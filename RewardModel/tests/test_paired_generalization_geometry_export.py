from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from datasets import Dataset

from experiments.paired_generalization import geometry_export as export
from experiments.paired_generalization import representation_geometry as geometry
from experiments.paired_generalization.qualify_controls import make_model
from reward_model.training.trainer import RewardModelTrainer
from test_paired_generalization_checkpoint_evaluation import examples_fixture


def fixture():
    model = make_model(None, {})
    model.protein_tokenizer = model.molecule_tokenizer = SimpleNamespace(pad_token_id=1)
    dataset = examples_fixture(model)
    frame = dataset.select_columns(geometry.IDENTITY+['pchembl_value']).to_pandas()
    memberships, _ = geometry.sampling_design(frame, 'A', 'val', cap=7)
    return model, dataset, frame, memberships


def test_alignment_uses_identities_despite_different_raw_cache_and_view_orders():
    _, dataset, frame, memberships = fixture()
    # The raw row positions are still frozen; only token-cache positions move.
    shuffled = dataset.select(list(reversed(range(len(dataset)))))
    selected, rows, members = export.align_cohorts(shuffled, frame, memberships.sample(frac=1, random_state=7))
    expected = memberships.drop_duplicates(geometry.IDENTITY).sort_values(geometry.IDENTITY)
    assert rows[geometry.IDENTITY].values.tolist() == expected[geometry.IDENTITY].values.tolist()
    assert rows.cache_row_index.tolist() == [len(dataset)-1-i for i in rows.source_row_index]
    assert selected['compound_id'] == rows.compound_id.tolist()
    for view, group in members.groupby('view'):
        wanted = memberships[memberships.view == view].sort_values('view_row_index')
        actual = rows.iloc[group.export_row_index]
        assert actual[geometry.IDENTITY].values.tolist() == wanted[geometry.IDENTITY].values.tolist()


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'label', 'raw_identity', 'view_order', 'source_index'])
def test_wrong_data_refused_before_any_forward_or_output(tmp_path, monkeypatch, defect):
    model, dataset, frame, memberships = fixture()
    def refuse(*args, **kwargs):
        raise AssertionError('Invalid data must not reach inference')
    monkeypatch.setattr(model, 'forward', refuse)
    index = int(memberships.source_row_index.iloc[0])
    if defect == 'missing':
        dataset = dataset.select([i for i in range(len(dataset)) if i != index])
    elif defect == 'duplicate':
        dataset = dataset.select(list(range(len(dataset)))+[index])
    elif defect == 'label':
        data = dataset.to_list(); data[index]['pchembl_value'] += 1.; dataset = Dataset.from_list(data)
    elif defect == 'raw_identity':
        frame.loc[index, 'compound_id'] = 'wrong'
    elif defect == 'view_order':
        memberships.loc[0, 'view_row_index'] = 999
    else:
        memberships.loc[0, 'source_row_index'] = len(frame)
    with pytest.raises(ValueError):
        export.export_geometry(model, dataset, frame, memberships, tmp_path/'out', {}, cpu=True)
    assert not (tmp_path/'out').exists()


def test_actual_model_export_metrics_and_parameter_preservation(tmp_path, monkeypatch):
    model, dataset, frame, memberships = fixture()
    def refuse(*args, **kwargs):
        raise AssertionError('Geometry must not train or construct an optimizer')
    monkeypatch.setattr(RewardModelTrainer, 'train', refuse)
    monkeypatch.setattr(RewardModelTrainer, 'create_optimizer', refuse)
    before = {k:v.clone() for k,v in model.state_dict().items()}
    source = {'fixture':'frozen_rows'}
    output = tmp_path/'export'
    result = export.export_geometry(model, dataset, frame, memberships, output, source, cpu=True, batch_size=2)
    assert result['cpu_qualification']
    analysis = result['analysis']
    assert len(analysis['statistics']) == 8
    assert analysis['preprojection_alignment'] is None
    for side in ('protein','molecule'):
        values = np.load(output/(side+'_encoder_pooled.npy'))
        members = pd.read_parquet(output/'memberships.parquet')
        indices = members[members.view == 'observations'].export_row_index.to_numpy()
        singular = torch.linalg.svdvals(torch.tensor(values[indices], dtype=torch.float64))
        probabilities = singular/singular.sum()
        probabilities = probabilities[probabilities > 0]
        expected = torch.exp(-(probabilities*torch.log(probabilities)).sum()).item()
        assert analysis['statistics']['observations/'+side+'/encoder_pooled']['rank']['value'] == pytest.approx(expected)
    assert analysis['alignment']['all_observed_pairs']['requested_pairs'] == 7
    rows = pd.read_parquet(output/'rows.parquet')
    indices = members[members.view == 'observations'].export_row_index.to_numpy()
    assert analysis['alignment']['strict_active_pairs']['requested_pairs'] == int((rows.iloc[indices].pchembl_value > 6).sum())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert export.verify_complete(output, source, cpu=True) == result
    with pytest.raises(ValueError, match='execution mode'):
        export.verify_complete(output, source, cpu=False)
    with pytest.raises(RuntimeError, match='Existing geometry'):
        export.export_geometry(model, dataset, frame, memberships, output, source, cpu=True)
    (output/'protein_encoder_pooled.npy').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='artifact changed'):
        export.verify_complete(output, source, cpu=True)


def test_capture_parity_rejects_wrong_projection_path(tmp_path, monkeypatch):
    model, dataset, frame, memberships = fixture()
    original = model.forward
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        result.cosine_similarity = result.cosine_similarity+1.
        return result
    monkeypatch.setattr(model, 'forward', changed)
    with pytest.raises(ValueError, match='native cosine pathway'):
        export.export_geometry(model, dataset, frame, memberships, tmp_path/'wrong', {}, cpu=True)
    assert not (tmp_path/'wrong/complete.json').exists()
    assert not model.contrastive_protein_projection._forward_hooks


def test_scientific_export_requires_atum_and_permitted_gpu_before_data_access(tmp_path, monkeypatch):
    model, dataset, frame, memberships = fixture()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    with pytest.raises(RuntimeError, match='authorized Atum'):
        export.export_geometry(model, dataset, frame, memberships, tmp_path/'out', {})
    assert not (tmp_path/'out').exists()
