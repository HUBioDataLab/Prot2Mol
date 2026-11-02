import copy
from dataclasses import replace
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.paired_generalization import adaptation_query as query
from experiments.paired_generalization import adaptation_fit as fit
from experiments.paired_generalization.adaptation_views import AdaptationViews
from experiments.paired_generalization.checkpoint_evaluation import read_native_predictions
from experiments.paired_generalization.qualify_controls import make_model
from experiments.paired_generalization.run import sha256
from reward_model.training.trainer import RewardModelTrainer
from test_paired_generalization_checkpoint_evaluation import config_fixture
import test_paired_generalization_adaptation_views as view_fixtures


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('ACCELERATE_USE_CPU', 'true')
    monkeypatch.setenv('WANDB_MODE', 'disabled')
    original_tokens = view_fixtures.token_dataset

    def tiny_tokens(raw):
        dataset = original_tokens(raw)
        values = dict(protein_input_ids=[[0, 5, 2]]*len(dataset),
            protein_attention_mask=[[1, 1, 1]]*len(dataset), protein_length=[3]*len(dataset),
            molecule_input_ids=[[0, 3+i%20, 4+(i*3)%20, 2] for i in range(len(dataset))],
            molecule_attention_mask=[[1, 1, 1, 1]]*len(dataset), molecule_length=[4]*len(dataset))
        for key, rows in values.items():
            dataset = dataset.remove_columns(key).add_column(key, rows)
        return dataset

    monkeypatch.setattr(view_fixtures, 'token_dataset', tiny_tokens)
    manifest, _ = view_fixtures.bundle(tmp_path/'views')
    views = AdaptationViews(manifest, sha256(manifest))
    model = make_model(None, {})
    config = config_fixture(model, tmp_path)
    config = replace(config, training=replace(config.training, seed=42,
        num_train_epochs=2, per_device_train_batch_size=2, gradient_accumulation_steps=2,
        learning_rate=.001, logging_steps=1, evaluate_test=False))
    return model, views, config


def trained(inputs, tmp_path, scope='heads', task_id='across_k1'):
    model, views, config = inputs
    base_state = copy.deepcopy(model.state_dict())
    base = query.parameter_digest(model)
    source = query.task_source(views, task_id, config.training.seed)
    result = fit.fit_support(model, views.load(source['task']['support_view']), config,
        tmp_path/'support_fit', scope=scope, base_parameter_sha256=base, source=source, cpu=True)
    model.load_state_dict(base_state)
    return base, result, dict(fit_directory=tmp_path/'support_fit',
                            fit_complete_sha256=sha256(tmp_path/'support_fit/complete.json'))


def forbid_training(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail('Query scoring must never fit or create an optimizer')
    monkeypatch.setattr(RewardModelTrainer, 'train', fail)
    monkeypatch.setattr(RewardModelTrainer, 'create_optimizer', fail)


@pytest.mark.parametrize('scope', fit.SCOPES)
def test_native_adapter_query_preserves_base_and_produces_selection_rows(inputs, tmp_path, monkeypatch, scope):
    model, views, config = inputs
    base, fitted, binding = trained(inputs, tmp_path, scope)
    forbid_training(monkeypatch)
    model.train(); model.protein_encoder.eval()
    modes = {m:m.training for m in model.modules()}
    flags = {n:p.requires_grad for n,p in model.named_parameters()}
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    original_config = copy.deepcopy(config)
    result = query.evaluate_query(model, views, config, tmp_path/'query', task_id='across_k1',
        role='validation', epoch=1, base_parameter_sha256=base, scope=scope, cpu=True, **binding)
    assert query.parameter_digest(model) == base and config == original_config
    assert {m:m.training for m in model.modules()} == modes
    assert {n:p.requires_grad for n,p in model.named_parameters()} == flags
    assert random.getstate() == python_rng
    np.testing.assert_equal(np.random.get_state(), numpy_rng)
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert result['base_restored'] and result['modes_and_rng_restored']
    assert not result['optimizer_created'] and not result['selection_performed']
    assert result['cpu_qualification'] and result['source']['query_role'] == 'validation'
    assert result['selection_records']
    assert all(r['role']=='validation' and r['seed']==42 and r['regime']=='B' and r['epoch']==1
               for r in result['selection_records'])
    assert len(result['assay_records']) == 3*result['assays']
    assert set(r['scorer'] for r in result['assay_records']) == set(query.SCORERS)
    restored = json.loads((tmp_path/'query/complete.json').read_text())
    assert restored == result
    with pytest.raises(RuntimeError, match='Existing query'):
        query.evaluate_query(model, views, config, tmp_path/'query', task_id='across_k1',
            role='validation', epoch=1, base_parameter_sha256=base, scope=scope, cpu=True, **binding)

    # Independent direct restoration must produce exactly the same native scores.
    snapshot = fitted['snapshots'][0]
    fit.restore_adapter(model, tmp_path/'support_fit'/snapshot['file'], snapshot['sha256'])
    _, examples, _ = query.query_view(views, 'across_k1', 'validation')
    direct = query.evaluate_fixed_model(model, examples, config, tmp_path/'direct', prefix='eval', cpu=True)
    actual, _ = read_native_predictions(result['prediction_manifest'])
    expected, _ = read_native_predictions(direct['prediction_manifest'])
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_k0_baseline_has_no_adapter_optimizer_or_selection_candidates(inputs, tmp_path, monkeypatch):
    model, views, config = inputs
    forbid_training(monkeypatch)
    base = query.parameter_digest(model)
    result = query.evaluate_query(model, views, config, tmp_path/'baseline', task_id='across_k0',
        role='test', epoch=0, base_parameter_sha256=base, cpu=True)
    assert result['selection_records'] == [] and result['adapter_sha256'] is None
    assert result['fit_complete_sha256'] is None and result['source']['scope'] is None
    assert result['source']['query_role'] == 'test' and query.parameter_digest(model) == base


def test_within_unit_queries_use_reserved_assays_and_keep_unit_identity(inputs, tmp_path, monkeypatch):
    model, views, config = inputs
    task_id = next(k for k,t in views.record['tasks'].items()
                   if t['kind']=='within' and t['probe_role']=='validation' and t['k']==1)
    base, _, binding = trained(inputs, tmp_path, task_id=task_id)
    forbid_training(monkeypatch)
    result = query.evaluate_query(model, views, config, tmp_path/'within', task_id=task_id,
        role='validation', epoch=2, scope='heads', base_parameter_sha256=base, cpu=True, **binding)
    assert result['units'] == 1
    assert {r['unit'] for r in result['assay_records']} == {views.record['tasks'][task_id]['unit']}
    support_groups = set(views.record['views'][views.record['tasks'][task_id]['support_view']]['group_ids'])
    assert not support_groups & {r['group_id'] for r in result['assay_records']}
    with pytest.raises(ValueError, match='query role'):
        query.evaluate_query(model, views, config, tmp_path/'wrong_role', task_id=task_id,
            role='test', epoch=2, scope='heads', base_parameter_sha256=base, cpu=True, **binding)


def test_undefined_and_ineligible_assays_are_retained_with_native_precision():
    frame = pd.DataFrame(dict(group_id=['p__a']*3+['p__b']*3+['p__c']*2,
        pchembl_value=[6., 6.25, 6.5-1e-8, 5., 6., 7., 5., 8.], binary_label=[0,1,1,0,0,1,0,1],
        ranking_score=[1.,1.,1.,3.,2.,1.,1.,2.], activity_logits=[1.,2.,3.,1.,2.,3.,1.,2.],
        cosine_similarity=[0.]*8))
    frame['pchembl_value'] = frame.pchembl_value.astype(np.float32)
    groups = pd.DataFrame(dict(group_id=['p__a','p__b','p__c'],unit=['u1','u1','u2'],
                               rows=[3,3,2],ranking_eligible=[True,True,False]))
    summary, records = query.pathway_records(frame, groups, .5)
    assert summary['ranking_score']['macro_spearman'] == -1.
    assert summary['ranking_score']['spearman_num_groups'] == 2
    assert summary['ranking_score']['spearman_num_defined_groups'] == 1
    assert summary['cosine_similarity']['macro_spearman'] is None
    assert len(records) == 9
    a = next(r for r in records if r['scorer']=='ranking_score' and r['group_id']=='p__a')
    assert a['eligible'] and not a['defined'] and a['rho'] is None
    assert all(not r['eligible'] for r in records if r['group_id']=='p__c')
    with pytest.raises(ValueError, match='complete frozen assays'):
        query.pathway_records(frame.iloc[:-1], groups, .5)
    altered = frame.copy(); altered.loc[0,'ranking_score'] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        query.pathway_records(altered, groups, .5)


def test_failure_restores_adapter_modes_and_rng_without_a_completion_marker(inputs, tmp_path, monkeypatch):
    model, views, config = inputs
    base, _, binding = trained(inputs, tmp_path)
    modes = {m:m.training for m in model.modules()}
    rng = torch.get_rng_state()

    def injected(*args, **kwargs):
        assert query.parameter_digest(model) != base
        model.eval(); torch.rand(3)
        raise RuntimeError('injected evaluation failure')

    monkeypatch.setattr(query, 'evaluate_fixed_model', injected)
    with pytest.raises(RuntimeError, match='injected'):
        query.evaluate_query(model, views, config, tmp_path/'failed', task_id='across_k1',
            role='validation', epoch=1, scope='heads', base_parameter_sha256=base, cpu=True, **binding)
    assert query.parameter_digest(model) == base
    assert {m:m.training for m in model.modules()} == modes and torch.equal(torch.get_rng_state(), rng)
    assert not (tmp_path/'failed/complete.json').exists()


@pytest.mark.parametrize('change', ['task', 'checksum', 'epoch', 'base', 'scope', 'cpu_mode'])
def test_mismatched_fit_is_rejected_before_query_inference(inputs, tmp_path, monkeypatch, change):
    model, views, config = inputs
    base, _, binding = trained(inputs, tmp_path)
    kwargs = dict(task_id='across_k1', role='validation', epoch=1, scope='heads',
                  base_parameter_sha256=base, cpu=True, **binding)
    if change=='task':kwargs['task_id']='across_k2'
    if change=='checksum':kwargs['fit_complete_sha256']='0'*64
    if change=='epoch':kwargs['epoch']=3
    if change=='base':kwargs['base_parameter_sha256']='0'*64
    if change=='scope':kwargs['scope']='heads_cross_attention'
    if change=='cpu_mode':kwargs['cpu']=False
    monkeypatch.setattr(query, 'evaluate_fixed_model', lambda *a,**k:pytest.fail('Must reject before inference'))
    with pytest.raises(ValueError):
        query.evaluate_query(model, views, config, tmp_path/'bad', **kwargs)
    assert query.parameter_digest(model)==base and not (tmp_path/'bad').exists()


@pytest.mark.parametrize('column', ['compound_id', 'binary_label'])
def test_query_export_identity_and_labels_are_checked_after_native_inference(inputs, tmp_path, monkeypatch, column):
    model, views, config = inputs
    base, _, binding = trained(inputs, tmp_path)
    real_reader = query.read_native_predictions

    def altered(path):
        frame, manifest = real_reader(path)
        frame.loc[0, column] = 'different_molecule' if column=='compound_id' else 1-frame.loc[0,column]
        return frame, manifest

    monkeypatch.setattr(query, 'read_native_predictions', altered)
    with pytest.raises(AssertionError):
        query.evaluate_query(model, views, config, tmp_path/'altered', task_id='across_k1',
            role='validation', epoch=1, scope='heads', base_parameter_sha256=base, cpu=True, **binding)
    assert query.parameter_digest(model)==base
    assert (tmp_path/'altered/native/complete.json').exists()
    assert not (tmp_path/'altered/complete.json').exists()


def test_support_query_overlap_is_rejected_before_inference(inputs, tmp_path, monkeypatch):
    model, views, config = inputs
    views.record['tasks']['across_k1']['support_view'] = 'across_validation'
    monkeypatch.setattr(query, 'evaluate_fixed_model', lambda *a,**k:pytest.fail('Must reject before inference'))
    with pytest.raises(ValueError, match='Support and query assays overlap'):
        query.evaluate_query(model, views, config, tmp_path/'overlap', task_id='across_k1',
            role='validation', epoch=1, scope='heads', base_parameter_sha256=query.parameter_digest(model),
            fit_directory=tmp_path/'unused', fit_complete_sha256='0'*64, cpu=True)
    assert not (tmp_path/'overlap').exists()
