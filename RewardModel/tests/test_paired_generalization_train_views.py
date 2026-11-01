import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset, load_from_disk

from experiments.paired_generalization.control_plan import token_cache_hashes
from experiments.paired_generalization.learning_curve_plan import nested_unit_indices, subset_statistics
from experiments.paired_generalization.run import sha256, write_json
from reward_model.training.config import RewardTrainingDataConfig
from reward_model.training.data import RewardAssayListDataset, RewardEvaluationDataset, select_fixed_ranking_evaluation_subset
from reward_model.training.known_positives import KnownPositiveIndex
from reward_model.training.train_views import check_whole_units, indexed_label_view, load_training_view, subset_pair_labels


def raw_fixture():
    frame = pd.DataFrame([
        dict(assay_id=f'a{group}',group_id=f'p__a{group}',target_chembl_id='p',
             compound_id='shared' if i == 0 and group < 2 else f'm{group}_{i}',
             pchembl_value=float(5+i if group != 1 else 7+i),binary_label=0,
             measurement_count=1,unit=f'a{group}')
        for group in range(4) for i in range(3)])
    frame['binary_label'] = subset_pair_labels(frame, np.arange(len(frame)))
    return frame


def dataset_fixture(frame):
    columns = frame.drop(columns=['unit','assay_id','measurement_count']).to_dict('list')
    columns.update(protein_input_ids=[[i+1,0] for i in range(len(frame))],
        protein_attention_mask=[[1,0]]*len(frame),molecule_input_ids=[[1,2]]*len(frame),
        molecule_attention_mask=[[1,1]]*len(frame),protein_length=[1]*len(frame),molecule_length=[2]*len(frame))
    return Dataset.from_dict(columns)


def test_nested_subsets_keep_whole_unequal_units_and_ignore_row_order():
    units = np.repeat(np.arange(20), np.arange(1,21))
    subsets, order = nested_unit_indices(units)
    previous = set()
    for fraction, indices in subsets.items():
        check_whole_units(units, indices)
        assert previous <= set(indices)
        assert len(np.unique(units[indices])) == int(np.floor(fraction*20))
        previous = set(indices)
    repeated, second_order = nested_unit_indices(units[::-1])
    np.testing.assert_array_equal(order, second_order)
    for f in subsets:
        assert set(units[subsets[f]]) == set(units[::-1][repeated[f]])
    with pytest.raises(ValueError, match='fragments'):
        check_whole_units(np.array(['a','a','b']), np.array([0,2]))
    for bad in (np.array([1,0]),np.array([0,0]),np.array([-1]),np.array([3])):
        with pytest.raises(ValueError):
            check_whole_units(np.array(['a','a','b']),bad)


def test_subset_labels_exclude_omitted_assay_measurements():
    raw = raw_fixture()
    indices = np.array([0,1,2,6,7,8])
    labels = subset_pair_labels(raw,indices)
    assert raw.loc[0,'binary_label'] == 1  # pooled pair median is six
    assert labels[0] == 0                 # selected assay measurement is five
    changed = raw.copy()
    changed.loc[3:5,'pchembl_value'] = 1000.
    np.testing.assert_array_equal(subset_pair_labels(changed,indices),labels)
    stats = subset_statistics(raw,indices,labels,{'test':raw[['group_id','target_chembl_id']].drop_duplicates()})
    assert stats['labels_changed_from_full_training'] == 1
    assert stats['rows'] == 6 and stats['units'] == 2


def test_indexed_label_view_shares_token_buffers_and_survives_dataset_consumers(tmp_path, monkeypatch):
    raw = raw_fixture()
    dataset_fixture(raw).save_to_disk(str(tmp_path/'cache'),num_shards=2)
    original = load_from_disk(str(tmp_path/'cache'))
    indices = np.array([0,1,2,6,7,8])
    labels = subset_pair_labels(raw,indices)
    def no_flatten(*args,**kwargs):
        raise AssertionError('Token index flattening would copy payloads')
    monkeypatch.setattr(Dataset,'flatten_indices',no_flatten)
    view = indexed_label_view(original,indices,labels,'fixture_view')
    before = original.data.column('protein_input_ids').chunks
    after = view.data.column('protein_input_ids').chunks
    assert [chunk.values.buffers()[1].address for chunk in before] == [chunk.values.buffers()[1].address for chunk in after]
    assert set(p['filename'] for p in view.cache_files) == set(p['filename'] for p in original.cache_files)
    assert view['binary_label'] == labels.tolist()
    assert original['binary_label'] == raw.binary_label.tolist()
    assert view['compound_id'] == raw.iloc[indices].compound_id.tolist()
    restored = pickle.loads(pickle.dumps(view))
    assert restored[:] == view[:]
    sampler = RewardAssayListDataset(view,include_all_assays_for_contrastive=True)
    assert sorted(sampler.epoch_example_indices()) == list(range(len(view)))
    probe = select_fixed_ranking_evaluation_subset(view,num_assays=2,seed=42,min_pchembl_span=.5)
    evaluator = RewardEvaluationDataset(probe,protein_shuffle_sensitivity=False)
    seen = [evaluator[i]['rows'][0]['binary_label'] for i in range(len(evaluator))]
    assert seen == labels.tolist()
    positives = KnownPositiveIndex.from_training_dataset(view,active_threshold=6.)
    assert 'shared' not in positives.positives_by_target['p']
    assert positives.provenance['source_rows'] == len(view)


def manifest_fixture(directory):
    raw = raw_fixture()
    source = directory/'train.parquet'
    raw.to_parquet(source,index=False)
    cache = directory/'tokens/train_examples'
    dataset_fixture(raw).save_to_disk(str(cache),num_shards=2)
    dataset = load_from_disk(str(cache))
    metadata = directory/'metadata.parquet'
    raw.to_parquet(metadata,index=False)
    indices = np.array([0,1,2,6,7,8])
    np.save(directory/'indices.npy',indices)
    np.save(directory/'labels.npy',subset_pair_labels(raw,indices))
    record = dict(format_version=1,label_policy='within_selected_training_pair_median',activity_threshold=6.,unit_definition='whole assay',
        source=dict(parquet_path=str(source),parquet_sha256=sha256(source),cache_dir=str(cache),
            cache_files_sha256=token_cache_hashes(cache),fingerprint=dataset._fingerprint,rows=len(dataset)),
        statistics={'rows':len(indices)})
    for key, name in [('indices','indices.npy'),('labels','labels.npy'),('source_metadata','metadata.parquet')]:
        record[key] = dict(path=str(directory/name),sha256=sha256(directory/name))
    manifest = directory/'view.json'
    write_json(manifest,record)
    config = RewardTrainingDataConfig(train_parquet_path=str(source),val_parquet_path=str(source),test_parquet_path=str(source),
        tokenized_dataset_dir=str(cache.parent),enforce_binary_label_threshold=False,
        train_view_manifest=str(manifest),train_view_sha256=sha256(manifest))
    return dataset,config,record


def test_training_view_verifies_bytes_and_recomputes_subset_labels(tmp_path):
    dataset,config,record = manifest_fixture(tmp_path)
    view,provenance = load_training_view(dataset,config)
    assert len(view) == 6 and view['binary_label'][0] == 0
    assert provenance['manifest_sha256'] == sha256(Path(config.train_view_manifest))
    assert not provenance['token_payloads_copied']
    # Even a consistently rehashed manifest cannot authorize pooled labels.
    labels_path = Path(record['labels']['path'])
    labels = np.load(labels_path)
    labels[0] = 1
    np.save(labels_path,labels)
    record['labels']['sha256'] = sha256(labels_path)
    write_json(Path(config.train_view_manifest),record)
    config.train_view_sha256 = sha256(Path(config.train_view_manifest))
    with pytest.raises(ValueError,match='solely from selected'):
        load_training_view(dataset,config)


@pytest.mark.parametrize('damage',['manifest','cache','source'])
def test_changed_source_or_manifest_rejects_subset(tmp_path,damage):
    dataset,config,record = manifest_fixture(tmp_path)
    if damage == 'manifest':
        Path(config.train_view_manifest).write_text('{}')
    elif damage == 'cache':
        (Path(record['source']['cache_dir'])/'dataset_info.json').write_text('{}')
    else:
        Path(config.train_parquet_path).write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):
        load_training_view(dataset,config)


def test_training_view_config_requires_pinned_hash():
    data = dict(train_parquet_path='train',val_parquet_path='val',test_parquet_path='test',tokenized_dataset_dir='cache')
    with pytest.raises(ValueError,match='together'):
        RewardTrainingDataConfig(**data,train_view_manifest='manifest')
    with pytest.raises(ValueError,match='64 lowercase'):
        RewardTrainingDataConfig(**data,train_view_manifest='manifest',train_view_sha256='invalid')
