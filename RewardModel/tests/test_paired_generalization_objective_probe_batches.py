"""Frozen diagnostic contexts must reconstruct actual native collator inputs."""
from types import SimpleNamespace

from datasets import Dataset
import numpy as np
import pandas as pd
import pytest
import torch

from experiments.paired_generalization import objective_probe_batches as m
from reward_model.training.data import RewardAssayListCollator


@pytest.fixture
def inputs():
    rows = []
    for g in range(64):
        for i in range(3 + g % 22):
            n, length = len(rows), 2 + (g+i) % 8
            rows.append(dict(assay_id=f'a{g}', target_chembl_id=f'p{g%7}', compound_id=f'm{n}',
                group_id=f'p{g%7}__a{g}', pchembl_value=4.+.3*i, binary_label=float(i > 6),
                protein_input_ids=[g+1]*3, protein_attention_mask=[1]*3,
                molecule_input_ids=[i+1]*length, molecule_attention_mask=[1]*length,
                protein_length=3, molecule_length=length))
    source = pd.DataFrame(rows)[m.RAW_COLUMNS]
    examples = Dataset.from_list(rows)
    config = SimpleNamespace(training=SimpleNamespace(seed=42, per_device_train_batch_size=12,
        length_bucketing=True, length_bucket_size_multiplier=2),
        data=SimpleNamespace(ranking_max_ligands=16, ranking_opportunity_divisor=32,
            ranking_min_pchembl_span=.5, max_classification_only_per_item=16, sample_all_assays=None),
        model=SimpleNamespace(contrastive_loss_weight=1., ranking_balance_assays=False))
    components = pd.Series({f'a{g}': g//2 for g in range(64)})
    return source, examples, config, components


def test_evenly_spaced_batch_selection_has_no_replacement():
    assert m.batch_numbers(100, 4) == [0, 33, 66, 99]
    assert m.batch_numbers(4, 4) == [0, 1, 2, 3]
    for total, count in [(2, 3), (0, 1), (2, 0), (1.5, 1)]:
        with pytest.raises(ValueError): m.batch_numbers(total, count)


@pytest.mark.parametrize('role', ['train', 'cold_validation'])
def test_frozen_contexts_equal_native_features_and_collated_tensors(inputs, role):
    source, examples, config, components = inputs
    metadata = m.aligned_metadata(source, examples, 'B', components)
    dataset, batches = m.sampler_layout(examples, config, role)
    result, rows = m.freeze_batches(dataset, batches, metadata, count=3)
    collator = RewardAssayListCollator()
    assert dataset.item_count_multiple == (48 if role == 'train' else 12)
    assert result['selected_batches'] == 3
    assert rows.cache_row_index.is_unique
    for record in result['batches']:
        native = [dataset[i] for i in batches[record['source_batch_index']]]
        frozen = m.materialize_batch(record, examples)
        assert native == frozen
        expected, actual = collator(native), collator(frozen)
        assert set(actual) == set(expected)
        for name in actual:
            if isinstance(actual[name], torch.Tensor): assert torch.equal(actual[name], expected[name]), name
            else: assert actual[name] == expected[name], name
        assert len(record['cache_row_indices']) == record['rows']
        assert len(record['items']) == 12


def test_cold_view_maps_noncontiguous_source_and_cache_orders_and_whole_units(inputs):
    source, examples, config, components = inputs
    permutation = np.random.default_rng(17).permutation(len(examples))
    examples = examples.select(permutation.tolist())
    metadata = m.aligned_metadata(source, examples, 'B', components)
    assert np.array_equal(metadata.source_row_index, permutation)
    cold = metadata[metadata.unit.isin(['component:1', 'component:3', 'component:7', 'component:11',
        'component:13', 'component:15', 'component:17', 'component:19'])][['source_row_index', 'unit', *m.IDENTITY]]
    view, selected = m.cold_view(examples, metadata, cold.sample(frac=1, random_state=8))
    dataset, batches = m.sampler_layout(view, config, 'cold_validation')
    result, _ = m.freeze_batches(dataset, batches, selected, count=1)
    record = result['batches'][0]
    frozen = m.materialize_batch(record, examples)
    native = [dataset[i] for i in batches[record['source_batch_index']]]
    for original, restored in zip(native, frozen):
        assert restored['rows'] == original['rows']
        assert restored['example_indices'] == selected.iloc[original['example_indices']].cache_row_index.tolist()
    with pytest.raises(ValueError, match='all rows'):
        m.cold_view(examples, metadata, cold.iloc[1:])


def test_zero_ranking_or_positive_contexts_are_retained_without_enrichment(inputs):
    source, examples, config, components = inputs
    data = examples.to_pandas(); data['pchembl_value'] = 5.; data['binary_label'] = 0.
    examples = Dataset.from_pandas(data, preserve_index=False)
    source = source.assign(pchembl_value=5., binary_label=0.)
    metadata = m.aligned_metadata(source, examples, 'A', components)
    dataset, batches = m.sampler_layout(examples, config, 'train')
    result, _ = m.freeze_batches(dataset, batches, metadata, count=3)
    assert all(r['flagged_ranking_lists'] == 0 and r['strict_active_rows'] == 0 for r in result['batches'])
    assert [r['source_batch_index'] for r in result['batches']] == m.batch_numbers(len(batches), 3)


@pytest.mark.parametrize('damage', ['label', 'identity', 'unit', 'length'])
def test_mapping_rejects_changed_labels_identities_units_or_lengths(inputs, damage):
    source, examples, config, components = inputs
    if damage == 'label': source.loc[0, 'pchembl_value'] += 1
    elif damage == 'identity': source.loc[0, 'compound_id'] = 'wrong'
    elif damage == 'unit': components = components.drop('a0')
    else:
        data = examples.to_dict(); data['protein_length'][0] = 0; examples = Dataset.from_dict(data)
    with pytest.raises(ValueError): m.aligned_metadata(source, examples, 'B', components)


@pytest.mark.parametrize('damage', ['wrong_role', 'wrong_seed', 'missing_lengths', 'balanced', 'disabled_contrastive'])
def test_sampler_cannot_silently_change_reference_context_recipe(inputs, damage):
    _, examples, config, _ = inputs; role = 'train'
    if damage == 'wrong_role': role = 'test'
    elif damage == 'wrong_seed': config.training.seed = 44
    elif damage == 'missing_lengths': examples = examples.remove_columns('protein_length')
    elif damage == 'balanced': config.model.ranking_balance_assays = True
    else: config.model.contrastive_loss_weight = 0.
    with pytest.raises(ValueError): m.sampler_layout(examples, config, role)


@pytest.mark.parametrize('damage', ['flattening', 'assay', 'duplicate_mapping'])
def test_reconstruction_does_not_accept_changed_contexts(inputs, damage):
    source, examples, config, components = inputs
    metadata = m.aligned_metadata(source, examples, 'A', components)
    dataset, batches = m.sampler_layout(examples, config, 'train')
    if damage == 'duplicate_mapping':
        metadata.loc[1, 'cache_row_index'] = metadata.loc[0, 'cache_row_index']
        with pytest.raises(ValueError, match='ambiguous'): m.freeze_batches(dataset, batches, metadata, count=1)
        return
    result, _ = m.freeze_batches(dataset, batches, metadata, count=1); record = result['batches'][0]
    if damage == 'flattening': record['cache_row_indices'] = list(reversed(record['cache_row_indices']))
    else: record['items'][0]['assay_lists'][0]['group_id'] = 'wrong'
    with pytest.raises(ValueError): m.materialize_batch(record, examples)
