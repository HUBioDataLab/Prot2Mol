"""Production reconstruction checks compare native inputs, not probe scores."""
import copy

from datasets import Dataset
import pytest
import torch

from experiments.paired_generalization import objective_probe_batches as design
from experiments.paired_generalization import verify_objective_probe_batches as verify
from reward_model.training.data import RewardAssayListCollator
from test_paired_generalization_objective_probe_batches import inputs


def test_collated_parity_records_values_shapes_dtypes_and_metadata():
    batch = dict(tokens=torch.tensor([[1, 2], [3, 4]]), labels=torch.tensor([0., 1.]),
                 empty=torch.tensor([], dtype=torch.long), assays=['p__a', 'p__b'])
    result = verify.compare_batches(batch, copy.deepcopy(batch))
    assert result['tokens']['dtype'] == 'torch.int64' and result['tokens']['shape'] == [2, 2]
    assert result['empty']['shape'] == [0]
    assert all(len(value['sha256']) == 64 for value in result.values())


@pytest.mark.parametrize('damage', ['field', 'shape', 'dtype', 'value', 'metadata', 'type'])
def test_collated_mismatch_cannot_pass_by_value_only(damage):
    expected = dict(x=torch.tensor([1, 2]), labels=['a', 'b']); actual = copy.deepcopy(expected)
    if damage == 'field': actual['extra'] = 0
    elif damage == 'shape': actual['x'] = actual['x'].reshape(1, 2)
    elif damage == 'dtype': actual['x'] = actual['x'].to(torch.int32)
    elif damage == 'value': actual['x'][0] = 9
    elif damage == 'metadata': actual['labels'].reverse()
    else: actual['labels'] = tuple(actual['labels'])
    with pytest.raises(ValueError): verify.compare_batches(expected, actual)


def context(inputs, role):
    source, examples, config, components = inputs
    metadata = design.aligned_metadata(source, examples, 'B', components)
    view = examples
    if role == 'cold_validation':
        cold = metadata[metadata.unit.isin([f'component:{i}' for i in range(0, 32, 2)])]
        view, metadata = design.cold_view(examples, metadata, cold)
    dataset, batches = design.sampler_layout(view, config, role)
    record, rows = design.freeze_batches(dataset, batches, metadata, count=2)
    return record, dataset, batches, metadata, examples, rows


@pytest.mark.parametrize('role', ['train', 'cold_validation'])
def test_reconstructs_every_tensor_including_original_cache_indices(inputs, role):
    args = context(inputs, role)
    collator = RewardAssayListCollator(protein_pad_token_id=1, molecule_pad_token_id=3)
    result = verify.verify_context(*args, collator)
    assert len(result['batches']) == 2
    assert result['exact_native_features'] and result['exact_native_collator_tensors']
    assert result['selected_unique_rows'] == len(args[-1])
    for batch in result['batches']:
        assert 'evaluation_example_indices' in batch['collated_fields']
        assert 'protein_input_ids' in batch['collated_fields']


@pytest.mark.parametrize('damage', ['selection', 'items', 'rows', 'tokens'])
def test_changed_manifest_rows_or_cached_tokens_require_inspection(inputs, damage):
    record, dataset, batches, metadata, examples, rows = context(inputs, 'cold_validation')
    if damage == 'selection': record['batches'][0]['source_batch_index'] += 1
    elif damage == 'items': record['batches'][0]['items'].reverse()
    elif damage == 'rows': rows = rows.iloc[1:].reset_index(drop=True)
    else:
        values = examples.to_dict()
        index = record['batches'][0]['cache_row_indices'][0]
        values['molecule_input_ids'][index][0] += 100
        examples = Dataset.from_dict(values)
    with pytest.raises((ValueError, AssertionError)):
        verify.verify_context(record, dataset, batches, metadata, examples, rows, RewardAssayListCollator())
