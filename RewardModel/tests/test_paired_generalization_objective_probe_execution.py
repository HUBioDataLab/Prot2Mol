import copy
import json

import pytest
import torch
from datasets import Dataset

from experiments.paired_generalization import objective_probe_execution as worker
from reward_model.training.trainer import RewardModelTrainer


def test_native_saved_model_batches_measurements_and_reload(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('ACCELERATE_USE_CPU', 'true')
    monkeypatch.setenv('WANDB_MODE', 'disabled')
    def forbidden(*args, **kwargs): raise AssertionError('Diagnostics must not fit or create an optimizer')
    monkeypatch.setattr(RewardModelTrainer, 'train', forbidden)
    monkeypatch.setattr(RewardModelTrainer, 'create_optimizer', forbidden)
    result = worker.qualify_cpu(tmp_path/'qualification')
    assert result['synthetic_pairs'] == 2 and result['native_checkpoint_roundtrip']
    assert result['parameters_and_rng_preserved'] and not result['production_gpu_qualified']
    completed = json.loads((tmp_path/'qualification/pairs/complete.json').read_text())
    assert completed['cpu_qualification'] and len(completed['artifacts_sha256']) == 2
    summary = completed['summary']
    assert not summary['independent_batch_intervals']
    block = summary['gradient_statistics']['protein_encoder.encoder.layer.0']
    assert block['objectives']['R']['states'] == {'nonzero': 2}
    assert block['cosines']['R__C']['defined'] == 2
    assert summary['gradient_statistics']['fusion']['objectives']['C']['states'] == {'disconnected': 2}
    assert summary['gradient_statistics']['fusion']['cosines']['R__C']['undefined'] == 2
    for source in ('R', 'B', 'C'):
        assert set(summary['lookahead'][source]) == {'encoders', 'interaction', 'shared'}
    empty = summary['lookahead']['C']['interaction']
    assert empty['states'] == {'no_effective_parameter_step': 2}
    assert empty['targets']['R']['relative_transfer']['undefined'] == 2
    rows = [json.loads((tmp_path/'qualification/pairs'/name).read_text()) for name in sorted(completed['artifacts_sha256'])]
    assert worker.summarize(rows) == summary
    rows[1]['measurement']['learning_rates']['protein_encoder'] *= 2
    with pytest.raises(ValueError, match='Cannot pool'): worker.summarize(rows)
    with pytest.raises(ValueError, match='every pair once'): worker.summarize(rows[:1]*2)


def test_distribution_preserves_zero_undefined_and_signed_values():
    result = worker.distribution([None, -2., 0., 1.])
    assert result == dict(total=4, defined=3, undefined=1, mean=-1/3,
        quantiles=dict(min=-2., q25=-1., median=0., q75=.5, max=1.), positive=1, negative=1, zero=1)
    assert worker.distribution([None])['mean'] is None
    for value in (float('nan'), float('inf')):
        with pytest.raises(ValueError, match='Nonfinite'): worker.distribution([value])


@pytest.mark.parametrize('damage', ['tokens', 'fingerprint', 'pair_order', 'parameter_identity', 'dtype', 'role'])
def test_invalid_input_cannot_reach_model_forward(tmp_path, monkeypatch, damage):
    model, examples, contexts, collator, training, evidence = worker.qualification_fixture()
    source = dict(parameter_sha256=worker.parameter_digest(model))
    if damage == 'tokens':
        values = examples['train'].to_dict()
        index = contexts['train']['batches'][0]['cache_row_indices'][0]
        values['protein_input_ids'][index][0] += 1
        examples['train'] = Dataset.from_dict(values)
    if damage == 'fingerprint': evidence['train']['batches'][0]['collated_fields']['protein_input_ids']['sha256'] = 'bad'
    if damage == 'pair_order': contexts['train']['batches'].reverse()
    if damage == 'parameter_identity': source['parameter_sha256'] = 'bad'
    if damage == 'dtype': model.double()
    if damage == 'role': examples['test'] = examples.pop('cold_validation')
    def forbidden(*args, **kwargs): raise AssertionError('Invalid input reached forward')
    monkeypatch.setattr(model, 'forward', forbidden)
    with pytest.raises(ValueError):
        worker.execute_pairs(model, examples, contexts, collator, training, evidence, tmp_path/'out', source, cpu=True)
    assert not (tmp_path/'out/complete.json').exists()


def test_failure_keeps_partial_output_visible_and_restores_precision(tmp_path, monkeypatch):
    args = worker.qualification_fixture()
    model = args[0]; before = worker.parameter_digest(model)
    source = dict(parameter_sha256=before)
    original = worker.probes.measure
    calls = []
    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2: raise RuntimeError('injected failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(worker.probes, 'measure', fail_second)
    precision = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                 torch.backends.cudnn.allow_tf32)
    with pytest.raises(RuntimeError, match='injected failure'):
        worker.execute_pairs(*args, tmp_path/'out', source, cpu=True)
    assert (tmp_path/'out/pair_000.json').is_file() and not (tmp_path/'out/complete.json').exists()
    assert worker.parameter_digest(model) == before
    assert precision == (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                         torch.backends.cudnn.allow_tf32)
    with pytest.raises(RuntimeError, match='Existing probe output'):
        worker.execute_pairs(*args, tmp_path/'out', source, cpu=True)


def test_saved_artifact_summary_or_cpu_mode_changes_are_rejected(tmp_path):
    args = worker.qualification_fixture()
    model, _, contexts, _, _, evidence = args
    source = dict(parameter_sha256=worker.parameter_digest(model))
    result = worker.execute_pairs(*args, tmp_path/'out', source, cpu=True)
    assert worker.verify_complete(tmp_path/'out', source, contexts, evidence, cpu=True) == result
    with pytest.raises(ValueError, match='execution mode'):
        worker.verify_complete(tmp_path/'out', source, contexts, evidence, cpu=False)
    path = tmp_path/'out/complete.json'
    changed = copy.deepcopy(result); changed['summary']['pairs'] = 9
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='distributions differ'):
        worker.verify_complete(tmp_path/'out', source, contexts, evidence, cpu=True)
    path.write_text(json.dumps(result))
    (tmp_path/'out/pair_000.json').write_text('{}')
    with pytest.raises(ValueError, match='Saved diagnostic pair changed'):
        worker.verify_complete(tmp_path/'out', source, contexts, evidence, cpu=True)


def test_scientific_path_cannot_run_on_local_or_unapproved_device(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    with pytest.raises(RuntimeError, match='authorized Atum'):
        worker.execute_pairs(None, {}, {}, None, {}, {}, tmp_path/'out', {})
    assert not (tmp_path/'out').exists()
