import copy
import json
from pathlib import Path

import pytest

from experiments.paired_generalization import control_plan, control_driver
from experiments.paired_generalization.run import sha256, write_json
from test_paired_generalization_control_plan import ROOT, noise_fixture


def test_gate_checks_all_baselines_and_matching_data(tmp_path):
    plan = dict(source_data_sha256={r: {'train': r+'train', 'val': r+'val', 'test': r+'test'} for r in ('A','B')})
    gate = control_driver.baseline_gate(tmp_path, plan)
    assert not gate['ready'] and len(gate['missing']) == 20
    for regime in ('A','B'):
        for template in control_driver.BASELINE_PATHS:
            write_json(tmp_path/'tier2'/template.format(regime=regime)/'complete.json',
                dict(regime=regime, data_sha256=plan['source_data_sha256'][regime], metrics={'val':{}, 'test':{}}))
    assert control_driver.baseline_gate(tmp_path, plan)['ready']
    plan['source_data_sha256']['B']['train'] = 'changed'
    with pytest.raises(ValueError, match='provenance'):
        control_driver.baseline_gate(tmp_path, plan)


def test_busy_gpu_detection_uses_uuid_allowlist():
    a, b = control_plan.GPU_UUIDS[:2]
    observed = f'{a}, 123\nGPU-unrelated, 456\n{b}, 789\n'
    assert control_driver.busy_authorized_devices(observed) == [dict(gpu_uuid=a,pid=123), dict(gpu_uuid=b,pid=789)]


def completed_tuning_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(control_plan, 'WORK', tmp_path)
    directory = tmp_path/'controlled'
    plan = control_plan.build_plan(ROOT, tmp_path, noise_fixture())
    write_json(directory/'plan.json', plan)
    digest = sha256(directory/'plan.json')
    for job in plan['tuning_jobs'][:6]:
        metadata = control_plan.materialize_job(job, directory, digest)
        output = Path(metadata['output_dir'])
        lr = job['learning_rate_multiplier']
        score = {1.: .3, .3: .2, 3.: .1}[lr] if job['regime'] == 'B' else {1.: .1, .3: .9, 3.: .2}[lr]
        write_json(output/'experiment_summary.json', dict(config_sha256=metadata['config_sha256'], test_evaluated=False, test_metrics={}))
        for rank in range(4):
            write_json(output/f'rank_{rank}_complete.json', dict(rank=rank,world_size=4,config_sha256=metadata['config_sha256']))
        write_json(output/'retained/retention.json', dict(config_sha256=metadata['config_sha256'], selections={'best_macro':{'step':10,'metric':score}}))
        snapshot = output/'prediction_exports/trajectory/eval_step00000010.json'
        snapshot.parent.mkdir(parents=True)
        snapshot.with_suffix('.npz').write_bytes(b'fixture-payload')
        write_json(snapshot, dict(predictions_sha256=sha256(snapshot.with_suffix('.npz')),
            metrics=dict(eval_macro_spearman=score,eval_spearman_defined_fraction=1.)))
    return plan


def test_confirmation_uses_complete_b_validation_and_rejects_test_exposure(tmp_path, monkeypatch):
    plan = completed_tuning_fixture(tmp_path, monkeypatch)
    result = control_plan.confirm('full')
    assert result['selection']['learning_rate_multiplier'] == 1.
    assert {(j['regime'],j['seed']) for j in result['confirmation_jobs']} == {('A',43),('B',43),('A',44),('B',44)}
    assert all(j['learning_rate_multiplier'] == 1. for j in result['confirmation_jobs'])
    assert control_plan.confirm('full') == result
    job = plan['tuning_jobs'][0]
    summary = Path(job['config']['training']['output_dir'])/'experiment_summary.json'
    changed = json.loads(summary.read_text())
    changed['test_evaluated'] = True
    write_json(summary, changed)
    with pytest.raises(ValueError, match='test data'):
        control_plan.confirm('full')


def test_missing_rank_prevents_confirmation_and_no_confirmation_configs_are_written(tmp_path, monkeypatch):
    plan = completed_tuning_fixture(tmp_path, monkeypatch)
    output = Path(plan['tuning_jobs'][1]['config']['training']['output_dir'])
    (output/'rank_3_complete.json').unlink()
    with pytest.raises(FileNotFoundError):
        control_plan.confirm('full')
    assert not (tmp_path/'controlled/selections/full.json').exists()
    assert not list((tmp_path/'controlled/configs').glob('*seed43*'))


def test_next_job_does_not_treat_summary_alone_as_verified_training_completion(tmp_path, monkeypatch):
    plan = completed_tuning_fixture(tmp_path, monkeypatch)
    directory = tmp_path/'controlled'
    next_job = control_driver.next_job(directory, plan)
    assert next_job['id'] == plan['tuning_jobs'][0]['id']
    for job in plan['tuning_jobs'][:6]:
        metadata = json.loads((directory/'jobs'/f"{job['id']}.json").read_text())
        output = Path(metadata['output_dir'])
        write_json(output/'controlled_training_complete.json', dict(
            config_sha256=metadata['config_sha256'], plan_sha256=metadata['plan_sha256'],
            evidence_sha256={'experiment_summary.json':sha256(output/'experiment_summary.json')}))
    # Point the driver's confirmation function at the same temporary workspace.
    next_job = control_driver.next_job(directory, plan)
    assert next_job['seed'] == 43 and next_job['regime'] == 'A'
    assert next_job['learning_rate_multiplier'] == 1.
