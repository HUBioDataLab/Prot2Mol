"""Bounded CPU audits wait for evidence, preserve failures and never create fits."""
import os
from pathlib import Path

import pytest

from experiments.paired_generalization import native_tuning_audits as m
from experiments.paired_generalization.run import sha256, write_json


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path/'project'; work = root/'outputs'; root.mkdir()
    monkeypatch.setattr(m, 'ROOT', root); monkeypatch.setattr(m, 'WORK', work)
    monkeypatch.setattr(m, 'runtime_guard', lambda: None)
    monkeypatch.setattr(m, 'validate_tuning', lambda tuning: None)
    monkeypatch.setattr(m.queue, 'boot_id', lambda: 'boot')
    source = root/'source.py'; source.write_text('unchanged')
    software = lambda: dict(source_sha256={'source.py': sha256(source)}, packages={})
    monkeypatch.setattr(m, 'software', software); monkeypatch.setattr(m.analysis, 'software', software)
    python = str(root/'.venv/bin/python')
    parent = dict(pid=100, start_ticks=1, uid=os.getuid(), cwd=str(root),
        argv=[python, 'RewardModel/experiments/paired_generalization/control_queue.py', 'run'])
    jobs = [dict(id=identity, config_path=str(work/'controlled/configs'/f'{identity}.yaml'),
        output_dir=str(work/'controlled/runs'/identity), config_sha256='config', plan_sha256='primary') for identity in m.IDENTITIES]
    current = dict(pid=101, start_ticks=2, uid=os.getuid(), cwd=str(root), argv=[python, '-m', 'torch.distributed.run',
        '--standalone', '--nproc_per_node=4', 'RewardModel/experiments/paired_generalization/run.py', 'train', '--config', jobs[0]['config_path']])
    live = {100: parent, 101: current}
    def inspect(pid):
        if pid == os.getpid():
            return dict(pid=pid, start_ticks=3, uid=os.getuid(), cwd=str(root), argv=['cpu-audit'])
        return live.get(pid)
    monkeypatch.setattr(m.base, 'inspect_process', inspect)
    registrations = {j['id']: dict(job_metadata_sha256=j['id'], launch_sha256='launch') for j in jobs}
    monkeypatch.setattr(m.analysis, 'registration', lambda identity: (next(j for j in jobs if j['id'] == identity), {}, registrations[identity]))
    qualification = root/'qualification.json'; write_json(qualification, {'sources': software()['source_sha256']})
    tuning = dict(successors=[{'id': 'preceding-A'}, *jobs])
    tuning_dir = work/'controlled/full_tuning_queue'
    write_json(tuning_dir/'plan.json', tuning)
    write_json(tuning_dir/'launch_process.json', dict(identity=parent, plan_sha256=sha256(tuning_dir/'plan.json')))
    write_json(tuning_dir/'status.json', dict(phase='executing_registered_job', job_id=jobs[0]['id']))
    write_json(work/'controlled/driver_status.json', dict(phase='training', job_id=jobs[0]['id'], driver_pid=100,
        pid=101, gpu_uuids=list(m.queue.GPU_UUIDS)))
    Path(jobs[0]['output_dir']).mkdir(parents=True)
    (Path(jobs[0]['output_dir'])/'runtime_qualification.json').write_bytes(qualification.read_bytes())
    previous = work/'previous_audit'; previous_dir = work/'previous_native'
    previous_completion = work/'previous_training/complete.json'; write_json(previous_completion, {'complete': True})
    write_json(previous/'plan.json', dict(directory=str(previous_dir), files_sha256={}, training_sources_sha256={}))
    write_json(previous/'started.json', dict(identity=dict(pid=99, start_ticks=1)))
    write_json(previous/'exit.json', dict(returncode=0))
    write_json(previous_dir/'trajectory_through_epoch15.json', dict(epochs=list(range(1, 16))))
    write_json(previous_dir/'verification_through_epoch15.json', dict(checks=[{}]*30,
        source_and_artifact_hashes_verified=True, test_predictions_read=False))
    write_json(previous/'complete.json', dict(plan_sha256=sha256(previous/'plan.json'), snapshots_verified=30,
        training_completion_sha256=sha256(previous_completion),
        trajectory_sha256=sha256(previous_dir/'trajectory_through_epoch15.json'),
        verification_sha256=sha256(previous_dir/'verification_through_epoch15.json')))
    monkeypatch.setattr(m.base, 'destination', lambda: previous)
    monkeypatch.setattr(m.base, 'validate_plan', lambda plan: None)
    monkeypatch.setattr(m.base, 'require_completion', lambda plan: previous_completion)
    plan = m.read(m.prepare())
    return dict(root=root, work=work, jobs=jobs, plan=plan, live=live, parent=parent, current=current,
        source=source, qualification=qualification, registrations=registrations, previous=previous,
        previous_dir=previous_dir, inspect=inspect, tuning_dir=tuning_dir)


def finish(p, index, *, remove_launcher=True):
    job = p['jobs'][index]; output = Path(job['output_dir']); output.mkdir(parents=True, exist_ok=True)
    (output/'runtime_qualification.json').write_bytes(p['qualification'].read_bytes())
    records = [dict(epoch=i, step=i*10, eval_macro_spearman=.2, eval_loss=1.) for i in range(1, 16)]
    write_json(output/'trainer_state.json', dict(epoch=15, log_history=records))
    write_json(output/'experiment_summary.json', dict(test_evaluated=False, test_metrics={}))
    write_json(output/'retained/retention.json', dict(complete=True))
    evidence = ['trainer_state.json', 'experiment_summary.json', 'retained/retention.json']
    for rank in range(4):
        name = f'rank_{rank}_complete.json'
        write_json(output/name, dict(rank=rank, world_size=4, config_sha256=job['config_sha256']))
        evidence.append(name)
    for row in records:
        for prefix in ('eval', 'train_eval'):
            write_json(output/'prediction_exports/trajectory'/f"{prefix}_step{row['step']:08d}.json", row)
    write_json(output/'controlled_training_complete.json', dict(job_id=job['id'], phase='training_complete_test_not_scored',
        config_sha256=job['config_sha256'], plan_sha256=job['plan_sha256'],
        evidence_sha256={name: sha256(output/name) for name in evidence}))
    if index == 0 and remove_launcher:
        p['live'].pop(101, None)


def test_prepare_freezes_three_existing_jobs_and_completed_predecessor(prepared):
    p = prepared; m.validate_plan(p['plan'])
    assert tuple(j['id'] for j in p['plan']['jobs']) == m.IDENTITIES
    assert p['plan']['parent'] == p['parent'] and p['plan']['current_launcher'] == p['current']
    assert not any(Path(j['output_dir']).exists() for j in p['jobs'][1:])


@pytest.mark.parametrize('damage', ['parent', 'current_command', 'previous_exit', 'previous_hash', 'previous_checks', 'output'])
def test_prepare_rejects_bad_parent_or_predecessor_or_existing_analysis(prepared, damage):
    p = prepared
    if damage == 'parent': p['live'].pop(100)
    elif damage == 'current_command': p['live'][101] = dict(p['current'], argv=['unexpected'])
    elif damage == 'previous_exit': write_json(p['previous']/'exit.json', dict(returncode=1))
    elif damage == 'previous_hash': (p['previous_dir']/'trajectory_through_epoch15.json').write_text('{}')
    elif damage == 'previous_checks':
        v = p['previous_dir']/'verification_through_epoch15.json'; value = m.read(v); value['checks'] = [{}]*29; write_json(v, value)
        c = p['previous']/'complete.json'; value = m.read(c); value['verification_sha256'] = sha256(v); write_json(c, value)
    else: m.analysis_directory(m.IDENTITIES[0]).mkdir(parents=True)
    with pytest.raises((ValueError, RuntimeError)): m.prepare()


def test_live_launcher_is_not_bypassed_by_an_early_receipt(prepared, monkeypatch):
    p = prepared; finish(p, 0, remove_launcher=False); sleeps = []
    def sleep(seconds): sleeps.append(seconds); p['live'].pop(101)
    monkeypatch.setattr(m.time, 'sleep', sleep)
    assert m.wait_for_completion(p['plan'], p['jobs'][0])['job_id'] == m.IDENTITIES[0]
    assert sleeps == [30]


def test_future_completed_receipt_can_be_audited_after_parent_exits(prepared):
    p = prepared; finish(p, 2); p['live'].clear()
    assert m.wait_for_completion(p['plan'], p['jobs'][2])['job_id'] == m.IDENTITIES[2]


def test_missing_completion_after_parent_exit_requires_inspection(prepared):
    p = prepared; p['live'].clear()
    with pytest.raises(RuntimeError, match='without this verified fit'):
        m.wait_for_completion(p['plan'], p['jobs'][0])


@pytest.mark.parametrize('damage', ['state', 'missing_rank', 'missing_probe', 'qualification', 'registration'])
def test_completion_provenance_failure_is_never_retried(prepared, monkeypatch, damage):
    p = prepared; finish(p, 1); job = p['jobs'][1]; output = Path(job['output_dir'])
    if damage == 'state': (output/'trainer_state.json').write_text('{}')
    elif damage == 'missing_rank': (output/'rank_2_complete.json').unlink()
    elif damage == 'missing_probe': (output/'prediction_exports/trajectory/train_eval_step00000150.json').unlink()
    elif damage == 'qualification': (output/'runtime_qualification.json').write_text('{}')
    else: monkeypatch.setattr(m.analysis, 'registration', lambda identity: ({}, {}, {}))
    monkeypatch.setattr(m.time, 'sleep', lambda _: pytest.fail('Evidence errors must not retry'))
    with pytest.raises((ValueError, FileNotFoundError)):
        m.wait_for_completion(p['plan'], job)


@pytest.mark.parametrize('damage', ['omitted_rank', 'wrong_rank', 'test_scored'])
def test_self_consistent_but_incomplete_or_wrong_receipt_is_rejected(prepared, damage):
    p = prepared; finish(p, 1); job = p['jobs'][1]; output = Path(job['output_dir'])
    path = output/'controlled_training_complete.json'; receipt = m.read(path)
    if damage == 'omitted_rank': del receipt['evidence_sha256']['rank_2_complete.json']
    else:
        name = 'rank_2_complete.json' if damage == 'wrong_rank' else 'experiment_summary.json'
        value = m.read(output/name)
        value['rank' if damage == 'wrong_rank' else 'test_evaluated'] = 3 if damage == 'wrong_rank' else True
        write_json(output/name, value); receipt['evidence_sha256'][name] = sha256(output/name)
    write_json(path, receipt)
    with pytest.raises(ValueError): m.completed_evidence(p['plan'], job)


@pytest.mark.parametrize('where', ['process', 'status'])
def test_transient_observation_repolls_the_same_parent(prepared, monkeypatch, where):
    p = prepared; calls = []; sleeps = []
    if where == 'process':
        def inspect(pid):
            calls.append(pid)
            if len(calls) == 1: raise m.base.ProcessObservationError('temporary')
            return p['inspect'](pid)
        monkeypatch.setattr(m.base, 'inspect_process', inspect)
    else:
        read = m.read
        def transient(path):
            if path == p['tuning_dir']/'status.json':
                calls.append(path)
                if len(calls) == 1: raise OSError('temporary')
            return read(path)
        monkeypatch.setattr(m, 'read', transient)
    def sleep(seconds): sleeps.append(seconds); finish(p, 0)
    monkeypatch.setattr(m.time, 'sleep', sleep)
    assert m.wait_for_completion(p['plan'], p['jobs'][0])['job_id'] == m.IDENTITIES[0]
    assert sleeps == [30]


def install_analysis(monkeypatch, *, fail_second=False):
    calls = []
    def analyze(identity):
        calls.append(identity); directory = m.analysis_directory(identity)
        result = dict(epochs_analyzed=list(range(1, 16)), complete_horizon_observed=True, test_predictions_read=False)
        write_json(directory/'trajectory_through_epoch15.json', result)
        if fail_second and len(calls) == 2: raise ValueError('injected numerical failure')
        return result
    def verify(directory):
        result = dict(checks=[{}]*30, source_and_artifact_hashes_verified=True, test_predictions_read=False,
            trajectory_sha256=sha256(directory/'trajectory_through_epoch15.json'))
        write_json(directory/'verification_through_epoch15.json', result)
        return result
    monkeypatch.setattr(m.analysis, 'main', analyze); monkeypatch.setattr(m.verifier, 'verify', verify)
    monkeypatch.setattr(m.driver, 'execute_job', lambda *args: pytest.fail('No training permitted'))
    return calls


def test_runs_exactly_three_audits_once_and_preserves_receipts(prepared, monkeypatch):
    p = prepared
    for i in range(3): finish(p, i)
    p['live'].clear(); calls = install_analysis(monkeypatch)
    result = m.run()
    assert calls == list(m.IDENTITIES) and len(result['jobs']) == 3
    assert result['new_fits_created'] == 0 and not result['model_inference'] and not result['test_predictions_read']
    assert m.read(m.destination()/'exit.json')['returncode'] == 0
    with pytest.raises(RuntimeError, match='already started'): m.run()


def test_partial_analysis_is_preserved_and_never_retried(prepared, monkeypatch):
    p = prepared
    for i in range(3): finish(p, i)
    calls = install_analysis(monkeypatch, fail_second=True)
    with pytest.raises(ValueError, match='injected numerical failure'): m.run()
    assert calls == list(m.IDENTITIES[:2])
    assert (m.destination()/'jobs'/m.IDENTITIES[0]/'complete.json').exists()
    assert not (m.destination()/'complete.json').exists()
    assert m.read(m.destination()/'exit.json')['returncode'] == 1
    with pytest.raises(RuntimeError, match='already started'): m.run()


def test_changed_source_stops_before_analysis(prepared, monkeypatch):
    p = prepared; p['source'].write_text('changed'); calls = install_analysis(monkeypatch)
    with pytest.raises(ValueError, match='source'): m.run()
    assert not calls


def test_validation_environment_is_restored_on_failure(monkeypatch):
    monkeypatch.setenv('ACCELERATE_USE_CPU', 'true'); monkeypatch.setenv('WANDB_MODE', 'disabled')
    def fail(plan):
        assert os.environ['ACCELERATE_USE_CPU'] == 'false' and os.environ['WANDB_MODE'] == 'online'
        raise ValueError('changed queue')
    monkeypatch.setattr(m.queue, 'validate_plan', fail)
    with pytest.raises(ValueError, match='changed queue'): m.validate_tuning({})
    assert os.environ['ACCELERATE_USE_CPU'] == 'true' and os.environ['WANDB_MODE'] == 'disabled'
