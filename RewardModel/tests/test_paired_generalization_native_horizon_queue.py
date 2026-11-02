"""One-shot CPU handoff, exact original handles and full-horizon evidence gates."""
import os
from pathlib import Path

import pytest

from experiments.paired_generalization import native_horizon_queue as module
from experiments.paired_generalization.run import sha256, write_json


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path/'project'; work = root/'outputs'; root.mkdir()
    monkeypatch.setattr(module, 'ROOT', root); monkeypatch.setattr(module, 'WORK', work)
    monkeypatch.setattr(module, 'runtime_guard', lambda: None)
    source = root/'source.py'; source.write_text('fixed synthetic source')
    def software(): return dict(source_sha256={'source.py': sha256(source)}, packages={})
    monkeypatch.setattr(module, 'software', software); monkeypatch.setattr(module.native_analysis, 'software', software)
    monkeypatch.setattr(module.control_queue, 'boot_id', lambda: 'same-boot')
    processes = {role: dict(pid=100+i, start_ticks=200+i, uid=1, cwd=str(root), argv=['synthetic', role])
        for i, role in enumerate(('driver', 'training', 'rank0', 'rank1', 'rank2', 'rank3'))}
    live = {p['pid']: p for p in processes.values()}
    def inspect(pid):
        if pid == os.getpid(): return dict(pid=pid, start_ticks=1, uid=1, cwd=str(root), argv=['synthetic-coordinator'])
        return live.get(pid)
    monkeypatch.setattr(module.control_queue, 'process_identity', inspect)
    output = work/'controlled/runs'/module.IDENTITY
    job = dict(id=module.IDENTITY, output_dir=str(output), config_sha256='configuration', plan_sha256='primary-plan')
    registration = dict(job_metadata_sha256='metadata', launch_sha256='launch')
    monkeypatch.setattr(module.native_analysis, 'registration', lambda identity: (job, {}, registration))
    write_json(work/'controlled/full_tuning_queue/plan.json', dict(predecessor=job, processes=processes))
    directory = work/'tier1/native_trajectories'/module.IDENTITY/module.native_analysis.digest_json(software())[:16]
    prefix = directory/'trajectory_through_epoch10.json'
    write_json(prefix, dict(epochs_analyzed=list(range(1, 11))))
    write_json(directory/'verification_through_epoch10.json', dict(trajectory_sha256=sha256(prefix),
        source_and_artifact_hashes_verified=True, test_predictions_read=False))
    path = module.prepare(); plan = module.read(path)
    records = [dict(epoch=i, step=i*10, eval_macro_spearman=.2, eval_loss=1.) for i in range(1, 16)]
    state = output/'trainer_state.json'; write_json(state, dict(epoch=15, global_step=150, log_history=records))
    for record in records:
        for prefix in ('eval', 'train_eval'):
            write_json(output/'prediction_exports/trajectory'/f"{prefix}_step{record['step']:08d}.json",
                dict(epoch=record['epoch'], step=record['step']))
    completion = output/'controlled_training_complete.json'
    write_json(completion, dict(config_sha256=job['config_sha256'], plan_sha256=job['plan_sha256'],
        evidence_sha256={'trainer_state.json': sha256(state)}))
    return dict(root=root, work=work, output=output, path=path, plan=plan, live=live, source=source, completion=completion, directory=directory)


def test_plan_freezes_the_original_handles_and_verified_ten_epoch_prefix(prepared):
    module.validate_plan(prepared['plan'])
    assert len(module.live_originals(prepared['plan'])) == 6
    assert prepared['plan']['previous_epoch'] == 10
    with pytest.raises(RuntimeError, match='still live'): module.require_completion(prepared['plan'])
    prepared['live'].clear()
    assert module.require_completion(prepared['plan']) == prepared['completion']


def test_pid_reuse_does_not_keep_the_old_process_alive(prepared):
    prepared['live'].clear()
    expected = prepared['plan']['processes']['driver']; reused = dict(expected, start_ticks=expected['start_ticks']+1)
    prepared['live'][expected['pid']] = reused
    assert not module.live_originals(prepared['plan'])


def test_observation_error_is_not_mistaken_for_process_exit(prepared, monkeypatch):
    def fail(pid): raise RuntimeError('transient observation failure')
    monkeypatch.setattr(module.control_queue, 'process_identity', fail)
    with pytest.raises(RuntimeError, match='observation failure'): module.live_originals(prepared['plan'])


@pytest.mark.parametrize('damage', ['missing_completion', 'changed_state', 'missing_probe', 'duplicate_snapshot', 'wrong_horizon'])
def test_terminal_processes_without_complete_evidence_cannot_start_analysis(prepared, damage):
    prepared['live'].clear(); output = prepared['output']
    if damage == 'missing_completion': prepared['completion'].unlink()
    elif damage in ('changed_state', 'wrong_horizon'):
        p = output/'trainer_state.json'; state = module.read(p); state['epoch'] = 14; write_json(p, state)
        if damage == 'wrong_horizon':
            complete = module.read(prepared['completion']); complete['evidence_sha256']['trainer_state.json'] = sha256(p); write_json(prepared['completion'], complete)
    elif damage == 'missing_probe': (output/'prediction_exports/trajectory/train_eval_step00000150.json').unlink()
    else: write_json(output/'prediction_exports/trajectory/eval_extra.json', dict(epoch=1, step=10))
    with pytest.raises((ValueError, RuntimeError)): module.require_completion(prepared['plan'])


def test_source_or_preceding_evidence_change_stops_handoff(prepared):
    prepared['source'].write_text('changed')
    with pytest.raises(ValueError, match='source'): module.validate_plan(prepared['plan'])


def test_cpu_runtime_boundary_rejects_gpu_visibility(monkeypatch):
    monkeypatch.setattr(module, 'ROOT', Path('/home/atabey/Prot2Mol'))
    monkeypatch.setenv('ACCELERATE_USE_CPU', 'true'); monkeypatch.setenv('WANDB_MODE', 'disabled')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(RuntimeError, match='CPU-only'): module.runtime_guard()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    module.runtime_guard()


def test_one_handoff_waits_then_analyzes_and_verifies_exactly_once(prepared, monkeypatch):
    calls = []; sleeps = []
    def sleep(seconds):
        sleeps.append(seconds); prepared['live'].clear()
    def analyze(identity):
        assert not module.live_originals(prepared['plan']); calls.append(('analyze', identity))
        return dict(epochs_analyzed=list(range(1, 16)), complete_horizon_observed=True, test_predictions_read=False)
    def verify(directory, previous_epoch):
        calls.append(('verify', directory, previous_epoch))
        result = dict(checks=[{} for _ in range(30)], source_and_artifact_hashes_verified=True, test_predictions_read=False, trajectory_sha256='finished-trajectory')
        write_json(directory/'verification_through_epoch15.json', result)
        return result
    monkeypatch.setattr(module.time, 'sleep', sleep)
    monkeypatch.setattr(module.native_analysis, 'main', analyze)
    monkeypatch.setattr(module.verify_native_analysis, 'verify', verify)
    result = module.run()
    assert sleeps == [60]
    assert calls == [('analyze', module.IDENTITY), ('verify', prepared['directory'], 10)]
    assert result['snapshots_verified'] == 30 and result['new_fits_created'] == 0 and not result['study_complete']
    assert module.read(module.destination()/'exit.json')['returncode'] == 0
    with pytest.raises(RuntimeError, match='already started'): module.run()
    assert len(calls) == 2


def test_failed_analysis_is_recorded_without_retry_or_completion(prepared, monkeypatch):
    prepared['live'].clear(); calls = []
    def fail(identity): calls.append(identity); raise ValueError('bad snapshot')
    monkeypatch.setattr(module.native_analysis, 'main', fail)
    with pytest.raises(ValueError, match='bad snapshot'): module.run()
    assert calls == [module.IDENTITY] and not (module.destination()/'complete.json').exists()
    assert module.read(module.destination()/'exit.json')['returncode'] == 1
    with pytest.raises(RuntimeError, match='already started'): module.run()
    assert calls == [module.IDENTITY]


def test_incomplete_independent_verification_cannot_claim_success(prepared, monkeypatch):
    prepared['live'].clear()
    monkeypatch.setattr(module.native_analysis, 'main', lambda identity: dict(epochs_analyzed=list(range(1, 16)), complete_horizon_observed=True, test_predictions_read=False))
    monkeypatch.setattr(module.verify_native_analysis, 'verify', lambda *a, **kw: dict(checks=[{}]*28, source_and_artifact_hashes_verified=True, test_predictions_read=False))
    with pytest.raises(ValueError, match='verification is incomplete'): module.run()
    assert not (module.destination()/'complete.json').exists()
