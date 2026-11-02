"""Lifecycle and evidence gates for one fit under a persistent coordinator."""
import os

import pytest

from experiments.paired_generalization import native_active_handoff as m
from experiments.paired_generalization.run import sha256, write_json


@pytest.fixture
def active(tmp_path, monkeypatch):
    root = tmp_path/'project'; root.mkdir(); work = root/'outputs'
    monkeypatch.setattr(m, 'ROOT', root); monkeypatch.setattr(m, 'WORK', work)
    monkeypatch.setattr(m, 'runtime_guard', lambda: None)
    monkeypatch.setattr(m.control_queue, 'boot_id', lambda: 'boot')
    source = root/'source.py'; source.write_text('frozen source')
    def software(): return dict(source_sha256={'source.py': sha256(source)}, packages={})
    monkeypatch.setattr(m, 'software', software)
    monkeypatch.setattr(m.native_analysis, 'software', software)
    python = str(root/'.venv/bin/python'); command = 'RewardModel/experiments/paired_generalization/run.py'
    config = str(work/'controlled/configs'/f'{m.IDENTITY}.yaml')
    tail = ['train', '--config', config]
    argv = dict(driver=[python, 'RewardModel/experiments/paired_generalization/control_queue.py', 'run'],
        training=[python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4', command, *tail],
        **{f'rank{i}': [python, '-u', command, *tail] for i in range(4)})
    identities = {role: dict(pid=100+i, start_ticks=200+i, uid=os.getuid(), cwd=str(root), argv=command)
                  for i, (role, command) in enumerate(argv.items())}
    live = {p['pid']: p for p in identities.values()}
    def inspect(pid):
        if pid == os.getpid(): return dict(pid=pid, start_ticks=1, uid=os.getuid(), cwd=str(root), argv=['worker'])
        return live.get(pid)
    monkeypatch.setattr(m.control_queue, 'process_identity', inspect)
    holders = [dict(gpu_uuid=gpu, pid=identities[f'rank{i}']['pid']) for i, gpu in enumerate(m.control_queue.GPU_UUIDS)]
    monkeypatch.setattr(m.control_queue, 'gpu_processes', lambda: holders)
    envs = {p['pid']: dict(CUDA_VISIBLE_DEVICES=','.join(m.control_queue.GPU_UUIDS), LOCAL_RANK=str(i))
            for i, p in enumerate(identities[r] for r in ('rank0', 'rank1', 'rank2', 'rank3'))}
    monkeypatch.setattr(m, 'rank_environment', lambda pid: envs[pid])
    output = work/'controlled/runs'/m.IDENTITY
    job = dict(id=m.IDENTITY, config_path=config, output_dir=str(output), config_sha256='config', plan_sha256='primary')
    registration = dict(job_metadata_sha256='job', launch_sha256='launch')
    monkeypatch.setattr(m.native_analysis, 'registration', lambda identity: (job, {}, registration))
    tuning = work/'controlled/full_tuning_queue'
    write_json(tuning/'plan.json', dict(successors=[job]))
    write_json(tuning/'launch_process.json', dict(identity=identities['driver'], plan_sha256=sha256(tuning/'plan.json')))
    write_json(work/'controlled/driver_status.json', dict(phase='training', job_id=m.IDENTITY,
        driver_pid=identities['driver']['pid'], pid=identities['training']['pid'], gpu_uuids=list(m.control_queue.GPU_UUIDS)))
    write_json(tuning/'status.json', dict(phase='executing_registered_job', job_id=m.IDENTITY))
    write_json(output/'runtime_qualification.json', dict(sources={'source.py': sha256(source)}))
    directory = work/'tier1/native_trajectories'/m.IDENTITY/m.native_analysis.digest_json(software())[:16]
    trajectory = directory/'trajectory_through_epoch05.json'
    verification = directory/'verification_through_epoch05.json'
    write_json(trajectory, dict(epochs_analyzed=list(range(1, 6))))
    write_json(verification, dict(trajectory_sha256=sha256(trajectory), checks=[{}]*10,
        source_and_artifact_hashes_verified=True, test_predictions_read=False))
    prefix = work/'tier1/native_trajectory_prefix'/m.IDENTITY/'epoch05'
    write_json(prefix/'plan.json', dict(job=job, registration=registration, native_software=software(),
        analysis_directory=str(directory), files_sha256={str(source): sha256(source)}))
    write_json(prefix/'complete.json', dict(plan_sha256=sha256(prefix/'plan.json'), epochs=list(range(1, 6)),
        snapshots_verified=10, trajectory_sha256=sha256(trajectory), verification_sha256=sha256(verification)))
    write_json(prefix/'exit.json', dict(returncode=0))
    write_json(prefix/'started.json', dict(identity=dict(pid=1, start_ticks=1)))
    return dict(root=root, work=work, source=source, output=output, job=job, live=live, identities=identities,
                inspect=inspect, envs=envs, holders=holders, directory=directory, prefix=prefix, tuning=tuning)


@pytest.fixture
def prepared(active):
    active['plan'] = m.read(m.prepare())
    records = [dict(epoch=i, step=i*10, eval_macro_spearman=.2, eval_loss=1.) for i in range(1, 16)]
    output = active['output']; state = output/'trainer_state.json'
    write_json(state, dict(epoch=15, log_history=records))
    for r in records:
        for prefix in ('eval', 'train_eval'):
            write_json(output/'prediction_exports/trajectory'/f"{prefix}_step{r['step']:08d}.json",
                       dict(epoch=r['epoch'], step=r['step']))
    active['completion'] = output/'controlled_training_complete.json'
    active['done'] = dict(config_sha256='config', plan_sha256='primary', evidence_sha256={'trainer_state.json': sha256(state)})
    return active


def finish(p):
    for role, identity in p['identities'].items():
        if role != 'driver': p['live'].pop(identity['pid'], None)
    write_json(p['completion'], p['done'])


def test_preparation_separates_persistent_driver_from_five_fit_handles(prepared):
    plan = prepared['plan']; m.validate_plan(plan)
    assert set(plan['processes']) == {'training', 'rank0', 'rank1', 'rank2', 'rank3'}
    assert plan['driver'] == prepared['identities']['driver']
    assert plan['previous_epoch'] == 5


@pytest.mark.parametrize('damage', ['device', 'rank', 'command', 'owner', 'launch', 'prefix', 'prefix_live', 'source'])
def test_preparation_rejects_unverified_process_or_prefix(active, damage):
    if damage == 'device': active['holders'][0]['gpu_uuid'] = 'unauthorized'
    elif damage == 'rank': active['envs'][active['identities']['rank0']['pid']]['LOCAL_RANK'] = '1'
    elif damage == 'command': active['identities']['training']['argv'][-1] = 'another.yaml'
    elif damage == 'owner': active['identities']['rank0']['uid'] += 1
    elif damage == 'launch': write_json(active['tuning']/'launch_process.json', dict(identity=active['identities']['driver'], plan_sha256='wrong'))
    elif damage == 'prefix': write_json(active['prefix']/'complete.json', {})
    elif damage == 'prefix_live': active['live'][1] = dict(pid=1, start_ticks=1)
    else: active['source'].write_text('changed')
    with pytest.raises((ValueError, RuntimeError, KeyError)): m.prepare()
    assert not (m.destination()/'plan.json').exists()


def test_completion_can_arrive_before_process_teardown(prepared, monkeypatch):
    write_json(prepared['completion'], prepared['done']); sleeps = []
    def sleep(seconds): sleeps.append(seconds); finish(prepared)
    monkeypatch.setattr(m.time, 'sleep', sleep)
    assert m.wait_for_completion(prepared['plan']) == prepared['completion']
    assert sleeps == [30] and m.observe(prepared['plan'])['driver_live']


def test_waits_through_retention_but_not_coordinator_lifetime(prepared, monkeypatch):
    finish(prepared); prepared['completion'].unlink(); sleeps = []
    def sleep(seconds): sleeps.append(seconds); write_json(prepared['completion'], prepared['done'])
    monkeypatch.setattr(m.time, 'sleep', sleep)
    assert m.wait_for_completion(prepared['plan']) == prepared['completion']
    assert sleeps == [30]


def test_pid_reuse_is_terminal_for_original_fit(prepared):
    finish(prepared); old = prepared['identities']['training']
    prepared['live'][old['pid']] = dict(old, start_ticks=old['start_ticks']+1)
    assert m.wait_for_completion(prepared['plan']) == prepared['completion']


@pytest.mark.parametrize('at', ['first_observation', 'completion_recheck', 'dynamic_status'])
def test_transient_observation_repolls_same_handles(prepared, monkeypatch, at):
    finish(prepared); sleeps = []; calls = 0
    def inspect(pid):
        nonlocal calls
        calls += 1
        if calls == (1 if at == 'first_observation' else 7): raise RuntimeError('transient process read')
        return prepared['inspect'](pid)
    if at == 'dynamic_status':
        original = m.read
        def read(path):
            nonlocal calls
            if path == prepared['work']/'controlled/driver_status.json':
                calls += 1
                if calls == 1: raise OSError('transient status read')
            return original(path)
        monkeypatch.setattr(m, 'read', read)
    else: monkeypatch.setattr(m.control_queue, 'process_identity', inspect)
    monkeypatch.setattr(m.time, 'sleep', lambda seconds: sleeps.append(seconds))
    assert m.wait_for_completion(prepared['plan']) == prepared['completion']
    assert sleeps == [30]


@pytest.mark.parametrize('failure', ['driver_gone', 'queue_failed', 'queue_advanced', 'state_changed', 'missing_snapshot', 'duplicate', 'short_horizon', 'source_changed'])
def test_missing_or_changed_completion_evidence_fails_without_retry(prepared, monkeypatch, failure):
    finish(prepared)
    if failure in ('driver_gone', 'queue_failed', 'queue_advanced'):
        prepared['completion'].unlink()
        if failure == 'driver_gone': prepared['live'].clear()
        else: write_json(prepared['tuning']/'status.json', dict(phase='stopped_for_inspection' if failure == 'queue_failed' else 'executing_registered_job', job_id='next'))
    elif failure in ('state_changed', 'short_horizon'):
        state = prepared['output']/'trainer_state.json'; s = m.read(state); s['epoch'] = 14; write_json(state, s)
        if failure == 'short_horizon':
            prepared['done']['evidence_sha256']['trainer_state.json'] = sha256(state)
            write_json(prepared['completion'], prepared['done'])
    elif failure == 'missing_snapshot': (prepared['output']/'prediction_exports/trajectory/train_eval_step00000150.json').unlink()
    elif failure == 'duplicate': write_json(prepared['output']/'prediction_exports/trajectory/eval_extra.json', dict(epoch=1, step=10))
    else: prepared['source'].write_text('changed')
    monkeypatch.setattr(m.time, 'sleep', lambda _: pytest.fail('Must not retry provenance or completion failure'))
    with pytest.raises((RuntimeError, ValueError)): m.wait_for_completion(prepared['plan'])


@pytest.mark.parametrize('failure', [None, 'analysis', 'short_result', 'verification', 'receipt'])
def test_exactly_one_analysis_and_verification_or_inspection(prepared, monkeypatch, failure):
    finish(prepared); calls = []
    def analyze(identity):
        calls.append('analysis')
        if failure == 'analysis': raise ValueError('bad snapshot')
        return dict(epochs_analyzed=list(range(1, 15 if failure == 'short_result' else 16)),
                    complete_horizon_observed=True, test_predictions_read=False)
    def verify(directory, previous_epoch):
        calls.append('verification'); assert previous_epoch == 5
        trajectory = directory/'trajectory_through_epoch15.json'; write_json(trajectory, dict(epochs=list(range(1, 16))))
        result = dict(checks=[{}]*(28 if failure == 'verification' else 30), source_and_artifact_hashes_verified=True,
                      test_predictions_read=False, trajectory_sha256=sha256(trajectory))
        write_json(directory/'verification_through_epoch15.json', result)
        if failure == 'receipt': trajectory.write_text('changed')
        return result
    monkeypatch.setattr(m.native_analysis, 'main', analyze)
    monkeypatch.setattr(m.verify_native_analysis, 'verify', verify)
    if failure:
        with pytest.raises(ValueError): m.run()
        assert m.read(m.destination()/'exit.json')['returncode'] == 1
        assert not (m.destination()/'complete.json').exists()
    else:
        done = m.run()
        assert done['snapshots_verified'] == 30 and done['new_fits_created'] == 0 and not done['study_complete']
        assert m.read(m.destination()/'exit.json')['returncode'] == 0
    previous_calls = calls.copy()
    with pytest.raises(RuntimeError, match='already started'): m.run()
    assert calls == previous_calls
