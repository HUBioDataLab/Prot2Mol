import copy
import json
import os
from pathlib import Path

import pytest

from experiments.paired_generalization import control_queue as queue
from experiments.paired_generalization.run import sha256, write_json


def test_quota_gate_uses_the_smaller_account_limit_and_handles_overage():
    text = 'Filesystem blocks quota limit grace files quota limit grace\n/dev/archive 100 200 300 1 0 0\n/dev/home 250* 200 300 7days 1 0 0\n'
    assert queue.quota_remaining(text, '/dev/archive') == 100*1024
    assert queue.quota_remaining(text, '/dev/home') == 0
    with pytest.raises(ValueError, match='one numeric quota'):
        queue.quota_remaining(text, '/dev/unknown')
    with pytest.raises(ValueError, match='finite account quota'):
        queue.quota_remaining('/dev/noquota 1 0 0 1 0 0', '/dev/noquota')


def test_process_identity_distinguishes_pid_reuse_and_zombies(tmp_path):
    path = tmp_path/'12'
    path.mkdir()
    (path/'cwd').symlink_to(tmp_path, target_is_directory=True)
    (path/'cmdline').write_bytes(b'python\0run.py\0')

    def state(value, ticks):
        fields = [value]+['0']*18+[str(ticks)]+['0']*10
        (path/'stat').write_text('12 (name with spaces) '+' '.join(fields))

    state('S', 123)
    first = queue.process_identity(12, tmp_path)
    assert first['start_ticks'] == 123 and first['argv'] == ['python', 'run.py']
    state('R', 124)
    assert queue.process_identity(12, tmp_path) != first
    state('Z', 124)
    assert queue.process_identity(12, tmp_path) is None
    assert queue.process_identity(99, tmp_path) is None


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path/'project'
    work = root/'RewardModel/outputs/paired_generalization'
    directory = work/'controlled'
    module = root/'RewardModel/experiments/paired_generalization/control_queue.py'
    module.parent.mkdir(parents=True)
    module.write_text('fixture queue source')
    source = root/'RewardModel/reward_model/core.py'
    source.parent.mkdir(parents=True)
    source.write_text('fixture training source')
    spec = root/'Paired input ranking generalization.md'
    spec.write_text('fixture scientific specification')
    monkeypatch.setattr(queue, 'ROOT', root)
    monkeypatch.setattr(queue, 'WORK', work)
    monkeypatch.setattr(queue, '__file__', str(module))
    monkeypatch.setattr(queue, 'runtime_guard', lambda: None)
    monkeypatch.setattr(queue, 'boot_id', lambda: 'test-boot')
    primary = dict(spec_sha256=sha256(spec), gpu_uuids=list(queue.GPU_UUIDS),
                   tuning_jobs=[dict(id=identity, arm='full') for identity in
                                ('full__A__seed42__lr1p0', queue.PREDECESSOR, *queue.SUCCESSORS)])
    write_json(directory/'plan.json', primary)
    plan_hash = sha256(directory/'plan.json')
    write_json(directory/'preparation_complete.json', dict(plan_sha256=plan_hash))
    jobs = []
    for identity in (queue.PREDECESSOR, *queue.SUCCESSORS):
        config = directory/'configs'/f'{identity}.yaml'
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text('fixture: training config')
        job = dict(id=identity, plan_sha256=plan_hash, config_path=str(config),
                   config_sha256=sha256(config), output_dir=str(directory/'runs'/identity))
        write_json(directory/'jobs'/f'{identity}.json', job)
        jobs.append(job)
    write_json(Path(jobs[0]['output_dir'])/'runtime_qualification.json',
               dict(sources={str(source.relative_to(root)):sha256(source)}))
    write_json(directory/'driver_status.json', dict(job_id=queue.PREDECESSOR,
               phase='training', driver_pid=1, pid=2))
    processes = {}
    for pid in range(1, 7):
        argv = ['python', 'control_driver.py'] if pid == 1 else [
            'python', *(['-m', 'torch.distributed.run'] if pid == 2 else []),
            'run.py', '--config', jobs[0]['config_path']]
        processes[pid] = dict(pid=pid, start_ticks=100+pid, argv=argv,
                              cwd=str(root), uid=os.getuid())
    monkeypatch.setattr(queue, 'process_identity', lambda pid: processes.get(pid))
    holders = [dict(gpu_uuid=gpu,pid=i+3) for i,gpu in enumerate(queue.GPU_UUIDS)]
    monkeypatch.setattr(queue, 'gpu_processes', lambda: holders)
    monkeypatch.setattr(queue.control_driver, 'next_job', lambda *args: next(
        (job for job in jobs if not queue.control_driver.completed_job(job)), None))
    path = queue.prepare(1, 2)
    plan = queue.read(path)
    queue.validate_plan(plan)
    return dict(root=root, work=work, directory=directory, destination=path.parent,
                jobs=jobs, processes=processes, holders=holders, plan=plan, source=source)


def complete(job):
    output = Path(job['output_dir'])
    write_json(output/'evidence.json', dict(test_evaluated=False, finished=True))
    write_json(output/'controlled_training_complete.json', dict(
        config_sha256=job['config_sha256'], plan_sha256=job['plan_sha256'],
        evidence_sha256={'evidence.json':sha256(output/'evidence.json')}))


def ready(prepared, monkeypatch):
    prepared['processes'].clear()
    prepared['holders'].clear()
    complete(prepared['jobs'][0])
    monkeypatch.setattr(queue.control_driver, 'baseline_gate', lambda *args: dict(ready=True))
    monkeypatch.setattr(queue, 'storage_gate', lambda: dict(ready=True, records=[]))
    called = []

    def execute(job, *_):
        called.append(job['id'])
        complete(job)
        return dict(phase='training_complete_test_not_scored')

    monkeypatch.setattr(queue.control_driver, 'execute_job', execute)
    return called


def test_plan_rejects_source_and_config_changes(prepared):
    prepared['source'].write_text('changed training implementation')
    with pytest.raises(ValueError, match='Frozen queue input changed'):
        queue.validate_plan(prepared['plan'])


def test_plan_rejects_changed_training_config(prepared):
    Path(prepared['jobs'][1]['config_path']).write_text('different learning rate')
    with pytest.raises(ValueError, match='Frozen queue input changed'):
        queue.validate_plan(prepared['plan'])


def test_runtime_guard_keeps_coordinator_cpu_and_children_gpu_online(monkeypatch):
    monkeypatch.setattr(queue, 'ROOT', Path('/home/atabey/Prot2Mol'))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('ACCELERATE_USE_CPU', 'false')
    monkeypatch.setenv('WANDB_MODE', 'online')
    queue.runtime_guard()
    monkeypatch.setenv('ACCELERATE_USE_CPU', 'true')
    with pytest.raises(RuntimeError, match='only children expose GPUs'):
        queue.runtime_guard()
    monkeypatch.setenv('ACCELERATE_USE_CPU', 'false')
    monkeypatch.setenv('WANDB_MODE', 'disabled')
    with pytest.raises(RuntimeError, match='WANDB_MODE=online'):
        queue.runtime_guard()


def test_plan_rejects_changed_registered_metadata(prepared):
    job = copy.deepcopy(prepared['jobs'][1])
    job['output_dir'] += '_different'
    write_json(prepared['directory']/'jobs'/f"{job['id']}.json", job)
    with pytest.raises(ValueError, match='metadata differs'):
        queue.validate_plan(prepared['plan'])


def test_prepare_refuses_existing_successor(prepared):
    Path(prepared['jobs'][1]['output_dir']).mkdir()
    with pytest.raises(ValueError, match='successor output already exists'):
        queue.prepare(1, 2)


def test_prepare_refuses_foreign_or_missing_live_ranks(prepared):
    prepared['processes'][3]['uid'] = os.getuid()+1
    with pytest.raises(ValueError, match='authorized owner'):
        queue.prepare(1, 2)
    del prepared['processes'][3]
    with pytest.raises(ValueError, match='six distinct original'):
        queue.prepare(1, 2)


def test_missing_predecessor_completion_never_starts_successors(prepared, monkeypatch):
    prepared['processes'].clear()
    monkeypatch.setattr(queue.control_driver, 'execute_job', lambda *args: pytest.fail('Must not train'))
    with pytest.raises(RuntimeError, match='without verified completion'):
        queue.run()
    assert queue.read(prepared['destination']/'status.json')['phase'] == 'stopped_for_inspection'
    with pytest.raises(RuntimeError, match='already invoked'):
        queue.run()


def test_live_predecessor_handles_are_waited_even_if_receipt_exists(prepared, monkeypatch):
    original = dict(prepared['processes'])
    called = ready(prepared, monkeypatch)
    prepared['processes'].update(original)
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        assert called == []
        assert queue.read(prepared['destination']/'status.json')['phase'] == 'waiting_for_original_processes'
        prepared['processes'].clear()

    monkeypatch.setattr(queue.time, 'sleep', sleep)
    result = queue.run()
    assert waits == [30] and called == list(queue.SUCCESSORS)
    assert not result['study_complete'] and len(result['evidence']) == 5


@pytest.mark.parametrize('gate', ['gpu', 'storage', 'lock'])
def test_waiting_gates_do_not_start_training_or_hold_global_lock(prepared, monkeypatch, gate):
    import fcntl
    called = ready(prepared, monkeypatch)
    lock = (prepared['directory']/'driver.lock').open('a')
    if gate == 'gpu':
        prepared['holders'].append(dict(gpu_uuid=queue.GPU_UUIDS[0],pid=99))
    elif gate == 'storage':
        monkeypatch.setattr(queue, 'storage_gate', lambda: dict(ready=False,records=[]))
    else:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        assert called == []
        fcntl.flock(lock, fcntl.LOCK_UN)
        # The queue must have released its own descriptor before waiting.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_UN)
        prepared['holders'].clear()
        monkeypatch.setattr(queue, 'storage_gate', lambda: dict(ready=True,records=[]))

    monkeypatch.setattr(queue.time, 'sleep', sleep)
    try:
        queue.run()
    finally:
        lock.close()
    assert waits == [30] and called == list(queue.SUCCESSORS)


def test_failed_or_partial_successor_stops_without_retry(prepared, monkeypatch):
    called = ready(prepared, monkeypatch)

    def fail(job, *_):
        called.append(job['id'])
        Path(job['output_dir']).mkdir()
        raise RuntimeError('training failed')

    monkeypatch.setattr(queue.control_driver, 'execute_job', fail)
    with pytest.raises(RuntimeError, match='training failed'):
        queue.run()
    assert called == [queue.SUCCESSORS[0]]
    assert not (prepared['destination']/'complete.json').exists()
    with pytest.raises(RuntimeError, match='already invoked'):
        queue.run()


def test_changed_completion_evidence_stops_queue(prepared, monkeypatch):
    ready(prepared, monkeypatch)
    (Path(prepared['jobs'][0]['output_dir'])/'evidence.json').write_text('changed')
    with pytest.raises(ValueError, match='evidence changed'):
        queue.run()


def test_driver_success_without_completion_receipt_is_not_accepted(prepared, monkeypatch):
    ready(prepared, monkeypatch)
    monkeypatch.setattr(queue.control_driver, 'execute_job', lambda *args: dict(phase='training_complete_test_not_scored'))
    with pytest.raises(RuntimeError, match='without verified completion'):
        queue.run()


def test_queue_uses_global_lock_during_execution(prepared, monkeypatch):
    import fcntl
    called = ready(prepared, monkeypatch)
    execute = queue.control_driver.execute_job

    def check(job, *args):
        with (prepared['directory']/'driver.lock').open('a') as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return execute(job, *args)

    monkeypatch.setattr(queue.control_driver, 'execute_job', check)
    queue.run()
    assert called == list(queue.SUCCESSORS)


def test_last_gpu_recheck_can_wait_without_retrying_a_started_fit(prepared, monkeypatch):
    called = ready(prepared, monkeypatch)
    execute = queue.control_driver.execute_job
    busy = [True]
    waits = []

    def recheck(job, *args):
        if busy.pop() if busy else False:
            assert not Path(job['output_dir']).exists()
            return dict(phase='waiting_for_authorized_gpus', live_processes=[dict(pid=99)])
        return execute(job, *args)

    monkeypatch.setattr(queue.control_driver, 'execute_job', recheck)
    monkeypatch.setattr(queue.time, 'sleep', lambda seconds: waits.append(seconds))
    queue.run()
    assert waits == [30] and called == list(queue.SUCCESSORS)
