"""Fixed requested configurations and one-shot scheduling, with simulated fits."""
import copy
import os
from pathlib import Path

import pytest

from experiments.paired_generalization import requested_single_seed as module
from experiments.paired_generalization.run import sha256, write_json
from test_paired_generalization_control_plan import ROOT, noise_fixture


def complete(job):
    output = Path(job['output_dir'])
    write_json(output / 'evidence.json', dict(finished=True, test_evaluated=False))
    write_json(output / 'controlled_training_complete.json', dict(
        config_sha256=job['config_sha256'], plan_sha256=job['plan_sha256'],
        evidence_sha256={'evidence.json': sha256(output / 'evidence.json')}))


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / 'project'; work = root / 'RewardModel/outputs/paired_generalization'
    directory = work / 'controlled'
    source = root / 'RewardModel/experiments/paired_generalization/requested_single_seed.py'
    source.parent.mkdir(parents=True); source.write_text('fixture requested queue')
    old_queue_source = source.with_name('control_queue.py'); old_queue_source.write_text('fixture old queue')
    training_source = root / 'RewardModel/reward_model/core.py'
    training_source.parent.mkdir(parents=True); training_source.write_text('fixture native training source')
    (root / 'Paired input ranking generalization.md').write_bytes((ROOT / 'Paired input ranking generalization.md').read_bytes())
    monkeypatch.setattr(module, 'ROOT', root); monkeypatch.setattr(module, 'WORK', work)
    monkeypatch.setattr(module, '__file__', str(source))
    monkeypatch.setattr(module.control_queue, '__file__', str(old_queue_source))
    monkeypatch.setattr(module.control_queue, 'runtime_guard', lambda: None)
    monkeypatch.setattr(module.control_queue, 'boot_id', lambda: 'fixture-boot')
    processes = {}; holders = []; calls = []
    monkeypatch.setattr(module.control_queue, 'process_identity', lambda pid: processes.get(pid))
    monkeypatch.setattr(module.control_queue, 'gpu_processes', lambda: holders)
    monkeypatch.setattr(module.control_queue, 'storage_gate', lambda: dict(ready=True))
    monkeypatch.setattr(module.control_driver, 'baseline_gate', lambda *args: dict(ready=True))
    primary = module.control_plan.build_plan(ROOT, work, noise_fixture())
    primary_path = directory / 'plan.json'; write_json(primary_path, primary)
    for job in primary['tuning_jobs']:
        if (job['arm'] in (*module.ARMS, 'full') and job['learning_rate_multiplier'] == 1.):
            module.control_plan.materialize_job(job, directory, sha256(primary_path))
    full = next(a for a in primary['arms'] if a['name'] == 'full')
    identity = module.control_plan.job_id('full', 'B', 43, 1.)
    config = module.control_plan.make_config(primary['bases']['B'], full, regime='B', seed=43,
        multiplier=1., output=directory / 'runs' / identity, training_noise_rms=primary['training_noise_rms'])
    module.control_plan.materialize_job(dict(id=identity, arm='full', regime='B', seed=43,
        learning_rate_multiplier=1., phase='confirmation', config=config), directory, sha256(primary_path))
    sources = {str(training_source.relative_to(root)): sha256(training_source)}
    signature = module.control_plan.digest_json(dict(sources=sources, plan_sha256=sha256(primary_path)))
    cache = directory / 'qualifications' / signature[:16]
    write_json(cache / 'proof.json', dict(qualified=True, world_size=4))
    qualification = dict(signature=signature, sources=sources, world_size=4,
        evidence_sha256={'proof.json': sha256(cache / 'proof.json')})
    write_json(cache / 'complete.json', qualification)
    for regime, seed in (('A', 42), ('B', 42), ('B', 43)):
        identity = module.control_plan.job_id('full', regime, seed, 1.)
        job = module.read(directory / 'jobs' / (identity + '.json')); complete(job)
        output = Path(job['output_dir'])
        write_json(output / 'runtime_qualification.json', qualification)
        peak = 6 if seed == 42 else 5; score = .18458 if seed == 42 else .17739
        write_json(output / 'trainer_state.json', dict(epoch=15, log_history=[dict(epoch=e,
            eval_loss=20-e, eval_macro_spearman=score - .001*abs(e-peak)) for e in range(1, 16)]))
    stop = work / 'protocol_amendments/stop_hpo_20261103'
    retired = dict(pid=11, start_ticks=100, uid=os.getuid(), cwd=str(root), argv=['old_queue'])
    write_json(stop / 'stop_intent.json', dict(groups={'old': dict(members=[retired])}))
    write_json(stop / 'stop_complete.json', dict(phase='user_requested_stop_complete',
        intent_sha256=sha256(stop / 'stop_intent.json'), parameter_optimization_cancelled=True, outputs_preserved=True))
    monkeypatch.setattr(module, 'STOP_SHA256', sha256(stop / 'stop_complete.json'))
    path = module.prepare(); plan = module.read(path); module.validate_plan(plan)
    def execute(job, *_):
        import fcntl
        with (directory / 'driver.lock').open('a') as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        calls.append(job); complete(job)
        return dict(phase='training_complete_test_not_scored')
    monkeypatch.setattr(module.control_driver, 'execute_job', execute)
    monkeypatch.setattr(module.control_driver, 'next_job', lambda *args: pytest.fail('No legacy grid traversal'))
    monkeypatch.setattr(module.control_plan, 'confirm', lambda *args: pytest.fail('No new seed confirmations'))
    return dict(root=root, work=work, directory=directory, source=source, training_source=training_source,
        cache=cache, primary=primary, plan=plan, target=path.parent, calls=calls, holders=holders,
        processes=processes, retired=retired)


def test_exact_requested_scope_uses_one_seed_and_one_lr_without_repeating_reference(prepared):
    plan = prepared['plan']; jobs = plan['jobs']
    assert len(jobs) == 24 and len({j['id'] for j in jobs}) == 24
    assert [j['arm'] for j in jobs[::2]] == list(module.ARMS)
    assert all((j['seed'], j['learning_rate_multiplier']) == (42, 1.) for j in jobs)
    assert {j['regime'] for j in jobs} == {'A', 'B'} and all(j['arm'] != 'full' for j in jobs)
    assert plan['seed_choice']['selected_seed'] == 42
    assert not plan['seed_choice']['test_used_for_choice'] and not plan['seed_choice']['seed_uncertainty_available']
    assert plan['accounting'] == dict(new_training_fits=24, reused_reference_fits=2,
        new_seed_fits=0, lr_candidates_per_variant=1, confirmation_fits=0, epochs_per_fit=15)
    result = module.run()
    assert [j['id'] for j in prepared['calls']] == [j['id'] for j in jobs]
    assert result['training_fits_created'] == 24 and not result['parameter_search']
    assert not result['test_scoring_complete'] and not result['scientific_study_complete']
    with pytest.raises(RuntimeError, match='already invoked'): module.run()


@pytest.mark.parametrize('damage', ['seed', 'lr', 'arm', 'followup', 'device', 'jobs', 'config',
    'source', 'native_source', 'qualification', 'reference', 'stop', 'old_process'])
def test_changed_scope_or_unverified_inputs_cannot_launch_any_fit(prepared, damage):
    path = prepared['target'] / 'plan.json'; plan = copy.deepcopy(prepared['plan'])
    if damage in ('seed', 'lr', 'arm', 'followup', 'device', 'jobs'):
        if damage == 'seed': plan['seed'] = 43
        elif damage == 'lr': plan['learning_rate_multiplier'] = .3
        elif damage == 'arm': plan['arms'].append('bce_low')
        elif damage == 'followup': plan['followup_experiments'].append('E59')
        elif damage == 'device': plan['gpu_uuids'][0] = 'not-authorized'
        else: plan['jobs'].pop()
        write_json(path, plan)
    elif damage == 'config': Path(plan['jobs'][0]['config_path']).write_text('changed')
    elif damage == 'source': prepared['source'].write_text('changed')
    elif damage == 'native_source': prepared['training_source'].write_text('changed')
    elif damage == 'qualification': (prepared['cache'] / 'complete.json').unlink()
    elif damage == 'reference': Path(plan['references']['completion'][0]['path']).unlink()
    elif damage == 'stop': (prepared['work'] / 'protocol_amendments/stop_hpo_20261103/stop_complete.json').write_text('{}')
    else: prepared['processes'][11] = prepared['retired']
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)): module.run()
    assert not prepared['calls']


@pytest.mark.parametrize('gate', ['gpu', 'quota', 'lock', 'last_gpu_check'])
def test_resource_wait_preserves_scope_without_duplicate_fitting(prepared, monkeypatch, gate):
    import fcntl
    handle = (prepared['directory'] / 'driver.lock').open('a'); original = module.control_driver.execute_job
    once = [True]
    if gate == 'gpu': prepared['holders'].append(dict(pid=9, gpu_uuid=module.control_plan.GPU_UUIDS[0]))
    elif gate == 'quota': monkeypatch.setattr(module.control_queue, 'storage_gate', lambda: dict(ready=False))
    elif gate == 'lock': fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    else:
        def busy(*args):
            if once: once.pop(); return dict(phase='waiting_for_authorized_gpus')
            return original(*args)
        monkeypatch.setattr(module.control_driver, 'execute_job', busy)
    waits = []
    def sleep(seconds):
        waits.append(seconds); assert not prepared['calls']
        prepared['holders'].clear(); fcntl.flock(handle, fcntl.LOCK_UN)
        monkeypatch.setattr(module.control_queue, 'storage_gate', lambda: dict(ready=True))
    monkeypatch.setattr(module.time, 'sleep', sleep)
    try: module.run()
    finally: handle.close()
    assert waits == [60] and len(prepared['calls']) == 24


@pytest.mark.parametrize('failure', ['partial_output', 'failed_worker', 'missing_completion', 'changed_reference'])
def test_interrupted_or_changed_work_stops_without_automatic_retry(prepared, monkeypatch, failure):
    if failure == 'partial_output': Path(prepared['plan']['jobs'][0]['output_dir']).mkdir()
    else:
        original = module.control_driver.execute_job
        def fail(job, *args):
            if failure == 'changed_reference':
                result = original(job, *args)
                Path(prepared['plan']['references']['completion'][0]['path']).write_text('{}')
                return result
            prepared['calls'].append(job); Path(job['output_dir']).mkdir()
            if failure == 'failed_worker': raise RuntimeError('fixture native failure')
            return dict(phase='training_complete_test_not_scored')
        monkeypatch.setattr(module.control_driver, 'execute_job', fail)
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)): module.run()
    assert not (prepared['target'] / 'complete.json').exists()
    assert module.read(prepared['target'] / 'status.json')['phase'] == 'stopped_for_inspection'
    before = len(prepared['calls'])
    with pytest.raises((ValueError, RuntimeError)): module.run()
    assert len(prepared['calls']) == before


def test_no_changed_or_expanded_fixed_config_can_be_registered(prepared):
    primary = copy.deepcopy(prepared['primary'])
    job = next(j for j in primary['tuning_jobs'] if j['arm'] == 'frozen' and j['learning_rate_multiplier'] == 1.)
    job['config']['training']['learning_rate'] *= 3
    with pytest.raises(ValueError, match='fixed configuration'): module.scope_jobs(primary)


def test_existing_seed_evidence_is_checked_without_creating_another_seed_fit(prepared):
    before = set((prepared['directory'] / 'runs').iterdir())
    result = module.seed_evidence()
    assert [r['best_validation_macro_spearman'] for r in result['candidates']] == pytest.approx([.18458, .17739])
    assert set((prepared['directory'] / 'runs').iterdir()) == before
