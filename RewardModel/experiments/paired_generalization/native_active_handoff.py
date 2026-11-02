"""Audit the current full-A LR0.3 fit after its own five training handles exit.

The tuning coordinator outlives this fit. Observe it separately so retention can
finish, but do not wait for its later fits. This CPU worker never starts fitting,
model inference, test scoring or a replacement process.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from . import native_horizon_queue as base
else:
    from run import ROOT, WORK, sha256, write_json
    import native_horizon_queue as base

control_queue = base.control_queue
control_driver = base.control_driver
native_analysis = base.native_analysis
verify_native_analysis = base.verify_native_analysis
read, immutable = control_queue.read, control_queue.immutable
runtime_guard = base.runtime_guard
IDENTITY = 'full__A__seed42__lr0p3'
PREVIOUS_EPOCH = 5


class ProcessObservationError(RuntimeError):
    pass


def inspect_process(pid):
    try:
        return control_queue.process_identity(pid)
    except (OSError, RuntimeError) as error:
        raise ProcessObservationError(str(error)) from error


def live_originals(plan):
    return {role: p for role, p in plan['processes'].items() if inspect_process(p['pid']) == p}


def require_completion(plan):
    if live_originals(plan):
        raise RuntimeError('Original training processes are still live')
    # Process observation is isolated above. Reuse the frozen horizon/evidence
    # validator without a second, unclassified process read inside that helper.
    return base.require_completion(dict(plan, processes={}))


def destination():
    return WORK/'tier1/native_trajectory_handoff'/IDENTITY


def software():
    result = base.software()
    path = Path(__file__)
    result['source_sha256'][str(path.relative_to(ROOT))] = sha256(path)
    return result


def rank_environment(pid):
    return dict(item.split('=', 1) for item in (Path('/proc')/str(pid)/'environ').read_text().split('\0') if '=' in item)


def capture_handles(job, launch):
    status = read(WORK/'controlled/driver_status.json')
    driver = launch['identity']
    python = str(ROOT/'.venv/bin/python')
    command = 'RewardModel/experiments/paired_generalization/run.py'
    config_args = ['train', '--config', job['config_path']]
    if (status.get('phase') != 'training' or status.get('job_id') != IDENTITY or
        status.get('driver_pid') != driver['pid'] or status.get('gpu_uuids') != list(control_queue.GPU_UUIDS) or
        driver['argv'] != [python, 'RewardModel/experiments/paired_generalization/control_queue.py', 'run']):
        raise ValueError('Expected the existing tuning coordinator and current four-GPU A fit')
    holders = control_queue.gpu_processes()
    if (len(holders) != 4 or {r['gpu_uuid'] for r in holders} != set(control_queue.GPU_UUIDS) or
        len({r['pid'] for r in holders}) != 4):
        raise ValueError('Expected four ranks on exactly the authorized GPUs')
    pids = dict(training=status['pid'], **{f'rank{i}': next(r['pid'] for r in holders if r['gpu_uuid'] == gpu)
        for i, gpu in enumerate(control_queue.GPU_UUIDS)})
    processes = {role: control_queue.process_identity(pid) for role, pid in pids.items()}
    if len({driver['pid'], *pids.values()}) != 6:
        raise ValueError('Expected six distinct process identities')
    for role, identity in dict(driver=driver, **processes).items():
        if (identity is None or control_queue.process_identity(identity['pid']) != identity or
            identity['uid'] != os.getuid() or identity['cwd'] != str(ROOT)):
            raise ValueError('Prepare only with the original owner/project processes live')
        if role == 'driver':
            continue
        expected = ([python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4', command]
                    if role == 'training' else [python, '-u', command]) + config_args
        if identity['argv'] != expected:
            raise ValueError('Training process command differs from this registered fit')
        if role.startswith('rank'):
            env = rank_environment(identity['pid'])
            if (env.get('CUDA_VISIBLE_DEVICES', '').split(',') != list(control_queue.GPU_UUIDS) or
                env.get('LOCAL_RANK') != role[4:]):
                raise ValueError('Rank device allocation differs from the authorized mapping')
    return driver, processes


def prepare():
    runtime_guard()
    tuning_path = WORK/'controlled/full_tuning_queue/plan.json'
    launch_path = tuning_path.with_name('launch_process.json')
    tuning, launch = read(tuning_path), read(launch_path)
    job, _, registration = native_analysis.registration(IDENTITY)
    if ([j for j in tuning['successors'] if j['id'] == IDENTITY] != [job] or
        launch['plan_sha256'] != sha256(tuning_path)):
        raise ValueError('Current job is not registered in the existing tuning queue')
    driver, processes = capture_handles(job, launch)
    prefix = WORK/'tier1/native_trajectory_prefix'/IDENTITY/'epoch05'
    previous_plan = read(prefix/'plan.json'); done = read(prefix/'complete.json')
    if previous_plan['native_software'] != native_analysis.software():
        raise ValueError('The verified prefix used different native analysis sources or packages')
    directory = WORK/'tier1/native_trajectories'/IDENTITY/native_analysis.digest_json(native_analysis.software())[:16]
    trajectory = directory/'trajectory_through_epoch05.json'
    verification = directory/'verification_through_epoch05.json'
    verified = read(verification)
    if (previous_plan['job'] != job or previous_plan['registration'] != registration or
        previous_plan['analysis_directory'] != str(directory) or
        done['plan_sha256'] != sha256(prefix/'plan.json') or
        done['epochs'] != list(range(1, 6)) or done['snapshots_verified'] != 10 or
        done['trajectory_sha256'] != sha256(trajectory) or done['verification_sha256'] != sha256(verification) or
        read(trajectory)['epochs_analyzed'] != list(range(1, 6)) or
        verified['trajectory_sha256'] != sha256(trajectory) or len(verified['checks']) != 10 or
        not verified['source_and_artifact_hashes_verified'] or verified['test_predictions_read'] or
        read(prefix/'exit.json')['returncode'] != 0):
        raise ValueError('Expected the independently verified five-epoch prefix')
    prefix_worker = read(prefix/'started.json')['identity']
    if control_queue.process_identity(prefix_worker['pid']) == prefix_worker:
        raise RuntimeError('The previous prefix analysis is still live')
    qualification = Path(job['output_dir'])/'runtime_qualification.json'
    sources = read(qualification)['sources']
    files = {str(Path(p).relative_to(ROOT)): h for p, h in previous_plan['files_sha256'].items()}
    paths = [tuning_path, launch_path, trajectory, verification, qualification,
             *[prefix/name for name in ('plan.json', 'complete.json', 'exit.json', 'started.json')]]
    for path in paths:
        files[str(path.relative_to(ROOT))] = sha256(path)
    for name, digest in {**files, **sources}.items():
        if sha256(ROOT/name) != digest:
            raise ValueError('Frozen prefix or runtime evidence changed: '+name)
    plan = dict(job=job, processes=processes, driver=driver, boot_id=control_queue.boot_id(),
        software=software(), registration=registration, directory=str(directory), previous_epoch=PREVIOUS_EPOCH,
        files_sha256=files, training_sources_sha256=sources,
        scope='One CPU audit of exactly15 completed validation/probe epochs; no fit, inference, test scoring or retries.')
    if capture_handles(job, launch) != (driver, processes):
        raise RuntimeError('Original process identities changed during preparation')
    immutable(destination()/'plan.json', plan)
    validate_plan(plan)
    return destination()/'plan.json'


def validate_plan(plan):
    runtime_guard()
    if (read(destination()/'plan.json') != plan or plan['boot_id'] != control_queue.boot_id() or
        plan['job']['id'] != IDENTITY or plan['previous_epoch'] != PREVIOUS_EPOCH or plan['software'] != software() or
        set(plan['processes']) != {'training', 'rank0', 'rank1', 'rank2', 'rank3'}):
        raise ValueError('CPU handoff source, scope or host changed')
    job, _, registration = native_analysis.registration(IDENTITY)
    if job != plan['job'] or registration != plan['registration']:
        raise ValueError('Training registration changed')
    for name, digest in {**plan['files_sha256'], **plan['training_sources_sha256']}.items():
        if sha256(ROOT/name) != digest:
            raise ValueError('Frozen handoff evidence changed: '+name)


def observe(plan):
    live = live_originals(plan)
    driver_live = inspect_process(plan['driver']['pid']) == plan['driver']
    statuses = {}
    if not live and driver_live:
        try:
            statuses = dict(driver=read(WORK/'controlled/driver_status.json'),
                            queue=read(WORK/'controlled/full_tuning_queue/status.json'))
        except (OSError, json.JSONDecodeError) as error:
            raise ProcessObservationError(str(error)) from error
    return dict(live_processes=live, driver_live=driver_live, statuses=statuses)


def wait_for_completion(plan):
    while True:
        # Provenance failures are never converted into transient observations.
        validate_plan(plan)
        try:
            observed = observe(plan)
            if not observed['live_processes'] and control_driver.completed_job(plan['job']):
                return require_completion(plan)
        except ProcessObservationError as error:
            write_json(destination()/'status.json', dict(phase='retrying_process_observation', checked_unix=time.time(),
                error_type=type(error).__name__, error=str(error)))
            time.sleep(30)
            continue
        if observed['live_processes']:
            phase = 'waiting_for_fit_processes'
        else:
            statuses = observed['statuses']
            if (not observed['driver_live'] or statuses['queue'].get('phase') == 'stopped_for_inspection' or
                statuses['driver'].get('job_id') != IDENTITY or statuses['queue'].get('job_id') != IDENTITY):
                raise RuntimeError('Fit ended without verified completion; inspect without restarting')
            phase = 'waiting_for_driver_completion_receipt'
        write_json(destination()/'status.json', dict(phase=phase, checked_unix=time.time(), **observed))
        time.sleep(30)


def run():
    import fcntl
    runtime_guard(); directory = destination(); plan = read(directory/'plan.json')
    with (directory/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate_plan(plan)
        if (directory/'started.json').exists():
            raise RuntimeError('This one-shot handoff already started; inspect without restarting')
        immutable(directory/'started.json', dict(identity=control_queue.process_identity(os.getpid()), started_unix=time.time()))
        try:
            completion = wait_for_completion(plan)
            write_json(directory/'status.json', dict(phase='analyzing_complete_trajectory', started_unix=time.time()))
            result = native_analysis.main(IDENTITY)
            if result['epochs_analyzed'] != list(range(1, 16)) or not result['complete_horizon_observed'] or result['test_predictions_read']:
                raise ValueError('Expected all fifteen validation/probe epochs only')
            write_json(directory/'status.json', dict(phase='independently_verifying_trajectory', started_unix=time.time()))
            verified = verify_native_analysis.verify(Path(plan['directory']), previous_epoch=PREVIOUS_EPOCH)
            if len(verified['checks']) != 30 or not verified['source_and_artifact_hashes_verified'] or verified['test_predictions_read']:
                raise ValueError('Independent verification is incomplete')
            wait_for_completion(plan)
            trajectory = Path(plan['directory'])/'trajectory_through_epoch15.json'
            verification = Path(plan['directory'])/'verification_through_epoch15.json'
            if read(verification) != verified or sha256(trajectory) != verified['trajectory_sha256']:
                raise ValueError('Final trajectory or verification receipt changed')
            done = dict(plan_sha256=sha256(directory/'plan.json'), training_completion_sha256=sha256(completion),
                trajectory_sha256=sha256(trajectory), verification_sha256=sha256(verification),
                finished_unix=time.time(), epochs=result['epochs_analyzed'], snapshots_verified=30,
                new_fits_created=0, model_inference=False, test_predictions_read=False, study_complete=False)
            immutable(directory/'complete.json', done)
            write_json(directory/'status.json', dict(phase='complete', finished_unix=done['finished_unix']))
            immutable(directory/'exit.json', dict(returncode=0, finished_unix=time.time()))
            return done
        except BaseException as error:
            failed = dict(returncode=1, finished_unix=time.time(), error_type=type(error).__name__, error=str(error))
            immutable(directory/'exit.json', failed)
            write_json(directory/'status.json', dict(phase='failed_requires_inspection', **failed))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    args = parser.parse_args()
    print(prepare() if args.action == 'prepare' else json.dumps(run()), flush=True)
