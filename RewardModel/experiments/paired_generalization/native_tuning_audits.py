"""CPU audits of the three remaining fits in the existing full tuning queue.

No fit, inference, selection or test scoring. Completion receipts are written by
the existing driver only after its subprocesses return and retention completes.
The first fit's observed launcher must also terminate. Future fits can complete
while this worker is analyzing an earlier one; their verified receipts suffice.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

if __package__:
    from . import native_active_handoff as base
    from .run import ROOT, WORK, sha256, write_json
else:
    import native_active_handoff as base
    from run import ROOT, WORK, sha256, write_json

queue = base.control_queue
driver = base.control_driver
analysis = base.native_analysis
verifier = base.verify_native_analysis
read, immutable, runtime_guard = base.read, base.immutable, base.runtime_guard
IDENTITIES = ('full__B__seed42__lr0p3', 'full__A__seed42__lr3p0', 'full__B__seed42__lr3p0')


def destination():
    return WORK/'tier1/native_remaining_tuning_audits'


def software():
    result = base.software()
    path = Path(__file__)
    result['source_sha256'][str(path.relative_to(ROOT))] = sha256(path)
    return result


def validate_tuning(tuning):
    # The training coordinator's validator expects its own environment, even
    # though validation only reads files. This worker never exposes a GPU.
    previous = {k: os.environ[k] for k in ('ACCELERATE_USE_CPU', 'WANDB_MODE')}
    try:
        os.environ.update(ACCELERATE_USE_CPU='false', WANDB_MODE='online')
        queue.validate_plan(tuning)
    finally:
        os.environ.update(previous)


def analysis_directory(identity):
    return WORK/'tier1/native_trajectories'/identity/analysis.digest_json(analysis.software())[:16]


def prepare():
    runtime_guard()
    tuning_dir = WORK/'controlled/full_tuning_queue'
    tuning = read(tuning_dir/'plan.json'); validate_tuning(tuning)
    jobs = tuning['successors'][1:]
    if tuple(j['id'] for j in jobs) != IDENTITIES:
        raise ValueError('Expected exactly the three remaining registered tuning fits')
    launch = read(tuning_dir/'launch_process.json'); parent = launch['identity']
    status = read(WORK/'controlled/driver_status.json')
    python = str(ROOT/'.venv/bin/python')
    expected = [python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                'RewardModel/experiments/paired_generalization/run.py', 'train', '--config', jobs[0]['config_path']]
    current = base.inspect_process(status['pid'])
    if (launch['plan_sha256'] != sha256(tuning_dir/'plan.json') or
        base.inspect_process(parent['pid']) != parent or parent['uid'] != os.getuid() or
        parent['cwd'] != str(ROOT) or
        parent['argv'] != [python, 'RewardModel/experiments/paired_generalization/control_queue.py', 'run'] or
        status.get('phase') != 'training' or status.get('job_id') != IDENTITIES[0] or
        status.get('driver_pid') != parent['pid'] or status.get('gpu_uuids') != list(queue.GPU_UUIDS) or
        current is None or current['argv'] != expected or current['uid'] != os.getuid() or
        current['cwd'] != str(ROOT)):
        raise RuntimeError('Prepare only with the existing B fit and tuning coordinator live')
    previous = read(base.destination()/'plan.json'); base.validate_plan(previous)
    done = read(base.destination()/'complete.json')
    previous_dir = Path(previous['directory'])
    previous_completion = base.require_completion(previous)
    previous_verification = read(previous_dir/'verification_through_epoch15.json')
    if (base.inspect_process(read(base.destination()/'started.json')['identity']['pid']) ==
        read(base.destination()/'started.json')['identity'] or
        read(base.destination()/'exit.json')['returncode'] != 0 or done['snapshots_verified'] != 30 or
        done['plan_sha256'] != sha256(base.destination()/'plan.json') or
        done['training_completion_sha256'] != sha256(previous_completion) or
        done['trajectory_sha256'] != sha256(previous_dir/'trajectory_through_epoch15.json') or
        done['verification_sha256'] != sha256(previous_dir/'verification_through_epoch15.json') or
        len(previous_verification['checks']) != 30 or not previous_verification['source_and_artifact_hashes_verified'] or
        previous_verification['test_predictions_read']):
        raise ValueError('The preceding A audit must have completed successfully')
    for job in jobs:
        if analysis_directory(job['id']).exists():
            raise RuntimeError('Native analysis output already exists; inspect without adopting or retrying')
    if any(Path(j['output_dir']).exists() for j in jobs[1:]):
        raise RuntimeError('Prepare before the two remaining higher-rate fits start')
    registered, _, registration = analysis.registration(IDENTITIES[0])
    if registered != jobs[0]:
        raise ValueError('Current B launch registration differs')
    qualification = Path(jobs[0]['output_dir'])/'runtime_qualification.json'
    files = dict(previous['files_sha256'])
    files.update(previous['training_sources_sha256'])
    paths = [tuning_dir/'plan.json', tuning_dir/'launch_process.json', qualification, previous_completion,
             previous_dir/'trajectory_through_epoch15.json', previous_dir/'verification_through_epoch15.json',
             *[base.destination()/n for n in ('plan.json', 'complete.json', 'exit.json', 'started.json')]]
    for path in paths:
        files[str(path.relative_to(ROOT))] = sha256(path)
    plan = dict(jobs=jobs, tuning=tuning, parent=parent, current_launcher=current,
        first_registration=registration, runtime_qualification_sha256=sha256(qualification),
        boot_id=queue.boot_id(), software=software(), files_sha256=files,
        scope='Exactly three CPU audits of existing full tuning fits; no training, inference, test data, LR selection or retries.')
    immutable(destination()/'plan.json', plan)
    validate_plan(plan)
    return destination()/'plan.json'


def validate_plan(plan):
    runtime_guard()
    if (read(destination()/'plan.json') != plan or plan['boot_id'] != queue.boot_id() or
        plan['software'] != software() or tuple(j['id'] for j in plan['jobs']) != IDENTITIES or
        plan['jobs'] != plan['tuning']['successors'][1:]):
        raise ValueError('CPU audit host, source or scope changed')
    validate_tuning(plan['tuning'])
    for name, digest in plan['files_sha256'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError('Frozen audit input changed: '+name)


def completed_evidence(plan, job):
    if not driver.completed_job(job):
        return None
    output = Path(job['output_dir'])
    receipt = read(output/'controlled_training_complete.json')
    expected = {'experiment_summary.json', 'trainer_state.json', 'retained/retention.json',
                *{f'rank_{rank}_complete.json' for rank in range(4)}}
    if (receipt['job_id'] != job['id'] or receipt['phase'] != 'training_complete_test_not_scored' or
        set(receipt['evidence_sha256']) != expected):
        raise ValueError('Training completion omits expected rank or retention evidence')
    summary = read(output/'experiment_summary.json')
    if summary['test_evaluated'] or summary['test_metrics']:
        raise ValueError('Tuning fit unexpectedly scored test examples')
    for rank in range(4):
        done = read(output/f'rank_{rank}_complete.json')
        if done['rank'] != rank or done['world_size'] != 4 or done['config_sha256'] != job['config_sha256']:
            raise ValueError('Completed rank differs from the registered four-GPU fit')
    if job['id'] == IDENTITIES[0] and base.inspect_process(plan['current_launcher']['pid']) == plan['current_launcher']:
        return None
    registered, _, registration = analysis.registration(job['id'])
    if registered != job or (job['id'] == IDENTITIES[0] and registration != plan['first_registration']):
        raise ValueError('Completed launch registration differs')
    qualification = Path(job['output_dir'])/'runtime_qualification.json'
    if sha256(qualification) != plan['runtime_qualification_sha256']:
        raise ValueError('Completed fit used a different runtime qualification')
    # The original driver waits for torchrun (and B allocation exports), then
    # records all four rank completions and retention. Reuse its full-horizon
    # validator; this worker does not need to observe every future fit live.
    completion = base.base.require_completion(dict(job=job, processes={}))
    return dict(job_id=job['id'], registration=registration,
        training_completion_sha256=sha256(completion), runtime_qualification_sha256=sha256(qualification))


def wait_for_completion(plan, job):
    while True:
        validate_plan(plan)
        try:
            evidence = completed_evidence(plan, job)
        except base.ProcessObservationError as error:
            write_json(destination()/'status.json', dict(phase='retrying_observation', job_id=job['id'],
                checked_unix=time.time(), error_type=type(error).__name__, error=str(error)))
            time.sleep(30)
            continue
        if evidence is not None:
            return evidence
        try:
            parent_live = base.inspect_process(plan['parent']['pid']) == plan['parent']
            status = read(WORK/'controlled/full_tuning_queue/status.json')
        except (base.ProcessObservationError, OSError, json.JSONDecodeError) as error:
            write_json(destination()/'status.json', dict(phase='retrying_observation', job_id=job['id'],
                checked_unix=time.time(), error_type=type(error).__name__, error=str(error)))
            time.sleep(30)
            continue
        if not parent_live or status.get('phase') == 'stopped_for_inspection':
            raise RuntimeError('Tuning coordinator ended without this verified fit; inspect without restarting')
        write_json(destination()/'status.json', dict(phase='waiting_for_registered_fit', job_id=job['id'],
            checked_unix=time.time(), parent=plan['parent'], tuning_status=status))
        time.sleep(30)


def run():
    import fcntl
    runtime_guard(); target = destination(); plan = read(target/'plan.json')
    with (target/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate_plan(plan)
        if (target/'started.json').exists():
            raise RuntimeError('This one-shot audit already started; inspect without restarting')
        immutable(target/'started.json', dict(identity=base.inspect_process(os.getpid()), started_unix=time.time()))
        completed = []
        try:
            for job in plan['jobs']:
                evidence = wait_for_completion(plan, job)
                directory = analysis_directory(job['id'])
                if directory.exists():
                    raise RuntimeError('Native analysis output already exists; inspect without adopting or retrying')
                immutable(target/'jobs'/job['id']/'input.json', evidence)
                write_json(target/'status.json', dict(phase='analyzing_complete_trajectory', job_id=job['id'], started_unix=time.time()))
                result = analysis.main(job['id'])
                if (result['epochs_analyzed'] != list(range(1, 16)) or not result['complete_horizon_observed'] or
                    result['test_predictions_read']):
                    raise ValueError('Expected exactly fifteen validation/probe epochs')
                write_json(target/'status.json', dict(phase='independently_verifying_trajectory', job_id=job['id'], started_unix=time.time()))
                verified = verifier.verify(directory)
                trajectory, verification = directory/'trajectory_through_epoch15.json', directory/'verification_through_epoch15.json'
                if (len(verified['checks']) != 30 or not verified['source_and_artifact_hashes_verified'] or
                    verified['test_predictions_read'] or read(verification) != verified or
                    sha256(trajectory) != verified['trajectory_sha256']):
                    raise ValueError('Independent trajectory verification is incomplete')
                if wait_for_completion(plan, job) != evidence:
                    raise ValueError('Training evidence changed during analysis')
                record = dict(**evidence, trajectory_sha256=sha256(trajectory), verification_sha256=sha256(verification),
                    snapshots_verified=30, finished_unix=time.time())
                immutable(target/'jobs'/job['id']/'complete.json', record)
                completed.append(record)
            validate_plan(plan)
            for job, record in zip(plan['jobs'], completed, strict=True):
                directory = analysis_directory(job['id'])
                if (completed_evidence(plan, job) != read(target/'jobs'/job['id']/'input.json') or
                    read(target/'jobs'/job['id']/'complete.json') != record or
                    sha256(directory/'trajectory_through_epoch15.json') != record['trajectory_sha256'] or
                    sha256(directory/'verification_through_epoch15.json') != record['verification_sha256']):
                    raise ValueError('Earlier audit evidence changed')
            done = dict(plan_sha256=sha256(target/'plan.json'), jobs=completed, finished_unix=time.time(),
                new_fits_created=0, model_inference=False, test_predictions_read=False, study_complete=False)
            immutable(target/'complete.json', done)
            immutable(target/'exit.json', dict(returncode=0, finished_unix=time.time()))
            write_json(target/'status.json', dict(phase='complete', finished_unix=time.time()))
            return done
        except BaseException as error:
            failed = dict(returncode=1, finished_unix=time.time(), error_type=type(error).__name__, error=str(error))
            immutable(target/'exit.json', failed)
            write_json(target/'status.json', dict(phase='failed_requires_inspection', **failed))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    args = parser.parse_args()
    print(prepare() if args.action == 'prepare' else json.dumps(run()), flush=True)
