"""One CPU handoff: completed full-B15 fit -> native trajectory -> verification.

Wait for the existing driver and all original training ranks. Never launch,
resume or retry training. This adds no fit or test-set evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from . import control_queue, control_driver, native_analysis, verify_native_analysis
else:
    from run import ROOT, WORK, sha256, write_json
    import control_queue, control_driver, native_analysis, verify_native_analysis

IDENTITY = 'full__B__seed42__lr1p0'
PREVIOUS_EPOCH = 10
read = control_queue.read
immutable = control_queue.immutable


def destination():
    return WORK/'tier1/native_trajectory_handoff'/IDENTITY


def runtime_guard():
    if (ROOT != Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or
        os.environ.get('ACCELERATE_USE_CPU', '').lower() != 'true' or os.environ.get('WANDB_MODE') != 'disabled'):
        raise RuntimeError('Use CPU-only Atum Prot2Mol with W&B disabled')


def software():
    result = native_analysis.software()
    for module in (control_queue, control_driver, verify_native_analysis):
        path = Path(module.__file__)
        result['source_sha256'][str(path.relative_to(ROOT))] = sha256(path)
    path = Path(__file__)
    result['source_sha256'][str(path.relative_to(ROOT))] = sha256(path)
    return result


def prepare():
    runtime_guard()
    tuning_path = WORK/'controlled/full_tuning_queue/plan.json'; tuning = read(tuning_path)
    job = tuning['predecessor']; processes = tuning['processes']
    if job['id'] != IDENTITY or set(processes) != {'driver', 'training', 'rank0', 'rank1', 'rank2', 'rank3'}:
        raise ValueError('Expected the existing full-B driver and four-rank fit')
    if any(control_queue.process_identity(p['pid']) != p for p in processes.values()):
        raise RuntimeError('Prepare only while the original six handles are live')
    _, _, registration = native_analysis.registration(IDENTITY)
    directory = WORK/'tier1/native_trajectories'/IDENTITY/native_analysis.digest_json(native_analysis.software())[:16]
    trajectory = directory/f'trajectory_through_epoch{PREVIOUS_EPOCH:02d}.json'
    verification = directory/f'verification_through_epoch{PREVIOUS_EPOCH:02d}.json'
    previous, verified = read(trajectory), read(verification)
    if (verified['trajectory_sha256'] != sha256(trajectory) or
        previous['epochs_analyzed'] != list(range(1, PREVIOUS_EPOCH+1)) or
        not verified['source_and_artifact_hashes_verified'] or verified['test_predictions_read']):
        raise ValueError('The preceding ten-epoch audit is not verified')
    plan = dict(job=job, processes=processes, boot_id=control_queue.boot_id(), software=software(),
        registration=registration, directory=str(directory), previous_epoch=PREVIOUS_EPOCH,
        files_sha256={str(p.relative_to(ROOT)): sha256(p) for p in (tuning_path, trajectory, verification)},
        scope='Wait for all original handles and verified training completion; analyze and independently verify exactly15 epochs on CPU. No fit, inference, test scoring, retry or study-completion claim.')
    immutable(destination()/'plan.json', plan)
    return destination()/'plan.json'


def validate_plan(plan):
    runtime_guard()
    if (read(destination()/'plan.json') != plan or plan['boot_id'] != control_queue.boot_id() or
        plan['job']['id'] != IDENTITY or plan['previous_epoch'] != PREVIOUS_EPOCH or plan['software'] != software()):
        raise ValueError('CPU handoff source, scope or host changed')
    job, _, registration = native_analysis.registration(IDENTITY)
    if job != plan['job'] or registration != plan['registration']:
        raise ValueError('CPU handoff training registration changed')
    for name, digest in plan['files_sha256'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError('Preceding queue or trajectory evidence changed')


def live_originals(plan):
    return {role: p for role, p in plan['processes'].items() if control_queue.process_identity(p['pid']) == p}


def require_completion(plan):
    if live_originals(plan):
        raise RuntimeError('Original training processes are still live')
    if not control_driver.completed_job(plan['job']):
        raise RuntimeError('Original driver ended without verified completion; inspect without restarting')
    run = Path(plan['job']['output_dir']); state = read(run/'trainer_state.json')
    records = [r for r in state['log_history'] if 'eval_macro_spearman' in r and 'eval_loss' in r]
    if state['epoch'] != 15 or [r['epoch'] for r in records] != list(range(1, 16)):
        raise ValueError('Completed fit does not cover the full fifteen-epoch horizon')
    for prefix in ('eval', 'train_eval'):
        snapshots = [read(p) for p in (run/'prediction_exports/trajectory').glob(prefix+'_*.json')]
        if sorted((s['epoch'], s['step']) for s in snapshots) != [(r['epoch'], r['step']) for r in records]:
            raise ValueError('Completed trajectory omits or duplicates epoch snapshots')
    return run/'controlled_training_complete.json'


def run():
    import fcntl
    runtime_guard(); directory = destination(); plan = read(directory/'plan.json')
    with (directory/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate_plan(plan)
        if (directory/'started.json').exists():
            raise RuntimeError('This one-shot handoff already started; inspect its process and receipts')
        immutable(directory/'started.json', dict(identity=control_queue.process_identity(os.getpid()), started_unix=time.time()))
        try:
            while live := live_originals(plan):
                write_json(directory/'status.json', dict(phase='waiting_for_original_processes', checked_unix=time.time(), live_processes=live))
                time.sleep(60)
            validate_plan(plan); completion = require_completion(plan)
            write_json(directory/'status.json', dict(phase='analyzing_complete_trajectory', started_unix=time.time()))
            result = native_analysis.main(IDENTITY)
            if result['epochs_analyzed'] != list(range(1, 16)) or not result['complete_horizon_observed'] or result['test_predictions_read']:
                raise ValueError('Expected the complete validation/training trajectory only')
            write_json(directory/'status.json', dict(phase='independently_verifying_trajectory', started_unix=time.time()))
            verified = verify_native_analysis.verify(Path(plan['directory']), previous_epoch=PREVIOUS_EPOCH)
            if len(verified['checks']) != 30 or not verified['source_and_artifact_hashes_verified'] or verified['test_predictions_read']:
                raise ValueError('Independent verification is incomplete')
            validate_plan(plan); require_completion(plan)
            verification_path = Path(plan['directory'])/'verification_through_epoch15.json'
            done = dict(plan_sha256=sha256(directory/'plan.json'), training_completion_sha256=sha256(completion),
                trajectory_sha256=verified['trajectory_sha256'], verification_sha256=sha256(verification_path),
                finished_unix=time.time(), epochs=result['epochs_analyzed'], snapshots_verified=30,
                new_fits_created=0, test_predictions_read=False, study_complete=False)
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
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=('prepare', 'run'))
    args = parser.parse_args()
    print(prepare() if args.action == 'prepare' else json.dumps(run()), flush=True)
