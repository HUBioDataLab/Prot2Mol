"""After the full tuning queue, run exactly four registered seed confirmations.

Wait for the original coordinator to terminate and verify all six tuning fits.
The existing validation-only rule chooses one shared LR; this adds no settings.
Stop after A/B seeds43/44, before test evaluation or another model arm.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import yaml

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from . import control_queue, control_driver, control_plan
else:
    from run import ROOT, WORK, sha256, write_json
    import control_queue, control_driver, control_plan

ORDER = ((43,'A'),(43,'B'),(44,'A'),(44,'B'))
TUNING_IDS = ('full__A__seed42__lr1p0',control_queue.PREDECESSOR,*control_queue.SUCCESSORS)


def read(path):
    return json.loads(Path(path).read_text())


def destination():
    return WORK/'controlled/full_confirmation_queue'


def prepare():
    control_queue.runtime_guard()
    directory = WORK/'controlled'
    parent = directory/'full_tuning_queue'
    parent_plan = read(parent/'plan.json')
    control_queue.validate_plan(parent_plan)
    invocation,launch = read(parent/'invocation.json'),read(parent/'launch_process.json')
    process = control_queue.process_identity(invocation['pid'])
    if (process is None or process != launch['identity'] or process['uid'] != os.getuid() or
        process['cwd'] != str(ROOT.resolve()) or
        invocation['plan_sha256'] != sha256(parent/'plan.json') or
        launch['plan_sha256'] != sha256(parent/'plan.json')):
        raise ValueError('Prepare only against the exact live, owned tuning coordinator')
    primary = read(directory/'plan.json')
    tuning = [j for j in primary['tuning_jobs'] if j['arm']=='full']
    if tuple(j['id'] for j in tuning) != TUNING_IDS:
        raise ValueError('Expected the six frozen full-arm tuning jobs')
    files = dict(parent_plan['files_sha256'])
    files.update(parent_plan['sources_sha256'])
    jobs = []
    for job in tuning:
        path = directory/'jobs'/f"{job['id']}.json"
        metadata = read(path)
        config = Path(metadata['config_path'])
        expected = {key:value for key,value in job.items() if key!='config'}
        expected.update(config_path=str(directory/'configs'/f"{job['id']}.yaml"),
            config_sha256=sha256(config),output_dir=str(directory/'runs'/job['id']),
            plan_sha256=sha256(directory/'plan.json'))
        if metadata != expected or yaml.safe_load(config.read_text()) != job['config']:
            raise ValueError('Full tuning registration differs from its frozen configuration')
        jobs.append(metadata)
        for item in (path,config):files[str(item.relative_to(ROOT))]=sha256(item)
    if (directory/'selections/full.json').exists():
        raise RuntimeError('A full-arm selection already exists; inspect before preparing a handoff')
    for seed,regime in ORDER:
        for multiplier in control_plan.LEARNING_RATE_MULTIPLIERS:
            identity = control_plan.job_id('full',regime,seed,multiplier)
            if any(path.exists() for path in (directory/'runs'/identity,
                directory/'jobs'/f'{identity}.json',directory/'configs'/f'{identity}.yaml')):
                raise RuntimeError('A confirmation registration/output already exists')
    gate = control_driver.baseline_gate(WORK,primary)
    if not gate['ready']:
        raise RuntimeError('Completed baseline gate is required')
    control_queue.require_completed(jobs[:1])
    paths = [parent/name for name in ('plan.json','invocation.json','launch_process.json')]
    paths.append(Path(jobs[0]['output_dir'])/'controlled_training_complete.json')
    paths += [Path(__file__).resolve(),*(Path(item['path']) for item in gate['evidence'])]
    for path in paths:files[str(path.relative_to(ROOT))]=sha256(path)
    plan = dict(parent_process=process,parent_plan_sha256=sha256(parent/'plan.json'),
        tuning_jobs=jobs,order=[list(pair) for pair in ORDER],files_sha256=files,
        boot_id=control_queue.boot_id(),gpu_uuids=list(control_plan.GPU_UUIDS),
        scope='Only full A/B seeds43/44 at one B-validation-selected LR; stop before test scoring or other arms; never retry partial fits')
    control_queue.immutable(destination()/'plan.json',plan)
    return destination()/'plan.json'


def validate_plan(plan):
    control_queue.runtime_guard()
    if read(destination()/'plan.json') != plan:
        raise ValueError('Confirmation queue plan changed')
    if (plan['boot_id'] != control_queue.boot_id() or plan['gpu_uuids'] != list(control_plan.GPU_UUIDS) or
        plan['order'] != [list(pair) for pair in ORDER] or
        tuple(j['id'] for j in plan['tuning_jobs']) != TUNING_IDS):
        raise ValueError('Confirmation host, devices or scope changed')
    parent = WORK/'controlled/full_tuning_queue/plan.json'
    if sha256(parent) != plan['parent_plan_sha256']:
        raise ValueError('Predecessor tuning plan changed')
    control_queue.validate_plan(read(parent))
    for name,digest in plan['files_sha256'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError('Frozen confirmation input changed: '+name)


def require_tuning_completion(plan):
    parent = WORK/'controlled/full_tuning_queue'
    original = read(parent/'plan.json')
    complete = read(parent/'complete.json')
    expected = control_queue.require_completed([original['predecessor'],*original['successors']])
    if (complete['phase'] != 'four_tuning_successors_complete_other_work_pending' or
        complete['plan_sha256'] != plan['parent_plan_sha256'] or complete['evidence'] != expected):
        raise ValueError('Predecessor queue did not verify its exact completed fits')
    return dict(parent_complete_sha256=sha256(parent/'complete.json'),
        tuning_evidence=control_queue.require_completed(plan['tuning_jobs']))


def verify_confirmation(selection):
    directory = WORK/'controlled'
    snapshot = destination()/'selection.json'
    if snapshot.exists() and read(snapshot) != selection:
        raise ValueError('Confirmation selection snapshot changed')
    selected = selection['selection']
    if read(directory/'selections/full.json') != selected:
        raise ValueError('Frozen B validation selection changed')
    multiplier = selected['learning_rate_multiplier']
    if multiplier not in control_plan.LEARNING_RATE_MULTIPLIERS:
        raise ValueError('Unregistered confirmation learning rate')
    jobs = selection['confirmation_jobs']
    if [(j['seed'],j['regime']) for j in jobs] != list(ORDER):
        raise ValueError('Confirmation jobs differ from the four planned seeds/splits')
    for job in jobs:
        identity = control_plan.job_id('full',job['regime'],job['seed'],multiplier)
        if (job['id'] != identity or job['arm'] != 'full' or job['phase'] != 'confirmation' or
            job['learning_rate_multiplier'] != multiplier or
            job['selection_sha256'] != sha256(directory/'selections/full.json') or
            job['plan_sha256'] != sha256(directory/'plan.json') or
            Path(job['config_path']) != directory/'configs'/f'{identity}.yaml' or
            Path(job['output_dir']) != directory/'runs'/identity or
            sha256(Path(job['config_path'])) != job['config_sha256'] or
            read(directory/'jobs'/f'{identity}.json') != job):
            raise ValueError('Frozen confirmation registration changed')


def run():
    import fcntl
    directory = WORK/'controlled'
    target = destination()
    plan = read(target/'plan.json')
    validate_plan(plan)
    with (target/'queue.lock').open('a') as queue_lock:
        fcntl.flock(queue_lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (target/'invocation.json').exists():
            raise RuntimeError('Confirmation queue already invoked; inspect, do not restart')
        control_queue.immutable(target/'invocation.json',dict(pid=os.getpid(),started_unix=time.time(),
            plan_sha256=sha256(target/'plan.json')))

        def status(phase,**values):
            write_json(target/'status.json',dict(phase=phase,pid=os.getpid(),checked_unix=time.time(),**values))

        completed,selection = [],None
        try:
            while control_queue.process_identity(plan['parent_process']['pid']) == plan['parent_process']:
                validate_plan(plan)
                status('waiting_for_tuning_coordinator',parent_process=plan['parent_process'])
                time.sleep(30)
            while selection is None:
                validate_plan(plan)
                evidence = require_tuning_completion(plan)
                with (directory/'driver.lock').open('a') as lock:
                    try:
                        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        pass
                    else:
                        # Every fit and predecessor handle is complete before
                        # the existing validation-only selector can run.
                        selection = control_plan.confirm('full')
                        verify_confirmation(selection)
                        if any(Path(j['output_dir']).exists() for j in selection['confirmation_jobs']):
                            raise RuntimeError('A confirmation output already exists; refusing automatic retry')
                        control_queue.immutable(target/'upstream_completion.json',evidence)
                        control_queue.immutable(target/'selection.json',selection)
                if selection is None:
                    status('waiting_for_global_driver_lock');time.sleep(30)
            for job in selection['confirmation_jobs']:
                while True:
                    validate_plan(plan)
                    if require_tuning_completion(plan) != read(target/'upstream_completion.json'):
                        raise ValueError('Upstream completed evidence changed')
                    verify_confirmation(selection)
                    waiting = None
                    with (directory/'driver.lock').open('a') as lock:
                        try:
                            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            waiting = dict(phase='waiting_for_global_driver_lock')
                        if waiting is None:
                            control_queue.require_completed(completed)
                            primary = read(directory/'plan.json')
                            if control_driver.next_job(directory,primary) != job:
                                raise RuntimeError('Next registered job differs from this bounded confirmation queue')
                            if Path(job['output_dir']).exists():
                                raise RuntimeError('Confirmation output exists; refusing automatic retry')
                            gate = control_driver.baseline_gate(WORK,primary)
                            if not gate['ready']:
                                raise RuntimeError('Completed baseline gate is no longer satisfied')
                            busy,storage = control_queue.gpu_processes(),control_queue.storage_gate()
                            if busy:
                                waiting = dict(phase='waiting_for_authorized_gpus',live_processes=busy)
                            elif not storage['ready']:
                                waiting = dict(phase='waiting_for_storage',storage=storage)
                            else:
                                status('executing_confirmation',job_id=job['id'],storage=storage)
                                result = control_driver.execute_job(job,primary,gate)
                                write_json(directory/'driver_status.json',result)
                                validate_plan(plan)
                                verify_confirmation(selection)
                                if result.get('phase') == 'waiting_for_authorized_gpus':
                                    waiting = result
                                else:
                                    control_queue.require_completed([job])
                                    completed.append(job)
                                    break
                    status(job_id=job['id'],**waiting);time.sleep(30)
            if require_tuning_completion(plan) != read(target/'upstream_completion.json'):
                raise ValueError('Upstream completed evidence changed')
            result = dict(phase='full_confirmation_training_complete_other_work_pending',
                plan_sha256=sha256(target/'plan.json'),selection_sha256=sha256(target/'selection.json'),
                evidence=control_queue.require_completed(completed),finished_unix=time.time(),study_complete=False)
            control_queue.immutable(target/'complete.json',result)
            status(result['phase'],completed_jobs=[j['id'] for j in completed])
            return result
        except Exception as error:
            status('stopped_for_inspection',error_type=type(error).__name__,error=str(error),
                completed_jobs=[j['id'] for j in completed])
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    args = parser.parse_args()
    print(prepare() if args.action=='prepare' else json.dumps(run(),indent=2))
