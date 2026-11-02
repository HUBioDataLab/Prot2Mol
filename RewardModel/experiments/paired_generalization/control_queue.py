"""Bounded continuation of the four already registered full-arm tuning fits.

Wait for the exact live B driver and ranks, require its verified completion,
then use the existing driver under its global lock. No failed run is retried.
This queue stops before confirmation/selection and does not finish the study.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from .control_plan import GPU_UUIDS, _write_immutable
    from . import control_driver
else:
    from run import ROOT, WORK, sha256, write_json
    from control_plan import GPU_UUIDS, _write_immutable
    import control_driver

PREDECESSOR = 'full__B__seed42__lr1p0'
SUCCESSORS = ('full__A__seed42__lr0p3', 'full__B__seed42__lr0p3',
              'full__A__seed42__lr3p0', 'full__B__seed42__lr3p0')
MINIMUM_ARCHIVE_BYTES = 30 * 1024**3
MINIMUM_HOME_BYTES = 4 * 1024**3


def read(path):
    return json.loads(Path(path).read_text())


def immutable(path, value):
    _write_immutable(path, json.dumps(value, sort_keys=True, indent=2)+'\n')


def process_identity(pid, proc=Path('/proc')):
    """Identify a Linux process across PID reuse; a zombie is terminal."""
    path = proc/str(int(pid))
    try:
        fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] in ('Z', 'X'):
            return None
        argv = (path/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        if not argv or not argv[0]:
            raise RuntimeError('Live process has no inspectable command')
        return dict(pid=int(pid), start_ticks=int(fields[19]), argv=argv,
                    cwd=str((path/'cwd').resolve(strict=True)), uid=path.stat().st_uid)
    except FileNotFoundError:
        return None


def original_handles(plan):
    return {role: identity for role, identity in plan['processes'].items()
            if process_identity(identity['pid']) == identity}


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def runtime_guard():
    if (ROOT != Path('/home/atabey/Prot2Mol') or
        os.environ.get('CUDA_VISIBLE_DEVICES') != '' or
        os.environ.get('ACCELERATE_USE_CPU', '').lower() != 'false' or
        os.environ.get('WANDB_MODE') != 'online'):
        raise RuntimeError('Use Atum Prot2Mol, CUDA_VISIBLE_DEVICES empty, ACCELERATE_USE_CPU=false, and WANDB_MODE=online; only children expose GPUs')


def gpu_processes():
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                     '--format=csv,noheader'], text=True)
    return control_driver.busy_authorized_devices(output)


def quota_remaining(text, filesystem):
    """Linux quota -w -v reports block counts in KiB, with optional '*' marks."""
    matches = [line.split() for line in text.splitlines()
               if line.split() and line.split()[0] == filesystem]
    if len(matches) != 1 or len(matches[0]) < 4:
        raise ValueError('Expected one numeric quota record for '+filesystem)
    used, soft, hard = [int(v.rstrip('*')) for v in matches[0][1:4]]
    limits = [v for v in (soft, hard) if v > 0]
    if used < 0 or not limits:
        raise ValueError('A finite account quota is required for unattended training')
    return max(0, min(limits)-used)*1024


def storage_gate():
    env = dict(os.environ, LC_ALL='C')
    quota = subprocess.check_output(['quota', '-w', '-v'], text=True, env=env)
    records = []
    for name, path, minimum in (('archive', WORK/'controlled', MINIMUM_ARCHIVE_BYTES),
                                ('home', ROOT, MINIMUM_HOME_BYTES)):
        lines = subprocess.check_output(['df', '--output=source', str(path.resolve())],
                                        text=True, env=env).strip().splitlines()
        if len(lines) != 2:
            raise ValueError('Cannot establish project storage filesystem')
        remaining = quota_remaining(quota, lines[1].strip())
        free = shutil.disk_usage(path).free
        records.append(dict(storage=name, filesystem=lines[1].strip(),
                            quota_remaining_bytes=remaining, filesystem_free_bytes=free,
                            minimum_bytes=minimum, ready=min(remaining, free) >= minimum))
    return dict(ready=all(r['ready'] for r in records), records=records)


def prepare(driver_pid, training_pid):
    runtime_guard()
    directory = WORK/'controlled'
    primary = read(directory/'plan.json')
    order = [j['id'] for j in primary['tuning_jobs'] if j['arm'] == 'full']
    if order != ['full__A__seed42__lr1p0', PREDECESSOR, *SUCCESSORS]:
        raise ValueError('Registered full-arm tuning order differs')
    ready = read(directory/'preparation_complete.json')
    if ready['plan_sha256'] != sha256(directory/'plan.json'):
        raise ValueError('Primary preparation is not verified')
    if (primary['gpu_uuids'] != list(GPU_UUIDS) or
        primary['spec_sha256'] != sha256(ROOT/'Paired input ranking generalization.md')):
        raise ValueError('Specification or authorized GPUs changed')
    jobs, files = [], {}
    for identity in (PREDECESSOR, *SUCCESSORS):
        path = directory/'jobs'/f'{identity}.json'
        job = read(path)
        config = Path(job['config_path'])
        if (job['id'] != identity or job['plan_sha256'] != sha256(directory/'plan.json') or
            job['config_sha256'] != sha256(config) or
            Path(job['output_dir']) != directory/'runs'/identity):
            raise ValueError('Registered job provenance differs')
        if identity != PREDECESSOR and Path(job['output_dir']).exists():
            raise ValueError('A successor output already exists; inspect before queuing')
        jobs.append(job)
        for item in (path, config):
            files[str(item.relative_to(ROOT))] = sha256(item)
    current = jobs[0]
    if control_driver.next_job(directory, primary) != current:
        raise ValueError('The running B job is not the next unfinished registered job')
    state = read(directory/'driver_status.json')
    if (state.get('job_id') != PREDECESSOR or state.get('phase') != 'training' or
        state.get('driver_pid') != driver_pid or state.get('pid') != training_pid):
        raise ValueError('Status does not identify the expected live B training')
    holders = gpu_processes()
    if (len(holders) != 4 or {r['gpu_uuid'] for r in holders} != set(GPU_UUIDS) or
        len({r['pid'] for r in holders}) != 4):
        raise ValueError('Expected exactly four live ranks on the authorized GPUs')
    pids = dict(driver=driver_pid, training=training_pid,
                **{f'rank{i}': next(r['pid'] for r in holders if r['gpu_uuid'] == gpu)
                   for i, gpu in enumerate(GPU_UUIDS)})
    processes = {role: process_identity(pid) for role, pid in pids.items()}
    if len(set(pids.values())) != 6 or any(v is None for v in processes.values()):
        raise ValueError('Prepare while all six distinct original processes are live')
    for role, identity in processes.items():
        if identity['uid'] != os.getuid() or identity['cwd'] != str(ROOT.resolve()):
            raise ValueError('Live process is outside the authorized owner/project')
        argv = identity['argv']
        expected = 'control_driver.py' if role == 'driver' else 'run.py'
        if not any(Path(arg).name == expected for arg in argv):
            raise ValueError('Unexpected process command for '+role)
        if role != 'driver' and current['config_path'] not in argv:
            raise ValueError('A process belongs to another training configuration')
        if role == 'training' and 'torch.distributed.run' not in argv:
            raise ValueError('Expected the original torchrun launcher')
    qualification = Path(current['output_dir'])/'runtime_qualification.json'
    sources = read(qualification)['sources']
    for relative, digest in sources.items():
        if sha256(ROOT/relative) != digest:
            raise ValueError('Active training source changed: '+relative)
    for path in (directory/'plan.json', directory/'preparation_complete.json', qualification,
                 ROOT/'Paired input ranking generalization.md', Path(__file__).resolve()):
        files[str(path.relative_to(ROOT))] = sha256(path)
    from importlib.metadata import version
    plan = dict(predecessor=current, successors=jobs[1:], processes=processes,
                boot_id=boot_id(), files_sha256=files, sources_sha256=sources,
                packages={name: version(name) for name in ('torch', 'transformers', 'datasets', 'numpy', 'PyYAML')},
                gpu_uuids=list(GPU_UUIDS), scope='Exactly four registered tuning fits; stop before confirmation or test selection; never retry partial work')
    destination = directory/'full_tuning_queue'
    immutable(destination/'plan.json', plan)
    return destination/'plan.json'


def validate_plan(plan):
    runtime_guard()
    directory = WORK/'controlled'
    if read(directory/'full_tuning_queue/plan.json') != plan:
        raise ValueError('Queue plan changed after it was loaded')
    if plan['boot_id'] != boot_id() or plan['gpu_uuids'] != list(GPU_UUIDS):
        raise ValueError('Host boot or authorized GPU set changed')
    if (plan['predecessor']['id'] != PREDECESSOR or
        tuple(j['id'] for j in plan['successors']) != SUCCESSORS):
        raise ValueError('Queue scope differs from the four registered successor fits')
    for job in (plan['predecessor'], *plan['successors']):
        if job != read(directory/'jobs'/f"{job['id']}.json"):
            raise ValueError('Queue metadata differs from the registered job')
    from importlib.metadata import version
    if any(version(name) != value for name, value in plan['packages'].items()):
        raise ValueError('Runtime packages changed after queue preparation')
    for relative, digest in {**plan['files_sha256'], **plan['sources_sha256']}.items():
        if sha256(ROOT/relative) != digest:
            raise ValueError('Frozen queue input changed: '+relative)


def require_completed(jobs):
    evidence = []
    for job in jobs:
        if not control_driver.completed_job(job):
            raise RuntimeError('Original processes ended without verified completion: '+job['id'])
        path = Path(job['output_dir'])/'controlled_training_complete.json'
        evidence.append(dict(job_id=job['id'], path=str(path), sha256=sha256(path)))
    return evidence


def run():
    import fcntl
    directory = WORK/'controlled'
    destination = directory/'full_tuning_queue'
    plan = read(destination/'plan.json')
    validate_plan(plan)
    with (destination/'queue.lock').open('a') as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        invocation = destination/'invocation.json'
        if invocation.exists():
            raise RuntimeError('Queue already invoked; inspect its exact handle, do not restart')
        immutable(invocation, dict(pid=os.getpid(), started_unix=time.time(),
                                   plan_sha256=sha256(destination/'plan.json')))

        def status(phase, **values):
            write_json(destination/'status.json', dict(phase=phase, pid=os.getpid(),
                                                      checked_unix=time.time(), **values))

        completed = [plan['predecessor']]
        try:
            while live := original_handles(plan):
                validate_plan(plan)
                status('waiting_for_original_processes', live_processes=live)
                time.sleep(30)
            validate_plan(plan)
            require_completed(completed)
            for job in plan['successors']:
                while True:
                    validate_plan(plan)
                    waiting = None
                    with (directory/'driver.lock').open('a') as lock:
                        try:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            waiting = dict(phase='waiting_for_global_driver_lock')
                        if waiting is None:
                            require_completed(completed)
                            primary = read(directory/'plan.json')
                            if control_driver.next_job(directory, primary) != job:
                                raise RuntimeError('Next registered job changed; inspect before continuing')
                            if Path(job['output_dir']).exists():
                                raise RuntimeError('Successor output exists; refusing automatic retry')
                            gate = control_driver.baseline_gate(WORK, primary)
                            if not gate['ready']:
                                raise RuntimeError('Completed baseline gate is no longer satisfied')
                            busy = gpu_processes()
                            storage = storage_gate()
                            if busy:
                                waiting = dict(phase='waiting_for_authorized_gpus', live_processes=busy)
                            elif not storage['ready']:
                                waiting = dict(phase='waiting_for_storage', storage=storage)
                            else:
                                status('executing_registered_job', job_id=job['id'], storage=storage)
                                result = control_driver.execute_job(job, primary, gate)
                                write_json(directory/'driver_status.json', result)
                                validate_plan(plan)
                                if result.get('phase') == 'waiting_for_authorized_gpus':
                                    waiting = result
                                else:
                                    require_completed([job])
                                    completed.append(job)
                                    break
                    status(job_id=job['id'], **waiting)
                    time.sleep(30)
            evidence = require_completed(completed)
            result = dict(phase='four_tuning_successors_complete_other_work_pending',
                          plan_sha256=sha256(destination/'plan.json'), evidence=evidence,
                          finished_unix=time.time(), study_complete=False)
            immutable(destination/'complete.json', result)
            status(result['phase'], completed_jobs=[j['id'] for j in completed])
            return result
        except Exception as error:
            status('stopped_for_inspection', error_type=type(error).__name__, error=str(error),
                   completed_jobs=[j['id'] for j in completed
                                   if (Path(j['output_dir'])/'controlled_training_complete.json').exists()])
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare_parser = commands.add_parser('prepare')
    prepare_parser.add_argument('--driver-pid', required=True, type=int)
    prepare_parser.add_argument('--training-pid', required=True, type=int)
    commands.add_parser('run')
    args = parser.parse_args()
    print(prepare(args.driver_pid, args.training_pid) if args.command == 'prepare'
          else json.dumps(run(), indent=2))
