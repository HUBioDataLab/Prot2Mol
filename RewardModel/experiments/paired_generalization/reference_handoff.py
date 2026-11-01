"""One bounded handoff: live A reference -> contrastive audit -> next grid job.

This watches registered process identities, never restarts training, and never
marks the study complete. A failed/partial child requires inspection. Preparing
or running this coordinator changes no live training source or configuration.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from .control_plan import GPU_UUIDS, digest_json, _write_immutable
    from . import control_driver, reference_contrastive, protect_trajectory
    from .retain_controlled import retain
else:
    from run import ROOT, WORK, sha256, write_json
    from control_plan import GPU_UUIDS, digest_json, _write_immutable
    import control_driver, reference_contrastive, protect_trajectory
    from retain_controlled import retain

REFERENCE = protect_trajectory.REFERENCE


def read(path):
    return json.loads(Path(path).read_text())


def immutable(path, value):
    _write_immutable(path, json.dumps(value, indent=2, sort_keys=True)+'\n')


def process_identity(pid, proc=Path('/proc')):
    """Missing/zombie handles are terminal; PID reuse is not the old process."""
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


def live_originals(identities, inspect=process_identity):
    return {role: expected for role, expected in identities.items()
            if inspect(expected['pid']) == expected}


def software():
    from importlib.metadata import version
    paths = sorted(Path(__file__).parent.glob('*.py'))
    paths += sorted((ROOT/'RewardModel/reward_model').rglob('*.py'))
    return dict(source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in paths},
                packages={name: version(name) for name in ('torch', 'transformers', 'datasets', 'numpy')})


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def runtime_guard():
    if ROOT != Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Run the coordinator CPU-only inside Atum Prot2Mol')


def validate_processes(identities, config_path):
    if len({r['pid'] for r in identities.values()}) != 3:
        raise ValueError('Expected three distinct live process identities')
    for role, identity in identities.items():
        if identity['uid'] != os.getuid() or Path(identity['cwd']) != ROOT.resolve():
            raise ValueError('Process is not owned in the authorized project')
        args = identity['argv']
        expected = {'driver': 'control_driver.py', 'training': 'run.py', 'protector': 'protect_trajectory.py'}[role]
        if not any(Path(arg).name == expected for arg in args):
            raise ValueError('Unexpected process command for '+role)
        if role == 'training' and ('torch.distributed.run' not in args or str(config_path) not in args):
            raise ValueError('Training launcher belongs to another configuration')


def prepare(driver_pid, training_pid, protector_pid):
    runtime_guard()
    if __package__:
        from .native_analysis import registration
    else:
        from native_analysis import registration
    directory = WORK/'controlled'
    job_path = directory/'jobs'/f'{REFERENCE}.json'
    job, primary, _ = registration(REFERENCE)
    if (job['id'] != REFERENCE or job['regime'] != 'A' or job['arm'] != 'full' or
        job['seed'] != 42 or job['learning_rate_multiplier'] != 1. or
        job['plan_sha256'] != sha256(directory/'plan.json') or
        job['config_sha256'] != sha256(Path(job['config_path']))):
        raise ValueError('Reference registration changed')
    if primary['spec_sha256'] != sha256(ROOT/'Paired input ranking generalization.md') or primary['gpu_uuids'] != list(GPU_UUIDS):
        raise ValueError('Specification or authorized devices differ from the original plan')
    state = read(directory/'driver_status.json')
    if (state['job_id'] != REFERENCE or state['phase'] != 'training' or
        state['driver_pid'] != driver_pid or state['pid'] != training_pid):
        raise ValueError('Driver status does not identify the live reference')
    identities = {role: process_identity(pid) for role, pid in
                  [('driver', driver_pid), ('training', training_pid), ('protector', protector_pid)]}
    if any(value is None for value in identities.values()):
        raise ValueError('Prepare while all three original handles are live')
    validate_processes(identities, job['config_path'])
    plan = dict(reference=job, job_metadata_sha256=sha256(job_path),
                spec_sha256=sha256(ROOT/'Paired input ranking generalization.md'),
                software=software(), boot_id=boot_id(),
                processes=identities, gpu_uuids=list(GPU_UUIDS),
                scope='Wait for original handles; verify all15 protected epochs and training completion; run reference contrastive inference; start exactly one next frozen-grid driver. No automatic retry or study-completion claim.')
    destination = directory/'reference_handoff'
    immutable(destination/'plan.json', plan)
    return destination/'plan.json'


def validate_plan(plan):
    runtime_guard()
    directory = WORK/'controlled'
    job = plan['reference']
    if (plan['boot_id'] != boot_id() or
        plan['gpu_uuids'] != list(GPU_UUIDS) or plan['software'] != software() or
        plan['spec_sha256'] != sha256(ROOT/'Paired input ranking generalization.md') or
        job['plan_sha256'] != sha256(directory/'plan.json') or
        job['config_sha256'] != sha256(Path(job['config_path'])) or
        plan['job_metadata_sha256'] != sha256(directory/'jobs'/f'{REFERENCE}.json') or
        job != read(directory/'jobs'/f'{REFERENCE}.json')):
        raise ValueError('Handoff source, host, specification or registered configuration changed')


def finish_reference(job):
    """Recover only verified postprocessing after a possible retention-hold race."""
    import fcntl
    directory = WORK/'controlled'
    with (directory/'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protection = protect_trajectory.verify_complete((directory/'protected_trajectories'/REFERENCE).resolve())
        if len(protection['epochs']) != 15 or protection['config_sha256'] != job['config_sha256']:
            raise ValueError('The full reference trajectory is not protected')
        output = Path(job['output_dir'])
        launch = read(output/'launch_provenance.json')
        if (launch['world_size'] != 4 or launch['config']['training']['num_train_epochs'] != 15 or
            launch['config_sha256'] != job['config_sha256']):
            raise ValueError('Expected the original four-rank15-epoch reference')
        if control_driver.completed_job(job):
            return read(output/'controlled_training_complete.json')
        summary = read(output/'experiment_summary.json')
        if summary['test_evaluated'] or summary['test_metrics']:
            raise ValueError('Reference training unexpectedly evaluated test data')
        # retain verifies every rank, final horizon, selected weights and native
        # prediction payload before pruning. It cannot turn partial training
        # into completed training or release the protector's marker hold.
        result = retain(output, prune=True)
        evidence = ['experiment_summary.json', 'trainer_state.json', 'retained/retention.json',
                    *(f'rank_{rank}_complete.json' for rank in range(4))]
        done = dict(config_sha256=job['config_sha256'], plan_sha256=job['plan_sha256'],
                    job_id=job['id'], phase='training_complete_test_not_scored',
                    selections=result['selections'], finished_unix=time.time(),
                    postprocessing_recovered_without_retraining=True,
                    evidence_sha256={name: sha256(output/name) for name in evidence})
        immutable(output/'controlled_training_complete.json', done)
        assert control_driver.completed_job(job)
        return done


def child_environment(training):
    env = control_driver.launch_environment(REFERENCE)
    # The coordinator is CPU-only. Do not accidentally force its GPU children
    # onto CPU through Accelerate's inherited environment setting.
    env.update(ACCELERATE_USE_CPU='false', PYTHONPATH=str(ROOT/'RewardModel'),
               WANDB_MODE='online' if training else 'disabled')
    return env


def start_child(destination, name, command, *, training):
    intent = destination/f'{name}_launch.json'
    if intent.exists():
        raise RuntimeError('A child launch already exists; inspect its exact handle')
    log = destination/f'{name}.log'
    immutable(intent, dict(command=command, log=str(log), launch_requested_unix=time.time()))
    with log.open('x') as handle:
        child = subprocess.Popen(command, cwd=ROOT, env=child_environment(training),
            stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(destination/f'{name}_process.json', dict(pid=child.pid, command=command, log=str(log)))
    return child


def verify_reference_audit(plan, output):
    if read(output/'plan.json') != plan:
        raise ValueError('Reference audit plan differs')
    result = read(output/'complete.json')
    if (result['plan_sha256'] != sha256(output/'plan.json') or result['exports'] != 30 or result['test_data_used']):
        raise ValueError('Reference audit is incomplete or used test data')
    seen = []
    for record in result['trajectory']:
        info = record['analysis']
        expected = output/f"epoch{int(info['epoch']):02d}"/info['prefix']
        if Path(record['path']) != expected or sha256(expected/'complete.json') != record['sha256']:
            raise ValueError('Reference audit export evidence changed')
        marker = read(expected/'complete.json')
        if marker['analysis'] != info or marker['cpu_qualification'] or not marker['parameters_unchanged']:
            raise ValueError('Reference export is not the unchanged scientific checkpoint')
        seen.append((info['epoch'], info['prefix']))
    if sorted(seen) != sorted((epoch, prefix) for epoch in range(1,16) for prefix in ('eval','train_eval')):
        raise ValueError('Reference audit omits or duplicates an epoch/partition')
    return sha256(output/'complete.json')


def run():
    import fcntl
    destination = WORK/'controlled/reference_handoff'
    plan = read(destination/'plan.json')
    validate_plan(plan)
    with (destination/'handoff.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        claim = destination/'invocation.json'
        if claim.exists():
            raise RuntimeError('Handoff was already invoked; inspect recorded handles and outputs')
        immutable(claim, dict(pid=os.getpid(), started_unix=time.time(), plan_sha256=sha256(destination/'plan.json')))

        def status(phase, **values):
            write_json(destination/'status.json', dict(phase=phase, pid=os.getpid(), checked_unix=time.time(), **values))

        try:
            while live := live_originals(plan['processes']):
                validate_plan(plan)
                status('waiting_for_original_processes', live_processes=live)
                time.sleep(30)
            validate_plan(plan)
            status('verifying_reference_completion')
            finish_reference(plan['reference'])
            # All inputs/weights are verified by the existing audit planner.
            audit_plan = reference_contrastive.build_plan(WORK/'controlled')
            output = WORK/'tier1/reference_contrastive'/REFERENCE/digest_json(audit_plan['software'])[:16]
            if output.exists() and not (output/'complete.json').exists():
                raise RuntimeError('Partial reference audit requires inspection, not automatic retry')
            while True:
                busy = control_driver.busy_authorized_devices(subprocess.check_output(
                    ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True))
                if not busy:
                    break
                status('waiting_for_authorized_gpus', live_processes=busy)
                time.sleep(30)
            validate_plan(plan)
            command = [str(ROOT/'.venv/bin/python'), '-u', str(Path(__file__).with_name('reference_contrastive.py')),
                       'run', '--workers', '4']
            child = start_child(destination, 'reference_audit', command, training=False)
            status('reference_contrastive_inference', child_pid=child.pid)
            code = child.wait()
            if code:
                raise RuntimeError(f'Reference audit exited {code}; no next training was launched')
            validate_plan(plan)
            audit_sha = verify_reference_audit(audit_plan, output)
            directory = WORK/'controlled'
            with (directory/'driver.lock').open('a') as training_lock:
                fcntl.flock(training_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                job = control_driver.next_job(directory, read(directory/'plan.json'))
                if job is None or Path(job['output_dir']).exists():
                    raise RuntimeError('Expected one fresh next grid job; inspect before proceeding')
            command = [str(ROOT/'.venv/bin/python'), '-u', str(Path(__file__).with_name('control_driver.py'))]
            child = start_child(destination, 'next_driver', command, training=True)
            status('next_driver_preflight', child_pid=child.pid, expected_job=job['id'], reference_audit_sha256=audit_sha)
            while child.poll() is None:
                current = read(directory/'driver_status.json')
                if (current.get('driver_pid') == child.pid and current.get('job_id') == job['id'] and
                    current.get('phase') == 'training'):
                    identity = process_identity(current['pid'])
                    if identity is not None and str(job['config_path']) in identity['argv']:
                        result = dict(phase='reference_audit_complete_next_training_started',
                            reference_audit=str(output/'complete.json'), reference_audit_sha256=audit_sha,
                            next_job=job['id'], next_driver_pid=child.pid, training_process=identity,
                            study_complete=False, next_training_complete=False, completed_unix=time.time())
                        immutable(destination/'complete.json', result)
                        status(result['phase'], **{k:v for k,v in result.items() if k != 'phase'})
                        return result
                time.sleep(5)
            raise RuntimeError(f'Next driver exited {child.returncode} before verified training start; inspect its log')
        except Exception as exc:
            status('failed_requires_inspection', error=str(exc))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    prepare_command = commands.add_parser('prepare')
    for name in ('driver', 'training', 'protector'):
        prepare_command.add_argument(f'--{name}-pid', type=int, required=True)
    commands.add_parser('run')
    args = parser.parse_args()
    if args.action == 'prepare':
        print(prepare(args.driver_pid, args.training_pid, args.protector_pid))
    else:
        print(json.dumps(run(), indent=2))
