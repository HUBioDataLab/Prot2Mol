"""Replace the held legacy dispatcher without refitting its current A model.

The original fixed scope, job configurations and native training source remain
unchanged. This controller adopts only the bound first fit after a zero launcher
exit and all native completions, retires the three superseded controllers, then
runs the other 23 fixed fits. No allocation inference or averaging is invoked.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import time

from .run import ROOT, WORK, sha256, write_json
from . import requested_single_seed as requested, requested_retention as retention

cq, driver = requested.control_queue, requested.control_driver
read, immutable = cq.read, cq.immutable
WAIT_SECONDS = 600
HOLD_SHA256 = '9cebe82e7194261b5827affc6e51ec498318d1996adca8972a5ef72d4e7244c1'


def destination():
    return WORK / 'controlled/requested_scoped_training_20261104'


def hold_path():
    return WORK / 'protocol_amendments/skip_unrequested_allocations_20261104/dispatch_held.json'


def process_state(pid):
    """Include terminal status; a missing handle is not proof of zero exit."""
    path = Path('/proc') / str(pid)
    try:
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, state=fields[0], ppid=int(fields[1]), start_ticks=int(fields[19]),
                    exit_code=int(fields[49]), uid=path.stat().st_uid)
    except FileNotFoundError:
        return None


def require_live(identity, state=None):
    if (identity['uid'] != os.getuid() or identity['cwd'] != str(ROOT) or
            cq.process_identity(identity['pid']) != identity):
        raise ValueError('Bound owned controller identity changed')
    current = process_state(identity['pid'])
    if current is None or (state is not None and current['state'] != state):
        raise ValueError('Bound controller state changed')
    return current


def current_terminal(plan):
    hold = plan['held']; parent = hold['training_controller']; launcher = hold['training_launcher']
    require_live(parent, 'T')
    current = process_state(launcher['pid'])
    if (current is None or current['start_ticks'] != launcher['start_ticks'] or
            current['ppid'] != parent['pid'] or current['uid'] != os.getuid()):
        raise RuntimeError('Cannot establish the original launcher exit status')
    if current['state'] != 'Z':
        require_live(launcher)
        if current['state'] in ('T', 't', 'X'):
            raise RuntimeError('Current training is unexpectedly stopped')
        return None
    if current['exit_code'] != 0:
        raise RuntimeError('Current launcher failed; preserve outputs for inspection')
    if any(cq.process_identity(r['pid']) == r for r in hold['training_ranks'].values()):
        raise RuntimeError('Launcher exited but an original training rank is still live')
    return current


def prepare():
    cq.runtime_guard()
    scope_path = requested.destination() / 'plan.json'; scope = read(scope_path)
    requested.validate_plan(scope)
    if sha256(hold_path()) != HOLD_SHA256:
        raise ValueError('Dispatcher hold evidence changed')
    held = read(hold_path()); require_live(held['training_controller'], 'T')
    for name in ('reporting_controller', 'probe_controller'):
        require_live(held[name])
    if held['current_job'] != scope['jobs'][0]['id'] or held['training_signal_sent']:
        raise ValueError('Adoption must preserve exactly the first live A fit')
    job = scope['jobs'][0]; output = Path(job['output_dir'])
    if read(output / 'controlled_run.json') != dict(format_version=1, allow_checkpoint_retention=True, **job):
        raise ValueError('Current fit registration differs from the fixed scope')
    requested.require_unstarted(scope['jobs'][1:])
    paths = [scope_path, scope_path.parent / 'invocation.json', hold_path(),
             Path(__file__), Path(retention.__file__),
             WORK / 'controlled/requested_reporting_queue/plan.json',
             WORK / 'controlled/requested_reporting_queue/invocation.json',
             WORK / 'controlled/requested_probe_queue/plan.json',
             WORK / 'controlled/requested_probe_queue/invocation.json']
    plan = dict(format_version=1, boot_id=cq.boot_id(), scope_plan=str(scope_path),
        scope_plan_sha256=sha256(scope_path), held=held, wait_seconds=WAIT_SECONDS,
        files_sha256={str(p.relative_to(ROOT)): sha256(p) for p in paths},
        policy='fixed_seed42_no_allocation_inference_or_averaging',
        seed=42, learning_rate_multiplier=1., adopted_fits=1, remaining_fits=23,
        parameter_search=False, additional_seed_fits=0, allocation_inference=False,
        old_plans_preserved=True, current_fit_restarted=False)
    current_terminal(plan)
    path = destination() / 'plan.json'; immutable(path, plan)
    return path


def validate(plan):
    cq.runtime_guard()
    if (read(destination() / 'plan.json') != plan or plan['boot_id'] != cq.boot_id() or
            plan['scope_plan'] != str(requested.destination() / 'plan.json') or
            sha256(Path(plan['scope_plan'])) != plan['scope_plan_sha256'] or
            plan['wait_seconds'] != WAIT_SECONDS or plan['seed'] != 42 or
            plan['learning_rate_multiplier'] != 1. or plan['adopted_fits'] != 1 or
            plan['remaining_fits'] != 23 or plan['parameter_search'] or
            plan['additional_seed_fits'] or plan['allocation_inference'] or plan['current_fit_restarted'] or
            plan['held'] != read(hold_path()) or sha256(hold_path()) != HOLD_SHA256):
        raise ValueError('Scoped replacement plan changed')
    for name, digest in plan['files_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Scoped replacement source/input changed: ' + name)
    scope = read(Path(plan['scope_plan'])); requested.validate_plan(scope)
    return scope


def native_completion(job, qualification):
    output = Path(job['output_dir'])
    marker = read(output / 'controlled_run.json')
    if marker != dict(format_version=1, allow_checkpoint_retention=True, **job):
        raise ValueError('Native fit metadata changed')
    summary = read(output / 'experiment_summary.json'); state = read(output / 'trainer_state.json')
    rows = [r for r in state['log_history'] if 'eval_macro_spearman' in r and 'eval_loss' in r]
    if (sha256(Path(job['config_path'])) != job['config_sha256'] or
            summary['config_sha256'] != job['config_sha256'] or summary['test_evaluated'] or summary['test_metrics'] or
            state['epoch'] != 15 or [r['epoch'] for r in rows] != list(range(1, 16)) or
            state['global_step'] != rows[-1]['step'] or
            any(a['step'] >= b['step'] for a, b in zip(rows, rows[1:])) or
            read(output / 'runtime_qualification.json') != qualification):
        raise ValueError('Native fit has not completed the unchanged fifteen-epoch configuration')
    for rank in range(4):
        done = read(output / f'rank_{rank}_complete.json')
        if (done['rank'], done['world_size'], done['config_sha256']) != (rank, 4, job['config_sha256']):
            raise ValueError('Native rank completion changed')
    names = ['controlled_run.json', 'launch_provenance.json', 'experiment_summary.json',
             'trainer_state.json', 'runtime_qualification.json', *[f'rank_{r}_complete.json' for r in range(4)]]
    return {name: sha256(output / name) for name in names}


def retire_controllers(plan, terminal, native):
    """Signal only the recorded controllers, never a launcher/rank/process group."""
    target = destination(); held = plan['held']
    if current_terminal(plan) != terminal:
        raise ValueError('Terminal training evidence changed before controller retirement')
    for name in ('reporting_controller', 'probe_controller'):
        require_live(held[name])
    intent = dict(plan_sha256=sha256(target / 'plan.json'), terminal_launcher=terminal,
        native_completion_sha256=native, controllers={n: held[n] for n in
        ('probe_controller', 'reporting_controller', 'training_controller')},
        training_signals_sent=False, created_unix=time.time())
    immutable(target / 'retirement_intent.json', intent)
    for name in ('probe_controller', 'reporting_controller', 'training_controller'):
        owner = held[name]
        require_live(owner, 'T' if name == 'training_controller' else None)
        # The parent is deliberately held in waitpid. SIGKILL avoids executing
        # any legacy instruction; its already successful child is untouched.
        chosen = signal.SIGKILL if name == 'training_controller' else signal.SIGTERM
        os.kill(owner['pid'], chosen)
        for _ in range(100):
            if cq.process_identity(owner['pid']) != owner:
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Superseded controller did not exit; no new fit may start')
        immutable(target / (name + '_retired.json'), dict(identity=owner, signal=int(chosen),
            verified_not_live=True, training_signals_sent=False, checked_unix=time.time()))
    record = dict(intent_sha256=sha256(target / 'retirement_intent.json'),
        retired={n: sha256(target / (n+'_retired.json')) for n in intent['controllers']},
        current_fit_restarted=False, training_signals_sent=False, finished_unix=time.time())
    immutable(target / 'retirement_complete.json', record)
    return record


def finish_job(job, qualification, *, adopted=False):
    native_completion(job, qualification)
    output = Path(job['output_dir'])
    result = retention.retain(output, prune=True)
    names = ['experiment_summary.json', 'trainer_state.json', 'retained/retention.json',
             *[f'rank_{r}_complete.json' for r in range(4)]]
    done = dict(config_sha256=job['config_sha256'], plan_sha256=job['plan_sha256'], job_id=job['id'],
        phase='training_complete_test_not_scored', selections=result['selections'],
        execution_amendment=str(destination() / 'plan.json'),
        execution_amendment_sha256=sha256(destination() / 'plan.json'),
        adopted_existing_fit=adopted, allocation_inference=False, averaging=False,
        evidence_sha256={name: sha256(output/name) for name in names}, finished_unix=time.time())
    immutable(output / 'controlled_training_complete.json', done)
    return done


def execute_job(job, scope):
    """Same native Trainer and qualification; no call to the legacy executor."""
    if job not in scope['jobs'] or sha256(Path(job['config_path'])) != job['config_sha256']:
        raise ValueError('Fit is outside the fixed requested configurations')
    requested.require_unstarted([job])
    primary = read(WORK / 'controlled/plan.json')
    driver.verify_training_data(primary, job['regime'])
    gate = driver.baseline_gate(WORK, primary)
    if not gate['ready']:
        raise ValueError('Baseline evidence is incomplete')
    busy = cq.gpu_processes()
    if busy:
        return dict(phase='waiting_for_authorized_gpus', live_processes=busy)
    output = Path(job['output_dir']); output.mkdir(parents=True)
    write_json(output / 'controlled_run.json', dict(format_version=1, allow_checkpoint_retention=True, **job))
    write_json(output / 'baseline_gate.json', gate)
    write_json(output / 'runtime_qualification.json', scope['references']['qualification'])
    command = [str(ROOT/'.venv/bin/python'), '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node=4', 'RewardModel/experiments/paired_generalization/run.py',
               'train', '--config', job['config_path']]
    driver._run_process(command, output/'train.log', 'training', job['id'])
    return finish_job(job, scope['references']['qualification'])


def run():
    import fcntl
    target = destination(); plan = read(target / 'plan.json'); scope = validate(plan)
    with (target / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (target / 'invocation.json').exists():
            raise RuntimeError('Scoped controller already invoked; inspect, do not restart')
        immutable(target / 'invocation.json', dict(pid=os.getpid(), started_unix=time.time(),
            plan_sha256=sha256(target / 'plan.json')))
        completed = []
        def status(phase, **values):
            write_json(target / 'status.json', dict(phase=phase, pid=os.getpid(), checked_unix=time.time(),
                completed_jobs=[j['id'] for j in completed], **values))
        try:
            while (terminal := current_terminal(plan)) is None:
                status('waiting_for_current_A_without_refit', next_check_seconds=WAIT_SECONDS)
                time.sleep(WAIT_SECONDS); validate(plan)
            first = scope['jobs'][0]
            native = native_completion(first, scope['references']['qualification'])
            retire_controllers(plan, terminal, native)
            for number, job in enumerate(scope['jobs']):
                while True:
                    validate(plan); cq.require_completed(completed)
                    with (WORK / 'controlled/driver.lock').open('a') as driver_lock:
                        try:
                            fcntl.flock(driver_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            waiting = dict(phase='waiting_for_global_driver_lock')
                        else:
                            busy, storage = cq.gpu_processes(), cq.storage_gate()
                            if busy:
                                waiting = dict(phase='waiting_for_authorized_gpus', live_processes=busy)
                            elif not storage['ready']:
                                waiting = dict(phase='waiting_for_storage', storage=storage)
                            else:
                                status('adopting_current_A' if number == 0 else 'training_fixed_configuration', job_id=job['id'])
                                if number == 0:
                                    if native_completion(job, scope['references']['qualification']) != native:
                                        raise ValueError('Adopted native evidence changed after retirement')
                                    result = finish_job(job, scope['references']['qualification'], adopted=True)
                                else:
                                    result = execute_job(job, scope)
                                if result['phase'] == 'training_complete_test_not_scored':
                                    validate(plan)
                                    evidence = cq.require_completed([job])[0]
                                    immutable(target / 'jobs' / (job['id']+'.json'), evidence)
                                    completed.append(job); break
                                if result['phase'] != 'waiting_for_authorized_gpus' or Path(job['output_dir']).exists():
                                    raise RuntimeError('Unexpected scoped training result')
                                waiting = result
                    status(**waiting, job_id=job['id'], next_check_seconds=WAIT_SECONDS)
                    time.sleep(WAIT_SECONDS)
            result = dict(phase='requested_fixed_config_training_complete_evaluations_and_probes_pending',
                plan_sha256=plan['scope_plan_sha256'], evidence=cq.require_completed(completed),
                training_fits_created=24, adopted_existing_fits=1, replacement_fits_launched=23,
                parameter_search=False, additional_seed_fits=0, allocation_inference=False, averaging=False,
                test_scoring_complete=False, scientific_study_complete=False,
                execution_amendment=str(target / 'plan.json'), execution_amendment_sha256=sha256(target / 'plan.json'),
                retirement_sha256=sha256(target / 'retirement_complete.json'), finished_unix=time.time())
            immutable(target / 'complete.json', result)
            # Existing evaluators consume the original fixed-scope completion.
            # This explicit bridge preserves that scope and names its executor.
            immutable(requested.destination() / 'complete.json', result)
            status(result['phase']); return result
        except BaseException as error:
            status('stopped_for_inspection', error_type=type(error).__name__, error=str(error))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run')); args = parser.parse_args()
    print(prepare() if args.action == 'prepare' else json.dumps(run(), indent=2))
