"""Wait for the requested seed42 fits, then evaluate and report their fixed models.

No fitting or model selection runs here. Waiting checks only the bound training
handle/receipts every ten minutes; native verification runs at stage boundaries.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from . import requested_evaluation as evaluation, requested_results as results
    from . import requested_probe_protocol as probes
else:
    from run import ROOT, WORK, sha256, write_json
    import requested_evaluation as evaluation, requested_results as results
    import requested_probe_protocol as probes

cq = evaluation.control_queue
read, immutable = cq.read, cq.immutable
WAIT_SECONDS = 600
WAIT_PHASES = {'waiting_for_global_driver_lock', 'waiting_for_authorized_gpus', 'waiting_for_storage'}


def destination():
    return WORK / 'controlled/requested_reporting_queue'


def job_ids(scope):
    jobs = [*scope['references']['jobs'], *scope['jobs']]
    expected = [(a, r) for a in ('full', *evaluation.requested.ARMS) for r in ('A', 'B')]
    if ([(j['arm'], j['regime']) for j in jobs] != expected or
        any((j['seed'], j['learning_rate_multiplier'], j['id']) !=
            (42, 1., f"{j['arm']}__{j['regime']}__seed42__lr1p0") for j in jobs)):
        raise ValueError('Reporting requires exactly the 26 fixed requested/reference fits')
    return [j['id'] for j in jobs]


def bind_training(pid, start_ticks, scope_path):
    identity = cq.process_identity(pid)
    expected_argv = [str(ROOT / '.venv/bin/python'), str(Path(evaluation.requested.__file__).absolute()), 'run']
    invocation = read(scope_path.parent / 'invocation.json')
    if (identity is None or identity['start_ticks'] != start_ticks or identity['uid'] != os.getuid() or
        identity['cwd'] != str(ROOT) or identity['argv'] != expected_argv or
        invocation['pid'] != pid or invocation['plan_sha256'] != sha256(scope_path)):
        raise ValueError('Require the exact live owned requested-training controller')
    return identity


def prepare(training_pid, training_start_ticks):
    cq.runtime_guard()
    scope_path, scope = evaluation.scope()
    protocol = probes.verify()
    if protocol['scope_plan_sha256'] != sha256(scope_path):
        raise ValueError('Probe and training scopes disagree')
    owner = bind_training(training_pid, training_start_ticks, scope_path)
    code = results.software()
    paths = [Path(__file__).absolute(), Path(probes.__file__).absolute(), scope_path,
        scope_path.parent / 'invocation.json', probes.destination() / 'plan.json',
        probes.destination() / 'complete.json']
    files = dict(code['source_sha256'])
    files.update({str(p.relative_to(ROOT)): sha256(p) for p in paths})
    plan = dict(format_version=1, boot_id=cq.boot_id(), seed=42, learning_rate_multiplier=1.,
        scope_plan=str(scope_path), scope_plan_sha256=sha256(scope_path),
        probe_plan_sha256=sha256(probes.destination() / 'plan.json'), training_controller=owner,
        job_ids=job_ids(scope), workers=4, gpu_uuids=list(evaluation.GPU_UUIDS),
        wait_seconds=WAIT_SECONDS, files_sha256=files, packages=code['packages'],
        new_fits=0, parameter_search=False, additional_seed_fits=0,
        description='After all24 requested training completions, evaluate exactly one best-macro checkpoint for each of26 fixed fits, in batches of at most4 authorized GPUs. Analyze saved predictions on CPU and summarize all13 settings. No fitting, test selection, automatic failure retry or legacy queue traversal. Tier4 probes and remaining audits are separate.')
    path = destination() / 'plan.json'; immutable(path, plan)
    return path


def validate(plan):
    cq.runtime_guard()
    scope_path, scope = evaluation.scope()
    if (read(destination() / 'plan.json') != plan or plan['boot_id'] != cq.boot_id() or
        plan['seed'] != 42 or plan['learning_rate_multiplier'] != 1. or plan['workers'] != 4 or
        plan['gpu_uuids'] != list(evaluation.GPU_UUIDS) or plan['wait_seconds'] != WAIT_SECONDS or
        plan['scope_plan'] != str(scope_path) or plan['scope_plan_sha256'] != sha256(scope_path) or
        plan['probe_plan_sha256'] != sha256(probes.destination() / 'plan.json') or
        plan['job_ids'] != job_ids(scope) or plan['new_fits'] != 0 or
        plan['parameter_search'] or plan['additional_seed_fits'] != 0):
        raise ValueError('Requested reporting scope or execution environment changed')
    from importlib.metadata import version
    if any(version(n) != v for n, v in plan['packages'].items()):
        raise ValueError('Requested reporting packages changed')
    for name, digest in plan['files_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Requested reporting source or input changed: ' + name)
    return scope


def training_ready(plan):
    """Cheap waiting check; never treat process exit as training completion."""
    path = Path(plan['scope_plan']); target = path.parent
    if cq.boot_id() != plan['boot_id'] or sha256(path) != plan['scope_plan_sha256']:
        raise ValueError('Bound training plan or boot changed while waiting')
    if (target / 'complete.json').exists():
        return True
    state = read(target / 'status.json')
    if state['phase'] == 'stopped_for_inspection':
        raise RuntimeError('Requested training stopped; no evaluation may start')
    owner = plan['training_controller']
    if cq.process_identity(owner['pid']) != owner:
        # Receipt creation and normal process exit can occur between reads.
        if (target / 'complete.json').exists():
            return True
        raise RuntimeError('Requested training controller ended without completion')
    return False


def upstream_completion(plan):
    scope = validate(plan); path = Path(plan['scope_plan']).parent / 'complete.json'
    record = read(path)
    evidence = cq.require_completed(scope['jobs'])
    if (record['phase'] != 'requested_fixed_config_training_complete_evaluations_and_probes_pending' or
        record['plan_sha256'] != plan['scope_plan_sha256'] or record['evidence'] != evidence or
        record['training_fits_created'] != 24 or record['parameter_search'] or
        record['additional_seed_fits'] != 0 or record['test_scoring_complete'] or
        record['scientific_study_complete']):
        raise ValueError('Requested training completion is incomplete or changed')
    return dict(path=str(path), sha256=sha256(path), evidence=evidence,
        references=cq.require_completed(scope['references']['jobs']))


def run():
    import fcntl
    target = destination(); plan = read(target / 'plan.json'); validate(plan)
    with (target / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (target / 'invocation.json').exists():
            raise RuntimeError('Reporting queue already invoked; inspect, do not restart')
        immutable(target / 'invocation.json', dict(pid=os.getpid(), started_unix=time.time(),
            plan_sha256=sha256(target / 'plan.json')))
        completed = []

        def status(phase, **values):
            write_json(target / 'status.json', dict(phase=phase, pid=os.getpid(),
                checked_unix=time.time(), completed_jobs=[r['job_id'] for r in completed], **values))

        try:
            while not training_ready(plan):
                status('waiting_for_requested_training', next_check_seconds=WAIT_SECONDS)
                time.sleep(WAIT_SECONDS)
            upstream = upstream_completion(plan)
            immutable(target / 'upstream_completion.json', upstream)
            for offset in range(0, len(plan['job_ids']), plan['workers']):
                identities = plan['job_ids'][offset:offset + plan['workers']]
                while True:
                    validate(plan)
                    if sha256(Path(upstream['path'])) != upstream['sha256']:
                        raise ValueError('Upstream training completion changed during reporting')
                    status('evaluating_best_macro', batch=identities)
                    batch = evaluation.run_batch(identities)
                    if batch['phase'] == 'requested_best_macro_batch_complete':
                        if batch['jobs'] != identities or batch['new_fits'] != 0:
                            raise ValueError('Native evaluator changed the requested batch')
                        break
                    if batch['phase'] not in WAIT_PHASES:
                        raise ValueError('Unexpected selected evaluation result')
                    status(batch['phase'], batch=identities, resource_state=batch,
                        next_check_seconds=WAIT_SECONDS)
                    time.sleep(WAIT_SECONDS)
                immutable(target / 'batches' / f'{offset:02d}.json', batch)
                for identity in identities:
                    path = evaluation.destination(identity) / 'plan.json'
                    evaluation.verify_complete(path)
                    status('analyzing_fixed_predictions', job_id=identity)
                    analysis = results.analyze(identity)
                    result = results.verify_analysis(analysis)
                    if result['job']['id'] != identity:
                        raise ValueError('Native analysis belongs to a different fit')
                    record = dict(job_id=identity, evaluation=str(path.parent / 'complete.json'),
                        evaluation_sha256=sha256(path.parent / 'complete.json'),
                        analysis=str(analysis), analysis_sha256=sha256(analysis))
                    immutable(target / 'jobs' / (identity + '.json'), record)
                    completed.append(record)
            status('summarizing_requested_variants')
            paths = [r['analysis'] for r in completed]
            summary = results.summarize(paths)
            gate = probes.require_tier3(paths)
            if upstream_completion(plan) != upstream:
                raise ValueError('Training evidence changed during reporting')
            for record in completed:
                for name in ('evaluation', 'analysis'):
                    if sha256(Path(record[name])) != record[name + '_sha256']:
                        raise ValueError('Completed reporting evidence changed')
            done = dict(phase='requested_selected_evaluations_and_comparisons_complete_probes_pending',
                plan_sha256=sha256(target / 'plan.json'), upstream_sha256=sha256(target / 'upstream_completion.json'),
                evidence=completed, summary=str(summary), summary_sha256=sha256(summary), tier3_gate=gate,
                new_fits=0, additional_seed_fits=0, parameter_search=False, test_used_for_selection=False,
                scientific_study_complete=False, finished_unix=time.time())
            immutable(target / 'complete.json', done); status(done['phase'])
            return done
        except BaseException as error:
            status('stopped_for_inspection', error_type=type(error).__name__, error=str(error))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    sub = actions.add_parser('prepare')
    sub.add_argument('--training-pid', required=True, type=int)
    sub.add_argument('--training-start-ticks', required=True, type=int)
    actions.add_parser('run')
    args = parser.parse_args()
    print(prepare(args.training_pid, args.training_start_ticks) if args.action == 'prepare' else json.dumps(run(), indent=2))
