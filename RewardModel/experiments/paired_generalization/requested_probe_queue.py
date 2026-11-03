"""Wait for the fixed seed42 report, then run the requested mechanistic probes.

Bind the existing reporting process by PID, start time, owner and command.
Never restart it or the training queue. Waiting is CPU-only and sparse; each
probe executor independently verifies its Tier3 inputs and owns the GPU lock.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from .run import ROOT, WORK, sha256, write_json
from . import requested_reporting_queue as reporting, requested_probe_protocol as protocol
from . import requested_adaptation_queue as adaptation
from . import requested_geometry as geometry, requested_objectives as objectives

cq = reporting.cq
read, immutable = cq.read, cq.immutable
ORDER = ('adaptation', 'geometry', 'objectives')
MODULES = dict(adaptation=adaptation, geometry=geometry, objectives=objectives)
WAIT_SECONDS = 600


class ObservationUnavailable(RuntimeError):
    """A transient process-inspection failure is not proof of process exit."""


def destination():
    return WORK / 'controlled/requested_probe_queue'


def software():
    value = dict(source_sha256={}, packages={})
    for module in (adaptation, geometry, objectives):
        current = module.software()
        for field in ('source_sha256', 'packages'):
            for name, digest in current[field].items():
                if name in value[field] and value[field][name] != digest:
                    raise ValueError('Probe executors disagree about a dependency')
                value[field][name] = digest
    for path in (Path(__file__), Path(reporting.__file__), Path(protocol.__file__)):
        value['source_sha256'][str(path.relative_to(ROOT))] = sha256(path)
    return value


def bind_reporting(pid, start_ticks, path):
    identity = cq.process_identity(pid); invocation = read(path.parent / 'invocation.json')
    expected = [str(ROOT / '.venv/bin/python'), str(Path(reporting.__file__).absolute()), 'run']
    if (identity is None or identity['start_ticks'] != start_ticks or identity['uid'] != os.getuid()
            or identity['cwd'] != str(ROOT) or identity['argv'] != expected
            or invocation['pid'] != pid or invocation['plan_sha256'] != sha256(path)):
        raise ValueError('Require the exact live owned requested-reporting controller')
    return identity


def prepare(reporting_pid, reporting_start_ticks):
    cq.runtime_guard()
    path = reporting.destination() / 'plan.json'; report_plan = read(path)
    reporting.validate(report_plan); probe = protocol.verify()
    if report_plan['probe_plan_sha256'] != sha256(protocol.destination() / 'plan.json'):
        raise ValueError('Reporting and mechanistic probe protocols differ')
    owner = bind_reporting(reporting_pid, reporting_start_ticks, path)
    code = software(); files = dict(code['source_sha256'])
    for item in (path, path.parent / 'invocation.json', protocol.destination() / 'plan.json', protocol.destination() / 'complete.json'):
        files[str(item.relative_to(ROOT))] = sha256(item)
    plan = dict(format_version=1, boot_id=cq.boot_id(), seed=42, learning_rate_multiplier=1.,
        reporting_plan=str(path), reporting_plan_sha256=sha256(path), reporting_controller=owner,
        job_ids=report_plan['job_ids'], probe_plan_sha256=sha256(protocol.destination() / 'plan.json'),
        order=list(ORDER), wait_seconds=WAIT_SECONDS, gpu_uuids=list(reporting.evaluation.GPU_UUIDS),
        files_sha256=files, packages=code['packages'], support_fit_upper_bound=probe['accounting']['adaptation_support_fits'],
        geometry_exports=probe['accounting']['geometry_exports'], objective_pairs=probe['accounting']['objective_batch_pairs'],
        additional_primary_fits=0, parameter_search=False, additional_seed_fits=0,
        description='After all26 fixed selected-model analyses, execute the frozen adaptation, geometry and objective probes. Keep every prescribed task/cohort/batch and explicit unavailable cases. No source/data redraw, new seed, LR search or upstream relaunch. Selected-audit coverage and final34 answers remain a separate completion audit.')
    target = destination() / 'plan.json'; immutable(target, plan)
    return target


def validate(plan):
    cq.runtime_guard()
    path = Path(plan['reporting_plan'])
    if (read(destination() / 'plan.json') != plan or plan['boot_id'] != cq.boot_id()
            or path != reporting.destination() / 'plan.json' or sha256(path) != plan['reporting_plan_sha256']
            or plan['probe_plan_sha256'] != sha256(protocol.destination() / 'plan.json')
            or plan['seed'] != 42 or plan['learning_rate_multiplier'] != 1.
            or plan['order'] != list(ORDER) or plan['wait_seconds'] != WAIT_SECONDS
            or plan['gpu_uuids'] != list(reporting.evaluation.GPU_UUIDS)
            or plan['additional_primary_fits'] or plan['parameter_search'] or plan['additional_seed_fits']):
        raise ValueError('Requested probe queue scope or execution environment changed')
    from importlib.metadata import version
    if any(version(n) != v for n, v in plan['packages'].items()):
        raise ValueError('Requested probe queue packages changed')
    for name, digest in plan['files_sha256'].items():
        if sha256(ROOT / name) != digest: raise ValueError('Requested probe input or source changed: '+name)
    report_plan = read(path); reporting.validate(report_plan)
    probe = read(protocol.destination() / 'plan.json')
    if (plan['job_ids'] != report_plan['job_ids'] or len(plan['job_ids']) != 26
            or len(set(plan['job_ids'])) != 26
            or plan['support_fit_upper_bound'] != probe['accounting']['adaptation_support_fits']
            or plan['geometry_exports'] != 6 or plan['objective_pairs'] != 64):
        raise ValueError('Requested probe inventory changed')
    return report_plan


def reporting_ready(plan):
    """Process exit is never interpreted as successful upstream completion."""
    path = Path(plan['reporting_plan']); target = path.parent
    if cq.boot_id() != plan['boot_id'] or sha256(path) != plan['reporting_plan_sha256']:
        raise ValueError('Bound reporting plan or boot changed while waiting')
    if (target / 'complete.json').exists(): return True
    status = read(target / 'status.json')
    if status['phase'] == 'stopped_for_inspection': raise RuntimeError('Reporting stopped; probes cannot start')
    owner = plan['reporting_controller']
    try:
        current = cq.process_identity(owner['pid'])
    except (OSError, RuntimeError) as error:
        raise ObservationUnavailable(str(error)) from error
    if current != owner:
        if (target / 'complete.json').exists(): return True
        raise RuntimeError('Reporting controller ended without completion')
    return False


def upstream_completion(plan):
    report_plan = validate(plan); path = Path(plan['reporting_plan']); target = path.parent
    done = read(target / 'complete.json'); upstream = read(target / 'upstream_completion.json')
    if (done['phase'] != 'requested_selected_evaluations_and_comparisons_complete_probes_pending'
            or done['plan_sha256'] != sha256(path) or done['upstream_sha256'] != sha256(target / 'upstream_completion.json')
            or upstream != reporting.upstream_completion(report_plan)
            or [r['job_id'] for r in done['evidence']] != plan['job_ids']
            or done['new_fits'] or done['additional_seed_fits'] or done['parameter_search']
            or done['test_used_for_selection'] or done['scientific_study_complete']):
        raise ValueError('Requested selected-model reporting is incomplete or changed')
    for record in done['evidence']:
        identity = record['job_id']
        expected = reporting.evaluation.destination(identity) / 'complete.json'
        if record['evaluation'] != str(expected): raise ValueError('Wrong selected-model evaluation')
        for name in ('evaluation', 'analysis'):
            item = Path(record[name])
            if not item.is_relative_to(ROOT) or sha256(item) != record[name+'_sha256']:
                raise ValueError('Selected-model evidence changed')
        reporting.evaluation.verify_complete(expected.parent / 'plan.json')
    analyses = [record['analysis'] for record in done['evidence']]
    gate = protocol.require_tier3(analyses)
    if gate != done['tier3_gate']: raise ValueError('Reporting Tier3 gate differs from verified native analyses')
    summary_path = Path(done['summary'])
    if not summary_path.is_relative_to(WORK / 'requested_summary') or sha256(summary_path) != done['summary_sha256']:
        raise ValueError('Selected-model comparison report changed')
    summary = read(summary_path)
    if (summary['phase'] != 'requested_fixed_seed_comparison_complete' or summary['training_fits'] != 26
            or summary['new_fits'] or summary['seed_uncertainty_available'] or summary['test_used_for_selection']
            or summary['source']['analyses_sha256'] != gate['analysis_sha256']
            or summary['source']['software'] != reporting.results.software()):
        raise ValueError('Requested variant summary omits fixed analyses or adds a selection')
    reporting.results.diagnostics.verify_artifacts(summary_path.parent, summary)
    return dict(path=str(target / 'complete.json'), sha256=sha256(target / 'complete.json'), analyses=analyses,
        tier3_gate=gate, summary=str(summary_path), summary_sha256=sha256(summary_path))


def verify_probe(name, path):
    path = Path(path); module = MODULES[name]
    if name == 'adaptation':
        record, report = module.verify_complete(path)
        return dict(path=str(path.parent / 'complete.json'), sha256=sha256(path.parent / 'complete.json'),
            report=record['report'], report_sha256=record['report_sha256'], support_fits=report['new_support_fits'])
    module.verify_plan(path); record = read(path.parent / 'complete.json')
    expected = 'requested_reference_geometry_complete' if name == 'geometry' else 'requested_reference_objective_probes_complete'
    if (record['phase'] != expected or record['plan_sha256'] != sha256(path) or record['new_fits']
            or record['scientific_study_complete'] or set(record['evidence_sha256']) != {'A','B'}
            or (name == 'objectives' and record['test_data_read'])):
        raise ValueError('Probe completion changed its scope')
    measured = {}
    for regime in ('A','B'):
        if record['evidence_sha256'][regime] != sha256(path.parent / regime / 'complete.json'):
            raise ValueError('Probe worker completion changed')
        values = module.verify_worker(path, regime)
        measured[regime] = ({split: value['analysis'] for split, value in values[1].items()} if name == 'geometry'
            else dict(summary=values[1]['summary'], qualification=values[2]))
    if measured != record['results']: raise ValueError('Probe summary differs from native worker evidence')
    return dict(path=str(path.parent / 'complete.json'), sha256=sha256(path.parent / 'complete.json'), new_fits=0)


def run():
    import fcntl
    target = destination(); plan = read(target / 'plan.json'); validate(plan)
    with (target / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (target / 'invocation.json').exists(): raise RuntimeError('Probe queue already invoked; inspect, do not restart')
        immutable(target / 'invocation.json', dict(pid=os.getpid(), started_unix=time.time(), plan_sha256=sha256(target / 'plan.json')))
        completed = {}
        def status(phase, **values):
            write_json(target / 'status.json', dict(phase=phase, pid=os.getpid(), checked_unix=time.time(),
                completed_probes=list(completed), **values))
        try:
            while True:
                try:
                    if reporting_ready(plan): break
                    status('waiting_for_requested_reporting', next_check_seconds=WAIT_SECONDS)
                except ObservationUnavailable as error:
                    status('waiting_for_reporting_observation', observation_error=str(error), next_check_seconds=WAIT_SECONDS)
                time.sleep(WAIT_SECONDS)
            upstream = upstream_completion(plan); immutable(target / 'upstream_completion.json', upstream)
            for name in ORDER:
                validate(plan)
                if sha256(Path(upstream['path'])) != upstream['sha256']:
                    raise ValueError('Reporting evidence changed during probes')
                status('preparing_'+name)
                path = MODULES[name].prepare(upstream['analyses'])
                immutable(target / f'{name}_plan.json', dict(path=str(path), sha256=sha256(path)))
                while True:
                    status('running_'+name, execution_plan=str(path))
                    result = MODULES[name].run(path)
                    if result['phase'] not in reporting.WAIT_PHASES: break
                    status(result['phase'], probe=name, resource_state=result, next_check_seconds=WAIT_SECONDS)
                    time.sleep(WAIT_SECONDS); validate(plan)
                completed[name] = verify_probe(name, path)
                immutable(target / f'{name}_complete.json', completed[name])
            validate(plan)
            for result in completed.values():
                if sha256(Path(result['path'])) != result['sha256']:
                    raise ValueError('Completed probe evidence changed')
            record = dict(phase='requested_mechanistic_probes_complete', plan_sha256=sha256(target / 'plan.json'),
                upstream_sha256=sha256(target / 'upstream_completion.json'), probes=completed,
                additional_primary_fits=0, parameter_search=False, additional_seed_fits=0,
                scientific_study_complete=False, finished_unix=time.time(),
                remaining='Audit selected E01-E05/E08-E09/E19-E20 coverage and update all34 answers under the user-narrowed scope. No additional seed, weight, rate or split-allocation search is authorized.')
            immutable(target / 'complete.json', record); status(record['phase'])
            return record
        except BaseException as error:
            status('stopped_for_inspection', error_type=type(error).__name__, error=str(error))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    sub = actions.add_parser('prepare'); sub.add_argument('--reporting-pid', type=int, required=True)
    sub.add_argument('--reporting-start-ticks', type=int, required=True)
    actions.add_parser('run')
    args = parser.parse_args()
    print(prepare(args.reporting_pid, args.reporting_start_ticks) if args.action == 'prepare' else json.dumps(run(), indent=2))
