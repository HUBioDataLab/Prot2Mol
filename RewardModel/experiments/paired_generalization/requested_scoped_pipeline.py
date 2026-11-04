"""Explicitly rebind existing reporting/probe executors in new namespaces.

Their scientific code and old plans remain unchanged. Only queue destinations
and exact upstream controller bindings change in this process. Every amended
plan pins this adapter, the replacement training plan, and original evidence.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path

from .run import ROOT, WORK, sha256
from . import requested_scoped_training as training
from . import requested_reporting_queue as reporting, requested_probe_queue as probes

cq = reporting.cq
read, immutable = cq.read, cq.immutable


def reporting_destination():
    return WORK / 'controlled/requested_scoped_reporting_20261104'


def probe_destination():
    return WORK / 'controlled/requested_scoped_probes_20261104'


def configure():
    reporting.destination = reporting_destination
    probes.destination = probe_destination
    if probes.reporting is not reporting:
        raise ValueError('Probe executor must use the same rebound reporting module')


def module_for(role):
    if role not in ('reporting', 'probes'):
        raise ValueError('Unknown rebound stage')
    return reporting if role == 'reporting' else probes


def expected_parent(role):
    prefix = [str(ROOT / '.venv/bin/python'), '-m']
    if role == 'reporting':
        return prefix + ['experiments.paired_generalization.requested_scoped_training', 'run']
    return prefix + ['experiments.paired_generalization.requested_scoped_pipeline', 'run', 'reporting']


def prepare(role, pid, start_ticks):
    cq.runtime_guard(); configure(); module = module_for(role)
    parent_path = (training.destination() if role == 'reporting' else reporting_destination()) / 'plan.json'
    parent_plan = read(parent_path)
    if role == 'reporting':
        training.validate(parent_plan)
    else:
        validate_rebinding('reporting', parent_plan)
    owner = cq.process_identity(pid); invocation = read(parent_path.parent / 'invocation.json')
    if (owner is None or owner['start_ticks'] != start_ticks or owner['uid'] != os.getuid() or
            owner['cwd'] != str(ROOT) or owner['argv'] != expected_parent(role) or
            invocation['pid'] != pid or invocation['plan_sha256'] != sha256(parent_path)):
        raise ValueError('Require exact live owned replacement upstream controller')
    original = WORK / ('controlled/requested_reporting_queue/plan.json' if role == 'reporting'
                       else 'controlled/requested_probe_queue/plan.json')
    plan = copy.deepcopy(read(original))
    if role == 'reporting':
        plan['training_controller'] = owner
    else:
        plan.update(reporting_plan=str(parent_path), reporting_plan_sha256=sha256(parent_path),
                    reporting_controller=owner)
    paths = [Path(__file__), Path(training.__file__), Path(training.retention.__file__), original,
             parent_path, parent_path.parent / 'invocation.json', training.destination() / 'plan.json']
    plan['files_sha256'].update({str(p.relative_to(ROOT)): sha256(p) for p in paths})
    plan['rebinding'] = dict(role=role, original_plan=str(original), original_plan_sha256=sha256(original),
        upstream_plan=str(parent_path), upstream_plan_sha256=sha256(parent_path),
        training_amendment=str(training.destination() / 'plan.json'),
        training_amendment_sha256=sha256(training.destination() / 'plan.json'),
        scientific_execution_unchanged=True, original_plans_preserved=True)
    path = module.destination() / 'plan.json'; immutable(path, plan)
    validate_rebinding(role, plan)
    return path


def validate_rebinding(role, plan):
    configure(); module = module_for(role); binding = plan['rebinding']
    parent = (training.destination() if role == 'reporting' else reporting_destination()) / 'plan.json'
    owner = plan['training_controller' if role == 'reporting' else 'reporting_controller']
    if (binding['role'] != role or binding['upstream_plan'] != str(parent) or
            binding['upstream_plan_sha256'] != sha256(parent) or
            binding['training_amendment'] != str(training.destination() / 'plan.json') or
            binding['training_amendment_sha256'] != sha256(training.destination() / 'plan.json') or
            sha256(Path(binding['original_plan'])) != binding['original_plan_sha256'] or
            owner['argv'] != expected_parent(role) or not binding['scientific_execution_unchanged'] or
            not binding['original_plans_preserved']):
        raise ValueError('Explicit queue rebinding changed')
    training.validate(read(training.destination() / 'plan.json'))
    module.validate(plan)


def run(role):
    configure(); module = module_for(role)
    validate_rebinding(role, read(module.destination() / 'plan.json'))
    return module.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('role', choices=('reporting', 'probes'))
    parser.add_argument('--parent-pid', type=int); parser.add_argument('--parent-start-ticks', type=int)
    args = parser.parse_args()
    if args.action == 'prepare' and (args.parent_pid is None or args.parent_start_ticks is None):
        parser.error('prepare requires the upstream PID and start ticks')
    print(prepare(args.role, args.parent_pid, args.parent_start_ticks) if args.action == 'prepare'
          else json.dumps(run(args.role), indent=2))
