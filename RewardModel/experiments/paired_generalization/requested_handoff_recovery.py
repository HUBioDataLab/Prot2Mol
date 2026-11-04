"""Recover the inspected controller-exit race without rerunning scientific work.

The first A fit finished successfully. The previous dispatcher stopped while
observing a reporter after SIGTERM, before releasing the held legacy parent.
New namespaces preserve that failed invocation and its partial retirement.
Native fitting, retention, reporting and probe executors remain unchanged.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import time

from .run import ROOT, WORK, sha256
from . import requested_scoped_training as training, requested_scoped_pipeline as pipeline

cq = training.cq
read, immutable = cq.read, cq.immutable
_native_validate = training.validate
INTENT_SHA256 = 'f0e36a59d850fd8d5b84f5b3892c478f3f2f032ea248ee0f01cf8f530a72c884'
FAILURE_SHA256 = '0cf9e00228d8e65d4e407076b9e22913d9fcef06802403ee456c31b2428a5b0a'
ACTIVATION_SHA256 = 'a96f704ab882ab93acebf80c8302e66519304dbb5833bb3dddf8b7dcadf3bea2'


def old_destination():
    return WORK / 'controlled/requested_scoped_training_20261104'


def destination():
    return WORK / 'controlled/requested_handoff_recovery_20261104'


def reporting_destination():
    return WORK / 'controlled/requested_recovered_reporting_20261104'


def probe_destination():
    return WORK / 'controlled/requested_recovered_probes_20261104'


def activation_path():
    return WORK / 'requested_scoped_runtime/e9d5b33/activation.json'


def expected_parent(role):
    return [str(ROOT / '.venv/bin/python'), '-m',
            'experiments.paired_generalization.requested_handoff_recovery', 'run',
            'training' if role == 'reporting' else 'reporting']


def configure():
    """Explicit adapters in this new process; no original source or plan edits."""
    training.destination = destination
    training.validate = validate
    training.retire_controllers = recover_retirement
    pipeline.reporting_destination = reporting_destination
    pipeline.probe_destination = probe_destination
    pipeline.expected_parent = expected_parent
    pipeline.configure()


def terminal_observation(owner):
    """None means live or temporarily unobservable, never inferred success."""
    state = training.process_state(owner['pid'])
    if state is None:
        return dict(reason='handle_missing', original=owner)
    if state['start_ticks'] != owner['start_ticks']:
        return dict(reason='pid_replaced', original=owner, observed=state)
    if state['uid'] != owner['uid']:
        raise ValueError('Original process ownership changed')
    if state['state'] in ('Z', 'X'):
        return dict(reason='terminal_state', original=owner, observed=state)
    try:
        current = cq.process_identity(owner['pid'])
    except (OSError, RuntimeError):
        # /proc can expose an empty command between live stat and zombie stat.
        # Retry this exact handle, without another signal or a new process.
        return None
    if current is None:
        return None  # Require a fresh stat observation on the next attempt.
    if current != owner:
        raise ValueError('Original process identity changed while still live')
    return None


def wait_terminal(owner, attempts=200):
    for _ in range(attempts):
        try:
            result = terminal_observation(owner)
        except OSError:
            result = None
        if result is not None:
            return result
        time.sleep(.05)
    raise RuntimeError('Terminal state remains unverified; preserve work for inspection')


def failure_evidence():
    old = old_destination()
    if (sha256(old/'retirement_intent.json') != INTENT_SHA256 or
            sha256(old/'status.json') != FAILURE_SHA256 or
            sha256(activation_path()) != ACTIVATION_SHA256):
        raise ValueError('Inspected handoff failure evidence changed')
    prior = read(old/'plan.json'); intent = read(old/'retirement_intent.json')
    status = read(old/'status.json'); activation = read(activation_path())
    if (status['phase'] != 'stopped_for_inspection' or status['completed_jobs'] or
            status['error'] != 'Live process has no inspectable command' or
            intent['plan_sha256'] != sha256(old/'plan.json') or intent['training_signals_sent'] or
            activation['plans']['training']['sha256'] != sha256(old/'plan.json')):
        raise ValueError('Recovery is restricted to the inspected pre-adoption exit race')
    if any((old/name).exists() for name in ('retirement_complete.json','training_controller_retired.json',
                                           'reporting_controller_retired.json','complete.json')):
        raise ValueError('Unexpected additional progress in the failed invocation')
    proof = read(old/'probe_controller_retired.json')
    if (proof['identity'] != prior['held']['probe_controller'] or proof['signal'] != int(signal.SIGTERM)
            or not proof['verified_not_live'] or proof['training_signals_sent']):
        raise ValueError('Partial probe retirement evidence differs')
    observations = {}
    for role, owner in activation['controllers'].items():
        if role not in ('training','reporting','probes'):
            raise ValueError('Unexpected failed-controller role')
        observations['failed_'+role] = wait_terminal(owner)
        path = Path(activation['plans'][role]['path'])
        if (sha256(path) != activation['plans'][role]['sha256'] or
                read(path.parent/'status.json')['phase'] != 'stopped_for_inspection' or
                (path.parent/'complete.json').exists()):
            raise ValueError('A failed downstream queue progressed unexpectedly')
    for role in ('probe_controller','reporting_controller'):
        observations[role] = wait_terminal(prior['held'][role])
    return prior, intent, activation, observations


def prepare():
    cq.runtime_guard()
    # Validate the original scope before applying any process-local adapters.
    previous = training.destination
    training.destination = old_destination
    try:
        _native_validate(read(old_destination()/'plan.json'))
    finally:
        training.destination = previous
    prior, intent, activation, observations = failure_evidence()
    if training.current_terminal(prior) != intent['terminal_launcher']:
        raise ValueError('Original successful A launcher evidence changed')
    scope = read(Path(prior['scope_plan'])); job = scope['jobs'][0]
    native = training.native_completion(job, scope['references']['qualification'])
    if native != intent['native_completion_sha256']:
        raise ValueError('Completed A evidence changed after the interrupted handoff')
    training.requested.require_unstarted(scope['jobs'][1:])
    output = Path(job['output_dir'])
    if any((output/name).exists() for name in ('controlled_training_complete.json','retained/retention.json')):
        raise ValueError('A adoption progressed; inspect rather than repeat it')
    paths = [Path(__file__), old_destination()/'plan.json', old_destination()/'invocation.json',
             old_destination()/'status.json', old_destination()/'retirement_intent.json',
             old_destination()/'probe_controller_retired.json', activation_path(),
             activation_path().parent/'training_controller.log']
    for item in activation['plans'].values():
        path = Path(item['path']); paths.extend([path,path.parent/'invocation.json',path.parent/'status.json'])
    paths.extend(output/name for name in native)
    plan = copy.deepcopy(prior)
    plan['files_sha256'].update({str(p.relative_to(ROOT)):sha256(p) for p in paths})
    plan['recovery'] = dict(original_plan=str(old_destination()/'plan.json'),
        original_plan_sha256=sha256(old_destination()/'plan.json'),
        failed_intent_sha256=INTENT_SHA256, failed_status_sha256=FAILURE_SHA256,
        terminal_launcher=intent['terminal_launcher'], native_completion_sha256=native,
        terminal_observations=observations, new_scientific_fits=0,
        reason='Reporter exited between /proc stat and command reads after SIGTERM; do not retrain A or repeat retired-controller signals.')
    configure(); path=destination()/'plan.json'; immutable(path,plan); validate(plan)
    return path


def validate(plan):
    _native_validate(plan)
    recovery = plan['recovery']
    if (recovery['original_plan'] != str(old_destination()/'plan.json') or
            recovery['original_plan_sha256'] != sha256(old_destination()/'plan.json') or
            recovery['failed_intent_sha256'] != INTENT_SHA256 or
            recovery['failed_status_sha256'] != FAILURE_SHA256 or recovery['new_scientific_fits']):
        raise ValueError('Inspected recovery scope changed')
    failure_evidence()
    return read(Path(plan['scope_plan']))


def signal_held_parent(owner):
    """Retire the exact held owner; use a pidfd when Python exposes that API."""
    if not (hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal')):
        # Some deployed Python builds omit pidfd APIs. This parent is stopped
        # and cannot voluntarily exit; revalidate UID, cwd, argv and start time
        # immediately before the same exact-PID signal used by the old driver.
        training.require_live(owner, 'T')
        os.kill(owner['pid'], signal.SIGKILL)
        return 'revalidated_held_pid'
    descriptor = os.pidfd_open(owner['pid'])
    try:
        training.require_live(owner, 'T')
        signal.pidfd_send_signal(descriptor, signal.SIGKILL)
        return 'pidfd'
    finally:
        os.close(descriptor)


def recover_retirement(plan, terminal, native):
    prior, failed, activation, observations = failure_evidence()
    if (training.current_terminal(plan) != terminal or terminal != failed['terminal_launcher'] or
            native != failed['native_completion_sha256'] or
            native != plan['recovery']['native_completion_sha256']):
        raise ValueError('Successful A evidence changed before releasing the held parent')
    target = destination(); held=plan['held']
    immutable(target/'retirement_intent.json',dict(plan_sha256=sha256(target/'plan.json'),
        failed_intent=str(old_destination()/'retirement_intent.json'), failed_intent_sha256=INTENT_SHA256,
        terminal_launcher=terminal,native_completion_sha256=native,controllers=failed['controllers'],
        training_signals_sent=False,created_unix=time.time()))
    for role in ('probe_controller','reporting_controller'):
        immutable(target/(role+'_retired.json'),dict(identity=held[role],
            terminal_observation=observations[role],signal_sent_by_recovery=False,
            earlier_retirement_intent_sha256=INTENT_SHA256,verified_not_live=True,checked_unix=time.time()))
    owner=held['training_controller']; mechanism=signal_held_parent(owner)
    observed=wait_terminal(owner)
    immutable(target/'training_controller_retired.json',dict(identity=owner,signal=int(signal.SIGKILL),
        terminal_observation=observed,signal_mechanism=mechanism,
        signal_sent_by_recovery=True,training_signals_sent=False,
        verified_not_live=True,checked_unix=time.time()))
    result=dict(intent_sha256=sha256(target/'retirement_intent.json'),
        retired={role:sha256(target/(role+'_retired.json')) for role in failed['controllers']},
        current_fit_restarted=False,training_signals_sent=False,finished_unix=time.time())
    immutable(target/'retirement_complete.json',result)
    return result


def run(role):
    configure()
    if role == 'training':
        return training.run()
    return pipeline.run(role)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('role',choices=('training','reporting','probes'))
    parser.add_argument('--parent-pid',type=int);parser.add_argument('--parent-start-ticks',type=int)
    args=parser.parse_args()
    if args.action=='run': result=run(args.role)
    elif args.role=='training': result=prepare()
    else:
        if args.parent_pid is None or args.parent_start_ticks is None:
            parser.error('Downstream preparation requires exact parent PID and start ticks')
        configure();result=pipeline.prepare(args.role,args.parent_pid,args.parent_start_ticks)
    print(json.dumps(result,indent=2) if isinstance(result,dict) else result)
