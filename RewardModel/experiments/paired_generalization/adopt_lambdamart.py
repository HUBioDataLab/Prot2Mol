"""Publish an independently completed grid setting to its live sequential driver."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time

if __package__:
    from .run import ROOT, WORK, sha256, write_json
else:
    from run import ROOT, WORK, sha256, write_json


def read(path):
    return json.loads(path.read_text())


def verify_trial(source, destination, number):
    """Require identical data, features, budget, parameters and complete selection history."""
    contract = read(source/'fit_contract.json')
    main = read(destination/'fit_contract.json')
    if contract.pop('precomputed_trial', None) != number or contract != main:
        raise ValueError('Independent and main fit contracts differ')
    complete = read(source/'complete.json')
    for key, value in main.items():
        if key != 'base_parameters' and complete.get(key) != value:
            raise ValueError(f'Worker completion differs from contract: {key}')
    if complete.get('precomputed_trial') != number:
        raise ValueError('Wrong completed worker')
    trial = source/f'trial{number:02d}'
    result = read(trial/'complete.json')
    if complete['trials'] != [dict(trial=number, **result)]:
        raise ValueError('Parent and trial completion disagree')
    expected = {**main['base_parameters'], **main['settings'][number]}
    if result['parameters'] != expected:
        raise ValueError('Trial parameters differ')
    history = read(trial/'validation_history.json')
    records = history['history']
    rounds = result['rounds_trained']
    every, maximum, patience = (main[key] for key in ('evaluate_every', 'max_rounds', 'patience_rounds'))
    if not 0 < rounds <= maximum or rounds % every:
        raise ValueError('Invalid completed round count')
    if [row['rounds'] for row in records] != list(range(every, rounds+1, every)):
        raise ValueError('Incomplete validation history')
    best = None
    for index, row in enumerate(records):
        score = row['macro_spearman']
        if score is not None:
            if not math.isfinite(score) or not -1 <= score <= 1:
                raise ValueError('Invalid validation score')
            if best is None or score > best['macro_spearman']:
                best = row
        stopped = best is not None and row['rounds'] - best['rounds'] >= patience
        if stopped and index != len(records)-1:
            raise ValueError('History continued beyond registered stopping rule')
    if best is None or history['selected'] != best or result['selected'] != best:
        raise ValueError('Selected checkpoint is not the first validation maximum')
    if rounds != maximum and rounds - best['rounds'] < patience:
        raise ValueError('Trial ended before its declared budget/stopping rule')
    if sha256(trial/'model.ubj') != result['model_sha256']:
        raise ValueError('Selected model checksum differs')
    return result


@contextmanager
def pause_driver(pid, regime):
    """Pin one owned Linux process; always resume it, including publication errors."""
    descriptor = os.pidfd_open(pid)
    sent = False
    try:
        proc = Path(f'/proc/{pid}')
        args = proc.joinpath('cmdline').read_bytes().decode().split('\0')[:-1]
        expected = ['-u', 'RewardModel/experiments/paired_generalization/lambdamart.py', '--regime', regime]
        if (proc.stat().st_uid != os.getuid() or proc.joinpath('cwd').resolve() != ROOT
                or args[1:] != expected or Path(args[0]).name != 'python'):
            raise ValueError('PID is not the owned sequential LambdaMART driver')
        states = [p.read_text().split(') ', 1)[1].split()[0] for p in proc.glob('task/*/stat')]
        if not states or any(s in ('T', 't', 'Z', 'X') for s in states):
            raise ValueError('Driver is already stopped or terminal')
        signal.pidfd_send_signal(descriptor, signal.SIGSTOP)
        sent = True
        deadline = time.monotonic()+10
        while True:
            states = [p.read_text().split(') ', 1)[1].split()[0] for p in proc.glob('task/*/stat')]
            if states and all(s == 'T' for s in states):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Driver did not reach a stopped state')
            time.sleep(.01)
        yield
    finally:
        try:
            if sent:
                signal.pidfd_send_signal(descriptor, signal.SIGCONT)
        finally:
            os.close(descriptor)


def publish(source, destination, number, guard):
    result = verify_trial(source, destination, number)
    trial = source/f'trial{number:02d}'
    target = destination/trial.name
    if target.exists():
        raise FileExistsError('Main driver already reached this setting; leave it intact')
    temporary = Path(tempfile.mkdtemp(prefix=f'.adopt_{number:02d}_', dir=destination))
    try:
        for name in ('model.ubj', 'complete.json', 'validation_history.json'):
            shutil.copyfile(trial/name, temporary/name)
            if sha256(trial/name) != sha256(temporary/name):
                raise ValueError('Copied trial checksum differs')
        write_json(temporary/'adoption.json', dict(source=str(source), trial=number,
            source_contract_sha256=sha256(source/'fit_contract.json'),
            source_completion_sha256=sha256(source/'complete.json'),
            source_history_sha256=sha256(trial/'validation_history.json'),
            model_sha256=result['model_sha256'], adopted_utc=datetime.now(timezone.utc).isoformat(),
            scope='Same fixed grid, inputs and stopping rule; selected using validation only'))
        # Copy/hash first. The sequential driver is paused only across the final
        # existence check and atomic directory rename, so it cannot start this fit.
        with guard:
            if target.exists() or (destination/'complete.json').exists():
                raise FileExistsError('Main advanced before publication; leave its work intact')
            if read(source/'fit_contract.json') != {**read(destination/'fit_contract.json'), 'precomputed_trial': number}:
                raise ValueError('Fit contract changed before publication')
            temporary.rename(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return result


def main(regime, number, pid):
    if ROOT != Path('/home/atabey/Prot2Mol'):
        raise RuntimeError('Use the authorized Atum project')
    source = WORK/'tier2/lambdamart_precomputed'/regime/f'worker{number:02d}'
    destination = WORK/'tier2/lambdamart'/regime
    with (source/'fit.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = publish(source, destination, number, pause_driver(pid, regime))
    print(json.dumps(dict(regime=regime, trial=number, main_pid=pid, result=result)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--regime', choices=['A', 'B'], required=True)
    parser.add_argument('--trial', type=int, choices=range(5), required=True)
    parser.add_argument('--main-pid', type=int, required=True)
    args = parser.parse_args()
    main(args.regime, args.trial, args.main_pid)
