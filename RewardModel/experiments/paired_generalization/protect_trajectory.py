"""Preserve the matched A reference's full trajectory before ordinary cleanup.

Native scalar exports cannot reconstruct cross-candidate contrastive losses or
representation geometry. Hard links retain each completed epoch without copying
its large payload. A reversible managed-run marker hold prevents the already
running driver's older retention function from deleting unprotected weights.
This changes no training configuration, optimizer, model or GPU process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from .control_plan import _write_immutable
else:
    from run import ROOT, WORK, sha256, write_json
    from control_plan import _write_immutable

REFERENCE = 'full__A__seed42__lr1p0'


def read(path):
    return json.loads(Path(path).read_text())


def immutable(path, record):
    _write_immutable(path,json.dumps(record,indent=2,sort_keys=True)+'\n')


def digest_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def atomic_bytes(path, payload):
    temporary = path.with_suffix(path.suffix+'.protection_tmp')
    with temporary.open('xb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def hold_retention(run, destination, config_sha256, horizon):
    """Hold only our managed retention; preserve the exact original marker."""
    run,destination = Path(run).resolve(),Path(destination)
    if int(horizon) != horizon or horizon < 1:
        raise ValueError('Expected a positive whole-epoch horizon')
    marker = run/'controlled_run.json'
    guard = destination/'guard.json'
    if guard.exists():
        record = read(guard)
        if (record['run'] != str(run) or record['config_sha256'] != config_sha256 or record['horizon'] != horizon or
            sha256(destination/'original_controlled_run.json') != record['original_marker_sha256']):
            raise ValueError('Existing protection belongs to different run provenance')
        if (destination/'complete.json').exists():
            verify_complete(destination)
            return record
        current = sha256(marker)
        if current not in (record['original_marker_sha256'],record['held_marker_sha256']):
            raise ValueError('Managed marker changed independently of this hold')
        # Crash recovery after publishing the guard, before installing its hold.
        if current == record['original_marker_sha256']:
            held = read(destination/'original_controlled_run.json')
            held['allow_checkpoint_retention'] = False
            payload = (json.dumps(held,indent=2,sort_keys=True)+'\n').encode()
            if digest_bytes(payload) != record['held_marker_sha256']:
                raise ValueError('Cannot reproduce held marker')
            atomic_bytes(marker,payload)
        return record
    if destination.exists():
        raise RuntimeError('Unregistered partial protection requires inspection')
    original = marker.read_bytes()
    original_record = json.loads(original)
    if (original_record.get('format_version') != 1 or original_record.get('allow_checkpoint_retention') is not True or
        Path(original_record['output_dir']).resolve() != run or original_record['config_sha256'] != config_sha256):
        raise ValueError('Expected this managed run before retention')
    if (run/'retained/retention.json').exists():
        raise ValueError('Install trajectory protection before any retention starts')
    held = dict(original_record,allow_checkpoint_retention=False)
    held_payload = (json.dumps(held,indent=2,sort_keys=True)+'\n').encode()
    record = dict(format_version=1,run=str(run),config_sha256=config_sha256,horizon=int(horizon),
        original_marker_sha256=digest_bytes(original),held_marker_sha256=digest_bytes(held_payload),
        purpose='Preserve all matched A epoch weights for MD contrastive-direction and representation analyses',
        training_configuration_changed=False,
        interpretation='Only post-training managed checkpoint cleanup is held. If the driver reaches retention before release, it fails safely without deleting payloads; inspect and finish retention afterward without retraining.')
    destination.mkdir(parents=True)
    with (destination/'original_controlled_run.json').open('xb') as handle:
        handle.write(original)
    immutable(guard,record)
    atomic_bytes(marker,held_payload)
    return record


def capture_checkpoint(run, destination, checkpoint, horizon):
    """Trainer writes trainer_state last; never capture an incomplete save."""
    run,checkpoint = Path(run).resolve(),Path(checkpoint)
    if checkpoint.is_symlink() or checkpoint.resolve().parent != run:
        raise ValueError('Checkpoint is not directly owned by this run')
    state_path = checkpoint/'trainer_state.json'
    if not state_path.exists():
        return None
    try:
        state = read(state_path)
    except json.JSONDecodeError:
        return None  # The writer has not committed a complete state yet.
    step = int(checkpoint.name.split('-')[1])
    epoch = state.get('epoch')
    if (epoch is None or not 1 <= epoch <= horizon or abs(epoch-round(epoch)) > 1e-6 or
        state['global_step'] != step):
        raise ValueError('Unexpected checkpoint epoch/step')
    target = destination/f'epoch{round(epoch):02d}'
    complete = target/'complete.json'
    if complete.exists():
        record = read(complete)
        if record['step'] != step or record['epoch'] != round(epoch):
            raise ValueError('One epoch has multiple checkpoint identities')
        return record
    files = ['pytorch_model.bin','config.json','trainer_state.json']
    if any(not (checkpoint/name).exists() for name in files):
        return None
    if any((checkpoint/name).is_symlink() for name in files):
        raise ValueError('Checkpoint payload must not be a symlink')
    hashes = {name:sha256(checkpoint/name) for name in files}
    target.mkdir(parents=True,exist_ok=True)
    for name in files:
        source, saved = checkpoint/name,target/name
        if not saved.exists():
            if name == 'pytorch_model.bin':
                # Both paths are under the already archived controlled tree.
                # A cross-filesystem failure is explicit; never silently copy GBs.
                os.link(source,saved)
            else:
                with saved.open('xb') as handle:
                    handle.write(source.read_bytes())
        if sha256(saved) != hashes[name] or sha256(source) != hashes[name]:
            raise ValueError('Checkpoint changed during protection')
    record = dict(epoch=round(epoch),step=step,source=str(checkpoint),path=str(target),
        files_sha256=hashes,weights_bytes=(target/'pytorch_model.bin').stat().st_size,
        storage='hard-linked weights; separately copied configuration and trainer state')
    immutable(complete,record)
    return record


def verify_epochs(destination, horizon):
    records = []
    for epoch in range(1,horizon+1):
        directory = destination/f'epoch{epoch:02d}'
        record = read(directory/'complete.json')
        if record['epoch'] != epoch or Path(record['path']) != directory:
            raise ValueError('Protected epoch identity changed')
        for name,digest in record['files_sha256'].items():
            if sha256(directory/name) != digest:
                raise ValueError('Protected checkpoint payload changed')
        records.append(record)
    if len({r['step'] for r in records}) != horizon or [r['step'] for r in records] != sorted(r['step'] for r in records):
        raise ValueError('Protected steps are duplicated or out of order')
    return records


def release_retention(destination):
    """Release only after every epoch's independent retained path verifies."""
    guard = read(destination/'guard.json')
    records = verify_epochs(destination,guard['horizon'])
    original = (destination/'original_controlled_run.json').read_bytes()
    if digest_bytes(original) != guard['original_marker_sha256']:
        raise ValueError('Original managed marker changed')
    marker = Path(guard['run'])/'controlled_run.json'
    if sha256(marker) not in (guard['held_marker_sha256'],guard['original_marker_sha256']):
        raise ValueError('Refusing to overwrite independently modified managed marker')
    if sha256(marker) != guard['original_marker_sha256']:
        atomic_bytes(marker,original)
    result = dict(format_version=1,run=guard['run'],config_sha256=guard['config_sha256'],
        guard_sha256=sha256(destination/'guard.json'),epochs=records,retention_hold_released=True,
        original_marker_restored=True,
        total_weights_bytes=sum(r['weights_bytes'] for r in records),
        interpretation='All epoch weights remain usable after ordinary source-path cleanup. No training or scientific evaluation is implied by protection completion.')
    immutable(destination/'complete.json',result)
    return result


def verify_complete(destination):
    result = read(destination/'complete.json')
    guard = read(destination/'guard.json')
    if result['guard_sha256'] != sha256(destination/'guard.json') or not result['retention_hold_released']:
        raise ValueError('Trajectory protection provenance changed')
    if result['epochs'] != verify_epochs(destination,guard['horizon']):
        raise ValueError('Protected trajectory differs from its completion')
    if sha256(Path(guard['run'])/'controlled_run.json') != guard['original_marker_sha256']:
        raise ValueError('Original managed marker is not restored')
    return result


def watch():
    import fcntl
    if ROOT != Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Run protection CPU-only in the authorized Atum project')
    directory = WORK/'controlled'
    metadata = read(directory/'jobs'/f'{REFERENCE}.json')
    config_path = Path(metadata['config_path'])
    if (metadata['id'] != REFERENCE or metadata['regime'] != 'A' or metadata['arm'] != 'full' or metadata['seed'] != 42 or
        metadata['learning_rate_multiplier'] != 1. or metadata['plan_sha256'] != sha256(directory/'plan.json') or
        metadata['config_sha256'] != sha256(config_path)):
        raise ValueError('Expected the registered matched A reference')
    run = Path(metadata['output_dir']).resolve()
    launch = read(run/'launch_provenance.json')
    horizon = launch['config']['training']['num_train_epochs']
    if horizon != 15 or launch['config_sha256'] != metadata['config_sha256']:
        raise ValueError('Reference launch configuration differs')
    destination = (directory/'protected_trajectories'/REFERENCE).resolve()
    with (run/'trajectory_protection.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        hold_retention(run,destination,metadata['config_sha256'],horizon)
        if (destination/'complete.json').exists():
            return verify_complete(destination)
        seen = set()
        while True:
            for checkpoint in sorted(run.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[1])):
                record = capture_checkpoint(run,destination,checkpoint,horizon)
                if record is not None and record['epoch'] not in seen:
                    print(json.dumps(dict(protected_epoch=record['epoch'],step=record['step'],weights_bytes=record['weights_bytes'])),flush=True)
                    seen.add(record['epoch'])
            if len(seen) == horizon:
                result = release_retention(destination)
                print(json.dumps(dict(phase='trajectory_protected',epochs=len(result['epochs']),retention_hold_released=True)),flush=True)
                return result
            write_json(destination/'status.json',dict(phase='protecting_completed_epochs',pid=os.getpid(),
                epochs=sorted(seen),horizon=horizon,retention_hold_active=True))
            time.sleep(5)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    watch()
