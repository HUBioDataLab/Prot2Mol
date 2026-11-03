"""E54/E55 on the two fixed full-reference models and six frozen cohorts.

Scientific execution requires all requested Tier3 analyses. Each independent
worker uses one authorized GPU, exports both projection stages, and never fits.
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
    from . import requested_probe_protocol as protocol, requested_evaluation as evaluation
    from . import geometry_export as geometry
    from .checkpoint_evaluation import parameter_digest
else:
    from run import ROOT, WORK, sha256, write_json
    import requested_probe_protocol as protocol, requested_evaluation as evaluation
    import geometry_export as geometry
    from checkpoint_evaluation import parameter_digest

read, immutable = evaluation.read, evaluation.immutable
PARTITIONS = ('train', 'val', 'test')


def destination():
    return WORK / 'tier4/requested_geometry'


def source_frame(path):
    import pandas as pd
    # Raw partitions store assay/target identity; group_id is constructed by
    # the native token preparation and the frozen geometry design.
    frame = pd.read_parquet(path, columns=['assay_id', 'target_chembl_id', 'compound_id', 'pchembl_value'])
    frame['group_id'] = frame.target_chembl_id.astype(str) + '__' + frame.assay_id.astype(str)
    return frame


def software():
    result = evaluation.software()
    for module in (protocol, geometry):
        path = Path(module.__file__).absolute()
        result['source_sha256'][str(path.relative_to(ROOT))] = sha256(path)
    for name in ('requested_geometry.py', 'representation_geometry.py'):
        path = Path(__file__).with_name(name).absolute()
        result['source_sha256'][str(path.relative_to(ROOT))] = sha256(path)
    return result


def inputs(probe):
    """Resolve only the frozen own-regime full-reference geometry inputs."""
    primary_path = WORK / 'controlled/plan.json'; primary = read(primary_path)
    files = {str(primary_path): sha256(primary_path)}
    design = probe['probes']['geometry']['design']
    records = {}
    for regime in ('A', 'B'):
        path = evaluation.destination(f'full__{regime}__seed42__lr1p0') / 'plan.json'
        base = read(path)
        if base != evaluation.build_plan(base['job']['id']):
            raise ValueError('Full-reference checkpoint evidence changed')
        resolved = probe['probes']['geometry']['resolved_bases'][regime]
        if ({k: base[k] for k in ('job', 'checkpoint', 'rules')} != resolved or
            (base['job']['arm'], base['job']['regime'], base['job']['seed'],
             base['job']['learning_rate_multiplier']) != ('full', regime, 42, 1.) or
            set(base['rules']) != {'best_macro'}):
            raise ValueError('Geometry requires the fixed full seed42/base-LR best-macro models')
        files[str(path)] = sha256(path)
        for name, digest in ((base['job']['config_path'], base['job']['config_sha256']),
            (str(Path(base['checkpoint']['path']) / 'pytorch_model.bin'), base['checkpoint']['weights_sha256']),
            (str(Path(base['checkpoint']['path']) / 'config.json'), base['checkpoint']['model_config_sha256'])):
            files[name] = digest
        partitions = {}
        for split in PARTITIONS:
            data = dict(source=primary['bases'][regime]['data'][split + '_parquet_path'],
                source_sha256=primary['source_data_sha256'][regime][split],
                cache=str(Path(primary['bases'][regime]['data']['tokenized_dataset_dir']) / (split + '_examples')),
                cache_sha256=primary['tokenized_data_sha256'][regime][split])
            if split in base['data'] and base['data'][split] != data:
                raise ValueError('Geometry partition differs from selected-model evaluation')
            membership = WORK / 'tier4/geometry_design' / f'{regime}_{split}.parquet'
            digest = design['artifacts_sha256'][membership.name]
            files[str(membership)] = digest
            partitions[split] = dict(data=data, membership=str(membership), membership_sha256=digest)
        records[regime] = dict(job=base['job'], checkpoint=base['checkpoint'], rules=base['rules'], partitions=partitions)
    return records, files


def build(analysis_paths):
    gate = protocol.require_tier3(analysis_paths)
    probe_path = protocol.destination() / 'plan.json'; probe = protocol.verify()
    models, files = inputs(probe)
    files.update(gate['analysis_sha256'])
    for path in (probe_path, probe_path.parent / 'complete.json',
        evaluation.requested.destination() / 'complete.json'):
        files[str(path)] = sha256(path)
    for name, digest in files.items():
        if not Path(name).is_relative_to(ROOT) or sha256(Path(name)) != digest:
            raise ValueError('Geometry input changed or lies outside the project')
    return dict(format_version=1, boot_id=evaluation.control_queue.boot_id(), experiments=['E54', 'E55'],
        seed=42, learning_rate_multiplier=1., models=models, tier3_gate=gate,
        protocol_sha256=sha256(probe_path), input_sha256=files, software=software(),
        gpu_uuids=list(evaluation.GPU_UUIDS), workers=2, batch_size=32,
        stages=list(geometry.STAGES), partitions=list(PARTITIONS), new_fits=0,
        interpretation='Six fixed own-regime cohorts, same two full-reference best-macro models. Before/after denotes encoder-pooled versus contrastive-projected representations. No resampling, checkpoint selection, parameter update or ranking-performance inference.')


def prepare(analysis_paths):
    evaluation.control_queue.runtime_guard()
    plan = build(analysis_paths)
    if ROOT == Path('/home/atabey/Prot2Mol'):
        if __package__: from .prepare_random import archive_directory
        else: from prepare_random import archive_directory
        archive_directory(destination())
    path = destination() / 'plan.json'; immutable(path, plan)
    immutable(path.parent / 'preparation.json', dict(plan_sha256=sha256(path),
        requested_tier3_verified=True, scientific_geometry_started=False))
    return path


def verify_plan(path):
    """Check the prepared gate and input bytes without rerunning prior statistics."""
    path = Path(path); plan = read(path)
    ready = read(path.parent / 'preparation.json')
    if (path != destination() / 'plan.json' or ready['plan_sha256'] != sha256(path) or
        not ready['requested_tier3_verified'] or plan['boot_id'] != evaluation.control_queue.boot_id() or
        plan['experiments'] != ['E54', 'E55'] or plan['seed'] != 42 or plan['learning_rate_multiplier'] != 1. or
        set(plan['models']) != {'A', 'B'} or plan['workers'] != 2 or plan['batch_size'] != 32 or
        plan['stages'] != list(geometry.STAGES) or plan['partitions'] != list(PARTITIONS) or
        plan['gpu_uuids'] != list(evaluation.GPU_UUIDS) or plan['new_fits'] != 0 or
        plan['tier3_gate']['requested_fits'] != 26 or len(plan['tier3_gate']['analysis_sha256']) != 26 or
        plan['software'] != software()):
        raise ValueError('Geometry preparation, scope, software or Tier3 gate changed')
    for name, digest in plan['input_sha256'].items():
        if not Path(name).is_relative_to(ROOT) or sha256(Path(name)) != digest:
            raise ValueError('Prepared geometry input changed: ' + name)
    if plan['protocol_sha256'] != sha256(protocol.destination() / 'plan.json'):
        raise ValueError('Fixed probe protocol changed')
    models, _ = inputs(read(protocol.destination() / 'plan.json'))
    if models != plan['models']:
        raise ValueError('Frozen geometry inputs changed')
    return plan


def verify_worker(path, regime, *, cpu=False):
    plan = read(path); output = path.parent / regime; done = read(output / 'complete.json')
    if (regime not in ('A', 'B') or done['plan_sha256'] != sha256(path) or
        done['regime'] != regime or done['cpu_qualification'] != cpu or
        set(done['evidence_sha256']) != set(PARTITIONS) or done['new_fits'] != 0 or
        (done['gpu_uuid'] is not None if cpu else done['gpu_uuid'] not in evaluation.GPU_UUIDS)):
        raise ValueError('Geometry completion changed its scope or execution mode')
    results = {}
    for split in PARTITIONS:
        source = dict(plan_sha256=sha256(path), regime=regime, partition=split,
            checkpoint=plan['models'][regime]['checkpoint'], parameter_sha256=done['parameter_sha256'],
            gpu_uuid=done['gpu_uuid'])
        directory = output / split
        if sha256(directory / 'complete.json') != done['evidence_sha256'][split]:
            raise ValueError('Saved geometry result changed')
        results[split] = geometry.verify_complete(directory, source, cpu=cpu)
        manifest = read(directory / 'export.json')
        if manifest['parameter_sha256'] != done['parameter_sha256']:
            raise ValueError('Geometry exports used different model parameters')
    return done, results


def worker(path, regime, gpu_index, *, cpu=False):
    if regime not in ('A', 'B') or type(gpu_index) is not int or not 0 <= gpu_index < 4:
        raise ValueError('Expected an A/B reference and one authorized GPU')
    gpu = None if cpu else evaluation.GPU_UUIDS[gpu_index]
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != ('' if cpu else gpu) or
        os.environ.get('ACCELERATE_USE_CPU', '').lower() != ('true' if cpu else 'false') or
        (not cpu and ROOT != Path('/home/atabey/Prot2Mol'))):
        raise RuntimeError('Use one authorized Atum GPU or explicit CPU qualification')
    path = Path(path); plan = verify_plan(path); output = path.parent / regime
    if (output / 'complete.json').exists():
        verify_worker(path, regime, cpu=cpu); return
    if output.exists():
        raise RuntimeError('Partial geometry output requires inspection; no automatic retry')
    import pandas as pd
    import torch
    from datasets import load_from_disk
    from reward_model.model import load_reward_model
    from reward_model.training.config import load_reward_training_config
    torch.set_num_threads(4)
    model_info = plan['models'][regime]
    config = load_reward_training_config(model_info['job']['config_path'])
    model = load_reward_model(model_info['checkpoint']['path'], device=torch.device('cpu'))
    evaluation.postselection.verify_model_config(model.config.to_dict(), config.model,
        encoder_widths={side: getattr(model, side + '_encoder').config.hidden_size for side in geometry.SIDES})
    if json.loads(json.dumps(model.config.to_dict())) != read(Path(model_info['checkpoint']['path']) / 'config.json'):
        raise ValueError('Loaded reference configuration changed')
    model.to('cpu' if cpu else 'cuda'); before = parameter_digest(model)
    output.mkdir(parents=True)
    immutable(output / 'invocation.json', dict(plan_sha256=sha256(path), regime=regime,
        gpu_uuid=gpu, cpu_qualification=cpu, started_unix=time.time()))
    evidence = {}
    for split in PARTITIONS:
        item = model_info['partitions'][split]; evaluation.postselection.verify_data(item['data'])
        if sha256(Path(item['membership'])) != item['membership_sha256']:
            raise ValueError('Geometry membership changed before export')
        source = dict(plan_sha256=sha256(path), regime=regime, partition=split,
            checkpoint=model_info['checkpoint'], parameter_sha256=before, gpu_uuid=gpu)
        geometry.export_geometry(model, load_from_disk(item['data']['cache']),
            source_frame(item['data']['source']),
            pd.read_parquet(item['membership']), output / split, source, cpu=cpu,
            batch_size=plan['batch_size'])
        evidence[split] = sha256(output / split / 'complete.json')
    if parameter_digest(model) != before or verify_plan(path) != plan:
        raise RuntimeError('Geometry execution changed the model or its inputs')
    immutable(output / 'complete.json', dict(plan_sha256=sha256(path), regime=regime,
        parameter_sha256=before, gpu_uuid=gpu, cpu_qualification=cpu,
        evidence_sha256=evidence, new_fits=0, finished_unix=time.time()))
    verify_worker(path, regime, cpu=cpu)


def run(path):
    import fcntl
    evaluation.control_queue.runtime_guard()
    path = Path(path); plan = verify_plan(path)
    with (WORK / 'controlled/driver.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return dict(phase='waiting_for_global_driver_lock')
        holders = evaluation.control_queue.gpu_processes()
        if holders: return dict(phase='waiting_for_authorized_gpus', live_processes=holders)
        storage = evaluation.control_queue.storage_gate()
        if not storage['ready']: return dict(phase='waiting_for_storage', storage=storage)
        for regime in ('A', 'B'):
            output = path.parent / regime
            if output.exists() and not (output / 'complete.json').exists():
                raise RuntimeError('Partial geometry worker requires inspection')
            if (output / 'complete.json').exists(): verify_worker(path, regime)
        children, handles = [], []
        try:
            for index, regime in enumerate(('A', 'B')):
                output = path.parent / regime
                if (output / 'complete.json').exists():
                    verify_worker(path, regime); continue
                if output.exists(): raise RuntimeError('Partial geometry worker requires inspection')
                env = evaluation.postselection.launch_environment(plan['models'][regime]['job']['id'])
                env.update(CUDA_VISIBLE_DEVICES=evaluation.GPU_UUIDS[index], WANDB_MODE='disabled')
                handle = (path.parent / f'worker_{regime}.log').open('a'); handles.append(handle)
                command = [str(ROOT / '.venv/bin/python'), str(Path(__file__)), 'worker',
                    '--plan', str(path), '--regime', regime, '--gpu-index', str(index)]
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                children.append(child)
                write_json(path.parent / f'status_{regime}.json', dict(phase='exporting_frozen_geometry',
                    pid=child.pid, gpu_uuid=evaluation.GPU_UUIDS[index], started_unix=time.time()))
            codes = [child.wait() for child in children]
            if any(codes): raise RuntimeError('Geometry worker failed; inspect without retry: ' + str(codes))
        finally:
            for child in children:
                if child.poll() is None: child.terminate(); child.wait()
            for handle in handles: handle.close()
        measured = {regime: verify_worker(path, regime)[1] for regime in ('A', 'B')}
        if verify_plan(path) != plan: raise ValueError('Geometry plan changed during execution')
        record = dict(phase='requested_reference_geometry_complete', plan_sha256=sha256(path),
            evidence_sha256={r: sha256(path.parent / r / 'complete.json') for r in ('A', 'B')},
            results={r: {s: value['analysis'] for s, value in values.items()} for r, values in measured.items()},
            new_fits=0, scientific_study_complete=False,
            interpretation=plan['interpretation'])
        immutable(path.parent / 'complete.json', record)
        return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    actions.add_parser('prepare').add_argument('--analyses', nargs='+', type=Path, required=True)
    actions.add_parser('run').add_argument('--plan', type=Path, required=True)
    sub = actions.add_parser('worker'); sub.add_argument('--plan', type=Path, required=True)
    sub.add_argument('--regime', choices=('A', 'B'), required=True)
    sub.add_argument('--gpu-index', type=int, required=True)
    args = parser.parse_args()
    if args.action == 'prepare': print(prepare(args.analyses))
    elif args.action == 'run': print(json.dumps(run(args.plan), indent=2))
    else: worker(args.plan, args.regime, args.gpu_index)
