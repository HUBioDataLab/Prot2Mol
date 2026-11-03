"""Fixed-seed E56/E57 execution on the frozen training/cold-validation pairs.

Use the two full-reference best-macro checkpoints and their nominal rates.
Two repeated measurements of the first pair qualify runtime precision; they
never choose a batch, rate or model. Scientific results retain all32 pairs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import time

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from . import requested_probe_protocol as protocol, requested_evaluation as evaluation
    from . import objective_probe_execution as execution
else:
    from run import ROOT, WORK, sha256, write_json
    import requested_probe_protocol as protocol, requested_evaluation as evaluation
    import objective_probe_execution as execution

read, immutable = evaluation.read, evaluation.immutable
reconstruction = execution.protocol.reconstruction
ROLES = {'train': 'train', 'cold_validation': 'val'}


def destination():
    return WORK / 'tier4/requested_objectives'


def software():
    result = evaluation.software()
    result['source_sha256'].update(execution.worker_sources())
    for path in (Path(__file__).absolute(), Path(protocol.__file__).absolute()):
        result['source_sha256'][str(path.relative_to(ROOT))] = sha256(path)
    return result


def inputs(probe):
    from reward_model.training.config import load_reward_training_config
    original = probe['probes']['objective']
    primary_path = WORK / 'controlled/plan.json'; primary = read(primary_path)
    directory, frozen = reconstruction.inputs()
    verified_path = directory / 'reconstruction_verification.json'; verified = read(verified_path)
    if (original['seeds'] != [42] or len(original['checkpoint_slots']) != 2 or
        original['scopes'] != list(execution.protocol.SCOPES) or
        original['precision'] != 'float32' or original['autocast'] or original['allow_tf32'] or original['dropout'] or
        verified['complete_sha256'] != sha256(directory / 'complete.json') or verified['batches_verified'] != 128):
        raise ValueError('Frozen objective scope, precision or reconstruction changed')
    files = dict(original['input_sha256'])
    files.update({str(primary_path): sha256(primary_path), str(verified_path): sha256(verified_path)})
    models = {}
    for regime in ('A', 'B'):
        identity = f'full__{regime}__seed42__lr1p0'
        path = evaluation.destination(identity) / 'plan.json'; base = read(path)
        if (base != evaluation.build_plan(identity) or
            {k: base[k] for k in ('job', 'checkpoint', 'rules')} != original['resolved_bases'][regime] or
            (base['job']['arm'], base['job']['regime'], base['job']['seed'], base['job']['learning_rate_multiplier']) != ('full', regime, 42, 1.) or
            set(base['rules']) != {'best_macro'}):
            raise ValueError('Objective probes require the fixed full seed42/base-LR best-macro models')
        slots = [s for s in original['checkpoint_slots'] if s['regime'] == regime]
        if (len(slots) != 1 or slots[0]['selected_job_id'] != identity or slots[0]['seed'] != 42 or
            slots[0]['candidate_job_ids'] != [identity] or slots[0]['selected_checkpoint'] != base['checkpoint'] or
            slots[0]['batch_pairs'] != 32):
            raise ValueError('Only the frozen32 pairs and single fixed reference are permitted')
        config = load_reward_training_config(base['job']['config_path'])
        rates = execution.protocol.step_rates(asdict(config.training))
        if rates != original['candidates'][identity]['nominal_step_rates']:
            raise ValueError('Nominal component rates differ from the frozen protocol')
        files[str(path)] = sha256(path)
        files[base['job']['config_path']] = base['job']['config_sha256']
        files[str(Path(base['checkpoint']['path']) / 'pytorch_model.bin')] = base['checkpoint']['weights_sha256']
        files[str(Path(base['checkpoint']['path']) / 'config.json')] = base['checkpoint']['model_config_sha256']
        contexts = {}
        for role, split in ROLES.items():
            context_path = directory / f'{regime}_{role}.json'
            context = read(context_path)
            if (context['selected_batches'] != 32 or len(context['batches']) != 32 or
                [v['pair_index'] for v in context['batches']] != list(range(32)) or
                frozen['regimes'][regime]['batch_pairs'] != 32 or
                context['regime'] != regime or context['role'] != role):
                raise ValueError('Frozen diagnostic pair inventory changed')
            contexts[role] = dict(path=str(context_path), sha256=sha256(context_path),
                data=dict(source=primary['bases'][regime]['data'][split + '_parquet_path'],
                    source_sha256=primary['source_data_sha256'][regime][split],
                    cache=str(Path(config.data.tokenized_dataset_dir) / (split + '_examples')),
                    cache_sha256=primary['tokenized_data_sha256'][regime][split]))
            files[str(context_path)] = sha256(context_path)
        models[regime] = dict(job=base['job'], checkpoint=base['checkpoint'], rules=base['rules'],
            contexts=contexts, nominal_rates=rates, reconstruction=str(verified_path),
            collator_provenance=verified['collators'][regime])
    return models, files


def build(analysis_paths):
    gate = protocol.require_tier3(analysis_paths); probe = protocol.verify()
    models, files = inputs(probe)
    files.update(gate['analysis_sha256'])
    for path in (protocol.destination() / 'plan.json', protocol.destination() / 'complete.json',
        evaluation.requested.destination() / 'complete.json'):
        files[str(path)] = sha256(path)
    for name, digest in files.items():
        if not Path(name).is_relative_to(ROOT) or sha256(Path(name)) != digest:
            raise ValueError('Objective input changed or lies outside the project')
    return dict(format_version=1, boot_id=evaluation.control_queue.boot_id(), experiments=['E56', 'E57'],
        seed=42, learning_rate_multiplier=1., models=models, tier3_gate=gate,
        protocol_sha256=sha256(protocol.destination() / 'plan.json'), input_sha256=files,
        software=software(), gpu_uuids=list(evaluation.GPU_UUIDS), workers=2, batch_pairs_per_model=32,
        qualification_pair_index=0, qualification_repetitions=2, scopes=list(execution.protocol.SCOPES),
        precision='float32', autocast=False, allow_tf32=False, new_fits=0, test_data_read=False,
        interpretation='Conditional local gradient/transference diagnostics on64 frozen own-regime train/cold-validation pairs. Nominal weighted-SGD steps do not fit a model. Preserve undefined states and precision limits; no independent-batch interval, seed spread or causal generalization claim.')


def prepare(analysis_paths):
    evaluation.control_queue.runtime_guard(); plan = build(analysis_paths)
    if ROOT == Path('/home/atabey/Prot2Mol'):
        if __package__: from .prepare_random import archive_directory
        else: from prepare_random import archive_directory
        archive_directory(destination())
    path = destination() / 'plan.json'; immutable(path, plan)
    immutable(path.parent / 'preparation.json', dict(plan_sha256=sha256(path), requested_tier3_verified=True,
        scientific_objective_probes_started=False))
    return path


def verify_plan(path):
    path = Path(path); plan = read(path); ready = read(path.parent / 'preparation.json')
    if (path != destination() / 'plan.json' or ready['plan_sha256'] != sha256(path) or
        not ready['requested_tier3_verified'] or plan['boot_id'] != evaluation.control_queue.boot_id() or
        plan['experiments'] != ['E56', 'E57'] or plan['seed'] != 42 or plan['learning_rate_multiplier'] != 1. or
        set(plan['models']) != {'A', 'B'} or plan['workers'] != 2 or plan['batch_pairs_per_model'] != 32 or
        plan['qualification_pair_index'] != 0 or plan['qualification_repetitions'] != 2 or
        plan['scopes'] != list(execution.protocol.SCOPES) or plan['precision'] != 'float32' or
        plan['autocast'] or plan['allow_tf32'] or plan['new_fits'] != 0 or plan['test_data_read'] or
        plan['gpu_uuids'] != list(evaluation.GPU_UUIDS) or plan['software'] != software() or
        plan['tier3_gate']['requested_fits'] != 26 or len(plan['tier3_gate']['analysis_sha256']) != 26):
        raise ValueError('Objective preparation, scope, software or Tier3 gate changed')
    for name, digest in plan['input_sha256'].items():
        if not Path(name).is_relative_to(ROOT) or sha256(Path(name)) != digest:
            raise ValueError('Prepared objective input changed: ' + name)
    if plan['protocol_sha256'] != sha256(protocol.destination() / 'plan.json'):
        raise ValueError('Fixed probe protocol changed')
    models, _ = inputs(read(protocol.destination() / 'plan.json'))
    if models != plan['models']: raise ValueError('Frozen objective inputs changed')
    return plan


def repeatability(first, second):
    """Record FP32 drift without reselecting steps or hiding sign changes."""
    import torch
    differences = []
    def compare(a, b, path):
        if type(a) is not type(b): raise ValueError('Repeated diagnostic structure changed: ' + path)
        if isinstance(a, dict):
            if a.keys() != b.keys(): raise ValueError('Repeated diagnostic fields changed: ' + path)
            for key in a: compare(a[key], b[key], path + '/' + str(key))
        elif isinstance(a, list):
            if len(a) != len(b): raise ValueError('Repeated diagnostic inventory changed: ' + path)
            for i, (x, y) in enumerate(zip(a, b)): compare(x, y, path + '/' + str(i))
        elif isinstance(a, float):
            if not math.isfinite(a) or not math.isfinite(b): raise ValueError('Nonfinite qualification measurement')
            differences.append((path, abs(a-b), abs(a-b)/max(1., abs(a), abs(b))))
        elif a != b: raise ValueError('Repeated diagnostic metadata or effective-step state changed: ' + path)
    compare(first, second, '')
    tolerance = 32 * torch.finfo(torch.float32).eps
    sign_changes, step_precision = [], {}
    for source in execution.probes.OBJECTIVES:
        step_precision[source] = {}
        for scope in execution.protocol.SCOPES:
            a, b = (r['lookahead'][source]['scopes'][scope] for r in (first, second))
            step_precision[source][scope] = dict(planned_step_norm=a['planned_step_norm'],
                actual_step_norm=a['actual_step_norm'], state=a['state'], targets={})
            for target in execution.probes.OBJECTIVES:
                x, y = a['transfer'][target], b['transfer'][target]
                drift = abs(x['before']-y['before']) + abs(x['after']-y['after'])
                change, repeated = x['absolute_loss_change'], y['absolute_loss_change']
                if (change > 0)-(change < 0) != (repeated > 0)-(repeated < 0):
                    sign_changes.append('/'.join((source, scope, target)))
                step_precision[source][scope]['targets'][target] = dict(
                    repeated_loss_drift=drift, absolute_loss_change=change,
                    change_within_observed_repeat_drift=abs(change) <= drift)
    maximum = max((v[2] for v in differences), default=0.)
    return dict(float_values_compared=len(differences), exact=all(v[1] == 0 for v in differences),
        maximum_absolute_difference=max((v[1] for v in differences), default=0.),
        maximum_scaled_difference=maximum, tolerance=tolerance, passed=maximum <= tolerance,
        transfer_sign_changes=sign_changes, first_pair_step_precision=step_precision,
        interpretation='Numerical runtime check on the first frozen pair only. Scaled drift uses max(1,abs(values));32 FP32 eps is fixed before measurement. Small transfer signs can remain precision-limited even when runtime qualifies. This is not an error bound for other pairs.')


def qualify(model, examples, contexts, collator, training, evidence, output, source, *, cpu=False):
    import torch
    from reward_model.training.trainer import RewardModelTrainer
    output = Path(output)
    if output.exists(): raise RuntimeError('Existing objective qualification requires inspection')
    device = next(model.parameters()).device
    if device.type != ('cpu' if cpu else 'cuda') or any(
        t.is_floating_point() and t.dtype != torch.float32 for t in model.state_dict().values()):
        raise ValueError('Objective qualification requires FP32 parameters on the assigned device')
    before = execution.parameter_digest(model)
    if source['parameter_sha256'] != before: raise ValueError('Qualification checkpoint identity changed')
    rates = execution.protocol.validate_parameter_rates(dict(model.named_parameters()), training)
    batches = {role: execution.checked_batch(contexts[role]['batches'][0], examples[role], collator,
        evidence[role]['batches'][0]) for role in ROLES}
    batches = {role: RewardModelTrainer._build_model_inputs(None,
        {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()})
        for role, batch in batches.items()}
    dtype_checks = []
    def inspect(_module, _args, result):
        if torch.is_autocast_enabled() or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise ValueError('Qualification forward did not use the frozen precision policy')
        dtypes = {key: str(getattr(result, field).dtype) for key, (field, _) in execution.probes.OBJECTIVES.items()}
        if set(dtypes.values()) != {'torch.float32'}: raise ValueError('Native objective forward is not FP32')
        dtype_checks.append(dtypes)
    handle = model.register_forward_hook(inspect)
    measurements = []
    if not cpu: torch.cuda.reset_peak_memory_stats(device)
    try:
        with execution.full_precision():
            for index in range(2):
                measured = execution.probes.measure(model, batches['train'], batches['cold_validation'],
                    rates, precision='float32', scopes=execution.protocol.SCOPES)
                immutable(output / f'repeat_{index}.json', measured); measurements.append(measured)
    finally:
        handle.remove()
        if execution.parameter_digest(model) != before:
            raise RuntimeError('Objective qualification changed model parameters')
    comparison = repeatability(*measurements)
    memory = None
    if not cpu:
        torch.cuda.synchronize(device)
        memory = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            total_device_bytes=torch.cuda.get_device_properties(device).total_memory)
    qualified = dict(source=source, cpu_qualification=cpu, parameter_sha256=before,
        first_pair_context_sha256={r: execution.digest_json(contexts[r]['batches'][0]) for r in ROLES},
        repetitions=2, repeatability=comparison, forward_dtype_checks=dtype_checks,
        memory=memory, parameters_unchanged=True, optimizer_created=False,
        production_gpu_qualified=not cpu and comparison['passed'] and bool(dtype_checks),
        artifacts_sha256={f'repeat_{i}.json': sha256(output / f'repeat_{i}.json') for i in range(2)})
    immutable(output / 'complete.json', qualified)
    if not comparison['passed'] or not dtype_checks:
        raise RuntimeError('Objective precision qualification failed; preserve evidence and inspect without tuning steps')
    return qualified


def context_inputs(model_info, *, tokens):
    contexts = {r: read(item['path']) for r, item in model_info['contexts'].items()}
    record = read(model_info['reconstruction'])
    evidence = {r: record['contexts'][model_info['job']['regime'] + '_' + r] for r in ROLES}
    if not tokens: return contexts, evidence
    from datasets import load_from_disk
    examples = {}
    for role in ROLES:
        data = model_info['contexts'][role]['data']; evaluation.postselection.verify_data(data)
        examples[role] = load_from_disk(data['cache'])
    return contexts, evidence, examples


def verify_worker(path, regime, *, cpu=False):
    path = Path(path); plan = read(path); output = path.parent / regime
    done = read(output / 'complete.json'); info = plan['models'][regime]
    if (done['plan_sha256'] != sha256(path) or done['regime'] != regime or done['cpu_qualification'] != cpu or
        done['new_fits'] != 0 or done['test_data_read'] or
        (done['gpu_uuid'] is not None if cpu else done['gpu_uuid'] not in evaluation.GPU_UUIDS)):
        raise ValueError('Objective completion scope or device changed')
    source = dict(plan_sha256=sha256(path), regime=regime, checkpoint=info['checkpoint'],
        parameter_sha256=done['parameter_sha256'], gpu_uuid=done['gpu_uuid'])
    qualified = read(output / 'qualification/complete.json')
    if (sha256(output / 'qualification/complete.json') != done['qualification_sha256'] or
        qualified['source'] != source or qualified['cpu_qualification'] != cpu or
        qualified['production_gpu_qualified'] != (not cpu) or not qualified['parameters_unchanged'] or
        qualified['optimizer_created'] or set(qualified['artifacts_sha256']) != {'repeat_0.json', 'repeat_1.json'}):
        raise ValueError('Objective runtime qualification changed')
    for name, digest in qualified['artifacts_sha256'].items():
        if sha256(output / 'qualification' / name) != digest: raise ValueError('Repeated measurement changed')
    comparison = repeatability(*[read(output / 'qualification' / f'repeat_{i}.json') for i in range(2)])
    if comparison != qualified['repeatability'] or not comparison['passed']:
        raise ValueError('Repeated runtime precision comparison changed')
    contexts, evidence = context_inputs(info, tokens=False)
    result = execution.verify_complete(output / 'pairs', source, contexts, evidence, cpu=cpu)
    if sha256(output / 'pairs/complete.json') != done['pairs_sha256']:
        raise ValueError('Saved objective measurements changed')
    return done, result, qualified


def worker(path, regime, gpu_index, *, cpu=False):
    if regime not in ('A', 'B') or type(gpu_index) is not int or not 0 <= gpu_index < 4:
        raise ValueError('Expected an A/B reference and one authorized GPU')
    gpu = None if cpu else evaluation.GPU_UUIDS[gpu_index]
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != ('' if cpu else gpu) or
        os.environ.get('ACCELERATE_USE_CPU', '').lower() != ('true' if cpu else 'false') or
        (not cpu and ROOT != Path('/home/atabey/Prot2Mol'))):
        raise RuntimeError('Use one authorized Atum GPU or explicit CPU qualification')
    path = Path(path); plan = verify_plan(path); output = path.parent / regime
    if (output / 'complete.json').exists(): verify_worker(path, regime, cpu=cpu); return
    if output.exists(): raise RuntimeError('Partial objective output requires inspection; no automatic retry')
    import torch
    from reward_model.model import load_reward_model
    from reward_model.training.config import load_reward_training_config
    torch.set_num_threads(2)
    info = plan['models'][regime]; config = load_reward_training_config(info['job']['config_path'])
    contexts, evidence, examples = context_inputs(info, tokens=True)
    collator, provenance = reconstruction.reference_collator(config)
    if provenance != info['collator_provenance']: raise ValueError('Native collator tokenizers or settings changed')
    model = load_reward_model(info['checkpoint']['path'], device=torch.device('cpu'))
    evaluation.postselection.verify_model_config(model.config.to_dict(), config.model,
        encoder_widths={s: getattr(model, s + '_encoder').config.hidden_size for s in ('protein', 'molecule')})
    if json.loads(json.dumps(model.config.to_dict())) != read(Path(info['checkpoint']['path']) / 'config.json'):
        raise ValueError('Loaded reference configuration changed')
    model.to('cpu' if cpu else 'cuda'); before = execution.parameter_digest(model)
    training = asdict(config.training)
    if execution.protocol.validate_parameter_rates(dict(model.named_parameters()), training) != info['nominal_rates']:
        raise ValueError('Loaded reference parameter rates changed')
    source = dict(plan_sha256=sha256(path), regime=regime, checkpoint=info['checkpoint'],
        parameter_sha256=before, gpu_uuid=gpu)
    output.mkdir(parents=True)
    immutable(output / 'invocation.json', dict(source=source, cpu_qualification=cpu, started_unix=time.time()))
    qualify(model, examples, contexts, collator, training, evidence, output / 'qualification', source, cpu=cpu)
    execution.execute_pairs(model, examples, contexts, collator, training, evidence, output / 'pairs', source, cpu=cpu)
    if execution.parameter_digest(model) != before or verify_plan(path) != plan:
        raise RuntimeError('Objective execution changed model or inputs')
    immutable(output / 'complete.json', dict(plan_sha256=sha256(path), regime=regime,
        parameter_sha256=before, gpu_uuid=gpu, cpu_qualification=cpu, new_fits=0, test_data_read=False,
        qualification_sha256=sha256(output / 'qualification/complete.json'),
        pairs_sha256=sha256(output / 'pairs/complete.json'), finished_unix=time.time()))
    verify_worker(path, regime, cpu=cpu)


def run(path):
    import fcntl
    evaluation.control_queue.runtime_guard(); path = Path(path); plan = verify_plan(path)
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
                raise RuntimeError('Partial objective worker requires inspection')
            if (output / 'complete.json').exists(): verify_worker(path, regime)
        children, handles = [], []
        try:
            for index, regime in enumerate(('A', 'B')):
                if (path.parent / regime / 'complete.json').exists(): continue
                env = evaluation.postselection.launch_environment(plan['models'][regime]['job']['id'])
                env.update(CUDA_VISIBLE_DEVICES=evaluation.GPU_UUIDS[index], WANDB_MODE='disabled')
                handle = (path.parent / f'worker_{regime}.log').open('a'); handles.append(handle)
                command = [str(ROOT / '.venv/bin/python'), str(Path(__file__)), 'worker',
                    '--plan', str(path), '--regime', regime, '--gpu-index', str(index)]
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                children.append(child)
                write_json(path.parent / f'status_{regime}.json', dict(phase='qualifying_and_measuring_objectives',
                    pid=child.pid, gpu_uuid=evaluation.GPU_UUIDS[index], started_unix=time.time()))
            codes = [child.wait() for child in children]
            if any(codes): raise RuntimeError('Objective worker failed; inspect without retry: ' + str(codes))
        finally:
            for child in children:
                if child.poll() is None: child.terminate(); child.wait()
            for handle in handles: handle.close()
        measured = {r: verify_worker(path, r) for r in ('A', 'B')}
        if verify_plan(path) != plan: raise ValueError('Objective plan changed during execution')
        record = dict(phase='requested_reference_objective_probes_complete', plan_sha256=sha256(path),
            evidence_sha256={r: sha256(path.parent / r / 'complete.json') for r in ('A', 'B')},
            results={r: dict(summary=v[1]['summary'], qualification=v[2]) for r, v in measured.items()},
            new_fits=0, test_data_read=False, scientific_study_complete=False, interpretation=plan['interpretation'])
        immutable(path.parent / 'complete.json', record)
        return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    actions.add_parser('prepare').add_argument('--analyses', nargs='+', type=Path, required=True)
    actions.add_parser('run').add_argument('--plan', type=Path, required=True)
    sub = actions.add_parser('worker'); sub.add_argument('--plan', type=Path, required=True)
    sub.add_argument('--regime', choices=('A', 'B'), required=True); sub.add_argument('--gpu-index', type=int, required=True)
    args = parser.parse_args()
    if args.action == 'prepare': print(prepare(args.analyses))
    elif args.action == 'run': print(json.dumps(run(args.plan), indent=2))
    else: worker(args.plan, args.regime, args.gpu_index)
