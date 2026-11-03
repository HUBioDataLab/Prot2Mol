"""Fixed-seed E49-E52 execution contract and staged task workers.

The coordinator must hold the shared GPU lock. Preparation requires all26
requested Tier3 analyses; remaining tasks require frozen B42 validation-only
epoch selection. This module never changes supports, seeds or learning rates.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import time

from .run import ROOT, WORK, sha256, write_json
from . import requested_probe_protocol as protocol, requested_evaluation as evaluation
from . import requested_adaptation_tasks as tasks

read, immutable = evaluation.read, evaluation.immutable
HORIZON = list(range(1, 16))


def destination():
    return WORK / 'tier4/requested_adaptation'


def software():
    value = evaluation.software()
    for name in ('requested_adaptation_execution.py', 'requested_adaptation_tasks.py',
                 'requested_probe_protocol.py', 'adaptation_fit.py', 'adaptation_query.py',
                 'adaptation_protocol.py', 'adaptation_views.py', 'adaptation_design.py'):
        path = Path(__file__).with_name(name)
        value['source_sha256'][str(path.relative_to(ROOT))] = sha256(path)
    return value


def inputs(probe):
    adaptation = probe['probes']['adaptation']; records, models = {}, {}
    files = dict(probe['inputs_sha256'])
    for regime in ('A', 'B'):
        identity = f'full__{regime}__seed42__lr1p0'
        path = evaluation.destination(identity) / 'plan.json'; base = read(path)
        reference = {key: base[key] for key in ('job', 'checkpoint', 'rules')}
        if (base != evaluation.build_plan(identity) or reference != adaptation['resolved_bases'][regime]
                or set(base['rules']) != {'best_macro'}):
            raise ValueError('Adaptation must use the fixed full A/B42 best-macro references')
        artifact = adaptation['view_manifests'][regime]
        record = read(artifact['path'])
        if record['regime'] != regime or sha256(Path(artifact['path'])) != artifact['sha256']:
            raise ValueError('Adaptation view identity changed')
        files.update({str(path): sha256(path), artifact['path']: artifact['sha256'],
            base['job']['config_path']: base['job']['config_sha256'],
            str(Path(base['checkpoint']['path']) / 'pytorch_model.bin'): base['checkpoint']['weights_sha256'],
            str(Path(base['checkpoint']['path']) / 'config.json'): base['checkpoint']['model_config_sha256']})
        models[regime] = dict(reference, views=artifact)
        records[regime] = record
    return models, records, files


def build(analysis_paths):
    gate = protocol.require_tier3(analysis_paths); probe = protocol.verify()
    models, records, files = inputs(probe)
    files.update(gate['analysis_sha256'])
    for path in (protocol.destination() / 'plan.json', protocol.destination() / 'complete.json',
                 evaluation.requested.destination() / 'complete.json'):
        files[str(path)] = sha256(path)
    for name, digest in files.items():
        if not Path(name).is_relative_to(ROOT) or sha256(Path(name)) != digest:
            raise ValueError('Adaptation input changed or lies outside the project')
    return dict(format_version=1, seed=42, experiments=['E49', 'E50', 'E51', 'E52'],
        learning_rate_multiplier=1., boot_id=evaluation.control_queue.boot_id(),
        tier3_gate=gate, models=models, adaptation=probe['probes']['adaptation'],
        inventory=tasks.inventory(probe['probes']['adaptation'], records),
        snapshot_budget=probe['adaptation_snapshot_budget'],
        input_sha256=files, software=software(), gpu_uuids=list(evaluation.GPU_UUIDS),
        workers=4, horizon=HORIZON, additional_seed_fits=0, additional_learning_rates=0,
        qualification_policy='First existing B across_k1 fit in each scope, all15 validation queries; reuse these fits. No additional support draw, fit or epoch candidate.',
        selection_policy='All frozen B42 validation assays at every epoch, native macro Spearman and earliest tie. No A/test data or k0 candidate. Unselectable cells have no fallback.',
        retention_policy='All15 snapshots for B selection fits; selected+final for remaining fits, always full15-epoch training.',
        interpretation='Separate model reset for every within-unit support. Single seed; no seed-spread inference. GPU scheduling and aggregate scientific reporting are separate from this execution contract.')


def prepare(analysis_paths):
    evaluation.control_queue.runtime_guard(); plan = build(analysis_paths)
    if ROOT == Path('/home/atabey/Prot2Mol'):
        from .prepare_random import archive_directory
        archive_directory(destination())
    path = destination() / 'plan.json'; immutable(path, plan)
    immutable(path.parent / 'preparation.json', dict(plan_sha256=sha256(path),
        requested_tier3_verified=True, scientific_adaptation_started=False))
    return path


def verify_plan(path):
    path = Path(path); plan = read(path); ready = read(path.parent / 'preparation.json')
    if (path != destination() / 'plan.json' or ready['plan_sha256'] != sha256(path)
            or not ready['requested_tier3_verified'] or plan['boot_id'] != evaluation.control_queue.boot_id()
            or plan['seed'] != 42 or plan['learning_rate_multiplier'] != 1.
            or plan['experiments'] != ['E49', 'E50', 'E51', 'E52'] or plan['horizon'] != HORIZON
            or plan['gpu_uuids'] != list(evaluation.GPU_UUIDS) or plan['workers'] != 4
            or plan['additional_seed_fits'] or plan['additional_learning_rates']
            or plan['tier3_gate']['requested_fits'] != 26 or len(plan['tier3_gate']['analysis_sha256']) != 26
            or plan['software'] != software()):
        raise ValueError('Adaptation scope, software, preparation or Tier3 gate changed')
    for name, digest in plan['input_sha256'].items():
        if not Path(name).is_relative_to(ROOT) or sha256(Path(name)) != digest:
            raise ValueError('Prepared adaptation input changed: ' + name)
    probe = read(protocol.destination() / 'plan.json')
    models, records, _ = inputs(probe)
    if (plan['models'] != models or plan['adaptation'] != probe['probes']['adaptation']
            or plan['snapshot_budget'] != probe['adaptation_snapshot_budget']
            or plan['inventory'] != tasks.inventory(plan['adaptation'], records)):
        raise ValueError('Frozen adaptation references or tasks changed')
    return plan


def operations(inventory, stage, selections=None):
    """One independent support fit per operation; no post-selection refits."""
    if stage not in ('qualification', 'selection', 'remaining'):
        raise ValueError('Unknown adaptation stage')
    fits = inventory['fits']; result, unavailable = [], []
    if stage == 'remaining' and selections is None:
        raise ValueError('Remaining tasks require frozen validation selections')
    for job in fits:
        if stage in ('qualification', 'selection'):
            if not job['selection_fit']: continue
            if stage == 'qualification' and (job['kind'], job['k']) != ('across', 1): continue
            fit, save, queries = True, HORIZON, [dict(role='validation', epoch=e) for e in HORIZON]
            prefix = 'selection_fit'
        else:
            selected = selections[job['cell_id']]
            if selected['cell_id'] != job['cell_id']:
                raise ValueError('Selection belongs to a different adaptation cell')
            if selected['status'] == 'not_selectable':
                if selected['selected'] is not None: raise ValueError('Unselectable cell has a selected epoch')
                unavailable.append(dict(job_id=job['id'], reason='No defined B42 validation epoch'))
                continue
            if selected['status'] != 'selected':
                raise ValueError('Invalid frozen adaptation selection')
            epoch = selected['selected']['epoch']
            if type(epoch) is not int or epoch not in HORIZON: raise ValueError('Invalid selected epoch')
            roles = [r for r in job['query_roles'] if not (job['selection_fit'] and r == 'validation')]
            if not roles: continue
            fit = not job['selection_fit']; save = sorted({epoch, HORIZON[-1]}) if fit else HORIZON
            queries = [dict(role=r, epoch=epoch) for r in roles]
            prefix = 'remaining_fit' if fit else 'reused_fit'
        result.append(dict(id=prefix+'__'+job['id'], type='adapted', regime=job['regime'],
            job_id=job['id'], fit=fit, save_epochs=save, queries=queries))
    if stage == 'qualification':
        matched = [j for j in fits if any(o['job_id'] == j['id'] for o in result)]
        if (len(matched) != 2 or {j['scope'] for j in matched} != set(tasks.protocol.SCOPES)
                or any(j['regime'] != 'B' or j['task_id'] != 'across_k1' for j in matched)):
            raise ValueError('Qualification must reuse exactly the two B across-k1 support fits')
    else:
        for baseline in inventory['baselines']:
            if baseline['selection_phase'] != (stage == 'selection'): continue
            result.append(dict(id='baseline__'+baseline['id'], type='baseline',
                regime=baseline['regime'], baseline_id=baseline['id'], fit=False,
                save_epochs=[], queries=[dict(role=baseline['role'], epoch=0)]))
    if len({o['id'] for o in result}) != len(result): raise ValueError('Duplicate stage operation')
    return dict(operations=sorted(result, key=lambda o: (o['regime'], o['id'])), unavailable=unavailable)


def runtime_source(path, regime):
    plan = read(path)
    return dict(plan_sha256=sha256(Path(path)), regime=regime, seed=42,
        reference=plan['models'][regime], software=plan['software'])


def artifact(path, root):
    path = Path(path)
    if not path.is_relative_to(root): raise ValueError('Task evidence escapes the execution directory')
    return dict(path=str(path), sha256=sha256(path))


def pin_query(runtime, task_id, query, job):
    scope = None if job is None else job['scope']
    output = runtime.query_directory(task_id, query['role'], query['epoch'], scope)
    result = read(output / 'complete.json'); manifest = Path(result['prediction_manifest'])
    payload = manifest.parent / read(manifest)['predictions_file']
    paths = [output / name for name in ('complete.json', 'execution.json', 'native/complete.json')]
    paths += [manifest, payload]
    return dict(task_id=task_id, scope=scope, **query,
        artifacts=[artifact(p, runtime.output) for p in paths])


def operation_directory(output, operation):
    identity = operation['id']
    if Path(identity).name != identity: raise ValueError('Invalid operation identifier')
    return Path(output) / 'operations' / identity


def execute_operation(runtime, operation, inventory, *, gpu_uuid=None):
    """Caller verifies the stage and device gates. Completed operations never refit."""
    if runtime.views.record['regime'] != operation['regime']:
        raise ValueError('Worker loaded a different regime')
    output = operation_directory(runtime.output, operation)
    if (output / 'complete.json').exists():
        return verify_operation(runtime.output, operation, runtime.source, cpu=runtime.cpu)
    if output.exists(): raise RuntimeError('Partial adaptation operation requires inspection')
    immutable(output / 'invocation.json', dict(operation=operation, source=runtime.source,
        gpu_uuid=gpu_uuid, cpu_qualification=runtime.cpu, started_unix=time.time()))
    job = None; fit_artifacts = []
    if operation['type'] == 'adapted':
        job = next(j for j in inventory['fits'] if j['id'] == operation['job_id'])
        if operation['fit']: runtime.fit_task(job, operation['save_epochs'])
        done = runtime.verify_fit(job, operation['save_epochs'])
        directory = runtime.fit_directory(job)
        paths = [directory / 'complete.json', directory / 'fit_plan.json', directory.parent / 'receipt.json']
        paths += [directory / s['file'] for s in done['snapshots']]
        fit_artifacts = [artifact(p, runtime.output) for p in paths]
        task_id = job['task_id']
    elif operation['type'] == 'baseline':
        baseline = next(b for b in inventory['baselines'] if b['id'] == operation['baseline_id'])
        task_id = baseline['task_id']
    else: raise ValueError('Unknown adaptation operation')
    evidence = []
    for request in operation['queries']:
        runtime.score(task_id, request['role'], request['epoch'], job=job)
        evidence.append(pin_query(runtime, task_id, request, job))
    runtime.base.restore()
    receipt = dict(operation=operation, source=runtime.source, queries=evidence,
        fit_artifacts=fit_artifacts, base_parameter_sha256=runtime.base.digest,
        base_restored=True, cpu_qualification=runtime.cpu, gpu_uuid=gpu_uuid,
        finished_unix=time.time())
    immutable(output / 'complete.json', receipt)
    return receipt


def verify_operation(output, operation, source, *, cpu=False):
    """Bind already independently verified native queries to their saved bytes."""
    directory = operation_directory(output, operation); done = read(directory / 'complete.json')
    started = read(directory / 'invocation.json')
    if (done['operation'] != operation or started['operation'] != operation
            or done['source'] != source or started['source'] != source
            or not done['base_restored'] or done['cpu_qualification'] != cpu
            or started['cpu_qualification'] != cpu or done['gpu_uuid'] != started['gpu_uuid']
            or (done['gpu_uuid'] is not None if cpu else done['gpu_uuid'] not in evaluation.GPU_UUIDS)
            or len(done['queries']) != len(operation['queries'])
            or bool(done['fit_artifacts']) != (operation['type'] == 'adapted')):
        raise ValueError('Adaptation operation completion changed')
    for proof, request in zip(done['queries'], operation['queries'], strict=True):
        if any(proof[k] != v for k, v in request.items()) or len(proof['artifacts']) != 5:
            raise ValueError('Query evidence differs from the stage operation')
    for item in done['fit_artifacts'] + [a for q in done['queries'] for a in q['artifacts']]:
        if artifact(item['path'], Path(output)) != item:
            raise ValueError('Saved adaptation evidence changed: '+item['path'])
    return done


def freeze_selection(path):
    """Only complete B validation operations can supply epoch-selection rows."""
    path = Path(path); plan = verify_plan(path); output = path.parent / 'tasks'
    rows = {cell['id']: [] for cell in plan['adaptation']['cells']}; evidence = {}
    sources = {regime: runtime_source(path, regime) for regime in ('A', 'B')}
    jobs = {j['id']: j for j in plan['inventory']['fits']}
    for operation in operations(plan['inventory'], 'selection')['operations']:
        done = verify_operation(output, operation, sources[operation['regime']])
        receipt = operation_directory(output, operation) / 'complete.json'
        evidence[str(receipt)] = sha256(receipt)
        if operation['type'] != 'adapted': continue
        job = jobs[operation['job_id']]
        for query in done['queries']:
            result = read(query['artifacts'][0]['path'])
            if (query['task_id'] != job['task_id'] or query['scope'] != job['scope']
                    or query['role'] != 'validation' or result['source']['epoch'] != query['epoch']):
                raise ValueError('Selection received a different task or non-validation query')
            rows[job['cell_id']].extend(result['selection_records'])
    chosen = {cell['id']: tasks.protocol.select_epoch(rows[cell['id']], cell)
        for cell in plan['adaptation']['cells']}
    result = dict(plan_sha256=sha256(path), selections=chosen, operation_sha256=evidence,
        test_queries_used=False, additional_seed_fits=0, source_sha256=software())
    target = path.parent / 'selection.json'; immutable(target, result)
    return target


def verify_selection(path):
    path = Path(path); selected = read(path.parent / 'selection.json'); plan = read(path)
    if (selected['plan_sha256'] != sha256(path) or selected['test_queries_used']
            or selected['additional_seed_fits'] or selected['source_sha256'] != software()
            or set(selected['selections']) != {c['id'] for c in plan['adaptation']['cells']}):
        raise ValueError('Adaptation epoch-selection identity changed')
    expected = {str(operation_directory(path.parent / 'tasks', o) / 'complete.json')
        for o in operations(plan['inventory'], 'selection')['operations']}
    if set(selected['operation_sha256']) != expected:
        raise ValueError('Adaptation selection omitted a frozen validation operation')
    for name, digest in selected['operation_sha256'].items():
        if sha256(Path(name)) != digest: raise ValueError('Adaptation selection evidence changed')
    # Rebuild scalar selection from the pinned, previously verified query rows.
    # Raw payload hashes remain available in each operation for the final audit.
    jobs = {j['id']: j for j in plan['inventory']['fits']}
    rows = {c['id']: [] for c in plan['adaptation']['cells']}
    for name in selected['operation_sha256']:
        done = read(name); op = done['operation']
        if op['type'] != 'adapted': continue
        job = jobs[op['job_id']]
        for query in done['queries']:
            item = query['artifacts'][0]
            if sha256(Path(item['path'])) != item['sha256']: raise ValueError('Selected query metrics changed')
            rows[job['cell_id']].extend(read(item['path'])['selection_records'])
    rebuilt = {c['id']: tasks.protocol.select_epoch(rows[c['id']], c) for c in plan['adaptation']['cells']}
    if rebuilt != selected['selections']: raise ValueError('Selected epoch differs from frozen validation evidence')
    return selected


def load_runtime(path, regime, *, cpu=False):
    import torch
    from reward_model.model import load_reward_model
    from reward_model.training.config import load_reward_training_config
    from .adaptation_views import AdaptationViews
    model_info = read(path)['models'][regime]
    config = load_reward_training_config(model_info['job']['config_path'])
    model = load_reward_model(model_info['checkpoint']['path'], device=torch.device('cpu'))
    evaluation.postselection.verify_model_config(model.config.to_dict(), config.model,
        encoder_widths={side: getattr(model, side+'_encoder').config.hidden_size for side in ('protein', 'molecule')})
    if json.loads(json.dumps(model.config.to_dict())) != read(Path(model_info['checkpoint']['path']) / 'config.json'):
        raise ValueError('Loaded adaptation base configuration changed')
    config = tasks.protocol.adaptation_config(config, path.parent / 'tasks')
    config = replace(config, model=model.config)
    views = AdaptationViews(model_info['views']['path'], model_info['views']['sha256'])
    model.to('cpu' if cpu else 'cuda')
    return tasks.TaskRuntime(model, views, config, path.parent / 'tasks', runtime_source(path, regime), cpu=cpu)


def worker(path, stage_path, index):
    """Run the coordinator's immutable shard on exactly one authorized device."""
    if type(index) is not int or index not in range(4): raise ValueError('Invalid GPU worker index')
    gpu = evaluation.GPU_UUIDS[index]
    if (ROOT != Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES') != gpu
            or os.environ.get('ACCELERATE_USE_CPU', '').lower() != 'false'
            or os.environ.get('WANDB_MODE') != 'disabled'):
        raise RuntimeError('Adaptation worker requires one authorized Atum GPU; aggregate logging belongs to the coordinator')
    path, stage_path = Path(path), Path(stage_path); plan = verify_plan(path); stage = read(stage_path)
    name = stage['stage']
    if stage_path != path.parent / 'stages' / name / 'plan.json' or stage['plan_sha256'] != sha256(path):
        raise ValueError('Worker stage belongs to a different execution')
    selected = verify_selection(path) if name == 'remaining' else None
    expected = operations(plan['inventory'], name, None if selected is None else selected['selections'])
    if (stage['work'] != expected or stage['selection_sha256'] != (sha256(path.parent / 'selection.json') if selected else None)
            or stage['workers'] != (1 if name == 'qualification' else 4)):
        raise ValueError('Worker stage contains an unregistered task or epoch')
    if index >= stage['workers']: raise ValueError('Worker is outside this stage')
    if name != 'qualification':
        qualified = read(path.parent / 'qualification.json')
        if qualified['plan_sha256'] != sha256(path) or not qualified['qualified']:
            raise ValueError('Scientific GPU fitting/query qualification is missing')
        for filename, digest in qualified['operation_sha256'].items():
            if sha256(Path(filename)) != digest: raise ValueError('GPU qualification evidence changed')
    # The controller must be the actual parent while it holds the shared lock.
    parent = evaluation.control_queue.process_identity(os.getppid())
    if parent != stage['controller']: raise ValueError('Adaptation worker lost its registered controller')
    import torch
    torch.set_num_threads(2)
    assigned = expected['operations'][index::stage['workers']]
    record_path = stage_path.parent / f'worker{index}.json'
    if record_path.exists(): raise RuntimeError('Worker receipt already exists; do not relaunch this shard')
    runtime = None; identities = {}; peaks = {}
    for operation in assigned:
        if evaluation.control_queue.process_identity(os.getppid()) != parent:
            raise RuntimeError('Adaptation controller is no longer the registered parent')
        if runtime is None or runtime.views.record['regime'] != operation['regime']:
            if runtime is not None:
                del runtime; torch.cuda.empty_cache()
            runtime = load_runtime(path, operation['regime'])
        torch.cuda.reset_peak_memory_stats()
        execute_operation(runtime, operation, plan['inventory'], gpu_uuid=gpu)
        peaks[operation['id']] = dict(allocated_bytes=torch.cuda.max_memory_allocated(),
            reserved_bytes=torch.cuda.max_memory_reserved())
        receipt = operation_directory(runtime.output, operation) / 'complete.json'
        identities[str(receipt)] = sha256(receipt)
        write_json(stage_path.parent / f'progress{index}.json', dict(completed=len(identities), total=len(assigned),
            latest_operation=operation['id'], gpu_uuid=gpu, updated_unix=time.time()))
    immutable(record_path, dict(stage_sha256=sha256(stage_path), index=index, gpu_uuid=gpu,
        operation_sha256=identities, gpu_peak_memory=peaks, completed=len(assigned), finished_unix=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()
    worker(args.plan, args.stage, args.index)
