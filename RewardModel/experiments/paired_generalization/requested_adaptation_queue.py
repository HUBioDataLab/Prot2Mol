"""Run the fixed E49-E52 stages under one GPU owner, then aggregate evidence.

No search, support redraw or automatic retry. The two qualification fits are
part of B validation; each within-unit support always starts from its base.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import time

from .run import ROOT, WORK, sha256, write_json
from . import requested_adaptation_execution as execution
from . import requested_results, adaptation_statistics as statistics

cq = execution.evaluation.control_queue
read, immutable = cq.read, cq.immutable
STAGES = ('qualification', 'selection', 'remaining')
WAIT_SECONDS = 600


def software():
    result = requested_results.software()
    result['source_sha256'].update(execution.software()['source_sha256'])
    for path in (Path(__file__), Path(statistics.__file__)):
        result['source_sha256'][str(path.relative_to(ROOT))] = sha256(path)
    from importlib.metadata import version
    for name in ('wandb', 'pandas', 'scipy'): result['packages'][name] = version(name)
    return result


def prepare(analysis_paths):
    path = execution.prepare(analysis_paths)
    immutable(path.parent / 'controller_plan.json', dict(plan_sha256=sha256(path),
        software=software(), stages=list(STAGES), wait_seconds=WAIT_SECONDS,
        logging='One aggregate W&B run with kind/scope/k/regime/seed rows; no per-unit W&B runs.',
        partial_policy='Stop on any failed stage or partial worker; preserve outputs and require inspection.'))
    return path


def verify_controller(path):
    path = Path(path); plan = execution.verify_plan(path); record = read(path.parent / 'controller_plan.json')
    if (record['plan_sha256'] != sha256(path) or record['software'] != software()
            or record['stages'] != list(STAGES) or record['wait_seconds'] != WAIT_SECONDS):
        raise ValueError('Adaptation coordinator source or execution contract changed')
    return plan


def storage_requirement(plan, stages, selections=None):
    """Conservative remaining-output reservation, including query serialization.

    Tensor sizes come from the real checkpoint; query rows from frozen views.
    The overhead allowance is a resource guard, never a scientific sample cap.
    """
    jobs = {j['id']: j for j in plan['inventory']['fits']}
    baselines = {b['id']: b for b in plan['inventory']['baselines']}
    records = {r: read(m['views']['path']) for r, m in plan['models'].items()}
    # Before B selection, epoch1 represents the maximum two-snapshot retention;
    # it is only an allocation bound and never a selected scientific epoch.
    if selections is None:
        selections = {c['id']: dict(cell_id=c['id'], status='selected', selected=dict(epoch=1))
            for c in plan['adaptation']['cells']}
    operations = {o['id']: o for stage in stages for o in
        execution.operations(plan['inventory'], stage, selections if stage == 'remaining' else None)['operations']}
    raw, query_rows, query_count = 0, 0, 0
    for op in operations.values():
        record = records[op['regime']]
        if op['type'] == 'adapted':
            job = jobs[op['job_id']]; task = record['tasks'][job['task_id']]
            if op['fit']:
                raw += len(op['save_epochs']) * plan['snapshot_budget']['bytes_per_snapshot_by_scope'][job['scope']]
        else:
            task = record['tasks'][baselines[op['baseline_id']]['task_id']]
        for query in op['queries']:
            view = task['validation_view'] if task['kind'] == 'across' and query['role'] == 'validation' else task['query_view']
            query_rows += record['views'][view]['statistics']['rows']; query_count += 1
    # Native payloads contain repeated IDs and three pathways; JSON receipts
    # also contain assay records. Reserve 4KiB/observation and 128KiB/query.
    overhead = query_rows * 4096 + query_count * 128 * 1024
    reserve = 16 * 1024**3
    required = (raw * 110 + 99)//100 + overhead + reserve
    return dict(raw_adapter_bytes=raw, query_rows=query_rows, queries=query_count,
        query_and_receipt_allowance_bytes=overhead, reserve_bytes=reserve,
        required_archive_bytes=required, allocation_only=True)


def storage_gate(requirement):
    state = cq.storage_gate()
    archive = next(r for r in state['records'] if r['storage'] == 'archive')
    available = min(archive['quota_remaining_bytes'], archive['filesystem_free_bytes'])
    return dict(ready=state['ready'] and available >= requirement['required_archive_bytes'],
        available_archive_bytes=available, requirement=requirement, native=state)


def verify_stage(path, stage, *, artifacts=True):
    path = Path(path); plan = read(path); target = path.parent / 'stages' / stage
    schedule = read(target / 'plan.json'); complete = read(target / 'complete.json')
    selected = execution.verify_selection(path) if stage == 'remaining' else None
    expected = execution.operations(plan['inventory'], stage, None if selected is None else selected['selections'])
    if (schedule['stage'] != stage or schedule['plan_sha256'] != sha256(path)
            or schedule['work'] != expected or schedule['workers'] != (1 if stage == 'qualification' else 4)
            or schedule['selection_sha256'] != (sha256(path.parent / 'selection.json') if selected else None)
            or complete['stage_sha256'] != sha256(target / 'plan.json')
            or set(complete['workers_sha256']) != {str(i) for i in range(schedule['workers'])}):
        raise ValueError('Adaptation stage completion changed')
    catalog = {}; sources = {r: execution.runtime_source(path, r) for r in ('A', 'B')}
    for index in range(schedule['workers']):
        receipt_path = target / f'worker{index}.json'; receipt = read(receipt_path)
        assigned = expected['operations'][index::schedule['workers']]
        files = {str(execution.operation_directory(path.parent / 'tasks', op) / 'complete.json'): op for op in assigned}
        if (complete['workers_sha256'][str(index)] != sha256(receipt_path)
                or receipt['stage_sha256'] != sha256(target / 'plan.json') or receipt['index'] != index
                or receipt['gpu_uuid'] != execution.evaluation.GPU_UUIDS[index]
                or receipt['completed'] != len(assigned) or set(receipt['operation_sha256']) != set(files)
                or set(receipt['gpu_peak_memory']) != {op['id'] for op in assigned}):
            raise ValueError('Adaptation worker omitted or changed an assigned operation')
        for name, op in files.items():
            if sha256(Path(name)) != receipt['operation_sha256'][name]:
                raise ValueError('Adaptation operation changed after worker completion')
            done = execution.verify_operation(path.parent / 'tasks', op, sources[op['regime']]) if artifacts else read(name)
            if done['operation'] != op or done['source'] != sources[op['regime']]:
                raise ValueError('Adaptation operation belongs to another stage contract')
            catalog[op['id']] = done
    return catalog


def qualify(path):
    """Verify the two actual support fits and all30 native validation queries."""
    path = Path(path); catalog = verify_stage(path, 'qualification'); plan = read(path)
    jobs = {j['id']: j for j in plan['inventory']['fits']}
    worker = read(path.parent / 'stages/qualification/worker0.json'); evidence, measures = {}, {}
    for identity, done in catalog.items():
        op = done['operation']; job = jobs[op['job_id']]
        directory = path.parent / 'tasks/fits' / job['id'] / 'native'
        native = read(directory / 'complete.json'); fit = read(directory / 'fit_plan.json')
        optimizer_names = [name for group in native['optimizer_groups'] for name in group['names']]
        if (job['regime'] != 'B' or job['task_id'] != 'across_k1' or native['cpu_qualification']
                or native['scope'] != job['scope'] or native['epochs'] != 15 or native['global_step'] < 1
                or native['query_examples_received'] or native['selection_performed']
                or not native['frozen_parameters_unchanged'] or not done['base_restored']
                or len(optimizer_names) != len(set(optimizer_names))
                or sorted(optimizer_names) != sorted(native['trainable_parameters'])
                or native['base_parameter_sha256'] != done['base_parameter_sha256']
                or [s['epoch'] for s in native['snapshots']] != execution.HORIZON
                or fit['training']['seed'] != 42 or fit['training']['per_device_train_batch_size'] != 12
                or fit['training']['gradient_accumulation_steps'] != 16 or fit['training']['report_to']):
            raise ValueError('Production adaptation optimizer, scope or frozen-state qualification failed')
        for query in done['queries']:
            result = read(query['artifacts'][0]['path']); forward = read(query['artifacts'][2]['path'])
            if (query['role'] != 'validation' or not result['base_restored'] or not result['modes_and_rng_restored']
                    or forward['cpu_qualification'] or forward['world_size'] != 1 or not forward['device'].startswith('cuda')
                    or forward['bf16'] != fit['training']['bf16'] or forward['fp16'] != fit['training']['fp16']
                    or not forward['parameters_unchanged'] or forward['optimizer_created']):
                raise ValueError('Production adaptation query precision/restoration qualification failed')
        peak = worker['gpu_peak_memory'][identity]
        if not 0 < peak['allocated_bytes'] <= peak['reserved_bytes']:
            raise ValueError('No valid production GPU memory evidence')
        name = execution.operation_directory(path.parent / 'tasks', op) / 'complete.json'
        evidence[str(name)] = sha256(name)
        measures[job['scope']] = dict(global_step=native['global_step'], support_rows=native['support_rows'],
            support_assays=native['support_assays'], sampler_stats=native['sampler_stats'],
            optimizer_groups=native['optimizer_groups'], gpu_peak_memory=peak,
            base_parameter_sha256=done['base_parameter_sha256'], validation_queries=len(done['queries']))
    if set(measures) != set(execution.tasks.protocol.SCOPES):
        raise ValueError('Both registered adaptation scopes must qualify')
    result = dict(plan_sha256=sha256(path), qualified=True, operation_sha256=evidence, measures=measures,
        additional_fits=0, scope='First actual B across-k1 fits, reused by selection; exact native optimizer/frozen-state/adapter/query checks. No support or parameter selection.')
    immutable(path.parent / 'qualification.json', result)
    return result


def run_stage(path, stage, run):
    """Called while the process holds controlled/driver.lock for the whole stage."""
    path = Path(path); plan = read(path); target = path.parent / 'stages' / stage
    if target.exists(): raise RuntimeError('Adaptation stage already exists; inspect before any restart')
    selected = execution.verify_selection(path) if stage == 'remaining' else None
    work = execution.operations(plan['inventory'], stage, None if selected is None else selected['selections'])
    controller = cq.process_identity(os.getpid())
    if controller is None: raise RuntimeError('Cannot identify adaptation GPU owner')
    schedule = dict(stage=stage, plan_sha256=sha256(path), work=work,
        selection_sha256=sha256(path.parent / 'selection.json') if selected else None,
        workers=1 if stage == 'qualification' else 4, controller=controller)
    immutable(target / 'plan.json', schedule)
    children, handles = [], []
    try:
        for index in range(schedule['workers']):
            env = execution.evaluation.postselection.launch_environment('requested_adaptation_seed42')
            env.update(CUDA_VISIBLE_DEVICES=execution.evaluation.GPU_UUIDS[index], WANDB_MODE='disabled',
                ACCELERATE_USE_CPU='false', PYTHONPATH=str(ROOT / 'RewardModel'),
                OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            command = [str(ROOT / '.venv/bin/python'), '-m', 'experiments.paired_generalization.requested_adaptation_execution',
                '--plan', str(path), '--stage', str(target / 'plan.json'), '--index', str(index)]
            handle = (target / f'worker{index}.log').open('x'); handles.append(handle)
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            children.append(child)
            immutable(target / f'launch{index}.json', dict(pid=child.pid, command=command,
                gpu_uuid=execution.evaluation.GPU_UUIDS[index], started_unix=time.time()))
        # A thread waits on each child; failure is noticed without waiting for a
        # slower shard, and no training process is polled in a tight loop.
        with ThreadPoolExecutor(max_workers=len(children)) as pool:
            futures = {pool.submit(child.wait): i for i, child in enumerate(children)}
            try:
                for future in as_completed(futures):
                    code = future.result(); index = futures[future]
                    if code: raise RuntimeError(f'Adaptation {stage} worker{index} failed with {code}')
                    run.log({f'progress/{stage}/finished_workers': sum(c.poll() is not None for c in children)})
            except BaseException:
                for child in children:
                    if child.poll() is None: child.terminate()
                raise
    finally:
        for child in children:
            if child.poll() is None: child.terminate()
            child.wait()
        for handle in handles: handle.close()
    immutable(target / 'complete.json', dict(stage_sha256=sha256(target / 'plan.json'),
        workers_sha256={str(i): sha256(target / f'worker{i}.json') for i in range(schedule['workers'])},
        finished_unix=time.time()))
    # Native operations verified their raw outputs before receipt creation.
    # The deep pass occurs for qualification, selection, and the final report.
    verify_stage(path, stage, artifacts=False)
    run.log({f'progress/{stage}/operations': len(work['operations']),
        f'progress/{stage}/fits': sum(o['fit'] for o in work['operations'])})


def query_catalog(path):
    """Verify every completed operation once, including unselected B epochs."""
    path = Path(path); queries, fits, receipt_hashes = {}, {}, {}
    for stage in ('selection', 'remaining'):
        for done in verify_stage(path, stage).values():
            operation = done['operation']; regime = operation['regime']
            receipt = execution.operation_directory(path.parent / 'tasks', operation) / 'complete.json'
            receipt_hashes[str(receipt)] = sha256(receipt)
            for proof in done['queries']:
                key = regime, proof['task_id'], proof['scope'], proof['role'], proof['epoch']
                if key in queries: raise ValueError('Adaptation query was scored more than once')
                queries[key] = proof
            if operation['type'] == 'adapted':
                native = path.parent / 'tasks/fits' / operation['job_id'] / 'native/complete.json'
                if operation['job_id'] in fits and fits[operation['job_id']] != str(native):
                    raise ValueError('Support task has more than one native fit')
                fits[operation['job_id']] = str(native)
    return queries, fits, receipt_hashes


def summarize(plan, selections, query_records, fit_records, view_records, groups):
    """Aggregate only selected-epoch queries supplied by the verified catalog."""
    import pandas as pd
    jobs = plan['inventory']['fits']; comparisons, tables, points, exposure = {}, {}, [], []
    baselines = []
    for regime in ('A', 'B'):
        record = view_records[regime]; indexed = groups[regime].set_index('group_id', drop=False)
        for kind in ('across', 'within'):
            for role in ('validation', 'test'):
                requests = [b for b in plan['inventory']['baselines'] if b['regime'] == regime and b['role'] == role
                    and record['tasks'][b['task_id']]['kind'] == kind]
                if not requests: continue
                ids, rows = [], []
                for request in requests:
                    ids.extend(record['views'][request['query_view']]['group_ids'])
                    rows.extend(query_records[(regime, request['task_id'], None, role, 0)]['assay_records'])
                for scorer, table in statistics.assay_tables(rows, indexed.loc[ids].reset_index(drop=True)).items():
                    baselines.append(dict(regime=regime, kind=kind, role=role, k=0, seed=42, scorer=scorer,
                        cohort='Union of frozen available query units; each positive-k comparison below uses its own matched baseline.',
                        **statistics.point(table)))
    for job in jobs:
        chosen = selections[job['cell_id']]
        if job['id'] not in fit_records:
            if chosen['status'] != 'not_selectable': raise ValueError('Missing selected support fit')
            continue
        done = fit_records[job['id']]
        epoch = chosen['selected']['epoch'] if chosen['status'] == 'selected' else None
        steps = next((s['step'] for s in done['snapshots'] if s['epoch'] == epoch), None)
        if done['epochs'] != 15 or (epoch is not None and steps is None):
            raise ValueError('Support trajectory does not contain its frozen selected epoch')
        exposure.append(dict(job_id=job['id'], regime=job['regime'], kind=job['kind'], scope=job['scope'],
            k=job['k'], seed=42, support_rows=done['support_rows'], support_assays=done['support_assays'],
            epochs=15, optimizer_steps=done['global_step'], selected_epoch=epoch, selected_optimizer_step=steps,
            sampler_stats_per_epoch=done['sampler_stats'],
            exposure_interpretation='Native sampler counts and observed optimizer steps. Sampled lists may repeat molecules; counts are not unique examples or realized nonzero-gradient pair counts.'))
    for regime in ('A', 'B'):
        record = view_records[regime]; frame = groups[regime].set_index('group_id', drop=False)
        for kind in ('across', 'within'):
            for scope in execution.tasks.protocol.SCOPES:
                for role in ('validation', 'test'):
                    active = [j for j in jobs if (j['regime'], j['kind'], j['scope']) == (regime, kind, scope)
                        and role in j['query_roles'] and selections[j['cell_id']]['status'] == 'selected']
                    if not active: continue
                    key = f'{regime}__{kind}__{scope}__{role}'
                    adapted, ids, base_keys = {}, {}, set()
                    for job in active:
                        epoch = selections[job['cell_id']]['selected']['epoch']; k = job['k']
                        query_key = regime, job['task_id'], scope, role, epoch
                        adapted.setdefault(k, []).extend(query_records[query_key]['assay_records'])
                        task = record['tasks'][job['task_id']]
                        view = task['validation_view'] if kind == 'across' and role == 'validation' else task['query_view']
                        ids.setdefault(k, []).extend(record['views'][view]['group_ids'])
                        base_keys.add((regime, job['baseline_task'], None, role, 0))
                    by_k = {k: frame.loc[identity].reset_index(drop=True) for k, identity in ids.items()}
                    baseline = [row for query_key in sorted(base_keys, key=str) for row in query_records[query_key]['assay_records']]
                    result = statistics.compare_budgets(baseline, adapted, by_k)
                    comparisons[key] = result
                    for k in sorted(adapted):
                        tables[(regime, kind, scope, role, k)] = statistics.assay_tables(adapted[k], by_k[k])
                        for scorer in statistics.SCORERS:
                            item = result['available'][scorer][str(k)]
                            cell = f'{kind}__{scope}__k{k}'
                            points.append(dict(regime=regime, kind=kind, scope=scope, role=role, k=k, seed=42,
                                selected_epoch=selections[cell]['selected']['epoch'], scorer=scorer,
                                baseline_macro=item['baseline']['macro_spearman'], adapted_macro=item['adapted']['macro_spearman'],
                                paired_change=item['adapted_minus_baseline']['mean_change'],
                                eligible_assays=item['adapted']['eligible_groups'], defined_assays=item['adapted']['defined_groups'],
                                units=item['adapted']['units'], paired_unit_bootstrap=item['adapted_minus_baseline']['unit_bootstrap']))
    scope_changes = {}
    for regime, kind, scope, role, k in tables:
        if scope != 'heads': continue
        other = regime, kind, 'heads_cross_attention', role, k
        if other not in tables: continue
        first = tables[(regime, kind, scope, role, k)]; second = tables[other]
        scope_changes[f'{regime}__{kind}__{role}__k{k}'] = {
            scorer: statistics.paired_group_change(first[scorer], second[scorer]) for scorer in statistics.SCORERS}
    return dict(comparisons=comparisons, scope_changes=scope_changes, points=points, baseline_points=baselines, support_exposures=exposure,
        unavailable=plan['inventory']['unavailable'], not_selectable_cells=[c for c, s in selections.items() if s['status'] != 'selected'],
        seed=42, seed_uncertainty_available=False, additional_seed_fits=0,
        interpretation='All supported budgets and both registered scopes retained. Changes use identical whole query units and native assay-macro correlations. Base validation was previously used for checkpoint selection; these are conditional mechanistic probes, not independent deployment validation. No support-redraw uncertainty or causal across-versus-within claim.')


def report(path, run):
    import pandas as pd
    import wandb
    path = Path(path); plan = verify_controller(path); selected = execution.verify_selection(path)
    target = path.parent / 'report'
    if target.exists(): raise RuntimeError('Adaptation report already exists; inspect before regenerating it')
    catalog, fits, evidence = query_catalog(path)
    records = {r: read(m['views']['path']) for r, m in plan['models'].items()}
    groups = {r: pd.read_parquet(v['groups']['path']) for r, v in records.items()}
    needed = {(b['regime'], b['task_id'], None, b['role'], 0) for b in plan['inventory']['baselines']}
    for job in plan['inventory']['fits']:
        choice = selected['selections'][job['cell_id']]
        if choice['status'] != 'selected': continue
        for role in job['query_roles']:
            needed.add((job['regime'], job['task_id'], job['scope'], role, choice['selected']['epoch']))
            needed.add((job['regime'], job['baseline_task'], None, role, 0))
    queries = {key: read(catalog[key]['artifacts'][0]['path']) for key in needed}
    summary = summarize(plan, selected['selections'], queries, {key: read(value) for key, value in fits.items()}, records, groups)
    immutable(target / 'summary.json', summary)
    immutable(target / 'selection.json', selected)
    flat = [{k: v for k, v in p.items() if k != 'paired_unit_bootstrap'} for p in summary['points']]
    run.log({'adaptation/results_by_cell': wandb.Table(dataframe=pd.DataFrame(flat)),
        'adaptation/unchanged_baselines': wandb.Table(dataframe=pd.DataFrame([
            {k: v for k, v in p.items() if k != 'unit_bootstrap'} for p in summary['baseline_points']])),
        'adaptation/support_fits': len(fits), 'adaptation/selected_cells': sum(s['status'] == 'selected' for s in selected['selections'].values()),
        'adaptation/additional_seeds': 0})
    lines = ['# Requested adaptation results (seed 42)', '',
        'Native macro Spearman; k=0 is the unchanged base on the same query assays. Paired changes use mutually defined assays.', '',
        '| Split | Probe | Scope | k | Epoch | Test base | Test adapted | Paired change | Defined/eligible |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    number = lambda x: 'undefined' if x is None else f'{x:.4f}'
    for p in flat:
        if p['role'] == 'test' and p['scorer'] == 'ranking_score':
            lines.append(f"| {p['regime']} | {p['kind']} | {p['scope']} | {p['k']} | {p['selected_epoch']} | {number(p['baseline_macro'])} | {number(p['adapted_macro'])} | {number(p['paired_change'])} | {p['defined_assays']}/{p['eligible_assays']} |")
    lines.extend(['', summary['interpretation'], '',
        'The JSON report includes all three pathways, validation coverage, available and common-unit/common-defined cohorts, paired whole-unit intervals, adjacent-k changes, scope contrasts, sampler exposure and optimizer steps.', '',
        'Unavailable design cells: '+json.dumps(summary['unavailable'], sort_keys=True),
        'No selectable validation epoch: '+json.dumps(summary['not_selectable_cells']), ''])
    destination = target / 'REPORT.md'
    execution.evaluation.control_queue._write_immutable(destination, '\n'.join(lines))
    completed = dict(plan_sha256=sha256(path), controller_plan_sha256=sha256(path.parent / 'controller_plan.json'),
        selection_sha256=sha256(path.parent / 'selection.json'), operation_sha256=evidence,
        artifacts_sha256={p.name: sha256(p) for p in target.iterdir() if p.is_file()},
        scientific_provenance_verified=True, requested_adaptation_complete=True,
        experiments=['E49', 'E50', 'E51', 'E52'], seed=42, new_support_fits=len(fits),
        test_used_for_selection=False, seed_uncertainty_available=False, finished_unix=time.time())
    immutable(target / 'complete.json', completed)
    bundle = wandb.Artifact('requested-adaptation-seed42-'+sha256(path)[:12], type='evaluation',
        metadata=dict(plan_sha256=sha256(path), seed=42, test_used_for_selection=False))
    for name in ('summary.json', 'selection.json', 'REPORT.md', 'complete.json'):
        bundle.add_file(str(target / name), name=name)
    run.log_artifact(bundle)
    return target / 'complete.json'


def verify_complete(path, *, deep=False):
    """Verify reporting provenance; optionally rehash every native task payload."""
    path = Path(path); plan = verify_controller(path); result = read(path.parent / 'complete.json')
    target = path.parent / 'report'; report_path = target / 'complete.json'; done = read(report_path)
    selected = execution.verify_selection(path)
    if (result['phase'] != 'requested_adaptation_complete' or result['plan_sha256'] != sha256(path)
            or result['report'] != str(report_path) or result['report_sha256'] != sha256(report_path)
            or result['scientific_study_complete'] or result['additional_seed_fits'] or result['parameter_search']
            or done['plan_sha256'] != sha256(path) or done['controller_plan_sha256'] != sha256(path.parent / 'controller_plan.json')
            or done['selection_sha256'] != sha256(path.parent / 'selection.json')
            or not done['scientific_provenance_verified'] or not done['requested_adaptation_complete']
            or done['experiments'] != ['E49','E50','E51','E52'] or done['seed'] != 42
            or done['test_used_for_selection'] or done['seed_uncertainty_available']
            or set(done['artifacts_sha256']) != {'summary.json','selection.json','REPORT.md'}):
        raise ValueError('Adaptation completion has changed or overstates its scope')
    for name, digest in done['artifacts_sha256'].items():
        if sha256(target / name) != digest: raise ValueError('Adaptation report artifact changed')
    operations = [o for stage in ('selection','remaining') for o in execution.operations(plan['inventory'], stage,
        selected['selections'] if stage == 'remaining' else None)['operations']]
    expected = {str(execution.operation_directory(path.parent / 'tasks', op) / 'complete.json') for op in operations}
    if (set(done['operation_sha256']) != expected
            or done['new_support_fits'] != sum(op['fit'] for op in operations)):
        raise ValueError('Adaptation report did not cover the complete frozen task inventory')
    for name, digest in done['operation_sha256'].items():
        if sha256(Path(name)) != digest: raise ValueError('Adaptation task evidence changed after reporting')
    if deep:
        for stage in STAGES: verify_stage(path, stage)
    return result, done


def run(path):
    import fcntl
    cq.runtime_guard(); path = Path(path); plan = verify_controller(path); target = path.parent
    # Keep logging/cache writes inside the authorized project, including the
    # parent controller rather than only its GPU children.
    logging_environment = execution.evaluation.postselection.launch_environment('requested_adaptation_seed42')
    for key in ('WANDB_DIR', 'WANDB_CACHE_DIR', 'WANDB_CONFIG_DIR', 'WANDB_DATA_DIR', 'TMPDIR'):
        os.environ[key] = logging_environment[key]
        Path(logging_environment[key]).mkdir(parents=True, exist_ok=True)
    os.environ.pop('WANDB_RUN_ID', None); os.environ.pop('WANDB_RESUME', None)
    import wandb
    with (target / 'controller.lock').open('a') as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (target / 'invocation.json').exists(): raise RuntimeError('Adaptation queue already invoked; inspect, do not restart')
        immutable(target / 'invocation.json', dict(pid=os.getpid(), controller=cq.process_identity(os.getpid()),
            plan_sha256=sha256(path), started_unix=time.time()))
        log = None
        try:
            log = wandb.init(project='Prot2Mol_RewardModel', entity='hubiodatascilab',
                group='paired_generalization_adaptation', job_type='mechanistic_probes', name='requested_adaptation_seed42',
                dir=str(WORK), config=dict(plan_sha256=sha256(path), seed=42, learning_rate_multiplier=1.,
                    experiments=plan['experiments'], support_fits=plan['inventory']['accounting']['support_fits'],
                    parameter_search=False, additional_seed_fits=0))
            immutable(target / 'wandb.json', dict(id=log.id, url=log.url, project=log.project, entity=log.entity))
            for position, stage in enumerate(STAGES):
                selected = execution.verify_selection(path)['selections'] if stage == 'remaining' else None
                requirement = storage_requirement(plan, STAGES[position:], selected)
                while True:
                    with (WORK / 'controlled/driver.lock').open('a') as lock:
                        acquired = False
                        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); acquired = True
                        except BlockingIOError: pass
                        holders = cq.gpu_processes() if acquired else None
                        storage = storage_gate(requirement) if acquired and not holders else None
                        if acquired and not holders and storage['ready']:
                            immutable(target / f'storage_{stage}.json', dict(checked_unix=time.time(), **storage))
                            write_json(target / 'status.json', dict(phase='running_'+stage, pid=os.getpid(), updated_unix=time.time()))
                            run_stage(path, stage, log)
                            break
                        phase = 'waiting_for_global_driver_lock' if not acquired else ('waiting_for_authorized_gpus' if holders else 'waiting_for_storage')
                        write_json(target / 'status.json', dict(phase=phase, stage=stage, storage=storage,
                            holders=holders, next_check_seconds=WAIT_SECONDS, updated_unix=time.time()))
                    time.sleep(WAIT_SECONDS)
                if stage == 'qualification': qualify(path)
                if stage == 'selection': execution.freeze_selection(path)
            report_path = report(path, log)
            log.finish(); log = None
            result = dict(phase='requested_adaptation_complete', plan_sha256=sha256(path),
                report=str(report_path), report_sha256=sha256(report_path),
                scientific_study_complete=False, additional_seed_fits=0, parameter_search=False, finished_unix=time.time())
            immutable(target / 'complete.json', result); write_json(target / 'status.json', result)
            verify_complete(path)
            return result
        except BaseException as error:
            write_json(target / 'status.json', dict(phase='stopped_for_inspection', error_type=type(error).__name__,
                error=str(error), updated_unix=time.time()))
            if log is not None: log.finish(exit_code=1)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.plan), indent=2))
