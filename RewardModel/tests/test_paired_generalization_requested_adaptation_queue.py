"""Exact stage ownership, resource gates and native metric aggregation."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.paired_generalization import requested_adaptation_queue as module
from experiments.paired_generalization.run import write_json
from test_paired_generalization_adaptation_protocol import fixture_protocol
from test_paired_generalization_adaptation_statistics import records as assay_records


@pytest.fixture
def scientific(tmp_path):
    primary, views, groups = fixture_protocol()
    adaptation = module.execution.tasks.protocol.build_protocol(primary, views, groups)
    adaptation['seeds'] = [42]
    for cell in adaptation['cells']: cell['total_fits'] //= 3
    adaptation['accounting']['support_fits'] //= 3
    inventory = module.execution.tasks.inventory(adaptation, views)
    models = {}
    for regime, record in views.items():
        indexed = groups[regime].set_index('group_id')
        for view in record['views'].values():
            view['statistics'] = dict(rows=int(indexed.loc[view['group_ids'], 'rows'].sum()))
        path = tmp_path / f'views_{regime}.json'; write_json(path, record)
        models[regime] = dict(views=dict(path=str(path), sha256=module.sha256(path)))
    choices = {c['id']: dict(cell_id=c['id'], status='selected', selected=dict(epoch=2)) for c in adaptation['cells']}
    plan = dict(inventory=inventory, adaptation=adaptation, models=models, experiments=['E49','E50','E51','E52'],
        snapshot_budget=dict(bytes_per_snapshot_by_scope=dict(heads=100, heads_cross_attention=200)))
    queries, fits = {}, {}
    for job in inventory['fits']:
        task = views[job['regime']]['tasks'][job['task_id']]
        fits[job['id']] = dict(epochs=15, global_step=15, support_rows=20, support_assays=job['k'],
            snapshots=[dict(epoch=2, step=2), dict(epoch=15, step=15)], sampler_stats=dict(num_dataset_items=1))
        for role in job['query_roles']:
            view = task['validation_view'] if job['kind'] == 'across' and role == 'validation' else task['query_view']
            frame = groups[job['regime']].set_index('group_id', drop=False).loc[views[job['regime']]['views'][view]['group_ids']]
            for scope, task_id, epoch, value in [(job['scope'], job['task_id'], 2, .2 if job['scope'] == 'heads' else .4),
                                                (None, job['baseline_task'], 0, .1)]:
                values = {r.group_id: value if r.ranking_eligible else None for r in frame.itertuples()}
                queries[(job['regime'], task_id, scope, role, epoch)] = dict(assay_records=assay_records(frame, values))
    return dict(plan=plan, choices=choices, queries=queries, fits=fits, views=views, groups=groups)


def summary(fixture):
    return module.summarize(fixture['plan'], fixture['choices'], fixture['queries'], fixture['fits'], fixture['views'], fixture['groups'])


def test_selected_queries_report_all_budgets_pathways_and_paired_scope_changes(scientific):
    result = summary(scientific)
    assert result['seed'] == 42 and not result['seed_uncertainty_available']
    assert len(result['baseline_points']) == 18
    assert all(p['macro_spearman'] == pytest.approx(.1) for p in result['baseline_points'])
    assert len(result['support_exposures']) == len(scientific['fits'])
    assert {p['scorer'] for p in result['points']} == set(module.statistics.SCORERS)
    assert {p['role'] for p in result['points']} == {'validation', 'test'}
    assert not any(p['regime'] == 'A' and p['kind'] == 'within' for p in result['points'])
    for point in result['points']:
        assert point['baseline_macro'] == pytest.approx(.1)
        expected = .2 if point['scope'] == 'heads' else .4
        assert point['adapted_macro'] == pytest.approx(expected)
        assert point['paired_change'] == pytest.approx(expected-.1)
        assert point['selected_epoch'] == 2
    for scopes in result['scope_changes'].values():
        assert all(s['mean_change'] == pytest.approx(.2) for s in scopes.values())
    for comparison in result['comparisons'].values():
        assert 'common_units_statistics' in comparison and 'common_defined_statistics' in comparison
        assert not comparison['scientific_provenance_verified']  # library correctly leaves verification to caller
    assert all(e['epochs'] == 15 and e['selected_optimizer_step'] == 2 for e in result['support_exposures'])
    json.dumps(result, allow_nan=False)


def test_missing_selected_fit_or_query_is_not_silently_dropped(scientific):
    key = next(iter(scientific['fits'])); native = scientific['fits'].pop(key)
    with pytest.raises(ValueError, match='Missing selected'): summary(scientific)
    scientific['fits'][key] = native
    scientific['queries'].pop(next(iter(scientific['queries'])))
    with pytest.raises(KeyError): summary(scientific)


def test_undefined_and_unselectable_cells_are_not_replaced_by_zero_or_final(scientific):
    cell = scientific['plan']['adaptation']['cells'][0]['id']
    scientific['choices'][cell].update(status='not_selectable', selected=None)
    for key, query in scientific['queries'].items():
        for row in query['assay_records']:
            if row['scorer'] == 'cosine_similarity': row.update(defined=False, rho=None)
    result = summary(scientific)
    assert result['not_selectable_cells'] == [cell]
    assert all(p['adapted_macro'] is None for p in result['points'] if p['scorer'] == 'cosine_similarity')
    kind, scope, k = cell.split('__'); k = int(k[1:])
    assert not any((p['kind'], p['scope'], p['k']) == (kind, scope, k) for p in result['points'])


def test_storage_budget_deduplicates_qualification_and_decreases_after_selection(scientific):
    plan = scientific['plan']
    whole = module.storage_requirement(plan, module.STAGES)
    no_prefix = module.storage_requirement(plan, ('selection', 'remaining'))
    assert whole == no_prefix
    remaining = module.storage_requirement(plan, ('remaining',), scientific['choices'])
    assert 0 < remaining['raw_adapter_bytes'] < whole['raw_adapter_bytes']
    expected = sum((15 if j['selection_fit'] else 2)*plan['snapshot_budget']['bytes_per_snapshot_by_scope'][j['scope']]
        for j in plan['inventory']['fits'])
    assert whole['raw_adapter_bytes'] == expected
    assert whole['required_archive_bytes'] > whole['raw_adapter_bytes']
    final = copy.deepcopy(scientific['choices'])
    for choice in final.values(): choice['selected']['epoch'] = 15
    assert module.storage_requirement(plan, ('remaining',), final)['raw_adapter_bytes']*2 == remaining['raw_adapter_bytes']


@pytest.mark.parametrize('quota,free,home,ready', [(999,5000,True,False),(5000,999,True,False),(5000,5000,False,False),(5000,5000,True,True)])
def test_storage_respects_both_quota_and_filesystem_space(monkeypatch, quota, free, home, ready):
    monkeypatch.setattr(module.cq, 'storage_gate', lambda: dict(ready=home, records=[dict(storage='archive',
        quota_remaining_bytes=quota, filesystem_free_bytes=free)]))
    assert module.storage_gate(dict(required_archive_bytes=1000))['ready'] == ready


@pytest.fixture
def scheduler(scientific, tmp_path, monkeypatch):
    path = tmp_path / 'plan.json'; write_json(path, scientific['plan'])
    (tmp_path / 'controlled').mkdir()
    monkeypatch.setattr(module, 'ROOT', tmp_path); monkeypatch.setattr(module, 'WORK', tmp_path)
    monkeypatch.setattr(module, 'verify_controller', lambda _: scientific['plan'])
    monkeypatch.setattr(module.cq, 'runtime_guard', lambda: None)
    monkeypatch.setattr(module.cq, 'process_identity', lambda pid: dict(pid=pid, fixture=True))
    monkeypatch.setattr(module.execution, 'runtime_source', lambda *args: dict(fixture=True))
    monkeypatch.setattr(module.execution, 'verify_selection', lambda _: dict(selections=scientific['choices']))
    launch = module.execution.evaluation.postselection.launch_environment
    def local_environment(identity):
        env = launch(identity)
        for key in ('WANDB_DIR','WANDB_CACHE_DIR','WANDB_CONFIG_DIR','WANDB_DATA_DIR','TMPDIR'):
            env[key] = str(tmp_path / key.lower())
        return env
    monkeypatch.setattr(module.execution.evaluation.postselection, 'launch_environment', local_environment)
    writes, children = [], []
    log = SimpleNamespace(log=writes.append, id='fixture', url='fixture', project='fixture', entity='fixture', finish=lambda **_: None)
    import wandb
    monkeypatch.setattr(wandb, 'init', lambda **_: log)
    monkeypatch.setattr(module.cq, 'gpu_processes', lambda: [])
    monkeypatch.setattr(module, 'storage_gate', lambda _: dict(ready=True))
    monkeypatch.setattr(module, 'qualify', lambda _: dict(qualified=True))
    monkeypatch.setattr(module.execution, 'freeze_selection', lambda _: write_json(path.parent / 'selection.json', dict(fixture=True)))
    monkeypatch.setattr(module, 'verify_stage', lambda *_args, **_kw: {})
    def report(_path, _run):
        target = tmp_path / 'report/complete.json'; write_json(target, dict(fixture=True)); return target
    monkeypatch.setattr(module, 'report', report)
    monkeypatch.setattr(module, 'verify_complete', lambda _: ({}, {}))
    class Child:
        def __init__(self, args, **kwargs):
            import fcntl
            with (tmp_path / 'controlled/driver.lock').open('a') as lock:
                with pytest.raises(BlockingIOError): fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.args, self.kwargs, self.pid, self.code, self.terminated = args, kwargs, 900+len(children), None, False
            children.append(self)
        def poll(self): return self.code
        def wait(self):
            stage = Path(self.args[self.args.index('--stage')+1]); index = int(self.args[-1])
            write_json(stage.parent / f'worker{index}.json', dict(fixture=True)); self.code = 0; return 0
        def terminate(self): self.terminated = True; self.code = -15
    monkeypatch.setattr(module.subprocess, 'Popen', Child)
    return dict(path=path, children=children, log=log, writes=writes, Child=Child)


def test_scheduler_holds_global_lock_and_routes_only_four_authorized_devices(scheduler):
    result = module.run(scheduler['path'])
    assert result['phase'] == 'requested_adaptation_complete' and not result['scientific_study_complete']
    assert len(scheduler['children']) == 9  # 1 qualification +4 selection +4 remaining
    for child, index in zip(scheduler['children'], [0,0,1,2,3,0,1,2,3], strict=True):
        env = child.kwargs['env']
        assert env['CUDA_VISIBLE_DEVICES'] == module.execution.evaluation.GPU_UUIDS[index]
        assert env['WANDB_MODE'] == 'disabled' and env['ACCELERATE_USE_CPU'] == 'false'
    with pytest.raises(RuntimeError, match='already invoked'): module.run(scheduler['path'])
    assert len(scheduler['children']) == 9


@pytest.mark.parametrize('blocked', ['gpu', 'storage', 'lock'])
def test_waiting_for_resource_does_not_launch_or_hold_global_lock(scheduler, monkeypatch, blocked):
    import fcntl
    held = None
    if blocked == 'gpu': monkeypatch.setattr(module.cq, 'gpu_processes', lambda: [dict(pid=7)])
    elif blocked == 'storage': monkeypatch.setattr(module, 'storage_gate', lambda _: dict(ready=False))
    else:
        held = (scheduler['path'].parent / 'controlled/driver.lock').open('a'); fcntl.flock(held, fcntl.LOCK_EX)
    def stop(seconds):
        assert seconds == 600 and not scheduler['children']
        if held is None:
            with (scheduler['path'].parent / 'controlled/driver.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise RuntimeError('stop fixture wait')
    monkeypatch.setattr(module.time, 'sleep', stop)
    try:
        with pytest.raises(RuntimeError, match='stop fixture wait'): module.run(scheduler['path'])
    finally:
        if held is not None: held.close()
    assert not (scheduler['path'].parent / 'stages').exists()


def test_worker_failure_stops_the_queue_without_selection_or_test_launch(scheduler, monkeypatch):
    monkeypatch.setattr(scheduler['Child'], 'wait', lambda self: setattr(self, 'code', 9) or 9)
    monkeypatch.setattr(module, 'qualify', lambda _: pytest.fail('Failed qualification was accepted'))
    with pytest.raises(RuntimeError, match='failed with 9'): module.run(scheduler['path'])
    assert len(scheduler['children']) == 1
    assert module.read(scheduler['path'].parent / 'status.json')['phase'] == 'stopped_for_inspection'
    assert not (scheduler['path'].parent / 'complete.json').exists()


@pytest.fixture
def qualification(scientific, tmp_path, monkeypatch):
    path = tmp_path / 'plan.json'; write_json(path, scientific['plan'])
    monkeypatch.setattr(module.execution, 'runtime_source', lambda *_: dict(fixture=True))
    monkeypatch.setattr(module.execution, 'verify_operation', lambda output, operation, source:
        module.read(module.execution.operation_directory(output, operation) / 'complete.json'))
    work = module.execution.operations(scientific['plan']['inventory'], 'qualification')
    stage = tmp_path / 'stages/qualification/plan.json'
    write_json(stage, dict(stage='qualification', plan_sha256=module.sha256(path), work=work, workers=1,
        selection_sha256=None, controller=dict(fixture=True)))
    jobs = {j['id']: j for j in scientific['plan']['inventory']['fits']}; operation_hashes, memory = {}, {}
    for op in work['operations']:
        job = jobs[op['job_id']]; native_dir = tmp_path / 'tasks/fits' / job['id'] / 'native'
        write_json(native_dir / 'complete.json', dict(cpu_qualification=False, scope=job['scope'], epochs=15,
            global_step=15, query_examples_received=False, selection_performed=False, frozen_parameters_unchanged=True,
            optimizer_groups=[dict(names=['private.weight'], learning_rate=.001, weight_decay=.01)],
            trainable_parameters=['private.weight'], base_parameter_sha256='fixture-base',
            snapshots=[dict(epoch=e) for e in module.execution.HORIZON], support_rows=20, support_assays=1,
            sampler_stats=dict(num_ranking_lists=1)))
        write_json(native_dir / 'fit_plan.json', dict(training=dict(seed=42, per_device_train_batch_size=12,
            gradient_accumulation_steps=16, report_to=[], bf16=True, fp16=False)))
        query_proofs = []
        for q in op['queries']:
            directory = tmp_path / 'tasks/mock_queries' / op['id'] / str(q['epoch'])
            write_json(directory / 'complete.json', dict(base_restored=True, modes_and_rng_restored=True))
            write_json(directory / 'execution.json', dict(fixture=True))
            write_json(directory / 'native.json', dict(cpu_qualification=False, world_size=1, device='cuda:0',
                bf16=True, fp16=False, parameters_unchanged=True, optimizer_created=False))
            query_proofs.append(dict(**q, artifacts=[dict(path=str(directory / n))
                for n in ('complete.json','execution.json','native.json')]))
        receipt = module.execution.operation_directory(tmp_path / 'tasks', op) / 'complete.json'
        write_json(receipt, dict(operation=op, source=dict(fixture=True), queries=query_proofs,
            base_restored=True, base_parameter_sha256='fixture-base'))
        operation_hashes[str(receipt)] = module.sha256(receipt)
        memory[op['id']] = dict(allocated_bytes=1000, reserved_bytes=2000)
    worker = stage.parent / 'worker0.json'
    write_json(worker, dict(stage_sha256=module.sha256(stage), index=0,
        gpu_uuid=module.execution.evaluation.GPU_UUIDS[0], completed=2,
        operation_sha256=operation_hashes, gpu_peak_memory=memory))
    write_json(stage.parent / 'complete.json', dict(stage_sha256=module.sha256(stage), workers_sha256={'0':module.sha256(worker)}))
    return dict(path=path, stage=stage, worker=worker, jobs=jobs, operations=work['operations'])


def test_qualification_requires_both_existing_fits_and_all_query_precision_proofs(qualification):
    result = module.qualify(qualification['path'])
    assert result['qualified'] and result['additional_fits'] == 0
    assert set(result['measures']) == {'heads','heads_cross_attention'}
    assert all(m['validation_queries'] == 15 for m in result['measures'].values())
    assert len(result['operation_sha256']) == 2


@pytest.mark.parametrize('damage', ['cpu','optimizer','horizon','frozen','batch','precision','memory'])
def test_qualification_rejects_bad_native_fit_or_query_contract(qualification, damage):
    path = qualification['path']; op = qualification['operations'][0]
    native = path.parent / 'tasks/fits' / op['job_id'] / 'native'
    if damage in ('cpu','optimizer','horizon','frozen'):
        target = native / 'complete.json'; record = module.read(target)
        if damage == 'cpu': record['cpu_qualification'] = True
        elif damage == 'optimizer': record['optimizer_groups'][0]['names'].append('encoder.weight')
        elif damage == 'horizon': record['epochs'] = 14
        else: record['frozen_parameters_unchanged'] = False
    elif damage == 'batch':
        target = native / 'fit_plan.json'; record = module.read(target); record['training']['gradient_accumulation_steps'] = 4
    elif damage == 'precision':
        target = path.parent / 'tasks/mock_queries' / op['id'] / '1/native.json'; record = module.read(target); record['bf16'] = False
    else:
        target = qualification['worker']; record = module.read(target); record['gpu_peak_memory'][op['id']]['allocated_bytes'] = 0
    write_json(target, record)
    if damage == 'memory':
        end = qualification['stage'].parent / 'complete.json'; end_record = module.read(end)
        end_record['workers_sha256']['0'] = module.sha256(target); write_json(end, end_record)
    with pytest.raises(ValueError): module.qualify(path)
    assert not (path.parent / 'qualification.json').exists()


def test_stage_cannot_omit_an_assigned_operation_or_change_device(qualification):
    path = qualification['worker']; original = module.read(path)
    for change in ('device','operation'):
        record = copy.deepcopy(original)
        if change == 'device': record['gpu_uuid'] = 'unapproved'
        else: record['operation_sha256'].pop(next(iter(record['operation_sha256'])))
        write_json(path, record)
        end = qualification['stage'].parent / 'complete.json'; end_record = module.read(end)
        end_record['workers_sha256']['0'] = module.sha256(path); write_json(end, end_record)
        with pytest.raises(ValueError, match='worker omitted'): module.verify_stage(qualification['path'], 'qualification')


def test_report_publishes_conditional_metrics_and_verifies_its_exact_coverage(scientific, tmp_path, monkeypatch):
    monkeypatch.setenv('WANDB_CACHE_DIR', str(tmp_path / 'wandb_cache'))
    path = tmp_path / 'plan.json'; write_json(path, scientific['plan'])
    write_json(tmp_path / 'controller_plan.json', dict(fixture=True))
    chosen = dict(selections=scientific['choices']); write_json(tmp_path / 'selection.json', chosen)
    monkeypatch.setattr(module, 'verify_controller', lambda _: scientific['plan'])
    monkeypatch.setattr(module.execution, 'verify_selection', lambda _: chosen)
    for regime in ('A','B'):
        view = scientific['views'][regime]
        group_path = tmp_path / f'groups_{regime}.parquet'; scientific['groups'][regime].to_parquet(group_path)
        view['groups'] = dict(path=str(group_path)); write_json(Path(scientific['plan']['models'][regime]['views']['path']), view)
    catalog, fits, hashes = {}, {}, {}
    for index, (key, query) in enumerate(scientific['queries'].items()):
        target = tmp_path / 'queries' / f'{index}.json'; write_json(target, query)
        catalog[key] = dict(artifacts=[dict(path=str(target))])
    for key, native in scientific['fits'].items():
        target = tmp_path / 'fits' / key / 'complete.json'; write_json(target, native); fits[key] = str(target)
    for stage in ('selection','remaining'):
        for op in module.execution.operations(scientific['plan']['inventory'], stage, scientific['choices'] if stage == 'remaining' else None)['operations']:
            target = module.execution.operation_directory(tmp_path / 'tasks', op) / 'complete.json'
            write_json(target, dict(fixture=True)); hashes[str(target)] = module.sha256(target)
    monkeypatch.setattr(module, 'query_catalog', lambda _: (catalog, fits, hashes))
    logged, artifacts = [], []
    log = SimpleNamespace(log=logged.append, log_artifact=artifacts.append)
    report_path = module.report(path, log)
    write_json(tmp_path / 'complete.json', dict(phase='requested_adaptation_complete', plan_sha256=module.sha256(path),
        report=str(report_path), report_sha256=module.sha256(report_path), scientific_study_complete=False,
        additional_seed_fits=0, parameter_search=False))
    _, complete = module.verify_complete(path)
    assert complete['new_support_fits'] == len(scientific['fits']) and len(artifacts) == 1
    text = (tmp_path / 'report/REPORT.md').read_text()
    assert 'Native macro Spearman' in text and '| Test base | Test adapted |' in text
    assert 'not independent deployment validation' in text
    assert logged[0]['adaptation/additional_seeds'] == 0
    with pytest.raises(RuntimeError, match='report already exists'): module.report(path, log)
    (tmp_path / 'report/summary.json').write_text('{}')
    with pytest.raises(ValueError, match='artifact changed'): module.verify_complete(path)
