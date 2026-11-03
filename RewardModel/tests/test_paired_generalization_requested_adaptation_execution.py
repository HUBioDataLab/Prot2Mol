"""Stage separation, exact task accounting, and native operation receipts."""
import copy
from pathlib import Path

import pytest

from experiments.paired_generalization import requested_adaptation_execution as module
from experiments.paired_generalization.run import write_json
from test_paired_generalization_requested_adaptation_tasks import frozen, runtime, job_for


def selected(frozen, epoch=3):
    return {c['id']: dict(cell_id=c['id'], status='selected', selected=dict(epoch=epoch))
        for c in frozen[0]['cells']}


def test_stages_preserve_all_fits_and_queries_with_no_extra_seed(frozen):
    inventory = module.tasks.inventory(*frozen)
    prefix = module.operations(inventory, 'qualification')['operations']
    first = module.operations(inventory, 'selection')['operations']
    second = module.operations(inventory, 'remaining', selected(frozen))['operations']
    assert len(prefix) == 2 and all(op in first for op in prefix)
    assert not {o['id'] for o in first} & {o['id'] for o in second}
    trained = [o['job_id'] for o in first+second if o['fit']]
    assert len(trained) == len(set(trained)) == inventory['accounting']['support_fits']
    assert set(trained) == {j['id'] for j in inventory['fits']}
    assert all(j['seed'] == 42 for j in inventory['fits'])
    assert all(o['regime'] == 'B' and all(q['role'] == 'validation' for q in o['queries']) for o in first)
    assert sum(len(o['queries']) for o in first if o['type'] == 'adapted') == inventory['accounting']['selection_queries']
    assert sum(len(o['queries']) for o in second if o['type'] == 'adapted') == inventory['accounting']['remaining_selected_queries']
    assert sum(o['type'] == 'baseline' for o in first+second) == inventory['accounting']['baseline_queries']
    assert all(o['save_epochs'] == module.HORIZON for o in first if o['fit'])
    assert all(o['save_epochs'] == [3, 15] for o in second if o['fit'])
    assert all(o['save_epochs'] == module.HORIZON for o in second if o['type'] == 'adapted' and not o['fit'])


def test_remaining_stage_requires_selection_and_keeps_full_training_horizon(frozen):
    inventory = module.tasks.inventory(*frozen)
    with pytest.raises(ValueError, match='frozen validation'): module.operations(inventory, 'remaining')
    result = module.operations(inventory, 'remaining', selected(frozen, epoch=15))
    assert all(o['save_epochs'] == [15] for o in result['operations'] if o['fit'])
    for epoch in (0, 16, True, 1.5):
        with pytest.raises(ValueError, match='epoch'):
            module.operations(inventory, 'remaining', selected(frozen, epoch))


def test_unselectable_cells_do_not_use_test_or_final_epoch_fallback(frozen):
    inventory = module.tasks.inventory(*frozen); choices = selected(frozen)
    for value in choices.values(): value.update(status='not_selectable', selected=None)
    result = module.operations(inventory, 'remaining', choices)
    assert all(o['type'] == 'baseline' for o in result['operations'])
    assert len(result['unavailable']) == len(inventory['fits'])
    assert not any(o['fit'] for o in result['operations'])


def test_prefix_cannot_gain_a_seed_scope_or_extra_task(frozen):
    inventory = module.tasks.inventory(*frozen)
    extra = copy.deepcopy(next(j for j in inventory['fits'] if j['selection_fit'] and j['k'] == 1 and j['kind'] == 'across'))
    extra['id'] += '_different'; inventory['fits'].append(extra)
    with pytest.raises(ValueError, match='exactly the two'): module.operations(inventory, 'qualification')


@pytest.fixture
def operation(runtime):
    job = job_for(runtime)
    inventory = dict(fits=[job], baselines=[])
    op = dict(id='selection_fit__'+job['id'], type='adapted', regime='B', job_id=job['id'],
        fit=True, save_epochs=[1, 2], queries=[dict(role='validation', epoch=e) for e in (1, 2)])
    return inventory, op


def test_actual_native_operation_pins_all_snapshots_and_query_bytes(runtime, operation, monkeypatch):
    inventory, op = operation
    done = module.execute_operation(runtime, op, inventory)
    assert done == module.verify_operation(runtime.output, op, runtime.source, cpu=True)
    assert len(done['fit_artifacts']) == 5 and len(done['queries']) == 2
    assert all(len(q['artifacts']) == 5 for q in done['queries'])
    assert done['base_restored'] and done['base_parameter_sha256'] == runtime.base.digest
    monkeypatch.setattr(runtime, 'fit_task', lambda *_: pytest.fail('Completed fit repeated'))
    monkeypatch.setattr(runtime, 'score', lambda *_args, **_kwargs: pytest.fail('Completed query repeated'))
    assert module.execute_operation(runtime, op, inventory) == done


@pytest.mark.parametrize('damage', ['payload', 'adapter', 'execution', 'source', 'cpu', 'query_epoch'])
def test_operation_cannot_be_used_after_evidence_changes(runtime, operation, damage):
    inventory, op = operation; done = module.execute_operation(runtime, op, inventory)
    if damage in ('payload', 'adapter', 'execution'):
        item = done['fit_artifacts'][-1] if damage == 'adapter' else done['queries'][0]['artifacts'][4 if damage == 'payload' else 1]
        Path(item['path']).write_bytes(b'changed')
    else:
        path = module.operation_directory(runtime.output, op) / 'complete.json'
        changed = module.read(path)
        if damage == 'source': changed['source'] = dict(wrong=True)
        elif damage == 'cpu': changed['cpu_qualification'] = False
        else: changed['queries'][0]['epoch'] = 15
        write_json(path, changed)
    with pytest.raises(ValueError): module.verify_operation(runtime.output, op, runtime.source, cpu=True)


def test_partial_operation_is_preserved_and_not_retried(runtime, operation, monkeypatch):
    inventory, op = operation
    monkeypatch.setattr(runtime, 'score', lambda *_args, **_kw: (_ for _ in ()).throw(RuntimeError('query failed')))
    with pytest.raises(RuntimeError, match='query failed'): module.execute_operation(runtime, op, inventory)
    assert (module.operation_directory(runtime.output, op) / 'invocation.json').exists()
    assert (runtime.fit_directory(inventory['fits'][0]) / 'complete.json').exists()
    with pytest.raises(RuntimeError, match='Partial adaptation operation'): module.execute_operation(runtime, op, inventory)


def test_runtime_source_does_not_bind_gpu_so_existing_fits_can_be_reused(tmp_path):
    path = tmp_path / 'plan.json'
    write_json(path, dict(models=dict(B=dict(checkpoint='fixed')), software=dict(source='fixed')))
    value = module.runtime_source(path, 'B')
    assert value['seed'] == 42 and value['regime'] == 'B' and 'gpu_uuid' not in value
    assert value['plan_sha256'] == module.sha256(path)


@pytest.mark.parametrize('index', [-1, 4, True])
def test_worker_refuses_wrong_device_index_before_reading_inputs(index):
    with pytest.raises(ValueError, match='GPU worker index'): module.worker('none', 'none', index)


def test_worker_refuses_local_execution_or_unapproved_gpu(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3')
    with pytest.raises(RuntimeError, match='authorized Atum GPU'): module.worker('none', 'none', 0)


@pytest.fixture
def selection_fixture(frozen, tmp_path, monkeypatch):
    adaptation, records = frozen; inventory = module.tasks.inventory(adaptation, records)
    plan = dict(adaptation=adaptation, inventory=inventory)
    path = tmp_path / 'plan.json'; write_json(path, plan)
    monkeypatch.setattr(module, 'verify_plan', lambda _path: plan)
    monkeypatch.setattr(module, 'runtime_source', lambda *_: dict(fixture=True))
    monkeypatch.setattr(module, 'software', lambda: dict(fixture=True))
    monkeypatch.setattr(module, 'verify_operation', lambda output, operation, source:
        module.read(module.operation_directory(output, operation) / 'complete.json'))
    jobs = {j['id']: j for j in inventory['fits']}; cells = {c['id']: c for c in adaptation['cells']}
    for operation in module.operations(inventory, 'selection')['operations']:
        directory = module.operation_directory(tmp_path / 'tasks', operation); queries = []
        if operation['type'] == 'adapted':
            job = jobs[operation['job_id']]; task = records['B']['tasks'][job['task_id']]
            view = task['validation_view'] if job['kind'] == 'across' else task['query_view']
            groups = set(records['B']['views'][view]['group_ids'])
            eligible = {g: u for g, u in cells[job['cell_id']]['validation_groups'].items() if g in groups}
            for request in operation['queries']:
                rows = [dict(regime='B', seed=42, role='validation', kind=job['kind'], scope=job['scope'],
                    k=job['k'], epoch=request['epoch'], group_id=g, unit=u,
                    rho=.8 if request['epoch'] in (2, 3) else .1) for g, u in eligible.items()]
                target = directory / f"query{request['epoch']}.json"
                write_json(target, dict(source=dict(epoch=request['epoch']), selection_records=rows))
                queries.append(dict(task_id=job['task_id'], scope=job['scope'], **request,
                    artifacts=[module.artifact(target, tmp_path / 'tasks')]))
        write_json(directory / 'complete.json', dict(operation=operation, queries=queries))
    return path


def test_selection_freeze_uses_all_b_validation_epochs_and_earliest_tie(selection_fixture):
    path = selection_fixture; target = module.freeze_selection(path)
    saved = module.read(target)
    assert all(s['selected']['epoch'] == 2 for s in saved['selections'].values())
    assert not saved['test_queries_used'] and saved['additional_seed_fits'] == 0
    assert module.verify_selection(path) == saved
    assert all(len(s['candidates']) == 15 for s in saved['selections'].values())


@pytest.mark.parametrize('damage', ['test_role', 'wrong_task', 'missing_assay', 'missing_epoch', 'wrong_seed'])
def test_selection_rejects_incomplete_or_other_role_seed_evidence(selection_fixture, damage):
    path = selection_fixture; plan = module.read(path)
    op = next(o for o in module.operations(plan['inventory'], 'selection')['operations'] if o['type'] == 'adapted')
    receipt = module.operation_directory(path.parent / 'tasks', op) / 'complete.json'; done = module.read(receipt)
    query = done['queries'][0]; target = Path(query['artifacts'][0]['path']); result = module.read(target)
    if damage == 'test_role': query['role'] = 'test'
    elif damage == 'wrong_task': query['task_id'] = 'wrong'
    elif damage == 'missing_epoch': done['queries'].pop()
    elif damage == 'missing_assay': result['selection_records'].pop()
    else: result['selection_records'][0]['seed'] = 43
    write_json(receipt, done); write_json(target, result)
    with pytest.raises(ValueError): module.freeze_selection(path)
    assert not (path.parent / 'selection.json').exists()


@pytest.mark.parametrize('damage', ['winner', 'receipt', 'metrics'])
def test_frozen_selection_cannot_change_before_remaining_tasks(selection_fixture, damage):
    path = selection_fixture; target = module.freeze_selection(path); chosen = module.read(target)
    receipt = next(Path(n) for n in chosen['operation_sha256'] if module.read(n)['operation']['type'] == 'adapted')
    if damage == 'winner':
        next(iter(chosen['selections'].values()))['selected']['epoch'] = 15; write_json(target, chosen)
    elif damage == 'receipt': receipt.write_bytes(b'changed')
    else:
        query = module.read(receipt)['queries'][0]['artifacts'][0]; Path(query['path']).write_bytes(b'changed')
    with pytest.raises(ValueError): module.verify_selection(path)


def test_tier3_completion_is_required_before_resolving_adaptation_inputs(monkeypatch):
    def stop(_): raise RuntimeError('Tier3 not complete')
    monkeypatch.setattr(module.protocol, 'require_tier3', stop)
    monkeypatch.setattr(module.protocol, 'verify', lambda: pytest.fail('Passed a missing Tier3 gate'))
    with pytest.raises(RuntimeError, match='Tier3 not complete'): module.build([])
