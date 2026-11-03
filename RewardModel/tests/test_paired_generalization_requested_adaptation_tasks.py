"""Frozen task accounting and actual small support/query runs with base restoration."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from experiments.paired_generalization import requested_adaptation_tasks as module
from experiments.paired_generalization import adaptation_query_qualification as qualification
from experiments.paired_generalization.qualify_controls import make_model
from experiments.paired_generalization.run import write_json
from reward_model.training.config import RewardTrainingConfigBundle, RewardTrainingDataConfig, RewardTrainerConfig
from test_paired_generalization_adaptation_protocol import fixture_protocol


@pytest.fixture
def frozen():
    primary, records, groups = fixture_protocol()
    plan = module.protocol.build_protocol(primary, records, groups)
    plan['seeds'] = [42]
    for cell in plan['cells']: cell['total_fits'] //= 3
    plan['accounting']['support_fits'] //= 3
    return plan, records


def test_inventory_preserves_each_unit_and_deduplicates_unchanged_baselines(frozen):
    plan, records = frozen; result = module.inventory(plan, records)
    assert len(result['fits']) == plan['accounting']['support_fits']
    assert len({j['id'] for j in result['fits']}) == len(result['fits'])
    assert not any(j['regime'] == 'A' and j['kind'] == 'within' for j in result['fits'])
    for job in result['fits']:
        assert job['seed'] == 42
        assert job['selection_fit'] == (job['regime'] == 'B' and 'validation' in job['query_roles'])
        task = records[job['regime']]['tasks'][job['task_id']]
        support = records[job['regime']]['views'][task['support_view']]['group_ids']
        query = records[job['regime']]['views'][task['query_view']]['group_ids']
        assert not set(support) & set(query)
        for role in job['query_roles']:
            found = [b for b in result['baselines'] if (b['regime'], b['task_id'], b['role']) ==
                (job['regime'], job['baseline_task'], role)]
            assert len(found) == 1
    assert len([b for b in result['baselines'] if b['task_id'] == 'across_k0']) == 4
    counts = result['accounting']
    assert counts['maximum_adapter_snapshots'] == 15*counts['selection_fits'] + 2*counts['remaining_fits']
    assert counts['selection_queries'] == 15*counts['selection_fits']


@pytest.mark.parametrize('damage', ['seeds', 'count', 'duplicate', 'role', 'support', 'baseline'])
def test_inventory_rejects_expansion_or_crossed_task_bindings(frozen, damage):
    plan, records = frozen; cell = plan['cells'][0]
    if damage == 'seeds': plan['seeds'] = [42, 43]
    elif damage == 'count': cell['total_fits'] += 1
    elif damage == 'duplicate': cell['task_bindings']['A']['across'] *= 2
    elif damage == 'role': cell['selection_role'] = 'test'
    elif damage == 'support': records['A']['tasks']['across_k1']['k'] = 2
    else: records['A']['tasks']['across_k0']['query_view'] = 'across_validation'
    with pytest.raises(ValueError): module.inventory(plan, records)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', ''); monkeypatch.setenv('ACCELERATE_USE_CPU', 'true')
    monkeypatch.setenv('WANDB_MODE', 'disabled'); torch.set_num_threads(2)
    views = qualification.synthetic_views(tmp_path / 'views', 'B')
    model = make_model(None, {})
    config = RewardTrainingConfigBundle(model=model.config,
        data=RewardTrainingDataConfig(train_parquet_path='synthetic_support', val_parquet_path='synthetic_query',
            test_parquet_path='synthetic_test', tokenized_dataset_dir='synthetic_cache',
            enforce_binary_label_threshold=False, evaluation_ranking_partitions=3),
        training=RewardTrainerConfig(output_dir='unused', training_mode='single_gpu', seed=42,
            num_train_epochs=2, per_device_train_batch_size=2, per_device_eval_batch_size=2,
            gradient_accumulation_steps=2, learning_rate=.001, logging_steps=1, report_to=[],
            evaluate_test=False, protein_shuffle_sensitivity=False))
    return module.TaskRuntime(model, views, config, tmp_path / 'tasks', dict(cpu_fixture=True), cpu=True)


def job_for(runtime, task_id='across_k1', scope='heads'):
    task = runtime.views.record['tasks'][task_id]
    roles = ['validation', 'test'] if task['kind'] == 'across' else [task['probe_role']]
    cell = f"{task['kind']}__{scope}__k{task['k']}"
    return dict(id=f'B__{cell}__{task_id}', regime='B', seed=42, cell_id=cell, task_id=task_id,
        kind=task['kind'], scope=scope, k=task['k'], query_roles=roles,
        selection_fit='validation' in roles, baseline_task=task['baseline_task'])


def test_real_support_queries_both_scopes_and_snapshot_hashes_reset_base(runtime, monkeypatch):
    base = runtime.base.digest
    baseline = runtime.score('across_k0', 'validation', 0)
    assert baseline['base_restored'] and not baseline['selection_records']
    for scope in ('heads', 'heads_cross_attention'):
        job = job_for(runtime, scope=scope); receipt = runtime.fit_task(job, [1, 2])
        assert receipt['base_restored'] and module.parameter_digest(runtime.model) == base
        done = runtime.verify_fit(job, [1, 2])
        assert done['query_examples_received'] is False and done['selection_performed'] is False
        for epoch in (1, 2):
            result = runtime.score(job['task_id'], 'validation', epoch, job=job)
            assert result['selection_records'] and module.parameter_digest(runtime.model) == base
        result = runtime.score(job['task_id'], 'test', 1, job=job)
        assert result['source']['query_role'] == 'test' and result['source']['epoch'] == 1
        snapshot = done['snapshots'][-1]; path = runtime.fit_directory(job) / snapshot['file']
        combined = runtime.base.adapter_digest(path, snapshot['sha256'], scope=scope, epoch=2, step=snapshot['step'])
        module.fit.restore_adapter(runtime.model, path, snapshot['sha256'])
        assert combined == module.parameter_digest(runtime.model) == done['final_parameter_sha256']
        runtime.base.restore()
    monkeypatch.setattr(module.fit, 'fit_support', lambda *_args, **_kw: pytest.fail('Completed support fit was repeated'))
    monkeypatch.setattr(module.query, 'evaluate_query', lambda *_args, **_kw: pytest.fail('Completed query was repeated'))
    runtime.fit_task(job, [1, 2]); runtime.score(job['task_id'], 'test', 1, job=job)
    assert module.parameter_digest(runtime.model) == base


def test_separate_within_unit_tasks_never_carry_parameters_or_supports(runtime):
    tasks = [identity for identity, task in runtime.views.record['tasks'].items()
        if task['kind'] == 'within' and task['k'] == 1 and task['probe_role'] == 'test']
    assert len(tasks) >= 2
    supports = []
    for identity in tasks[:2]:
        job = job_for(runtime, identity); supports.append(set(runtime.views.record['views'][runtime.views.record['tasks'][identity]['support_view']]['group_ids']))
        runtime.fit_task(job, [1, 2]); result = runtime.score(identity, 'test', 1, job=job)
        assert result['units'] == 1 and module.parameter_digest(runtime.model) == runtime.base.digest
    assert not supports[0] & supports[1]


def test_selection_fit_cannot_omit_epochs_and_k0_cannot_fit(runtime):
    job = job_for(runtime)
    with pytest.raises(ValueError, match='every positive'): runtime.fit_task(job, [2])
    with pytest.raises(ValueError, match='k=0'): runtime.score('across_k0', 'validation', 1)
    with pytest.raises(ValueError, match='k=0'): runtime.score('across_k0', 'test', 0, job=job)
    assert not runtime.output.exists()


@pytest.fixture
def scored(runtime):
    job = job_for(runtime); runtime.fit_task(job, [1, 2]); runtime.score(job['task_id'], 'validation', 1, job=job)
    return runtime, job


@pytest.mark.parametrize('damage', ['adapter', 'fit_config', 'fit_receipt', 'metric', 'selection_rows', 'native_weights', 'native_payload'])
def test_changed_adapter_or_query_cannot_supply_selection_evidence(scored, damage):
    runtime, job = scored
    output = runtime.query_directory(job['task_id'], 'validation', 1, job['scope'])
    fit_directory = runtime.fit_directory(job)
    if damage == 'adapter': (fit_directory / 'epoch01.pt').write_bytes(b'changed')
    elif damage == 'fit_config':
        path = fit_directory / 'fit_plan.json'; record = module.read(path); record['training']['seed'] = 43; write_json(path, record)
    elif damage == 'fit_receipt':
        path = fit_directory.parent / 'receipt.json'; record = module.read(path); record['source'] = {}; write_json(path, record)
    elif damage == 'native_payload':
        manifest = module.read(Path(module.read(output / 'complete.json')['prediction_manifest']))
        payload = Path(module.read(output / 'complete.json')['prediction_manifest']).parent / manifest['predictions_file']
        payload.write_bytes(b'changed')
    elif damage == 'native_weights':
        path = output / 'native/complete.json'; record = module.read(path); record['parameter_sha256'] = 'different'; write_json(path, record)
    else:
        path = output / 'complete.json'; record = module.read(path)
        if damage == 'metric': record['pathways']['ranking_score']['macro_spearman'] = -99.
        else: record['selection_records'].pop()
        write_json(path, record)
    with pytest.raises(ValueError): runtime.verify_query(job['task_id'], 'validation', 1, job=job)


def test_fit_failure_restores_base_and_preserves_partial_work(runtime, monkeypatch):
    job = job_for(runtime); original = module.fit.fit_support
    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError('failure after native support completion')
    monkeypatch.setattr(module.fit, 'fit_support', fail)
    with pytest.raises(RuntimeError, match='failure after native'): runtime.fit_task(job, [1, 2])
    assert module.parameter_digest(runtime.model) == runtime.base.digest
    assert (runtime.fit_directory(job) / 'complete.json').exists()
    assert not (runtime.fit_directory(job).parent / 'receipt.json').exists()
    monkeypatch.setattr(module.fit, 'fit_support', lambda *_args, **_kw: pytest.fail('Uninspected work was restarted'))
    with pytest.raises(FileNotFoundError): runtime.fit_task(job, [1, 2])


def test_frozen_parameter_drift_is_not_hidden_by_reset(runtime):
    name = next(n for n in dict(runtime.model.named_parameters()) if n.startswith('protein_encoder.'))
    with torch.no_grad(): dict(runtime.model.named_parameters())[name].add_(1.)
    with pytest.raises(RuntimeError, match='frozen base state'): runtime.base.restore()


def test_partial_query_and_wrong_task_binding_are_rejected(runtime):
    job = job_for(runtime); job['seed'] = 43
    with pytest.raises(ValueError, match='binding'): runtime.fit_task(job, [1, 2])
    path = runtime.query_directory('across_k0', 'validation', 0, None); path.mkdir(parents=True)
    with pytest.raises(RuntimeError, match='Partial adaptation query'): runtime.score('across_k0', 'validation', 0)


def test_scientific_runtime_refuses_cpu_or_unapproved_device(runtime, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    config = replace(runtime.config, training=replace(runtime.config.training,
        num_train_epochs=15, per_device_train_batch_size=12, gradient_accumulation_steps=16))
    with pytest.raises(ValueError, match='authorized Atum GPU'):
        module.TaskRuntime(runtime.model, runtime.views, config, runtime.output, dict(fixture=True))
