"""Native tiny-model diagnostics; scientific gates and scheduler tested separately."""
import copy
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from experiments.paired_generalization import requested_objectives as module
from experiments.paired_generalization import control_plan
from experiments.paired_generalization.run import sha256, write_json
from reward_model.training.config import load_reward_training_config
from test_paired_generalization_requested_evaluation import native
from test_paired_generalization_requested_geometry import prepared as geometry_prepared


@pytest.fixture
def prepared(geometry_prepared, monkeypatch):
    value = geometry_prepared; root = value['root']
    monkeypatch.setattr(module, 'ROOT', root); monkeypatch.setattr(module, 'WORK', root)
    monkeypatch.setattr(module, 'software', lambda: dict(source_sha256={}, packages={}))
    probe_path = root / 'protocol/plan.json'; probe = module.read(probe_path)
    bases = probe['probes']['geometry']['resolved_bases']
    folder = root / 'objective_batches'; folder.mkdir()
    frozen = dict(regimes={r: dict(batch_pairs=32) for r in ('A', 'B')})
    write_json(folder / 'complete.json', frozen)
    verified = dict(complete_sha256=sha256(folder / 'complete.json'), batches_verified=128,
        collators={r: dict(fixture=True) for r in ('A', 'B')})
    write_json(folder / 'reconstruction_verification.json', verified)
    candidates, slots = {}, []
    for regime, base in bases.items():
        config = load_reward_training_config(base['job']['config_path'])
        candidates[base['job']['id']] = dict(nominal_step_rates=module.execution.protocol.step_rates(asdict(config.training)))
        slots.append(dict(regime=regime, seed=42, selected_job_id=base['job']['id'],
            selected_checkpoint=base['checkpoint'], candidate_job_ids=[base['job']['id']], batch_pairs=32))
        for role in module.ROLES:
            write_json(folder / f'{regime}_{role}.json', dict(regime=regime, role=role,
                selected_batches=32, batches=[dict(pair_index=i) for i in range(32)]))
    probe['probes']['objective'] = dict(seeds=[42], checkpoint_slots=slots,
        scopes=list(module.execution.protocol.SCOPES), precision='float32', autocast=False,
        allow_tf32=False, dropout=False, input_sha256={}, resolved_bases=bases, candidates=candidates)
    write_json(probe_path, probe)
    monkeypatch.setattr(module.reconstruction, 'inputs', lambda: (folder, module.read(folder / 'complete.json')))
    path = module.prepare(value['analyses']); module.verify_plan(path)
    value.update(objective_path=path, objective_plan=module.read(path), objective_folder=folder, probe_path=probe_path)
    return value


def test_plan_is_exact_two_full_references_64_pairs_and_no_test_partition(prepared):
    plan = prepared['objective_plan']
    assert plan['seed'] == 42 and plan['learning_rate_multiplier'] == 1.
    assert plan['batch_pairs_per_model'] == 32 and len(plan['models']) == 2
    assert plan['new_fits'] == 0 and not plan['test_data_read']
    for r, base in plan['models'].items():
        assert base['job']['id'] == f'full__{r}__seed42__lr1p0'
        assert set(base['contexts']) == {'train', 'cold_validation'}
        assert all('test_examples' not in c['data']['cache'] for c in base['contexts'].values())
        assert set(base['rules']) == {'best_macro'}


@pytest.mark.parametrize('damage', ['seed', 'rate', 'slot', 'pair_count', 'pair_order', 'checkpoint', 'precision', 'analysis'])
def test_plan_rejects_changed_scope_or_prerequisite(prepared, damage):
    probe = module.read(prepared['probe_path']); objective = probe['probes']['objective']
    if damage == 'seed': objective['seeds'] = [42, 43]
    elif damage == 'rate': next(iter(objective['candidates'].values()))['nominal_step_rates']['interaction'] *= 3
    elif damage == 'slot': objective['checkpoint_slots'][0]['candidate_job_ids'].append('alternate')
    elif damage == 'checkpoint': objective['resolved_bases']['A']['checkpoint']['id'] = 'different'
    elif damage == 'precision': objective['allow_tf32'] = True
    elif damage == 'analysis': prepared['analyses'][0].write_text('{}')
    else:
        path = prepared['objective_folder'] / 'A_train.json'; context = module.read(path)
        if damage == 'pair_count': context['selected_batches'] = 2
        else: context['batches'].reverse()
        write_json(path, context)
    write_json(prepared['probe_path'], probe)
    with pytest.raises(ValueError): module.verify_plan(prepared['objective_path'])


@pytest.fixture
def runtime(native, tmp_path, monkeypatch):
    """Two-pair CPU fixture exercises real checkpoints and gradients, not Tier3 acceptance."""
    monkeypatch.setattr(module, 'ROOT', tmp_path); monkeypatch.setattr(module, 'WORK', tmp_path)
    path = tmp_path / 'runtime/plan.json'
    original, examples, contexts, collator, _, evidence = module.execution.qualification_fixture()
    job = native['jobs'][0]; evaluation_path = module.evaluation.prepare(job['id'])
    base = module.read(evaluation_path)
    data = {}
    for role, dataset in examples.items():
        cache = tmp_path / 'runtime_data' / (role + '_examples'); dataset.save_to_disk(str(cache))
        source = cache.parent / (role + '.parquet'); dataset.to_pandas().to_parquet(source, index=False)
        context_path = cache.parent / (role + '.json'); write_json(context_path, contexts[role])
        data[role] = dict(path=str(context_path), sha256=sha256(context_path), data=dict(source=str(source),
            source_sha256=sha256(source), cache=str(cache), cache_sha256=control_plan.token_cache_hashes(cache)))
    reconstruction = tmp_path / 'runtime_data/reconstruction.json'
    write_json(reconstruction, dict(contexts={'A_' + role: record for role, record in evidence.items()}))
    config = load_reward_training_config(job['config_path'])
    info = dict(job=job, checkpoint=base['checkpoint'], rules=base['rules'], contexts=data,
        reconstruction=str(reconstruction), collator_provenance=dict(cpu_fixture=True),
        nominal_rates=module.execution.protocol.step_rates(asdict(config.training)))
    plan = dict(models={'A': info, 'B': dict(info, job=dict(job, regime='B'))}, interpretation='CPU fixture')
    write_json(path, plan)
    monkeypatch.setattr(module, 'verify_plan', module.read)
    monkeypatch.setattr(module.reconstruction, 'reference_collator', lambda _: (collator, dict(cpu_fixture=True)))
    return dict(path=path, plan=plan, model=original, examples=examples, contexts=contexts,
        collator=collator, training=asdict(config.training), evidence=evidence, root=tmp_path)


def test_actual_saved_model_qualification_all_pairs_and_reload(runtime, monkeypatch):
    from reward_model.training.trainer import RewardModelTrainer
    monkeypatch.setattr(RewardModelTrainer, 'train', lambda *_: pytest.fail('No diagnostic fitting'))
    monkeypatch.setattr(RewardModelTrainer, 'create_optimizer', lambda *_: pytest.fail('No diagnostic optimizer'))
    path = runtime['path']; weights = Path(runtime['plan']['models']['A']['checkpoint']['path']) / 'pytorch_model.bin'
    original_hash = sha256(weights)
    module.worker(path, 'A', 0, cpu=True)
    done, measured, qualified = module.verify_worker(path, 'A', cpu=True)
    assert qualified['repetitions'] == 2 and qualified['repeatability']['passed']
    assert qualified['forward_dtype_checks'] and qualified['memory'] is None
    assert qualified['cpu_qualification'] and not qualified['production_gpu_qualified']
    assert measured['summary']['pairs'] == 2 and not measured['summary']['independent_batch_intervals']
    assert done['new_fits'] == 0 and not done['test_data_read']
    assert sha256(weights) == original_hash
    monkeypatch.setattr(module.execution, 'execute_pairs', lambda *_args, **_kw: pytest.fail('Completed pairs were rerun'))
    module.worker(path, 'A', 0, cpu=True)
    with pytest.raises(ValueError, match='scope or device'): module.verify_worker(path, 'A')
    (path.parent / 'A/qualification/repeat_0.json').write_text('{}')
    with pytest.raises(ValueError, match='Repeated measurement'): module.verify_worker(path, 'A', cpu=True)


def test_runtime_reports_small_transfer_sign_changes_without_rate_search(runtime):
    args = runtime
    result = module.qualify(args['model'], args['examples'], args['contexts'], args['collator'],
        args['training'], args['evidence'], args['root'] / 'qualification',
        dict(parameter_sha256=module.execution.parameter_digest(args['model'])), cpu=True)
    first = module.read(args['root'] / 'qualification/repeat_0.json'); second = copy.deepcopy(first)
    for record, change in ((first, -1e-9), (second, 1e-9)):
        value = record['lookahead']['R']['scopes']['shared']['transfer']['B']
        value['absolute_loss_change'] = change; value['after'] = value['before'] + change
    compared = module.repeatability(first, second)
    assert compared['passed'] and 'R/shared/B' in compared['transfer_sign_changes']
    assert compared['first_pair_step_precision']['R']['shared']['targets']['B']['change_within_observed_repeat_drift']
    assert result['parameters_unchanged'] and not args['model']._forward_hooks


@pytest.mark.parametrize('failure', ['drift', 'dtype', 'tokens', 'collator', 'partial'])
def test_bad_qualification_or_inputs_never_start_scientific_pair_loop(runtime, monkeypatch, failure):
    path = runtime['path']; info = runtime['plan']['models']['A']
    if failure == 'drift':
        original = module.execution.probes.measure; calls = []
        def drift(*args, **kwargs):
            measured = original(*args, **kwargs); calls.append(1)
            if len(calls) == 2: measured['training_losses']['R'] += 1.
            return measured
        monkeypatch.setattr(module.execution.probes, 'measure', drift)
    elif failure == 'dtype':
        from reward_model import model as api
        original = api.load_reward_model
        monkeypatch.setattr(api, 'load_reward_model', lambda *args, **kwargs: original(*args, **kwargs).double())
    elif failure == 'tokens':
        next(Path(info['contexts']['train']['data']['cache']).glob('*.arrow')).write_bytes(b'changed')
    elif failure == 'collator':
        monkeypatch.setattr(module.reconstruction, 'reference_collator', lambda _: (runtime['collator'], {}))
    else: (path.parent / 'A').mkdir()
    monkeypatch.setattr(module.execution, 'execute_pairs', lambda *_args, **_kw: pytest.fail('Invalid qualification reached pair loop'))
    with pytest.raises((ValueError, RuntimeError)): module.worker(path, 'A', 0, cpu=True)
    assert not (path.parent / 'A/complete.json').exists()
    if failure == 'drift':
        assert not module.read(path.parent / 'A/qualification/complete.json')['repeatability']['passed']


def test_first_pair_failure_restores_hooks_precision_and_weights(runtime, monkeypatch):
    model = runtime['model']; before = module.execution.parameter_digest(model)
    precision = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    monkeypatch.setattr(module.execution.probes, 'measure', lambda *_args, **_kw: (_ for _ in ()).throw(RuntimeError('forward failure')))
    with pytest.raises(RuntimeError, match='forward failure'):
        module.qualify(model, runtime['examples'], runtime['contexts'], runtime['collator'],
            runtime['training'], runtime['evidence'], runtime['root'] / 'failure', dict(parameter_sha256=before), cpu=True)
    assert module.execution.parameter_digest(model) == before and not model._forward_hooks
    assert precision == (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)


@pytest.fixture
def scheduler(runtime, monkeypatch):
    monkeypatch.setattr(module.evaluation.control_queue, 'runtime_guard', lambda: None)
    monkeypatch.setattr(module.evaluation.control_queue, 'gpu_processes', lambda: [])
    monkeypatch.setattr(module.evaluation.control_queue, 'storage_gate', lambda: dict(ready=True))
    calls = []
    def verify(path, regime, **_):
        assert (path.parent / regime / 'complete.json').exists()
        return {}, dict(summary=dict(simulated=True)), dict(simulated=True)
    monkeypatch.setattr(module, 'verify_worker', verify)
    class Child:
        def __init__(self, command, **kwargs):
            self.command = command; self.kwargs = kwargs; self.pid = 900 + len(calls); self.code = None; self.terminated = False
            calls.append(self)
            import fcntl
            with (runtime['root'] / 'controlled/driver.lock').open('a') as lock:
                with pytest.raises(BlockingIOError): fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def wait(self):
            self.code = 0; regime = self.command[self.command.index('--regime') + 1]
            write_json(runtime['path'].parent / regime / 'complete.json', dict(simulated=True)); return 0
        def poll(self): return self.code
        def terminate(self): self.terminated = True
    monkeypatch.setattr(module.subprocess, 'Popen', Child)
    runtime.update(calls=calls, Child=Child); return runtime


def test_scientific_scheduler_uses_only_authorized_devices_and_global_lock(scheduler):
    result = module.run(scheduler['path'])
    assert len(scheduler['calls']) == 2 and set(result['results']) == {'A', 'B'}
    assert result['new_fits'] == 0 and not result['test_data_read'] and not result['scientific_study_complete']
    for i, child in enumerate(scheduler['calls']):
        assert child.kwargs['env']['CUDA_VISIBLE_DEVICES'] == module.evaluation.GPU_UUIDS[i]
        assert child.kwargs['env']['WANDB_MODE'] == 'disabled'
        assert child.command[-1] == str(i)


@pytest.mark.parametrize('gate', ['gpu', 'storage', 'lock', 'partial'])
def test_resource_or_partial_state_starts_no_workers(scheduler, monkeypatch, gate):
    import fcntl
    handle = (scheduler['root'] / 'controlled/driver.lock').open('a')
    if gate == 'gpu': monkeypatch.setattr(module.evaluation.control_queue, 'gpu_processes', lambda: [dict(pid=99)])
    elif gate == 'storage': monkeypatch.setattr(module.evaluation.control_queue, 'storage_gate', lambda: dict(ready=False))
    elif gate == 'lock': fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    else: (scheduler['path'].parent / 'B').mkdir()
    try:
        if gate == 'partial':
            with pytest.raises(RuntimeError, match='Partial objective'): module.run(scheduler['path'])
        else: assert module.run(scheduler['path'])['phase'].startswith('waiting_for_')
    finally: handle.close()
    assert not scheduler['calls']


def test_launch_failure_cleans_only_owned_child(scheduler, monkeypatch):
    def launch(*args, **kwargs):
        if scheduler['calls']: raise OSError('launch failure')
        return scheduler['Child'](*args, **kwargs)
    monkeypatch.setattr(module.subprocess, 'Popen', launch)
    with pytest.raises(OSError, match='launch failure'): module.run(scheduler['path'])
    assert len(scheduler['calls']) == 1 and scheduler['calls'][0].terminated
    assert not (scheduler['path'].parent / 'complete.json').exists()
