"""Fixed selection and actual tiny native inference; no scientific claims."""
import copy
import json
from pathlib import Path

import pytest

from experiments.paired_generalization import requested_evaluation as module, control_plan
from experiments.paired_generalization.run import sha256, write_json
from experiments.paired_generalization.qualify_controls import make_model
from experiments.paired_generalization.retain_controlled import selection_rules
from test_paired_generalization_checkpoint_evaluation import examples_fixture
from test_paired_generalization_control_plan import ROOT, noise_fixture
from test_paired_generalization_postselection import complete_fixture
from test_paired_generalization_requested_single_seed import prepared
import test_paired_generalization_postselection as old_fixture
import test_paired_generalization_retention as retention_fixture


@pytest.fixture
def native(tmp_path, monkeypatch, request):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('ACCELERATE_USE_CPU', 'true')
    monkeypatch.setenv('WANDB_MODE', 'disabled')
    monkeypatch.setattr(module, 'WORK', tmp_path)
    monkeypatch.setattr(module.postselection, 'WORK', tmp_path)
    model = make_model(None, getattr(request, 'param', {}))
    examples = examples_fixture(model)
    sources, caches = {}, {}
    for split in ('train', 'val', 'test'):
        source = tmp_path / 'data' / (split + '.parquet'); source.parent.mkdir(exist_ok=True)
        examples.to_pandas().to_parquet(source)
        cache = tmp_path / 'cache' / (split + '_examples'); examples.save_to_disk(str(cache))
        sources[split] = sha256(source); caches[split] = control_plan.token_cache_hashes(cache)
    primary = control_plan.build_plan(ROOT, tmp_path, noise_fixture())
    primary['tuning_jobs'] = [j for j in primary['tuning_jobs'] if j['arm'] == 'full' and j['learning_rate_multiplier'] == 1.]
    for regime in ('A', 'B'):
        primary['bases'][regime]['data'].update(tokenized_dataset_dir=str(tmp_path / 'cache'),
            **{f'{s}_parquet_path': str(tmp_path / 'data' / (s + '.parquet')) for s in sources})
    for job in primary['tuning_jobs']:
        job['config']['model'] = model.config.to_dict()
        job['config']['model'].update(protein_hidden_size=None, molecule_hidden_size=None)
        job['config']['data'] = copy.deepcopy(primary['bases'][job['regime']]['data'])
        job['config']['training'].update(num_train_epochs=15, bf16=False, fp16=False,
            dataloader_num_workers=0, protein_shuffle_sensitivity=False)
    primary.update(source_data_sha256={r: sources for r in ('A', 'B')},
        tokenized_data_sha256={r: caches for r in ('A', 'B')})
    directory = tmp_path / 'controlled'; write_json(directory / 'plan.json', primary)
    original_records = retention_fixture.records_fixture
    def history():
        return original_records() + [dict(epoch=e, step=e*10, eval_loss=1.2, eval_macro_spearman=.1) for e in range(7, 16)]
    monkeypatch.setattr(retention_fixture, 'records_fixture', history)
    old_make = old_fixture.make_run
    def fifteen(run):
        old_make(run)
        state = module.read(run / 'trainer_state.json'); state.update(epoch=15, global_step=150)
        write_json(run / 'trainer_state.json', state)
        args = retention_fixture.snapshot_arguments(run)
        args.update(phase='selected', step=150, epoch=15,
            metadata={'metrics': dict(eval_macro_spearman=.6, eval_loss=1.)})
        retention_fixture.export_evaluation_predictions(**args)
    monkeypatch.setattr(old_fixture, 'make_run', fifteen)
    jobs = []
    qualification = dict(signature='fixture qualified runtime', world_size=4)
    for job in primary['tuning_jobs']:
        metadata = control_plan.materialize_job(job, directory, sha256(directory / 'plan.json'))
        jobs.append(metadata); complete_fixture(metadata, model)
        run = Path(metadata['output_dir'])
        write_json(run / 'runtime_qualification.json', qualification)
        receipt = module.read(run / 'controlled_training_complete.json')
        receipt.update(job_id=metadata['id'], phase='training_complete_test_not_scored')
        write_json(run / 'controlled_training_complete.json', receipt)
    # Only these two completed fixed-rate fixtures exist. There are no other
    # LR candidates or selection files; no search is needed to evaluate them.
    scope_path = tmp_path / 'scope.json'; write_json(scope_path, dict(seed=42))
    authorized = dict(jobs=[], references=dict(jobs=jobs, qualification=qualification))
    monkeypatch.setattr(module, 'scope', lambda: (scope_path, authorized))
    monkeypatch.setattr(module.postselection, 'verify_selection', lambda *_: pytest.fail('No LR selection'))
    from reward_model import model as model_api
    from reward_model.model import LoadedEncoder
    loader = model_api.load_reward_model
    def tiny_loader(checkpoint, **kwargs):
        original = make_model(None, {})
        return loader(checkpoint, **kwargs,
            protein_bundle=LoadedEncoder('tiny/protein', None, original.protein_encoder, 12),
            molecule_bundle=LoadedEncoder('tiny/molecule', None, original.molecule_encoder, 16))
    monkeypatch.setattr(model_api, 'load_reward_model', tiny_loader)
    return dict(jobs=jobs, primary=primary, directory=directory, model=model, authorized=authorized)


def test_scope_accepts_exact_live_authorization_and_rejects_an_added_seed(prepared, monkeypatch):
    monkeypatch.setattr(module, 'ROOT', prepared['root']); monkeypatch.setattr(module, 'WORK', prepared['work'])
    path, plan = module.scope()
    assert len(plan['jobs']) == 24 and len(plan['references']['jobs']) == 2
    plan['seed'] = 43; write_json(path, plan)
    with pytest.raises(ValueError, match='single-seed'): module.scope()


def test_plan_needs_only_completed_fixed_rate_fit_and_uses_validation_macro(native):
    job = native['jobs'][0]; plan = module.build_plan(job['id'])
    assert set(plan['rules']) == {'best_macro'}
    assert plan['rules']['best_macro']['epoch'] == 5
    assert plan['checkpoint']['id'] == 'checkpoint-50'
    assert not plan['parameter_search'] and not plan['test_used_for_selection']
    assert plan['seed'] == 42 and plan['learning_rate_multiplier'] == 1.
    assert not (native['directory'] / 'selections').exists()
    assert len(list((native['directory'] / 'runs').iterdir())) == 2
    with pytest.raises(ValueError, match='outside'): module.build_plan('full__A__seed43__lr1p0')
    with pytest.raises(ValueError, match='outside'): module.build_plan('full__A__seed42__lr0p3')


@pytest.mark.parametrize('damage', ['weight', 'config', 'rank', 'history', 'rules', 'runtime', 'premature_test', 'empty_receipt'])
def test_invalid_completed_training_cannot_be_evaluated(native, damage):
    job = native['jobs'][0]; run = Path(job['output_dir'])
    if damage == 'weight': (run / 'checkpoint-50/pytorch_model.bin').write_bytes(b'changed')
    elif damage == 'config': (run / 'checkpoint-50/config.json').write_text('{}')
    elif damage == 'rank': (run / 'rank_3_complete.json').unlink()
    elif damage == 'runtime': write_json(run / 'runtime_qualification.json', {})
    else:
        name = {'history':'trainer_state.json', 'rules':'retained/retention.json',
            'premature_test':'experiment_summary.json', 'empty_receipt':'controlled_training_complete.json'}[damage]
        record = module.read(run / name)
        if damage == 'history': record['epoch'] = 14
        elif damage == 'rules': record['selections']['best_macro']['step'] = 10
        elif damage == 'premature_test': record['test_evaluated'] = True
        else: record['evidence_sha256'] = {}
        write_json(run / name, record)
        if damage != 'empty_receipt':
            receipt = module.read(run / 'controlled_training_complete.json')
            receipt['evidence_sha256'][name] = sha256(run / name)
            write_json(run / 'controlled_training_complete.json', receipt)
    with pytest.raises((ValueError, FileNotFoundError)): module.build_plan(job['id'])


@pytest.mark.parametrize('native', [{}, {'freeze_protein_encoder':True, 'freeze_molecule_encoder':True},
    {'encoder_lora_rank':4, 'encoder_lora_alpha':8.}, {'contrastive_mask_policy':'training_known'}], indirect=True)
def test_actual_native_inference_preserves_weights_and_both_partitions(native):
    job = native['jobs'][0]; path = module.prepare(job['id'])
    weights = Path(module.read(path)['checkpoint']['path']) / 'pytorch_model.bin'
    before = sha256(weights)
    module.worker(path, 0, cpu_qualification=True)
    plan, results = module.verify_complete(path, cpu_qualification=True)
    assert set(results) == {'val', 'test'} and sha256(weights) == before
    assert all(r['parameters_unchanged'] and not r['optimizer_created'] for r in results.values())
    assert all(r['ranking_partition_seed'] == 42 for r in results.values())
    with pytest.raises(ValueError): module.verify_complete(path)
    done = sha256(path.parent / 'complete.json')
    module.worker(path, 0, cpu_qualification=True)
    assert sha256(path.parent / 'complete.json') == done
    completion = module.read(path.parent / 'complete.json'); completion['evidence_sha256'].pop('test')
    write_json(path.parent / 'complete.json', completion)
    with pytest.raises(ValueError, match='partition'): module.verify_complete(path, cpu_qualification=True)


def test_worker_rejects_partial_outputs_changed_source_and_unauthorized_gpu(native, monkeypatch):
    path = module.prepare(native['jobs'][0]['id']); plan = module.read(path)
    for index in (-1, 4, True):
        with pytest.raises(ValueError, match='GPU index'): module.worker(path, index, cpu_qualification=True)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'unapproved')
    with pytest.raises(RuntimeError, match='assigned'): module.worker(path, 0)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    output = path.parent / plan['checkpoint']['id'] / 'test'; output.mkdir(parents=True)
    with pytest.raises(RuntimeError, match='Partial'): module.worker(path, 0, cpu_qualification=True)
    assert output.exists() and not (path.parent / 'complete.json').exists()
    source = Path(plan['data']['val']['source']); source.write_bytes(b'changed')
    with pytest.raises(ValueError, match='data differs'): module.worker(path, 0, cpu_qualification=True)


def test_only_complete_matching_legacy_outputs_are_reused(native, monkeypatch):
    job = native['jobs'][0]; path = module.prepare(job['id']); plan = module.read(path)
    module.worker(path, 0, cpu_qualification=True)
    # Preserve actual native payloads, but supply an explicitly simulated GPU
    # receipt here to exercise reuse routing. This is not GPU qualification.
    legacy = module.WORK / 'postselection' / job['id']; legacy.mkdir(parents=True)
    old_plan = dict(job=job, checkpoints=[plan['checkpoint']], rules=plan['rules'], data=plan['data'])
    old_path = legacy / 'plan.json'; write_json(old_path, old_plan)
    monkeypatch.setattr(module.postselection, 'build_evaluation_plan', lambda *_: old_plan)
    old_output = legacy / plan['checkpoint']['id'] / 'val'; old_output.mkdir(parents=True)
    source = dict(plan_sha256=sha256(old_path), checkpoint=plan['checkpoint'], split='val', gpu_uuid=module.GPU_UUIDS[1])
    write_json(old_output / 'complete.json', dict(source=source))
    partial = legacy / plan['checkpoint']['id'] / 'test'; partial.mkdir(); (partial / 'partial.txt').write_text('preserve')
    calls = []
    monkeypatch.setattr(module.postselection, 'verify_evaluation', lambda output, observed, **_: calls.append((output, observed)))
    reused = module.legacy_completed(job['id'], plan['checkpoint'], job, plan['data'])
    assert set(reused) == {'val'} and calls == [(old_output, source)]
    assert (partial / 'partial.txt').read_text() == 'preserve'
    changed = copy.deepcopy(plan['data']); changed['val']['source_sha256'] = 'changed'
    with pytest.raises(ValueError, match='same selected'): module.legacy_completed(job['id'], plan['checkpoint'], job, changed)


@pytest.mark.parametrize('gate', ['lock', 'gpu', 'storage'])
def test_busy_resources_cannot_spawn_an_evaluator(native, monkeypatch, gate):
    import fcntl
    monkeypatch.setattr(module.control_queue, 'runtime_guard', lambda: None)
    monkeypatch.setattr(module.control_queue, 'gpu_processes', lambda: [{'pid':1}] if gate == 'gpu' else [])
    monkeypatch.setattr(module.control_queue, 'storage_gate', lambda: dict(ready=gate != 'storage'))
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **k: pytest.fail('No launch while busy'))
    with (native['directory'] / 'driver.lock').open('a') as lock:
        if gate == 'lock': fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = module.run_batch([native['jobs'][0]['id']])
    assert result['phase'].startswith('waiting_for_')


def test_batch_guard_rejects_duplicate_or_excessive_jobs(native, monkeypatch):
    monkeypatch.setattr(module.control_queue, 'runtime_guard', lambda: None)
    for jobs in ([], ['same', 'same'], list('abcde')):
        with pytest.raises(ValueError, match='distinct'): module.run_batch(jobs)


@pytest.mark.parametrize('launch_failure', [False, True])
def test_batch_assigns_four_authorized_gpus_and_cleans_up_launch_failure(tmp_path, monkeypatch, launch_failure):
    monkeypatch.setattr(module, 'WORK', tmp_path)
    (tmp_path / 'controlled').mkdir()
    monkeypatch.setattr(module.control_queue, 'runtime_guard', lambda: None)
    monkeypatch.setattr(module.control_queue, 'gpu_processes', lambda: [])
    monkeypatch.setattr(module.control_queue, 'storage_gate', lambda: dict(ready=True))
    paths = {}
    for i in range(4):
        p = tmp_path / str(i) / 'plan.json'; write_json(p, dict(job=i)); paths[str(i)] = p
    monkeypatch.setattr(module, 'prepare', lambda identity: paths[identity])
    monkeypatch.setattr(module, 'verify_complete', lambda path: None)
    calls, children = [], []
    class Child:
        def __init__(self, index): self.pid = 10+index; self.code = None; self.terminated = False
        def wait(self): self.code = 0; return 0
        def poll(self): return self.code
        def terminate(self): self.terminated = True; self.code = -15
    def spawn(command, **kwargs):
        index = len(calls)
        if launch_failure and index == 2: raise OSError('fixture launch failure')
        calls.append((command, kwargs)); child = Child(index); children.append(child)
        # Minimal completion file for the scheduler's final hashing only.
        write_json(paths[str(index)].parent / 'complete.json', dict(fixture=True))
        return child
    monkeypatch.setattr(module.subprocess, 'Popen', spawn)
    if launch_failure:
        with pytest.raises(OSError, match='launch failure'): module.run_batch(list('0123'))
        assert all(c.terminated for c in children)
    else:
        result = module.run_batch(list('0123'))
        assert result['phase'] == 'requested_best_macro_batch_complete'
        assert len(calls) == 4 and not any(c.terminated for c in children)
    for i, (command, kwargs) in enumerate(calls):
        assert command[-1] == str(i)
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == module.GPU_UUIDS[i]
        assert kwargs['env']['WANDB_MODE'] == 'disabled'
