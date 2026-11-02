"""E41 provenance gates and real tiny-model native inference, without training."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from datasets import Dataset, load_from_disk

from experiments.paired_generalization import learning_curve_evaluation as module
from experiments.paired_generalization.run import sha256, write_json
from experiments.paired_generalization.qualify_controls import make_model
from test_paired_generalization_checkpoint_evaluation import examples_fixture
from test_paired_generalization_control_plan import ROOT, noise_fixture


@pytest.fixture
def grid(tmp_path, monkeypatch):
    """Training histories are synthetic. Worker tests use actual saved tiny weights."""
    for name, value in [('CUDA_VISIBLE_DEVICES', ''), ('ACCELERATE_USE_CPU', 'true'), ('WANDB_MODE', 'disabled')]:
        monkeypatch.setenv(name, value)
    for owner in (module, module.learning_curve_driver, module.control_plan):
        monkeypatch.setattr(owner, 'WORK', tmp_path)
    model = make_model(None, {})
    dataset = examples_fixture(model); raw = dataset.to_pandas()
    from reward_model.training.train_views import subset_pair_labels
    raw['binary_label'] = subset_pair_labels(raw, np.arange(len(raw)))
    training = Dataset.from_pandas(raw, preserve_index=False)
    sources = {}; caches = {}
    for role in ('train', 'val', 'test'):
        rows = training if role == 'train' else dataset
        source = tmp_path/'data'/f'{role}.parquet'; source.parent.mkdir(exist_ok=True)
        rows.to_pandas().to_parquet(source, index=False); sources[role] = sha256(source)
        cache = tmp_path/'cache'/f'{role}_examples'; rows.save_to_disk(str(cache))
        caches[role] = module.control_plan.token_cache_hashes(cache)
    training = load_from_disk(str(tmp_path/'cache/train_examples'))
    primary = module.control_plan.build_plan(ROOT, tmp_path, noise_fixture())
    full = primary['arms'][0]; primary['tuning_jobs'] = []
    for regime in ('A', 'B'):
        base = primary['bases'][regime]
        base['model'] = model.config.to_dict()
        base['model'].update(protein_hidden_size=None, molecule_hidden_size=None)
        base['data'].update(tokenized_dataset_dir=str(tmp_path/'cache'),
            **{f'{s}_parquet_path': str(tmp_path/'data'/f'{s}.parquet') for s in sources})
        base['training'].update(bf16=False, fp16=False, dataloader_num_workers=0, protein_shuffle_sensitivity=False)
    for lr in (1., .3, 3.):
        for regime in ('A', 'B'):
            identity = module.control_plan.job_id('full', regime, 42, lr)
            config = module.control_plan.make_config(primary['bases'][regime], full, regime=regime, seed=42,
                multiplier=lr, output=tmp_path/'controlled/runs'/identity, training_noise_rms=primary['training_noise_rms'])
            primary['tuning_jobs'].append(dict(id=identity, arm='full', regime=regime, seed=42,
                learning_rate_multiplier=lr, phase='tuning', config=config))
    primary.update(source_data_sha256={r: sources for r in ('A', 'B')}, tokenized_data_sha256={r: caches for r in ('A', 'B')})
    primary_path = tmp_path/'controlled/plan.json'; write_json(primary_path, primary)
    qualified_source = 'RewardModel/experiments/paired_generalization/learning_curve_evaluation.py'
    code = {qualified_source: sha256(ROOT/qualified_source)}
    runtime = dict(world_size=4, sources=code,
        signature=module.control_plan.digest_json(dict(sources=code, plan_sha256=sha256(primary_path))))
    curve_dir = tmp_path/'learning_curves'; views = {}; statistics = {}; artifacts = {}
    for regime in ('A', 'B'):
        base = curve_dir/'views'/regime; base.mkdir(parents=True)
        metadata = raw.copy()
        metadata['unit'] = metadata.assay_id if regime == 'A' else metadata.assay_id.map({'a0': 0, 'a1': 1, 'a2': 2})
        source_metadata = base/'source_metadata.parquet'; metadata.to_parquet(source_metadata, index=False)
        subsets, order = module.learning_curve_plan.nested_unit_indices(metadata.unit.to_numpy())
        np.save(base/'unit_order.npy', order.astype(str) if order.dtype.kind == 'O' else order, allow_pickle=False)
        views[regime] = {}; statistics[regime] = {}
        for fraction, indices in subsets.items():
            name = module.learning_curve_plan.fraction_name(fraction); labels = subset_pair_labels(metadata, indices)
            np.save(base/f'{name}_indices.npy', indices); np.save(base/f'{name}_labels.npy', labels)
            stats = dict(rows=len(indices), labels_changed_from_full_training=int(np.count_nonzero(labels != metadata.iloc[indices].binary_label)))
            record = dict(format_version=1, regime=regime, fraction_requested=fraction, subset_seed=42,
                label_policy='within_selected_training_pair_median', activity_threshold=6.,
                unit_definition='whole assay' if regime == 'A' else 'whole scaffold-assay component',
                source=dict(parquet_path=str(tmp_path/'data/train.parquet'), parquet_sha256=sources['train'],
                    cache_dir=str(tmp_path/'cache/train_examples'), cache_files_sha256=caches['train'],
                    fingerprint=training._fingerprint, rows=len(training)),
                statistics=stats, unit_order_sha256=sha256(base/'unit_order.npy'))
            for key, path in [('indices', base/f'{name}_indices.npy'), ('labels', base/f'{name}_labels.npy'), ('source_metadata', source_metadata)]:
                record[key] = dict(path=str(path), sha256=sha256(path))
            path = base/f'{name}.json'; write_json(path, record)
            views[regime][name] = dict(path=str(path), sha256=sha256(path)); statistics[regime][name] = stats
        artifacts[regime] = dict(unit_order_sha256=sha256(base/'unit_order.npy'), source_metadata_sha256=sha256(source_metadata))
    jobs = []
    for fraction in module.learning_curve_plan.FRACTIONS[:-1]:
        name = module.learning_curve_plan.fraction_name(fraction)
        for source in primary['tuning_jobs']:
            job = copy.deepcopy(source); job['arm'] = 'unit_curve_'+name; job['fraction'] = fraction
            job['id'] = module.control_plan.job_id(job['arm'], job['regime'], 42, job['learning_rate_multiplier'])
            job['config']['training']['output_dir'] = str(curve_dir/'runs'/job['id'])
            view = views[job['regime']][name]
            job['config']['data'].update(train_view_manifest=view['path'], train_view_sha256=view['sha256'])
            jobs.append(job)
    curve = dict(primary_plan_sha256=sha256(primary_path), spec_sha256=primary['spec_sha256'],
        fractions=list(module.learning_curve_plan.FRACTIONS), subset_seed=42, views=views,
        statistics=statistics, artifacts=artifacts, tuning_jobs=jobs)
    curve_path = curve_dir/'plan.json'; write_json(curve_path, curve)
    write_json(curve_dir/'preparation_complete.json', dict(plan_sha256=sha256(curve_path)))
    write_json(curve_dir/'qualified_data_views.json', dict(plan_sha256=sha256(curve_path), views=[dict(regime=r) for r in ('A', 'B')]))

    def finish(metadata, view=None):
        run = Path(metadata['output_dir']); score = ({.3: .8, 1.: .2, 3.: .1} if metadata['regime'] == 'A' else {.3: .2, 1.: .3, 3.: .1})[metadata['learning_rate_multiplier']]
        history = [dict(epoch=e, step=e*10, eval_loss=.6+abs(e-3)*.1,
            eval_macro_spearman=score if e == 5 else score-.1-.005*e) for e in range(1, 16)]
        rules = module.selection_rules(history); preserved = []
        write_json(run/'config.json', model.config.to_dict())
        for name in ('checkpoint-10', 'checkpoint-30', 'checkpoint-50', 'retained/average_from_loss_minimum'):
            path = run/name; write_json(path/'config.json', model.config.to_dict())
            torch.save(model.state_dict(), path/'pytorch_model.bin')
            preserved.append(dict(path=str((path/'pytorch_model.bin').relative_to(run)), sha256=sha256(path/'pytorch_model.bin')))
        write_json(run/'trainer_state.json', dict(epoch=15, global_step=150, log_history=history))
        write_json(run/'retained/retention.json', dict(config_sha256=metadata['config_sha256'], selections=rules,
            averaging=dict(epochs=list(range(3, 16)), singleton=False), preserved_weights=preserved))
        write_json(run/'runtime_qualification.json', runtime)
        write_json(run/'experiment_summary.json', dict(config_sha256=metadata['config_sha256'], test_evaluated=False,
            test_metrics={}, training_view=view['provenance'] if view else None,
            train_examples=view['provenance']['statistics']['rows'] if view else len(raw)))
        manifest = run/'prediction_exports/trajectory/eval_step00000050.json'; manifest.parent.mkdir(parents=True)
        manifest.with_suffix('.npz').write_bytes(b'synthetic training validation payload')
        write_json(manifest, dict(predictions_sha256=sha256(manifest.with_suffix('.npz')),
            metrics=dict(eval_macro_spearman=score, eval_spearman_defined_fraction=1.)))
        for rank in range(4):
            write_json(run/f'rank_{rank}_complete.json', dict(rank=rank, world_size=4, config_sha256=metadata['config_sha256']))
        files = ['experiment_summary.json', 'trainer_state.json', 'retained/retention.json', *(f'rank_{i}_complete.json' for i in range(4))]
        write_json(run/'controlled_training_complete.json', dict(config_sha256=metadata['config_sha256'], plan_sha256=metadata['plan_sha256'],
            phase='training_complete_test_not_scored', evidence_sha256={n: sha256(run/n) for n in files}))
    for job in primary['tuning_jobs']:
        finish(module.control_plan.materialize_job(job, primary_path.parent, sha256(primary_path)))
    primary_confirmation = module.control_plan.confirm('full')
    for metadata in primary_confirmation['confirmation_jobs']:
        finish(metadata)
    curve_metadata = []
    for job in jobs:
        metadata = module.control_plan.materialize_job(job, curve_dir, sha256(curve_path)); curve_metadata.append(metadata)
        if job['fraction'] == .1:
            finish(metadata, module.view_contract(curve, primary, job['regime'], .1))
    confirmation = module.learning_curve_driver.confirm_fraction(.1, curve_dir)
    for metadata in confirmation['confirmation_jobs']:
        finish(metadata, module.view_contract(curve, primary, metadata['regime'], .1))
    return dict(work=tmp_path, curve=curve, primary=primary, model=model, metadata=curve_metadata,
        confirmations=confirmation['confirmation_jobs'], primary_confirmations=primary_confirmation['confirmation_jobs'])


def job_id(regime='A', seed=42, lr=1.):
    return module.control_plan.job_id('unit_curve_fraction010', regime, seed, lr)


def change_summary(grid, identity, field, value):
    run = grid['work']/'learning_curves/runs'/identity; path = run/'experiment_summary.json'
    summary = module.read(path); summary[field] = value; write_json(path, summary)
    path = run/'controlled_training_complete.json'; done = module.read(path)
    done['evidence_sha256']['experiment_summary.json'] = sha256(run/'experiment_summary.json'); write_json(path, done)


def test_selection_uses_b_validation_and_reuses_full_endpoint_without_mutation(grid):
    before = {str(p): sha256(p) for p in grid['work'].rglob('*') if p.is_file()}
    selected = module.build_evaluation_plan(job_id())
    assert selected['selected_learning_rate'] and len(selected['rules']) == 5
    assert len(selected['checkpoints']) == 4 and len(selected['shared_full_endpoint']['job_ids']) == 10
    assert not selected['shared_full_endpoint']['refitted'] and selected['new_fits_created'] == 0
    assert not selected['test_used_for_selection']
    assert selected['training_view']['provenance']['statistics']['rows'] == 3
    assert selected['data']['test']['source'] == str(grid['work']/'data/test.parquet')
    other = module.build_evaluation_plan(job_id(lr=.3))
    assert not other['selected_learning_rate'] and set(other['rules']) == {'best_macro'}
    confirmation = module.build_evaluation_plan(job_id(seed=43))
    assert confirmation['seed_role'] == 'confirmation' and len(confirmation['rules']) == 5
    assert {str(p): sha256(p) for p in grid['work'].rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('damage', ['full_confirmation', 'other_tuning', 'selection', 'view_used', 'rows_used',
    'test_exposed', 'rank', 'runtime', 'weights', 'model', 'partial_history', 'confirmation_lr'])
def test_missing_or_changed_evidence_cannot_create_evaluation_plan(grid, damage):
    work = grid['work']; identity = job_id(); run = work/'learning_curves/runs'/identity
    if damage == 'full_confirmation':
        (Path(grid['primary_confirmations'][0]['output_dir'])/'controlled_training_complete.json').unlink()
    elif damage == 'other_tuning':
        (work/'learning_curves/runs'/job_id('B', lr=3.)/'controlled_training_complete.json').unlink()
    elif damage == 'selection':
        p = work/'learning_curves/selections/fraction010.json'; r = module.read(p); r['learning_rate_multiplier'] = .3; write_json(p, r)
    elif damage == 'view_used':
        change_summary(grid, identity, 'training_view', None)
    elif damage == 'rows_used':
        change_summary(grid, identity, 'train_examples', 9)
    elif damage == 'test_exposed':
        change_summary(grid, identity, 'test_evaluated', True)
    elif damage == 'rank':
        (run/'rank_3_complete.json').unlink()
    elif damage == 'runtime':
        p = run/'runtime_qualification.json'; r = module.read(p); r['world_size'] = 2; write_json(p, r)
    elif damage == 'weights':
        (run/'checkpoint-50/pytorch_model.bin').write_bytes(b'changed')
    elif damage == 'model':
        p = run/'checkpoint-50/config.json'; r = module.read(p); r['dropout'] = .7; write_json(p, r)
    elif damage == 'partial_history':
        p = run/'trainer_state.json'; r = module.read(p); r['epoch'] = 14; write_json(p, r)
    else:
        identity = job_id(seed=43); p = work/'learning_curves/jobs'/f'{identity}.json'
        r = module.read(p); r['selection_sha256'] = 'changed'; write_json(p, r)
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        module.prepare(identity)
    assert not list((work/'learning_curve_evaluation').glob('*/plan.json'))


@pytest.mark.parametrize('damage', ['labels', 'fragmented_unit', 'permutation', 'full_labels'])
def test_rehashed_subset_payloads_cannot_change_the_registered_sampling_or_labels(grid, damage):
    curve = copy.deepcopy(grid['curve']); regime = 'A'; fraction = 1. if damage == 'full_labels' else .1
    name = module.learning_curve_plan.fraction_name(fraction); item = curve['views'][regime][name]
    p = Path(item['path']); record = module.read(p)
    if damage in ('labels', 'full_labels'):
        target = Path(record['labels']['path']); values = np.load(target); values[0] = 1-values[0]; np.save(target, values)
        record['labels']['sha256'] = sha256(target)
    elif damage == 'fragmented_unit':
        target = Path(record['indices']['path']); values = np.load(target)[1:]; np.save(target, values)
        record['indices']['sha256'] = sha256(target)
    else:
        target = p.parent/'unit_order.npy'; values = np.load(target); np.save(target, values[::-1])
        record['unit_order_sha256'] = sha256(target); curve['artifacts'][regime]['unit_order_sha256'] = sha256(target)
    write_json(p, record); item['sha256'] = sha256(p)
    with pytest.raises(ValueError):
        module.view_contract(curve, grid['primary'], regime, fraction)


def test_grid_rejects_extra_fit_budget_model_or_full_endpoint_changes(grid):
    for damage in ('missing', 'budget', 'mask', 'fraction'):
        curve = copy.deepcopy(grid['curve'])
        if damage == 'missing':
            curve['tuning_jobs'].pop()
        elif damage == 'budget':
            curve['tuning_jobs'][0]['config']['training']['num_train_epochs'] = 10
        elif damage == 'mask':
            curve['tuning_jobs'][0]['config']['model']['contrastive_mask_policy'] = 'training_known'
        else:
            curve['tuning_jobs'][0]['fraction'] = 1.
        with pytest.raises(ValueError):
            module.validate_grid(curve, grid['primary'])
    before = {str(p): sha256(p) for p in grid['work'].rglob('*') if p.is_file()}
    result = module.preflight()
    assert result['registered_tuning_jobs'] == 18 and not result['scientific_evaluation_created']
    assert result['new_fits_created'] == 0
    assert {str(p): sha256(p) for p in grid['work'].rglob('*') if p.is_file()} == before


def test_rehashed_partial_nonselected_candidate_cannot_enter_lr_selection(grid):
    run = grid['work']/'learning_curves/runs'/job_id('B', lr=3.)
    path = run/'trainer_state.json'; state = module.read(path)
    state['epoch'] = 14; state['global_step'] = 140; state['log_history'].pop(); write_json(path, state)
    path = run/'controlled_training_complete.json'; done = module.read(path)
    done['evidence_sha256']['trainer_state.json'] = sha256(run/'trainer_state.json'); write_json(path, done)
    with pytest.raises(ValueError, match='finish all fifteen epochs'):
        module.prepare(job_id())
    assert not list((grid['work']/'learning_curve_evaluation').glob('*/plan.json'))


def test_real_saved_tiny_weights_evaluate_heldout_rows_without_training(grid, monkeypatch):
    import reward_model.model as model_api
    from reward_model.model import LoadedEncoder
    from reward_model.training.trainer import RewardModelTrainer
    original = model_api.load_reward_model; loads = []
    def loader(path, **kwargs):
        model = make_model(None, {}); loads.append(path)
        return original(path, **kwargs, protein_bundle=LoadedEncoder('tiny/protein', None, model.protein_encoder, 12),
            molecule_bundle=LoadedEncoder('tiny/molecule', None, model.molecule_encoder, 16))
    def no_training(*args, **kwargs):
        raise AssertionError('Curve evaluation cannot train or create an optimizer')
    monkeypatch.setattr(model_api, 'load_reward_model', loader)
    monkeypatch.setattr(RewardModelTrainer, 'train', no_training)
    monkeypatch.setattr(RewardModelTrainer, 'create_optimizer', no_training)
    before = {str(p): sha256(p) for p in (grid['work']/'learning_curves').rglob('*') if p.is_file()}
    path = module.prepare(job_id(lr=.3)); module.evaluate_worker(path, 0, 1, cpu_qualification=True)
    assert len(loads) == 1
    evidence = {}
    for role in ('val', 'test'):
        output = path.parent/'checkpoint-50'/role; result = module.read(output/'complete.json')
        assert result['rows'] == 9 and result['parameters_unchanged'] and not result['optimizer_created']
        assert result['cpu_qualification'] and result['known_positive_index'] is None
        evidence[f'checkpoint-50/{role}'] = sha256(output/'complete.json')
    module.immutable(path.parent/'complete.json', dict(phase='curve_fixed_weight_evaluation_complete',
        plan_sha256=sha256(path), evidence_sha256=evidence))
    module.verify_complete(path.parent, cpu_qualification=True)
    with pytest.raises(ValueError, match='device provenance'):
        module.verify_complete(path.parent)
    module.evaluate_worker(path, 0, 1, cpu_qualification=True); assert len(loads) == 1
    assert {str(p): sha256(p) for p in (grid['work']/'learning_curves').rglob('*') if p.is_file()} == before
    receipt = module.read(path.parent/'checkpoint-50/test/complete.json')
    Path(receipt['prediction_manifest']).with_suffix('.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='payload differs'):
        module.evaluate_worker(path, 0, 1, cpu_qualification=True)


def test_workers_reject_wrong_gpu_and_launcher_respects_training_lock(tmp_path, monkeypatch):
    import fcntl
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    with pytest.raises(RuntimeError, match='authorized GPU'):
        module.evaluate_worker(Path('unused'), 0, 4)
    monkeypatch.setattr(module, 'ROOT', Path('/home/atabey/Prot2Mol')); monkeypatch.setattr(module, 'WORK', tmp_path)
    (tmp_path/'controlled').mkdir()
    with (tmp_path/'controlled/driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            module.run_evaluation('unused')
