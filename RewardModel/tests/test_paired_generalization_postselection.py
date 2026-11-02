"""Selection/provenance failures plus real fixed-weight inference on a tiny grid.

The retained training histories are synthetic fixtures. Inference uses actual
saved tiny ESM/RoBERTa weights and the native Trainer; it is not science evidence.
"""
import copy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
import yaml

from experiments.paired_generalization import control_plan, postselection
from experiments.paired_generalization.run import sha256, write_json
from experiments.paired_generalization.retain_controlled import retain
from experiments.paired_generalization.qualify_controls import make_model
from test_paired_generalization_checkpoint_evaluation import examples_fixture
from test_paired_generalization_control_plan import noise_fixture
from test_paired_generalization_retention import make_run

ROOT = Path(__file__).resolve().parents[2]


def test_saved_config_accepts_inferred_or_explicit_widths_without_mutating_inputs():
    model = make_model(None, {})
    launch = replace(model.config, protein_hidden_size=None, molecule_hidden_size=None)
    saved = json.loads(json.dumps(model.config.to_dict()))
    before = copy.deepcopy((saved, launch.to_dict()))
    for expected in (launch, model.config):
        postselection.verify_model_config(saved, expected)
        postselection.verify_model_config(saved, expected,
            encoder_widths={'protein':12, 'molecule':16})
    assert (saved, launch.to_dict()) == before


@pytest.mark.parametrize('change', [
    {'protein_hidden_size':15}, {'molecule_hidden_size':20},
    {'dropout':.05}, {'contrastive_loss_weight':.5},
    {'ranking_head_hidden_dims':[9]}, {'molecule_model_name_or_path':'other/model'},
])
def test_config_resolution_does_not_allow_other_launch_changes(change):
    model = make_model(None, {})
    launch = replace(model.config, protein_hidden_size=None, molecule_hidden_size=None, **{
        k:v for k,v in change.items() if k not in ('protein_hidden_size','molecule_hidden_size')})
    for key in ('protein_hidden_size','molecule_hidden_size'):
        if key in change:
            setattr(launch, key, change[key])
    with pytest.raises(ValueError, match='differs from the launch'):
        postselection.verify_model_config(model.config.to_dict(), launch)


@pytest.mark.parametrize('width', [None, 0, -1, True, 12.5])
def test_saved_width_must_be_resolved(width):
    model = make_model(None, {})
    launch = replace(model.config, protein_hidden_size=None, molecule_hidden_size=None)
    saved = model.config.to_dict()
    saved['protein_hidden_size'] = width
    with pytest.raises(ValueError, match='resolved positive integer'):
        postselection.verify_model_config(saved, launch)


def test_missing_null_field_and_changed_encoder_cannot_pass_config_verification():
    model = make_model(None, {})
    launch = replace(model.config, protein_hidden_size=None, molecule_hidden_size=None)
    saved = model.config.to_dict()
    with pytest.raises(ValueError, match='loaded encoder'):
        postselection.verify_model_config(saved, launch,
            encoder_widths={'protein':15, 'molecule':16})
    del saved['protein_tokenizer_name_or_path']
    with pytest.raises(ValueError, match='fields differ'):
        postselection.verify_model_config(saved, launch)


def complete_fixture(metadata, model):
    run = Path(metadata['output_dir'])
    make_run(run)
    config = yaml.safe_load(Path(metadata['config_path']).read_text())
    write_json(run/'controlled_run.json', dict(format_version=1, allow_checkpoint_retention=True, **metadata))
    # Allocation-specific retention has its own tests. This fixture supplies the
    # primary retained artifacts directly for B without claiming allocation work.
    marker = postselection.read_json(run/'controlled_run.json')
    marker.pop('regime')
    write_json(run/'controlled_run.json', marker)
    write_json(run/'launch_provenance.json', dict(config_sha256=metadata['config_sha256'], world_size=4, config=config))
    write_json(run/'config.json', model.config.to_dict())
    for rank in range(4):
        write_json(run/f'rank_{rank}_complete.json', dict(rank=rank, world_size=4, config_sha256=metadata['config_sha256']))
    weights = model.state_dict()
    for checkpoint in run.glob('checkpoint-*'):
        write_json(checkpoint/'config.json', model.config.to_dict())
        torch.save(weights, checkpoint/'pytorch_model.bin')
    torch.save(weights, run/'pytorch_model.bin')
    summary = postselection.read_json(run/'experiment_summary.json')
    summary.update(config_sha256=metadata['config_sha256'], test_evaluated=False, test_metrics={})
    write_json(run/'experiment_summary.json', summary)
    if model.config.contrastive_mask_policy == 'training_known':
        from datasets import load_from_disk
        from reward_model.training.known_positives import KnownPositiveIndex
        source = load_from_disk(str(Path(config['data']['tokenized_dataset_dir'])/'train_examples'))
        index = KnownPositiveIndex.from_training_dataset(source, active_threshold=model.config.contrastive_active_threshold)
        write_json(run/'known_positive_index.json', index.provenance)
    for path in (run/'prediction_exports/trajectory').glob('eval_*.json'):
        record = postselection.read_json(path)
        record['metrics']['eval_spearman_defined_fraction'] = 1.
        write_json(path, record)
    retain(run, prune=True)
    files = ['experiment_summary.json', 'trainer_state.json', 'retained/retention.json',
             *(f'rank_{rank}_complete.json' for rank in range(4))]
    write_json(run/'controlled_training_complete.json', dict(config_sha256=metadata['config_sha256'],
        plan_sha256=metadata['plan_sha256'], evidence_sha256={name:sha256(run/name) for name in files}))


@pytest.fixture
def grid(tmp_path, monkeypatch, request):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('ACCELERATE_USE_CPU', 'true')
    monkeypatch.setenv('WANDB_MODE', 'disabled')
    monkeypatch.setattr(postselection, 'WORK', tmp_path)
    model = make_model(None, {'contrastive_mask_policy':'training_known'} if getattr(request, 'param', False) else {})
    dataset = examples_fixture(model)
    sources, caches = {}, {}
    for split in ('train', 'val', 'test'):
        source = tmp_path/'data'/f'{split}.parquet'
        source.parent.mkdir(exist_ok=True)
        rows = dataset.select(range(6)) if split == 'train' else dataset
        rows.to_pandas().to_parquet(source)
        cache = tmp_path/'cache'/f'{split}_examples'
        rows.save_to_disk(str(cache))
        sources[split] = sha256(source)
        caches[split] = control_plan.token_cache_hashes(cache)
    plan = control_plan.build_plan(ROOT, tmp_path, noise_fixture())
    plan['arms'] = plan['arms'][:1]
    plan['tuning_jobs'] = plan['tuning_jobs'][:6]
    for regime in ('A', 'B'):
        plan['bases'][regime]['data'].update(tokenized_dataset_dir=str(tmp_path/'cache'),
            **{f'{s}_parquet_path':str(tmp_path/'data'/f'{s}.parquet') for s in sources})
    for job in plan['tuning_jobs']:
        job['config']['model'] = model.config.to_dict()
        # Production launch configs infer these widths from the encoders; saved
        # checkpoints contain the resolved integers. Exercise that real path.
        job['config']['model'].update(protein_hidden_size=None, molecule_hidden_size=None)
        job['config']['data'] = copy.deepcopy(plan['bases'][job['regime']]['data'])
        job['config']['training'].update(num_train_epochs=6, bf16=False, fp16=False,
            dataloader_num_workers=0, protein_shuffle_sensitivity=False)
    plan.update(source_data_sha256={r:sources for r in ('A','B')},
                tokenized_data_sha256={r:caches for r in ('A','B')})
    directory = tmp_path/'controlled'
    write_json(directory/'plan.json', plan)
    candidates, evidence, metadata = [], [], []
    for job in plan['tuning_jobs']:
        item = control_plan.materialize_job(job, directory, sha256(directory/'plan.json'))
        metadata.append(item)
        complete_fixture(item, model)
        candidate, source = control_plan.validation_candidate(item)
        evidence.append(source)
        if item['regime'] == 'B':
            candidates.append(candidate)
    decision = dict(arm='full', plan_sha256=sha256(directory/'plan.json'),
                    **control_plan.select_learning_rate(candidates), evidence=evidence)
    write_json(directory/'selections/full.json', decision)
    return directory, plan, metadata, model


def selected_job(grid):
    return next(item for item in grid[2] if item['regime'] == 'A' and item['learning_rate_multiplier'] == .3)


def test_plan_compares_all_rules_only_at_selected_lr_and_deduplicates_same_epoch(grid):
    directory, _, metadata, _ = grid
    selected = postselection.build_evaluation_plan(directory, selected_job(grid)['id'])
    assert selected['selected_learning_rate']
    assert len(selected['rules']) == 5 and len(selected['checkpoints']) == 4
    assert selected['rules']['best_loss']['checkpoint'] == selected['rules']['patience3_loss']['checkpoint']
    assert selected['rules']['best_macro']['step'] == 50
    assert selected['rules']['patience3_macro']['step'] == 10
    assert selected['rules']['average_from_loss_minimum']['epochs'] == [3, 4, 5, 6]
    assert selected['seed_role'] == 'tuning_exposed'
    other = postselection.build_evaluation_plan(directory, metadata[0]['id'])
    assert not other['selected_learning_rate']
    assert set(other['rules']) == {'best_macro'}
    assert len(other['checkpoints']) == 1


def test_checkpoint_configuration_must_match_the_run_config(grid):
    directory, _, _, _ = grid
    target = selected_job(grid)
    path = Path(target['output_dir'])/'checkpoint-50/config.json'
    config = postselection.read_json(path)
    config['protein_hidden_size'] += 3
    write_json(path, config)
    with pytest.raises(ValueError, match='Retained model configuration changed'):
        postselection.build_evaluation_plan(directory, target['id'])


@pytest.mark.parametrize('damage', ['missing_rank', 'changed_config', 'changed_selection',
                                   'changed_weights', 'partial_training', 'premature_test'])
def test_incomplete_or_changed_evidence_cannot_create_test_jobs(grid, damage):
    directory, _, metadata, _ = grid
    target = selected_job(grid)
    run = Path(target['output_dir'])
    if damage == 'missing_rank':
        (run/'rank_3_complete.json').unlink()
    elif damage == 'changed_config':
        with Path(target['config_path']).open('a') as handle:
            handle.write('\n# modified\n')
    elif damage == 'changed_selection':
        path = directory/'selections/full.json'
        record = postselection.read_json(path)
        record['learning_rate_multiplier'] = 3.
        write_json(path, record)
    elif damage == 'changed_weights':
        (run/'checkpoint-50/pytorch_model.bin').write_bytes(b'changed')
    elif damage == 'partial_training':
        # Even the other split's non-winning candidate must finish first.
        (Path(metadata[1]['output_dir'])/'controlled_training_complete.json').unlink()
    else:
        path = run/'experiment_summary.json'
        record = postselection.read_json(path)
        record['test_evaluated'] = True
        write_json(path, record)
        complete = postselection.read_json(run/'controlled_training_complete.json')
        complete['evidence_sha256']['experiment_summary.json'] = sha256(path)
        write_json(run/'controlled_training_complete.json', complete)
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        postselection.build_evaluation_plan(directory, target['id'])


def test_confirmation_requires_the_same_frozen_lr_and_records_seed_role(grid):
    directory, plan, _, model = grid
    selected = selected_job(grid)
    source = next(j for j in plan['tuning_jobs'] if j['id'] == selected['id'])
    job = copy.deepcopy(source)
    job.update(id=control_plan.job_id('full', 'A', 43, .3), seed=43, phase='confirmation',
               selection_sha256=sha256(directory/'selections/full.json'))
    job['config']['training'].update(seed=43, output_dir=str(directory/'runs'/job['id']))
    metadata = control_plan.materialize_job(job, directory, sha256(directory/'plan.json'))
    complete_fixture(metadata, model)
    result = postselection.build_evaluation_plan(directory, job['id'])
    assert result['seed_role'] == 'confirmation' and len(result['rules']) == 5
    metadata['selection_sha256'] = 'wrong'
    write_json(directory/'jobs'/f"{job['id']}.json", metadata)
    with pytest.raises(ValueError, match='frozen B-selected'):
        postselection.build_evaluation_plan(directory, job['id'])


@pytest.mark.parametrize('grid', [False, True], indirect=True, ids=['reference_mask','training_known_mask'])
def test_saved_tiny_weights_flow_through_worker_and_native_exports_without_training(grid, monkeypatch):
    import reward_model.model as model_api
    from reward_model.model import LoadedEncoder
    from reward_model.training.trainer import RewardModelTrainer
    from experiments.paired_generalization.checkpoint_evaluation import read_native_predictions
    directory, _, metadata, _ = grid
    # Use one nonselected LR: same worker path, one actual checkpoint, both splits.
    path = postselection.prepare(metadata[0]['id'])
    original_loader = model_api.load_reward_model
    loads = []
    def tiny_loader(checkpoint, **kwargs):
        model = make_model(None, {})
        loads.append(checkpoint)
        return original_loader(checkpoint, **kwargs,
            protein_bundle=LoadedEncoder('tiny/protein', None, model.protein_encoder, 12),
            molecule_bundle=LoadedEncoder('tiny/molecule', None, model.molecule_encoder, 16))
    def no_training(*args, **kwargs):
        raise AssertionError('Postselection must not train')
    monkeypatch.setattr(model_api, 'load_reward_model', tiny_loader)
    monkeypatch.setattr(RewardModelTrainer, 'train', no_training)
    monkeypatch.setattr(RewardModelTrainer, 'create_optimizer', no_training)
    before = {str(p):sha256(p) for p in directory.rglob('*') if p.is_file()}
    postselection.evaluate_worker(path, 0, 1, cpu_qualification=True)
    assert len(loads) == 1
    for split in ('val', 'test'):
        output = path.parent/'checkpoint-50'/split
        complete = postselection.read_json(output/'complete.json')
        assert complete['rows'] == 9 and complete['parameters_unchanged']
        if grid[3].config.contrastive_mask_policy == 'training_known':
            assert complete['known_positive_index']['source_rows'] == 6
            assert complete['known_positive_index']['heldout_labels_used'] is False
        else:
            assert complete['known_positive_index'] is None
        frame, _ = read_native_predictions(complete['prediction_manifest'])
        assert len(frame) == 9
        assert complete['metrics'][('eval' if split == 'val' else 'test')+'_macro_spearman'] is not None
        postselection.verify_evaluation(output, complete['source'], cpu_qualification=True)
        with pytest.raises(ValueError, match='provenance changed'):
            postselection.verify_evaluation(output, complete['source'])
    assert {str(p):sha256(p) for p in directory.rglob('*') if p.is_file()} == before
    postselection.evaluate_worker(path, 0, 1, cpu_qualification=True)
    assert len(loads) == 1  # Verified reuse does not even load the model.
    output = path.parent/'checkpoint-50/test'
    result = postselection.read_json(output/'complete.json')
    Path(result['prediction_manifest']).with_suffix('.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='payload differs'):
        postselection.evaluate_worker(path, 0, 1, cpu_qualification=True)


def test_scientific_worker_rejects_unapproved_device_before_reading_data(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    with pytest.raises(RuntimeError, match='authorized GPU'):
        postselection.evaluate_worker(Path('unused'), 0, 4)


def test_postselection_obeys_the_existing_training_lock(tmp_path, monkeypatch):
    import fcntl
    directory = tmp_path/'controlled'
    directory.mkdir()
    monkeypatch.setattr(postselection, 'ROOT', Path('/home/atabey/Prot2Mol'))
    monkeypatch.setattr(postselection, 'WORK', tmp_path)
    with (directory/'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            postselection.run_evaluation('unused')
