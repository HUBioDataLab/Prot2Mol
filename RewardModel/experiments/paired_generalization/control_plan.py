"""Preregistered configurations and validation-only selection for Tier 3.

This is the implementable control grid, not a claim that the entire MD is
covered: unit subsets, the clarified 2x2 design and conditional methods still
need their own manifests. Preparing a plan does not start any training.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import yaml

if __package__:
    from .run import ROOT, WORK, sha256, write_json
else:
    from run import ROOT, WORK, sha256, write_json

from reward_model.training.config import load_reward_training_config

VERSION = 1
LEARNING_RATE_MULTIPLIERS = (0.3, 1.0, 3.0)
TUNING_SEED = 42
CONFIRMATION_SEEDS = (43, 44)
GPU_UUIDS = (
    'GPU-970f3ca6-f824-616b-ec10-b972e15edb47',
    'GPU-8f691160-19aa-95bb-14b7-82706085df29',
    'GPU-bd4fd417-ac81-eb97-55b4-4cd60e19847a',
    'GPU-bf841640-8bdb-c7a0-ad1e-9fadf5635d3a',
)


def control_arms():
    arms = []

    def add(name, family, experiments, model=None, training=None):
        arms.append(dict(name=name, family=family, experiment_ids=experiments,
                         model=model or {}, training=training or {}))

    add('full', 'encoder_freedom', ['E33', 'E37', 'E38', 'E48'])
    # A declared regularization comparison precedes encoder/objective controls.
    # Each setting has the same LR budget; family-level budgets are explicit.
    add('weight_decay_0p001', 'regularization', ['E33'], training={'weight_decay': .001})
    add('weight_decay_0p1', 'regularization', ['E33'], training={'weight_decay': .1})
    add('frozen', 'encoder_freedom', ['E34', 'E48'],
        model={'freeze_protein_encoder': True, 'freeze_molecule_encoder': True})
    add('last_two_blocks', 'encoder_freedom', ['E35', 'E48'],
        model={'protein_unfrozen_last_layers': 2, 'molecule_unfrozen_last_layers': 2})
    for rank in (4, 16, 64):
        add(f'lora{rank}', 'encoder_freedom', ['E36', 'E48'],
            model={'encoder_lora_rank': rank, 'encoder_lora_alpha': 2.*rank})
    for subset in ('R', 'B', 'C', 'RB', 'RC', 'BC'):
        add(f'loss_{subset}', 'loss_subsets', ['E38'], model={
            'ranking_loss_weight': .25 if 'R' in subset else 0.,
            'classification_loss_weight': .25 if 'B' in subset else 0.,
            'contrastive_loss_weight': 1. if 'C' in subset else 0.,
            'ranking_readout': 'ranking_head' if 'R' in subset else ('activity_logit' if 'B' in subset else 'cosine'),
        })
    # Axis sweep: R stays .25. Four alternatives plus the shared full reference.
    for name, key, value in (
        ('bce_low', 'classification_loss_weight', .0625),
        ('bce_high', 'classification_loss_weight', 1.),
        ('contrastive_low', 'contrastive_loss_weight', .25),
        ('contrastive_high', 'contrastive_loss_weight', 4.),
    ):
        add(name, 'static_weight_sweep', ['E39'], model={key: value})
    add('unmasked', 'contrastive_controls', ['E42'], model={'contrastive_mask_policy': 'none'})
    add('training_known_mask', 'contrastive_controls', ['E42'], model={'contrastive_mask_policy': 'training_known'})
    add('fixed_scale', 'contrastive_controls', ['E43'], model={'learn_contrastive_scale': False})
    add('detach_contrastive', 'contrastive_controls', ['E44'], model={'contrastive_detach_encoders': True})
    add('listwise_normalized', 'ranking_controls', ['E45'], model={'ranking_normalize_per_group': True})
    add('assay_balanced', 'ranking_controls', ['E45'],
        model={'ranking_normalize_per_group': True, 'ranking_balance_assays': True})
    for weighting in ('top', 'uniform'):
        add(f'pairwise_{weighting}', 'ranking_controls', ['E46'],
            model={'ranking_objective': 'pairwise_logistic', 'ranking_position_weighting': weighting})
    add('pairwise_hard_zero_margin', 'ranking_controls', ['E47'],
        model={'ranking_objective': 'pairwise_logistic', 'ranking_position_weighting': 'uniform', 'ranking_affinity_margin': 0.})
    add('pairwise_noise_soft', 'ranking_controls', ['E47'], model={
        'ranking_objective': 'pairwise_logistic', 'ranking_position_weighting': 'uniform',
        'ranking_affinity_margin': 0., 'ranking_pair_target_temperature': 'training_replicate_rms',
    })
    return arms


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def job_id(arm, regime, seed, multiplier):
    if regime not in ('A', 'B') or seed not in (TUNING_SEED, *CONFIRMATION_SEEDS) or multiplier not in LEARNING_RATE_MULTIPLIERS:
        raise ValueError('Job is outside the declared regime/seed/LR grid')
    return f"{arm}__{regime}__seed{seed}__lr{str(multiplier).replace('.', 'p')}"


def make_config(base, arm, *, regime, seed, multiplier, output, training_noise_rms):
    result = copy.deepcopy(base)
    result['model'].update(arm['model'])
    if result['model'].get('ranking_pair_target_temperature') == 'training_replicate_rms':
        noise = float(training_noise_rms[regime])
        if not math.isfinite(noise) or noise <= 0:
            raise ValueError('Soft pair targets require a finite positive training noise RMS')
        result['model']['ranking_pair_target_temperature'] = math.sqrt(3.) * noise / math.pi
    result['training'].update(arm['training'])
    result['training'].update(output_dir=str(output), seed=seed,
        num_train_epochs=15, max_steps=None, eval_steps=None,
        per_device_train_batch_size=12, gradient_accumulation_steps=4,
        training_mode='multi_gpu', export_evaluation_predictions=True,
        fixed_train_eval_assays=2048, evaluate_test=False, save_total_limit=16,
        metric_for_best_model='eval_macro_spearman', greater_is_better=True)
    result['data']['sample_all_assays'] = True
    for key in ('learning_rate', 'encoder_learning_rate', 'protein_encoder_learning_rate',
                'molecule_encoder_learning_rate', 'projection_learning_rate'):
        if result['training'].get(key) is not None:
            result['training'][key] *= multiplier
    return result


def _base_config(path):
    bundle = load_reward_training_config(str(path))
    return dict(model=bundle.model.to_dict(), data=asdict(bundle.data), training=asdict(bundle.training))


def build_plan(root, work, noise_summary):
    root, work = Path(root), Path(work)
    paths = dict(A=root/'RewardModel/configs/reward_train_paired_generalization_random.yaml',
                 B=root/'RewardModel/configs/reward_train_paired_generalization_baseline.yaml')
    bases = {regime: _base_config(path) for regime, path in paths.items()}
    noise = {regime: noise_summary[regime]['train']['noise_scale']['rms_replicate_difference'] for regime in ('A', 'B')}
    arms, jobs = control_arms(), []
    # The base-LR matched A/B references run before the other tuning candidates.
    for arm in arms:
        for multiplier in (1., .3, 3.):
            for regime in ('A', 'B'):
                identity = job_id(arm['name'], regime, TUNING_SEED, multiplier)
                config = make_config(bases[regime], arm, regime=regime, seed=TUNING_SEED,
                                     multiplier=multiplier, output=work/'controlled/runs'/identity,
                                     training_noise_rms=noise)
                jobs.append(dict(id=identity, arm=arm['name'], regime=regime, seed=TUNING_SEED,
                                 learning_rate_multiplier=multiplier, phase='tuning', config=config))
    return dict(format_version=VERSION, spec_sha256=sha256(root/'Paired input ranking generalization.md'),
        base_config_sha256={key: sha256(path) for key, path in paths.items()},
        bases=bases, arms=arms, tuning_jobs=jobs, training_noise_rms=noise,
        training_noise_source='Only each regime train.noise_scale.rms_replicate_difference',
        learning_rate_multipliers=list(LEARNING_RATE_MULTIPLIERS), tuning_seed=TUNING_SEED,
        confirmation_seeds=list(CONFIRMATION_SEEDS), gpu_uuids=list(GPU_UUIDS),
        accounting=dict(settings=len(arms), tuning_runs=len(jobs), confirmation_runs=4*len(arms),
                        total_runs=10*len(arms), epochs_per_run=15,
                        original_reference_reused=False),
        shared_reference=dict(loss_RBC='full', original_weights='full', weight_decay_0p01='full',
                              learned_scale='full', reference_masks='full', original_ranking='full'),
        interpretation='Each setting gets three LR candidates on both splits, then seeds 43/44 at the same B-selected LR. Shared references are not repeated. Report family-level trial counts. Averaging and checkpoint-rule comparisons use the common selected LR to isolate those changes; they are not independently tuned algorithms.',
        not_covered=['E40 molecule-only/full-pair x leak-in/out 2x2: user confirmed axes; matched subset manifests pending',
                     'E41 whole-unit learning curves: subset manifests pending',
                     'additional B realizations: affordability assessment pending',
                     'Tier 4 adaptation/probes and applicable method-catalog experiments'])


def _write_immutable(path, text):
    path = Path(path)
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f'Refusing to change prepared artifact: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as handle:
            handle.write(text)


def materialize_job(job, directory, plan_sha256):
    config_path = Path(directory)/'configs'/f"{job['id']}.yaml"
    _write_immutable(config_path, yaml.safe_dump(job['config'], sort_keys=True))
    # Round-trip through the same production loader before declaring it ready.
    loaded = load_reward_training_config(str(config_path))
    if loaded.training.per_device_train_batch_size * loaded.training.gradient_accumulation_steps * len(GPU_UUIDS) != 192:
        raise RuntimeError('Controlled effective batch changed')
    metadata = {key: value for key, value in job.items() if key != 'config'}
    metadata.update(config_path=str(config_path), config_sha256=sha256(config_path),
                    output_dir=loaded.training.output_dir, plan_sha256=plan_sha256)
    _write_immutable(Path(directory)/'jobs'/f"{job['id']}.json",
                     json.dumps(metadata, indent=2, sort_keys=True)+'\n')
    return metadata


def token_cache_hashes(directory):
    """Freeze the actual Arrow shards referenced by a saved Dataset's state."""
    directory = Path(directory)
    state = json.loads((directory/'state.json').read_text())
    names = ['state.json', 'dataset_info.json'] + [item['filename'] for item in state['_data_files']]
    if len(names) < 3 or len(set(names)) != len(names):
        raise ValueError('Token cache must reference distinct nonempty Arrow shards')
    if any(Path(name).name != name for name in names):
        raise ValueError('Token shards must belong directly to their saved dataset directory')
    return {name: sha256(directory/name) for name in names}


def prepare():
    if __package__:
        from .prepare_random import archive_directory
    else:
        from prepare_random import archive_directory
    if ROOT != Path('/home/atabey/Prot2Mol'):
        raise RuntimeError('Runtime preparation is restricted to Atum Prot2Mol')
    directory = WORK/'controlled'
    archive_directory(directory)
    noise_path = WORK/'tier1/replicates/noise_summary.json'
    plan = build_plan(ROOT, WORK, json.loads(noise_path.read_text()))
    plan['noise_summary_sha256'] = sha256(noise_path)
    plan['source_data_sha256'] = {
        regime: {split: sha256(Path(plan['bases'][regime]['data'][f'{split}_parquet_path']))
                 for split in ('train', 'val', 'test')} for regime in ('A', 'B')}
    plan['tokenized_data_sha256'] = {
        regime: {split: token_cache_hashes(Path(plan['bases'][regime]['data']['tokenized_dataset_dir'])/f'{split}_examples')
                 for split in ('train', 'val', 'test')} for regime in ('A', 'B')}
    path = directory/'plan.json'
    _write_immutable(path, json.dumps(plan, indent=2, sort_keys=True)+'\n')
    digest = sha256(path)
    for job in plan['tuning_jobs']:
        materialize_job(job, directory, digest)
    write_json(directory/'preparation_complete.json', dict(plan_sha256=digest,
        tuning_configurations=len(plan['tuning_jobs']), accounting=plan['accounting'],
        action='prepared only; no training launched'))
    return plan['accounting']


def select_learning_rate(candidates):
    """Read only the explicitly supplied validation results, never test scores."""
    if len(candidates) != len(LEARNING_RATE_MULTIPLIERS) or {c['learning_rate_multiplier'] for c in candidates} != set(LEARNING_RATE_MULTIPLIERS):
        raise ValueError('All three distinct B validation candidates are required')
    valid = [c for c in candidates if c['validation_macro_spearman'] is not None
             and math.isfinite(c['validation_macro_spearman'])
             and c['validation_defined_fraction'] is not None
             and 0 < c['validation_defined_fraction'] <= 1]
    if not valid:
        raise ValueError('No finite, defined B validation candidate')
    winner = max(valid, key=lambda c: (c['validation_macro_spearman'], -c['learning_rate_multiplier']))
    return dict(learning_rate_multiplier=winner['learning_rate_multiplier'],
                validation_macro_spearman=winner['validation_macro_spearman'],
                validation_defined_fraction=winner['validation_defined_fraction'],
                criterion='maximum B validation macro Spearman; ties choose lower LR; coverage reported',
                candidates=[{key: c[key] for key in ('learning_rate_multiplier', 'validation_macro_spearman', 'validation_defined_fraction')} for c in candidates])


def validation_candidate(metadata):
    """Read a completed four-rank fit's validation evidence, never test metrics."""
    if sha256(Path(metadata['config_path'])) != metadata['config_sha256']:
        raise ValueError('Prepared job provenance changed')
    output = Path(metadata['output_dir'])
    summary = json.loads((output/'experiment_summary.json').read_text())
    if summary['config_sha256'] != metadata['config_sha256'] or summary['test_evaluated'] or summary['test_metrics']:
        raise ValueError('Tuning completion has changed provenance or already scored test data')
    for rank in range(len(GPU_UUIDS)):
        rank_done = json.loads((output/f'rank_{rank}_complete.json').read_text())
        if rank_done['rank'] != rank or rank_done['world_size'] != len(GPU_UUIDS) or rank_done['config_sha256'] != metadata['config_sha256']:
            raise ValueError('All four ranks must finish the same tuning configuration')
    retained = json.loads((output/'retained/retention.json').read_text())
    if retained['config_sha256'] != metadata['config_sha256']:
        raise ValueError('Retention belongs to another configuration')
    selection = retained['selections']['best_macro']
    snapshot_path = output/'prediction_exports/trajectory'/f"eval_step{selection['step']:08d}.json"
    snapshot = json.loads(snapshot_path.read_text())
    if sha256(snapshot_path.with_suffix('.npz')) != snapshot['predictions_sha256']:
        raise ValueError('Selected validation predictions changed')
    if snapshot['metrics']['eval_macro_spearman'] != selection['metric']:
        raise ValueError('Retention and selected snapshot disagree')
    evidence = dict(job_id=metadata['id'], summary_sha256=sha256(output/'experiment_summary.json'),
                    validation_snapshot_sha256=sha256(snapshot_path))
    candidate = dict(learning_rate_multiplier=metadata['learning_rate_multiplier'],
        validation_macro_spearman=selection['metric'],
        validation_defined_fraction=snapshot['metrics']['eval_spearman_defined_fraction'])
    return candidate,evidence


def confirm(arm_name):
    directory = WORK/'controlled'
    plan_path = directory/'plan.json'
    plan = json.loads(plan_path.read_text())
    digest = sha256(plan_path)
    arm = next(a for a in plan['arms'] if a['name'] == arm_name)
    candidates, evidence = [], []
    for job in (j for j in plan['tuning_jobs'] if j['arm'] == arm_name):
        metadata_path = directory/'jobs'/f"{job['id']}.json"
        metadata = json.loads(metadata_path.read_text())
        if metadata['plan_sha256'] != digest:
            raise ValueError('Prepared job provenance changed')
        candidate,source = validation_candidate(metadata)
        evidence.append(source)
        if job['regime'] == 'B':
            candidates.append(candidate)
    selected = dict(arm=arm_name, plan_sha256=digest, **select_learning_rate(candidates), evidence=evidence)
    selection_path = directory/'selections'/f'{arm_name}.json'
    _write_immutable(selection_path, json.dumps(selected, indent=2, sort_keys=True)+'\n')
    jobs = []
    for seed in CONFIRMATION_SEEDS:
        for regime in ('A', 'B'):
            multiplier = selected['learning_rate_multiplier']
            identity = job_id(arm_name, regime, seed, multiplier)
            config = make_config(plan['bases'][regime], arm, regime=regime, seed=seed, multiplier=multiplier,
                                 output=WORK/'controlled/runs'/identity, training_noise_rms=plan['training_noise_rms'])
            job = dict(id=identity, arm=arm_name, regime=regime, seed=seed,
                       learning_rate_multiplier=multiplier, phase='confirmation', config=config,
                       selection_sha256=sha256(selection_path))
            jobs.append(materialize_job(job, directory, digest))
    return dict(selection=selected, confirmation_jobs=jobs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='action', required=True)
    subparsers.add_parser('prepare')
    selection = subparsers.add_parser('confirm')
    selection.add_argument('--arm', required=True, choices=[a['name'] for a in control_arms()])
    args = parser.parse_args()
    print(json.dumps(prepare() if args.action == 'prepare' else confirm(args.arm), indent=2))
