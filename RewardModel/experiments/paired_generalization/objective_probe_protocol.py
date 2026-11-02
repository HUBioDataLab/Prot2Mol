"""Freeze E56/E57 reference checkpoints and nominal steps, without selecting a run.

This prepares rules for the existing numerical worker. It does not make a GPU
queue, load a checkpoint, choose an LR, or waive the MD's Tier 3 prerequisite.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from reward_model.training.trainer import RewardModelTrainer

if __package__:
    from .run import ROOT, WORK, sha256
    from .control_plan import GPU_UUIDS, LEARNING_RATE_MULTIPLIERS, digest_json, job_id, _write_immutable
    from . import objective_probes as probes
    from . import objective_probe_batches as batches
    from . import verify_objective_probe_batches as reconstruction
else:
    from run import ROOT, WORK, sha256
    from control_plan import GPU_UUIDS, LEARNING_RATE_MULTIPLIERS, digest_json, job_id, _write_immutable
    import objective_probes as probes
    import objective_probe_batches as batches
    import verify_objective_probe_batches as reconstruction

SEEDS = (42, 43, 44)
SCOPES = ('encoders', 'interaction', 'shared')
RATE_FIELDS = ('encoder_learning_rate', 'protein_encoder_learning_rate',
               'molecule_encoder_learning_rate', 'projection_learning_rate')


def native_rate(name, training):
    return RewardModelTrainer._component_learning_rate(name,
        default_learning_rate=training['learning_rate'],
        **{key: training.get(key) for key in RATE_FIELDS})


def step_rates(training):
    """Use nominal Trainer rates, including its side/common/default fallbacks.

The numerical worker has one interaction rate. A configuration that gives its
projections and fusion different rates cannot be represented by that worker.
"""
    rates = {side: native_rate(side+'.weight', training)
             for side in ('protein_encoder', 'molecule_encoder')}
    interaction = {native_rate(root+'.weight', training) for root in probes.INTERACTION_ROOTS}
    if len(interaction) != 1:
        raise ValueError('Unequal interaction rates require a different numerical worker')
    rates['interaction'] = interaction.pop()
    if any(not math.isfinite(value) or value <= 0 for value in rates.values()):
        raise ValueError('Nominal probe rates must be finite and positive')
    return rates


def validate_parameter_rates(names, training):
    """Scientific callers must check the actual loaded model's parameter names."""
    rates = step_rates(training)
    categories = set()
    for name in names:
        category = probes.parameter_scope(name)
        if category != 'private':
            categories.add(category)
            if native_rate(name, training) != rates[category]:
                raise ValueError('Native parameter rate differs from the frozen probe rate: '+name)
    if categories != set(rates):
        raise ValueError('Expected both encoder sides and interaction parameters')
    return rates


def build_protocol(primary, frozen):
    if (primary['tuning_seed'] != 42 or primary['confirmation_seeds'] != [43, 44]
        or primary['gpu_uuids'] != list(GPU_UUIDS)
        or primary['learning_rate_multipliers'] != list(LEARNING_RATE_MULTIPLIERS)
        or primary['spec_sha256'] != frozen['spec_sha256']):
        raise ValueError('Expected the registered primary grid and governing MD')
    if set(frozen['regimes']) != {'A', 'B'}:
        raise ValueError('Both own-regime cold context sets are required')
    for regime, record in frozen['regimes'].items():
        if record['batch_pairs'] != batches.BATCH_PAIRS or set(record['contexts']) != {'train', 'cold_validation'}:
            raise ValueError('Expected the frozen 32 diagnostic pairs per regime')
        for role, context in record['contexts'].items():
            if any(context[key] != value for key, value in dict(
                regime=regime, role=role, context_seed=42, epoch=0, local_items=12,
                selected_batches=batches.BATCH_PAIRS).items()):
                raise ValueError('Frozen context role, seed, epoch or local batch changed')
    expected = {job_id('full', regime, 42, lr)
                for regime in ('A', 'B') for lr in LEARNING_RATE_MULTIPLIERS}
    candidates = {}
    for job in primary['tuning_jobs']:
        if job['arm'] != 'full':
            continue
        if (job['id'] not in expected or job['id'] in candidates or job['seed'] != 42
            or job['id'] != job_id('full', job['regime'], job['seed'], job['learning_rate_multiplier'])):
            raise ValueError('Invalid or duplicated full-reference candidate')
        config = job['config']; training, model = config['training'], config['model']
        if any(training.get(key) != value for key, value in dict(
            seed=42, num_train_epochs=15, max_steps=None, per_device_train_batch_size=12,
            gradient_accumulation_steps=4, evaluate_test=False,
            metric_for_best_model='eval_macro_spearman', greater_is_better=True).items()):
            raise ValueError('Full-reference training or checkpoint rule changed')
        weights = {key: model[field] for key, (_, field) in probes.OBJECTIVES.items()}
        if weights != dict(R=.25, B=.25, C=1.) or model['pair_scoring_mode'] != 'mlp_contrastive':
            raise ValueError('Expected the original full R+B+C reference objectives')
        candidates[job['id']] = dict(regime=job['regime'],
            learning_rate_multiplier=job['learning_rate_multiplier'],
            config_sha256=digest_json(config), nominal_step_rates=step_rates(training), weights=weights)
    if set(candidates) != expected:
        raise ValueError('All six full-reference tuning configurations are required')
    slots = [dict(regime=regime, seed=seed,
        candidate_job_ids=[job_id('full', regime, seed, lr) for lr in LEARNING_RATE_MULTIPLIERS],
        selected_job_id=None, selected_checkpoint=None, batch_pairs=batches.BATCH_PAIRS)
        for regime in ('A', 'B') for seed in SEEDS]
    return dict(format_version=1, experiments=['E56', 'E57'], seeds=list(SEEDS),
        base_policy='Full R+B+C A/B references at the common B-validation-selected LR, seeds42/43/44; each fit uses its own best validation-macro checkpoint. Same reference policy as head adaptation. No alternative-arm or test-based winner.',
        selection_requirement='Resolve controlled/selections/full.json with postselection.verify_selection after all registered tuning fits; then verify each existing fit, retained checkpoint and training qualification. This protocol makes no selection.',
        candidates=candidates, checkpoint_slots=slots, selection_resolved=False,
        context_policy='Use the saved own-regime training/cold-validation pairs in pair_index order. One seed42 epoch-zero context set is shared across model seeds. Never substitute global B validation for A held-out data.',
        step_policy='Exactly one stateless weighted-SGD step per source objective and scope, using nominal configured Trainer component rates at the selected LR. No additional step-size grid, scheduler value, clipping, decay or optimizer-state replay. Private parameters remain fixed.',
        precision='float32', parameter_dtype='float32', autocast=False,
        allow_tf32=False, dropout=False, scopes=list(SCOPES),
        source_objectives=list(probes.OBJECTIVES), target_objectives=list(probes.OBJECTIVES),
        reporting_policy='Report per-block raw/weighted norms and pairwise cosines, then per-source/scope/target absolute loss changes and relative transfer. Preserve every frozen batch, undefined state and coverage. Summarize batch distributions separately for each regime/model seed; compare seeds on identical contexts. Batches can share units: no independent-batch confidence intervals. Unchanged losses do not establish absence of interference.',
        interpretation='Conditional local diagnostics of the selected models. Positive transfer means lower cold-batch objective loss after this particular nominal SGD step; it is not trained-model improvement or evidence that gradient conflict causes the generalization gap.',
        scientific_gates=['Tier 3 experiments and selected-model evaluations complete',
            'Verify selected fits, retained checkpoint/configuration/source hashes and actual parameter-rate mapping',
            'Verify complete training/validation token payloads and frozen reconstruction evidence',
            'Qualify production float32 parameters/forwards with TF32 disabled, GPU memory use, repeatability and effective step sizes; report unresolved precision limits rather than tune steps on observed transfer',
            'Acquire the existing shared GPU lock and expose only an assigned authorized GPU UUID'],
        accounting=dict(existing_model_slots=len(slots), diagnostic_pairs=len(slots)*batches.BATCH_PAIRS,
            nominal_steps_per_source_and_scope=1, additional_fits=0, additional_lr_candidates=0),
        model_inference=False, scientific_probe_results=0, production_runtime_qualified=False,
        executable_gpu_queue=False)


def prepare():
    if (ROOT != Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
        or os.environ.get('ACCELERATE_USE_CPU', '').lower() != 'true'
        or os.environ.get('WANDB_MODE') != 'disabled'):
        raise RuntimeError('Use CPU-only, W&B-disabled preparation inside Atum Prot2Mol')
    directory, frozen = reconstruction.inputs()
    primary_path = WORK/'controlled/plan.json'
    batches.checked(primary_path, frozen['input_sha256'][str(primary_path)])
    primary = batches.read(primary_path)
    batches.checked(ROOT/'Paired input ranking generalization.md', primary['spec_sha256'])
    verified_path = directory/'reconstruction_verification.json'
    verified = batches.read(verified_path)
    if (verified['complete_sha256'] != sha256(directory/'complete.json')
        or verified['verifier_sha256'] != sha256(Path(reconstruction.__file__))
        or verified['encoder_source_sha256'] != sha256(ROOT/'RewardModel/reward_model/model/encoders.py')
        or set(verified['contexts']) != {'A_train', 'A_cold_validation', 'B_train', 'B_cold_validation'}
        or verified['batches_verified'] != 128
        or not all(c['exact_native_features'] and c['exact_native_collator_tensors'] and c['source_row_table_verified']
                   for c in verified['contexts'].values())
        or any(verified[key] for key in ('model_weights_loaded', 'model_inference', 'new_fits_created', 'scientific_probe_results', 'test_data_read'))):
        raise ValueError('Production batch reconstruction evidence is incomplete or inconsistent')
    sources = [Path(__file__), Path(probes.__file__), Path(batches.__file__), Path(reconstruction.__file__),
        ROOT/'RewardModel/reward_model/training/trainer.py', Path(__file__).with_name('control_plan.py')]
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    inputs = dict(frozen['input_sha256'])
    inputs.update({str(directory/name): digest for name, digest in frozen['artifacts_sha256'].items()})
    inputs.update({str(path): sha256(path) for path in (directory/'complete.json', verified_path)})
    result = build_protocol(primary, frozen)
    result.update(spec_sha256=primary['spec_sha256'], primary_plan_sha256=sha256(primary_path),
        input_sha256=inputs, source_sha256=source_hashes, packages=frozen['packages'])
    for path, digest in inputs.items(): batches.checked(path, digest)
    for path, digest in source_hashes.items(): batches.checked(ROOT/path, digest)
    output = WORK/'tier4/objective_probe_protocol'
    plan_path = output/'plan.json'
    _write_immutable(plan_path, json.dumps(result, indent=2, sort_keys=True)+'\n')
    done = dict(plan_sha256=sha256(plan_path), accounting=result['accounting'],
        source_files_verified=len(source_hashes), input_files_verified=len(inputs),
        selection_resolved=False, new_fits_created=0, model_inference=False,
        scientific_probe_results=0, executable_gpu_queue=False)
    _write_immutable(output/'complete.json', json.dumps(done, indent=2, sort_keys=True)+'\n')
    return done


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(prepare(), indent=2), flush=True)
