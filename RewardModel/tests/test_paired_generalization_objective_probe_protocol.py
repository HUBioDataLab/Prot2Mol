import copy

import pytest

from experiments.paired_generalization import objective_probe_protocol as protocol
from experiments.paired_generalization.control_plan import GPU_UUIDS, job_id
from experiments.paired_generalization.qualify_controls import make_model


def fixture_inputs():
    primary = dict(tuning_seed=42, confirmation_seeds=[43, 44], gpu_uuids=list(GPU_UUIDS),
        learning_rate_multipliers=[.3, 1., 3.], spec_sha256='spec', tuning_jobs=[])
    for regime in ('A', 'B'):
        for lr in (.3, 1., 3.):
            config = dict(model=dict(pair_scoring_mode='mlp_contrastive', ranking_loss_weight=.25,
                classification_loss_weight=.25, contrastive_loss_weight=1.),
                training=dict(seed=42, num_train_epochs=15, max_steps=None,
                    per_device_train_batch_size=12, gradient_accumulation_steps=4, evaluate_test=False,
                    metric_for_best_model='eval_macro_spearman', greater_is_better=True,
                    learning_rate=lr*1e-4, encoder_learning_rate=lr*1e-5,
                    projection_learning_rate=lr*1e-4))
            primary['tuning_jobs'].append(dict(id=job_id('full', regime, 42, lr), arm='full',
                regime=regime, seed=42, learning_rate_multiplier=lr, config=config))
    frozen = dict(spec_sha256='spec', regimes={regime: dict(batch_pairs=32, contexts={
        role: dict(regime=regime, role=role, context_seed=42, epoch=0, local_items=12, selected_batches=32)
        for role in ('train', 'cold_validation')}) for regime in ('A', 'B')})
    return primary, frozen


def test_six_symbolic_reference_slots_do_not_select_or_add_fits():
    primary, frozen = fixture_inputs()
    before = copy.deepcopy((primary, frozen))
    result = protocol.build_protocol(primary, frozen)
    assert (primary, frozen) == before
    assert result['accounting'] == dict(existing_model_slots=6, diagnostic_pairs=192,
        nominal_steps_per_source_and_scope=1, additional_fits=0, additional_lr_candidates=0)
    assert {(r['regime'], r['seed']) for r in result['checkpoint_slots']} == {
        (r, s) for r in ('A', 'B') for s in (42, 43, 44)}
    assert not result['selection_resolved'] and not result['executable_gpu_queue']
    assert result['precision'] == 'float32' and not result['autocast']
    for slot in result['checkpoint_slots']:
        assert slot['selected_job_id'] is slot['selected_checkpoint'] is None
        assert slot['candidate_job_ids'] == [job_id('full', slot['regime'], slot['seed'], lr) for lr in (.3, 1., 3.)]
    for candidate in result['candidates'].values():
        lr = candidate['learning_rate_multiplier']
        assert candidate['nominal_step_rates'] == dict(protein_encoder=lr*1e-5,
            molecule_encoder=lr*1e-5, interaction=lr*1e-4)


def test_actual_architecture_rate_mapping_and_optional_fallbacks():
    names = list(dict(make_model(None, {}).named_parameters()))
    training = dict(learning_rate=1e-4, encoder_learning_rate=1e-5,
        protein_encoder_learning_rate=2e-5, molecule_encoder_learning_rate=None,
        projection_learning_rate=None)
    assert protocol.validate_parameter_rates(names, training) == dict(
        protein_encoder=2e-5, molecule_encoder=1e-5, interaction=1e-4)
    training['protein_encoder_learning_rate'] = 0.  # Trainer's explicit fallback semantics.
    assert protocol.step_rates(training)['protein_encoder'] == 1e-5
    training['encoder_learning_rate'] = None
    assert protocol.validate_parameter_rates(names, training) == dict(
        protein_encoder=1e-4, molecule_encoder=1e-4, interaction=1e-4)


def test_unequal_projection_and_fusion_rates_are_not_silently_averaged():
    with pytest.raises(ValueError, match='Unequal interaction'):
        protocol.step_rates(dict(learning_rate=1e-4, projection_learning_rate=1e-5))


@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf')])
def test_invalid_resolved_nominal_rates_are_rejected(value):
    with pytest.raises(ValueError, match='finite and positive|Unequal interaction'):
        protocol.step_rates(dict(learning_rate=value))


def test_unknown_or_incomplete_model_parameter_map_is_rejected():
    training = dict(learning_rate=1e-4)
    with pytest.raises(ValueError, match='Unregistered parameter scope'):
        protocol.validate_parameter_rates(['unknown.weight'], training)
    with pytest.raises(ValueError, match='both encoder sides'):
        protocol.validate_parameter_rates(['protein_encoder.weight', 'ranking_head.weight'], training)


@pytest.mark.parametrize('change', ['missing_candidate', 'duplicate_candidate', 'loss_selection',
    'test_selection', 'different_objective', 'changed_context', 'different_spec', 'different_seed'])
def test_changed_reference_or_context_contract_is_rejected(change):
    primary, frozen = fixture_inputs()
    if change == 'missing_candidate': primary['tuning_jobs'].pop()
    if change == 'duplicate_candidate': primary['tuning_jobs'].append(primary['tuning_jobs'][0])
    if change == 'loss_selection': primary['tuning_jobs'][0]['config']['training']['metric_for_best_model'] = 'eval_loss'
    if change == 'test_selection': primary['tuning_jobs'][0]['config']['training']['evaluate_test'] = True
    if change == 'different_objective': primary['tuning_jobs'][0]['config']['model']['ranking_loss_weight'] = 1.
    if change == 'changed_context': frozen['regimes']['A']['contexts']['cold_validation']['context_seed'] = 43
    if change == 'different_spec': frozen['spec_sha256'] = 'other'
    if change == 'different_seed': primary['confirmation_seeds'] = [43, 45]
    with pytest.raises(ValueError): protocol.build_protocol(primary, frozen)
