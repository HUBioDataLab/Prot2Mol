import copy
import json
import math
from pathlib import Path

import pytest

from experiments.paired_generalization import control_plan
from reward_model.training.config import load_reward_training_config

ROOT = Path(__file__).resolve().parents[2]


def noise_fixture():
    # Held-out values are deliberately implausible: they must have no effect.
    return {regime: dict(train=dict(noise_scale=dict(rms_replicate_difference=value)),
                        val=dict(noise_scale=dict(rms_replicate_difference=999.)),
                        test=dict(noise_scale=dict(rms_replicate_difference=-999.)))
            for regime, value in [('A', .78), ('B', .76)]}


def test_all_declared_configs_round_trip_and_preserve_matched_budgets(tmp_path):
    plan = control_plan.build_plan(ROOT, tmp_path, noise_fixture())
    assert plan['accounting'] == dict(settings=28, tuning_runs=168, confirmation_runs=112,
                                      total_runs=280, epochs_per_run=15, original_reference_reused=False)
    assert len({job['id'] for job in plan['tuning_jobs']}) == 168
    assert [(j['arm'], j['regime'], j['learning_rate_multiplier']) for j in plan['tuning_jobs'][:2]] == [('full','A',1.), ('full','B',1.)]
    for job in plan['tuning_jobs']:
        prepared = control_plan.materialize_job(job, tmp_path/'control', 'plan-digest')
        loaded = load_reward_training_config(prepared['config_path'])
        assert loaded.training.num_train_epochs == 15
        assert loaded.training.evaluate_test is False
        assert loaded.training.fixed_train_eval_assays == 2048
        assert loaded.training.per_device_train_batch_size * loaded.training.gradient_accumulation_steps * len(control_plan.GPU_UUIDS) == 192
        assert loaded.data.sample_all_assays is True
        assert loaded.training.learning_rate == pytest.approx(1e-4*job['learning_rate_multiplier'])
        assert loaded.training.protein_encoder_learning_rate == pytest.approx(1e-5*job['learning_rate_multiplier'])
        if job['arm'].startswith('loss_'):
            subset = job['arm'].split('_')[1]
            assert (loaded.model.ranking_loss_weight > 0) == ('R' in subset)
            assert (loaded.model.classification_loss_weight > 0) == ('B' in subset)
            assert (loaded.model.contrastive_loss_weight > 0) == ('C' in subset)
            assert loaded.model.ranking_readout == ('ranking_head' if 'R' in subset else ('activity_logit' if 'B' in subset else 'cosine'))
        if job['arm'] == 'pairwise_noise_soft':
            noise = plan['training_noise_rms'][job['regime']]
            assert loaded.model.ranking_pair_target_temperature == pytest.approx(math.sqrt(3)*noise/math.pi)
            assert loaded.model.ranking_affinity_margin == 0
    changed = noise_fixture()
    for regime in changed:
        changed[regime]['test']['noise_scale']['rms_replicate_difference'] = 12345.
    assert control_plan.build_plan(ROOT, tmp_path, changed) == plan


def test_prepared_job_is_immutable_and_confirmation_keeps_common_selected_lr(tmp_path):
    plan = control_plan.build_plan(ROOT, tmp_path, noise_fixture())
    job = plan['tuning_jobs'][0]
    control_plan.materialize_job(job, tmp_path/'control', 'plan-digest')
    assert control_plan.materialize_job(job, tmp_path/'control', 'plan-digest')['phase'] == 'tuning'
    changed = copy.deepcopy(job)
    changed['config']['training']['seed'] = 43
    with pytest.raises(RuntimeError, match='Refusing to change'):
        control_plan.materialize_job(changed, tmp_path/'control', 'plan-digest')
    arm = plan['arms'][0]
    for seed in (43, 44):
        for regime in ('A','B'):
            result = control_plan.make_config(plan['bases'][regime], arm, regime=regime, seed=seed,
                multiplier=.3, output=tmp_path/f'{regime}{seed}', training_noise_rms=plan['training_noise_rms'])
            assert result['training']['seed'] == seed
            assert result['training']['learning_rate'] == pytest.approx(3e-5)


def candidates():
    return [dict(learning_rate_multiplier=m, validation_macro_spearman=score,
                 validation_defined_fraction=1., test_macro_spearman=test)
            for m, score, test in [(.3,.2,.9),(1.,.3,.0),(3.,.1,1.)]]


def test_selector_uses_only_validation_and_lower_lr_ties():
    inputs = candidates()
    result = control_plan.select_learning_rate(inputs)
    assert result['learning_rate_multiplier'] == 1.
    assert 'test' not in json.dumps(result)
    for row in inputs:
        row['test_macro_spearman'] *= -100
    assert control_plan.select_learning_rate(inputs) == result
    inputs[0]['validation_macro_spearman'] = .3
    assert control_plan.select_learning_rate(inputs)['learning_rate_multiplier'] == .3
    inputs[0]['validation_macro_spearman'] = float('nan')
    inputs[1]['validation_defined_fraction'] = 0.
    assert control_plan.select_learning_rate(inputs)['learning_rate_multiplier'] == 3.
    inputs[2]['validation_macro_spearman'] = None
    with pytest.raises(ValueError, match='No finite'):
        control_plan.select_learning_rate(inputs)
    with pytest.raises(ValueError, match='All three'):
        control_plan.select_learning_rate(candidates()[:2])


def test_cache_snapshot_tracks_referenced_bytes_and_rejects_outside_paths(tmp_path):
    (tmp_path/'state.json').write_text(json.dumps({'_data_files':[{'filename':'data.arrow'}]}))
    (tmp_path/'dataset_info.json').write_text('{}')
    (tmp_path/'data.arrow').write_bytes(b'original-shard')
    original = control_plan.token_cache_hashes(tmp_path)
    (tmp_path/'data.arrow').write_bytes(b'modified-shard')
    assert control_plan.token_cache_hashes(tmp_path) != original
    (tmp_path/'state.json').write_text(json.dumps({'_data_files':[{'filename':'../elsewhere.arrow'}]}))
    with pytest.raises(ValueError, match='directly'):
        control_plan.token_cache_hashes(tmp_path)
