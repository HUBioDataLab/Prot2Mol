import copy
from dataclasses import replace

import pytest
import torch

from experiments.paired_generalization import adaptation_fit as fit
from experiments.paired_generalization.qualify_controls import make_model, make_batch
from experiments.paired_generalization.run import sha256
from reward_model.training.trainer import RewardModelTrainer
from test_paired_generalization_checkpoint_evaluation import examples_fixture, config_fixture


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('ACCELERATE_USE_CPU', 'true')
    monkeypatch.setenv('WANDB_MODE', 'disabled')
    model = make_model(None, {})
    examples = examples_fixture(model)
    config = config_fixture(model, tmp_path)
    config = replace(config, training=replace(config.training, num_train_epochs=2,
        per_device_train_batch_size=2, gradient_accumulation_steps=2,
        learning_rate=.001, logging_steps=1, evaluate_test=False))
    return model, examples, config


@pytest.mark.parametrize('scope', fit.SCOPES)
def test_freeze_scope_keeps_feature_dropout_fixed_and_restores_modes_on_failure(inputs, scope):
    model, _, _ = inputs
    model.protein_encoder.eval()
    flags = {name:p.requires_grad for name,p in model.named_parameters()}
    modes = {module:module.training for module in model.modules()}
    weights = fit.parameter_digest(model)
    with pytest.raises(RuntimeError, match='injected'):
        with fit.adaptation_scope(model, scope) as names:
            for _ in range(2):
                model.eval()
                model.train()
                assert not model.protein_encoder.training and not model.molecule_encoder.training
                assert not model.protein_projection.training and not model.molecule_projection.training
                assert not model.projection_dropout.training
                assert model.ranking_head.training and model.classification_head.training
                assert model.fusion.training == (scope == 'heads_cross_attention')
            assert {name for name,p in model.named_parameters() if p.requires_grad} == set(names)
            assert not any(name.startswith(('protein_encoder.', 'molecule_encoder.',
                                            'protein_projection.', 'molecule_projection.')) for name in names)
            raise RuntimeError('injected failure')
    assert fit.parameter_digest(model) == weights
    assert {name:p.requires_grad for name,p in model.named_parameters()} == flags
    assert {module:module.training for module in model.modules()} == modes


@pytest.mark.parametrize('scope', fit.SCOPES)
def test_native_support_training_updates_only_scoped_weights_and_round_trips(tmp_path, inputs, monkeypatch, scope):
    model, examples, config = inputs
    original_config = copy.deepcopy(config)
    original = {name:tensor.clone() for name,tensor in model.state_dict().items()}
    base = fit.parameter_digest(model)

    def forbidden(*args, **kwargs):
        pytest.fail('Support fitting must not evaluate queries, select a best model or save the full model')

    monkeypatch.setattr(RewardModelTrainer, 'evaluate', forbidden)
    monkeypatch.setattr(RewardModelTrainer, '_load_best_model', forbidden)
    monkeypatch.setattr(RewardModelTrainer, 'save_model', forbidden)
    output = tmp_path/'fit'
    result = fit.fit_support(model, examples, config, output, scope=scope,
                             base_parameter_sha256=base, cpu=True)
    assert config == original_config
    assert result['epochs'] == 2 and result['global_step'] == 2
    assert result['frozen_parameters_unchanged'] and not result['selection_performed']
    assert not result['query_examples_received'] and result['cpu_qualification']
    assert [s['epoch'] for s in result['snapshots']] == [1, 2]
    names = set(result['trainable_parameters'])
    changed = {name for name,tensor in model.state_dict().items() if not torch.equal(tensor, original[name])}
    assert changed and changed <= names
    assert any(name.startswith('ranking_head.') for name in changed)
    assert any(name.startswith('fusion.') for name in changed) == (scope == 'heads_cross_attention')
    actual_optimizer_names = [name for group in result['optimizer_groups'] for name in group['names']]
    assert set(actual_optimizer_names) == names and len(actual_optimizer_names) == len(names)
    assert not list(output.glob('checkpoint-*')) and not (output/'pytorch_model.bin').exists()
    snapshot = result['snapshots'][-1]
    fresh = make_model(None, {})
    fit.restore_adapter(fresh, output/snapshot['file'], snapshot['sha256'])
    assert fit.parameter_digest(fresh) == fit.parameter_digest(model)
    batch = make_batch(model, torch.device('cpu'), False)
    model.eval(); fresh.eval()
    with torch.no_grad():
        expected, actual = model(**batch), fresh(**batch)
    for key in ('ranking_score', 'activity_logits', 'cosine_similarity'):
        torch.testing.assert_close(getattr(actual, key), getattr(expected, key), rtol=0, atol=0)
    with pytest.raises(ValueError, match='exact unchanged base'):
        fit.fit_support(model, examples, config, tmp_path/'carryover', scope=scope,
                        base_parameter_sha256=base, cpu=True)
    with pytest.raises(RuntimeError, match='Existing adaptation output'):
        fit.fit_support(make_model(None, {}), examples, config, output, scope=scope,
                        base_parameter_sha256=base, cpu=True)


def test_low_k_without_ranking_supervision_still_trains_support_bce(tmp_path, inputs):
    model, examples, config = inputs
    config = replace(config, training=replace(config.training, num_train_epochs=1))
    # Two observations cannot enter the three-item headline ranking objective.
    support = examples.select([0, 1])
    result = fit.fit_support(model, support, config, tmp_path/'small', scope='heads',
        base_parameter_sha256=fit.parameter_digest(model), cpu=True)
    assert result['support_assays'] == 1 and result['support_rows'] == 2
    assert result['sampler_stats']['num_ranking_lists'] == 0
    assert result['global_step'] == 1 and result['snapshots'][0]['epoch'] == 1


def test_disabled_objectives_and_fixed_scale_are_not_reenabled():
    model = make_model(None, dict(ranking_loss_weight=0., contrastive_loss_weight=0.,
                                  ranking_readout='activity_logit', learn_contrastive_scale=False))
    for scope in fit.SCOPES:
        with fit.adaptation_scope(model, scope) as names:
            assert all(not name.startswith(('ranking_head.', 'contrastive_', 'logit_scale')) for name in names)
            assert not model.logit_scale.requires_grad


def test_restore_rejects_rehashed_out_of_scope_or_nonfinite_weights_atomically(tmp_path, inputs):
    model, examples, config = inputs
    base = fit.parameter_digest(model)
    result = fit.fit_support(model, examples, config, tmp_path/'fit', scope='heads',
                            base_parameter_sha256=base, cpu=True, save_epochs=[2])
    assert [r['epoch'] for r in result['snapshots']] == [2]
    payload = torch.load(tmp_path/'fit/epoch02.pt', weights_only=True)
    for kind in ('scope', 'nonfinite', 'shape'):
        bad = copy.deepcopy(payload)
        first = sorted(bad['parameters'])[0]
        if kind == 'scope':
            bad['parameters']['protein_encoder.unregistered'] = torch.zeros(1)
        elif kind == 'nonfinite':
            bad['parameters'][first].fill_(float('nan'))
        else:
            bad['parameters'][first] = torch.zeros(999)
        path = tmp_path/f'{kind}.pt'
        torch.save(bad, path)
        fresh = make_model(None, {})
        with pytest.raises(ValueError, match='scope|Invalid adaptation tensor'):
            fit.restore_adapter(fresh, path, sha256(path))
        assert fit.parameter_digest(fresh) == base
    with pytest.raises(ValueError, match='checksum'):
        fit.restore_adapter(make_model(None, {}), tmp_path/'fit/epoch02.pt', '0'*64)


def test_support_preflight_rejects_wrong_base_or_missing_positive_index(tmp_path, inputs):
    model, examples, config = inputs
    with pytest.raises(ValueError, match='exact unchanged base'):
        fit.fit_support(model, examples, config, tmp_path/'bad_base', scope='heads',
                        base_parameter_sha256='0'*64, cpu=True)
    masked = make_model(None, dict(contrastive_mask_policy='training_known'))
    config = replace(config, model=masked.config)
    with pytest.raises(ValueError, match='support-safe positive index'):
        fit.fit_support(masked, examples, config, tmp_path/'missing_index', scope='heads',
                        base_parameter_sha256=fit.parameter_digest(masked), cpu=True)
    assert not (tmp_path/'bad_base').exists() and not (tmp_path/'missing_index').exists()


@pytest.mark.parametrize('epochs', [[1], [2, 1], [1, 1, 2], [0, 2], [1.5, 2]])
def test_rejects_ambiguous_snapshot_horizons_before_fitting(tmp_path, inputs, epochs):
    model, examples, config = inputs
    with pytest.raises(ValueError, match='Save epochs'):
        fit.fit_support(model, examples, config, tmp_path/'fit', scope='heads',
                        base_parameter_sha256=fit.parameter_digest(model), save_epochs=epochs, cpu=True)
    assert not (tmp_path/'fit').exists()
