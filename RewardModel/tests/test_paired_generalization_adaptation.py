import pytest
import torch
from transformers import EsmConfig, EsmModel, RobertaConfig, RobertaModel

from conftest import DummyTokenizer
from reward_model.model import LoadedEncoder, RewardModel, RewardModelConfig, load_reward_model, save_reward_model
from reward_model.model.adaptation import LowRankLinear


def bundles():
    torch.manual_seed(5)
    protein = EsmModel(EsmConfig(vocab_size=33, hidden_size=12, num_hidden_layers=3,
        num_attention_heads=3, intermediate_size=24, pad_token_id=1, mask_token_id=32,
        hidden_dropout_prob=.1, attention_probs_dropout_prob=0., token_dropout=False), add_pooling_layer=False)
    molecule = RobertaModel(RobertaConfig(vocab_size=33, hidden_size=16, num_hidden_layers=3,
        num_attention_heads=4, intermediate_size=32, pad_token_id=1,
        hidden_dropout_prob=.1, attention_probs_dropout_prob=0.), add_pooling_layer=False)
    return (LoadedEncoder('protein/tiny', DummyTokenizer(1), protein, 12),
            LoadedEncoder('molecule/tiny', DummyTokenizer(1), molecule, 16))


def model(**changes):
    protein, molecule = bundles()
    torch.manual_seed(29)
    settings = dict(protein_model_name_or_path='protein/tiny',
        molecule_model_name_or_path='molecule/tiny', fusion_hidden_dim=12,
        fusion_num_heads=3, fusion_attention_backend='sdpa', dropout=0.,
        pair_scoring_mode='mlp_contrastive', ranking_head_hidden_dims=(8,),
        classification_head_hidden_dims=(8,), ranking_loss_weight=.25,
        classification_loss_weight=.25, contrastive_loss_weight=1.,
        contrastive_active_threshold=6., contrastive_strict_active_only=True)
    settings.update(changes)
    config = RewardModelConfig(**settings)
    return RewardModel(config, protein_bundle=protein, molecule_bundle=molecule)


def batch():
    proteins = torch.tensor([[3, 4, 1]] * 3 + [[5, 6, 1]] * 3 + [[7, 8, 1]] * 3)
    molecules = torch.tensor([[i + 3, i + 4, 1] for i in range(9)])
    labels = torch.tensor([8., 6.5, 4.] * 3)
    owners = torch.arange(3).repeat_interleave(3)
    return dict(protein_input_ids=proteins, protein_attention_mask=(proteins != 1).long(),
        molecule_input_ids=molecules, molecule_attention_mask=(molecules != 1).long(),
        pchembl_values=labels, activity_labels=(labels > 6).float(),
        ranking_group_ids=owners, contrastive_group_ids=owners,
        contrastive_target_ids=owners, contrastive_molecule_ids=torch.arange(9))


def test_lora_preserves_initial_function_and_common_head_rng_then_updates_only_adapters():
    reference = model().eval()
    adapted = model(encoder_lora_rank=2, encoder_lora_alpha=4.).eval()
    for name, value in reference.state_dict().items():
        if not name.startswith(('protein_encoder.', 'molecule_encoder.')):
            assert torch.equal(value, adapted.state_dict()[name]), name
    expected, actual = reference(**batch()), adapted(**batch())
    torch.testing.assert_close(actual.ranking_score, expected.ranking_score, rtol=0, atol=0)
    torch.testing.assert_close(actual.cosine_similarity, expected.cosine_similarity, rtol=0, atol=0)
    original = {name: value.detach().clone() for name, value in adapted.named_parameters() if '.base.' in name}
    optimizer = torch.optim.AdamW([p for p in adapted.parameters() if p.requires_grad], lr=.01)
    actual.loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in adapted.named_parameters() if 'lora_B' in n)
    optimizer.step()
    for name, value in adapted.named_parameters():
        if '.base.' in name:
            assert not value.requires_grad and value.grad is None
            assert torch.equal(value, original[name])
    assert sum(isinstance(m, LowRankLinear) for m in adapted.modules()) == 12


def test_partial_unfreezing_preserves_lower_blocks_and_their_eval_mode():
    adapted = model(protein_unfrozen_last_layers=1, molecule_unfrozen_last_layers=1).train()
    for side in ('protein', 'molecule'):
        encoder = getattr(adapted, side + '_encoder')
        assert not encoder.embeddings.training
        assert all(not p.requires_grad for p in encoder.embeddings.parameters())
        assert not encoder.encoder.layer[0].training
        assert all(not p.requires_grad for p in encoder.encoder.layer[0].parameters())
        assert encoder.encoder.layer[-1].training
        assert all(p.requires_grad for p in encoder.encoder.layer[-1].parameters())
    adapted(**batch()).loss.backward()
    assert any(p.grad is not None for p in adapted.protein_encoder.encoder.layer[-1].parameters())
    assert all(p.grad is None for p in adapted.protein_encoder.encoder.layer[0].parameters())


def test_contrastive_detach_blocks_only_encoder_gradient_and_fixed_scale_stays_fixed():
    adapted = model(contrastive_detach_encoders=True, learn_contrastive_scale=False).eval()
    reference = model().eval()
    output = adapted(**batch())
    torch.testing.assert_close(output.contrastive_loss, reference(**batch()).contrastive_loss)
    output.contrastive_loss.backward()
    assert all(p.grad is None for p in adapted.protein_encoder.parameters())
    assert all(p.grad is None for p in adapted.molecule_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapted.contrastive_molecule_projection.parameters())
    assert not adapted.logit_scale.requires_grad and adapted.logit_scale.grad is None
    adapted.zero_grad(set_to_none=True)
    adapted(**batch()).ranking_loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapted.protein_encoder.parameters())


@pytest.mark.parametrize('controls', [
    dict(encoder_lora_rank=2, encoder_lora_alpha=4., learn_contrastive_scale=False,
         contrastive_detach_encoders=True),
    dict(ranking_loss_weight=0, contrastive_loss_weight=0, ranking_readout='activity_logit'),
    dict(ranking_loss_weight=0, classification_loss_weight=0, ranking_readout='cosine'),
])
def test_adapted_checkpoint_round_trip_preserves_outputs_and_training_controls(tmp_path, controls):
    adapted = model(**controls).eval()
    with torch.no_grad():
        for name, parameter in adapted.named_parameters():
            if 'lora_B' in name:
                parameter.fill_(.03)
    expected = adapted(**batch()).ranking_score
    save_reward_model(adapted, str(tmp_path))
    protein, molecule = bundles()
    restored = load_reward_model(str(tmp_path), protein_bundle=protein, molecule_bundle=molecule).eval()
    assert restored.config == adapted.config
    assert restored.logit_scale.requires_grad == adapted.logit_scale.requires_grad
    assert restored.encoder_adaptation_report == adapted.encoder_adaptation_report
    torch.testing.assert_close(restored(**batch()).ranking_score, expected, rtol=0, atol=0)


@pytest.mark.parametrize('changes', [
    {'encoder_lora_rank': 2, 'freeze_protein_encoder': True},
    {'encoder_lora_rank': 2, 'molecule_unfrozen_last_layers': 1},
    {'protein_unfrozen_last_layers': 0}, {'encoder_lora_rank': -1},
])
def test_conflicting_or_invalid_adaptation_modes_rejected(changes):
    with pytest.raises(ValueError):
        model(**changes)


@pytest.mark.parametrize('r,b,c,readout', [
    (1, 0, 0, 'ranking_head'), (0, 1, 0, 'activity_logit'), (0, 0, 1, 'cosine'),
    (1, 1, 0, 'ranking_head'), (1, 0, 1, 'ranking_head'),
    (0, 1, 1, 'activity_logit'), (1, 1, 1, 'ranking_head'),
])
def test_loss_subsets_score_trained_pathway_and_freeze_disabled_branches(r, b, c, readout):
    adapted = model(ranking_loss_weight=r, classification_loss_weight=b,
                    contrastive_loss_weight=c, ranking_readout=readout).eval()
    output = adapted(**batch())
    if readout == 'activity_logit':
        assert output.ranking_score is output.activity_logits
    elif readout == 'cosine':
        assert output.ranking_score is output.cosine_similarity
    output.loss.backward()
    for module, enabled in ((adapted.ranking_head, r), (adapted.classification_head, b),
                            (adapted.contrastive_protein_projection, c),
                            (adapted.contrastive_molecule_projection, c),
                            (adapted.fusion, r or b)):
        if enabled:
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
        else:
            assert all(not p.requires_grad and p.grad is None for p in module.parameters())
    assert adapted.logit_scale.requires_grad == bool(c)


@pytest.mark.parametrize('changes', [
    {'ranking_readout': 'cosine'},
    {'ranking_readout': 'activity_logit', 'ranking_loss_weight': 0, 'classification_loss_weight': 0},
    {'ranking_readout': 'cosine', 'ranking_loss_weight': 0, 'contrastive_loss_weight': 0},
])
def test_readout_cannot_use_an_untrained_objective_or_replace_an_active_ranking_objective(changes):
    with pytest.raises(ValueError):
        model(**changes)
