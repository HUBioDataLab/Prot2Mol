import pytest
import torch

from conftest import DummyEncoder, DummyTokenizer
from reward_model.model import LoadedEncoder, RewardModel, RewardModelConfig, RewardModelOutput


def _build_model(**config_overrides):
    protein_bundle = LoadedEncoder(
        name_or_path="protein/dummy",
        tokenizer=DummyTokenizer(),
        model=DummyEncoder(hidden_size=6),
        hidden_size=6,
    )
    molecule_bundle = LoadedEncoder(
        name_or_path="molecule/dummy",
        tokenizer=DummyTokenizer(),
        model=DummyEncoder(hidden_size=8),
        hidden_size=8,
    )
    dropout = config_overrides.pop("dropout", 0.0)
    config = RewardModelConfig(
        protein_model_name_or_path="protein/dummy",
        molecule_model_name_or_path="molecule/dummy",
        fusion_hidden_dim=10,
        fusion_num_heads=2,
        dropout=dropout,
        pooling_type="mean",
        **config_overrides,
    )
    return RewardModel(config=config, protein_bundle=protein_bundle, molecule_bundle=molecule_bundle)


def test_reward_model_forward_handles_hidden_dim_mismatch_and_losses():
    model = _build_model()

    outputs = model(
        protein_input_ids=torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.long),
        protein_attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long),
        molecule_input_ids=torch.tensor([[7, 8, 9, 0], [1, 2, 3, 4]], dtype=torch.long),
        molecule_attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.long),
        activity_labels=torch.tensor([1.0, 0.0], dtype=torch.float32),
        positive_indices=torch.tensor([0], dtype=torch.long),
        negative_indices=torch.tensor([1], dtype=torch.long),
        return_token_embeddings=True,
    )

    assert isinstance(outputs, RewardModelOutput)
    assert outputs["ranking_score"].shape == (2,)
    assert outputs["activity_logits"].shape == (2,)
    assert outputs["activity_probability"].shape == (2,)
    assert outputs["joint_embedding"].shape == (2, 20)
    assert outputs["fused_protein_tokens"].shape == (2, 3, 10)
    assert outputs["fused_molecule_tokens"].shape == (2, 4, 10)
    assert "pair_loss" in outputs
    assert "classification_loss" in outputs
    assert "loss" in outputs
    assert model.config.protein_hidden_size == 6
    assert model.config.molecule_hidden_size == 8
    expected_head_dims = [2048, 1024, 512, 256, 128]
    assert [
        model.ranking_head.fc1.out_features,
        *(layer.out_features for layer in model.ranking_head.hidden_layers),
    ] == expected_head_dims
    assert [
        model.classification_head.fc1.out_features,
        *(layer.out_features for layer in model.classification_head.hidden_layers),
    ] == expected_head_dims


def test_reward_model_passes_fusion_residual_config_to_fusion():
    baseline = _build_model(fusion_residual=False)
    residual = _build_model(fusion_residual=True)

    assert baseline.fusion.residual is False
    assert baseline.fusion.protein_residual_norm is None
    assert baseline.fusion.molecule_residual_norm is None
    assert residual.fusion.residual is True
    assert residual.fusion.protein_residual_norm is not None
    assert residual.fusion.molecule_residual_norm is not None


def test_reward_model_applies_configured_dropout_outside_encoders():
    model = _build_model(dropout=0.1, pair_scoring_mode="scaled_cosine")

    assert model.projection_dropout.p == pytest.approx(0.1)
    assert model.fusion.dropout == pytest.approx(0.1)


def test_scaled_cosine_mode_shares_one_geometry_and_scale_between_objectives():
    model = _build_model(
        pair_scoring_mode="scaled_cosine",
        cosine_scale_init=13.0,
        cosine_classification_bias_init=-0.75,
        fusion_residual=True,
    )
    model.eval()

    outputs = model(
        protein_input_ids=torch.tensor(
            [[1, 2, 0], [3, 4, 5], [1, 6, 7]], dtype=torch.long
        ),
        protein_attention_mask=torch.tensor(
            [[1, 1, 0], [1, 1, 1], [1, 1, 1]], dtype=torch.long
        ),
        molecule_input_ids=torch.tensor(
            [[7, 8, 9], [1, 2, 3], [4, 5, 6]], dtype=torch.long
        ),
        molecule_attention_mask=torch.ones((3, 3), dtype=torch.long),
        activity_labels=torch.tensor([1.0, 0.0, 1.0]),
        pchembl_values=torch.tensor([7.0, 6.0, 5.0]),
        ranking_group_ids=torch.tensor([0, 0, 0]),
    )

    assert model.ranking_head is None
    assert model.classification_head is None
    assert outputs.cosine_similarity.shape == (3,)
    assert torch.all(outputs.cosine_similarity >= -1.0)
    assert torch.all(outputs.cosine_similarity <= 1.0)
    assert torch.allclose(
        outputs.normalized_protein_embedding.norm(dim=-1), torch.ones(3)
    )
    assert torch.allclose(
        outputs.normalized_molecule_embedding.norm(dim=-1), torch.ones(3)
    )
    assert outputs.score_scale.item() == pytest.approx(13.0)
    assert torch.allclose(
        outputs.ranking_score,
        outputs.score_scale * outputs.cosine_similarity,
    )
    assert torch.allclose(
        outputs.activity_logits,
        outputs.ranking_score - 0.75,
    )


def test_zero_classification_weight_removes_classification_loss_and_gradient():
    model = _build_model(
        pair_scoring_mode="scaled_cosine",
        classification_loss_weight=0.0,
        fusion_residual=True,
    )
    outputs = model(
        protein_input_ids=torch.tensor(
            [[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype=torch.long
        ),
        protein_attention_mask=torch.ones((3, 3), dtype=torch.long),
        molecule_input_ids=torch.tensor(
            [[7, 8, 9], [1, 2, 3], [4, 5, 6]], dtype=torch.long
        ),
        molecule_attention_mask=torch.ones((3, 3), dtype=torch.long),
        activity_labels=torch.tensor([1.0, 0.0, 1.0]),
        pchembl_values=torch.tensor([7.0, 6.0, 5.0]),
        ranking_group_ids=torch.tensor([0, 0, 0]),
    )

    assert outputs.classification_loss is None
    assert outputs.ranking_loss is not None
    assert torch.allclose(outputs.loss, outputs.ranking_loss)
    outputs.loss.backward()
    assert model.classification_logit_bias.grad is None
    assert model.logit_scale.grad is not None
    assert torch.isfinite(model.logit_scale.grad)


@pytest.mark.parametrize("objective", ["ranking", "classification"])
def test_each_scaled_cosine_objective_updates_shared_pair_geometry(objective):
    model = _build_model(
        pair_scoring_mode="scaled_cosine",
        cosine_scale_init=13.0,
        fusion_residual=True,
    )
    common_inputs = {
        "protein_input_ids": torch.tensor(
            [[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype=torch.long
        ),
        "protein_attention_mask": torch.ones((3, 3), dtype=torch.long),
        "molecule_input_ids": torch.tensor(
            [[7, 8, 9], [1, 2, 3], [4, 5, 6]], dtype=torch.long
        ),
        "molecule_attention_mask": torch.ones((3, 3), dtype=torch.long),
    }
    if objective == "ranking":
        common_inputs.update(
            pchembl_values=torch.tensor([7.0, 6.0, 5.0]),
            ranking_group_ids=torch.tensor([0, 0, 0]),
        )
    else:
        common_inputs["activity_labels"] = torch.tensor([1.0, 0.0, 1.0])

    outputs = model(**common_inputs)
    outputs.loss.backward()

    assert model.logit_scale.grad is not None
    assert model.logit_scale.grad.abs().item() > 0.0
    assert model.protein_projection.weight.grad is not None
    assert model.protein_projection.weight.grad.abs().sum() > 0.0
    assert model.molecule_projection.weight.grad is not None
    assert model.molecule_projection.weight.grad.abs().sum() > 0.0
    if objective == "classification":
        assert model.classification_logit_bias.grad is not None
        assert model.classification_logit_bias.grad.abs().item() > 0.0
    else:
        assert model.classification_logit_bias.grad is None


def test_scaled_cosine_scale_is_positive_and_capped():
    model = _build_model(
        pair_scoring_mode="scaled_cosine",
        cosine_scale_init=13.0,
        cosine_scale_max=20.0,
    )
    with torch.no_grad():
        model.logit_scale.fill_(100.0)
    outputs = model(
        protein_input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        protein_attention_mask=torch.ones((1, 2), dtype=torch.long),
        molecule_input_ids=torch.tensor([[3, 4]], dtype=torch.long),
        molecule_attention_mask=torch.ones((1, 2), dtype=torch.long),
    )
    assert outputs.score_scale.item() == pytest.approx(20.0)
    assert torch.isfinite(outputs.ranking_score).all()

    with torch.no_grad():
        model.logit_scale.fill_(-100.0)
    outputs = model(
        protein_input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        protein_attention_mask=torch.ones((1, 2), dtype=torch.long),
        molecule_input_ids=torch.tensor([[3, 4]], dtype=torch.long),
        molecule_attention_mask=torch.ones((1, 2), dtype=torch.long),
    )
    assert outputs.score_scale.item() > 0.0


def test_scaled_cosine_zero_vectors_remain_finite():
    model = _build_model(pair_scoring_mode="scaled_cosine")
    zeros = torch.zeros((2, model.config.fusion_hidden_dim))
    joint = torch.cat([zeros, zeros], dim=-1)

    ranking_score, activity_logits, cosine, scale, protein, molecule = (
        model._score_pooled_pair(zeros, zeros, joint)
    )

    assert torch.equal(cosine, torch.zeros(2))
    assert torch.isfinite(ranking_score).all()
    assert torch.isfinite(activity_logits).all()
    assert torch.isfinite(scale)
    assert torch.isfinite(protein).all()
    assert torch.isfinite(molecule).all()


def test_reward_model_encoders_can_be_frozen_independently():
    protein_frozen = _build_model(
        freeze_protein_encoder=True,
        freeze_molecule_encoder=False,
    )
    protein_frozen.train()
    assert not any(
        parameter.requires_grad
        for parameter in protein_frozen.protein_encoder.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in protein_frozen.molecule_encoder.parameters()
    )
    assert protein_frozen.protein_encoder.training is False
    assert protein_frozen.molecule_encoder.training is True

    molecule_frozen = _build_model(
        freeze_protein_encoder=False,
        freeze_molecule_encoder=True,
    )
    molecule_frozen.train()
    assert all(
        parameter.requires_grad
        for parameter in molecule_frozen.protein_encoder.parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in molecule_frozen.molecule_encoder.parameters()
    )
    assert molecule_frozen.protein_encoder.training is True
    assert molecule_frozen.molecule_encoder.training is False


def test_reward_model_requires_pair_indices_together():
    model = _build_model()

    try:
        model(
            protein_input_ids=torch.tensor([[1, 2]], dtype=torch.long),
            protein_attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
            molecule_input_ids=torch.tensor([[1, 2]], dtype=torch.long),
            molecule_attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
            positive_indices=torch.tensor([0], dtype=torch.long),
        )
        assert False, "Expected ValueError when only one pair index tensor is provided"
    except ValueError as exc:
        assert "provided together" in str(exc)


def test_reward_model_can_return_tuple_output():
    model = _build_model()

    outputs = model(
        protein_input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        protein_attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        molecule_input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        molecule_attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.long),
        return_dict=False,
    )

    assert isinstance(outputs, tuple)
    assert len(outputs) == 4
