import pytest
import torch
import torch.nn.functional as F

from conftest import DummyEncoder, DummyTokenizer
from reward_model.model import (
    LoadedEncoder,
    RewardModel,
    RewardModelConfig,
    RewardModelOutput,
    ligunity_bidirectional_contrastive_loss,
)
from reward_model.model.fusion import masked_pool


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
    pooling_type = config_overrides.pop("pooling_type", "mean")
    config = RewardModelConfig(
        protein_model_name_or_path="protein/dummy",
        molecule_model_name_or_path="molecule/dummy",
        fusion_hidden_dim=10,
        fusion_num_heads=2,
        dropout=dropout,
        pooling_type=pooling_type,
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


def test_simple_cosine_is_exact_cls_pool_projection_normalize_ranking_path():
    torch.manual_seed(7)
    model = _build_model(
        pair_scoring_mode="cosine",
        pooling_type="cls",
        projection_type="nonlinear",
        classification_loss_weight=0.0,
        dropout=0.1,
    )
    model.eval()
    protein_input_ids = torch.tensor(
        [[1, 2, 0], [3, 4, 5], [1, 6, 7]], dtype=torch.long
    )
    protein_mask = torch.tensor(
        [[1, 1, 0], [1, 1, 1], [1, 1, 1]], dtype=torch.long
    )
    molecule_input_ids = torch.tensor(
        [[7, 8, 0], [1, 2, 3], [4, 5, 6]], dtype=torch.long
    )
    molecule_mask = torch.tensor(
        [[1, 1, 0], [1, 1, 1], [1, 1, 1]], dtype=torch.long
    )

    projection_input_shapes = []
    hooks = [
        model.protein_projection.register_forward_pre_hook(
            lambda _module, args: projection_input_shapes.append(tuple(args[0].shape))
        ),
        model.molecule_projection.register_forward_pre_hook(
            lambda _module, args: projection_input_shapes.append(tuple(args[0].shape))
        ),
    ]
    try:
        outputs = model(
            protein_input_ids=protein_input_ids,
            protein_attention_mask=protein_mask,
            molecule_input_ids=molecule_input_ids,
            molecule_attention_mask=molecule_mask,
            activity_labels=torch.tensor([1.0, 0.0, 1.0]),
            pchembl_values=torch.tensor([7.0, 6.0, 5.0]),
            ranking_group_ids=torch.tensor([0, 0, 0]),
            return_token_embeddings=True,
        )
    finally:
        for hook in hooks:
            hook.remove()

    with torch.no_grad():
        protein_tokens = model.encode_protein(protein_input_ids, protein_mask)
        molecule_tokens = model.encode_molecule(molecule_input_ids, molecule_mask)
        projected_protein = model.protein_projection(
            masked_pool(protein_tokens, protein_mask.bool(), "cls")
        )
        projected_molecule = model.molecule_projection(
            masked_pool(molecule_tokens, molecule_mask.bool(), "cls")
        )
        expected_protein = F.normalize(
            projected_protein,
            p=2,
            dim=-1,
            eps=1e-6,
        )
        expected_molecule = F.normalize(
            projected_molecule,
            p=2,
            dim=-1,
            eps=1e-6,
        )
        expected_score = (expected_protein * expected_molecule).sum(dim=-1)

    assert model.protein_norm is None
    assert model.molecule_norm is None
    assert model.projection_dropout.p == pytest.approx(0.1)
    assert model.fusion is None
    assert model.ranking_head is None
    assert model.classification_head is None
    assert model.logit_scale is None
    assert model.classification_logit_bias is None
    assert projection_input_shapes == [(3, 6), (3, 8)]
    assert outputs.fused_protein_tokens is None
    assert outputs.fused_molecule_tokens is None
    assert outputs.score_scale is None
    assert outputs.classification_logit_bias is None
    assert torch.allclose(outputs.protein_token_embeddings, protein_tokens)
    assert torch.allclose(outputs.molecule_token_embeddings, molecule_tokens)
    assert torch.allclose(outputs.normalized_protein_embedding, expected_protein)
    assert torch.allclose(outputs.normalized_molecule_embedding, expected_molecule)
    assert torch.allclose(outputs.ranking_score, expected_score)
    assert torch.allclose(outputs.cosine_similarity, expected_score)
    assert torch.all(outputs.ranking_score >= -1.0)
    assert torch.all(outputs.ranking_score <= 1.0)
    assert outputs.classification_loss is None
    assert outputs.ranking_loss is not None
    assert torch.allclose(outputs.loss, outputs.ranking_loss)

    outputs.loss.backward()
    for projection in (model.protein_projection, model.molecule_projection):
        assert projection.linear1.weight.grad is not None
        assert projection.linear1.weight.grad.abs().sum() > 0.0
        assert projection.linear2.weight.grad is not None
        assert projection.linear2.weight.grad.abs().sum() > 0.0


def test_simple_cosine_supports_ligunity_style_nonlinear_projection():
    model = _build_model(
        pair_scoring_mode="cosine",
        pooling_type="cls",
        projection_type="nonlinear",
        classification_loss_weight=0.0,
    )
    outputs = model(
        protein_input_ids=torch.tensor([[1, 2, 3], [4, 5, 6]]),
        protein_attention_mask=torch.ones(2, 3, dtype=torch.long),
        molecule_input_ids=torch.tensor([[7, 8, 9], [1, 2, 3]]),
        molecule_attention_mask=torch.ones(2, 3, dtype=torch.long),
        pchembl_values=torch.tensor([7.0, 5.0]),
        ranking_group_ids=torch.tensor([0, 0]),
    )

    assert model.protein_projection.linear1.in_features == 6
    assert model.protein_projection.linear1.out_features == 6
    assert model.protein_projection.linear2.out_features == 10
    assert model.molecule_projection.linear1.in_features == 8
    assert model.molecule_projection.linear1.out_features == 8
    assert model.molecule_projection.linear2.out_features == 10
    assert outputs.normalized_protein_embedding.shape == (2, 10)
    assert outputs.normalized_molecule_embedding.shape == (2, 10)

    outputs.ranking_score.sum().backward()
    for projection in (model.protein_projection, model.molecule_projection):
        assert projection.linear1.weight.grad is not None
        assert projection.linear1.weight.grad.abs().sum() > 0.0
        assert projection.linear2.weight.grad is not None
        assert projection.linear2.weight.grad.abs().sum() > 0.0


def test_simple_cosine_projection_dropout_is_train_only():
    model = _build_model(
        pair_scoring_mode="cosine",
        classification_loss_weight=0.0,
        dropout=0.5,
    )
    inputs = {
        "protein_input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "protein_attention_mask": torch.ones(2, 3, dtype=torch.long),
        "molecule_input_ids": torch.tensor([[7, 8, 9], [1, 2, 3]]),
        "molecule_attention_mask": torch.ones(2, 3, dtype=torch.long),
        "return_token_embeddings": True,
    }

    model.train()
    torch.manual_seed(11)
    first_train = model(**inputs)
    torch.manual_seed(12)
    second_train = model(**inputs)
    assert not torch.equal(
        first_train.normalized_protein_embedding,
        second_train.normalized_protein_embedding,
    )
    assert not torch.equal(
        first_train.normalized_molecule_embedding,
        second_train.normalized_molecule_embedding,
    )

    model.eval()
    first_eval = model(**inputs)
    second_eval = model(**inputs)
    assert torch.equal(
        first_eval.normalized_protein_embedding,
        second_eval.normalized_protein_embedding,
    )
    assert torch.equal(
        first_eval.normalized_molecule_embedding,
        second_eval.normalized_molecule_embedding,
    )


def test_simple_cosine_shares_one_scaled_matrix_between_ranking_and_contrastive():
    torch.manual_seed(17)
    model = _build_model(
        pair_scoring_mode="cosine",
        classification_loss_weight=0.0,
        ranking_loss_weight=0.5,
        contrastive_loss_weight=0.5,
        ranking_temperature=0.1,
    )
    protein_input_ids = torch.tensor(
        [
            [1, 2, 0],
            [1, 2, 0],
            [1, 2, 0],
            [3, 4, 0],
            [3, 4, 0],
            [3, 4, 0],
        ],
        dtype=torch.long,
    )
    molecule_input_ids = torch.tensor(
        [[5, 1, 0], [6, 1, 0], [7, 1, 0], [8, 1, 0], [9, 1, 0], [2, 1, 0]],
        dtype=torch.long,
    )
    pchembl = torch.tensor([8.0, 7.0, 4.5, 8.5, 6.5, 5.5])
    group_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    target_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    molecule_ids = torch.arange(6)

    outputs = model(
        protein_input_ids=protein_input_ids,
        protein_attention_mask=protein_input_ids.ne(0).long(),
        molecule_input_ids=molecule_input_ids,
        molecule_attention_mask=molecule_input_ids.ne(0).long(),
        pchembl_values=pchembl,
        ranking_group_ids=group_ids,
        contrastive_group_ids=group_ids,
        contrastive_target_ids=target_ids,
        contrastive_molecule_ids=molecule_ids,
    )

    shared_score_matrix = torch.matmul(
        outputs.normalized_protein_embedding[[0, 3]].float(),
        outputs.normalized_molecule_embedding.float().transpose(0, 1),
    ) / model.config.ranking_temperature
    expected_contrastive = ligunity_bidirectional_contrastive_loss(
        shared_score_matrix,
        pchembl,
        group_ids,
        torch.tensor([0, 1]),
        molecule_ids,
    )

    assert torch.allclose(
        shared_score_matrix[group_ids, torch.arange(6)],
        outputs.ranking_score / model.config.ranking_temperature,
    )
    assert outputs.classification_loss is None
    assert torch.allclose(outputs.contrastive_loss, expected_contrastive[0])
    assert torch.allclose(
        outputs.contrastive_protein_to_molecule_loss,
        expected_contrastive[1],
    )
    assert torch.allclose(
        outputs.contrastive_molecule_to_protein_loss,
        expected_contrastive[2],
    )
    assert torch.allclose(
        outputs.loss,
        0.5 * outputs.ranking_loss + 0.5 * outputs.contrastive_loss,
    )

    outputs.loss.backward()
    for parameter in (
        model.protein_encoder.proj.weight,
        model.molecule_encoder.proj.weight,
        model.protein_projection.weight,
        model.molecule_projection.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0.0


def test_contrastive_only_objective_excludes_ranking_loss_from_total():
    model = _build_model(
        pair_scoring_mode="cosine",
        classification_loss_weight=0.0,
        ranking_loss_weight=0.0,
        contrastive_loss_weight=1.0,
        ranking_temperature=0.1,
    )
    protein_input_ids = torch.tensor(
        [[1, 2, 0], [1, 2, 0], [1, 2, 0], [3, 4, 0], [3, 4, 0], [3, 4, 0]],
        dtype=torch.long,
    )
    molecule_input_ids = torch.tensor(
        [[5, 1, 0], [6, 1, 0], [7, 1, 0], [8, 1, 0], [9, 1, 0], [2, 1, 0]],
        dtype=torch.long,
    )

    outputs = model(
        protein_input_ids=protein_input_ids,
        protein_attention_mask=protein_input_ids.ne(0).long(),
        molecule_input_ids=molecule_input_ids,
        molecule_attention_mask=molecule_input_ids.ne(0).long(),
        pchembl_values=torch.tensor([8.0, 7.0, 4.5, 8.5, 6.5, 5.5]),
        ranking_group_ids=torch.tensor([0, 0, 0, 1, 1, 1]),
        contrastive_group_ids=torch.tensor([0, 0, 0, 1, 1, 1]),
        contrastive_target_ids=torch.tensor([0, 0, 0, 1, 1, 1]),
        contrastive_molecule_ids=torch.arange(6),
    )

    assert outputs.ranking_loss is None
    assert outputs.contrastive_loss is not None
    assert torch.equal(outputs.loss, outputs.contrastive_loss)


def test_cosine_contrastive_and_mlp_classification_share_embeddings():
    model = _build_model(
        pair_scoring_mode="cosine",
        cosine_classification_mlp=True,
        classification_loss_weight=0.5,
        ranking_loss_weight=0.0,
        contrastive_loss_weight=0.5,
        contrastive_active_threshold=6.0,
        contrastive_strict_active_only=True,
        ranking_temperature=1.0 / 13.0,
    )
    protein_input_ids = torch.tensor(
        [[1, 2, 0], [1, 2, 0], [1, 2, 0], [3, 4, 0], [3, 4, 0], [3, 4, 0]],
        dtype=torch.long,
    )
    molecule_input_ids = torch.tensor(
        [[5, 1, 0], [6, 1, 0], [7, 1, 0], [8, 1, 0], [9, 1, 0], [2, 1, 0]],
        dtype=torch.long,
    )
    pchembl = torch.tensor([8.0, 7.0, 4.5, 8.5, 6.0, 5.5])
    group_ids = torch.tensor([0, 0, 0, 1, 1, 1])

    outputs = model(
        protein_input_ids=protein_input_ids,
        protein_attention_mask=protein_input_ids.ne(0).long(),
        molecule_input_ids=molecule_input_ids,
        molecule_attention_mask=molecule_input_ids.ne(0).long(),
        activity_labels=(pchembl >= 6.0).float(),
        pchembl_values=pchembl,
        ranking_group_ids=group_ids,
        contrastive_group_ids=group_ids,
        contrastive_target_ids=group_ids,
        contrastive_molecule_ids=torch.arange(6),
    )

    normalized_joint = torch.cat(
        [
            outputs.normalized_protein_embedding,
            outputs.normalized_molecule_embedding,
        ],
        dim=-1,
    )
    expected_logits = model.classification_head(normalized_joint)
    assert model.classification_head is not None
    assert torch.allclose(outputs.activity_logits, expected_logits)
    assert outputs.ranking_loss is None
    assert outputs.contrastive_loss is not None
    assert outputs.classification_loss is not None
    assert torch.allclose(
        outputs.loss,
        0.5 * outputs.contrastive_loss + 0.5 * outputs.classification_loss,
    )

    outputs.loss.backward()
    assert model.classification_head.fc1.weight.grad is not None
    assert model.classification_head.fc1.weight.grad.abs().sum() > 0.0


def test_cosine_marginal_biases_are_excluded_from_contrastive_retrieval():
    torch.manual_seed(23)
    model = _build_model(
        pair_scoring_mode="cosine",
        cosine_marginal_biases=True,
        cosine_scale_init=13.0,
        protein_pooling_type="mean",
        molecule_pooling_type="cls",
        ranking_loss_weight=0.0,
        contrastive_loss_weight=0.5,
        classification_loss_weight=0.5,
        contrastive_active_threshold=6.0,
        contrastive_strict_active_only=True,
    )
    protein_input_ids = torch.tensor(
        [[1, 2, 0], [1, 2, 0], [3, 4, 5], [3, 4, 5]],
        dtype=torch.long,
    )
    protein_mask = protein_input_ids.ne(0).long()
    molecule_input_ids = torch.tensor(
        [[5, 1, 0], [6, 2, 0], [7, 3, 0], [8, 4, 0]],
        dtype=torch.long,
    )
    molecule_mask = molecule_input_ids.ne(0).long()
    pchembl = torch.tensor([8.0, 5.0, 7.0, 4.0])
    inputs = {
        "protein_input_ids": protein_input_ids,
        "protein_attention_mask": protein_mask,
        "molecule_input_ids": molecule_input_ids,
        "molecule_attention_mask": molecule_mask,
        "activity_labels": (pchembl >= 6.0).float(),
        "pchembl_values": pchembl,
        "ranking_group_ids": torch.tensor([0, 0, 1, 1]),
        "contrastive_group_ids": torch.tensor([0, 0, 1, 1]),
        "contrastive_target_ids": torch.tensor([0, 0, 1, 1]),
        "contrastive_molecule_ids": torch.arange(4),
    }

    projection_inputs = {}
    hooks = [
        model.protein_projection.register_forward_pre_hook(
            lambda _module, args: projection_inputs.update(
                protein=args[0].detach().clone()
            )
        ),
        model.molecule_projection.register_forward_pre_hook(
            lambda _module, args: projection_inputs.update(
                molecule=args[0].detach().clone()
            )
        ),
    ]
    try:
        baseline = model(**inputs)
    finally:
        for hook in hooks:
            hook.remove()

    with torch.no_grad():
        protein_tokens = model.encode_protein(protein_input_ids, protein_mask)
        molecule_tokens = model.encode_molecule(molecule_input_ids, molecule_mask)
        assert torch.allclose(
            projection_inputs["protein"],
            masked_pool(protein_tokens, protein_mask.bool(), "mean"),
        )
        assert torch.allclose(
            projection_inputs["molecule"],
            masked_pool(molecule_tokens, molecule_mask.bool(), "cls"),
        )
        model.molecule_bias_head.weight.fill_(0.25)
        model.molecule_bias_head.bias.fill_(0.5)
        model.protein_bias_head.weight.fill_(-0.2)
        model.protein_bias_head.bias.fill_(-0.3)
    biased = model(**inputs)

    assert baseline.score_scale.item() == pytest.approx(13.0)
    with torch.no_grad():
        pooled_protein = model.protein_projection(projection_inputs["protein"])
        pooled_molecule = model.molecule_projection(projection_inputs["molecule"])
        expected_ranking = (
            biased.score_scale * biased.cosine_similarity
            + model.molecule_bias_head(
                F.normalize(pooled_molecule, p=2, dim=-1, eps=1e-6)
            ).squeeze(-1)
        )
        expected_classification = (
            expected_ranking
            + model.protein_bias_head(
                F.normalize(pooled_protein, p=2, dim=-1, eps=1e-6)
            ).squeeze(-1)
        )
    assert torch.allclose(biased.ranking_score, expected_ranking)
    assert torch.allclose(biased.activity_logits, expected_classification)
    assert not torch.allclose(baseline.ranking_score, biased.ranking_score)
    assert not torch.allclose(baseline.activity_logits, biased.activity_logits)
    assert torch.allclose(baseline.contrastive_loss, biased.contrastive_loss)
    bias_gradients = torch.autograd.grad(
        biased.contrastive_loss,
        tuple(model.molecule_bias_head.parameters())
        + tuple(model.protein_bias_head.parameters()),
        allow_unused=True,
        retain_graph=True,
    )
    assert all(gradient is None for gradient in bias_gradients)

    biased.loss.backward()
    assert model.molecule_bias_head.weight.grad is not None
    assert model.molecule_bias_head.weight.grad.abs().sum() > 0.0
    assert model.protein_bias_head.weight.grad is not None
    assert model.protein_bias_head.weight.grad.abs().sum() > 0.0


def test_contrastive_objective_requires_identity_metadata():
    model = _build_model(
        pair_scoring_mode="cosine",
        classification_loss_weight=0.0,
        contrastive_loss_weight=0.5,
    )
    with pytest.raises(ValueError, match="target and molecule identity ids"):
        model(
            protein_input_ids=torch.tensor([[1, 2], [1, 2], [1, 2]]),
            protein_attention_mask=torch.ones((3, 2), dtype=torch.long),
            molecule_input_ids=torch.tensor([[3, 4], [4, 5], [5, 6]]),
            molecule_attention_mask=torch.ones((3, 2), dtype=torch.long),
            pchembl_values=torch.tensor([7.0, 6.0, 5.0]),
            ranking_group_ids=torch.tensor([0, 0, 0]),
            contrastive_group_ids=torch.tensor([0, 0, 0]),
        )


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
