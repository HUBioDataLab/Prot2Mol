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
    config = RewardModelConfig(
        protein_model_name_or_path="protein/dummy",
        molecule_model_name_or_path="molecule/dummy",
        fusion_hidden_dim=10,
        fusion_num_heads=2,
        dropout=0.0,
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
