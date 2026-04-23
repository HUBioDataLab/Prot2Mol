import copy

import torch

from conftest import DummyEncoder, DummyTokenizer
from reward_model.model import LoadedEncoder, RewardModel, RewardModelConfig


def _build_model():
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
    )
    return RewardModel(config=config, protein_bundle=protein_bundle, molecule_bundle=molecule_bundle)


def _dummy_batch():
    return {
        "protein_input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.long),
        "protein_attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long),
        "molecule_input_ids": torch.tensor([[7, 8, 9, 0], [1, 2, 3, 4]], dtype=torch.long),
        "molecule_attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.long),
        "activity_labels": torch.tensor([1.0, 0.0], dtype=torch.float32),
        "positive_indices": torch.tensor([0], dtype=torch.long),
        "negative_indices": torch.tensor([1], dtype=torch.long),
    }


def test_reward_model_backward_produces_gradients_for_all_trainable_blocks():
    torch.manual_seed(0)
    model = _build_model()
    batch = _dummy_batch()

    outputs = model(**batch)
    assert outputs.loss is not None

    outputs.loss.backward()

    grad_targets = {
        "protein_encoder": model.protein_encoder.proj.weight.grad,
        "molecule_encoder": model.molecule_encoder.proj.weight.grad,
        "protein_projection": model.protein_projection.weight.grad,
        "molecule_projection": model.molecule_projection.weight.grad,
        "fusion_query_p": model.fusion.query_p.weight.grad,
        "fusion_query_m": model.fusion.query_m.weight.grad,
        "ranking_head": model.ranking_head.fc1.weight.grad,
        "classification_head": model.classification_head.fc1.weight.grad,
    }

    for name, grad in grad_targets.items():
        assert grad is not None, f"Expected gradient for {name}"
        assert torch.isfinite(grad).all(), f"Expected finite gradient values for {name}"
        assert grad.abs().sum().item() > 0.0, f"Expected non-zero gradient magnitude for {name}"


def test_reward_model_optimizer_step_updates_parameters():
    torch.manual_seed(0)
    model = _build_model()
    batch = _dummy_batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    tracked_before = {
        "protein_projection": copy.deepcopy(model.protein_projection.weight.detach()),
        "molecule_projection": copy.deepcopy(model.molecule_projection.weight.detach()),
        "fusion_query_p": copy.deepcopy(model.fusion.query_p.weight.detach()),
        "ranking_head": copy.deepcopy(model.ranking_head.fc2.weight.detach()),
        "classification_head": copy.deepcopy(model.classification_head.fc2.weight.detach()),
    }

    outputs = model(**batch)
    assert outputs.loss is not None
    outputs.loss.backward()
    optimizer.step()

    tracked_after = {
        "protein_projection": model.protein_projection.weight.detach(),
        "molecule_projection": model.molecule_projection.weight.detach(),
        "fusion_query_p": model.fusion.query_p.weight.detach(),
        "ranking_head": model.ranking_head.fc2.weight.detach(),
        "classification_head": model.classification_head.fc2.weight.detach(),
    }

    changed = [
        name
        for name, before in tracked_before.items()
        if not torch.equal(before, tracked_after[name])
    ]

    assert changed, "Expected at least one tracked parameter tensor to change after optimizer.step()"
