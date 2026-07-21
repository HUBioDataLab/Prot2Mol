import copy

import torch

from conftest import DummyEncoder, DummyTokenizer
from reward_model.model import LoadedEncoder, RewardModel, RewardModelConfig


class CountingDummyEncoder(DummyEncoder):
    def __init__(self, hidden_size: int):
        super().__init__(hidden_size=hidden_size)
        self.batch_sizes = []

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        self.batch_sizes.append(int(input_ids.size(0)))
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )


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


def _duplicate_batch():
    return {
        "protein_input_ids": torch.tensor(
            [[1, 2, 0], [1, 2, 0], [3, 4, 0], [1, 2, 0]],
            dtype=torch.long,
        ),
        "protein_attention_mask": torch.tensor(
            [[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]],
            dtype=torch.long,
        ),
        "molecule_input_ids": torch.tensor(
            [[5, 6, 0], [7, 8, 0], [5, 6, 0], [9, 1, 0]],
            dtype=torch.long,
        ),
        "molecule_attention_mask": torch.tensor(
            [[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]],
            dtype=torch.long,
        ),
        "activity_labels": torch.tensor([1.0, 1.0, 0.0, 0.0]),
        "positive_indices": torch.tensor([0, 1], dtype=torch.long),
        "negative_indices": torch.tensor([2, 3], dtype=torch.long),
    }


def _build_counting_model(*, deduplicate: bool):
    protein_encoder = CountingDummyEncoder(hidden_size=6)
    molecule_encoder = CountingDummyEncoder(hidden_size=8)
    model = RewardModel(
        config=RewardModelConfig(
            protein_model_name_or_path="protein/dummy",
            molecule_model_name_or_path="molecule/dummy",
            fusion_hidden_dim=10,
            fusion_num_heads=2,
            dropout=0.0,
            deduplicate_protein_inputs=deduplicate,
            deduplicate_molecule_inputs=deduplicate,
        ),
        protein_bundle=LoadedEncoder(
            name_or_path="protein/dummy",
            tokenizer=DummyTokenizer(),
            model=protein_encoder,
            hidden_size=6,
        ),
        molecule_bundle=LoadedEncoder(
            name_or_path="molecule/dummy",
            tokenizer=DummyTokenizer(),
            model=molecule_encoder,
            hidden_size=8,
        ),
    )
    return model, protein_encoder, molecule_encoder


def test_reward_model_deduplicates_protein_and_molecule_encoder_inputs():
    model, protein_encoder, molecule_encoder = _build_counting_model(deduplicate=True)

    outputs = model(**_duplicate_batch())

    assert outputs.loss is not None
    assert protein_encoder.batch_sizes == [2]
    assert molecule_encoder.batch_sizes == [3]


def test_reward_model_can_disable_deduplication_for_legacy_execution():
    model, protein_encoder, molecule_encoder = _build_counting_model(deduplicate=False)

    model(**_duplicate_batch())

    assert protein_encoder.batch_sizes == [4]
    assert molecule_encoder.batch_sizes == [4]


def test_deduplicated_forward_and_backward_match_legacy_execution():
    torch.manual_seed(7)
    optimized_model, _, _ = _build_counting_model(deduplicate=True)
    legacy_model = copy.deepcopy(optimized_model)
    legacy_model.config.deduplicate_protein_inputs = False
    legacy_model.config.deduplicate_molecule_inputs = False
    batch = _duplicate_batch()

    optimized_outputs = optimized_model(**batch)
    legacy_outputs = legacy_model(**batch)

    assert torch.allclose(optimized_outputs.ranking_score, legacy_outputs.ranking_score)
    assert torch.allclose(optimized_outputs.activity_logits, legacy_outputs.activity_logits)
    assert torch.allclose(optimized_outputs.loss, legacy_outputs.loss)

    optimized_outputs.loss.backward()
    legacy_outputs.loss.backward()
    legacy_parameters = dict(legacy_model.named_parameters())
    for name, optimized_parameter in optimized_model.named_parameters():
        legacy_parameter = legacy_parameters[name]
        assert optimized_parameter.grad is not None, name
        assert legacy_parameter.grad is not None, name
        assert torch.allclose(
            optimized_parameter.grad,
            legacy_parameter.grad,
            atol=1e-6,
            rtol=1e-5,
        ), name


def test_dynamic_trailing_padding_removal_preserves_forward_and_backward():
    torch.manual_seed(13)
    fixed_width_model = _build_model()
    dynamic_width_model = copy.deepcopy(fixed_width_model)
    fixed_width_batch = _duplicate_batch()
    dynamic_width_batch = dict(fixed_width_batch)
    for key in (
        "protein_input_ids",
        "protein_attention_mask",
        "molecule_input_ids",
        "molecule_attention_mask",
    ):
        dynamic_width_batch[key] = dynamic_width_batch[key][:, :2]

    fixed_outputs = fixed_width_model(**fixed_width_batch)
    dynamic_outputs = dynamic_width_model(**dynamic_width_batch)

    assert torch.allclose(fixed_outputs.ranking_score, dynamic_outputs.ranking_score)
    assert torch.allclose(fixed_outputs.activity_logits, dynamic_outputs.activity_logits)
    assert torch.allclose(fixed_outputs.loss, dynamic_outputs.loss)

    fixed_outputs.loss.backward()
    dynamic_outputs.loss.backward()
    dynamic_parameters = dict(dynamic_width_model.named_parameters())
    for name, fixed_parameter in fixed_width_model.named_parameters():
        dynamic_parameter = dynamic_parameters[name]
        assert fixed_parameter.grad is not None, name
        assert dynamic_parameter.grad is not None, name
        assert torch.allclose(
            fixed_parameter.grad,
            dynamic_parameter.grad,
            atol=1e-6,
            rtol=1e-5,
        ), name


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
