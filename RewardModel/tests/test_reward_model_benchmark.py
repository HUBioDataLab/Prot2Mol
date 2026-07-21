from types import SimpleNamespace

import torch

from conftest import DummyTokenizer

from benchmark_reward_model import (
    VARIANTS,
    _build_synthetic_pair_dataset,
    _gradient_norms_close,
    _max_gradient_signature_errors,
    _metrics_close,
    _percentile,
)


def test_benchmark_variants_preserve_expected_before_after_switches():
    baseline = VARIANTS["baseline"]
    optimized = VARIANTS["optimized"]
    optimized_sdpa = VARIANTS["optimized_sdpa"]
    optimized_sdpa_fused = VARIANTS["optimized_sdpa_fused"]

    assert baseline.dynamic_padding is False
    assert baseline.length_bucketing is False
    assert baseline.deduplicate_inputs is False
    assert optimized.dynamic_padding is True
    assert optimized.length_bucketing is True
    assert optimized.deduplicate_inputs is True
    assert optimized.fusion_attention_backend == "manual"
    assert optimized_sdpa.fusion_attention_backend == "sdpa"
    assert optimized_sdpa.fused_optimizer is False
    assert optimized_sdpa_fused.fused_optimizer is True


def test_benchmark_percentile_uses_nearest_rank():
    assert _percentile([4.0, 1.0, 3.0, 2.0], 0.5) == 2.0
    assert _percentile([4.0, 1.0, 3.0, 2.0], 0.95) == 4.0


def test_benchmark_gradient_signature_error_is_zero_for_equal_signatures():
    signature = {
        "encoder.weight": (3.0, -1.0),
        "head.weight": (2.0, 0.5),
    }

    errors = _max_gradient_signature_errors(signature, dict(signature))

    assert errors == {
        "gradient_norm_max_absolute_error": 0.0,
        "gradient_norm_max_relative_error": 0.0,
        "gradient_sum_max_absolute_error": 0.0,
    }


def test_benchmark_gradient_signature_error_reports_relative_norm_change():
    errors = _max_gradient_signature_errors(
        {"weight": (2.0, 1.0)},
        {"weight": (1.0, 1.5)},
    )

    assert torch.isclose(
        torch.tensor(errors["gradient_norm_max_relative_error"]),
        torch.tensor(0.5),
    )
    assert errors["gradient_norm_max_absolute_error"] == 1.0
    assert errors["gradient_sum_max_absolute_error"] == 0.5


def test_benchmark_gradient_closeness_handles_near_zero_norms():
    baseline = {"small": (1e-8, 0.0), "large": (100.0, 0.0)}
    optimized = {"small": (2e-8, 0.0), "large": (100.005, 0.0)}

    assert _gradient_norms_close(baseline, optimized)
    assert not _gradient_norms_close(baseline, {**optimized, "large": (100.02, 0.0)})


def test_synthetic_benchmark_dataset_has_fixed_storage_and_pair_protein_reuse():
    model = SimpleNamespace(
        config=SimpleNamespace(protein_max_length=16, molecule_max_length=8),
        protein_encoder=SimpleNamespace(config=SimpleNamespace(vocab_size=24)),
        molecule_encoder=SimpleNamespace(config=SimpleNamespace(vocab_size=32)),
        protein_tokenizer=DummyTokenizer(pad_token_id=1),
        molecule_tokenizer=DummyTokenizer(pad_token_id=0),
    )

    dataset = _build_synthetic_pair_dataset(model, num_pairs=8, seed=42)

    assert len(dataset) == 8
    assert len(dataset.example_dataset) == 16
    first_pair = dataset[0]
    assert len(first_pair["positive"]["protein_input_ids"]) == 16
    assert len(first_pair["positive"]["molecule_input_ids"]) == 8
    assert (
        first_pair["positive"]["protein_input_ids"]
        == first_pair["negative"]["protein_input_ids"]
    )
    assert first_pair["positive"]["molecule_input_ids"] != first_pair["negative"][
        "molecule_input_ids"
    ]


def test_validation_metric_comparison_treats_matching_nans_as_equal():
    close, max_error = _metrics_close(
        {"eval_loss": 0.5, "eval_spearman": float("nan")},
        {"eval_loss": 0.5, "eval_spearman": float("nan")},
    )

    assert close
    assert max_error == 0.0
