import torch

from benchmark_reward_model import (
    VARIANTS,
    _gradient_norms_close,
    _max_gradient_signature_errors,
    _percentile,
)


def test_benchmark_variants_preserve_expected_before_after_switches():
    baseline = VARIANTS["baseline"]
    optimized = VARIANTS["optimized"]

    assert baseline.dynamic_padding is False
    assert baseline.length_bucketing is False
    assert baseline.deduplicate_inputs is False
    assert optimized.dynamic_padding is True
    assert optimized.length_bucketing is True
    assert optimized.deduplicate_inputs is True


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
