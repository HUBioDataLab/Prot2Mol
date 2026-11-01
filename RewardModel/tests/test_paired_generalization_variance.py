import numpy as np
import pandas as pd
import pytest
from scipy.optimize import nnls

from experiments.paired_generalization.variance_audit import covariance_moments, fit_components, fit_diagnostics


def test_categorical_trace_fit_matches_explicit_dense_covariance_regression():
    rng = np.random.default_rng(45)
    y = rng.normal(size=12)
    labels = {"x": np.repeat(np.arange(3), 4), "c": np.tile(np.arange(4), 3), "obs": np.repeat(np.arange(6), 2)}
    names, gram, rhs = covariance_moments(y, labels)
    project = np.eye(len(y)) - np.ones((len(y), len(y))) / len(y)
    matrices = [project @ (codes[:, None] == codes[None, :]) @ project for codes in labels.values()] + [project]
    dense_design = np.stack([a.reshape(-1) for a in matrices], axis=1)
    dense_target = np.outer(project @ y, project @ y).reshape(-1)
    np.testing.assert_allclose(gram, dense_design.T @ dense_design, atol=1e-12)
    np.testing.assert_allclose(rhs, dense_design.T @ dense_target, atol=1e-12)
    expected, _ = nnls(dense_design, dense_target)
    fitted = fit_components(y, labels)
    assert fitted["identifiable"]
    np.testing.assert_allclose([fitted["variance_components"][n] for n in names], expected, atol=1e-10)


def test_nested_duplicate_effects_are_reported_as_nonidentifiable():
    labels = np.repeat(np.arange(4), 3)
    result = fit_components(np.arange(12, dtype=float), {"cluster": labels, "interaction": labels.copy()})
    assert not result["identifiable"]
    assert result["variance_components"] is None
    assert result["design_rank"] == 2
    with pytest.raises(ValueError):
        fit_components([1., np.nan], {"x": [0, 1]})


def test_replicate_contrast_diagnostic_uses_within_observation_degrees_of_freedom():
    frame = pd.DataFrame({"pchembl_value": [0., 2., 8., 10., 12., 100.], "observation_id": [0, 0, 1, 1, 1, 2]})
    fit = fit_components(frame.pchembl_value, {"observation": frame.observation_id})
    diagnosis = fit_diagnostics(frame, fit)
    assert diagnosis["replicate_contrast_degrees_of_freedom"] == 3
    # SSE: 2 for the first observation, 8 for the second, zero for the singleton.
    assert diagnosis["direct_within_observation_noise_variance"] == pytest.approx(10 / 3)
    assert diagnosis["observed_centered_variance"] == pytest.approx(frame.pchembl_value.var(ddof=0))


def test_weighted_covariance_fit_matches_dense_weighted_intercept_projection():
    y = np.array([0., 2., 4., 5., 1., 6., 3., 8.])
    weights = np.array([.2, .2, .2, .2, 1., 1., 2., 2.])
    labels = {"a": np.array([0, 0, 1, 1, 2, 2, 3, 3]), "b": np.tile([0, 1], 4)}
    names, gram, rhs = covariance_moments(y, labels, weights)
    w = np.diag(np.sqrt(weights))
    intercept = np.sqrt(weights)
    project = np.eye(len(y)) - np.outer(intercept, intercept) / weights.sum()
    matrices = [project @ w @ (codes[:, None] == codes[None, :]) @ w @ project for codes in labels.values()]
    matrices.append(project @ w @ w @ project)
    design = np.stack([matrix.reshape(-1) for matrix in matrices], axis=1)
    residual = project @ w @ y
    target = np.outer(residual, residual).reshape(-1)
    np.testing.assert_allclose(gram, design.T @ design, atol=1e-12)
    np.testing.assert_allclose(rhs, design.T @ target, atol=1e-12)
    expected, _ = nnls(design, target)
    fit = fit_components(y, labels, weights)
    np.testing.assert_allclose([fit["variance_components"][name] for name in names], expected, atol=1e-9)
