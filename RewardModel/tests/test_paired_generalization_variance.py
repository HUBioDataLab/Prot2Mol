import numpy as np
import pytest
from scipy.optimize import nnls

from experiments.paired_generalization.variance_audit import covariance_moments, fit_components


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
