import numpy as np
import pytest
from sklearn.linear_model import Ridge

from experiments.paired_generalization.independent_ridge import predict_tasks, ridge_path


@pytest.mark.parametrize("entities,dimensions", [(4, 8), (12, 3)])
def test_primal_and_dual_paths_match_expanded_observed_measurements(entities, dimensions):
    rng = np.random.default_rng(51)
    x = rng.normal(size=(entities, dimensions))
    y = rng.normal(size=entities)
    counts = rng.integers(1, 5, size=entities)
    strengths = [.0001, .1, 10.]
    coefficients, intercepts = ridge_path(x, y, counts, strengths)
    expanded_x = np.repeat(x, counts, axis=0)
    expanded_y = np.repeat(y, counts)
    for i, strength in enumerate(strengths):
        expected = Ridge(alpha=strength * counts.sum()).fit(expanded_x, expanded_y)
        np.testing.assert_allclose(x @ coefficients[i] + intercepts[i], expected.predict(x), atol=1e-9)


def test_single_observed_molecule_returns_constant_and_unseen_target_uses_explicit_fallback():
    coefficients, intercepts = ridge_path([[1., 2., 3.]], [7.], [5], [.01, 1.])
    np.testing.assert_array_equal(coefficients, 0.)
    np.testing.assert_array_equal(intercepts, 7.)
    predictions = predict_tasks(np.eye(3), np.array([0, 1]), np.array([0, -1]), coefficients[:1], intercepts[:1], 6.)
    np.testing.assert_array_equal(predictions, [7., 6.])
