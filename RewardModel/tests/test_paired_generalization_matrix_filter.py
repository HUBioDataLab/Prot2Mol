import numpy as np

from experiments.paired_generalization.matrix_filter import fit_smoother, observed_pairs, predict_smoother


def test_sparse_smoother_matches_exact_expanded_observation_kernel_ridge():
    targets = np.array(["a", "a", "a", "b", "b", "c", "c"])
    molecules = np.array(["1", "1", "2", "1", "3", "2", "3"])
    y = np.array([2., 4., 8., 5., 1., 4., 6.])
    alpha = .3
    kernel = (1. + (targets[:, None] == targets[None, :])) * (1. + (molecules[:, None] == molecules[None, :]))
    direct_dual = np.linalg.solve(kernel + alpha * np.eye(len(y)), y)
    pairs = observed_pairs(targets, molecules, y)
    model = fit_smoother(pairs, alpha)
    query_t = np.array(["a", "b", "a", "a", "new"])
    query_m = np.array(["1", "2", "new1", "new2", "new1"])
    query_kernel = (1. + (query_t[:, None] == targets[None, :])) * (1. + (query_m[:, None] == molecules[None, :]))
    result = predict_smoother(pairs, model, query_t, query_m)
    np.testing.assert_allclose(result, query_kernel @ direct_dual, atol=1e-7)
    assert result[2] == result[3]
    assert model["relative_residual"] < 2e-8
