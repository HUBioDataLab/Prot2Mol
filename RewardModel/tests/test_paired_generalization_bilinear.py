import numpy as np
import torch

from experiments.paired_generalization.bilinear_completion import factor_objective, fit_factors
from experiments.paired_generalization.kronecker_ridge import ObservedKronecker


def test_factor_gradient_matches_explicit_observation_autograd_with_intercept():
    rng = np.random.default_rng(17)
    x, y = rng.normal(size=(5, 3)), rng.normal(size=(8, 4))
    p, m = rng.integers(5, size=40), rng.integers(8, size=40)
    labels = rng.normal(size=40)
    problem = ObservedKronecker(x, y, p, m, labels, batch_size=7)
    left = torch.tensor(rng.normal(size=(3, 2)), requires_grad=True)
    right = torch.tensor(rng.normal(size=(4, 2)), requires_grad=True)
    raw = ((torch.tensor(x[p]) @ left) * (torch.tensor(y[m]) @ right)).sum(1)
    error = raw - raw.mean() + labels.mean() - torch.tensor(labels)
    explicit = error.square().mean() / 2 + .03 * (left.square().sum() + right.square().sum()) / 2
    explicit.backward()
    value, grad_l, grad_r = factor_objective(problem, left.detach(), right.detach(), .03)
    np.testing.assert_allclose(float(value) + np.var(labels) / 2, float(explicit.detach()), atol=1e-10)
    np.testing.assert_allclose(grad_l, left.grad, atol=1e-10)
    np.testing.assert_allclose(grad_r, right.grad, atol=1e-10)


def test_rank_one_fit_recovers_held_out_pairs_from_noiseless_sparse_observations():
    rng = np.random.default_rng(55)
    x, y = rng.normal(size=(15, 3)), rng.normal(size=(18, 4))
    true = np.outer(rng.normal(size=3), rng.normal(size=4))
    p, m = rng.integers(15, size=120), rng.integers(18, size=120)
    labels = np.einsum('ni,ij,nj->n', x[p], true, y[m]) + 2.
    problem = ObservedKronecker(x, y, p, m, labels)
    left, right, bias, fit = fit_factors(problem, 1, 1e-8, max_iterations=400, tolerance=1e-9)
    query_p, query_m = np.repeat(np.arange(15), 18), np.tile(np.arange(18), 15)
    expected = np.einsum('ni,ij,nj->n', x[query_p], true, y[query_m]) + 2.
    np.testing.assert_allclose(problem.predict(left @ right.T, bias, query_p, query_m), expected, atol=1e-4)
    assert fit['final_centered_objective'] <= fit['initial_centered_objective']
