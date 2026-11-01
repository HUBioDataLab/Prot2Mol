import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import Ridge

from experiments.paired_generalization.kronecker_ridge import ObservedKronecker, shuffle_proteins_by_group


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_observed_kronecker_matches_expanded_ridge_with_missing_and_repeated_pairs(device):
    rng = np.random.default_rng(123)
    x = np.column_stack([rng.normal(size=(4, 3)), np.ones(4)])
    y = np.column_stack([rng.normal(size=(6, 2)), np.ones(6)])
    p = np.array([0, 0, 0, 1, 2, 2, 3, 3, 3])
    m = np.array([0, 0, 1, 3, 1, 4, 2, 4, 5])
    labels = rng.normal(size=len(p))
    z = np.einsum('ni,nj->nij', x[p], y[m]).reshape(len(p), -1)
    problem = ObservedKronecker(x, y, p, m, labels, batch_size=3, device=device)
    query_p = np.array([0, 1, 2, 3])
    query_m = np.array([0, 5, 2, 3])
    query_z = np.einsum('ni,nj->nij', x[query_p], y[query_m]).reshape(4, -1)
    previous = None
    for strength in (1., .01, 1e-4):
        reference = Ridge(alpha=len(p) * strength).fit(z, labels)
        previous, intercept, fit = problem.solve(strength, initial=previous, tolerance=1e-11)
        result = problem.predict(previous, intercept, query_p, query_m)
        np.testing.assert_allclose(result, reference.predict(query_z), atol=1e-8)
        np.testing.assert_allclose(previous.cpu().numpy().ravel(), reference.coef_, atol=1e-8)
        assert fit['relative_residual'] <= 2e-11


def test_protein_shuffle_keeps_assays_whole_and_preserves_group_frequencies():
    groups = np.repeat(np.arange(12).astype(str), np.arange(1, 13))
    frame = pd.DataFrame({'group_id': groups})
    original = np.repeat(np.arange(12) % 4, np.arange(1, 13))
    shuffled, unchanged = shuffle_proteins_by_group(frame, original, 42)
    by_group = frame.assign(original=original, shuffled=shuffled).groupby('group_id')
    assert (by_group.shuffled.nunique() == 1).all()
    np.testing.assert_array_equal(np.sort(by_group.original.first()), np.sort(by_group.shuffled.first()))
    assert unchanged < 1.
    np.testing.assert_array_equal(shuffled, shuffle_proteins_by_group(frame, original, 42)[0])
