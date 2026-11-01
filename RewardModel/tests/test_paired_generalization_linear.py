import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from experiments.paired_generalization.baseline_common import RankingSelection
from experiments.paired_generalization.linear_baselines import logistic_parameters, predict, ridge_parameters, sufficient_statistics
from reward_model.training.evaluation import _compute_groupwise_spearman_with_records


def fixture():
    features = np.random.default_rng(7).normal(size=(7, 4))
    features[-1] = 1e6  # A held-out entity must not affect training standardization.
    indices = np.array([0, 1, 1, 2, 3, 3, 3, 4, 5, 5])
    return features, indices


def test_aggregated_entity_ridge_matches_expanded_observation_fit():
    features, indices = fixture()
    y = np.array([1., 2., 4., 7., 9., 8., 6., 3., 2., 1.])
    stats = sufficient_statistics(features, indices, y, batch_size=2)
    x = features[indices]
    np.testing.assert_allclose(stats["mean"], x.mean(axis=0), atol=1e-14)
    np.testing.assert_allclose(stats["scale"], x.std(axis=0), atol=1e-14)
    regularization = .03
    expected = Ridge(alpha=regularization * len(y)).fit((x - x.mean(axis=0)) / x.std(axis=0), y)
    coefficient, intercept = ridge_parameters(stats, regularization)
    np.testing.assert_allclose(predict(features, indices, coefficient, intercept, batch_size=2),
                               expected.predict((x - x.mean(axis=0)) / x.std(axis=0)), atol=1e-10)
    assert 6 not in stats["selected"]


def test_aggregated_binomial_fit_matches_expanded_binary_logistic_fit():
    features, indices = fixture()
    y = np.array([0., 1., 0., 1., 1., 1., 0., 0., 1., 0.])
    stats = sufficient_statistics(features, indices, y, batch_size=3)
    x = (features[indices] - stats["mean"]) / stats["scale"]
    regularization = .03
    expected = LogisticRegression(C=1 / (regularization * len(y)), tol=1e-11, max_iter=10000).fit(x, y)
    coefficient, intercept, fit = logistic_parameters(features, stats, regularization)
    result = predict(features, indices, coefficient, intercept, batch_size=2)
    np.testing.assert_allclose(result, expected.decision_function(x), atol=2e-5)
    assert fit["maximum_gradient"] < 1e-5
    assert result[1] == result[2]
    assert result[4] == result[5] == result[6]


def test_fast_selection_matches_original_eligibility_ties_and_constant_predictions():
    frame = pd.DataFrame({"group_id": ["a"] * 3 + ["b"] * 3 + ["c"] * 2 + ["d"] * 3 + ["e"] * 4,
                          "pchembl_value": [0, 1, 2, 0, 1, 2, 0, 2, 0, .1, .2, 0, 0, 2, 3]})
    frame["group_id"] = "target__" + frame.group_id
    scores = np.array([0, 1, 2, 1, 1, 1, 0, 1, 0, 1, 2, 4, 3, 2, 1], dtype=float)
    expected, _ = _compute_groupwise_spearman_with_records(group_ids=frame.group_id,
        ranking_scores=scores, pchembl_values=frame.pchembl_value, min_group_size=3, min_pchembl_span=.5)
    result = RankingSelection(frame)(scores)
    assert result["macro_spearman"] == pytest.approx(expected["eval_macro_spearman"])
    assert result["eligible_groups"] == 3
    assert result["defined_groups"] == 2
    assert result["defined_fraction"] == pytest.approx(2 / 3)
