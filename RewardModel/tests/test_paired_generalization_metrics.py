import itertools

import numpy as np
import pytest

from experiments.paired_generalization.metrics import aggregate_groups, group_metrics, pair_counts


@pytest.mark.parametrize("epsilon", [0., .25, .5, 1.])
def test_fast_pair_counts_match_explicit_pairs_with_ties(epsilon):
    rng = np.random.default_rng(101)
    y = rng.integers(0, 8, size=80) / 4
    s = rng.integers(0, 6, size=80) / 3
    expected = {"concordant": 0, "discordant": 0, "prediction_ties": 0, "label_band_pairs": 0}
    for i, j in itertools.combinations(range(len(y)), 2):
        if abs(y[i] - y[j]) <= epsilon:
            expected["label_band_pairs"] += 1
        elif s[i] == s[j]:
            expected["prediction_ties"] += 1
        elif (y[i] - y[j]) * (s[i] - s[j]) > 0:
            expected["concordant"] += 1
        else:
            expected["discordant"] += 1
    result = pair_counts(y, s, epsilon)
    for key, value in expected.items():
        assert getattr(result, key) == value
    assert result.comparable_pairs + result.label_band_pairs == result.all_pairs


def test_small_group_rank_values_and_macro_pair_weighting_differ():
    small = group_metrics([1, 2], [2, 1])
    large = group_metrics([1, 2, 3, 4], [1, 2, 3, 4])
    result = aggregate_groups([small, large])
    assert small["spearman"] == pytest.approx(-1)
    assert result["macro_spearman"] == pytest.approx(0)
    assert result["pooled_within_group_concordance"] == pytest.approx(5 / 7)
    assert result["original_eligible_macro_spearman"] == pytest.approx(1)
    values = {round(group_metrics([1, 2, 3], p)["spearman"], 8) for p in itertools.permutations([1, 2, 3])}
    assert values == {-1., -.5, .5, 1.}


def test_constant_groups_are_explicitly_undefined_and_predicted_ties_get_half_credit():
    constant_y = group_metrics([2, 2, 2], [1, 2, 3])
    constant_s = group_metrics([1, 2, 3], [2, 2, 2])
    singleton = group_metrics([1], [1])
    assert all(row["spearman"] is None for row in [constant_y, constant_s, singleton])
    assert constant_s["ordering_accuracy"] == .5
    assert constant_s["kendall_tau_b"] is None
    assert aggregate_groups([constant_y, constant_s, singleton])["defined_fraction"] == 0


def test_nonfinite_values_are_not_silently_dropped():
    with pytest.raises(ValueError):
        pair_counts([1, np.nan], [1, 2])
