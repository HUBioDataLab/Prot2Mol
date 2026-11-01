from itertools import combinations

import numpy as np
import pandas as pd

from experiments.paired_generalization.noise_band_audit import score_bands


def test_noise_band_exact_boundary_ties_and_macro_micro_denominators():
    frame = pd.DataFrame({"group_id": ["a"] * 4 + ["b"] * 2 + ["c"],
                          "pchembl_value": [0., 1., 2., 3., 0., 3., 0.],
                          "score": [3., 1., 1., 0., 0., 3., 1.]})
    summaries, tables = score_bands(frame, {"replicate": 1.}, ["score"])
    result = summaries["replicate/score"]
    expected = dict(all_pairs=0, comparable_pairs=0, concordant=0, discordant=0, prediction_ties=0, label_band_pairs=0)
    for _, group in frame.groupby("group_id"):
        for a, b in combinations(group.index, 2):
            expected["all_pairs"] += 1
            gap = frame.loc[a, "pchembl_value"] - frame.loc[b, "pchembl_value"]
            if abs(gap) <= 1.:
                expected["label_band_pairs"] += 1
                continue
            expected["comparable_pairs"] += 1
            product = gap * (frame.loc[a, "score"] - frame.loc[b, "score"])
            expected["concordant" if product > 0 else "discordant" if product < 0 else "prediction_ties"] += 1
    assert all(result[name] == value for name, value in expected.items())
    assert result["near_tie_pair_fraction"] == 3 / 7
    assert result["pooled_within_group_concordance"] == -2 / 7
    assert result["pooled_ordering_accuracy"] == .25
    assert result["macro_ordering_accuracy"] == .5
    assert result["groups_with_comparable_pairs"] == 2
    assert len(tables["replicate/score"]) == 3


def test_prediction_ties_get_half_credit_outside_label_noise_band():
    frame = pd.DataFrame({"group_id": ["x"] * 3, "pchembl_value": [0., 1., 3.], "score": [2., 2., 2.]})
    result, _ = score_bands(frame, {"noise": 1.}, ["score"])
    assert result["noise/score"]["pooled_ordering_accuracy"] == .5
    assert result["noise/score"]["pooled_within_group_concordance"] == 0.
