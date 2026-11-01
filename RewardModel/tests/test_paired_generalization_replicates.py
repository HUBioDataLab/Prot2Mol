import numpy as np
import pandas as pd
import pytest

from experiments.paired_generalization.replicate_audit import noise_scale, reconstruct, split_half


def fixture():
    raw = pd.DataFrame({"activity_id": range(8), "assay_id": ["a"] * 6 + ["b"] * 2,
                        "target_chembl_id": ["x"] * 8, "compound_id": ["1", "1", "2", "2", "3", "3", "4", "4"],
                        "pchembl_value": [4., 4., 6., 6., 8., 8., 5., 7.]})
    source = pd.DataFrame({"assay_id": ["a"] * 3 + ["b"], "target_chembl_id": ["x"] * 4,
                           "compound_id": ["1", "2", "3", "4"], "pchembl_value": [4., 6., 8., 6.],
                           "measurement_count": [2] * 4})
    return raw, source


def test_recovery_checks_medians_and_counts_separately():
    raw, source = fixture()
    obs, matched = reconstruct(raw, source)
    assert obs.verified.all() and len(matched) == 8
    broken = source.copy()
    broken.loc[0, "pchembl_value"] += 1
    broken.loc[1, "measurement_count"] += 1
    obs, _ = reconstruct(raw, broken)
    assert obs.median_matches.tolist() == [False, True, True, True]
    assert obs.count_matches.tolist() == [True, False, True, True]
    with pytest.raises(ValueError):
        reconstruct(pd.concat([raw, raw.iloc[:1]]), source)


def test_noise_scale_equals_direct_replicate_pair_rms():
    raw, source = fixture()
    obs, _ = reconstruct(raw, source)
    expected = np.sqrt(np.mean([0., 0., 0., 4.]))
    assert noise_scale(obs)["rms_replicate_difference"] == expected


def test_split_half_reports_coverage_and_perfect_replicate_reliability():
    raw, source = fixture()
    obs, matched = reconstruct(raw, source)
    result = split_half(obs, matched, repetitions=5)
    assert result["groups_with_at_least_three_replicated_members"] == 1
    assert result["replicated_members"] == 3
    assert result["mean_split_half_macro_spearman"] == pytest.approx(1)
    assert result["mean_spearman_brown_heuristic"] == pytest.approx(1)
    singleton = obs.assign(replicate_count=1)
    assert split_half(singleton, matched)["mean_split_half_macro_spearman"] is None
