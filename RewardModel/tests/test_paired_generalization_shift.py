import numpy as np
import pandas as pd
import pytest

from experiments.paired_generalization.shift_audit import c2st, estimate_label_prior, holm_adjust, unit_domain_weights, unit_test


def test_bbse_recovers_shift_without_target_labels_and_rejects_singular_classifier():
    # Columns are true labels: source is balanced, classifier has TPR .8 and FPR .1.
    counts = np.array([[900, 200], [100, 800]])
    target = np.array([410, 590])  # A .7 target positive prior under the same confusion.
    result = estimate_label_prior(counts, target)
    assert result["identifiable"]
    assert result["estimated_positive_prior"] == pytest.approx(.7)
    assert result["source_positive_prior"] == .5
    assert result["marginal_residual_l2"] < 1e-14
    assert not estimate_label_prior([[500, 500], [500, 500]], target)["identifiable"]
    outside = estimate_label_prior(counts, [0, 1000])
    assert outside["estimated_positive_prior"] == 1.
    assert outside["boundary_constraint_active"]


def test_c2st_giant_unit_cannot_dominate_training_mass_and_domains_are_balanced():
    frame = pd.DataFrame({"unit": ["giant"] * 100 + ["small", "right1", "right2"], "domain": [0] * 101 + [1, 1]})
    frame["weight"] = unit_domain_weights(frame)
    by_unit = frame.groupby("unit").weight.sum()
    np.testing.assert_allclose(by_unit, len(frame) / 4)
    np.testing.assert_allclose(frame.groupby("domain").weight.sum(), len(frame) / 2)
    bad = pd.DataFrame({"unit": ["same", "same"], "domain": [0, 1]})
    with pytest.raises(ValueError):
        unit_domain_weights(bad)


def test_unit_permutation_test_detects_separation_but_not_constant_predictions():
    domains = np.repeat([0, 1], 20)
    separated = unit_test(domains, domains, repetitions=99)
    assert separated["unit_balanced_accuracy"] == 1.
    assert separated["whole_unit_permutation_p_one_sided"] == .01
    constant = unit_test(np.ones(40), domains, repetitions=99)
    assert constant["unit_balanced_accuracy"] == .5
    assert constant["whole_unit_permutation_p_one_sided"] == 1.
    assert holm_adjust([.01, .03, .9]) == [.03, .06, .9]


def test_c2st_holds_complete_components_out_of_classifier_fit():
    frames = []
    for domain in (0, 1):
        frame = pd.DataFrame({"component": np.repeat(np.arange(40) + domain * 40, 3),
                              "group_id": [f"{domain}_{i}" for i in range(120)]})
        for field in ("size", "variance", "span", "replicate_mean", "replicate_max", "target_training_groups"):
            frame[field] = 1.
        frame["positive_rate"] = domain
        frame["low"], frame["high"] = 6., 7.
        frames.append(frame)
    result, heldout = c2st(*frames, regime="B", repetitions=19)
    assert not (set(result["training_units"]) & set(result["heldout_units"]))
    assert set(heldout.unit) == set(result["heldout_units"])
    assert (heldout.groupby("unit").size() == 3).all()
    assert result["unit_balanced_accuracy"] == 1.
