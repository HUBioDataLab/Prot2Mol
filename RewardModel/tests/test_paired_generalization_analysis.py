import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from experiments.paired_generalization.analyze_predictions import binary_logloss, fit_temperature, rescore, unit_bootstrap
from reward_model.training.evaluation import _compute_groupwise_spearman_with_records


def test_unit_bootstrap_preserves_multiple_groups_in_each_unit():
    frame = pd.DataFrame({"component": [0, 0, 1], "spearman": [1., -1., 1.]})
    result = unit_bootstrap(frame, repetitions=2000)
    assert result["units"] == 2
    assert result["interval"] == [0., 1.]
    assert unit_bootstrap(frame.assign(component=0))["interval"] is None


def test_calibration_reduces_validation_logloss_without_changing_ordering():
    labels = np.array([0, 1, 0, 1, 0, 1])
    logits = np.array([-12, 12, 10, -10, -9, 9])
    temperature = fit_temperature(labels, logits)
    assert temperature > 1
    assert binary_logloss(labels, logits, temperature) < binary_logloss(labels, logits)
    assert roc_auc_score(labels, logits) == roc_auc_score(labels, logits / temperature)


def test_rescore_reproduces_original_macro_and_reports_excluded_small_groups():
    frame = pd.DataFrame({
        "group_id": ["x__a"] * 3 + ["x__b"] * 2 + ["x__c"] * 3,
        "pchembl_value": [4., 5., 6., 7., 8., 6., 6., 6.],
        "ranking_score": [1., 2., 3., 2., 1., 1., 2., 3.],
        "activity_logits": [-1., 0., 1., 1., -1., 1., 1., 1.],
        "cosine_similarity": [.1, .2, .3, .2, .1, .1, .2, .3],
        "binary_label": [0, 0, 1, 1, 1, 1, 1, 1],
    })
    summaries, records = rescore(frame, {"a": 0, "b": 1, "c": 2})
    original, _ = _compute_groupwise_spearman_with_records(
        group_ids=frame.group_id, ranking_scores=frame.ranking_score,
        pchembl_values=frame.pchembl_value, min_pchembl_span=.5)
    ranking = summaries["ranking_score"]
    assert ranking["original_eligible_macro_spearman"] == pytest.approx(original["eval_macro_spearman"])
    assert ranking["macro_spearman"] == pytest.approx(0)
    assert ranking["by_group_size"]["2"]["macro_spearman"] == pytest.approx(-1)
    assert len(records["ranking_score"]) == 3
