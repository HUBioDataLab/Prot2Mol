import pandas as pd
import pytest

from reward_model.training.label_harmonization import harmonize_labels_by_pair_median


def _frame(rows):
    return pd.DataFrame(rows, columns=["target_chembl_id", "compound_id", "pchembl_value", "binary_label"])


def test_majority_active_pair_becomes_active_everywhere():
    train = _frame([("T1", "C1", 5.70, 0), ("T1", "C1", 6.67, 1)])
    test = _frame([("T1", "C1", 6.97, 1)])
    out, summary = harmonize_labels_by_pair_median({"train": train, "test": test})
    assert out["train"]["binary_label"].tolist() == [1, 1]
    assert out["test"]["binary_label"].tolist() == [1]
    assert summary["pairs_conflicting"] == 1
    assert summary["rows_negative_to_positive"] == 1


def test_majority_inactive_pair_becomes_inactive_everywhere():
    train = _frame([("T1", "C1", 5.07, 0), ("T1", "C1", 5.11, 0), ("T1", "C1", 6.26, 1)])
    out, summary = harmonize_labels_by_pair_median({"train": train})
    assert out["train"]["binary_label"].tolist() == [0, 0, 0]
    assert summary["rows_positive_to_negative"] == 1


def test_outlier_does_not_flip_a_potent_pair():
    train = _frame([("T1", "C1", 5.55, 0), ("T1", "C1", 8.00, 1),
                    ("T1", "C1", 8.70, 1), ("T1", "C1", 8.70, 1)])
    out, _ = harmonize_labels_by_pair_median({"train": train})
    assert out["train"]["binary_label"].tolist() == [1, 1, 1, 1]


def test_even_count_uses_mean_of_middle_values():
    train = _frame([("T1", "C1", 5.27, 0), ("T1", "C1", 6.23, 1)])
    out, _ = harmonize_labels_by_pair_median({"train": train})
    assert out["train"]["binary_label"].tolist() == [0, 0]


def test_median_exactly_at_threshold_is_active():
    train = _frame([("T1", "C1", 5.5, 0), ("T1", "C1", 6.5, 1)])
    out, _ = harmonize_labels_by_pair_median({"train": train})
    assert out["train"]["binary_label"].tolist() == [1, 1]


def test_non_conflicting_pairs_are_untouched():
    train = _frame([("T1", "C1", 7.0, 1), ("T2", "C2", 4.0, 0)])
    out, summary = harmonize_labels_by_pair_median({"train": train})
    assert out["train"]["binary_label"].tolist() == [1, 0]
    assert summary["rows_relabelled"] == 0


def test_pchembl_values_are_never_modified():
    train = _frame([("T1", "C1", 5.70, 0), ("T1", "C1", 6.67, 1)])
    out, _ = harmonize_labels_by_pair_median({"train": train})
    assert out["train"]["pchembl_value"].tolist() == [5.70, 6.67]


def test_source_frames_are_not_mutated():
    train = _frame([("T1", "C1", 5.70, 0), ("T1", "C1", 6.67, 1)])
    harmonize_labels_by_pair_median({"train": train})
    assert train["binary_label"].tolist() == [0, 1]


def test_labels_agree_across_splits_for_shared_pairs():
    train = _frame([("T1", "C1", 6.90, 1)])
    val = _frame([("T1", "C1", 5.10, 0)])
    out, _ = harmonize_labels_by_pair_median({"train": train, "val": val})
    assert out["train"]["binary_label"].tolist() == out["val"]["binary_label"].tolist()


def test_pair_key_can_include_activity_type():
    frame = pd.DataFrame(
        [("T1", "C1", "IC50", 5.0, 0), ("T1", "C1", "Ki", 7.0, 1)],
        columns=["target_chembl_id", "compound_id", "activity_type", "pchembl_value", "binary_label"],
    )
    out, _ = harmonize_labels_by_pair_median(
        {"train": frame}, pair_key=("target_chembl_id", "compound_id", "activity_type")
    )
    assert out["train"]["binary_label"].tolist() == [0, 1]


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="missing columns"):
        harmonize_labels_by_pair_median({"train": pd.DataFrame({"target_chembl_id": ["T1"]})})


def test_null_pchembl_raises():
    train = _frame([("T1", "C1", None, 0)])
    with pytest.raises(ValueError, match="null pchembl_value"):
        harmonize_labels_by_pair_median({"train": train})


def test_prepare_split_rows_respects_the_enforcement_flag():
    from reward_model.training.data import _prepare_split_rows

    # harmonised row: label says active, its own pchembl says inactive
    rows = [{
        "target_chembl_id": "T1", "assay_id": "A1", "compound_id": "C1",
        "protein_sequence": "MKV", "compound_selfies": "[C]",
        "pchembl_value": 5.4, "binary_label": 1, "activity_type": "IC50",
    }]
    with pytest.raises(ValueError, match="does not match the configured pChEMBL"):
        _prepare_split_rows(rows, activity_threshold=6.0)

    prepared = _prepare_split_rows(rows, activity_threshold=6.0,
                                   enforce_binary_label_threshold=False)
    assert prepared[0]["binary_label"] == 1
    assert prepared[0]["pchembl_value"] == 5.4   # raw value preserved
