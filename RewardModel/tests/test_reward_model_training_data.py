import csv
import math

import torch
from datasets import Dataset

from conftest import DummyTokenizer
from reward_model.model import RewardModelConfig
from reward_model.training import (
    RewardPairCollator,
    RewardPairDataset,
    RewardTrainingDataConfig,
    build_pair_records,
    load_tokenized_example_dataset,
    prepare_tokenized_example_dataset,
    split_tokenized_examples_by_group,
)
from reward_model.training.data import (
    is_valid_negative_for_positive,
    required_fold_change_for_positive_pchembl,
    required_pchembl_margin_for_positive,
)


CURATED_FIELDNAMES = [
    "target_chembl_id",
    "protein_sequence",
    "assay_chembl_id",
    "parent_molregno",
    "molecule_chembl_id",
    "compound_selfies",
    "pchembl_value",
    "activity_label",
]


def _write_curated_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CURATED_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _tokenized_example_rows():
    return [
        {
            "example_id": 0,
            "group_id": "T1__A1",
            "target_chembl_id": "T1",
            "assay_chembl_id": "A1",
            "molecule_chembl_id": "M0",
            "protein_input_ids": [1, 1, 1, 0],
            "protein_attention_mask": [1, 1, 1, 0],
            "molecule_input_ids": [5, 5, 0, 0, 0],
            "molecule_attention_mask": [1, 1, 0, 0, 0],
            "activity_label": 0,
            "pchembl_value": 4.0,
        },
        {
            "example_id": 1,
            "group_id": "T1__A1",
            "target_chembl_id": "T1",
            "assay_chembl_id": "A1",
            "molecule_chembl_id": "M1",
            "protein_input_ids": [1, 1, 1, 0],
            "protein_attention_mask": [1, 1, 1, 0],
            "molecule_input_ids": [6, 6, 0, 0, 0],
            "molecule_attention_mask": [1, 1, 0, 0, 0],
            "activity_label": 1,
            "pchembl_value": 6.0,
        },
        {
            "example_id": 2,
            "group_id": "T1__A1",
            "target_chembl_id": "T1",
            "assay_chembl_id": "A1",
            "molecule_chembl_id": "M2",
            "protein_input_ids": [1, 1, 1, 0],
            "protein_attention_mask": [1, 1, 1, 0],
            "molecule_input_ids": [7, 7, 0, 0, 0],
            "molecule_attention_mask": [1, 1, 0, 0, 0],
            "activity_label": 1,
            "pchembl_value": 7.0,
        },
        {
            "example_id": 3,
            "group_id": "T2__A2",
            "target_chembl_id": "T2",
            "assay_chembl_id": "A2",
            "molecule_chembl_id": "M3",
            "protein_input_ids": [2, 2, 2, 0],
            "protein_attention_mask": [1, 1, 1, 0],
            "molecule_input_ids": [8, 8, 0, 0, 0],
            "molecule_attention_mask": [1, 1, 0, 0, 0],
            "activity_label": 0,
            "pchembl_value": 3.5,
        },
        {
            "example_id": 4,
            "group_id": "T2__A2",
            "target_chembl_id": "T2",
            "assay_chembl_id": "A2",
            "molecule_chembl_id": "M4",
            "protein_input_ids": [2, 2, 2, 0],
            "protein_attention_mask": [1, 1, 1, 0],
            "molecule_input_ids": [9, 9, 0, 0, 0],
            "molecule_attention_mask": [1, 1, 0, 0, 0],
            "activity_label": 1,
            "pchembl_value": 8.0,
        },
        {
            "example_id": 5,
            "group_id": "T3__A3",
            "target_chembl_id": "T3",
            "assay_chembl_id": "A3",
            "molecule_chembl_id": "M5",
            "protein_input_ids": [3, 3, 3, 0],
            "protein_attention_mask": [1, 1, 1, 0],
            "molecule_input_ids": [4, 4, 0, 0, 0],
            "molecule_attention_mask": [1, 1, 0, 0, 0],
            "activity_label": 0,
            "pchembl_value": 5.5,
        },
        {
            "example_id": 6,
            "group_id": "T3__A3",
            "target_chembl_id": "T3",
            "assay_chembl_id": "A3",
            "molecule_chembl_id": "M6",
            "protein_input_ids": [3, 3, 3, 0],
            "protein_attention_mask": [1, 1, 1, 0],
            "molecule_input_ids": [4, 5, 0, 0, 0],
            "molecule_attention_mask": [1, 1, 0, 0, 0],
            "activity_label": 0,
            "pchembl_value": 5.5,
        },
        {
            "example_id": 7,
            "group_id": "T4__A4",
            "target_chembl_id": "T4",
            "assay_chembl_id": "A4",
            "molecule_chembl_id": "M7",
            "protein_input_ids": [4, 4, 4, 0],
            "protein_attention_mask": [1, 1, 1, 0],
            "molecule_input_ids": [3, 3, 0, 0, 0],
            "molecule_attention_mask": [1, 1, 0, 0, 0],
            "activity_label": 0,
            "pchembl_value": 4.4,
        },
        {
            "example_id": 8,
            "group_id": "T4__A4",
            "target_chembl_id": "T4",
            "assay_chembl_id": "A4",
            "molecule_chembl_id": "M8",
            "protein_input_ids": [4, 4, 4, 0],
            "protein_attention_mask": [1, 1, 1, 0],
            "molecule_input_ids": [3, 6, 0, 0, 0],
            "molecule_attention_mask": [1, 1, 0, 0, 0],
            "activity_label": 0,
            "pchembl_value": 4.9,
        },
    ]


def test_prepare_tokenized_example_dataset_writes_expected_columns(tmp_path, monkeypatch):
    curated_rows = [
        {
            "target_chembl_id": "T1",
            "protein_sequence": "MKTAA",
            "assay_chembl_id": "A1",
            "parent_molregno": 10,
            "molecule_chembl_id": "M10",
            "compound_selfies": "[C][O]",
            "pchembl_value": 7.0,
            "activity_label": 1,
        },
        {
            "target_chembl_id": "T1",
            "protein_sequence": "MKTAA",
            "assay_chembl_id": "A1",
            "parent_molregno": 11,
            "molecule_chembl_id": "M11",
            "compound_selfies": "[N]",
            "pchembl_value": 4.0,
            "activity_label": 0,
        },
    ]
    curated_csv_path = tmp_path / "chembl_assay_rows.csv"
    _write_curated_csv(curated_csv_path, curated_rows)

    monkeypatch.setattr("reward_model.training.data.load_tokenizer", lambda *_args, **_kwargs: DummyTokenizer())

    artifacts = prepare_tokenized_example_dataset(
        RewardTrainingDataConfig(
            curated_data_path=str(curated_csv_path),
            tokenized_dataset_dir=str(tmp_path / "tokenized_examples"),
            tokenization_batch_size=2,
        ),
        RewardModelConfig(
            protein_model_name_or_path="dummy/protein",
            molecule_model_name_or_path="dummy/molecule",
            protein_max_length=6,
            molecule_max_length=8,
        ),
    )
    dataset = load_tokenized_example_dataset(artifacts.dataset_path)

    expected_columns = {
        "example_id",
        "group_id",
        "protein_input_ids",
        "protein_attention_mask",
        "molecule_input_ids",
        "molecule_attention_mask",
        "activity_label",
        "pchembl_value",
        "target_chembl_id",
        "assay_chembl_id",
        "molecule_chembl_id",
    }
    assert expected_columns.issubset(set(dataset.column_names))
    assert len(dataset) == 2
    assert len(dataset[0]["protein_input_ids"]) == 6
    assert len(dataset[0]["molecule_input_ids"]) == 8
    assert dataset[0]["group_id"] == "T1__A1"


def test_split_tokenized_examples_by_group_is_deterministic_and_has_no_overlap():
    dataset = Dataset.from_list(_tokenized_example_rows())

    train_a, eval_a, stats_a = split_tokenized_examples_by_group(
        dataset,
        eval_split_ratio=0.25,
        split_seed=7,
    )
    train_b, eval_b, stats_b = split_tokenized_examples_by_group(
        dataset,
        eval_split_ratio=0.25,
        split_seed=7,
    )

    assert stats_a == stats_b
    assert train_a["example_id"] == train_b["example_id"]
    assert eval_a["example_id"] == eval_b["example_id"]
    assert set(train_a["group_id"]).isdisjoint(set(eval_a["group_id"]))


def test_build_pair_records_creates_all_valid_pairs_and_skips_ties():
    dataset = Dataset.from_list(_tokenized_example_rows())

    pair_records, stats = build_pair_records(dataset)

    assert stats.num_groups == 4
    assert stats.num_groups_with_pairs == 2
    assert stats.num_pairs == 4
    assert all(record["group_id"] in {"T1__A1", "T2__A2"} for record in pair_records)
    assert all(record["positive_pchembl"] > record["negative_pchembl"] for record in pair_records)
    assert all(
        is_valid_negative_for_positive(record["positive_pchembl"], record["negative_pchembl"])
        for record in pair_records
    )

    t1_pairs = [record for record in pair_records if record["group_id"] == "T1__A1"]
    assert len(t1_pairs) == 3
    assert {record["positive_example_id"] for record in t1_pairs} == {1, 2}
    assert all(record["group_id"] == "T1__A1" for record in t1_pairs)

    t2_pairs = [record for record in pair_records if record["group_id"] == "T2__A2"]
    assert len(t2_pairs) == 1
    assert t2_pairs[0]["positive_example_id"] == 4
    assert t2_pairs[0]["negative_example_id"] == 3

    assert not any(record["group_id"] == "T4__A4" for record in pair_records)


def test_positive_dependent_pair_rule_boundaries_and_interpolation():
    assert required_fold_change_for_positive_pchembl(5.0) == 10.0
    assert math.isclose(required_pchembl_margin_for_positive(5.0), 1.0, rel_tol=1e-9)

    assert required_fold_change_for_positive_pchembl(8.0) == 2.0
    assert math.isclose(
        required_pchembl_margin_for_positive(8.0),
        math.log10(2.0),
        rel_tol=1e-9,
    )

    assert math.isclose(required_fold_change_for_positive_pchembl(6.5), 6.0, rel_tol=1e-9)
    assert math.isclose(
        required_pchembl_margin_for_positive(6.5),
        math.log10(6.0),
        rel_tol=1e-9,
    )


def test_positive_dependent_pair_rule_accepts_and_rejects_expected_negatives():
    assert is_valid_negative_for_positive(5.0, 4.0)
    assert not is_valid_negative_for_positive(5.0, 4.01)

    threshold_at_eight = 8.0 - math.log10(2.0)
    assert is_valid_negative_for_positive(8.0, threshold_at_eight)
    assert not is_valid_negative_for_positive(8.0, 7.8)

    threshold_at_mid = 6.5 - math.log10(6.0)
    assert is_valid_negative_for_positive(6.5, threshold_at_mid)
    assert not is_valid_negative_for_positive(6.5, 5.8)


def test_reward_pair_collator_flattens_pairs_and_preserves_duplicate_rows():
    dataset = Dataset.from_list(_tokenized_example_rows())
    pair_records = [
        {
            "positive_example_id": 2,
            "negative_example_id": 0,
            "group_id": "T1__A1",
            "positive_pchembl": 7.0,
            "negative_pchembl": 4.0,
        },
        {
            "positive_example_id": 2,
            "negative_example_id": 1,
            "group_id": "T1__A1",
            "positive_pchembl": 7.0,
            "negative_pchembl": 6.0,
        },
    ]
    pair_dataset = RewardPairDataset(dataset, pair_records)
    collator = RewardPairCollator()

    batch = collator([pair_dataset[0], pair_dataset[1]])

    assert batch["protein_input_ids"].shape == (4, 4)
    assert batch["molecule_input_ids"].shape == (4, 5)
    assert batch["activity_labels"].shape == (4,)
    assert batch["positive_indices"].tolist() == [0, 1]
    assert batch["negative_indices"].tolist() == [2, 3]
    assert int(batch["num_pairs"].item()) == 2
    assert batch["group_ids"] == ["T1__A1", "T1__A1"]
    assert torch.equal(batch["protein_input_ids"][0], batch["protein_input_ids"][1])
    assert batch["pchembl_values"].tolist() == [7.0, 7.0, 4.0, 6.0]
