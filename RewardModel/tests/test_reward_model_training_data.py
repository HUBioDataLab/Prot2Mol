import math

import pytest
import torch
from datasets import Dataset

from conftest import DummyTokenizer
from reward_model.model import RewardModelConfig
from reward_model.training import (
    PAIR_DATASET_COLUMNS,
    RewardPairCollator,
    RewardPairDataset,
    RewardTrainingDataConfig,
    build_pair_records,
    get_saved_pair_dataset_paths,
    get_tokenized_split_dataset_paths,
    load_saved_pair_dataset,
    load_tokenized_example_dataset,
    prepare_tokenized_split_datasets,
    save_pair_dataset_from_example_dataset,
)
from reward_model.training.data import (
    TOKENIZED_DATASET_COLUMNS,
    is_valid_negative_for_positive,
    required_fold_change_for_positive_pchembl,
    required_pchembl_margin_for_positive,
)


def _write_split_parquet(path, rows):
    pd = pytest.importorskip("pandas")
    pd.DataFrame(rows).to_parquet(path, index=False)


def _split_parquet_rows():
    return {
        "train": [
            {
                "source": "Papyrus",
                "assay_id": "A1",
                "target_id": "TARGET_A",
                "target_chembl_id": "T1",
                "protein_sequence": "MKTAA",
                "compound_id": "M0",
                "compound_selfies": "[C][O]",
                "pchembl_value": 4.0,
                "binary_label": 0,
                "assay_group_id": "NOT_USED_FOR_PAIRING",
                "split": "train",
            },
            {
                "source": "Papyrus",
                "assay_id": "A1",
                "target_id": "TARGET_A",
                "target_chembl_id": "T1",
                "protein_sequence": "MKTAA",
                "compound_id": "M1",
                "compound_selfies": "[N]",
                "pchembl_value": 6.0,
                "binary_label": 1,
                "assay_group_id": "NOT_USED_FOR_PAIRING",
                "split": "train",
            },
            {
                "source": "Papyrus",
                "assay_id": "A1",
                "target_id": "TARGET_A",
                "target_chembl_id": "T1",
                "protein_sequence": "MKTAA",
                "compound_id": "M2",
                "compound_selfies": "[O]",
                "pchembl_value": 7.0,
                "binary_label": 1,
                "assay_group_id": "NOT_USED_FOR_PAIRING",
                "split": "train",
            },
            {
                "source": "Papyrus",
                "assay_id": "A2",
                "target_id": "TARGET_B",
                "target_chembl_id": "T2",
                "protein_sequence": "GGGGG",
                "compound_id": "M3",
                "compound_selfies": "[C][C]",
                "pchembl_value": 3.5,
                "binary_label": 0,
                "assay_group_id": "NOT_USED_FOR_PAIRING",
                "split": "train",
            },
            {
                "source": "Papyrus",
                "assay_id": "A2",
                "target_id": "TARGET_B",
                "target_chembl_id": "T2",
                "protein_sequence": "GGGGG",
                "compound_id": "M4",
                "compound_selfies": "[C][N]",
                "pchembl_value": 8.0,
                "binary_label": 1,
                "assay_group_id": "NOT_USED_FOR_PAIRING",
                "split": "train",
            },
        ],
        "val": [
            {
                "source": "Papyrus",
                "assay_id": "A3",
                "target_id": "TARGET_C",
                "target_chembl_id": "T3",
                "protein_sequence": "TTTTT",
                "compound_id": "M5",
                "compound_selfies": "[S]",
                "pchembl_value": 4.0,
                "binary_label": 0,
                "assay_group_id": "ALSO_NOT_USED",
                "split": "val",
            },
            {
                "source": "Papyrus",
                "assay_id": "A3",
                "target_id": "TARGET_C",
                "target_chembl_id": "T3",
                "protein_sequence": "TTTTT",
                "compound_id": "M6",
                "compound_selfies": "[P]",
                "pchembl_value": 7.0,
                "binary_label": 1,
                "assay_group_id": "ALSO_NOT_USED",
                "split": "val",
            },
        ],
        "test": [
            {
                "source": "Papyrus",
                "assay_id": "A4",
                "target_id": "TARGET_D",
                "target_chembl_id": "T4",
                "protein_sequence": "CCCCC",
                "compound_id": "M7",
                "compound_selfies": "[F]",
                "pchembl_value": 5.0,
                "binary_label": 0,
                "assay_group_id": "UNUSED",
                "split": "test",
            },
            {
                "source": "Papyrus",
                "assay_id": "A4",
                "target_id": "TARGET_D",
                "target_chembl_id": "T4",
                "protein_sequence": "CCCCC",
                "compound_id": "M8",
                "compound_selfies": "[Cl]",
                "pchembl_value": 8.0,
                "binary_label": 1,
                "assay_group_id": "UNUSED",
                "split": "test",
            },
        ],
    }


def _tokenized_example_rows():
    return [
        {
            "example_id": 0,
            "group_id": "T1__A1",
            "target_chembl_id": "T1",
            "assay_id": "A1",
            "compound_id": "M0",
            "protein_input_ids": [1, 1, 1, 0],
            "protein_attention_mask": [1, 1, 1, 0],
            "molecule_input_ids": [5, 5, 0, 0, 0],
            "molecule_attention_mask": [1, 1, 0, 0, 0],
            "binary_label": 0,
            "pchembl_value": 4.0,
        },
        {
            "example_id": 1,
            "group_id": "T1__A1",
            "target_chembl_id": "T1",
            "assay_id": "A1",
            "compound_id": "M1",
            "protein_input_ids": [1, 1, 1, 0],
            "protein_attention_mask": [1, 1, 1, 0],
            "molecule_input_ids": [6, 6, 0, 0, 0],
            "molecule_attention_mask": [1, 1, 0, 0, 0],
            "binary_label": 1,
            "pchembl_value": 6.0,
        },
        {
            "example_id": 2,
            "group_id": "T1__A1",
            "target_chembl_id": "T1",
            "assay_id": "A1",
            "compound_id": "M2",
            "protein_input_ids": [1, 1, 1, 0],
            "protein_attention_mask": [1, 1, 1, 0],
            "molecule_input_ids": [7, 7, 0, 0, 0],
            "molecule_attention_mask": [1, 1, 0, 0, 0],
            "binary_label": 1,
            "pchembl_value": 7.0,
        },
        {
            "example_id": 3,
            "group_id": "T2__A2",
            "target_chembl_id": "T2",
            "assay_id": "A2",
            "compound_id": "M3",
            "protein_input_ids": [2, 2, 2, 0],
            "protein_attention_mask": [1, 1, 1, 0],
            "molecule_input_ids": [8, 8, 0, 0, 0],
            "molecule_attention_mask": [1, 1, 0, 0, 0],
            "binary_label": 0,
            "pchembl_value": 3.5,
        },
        {
            "example_id": 4,
            "group_id": "T2__A2",
            "target_chembl_id": "T2",
            "assay_id": "A2",
            "compound_id": "M4",
            "protein_input_ids": [2, 2, 2, 0],
            "protein_attention_mask": [1, 1, 1, 0],
            "molecule_input_ids": [9, 9, 0, 0, 0],
            "molecule_attention_mask": [1, 1, 0, 0, 0],
            "binary_label": 1,
            "pchembl_value": 8.0,
        },
        {
            "example_id": 5,
            "group_id": "T3__A3",
            "target_chembl_id": "T3",
            "assay_id": "A3",
            "compound_id": "M5",
            "protein_input_ids": [3, 3, 3, 0],
            "protein_attention_mask": [1, 1, 1, 0],
            "molecule_input_ids": [4, 4, 0, 0, 0],
            "molecule_attention_mask": [1, 1, 0, 0, 0],
            "binary_label": 0,
            "pchembl_value": 5.5,
        },
        {
            "example_id": 6,
            "group_id": "T3__A3",
            "target_chembl_id": "T3",
            "assay_id": "A3",
            "compound_id": "M6",
            "protein_input_ids": [3, 3, 3, 0],
            "protein_attention_mask": [1, 1, 1, 0],
            "molecule_input_ids": [4, 5, 0, 0, 0],
            "molecule_attention_mask": [1, 1, 0, 0, 0],
            "binary_label": 0,
            "pchembl_value": 5.5,
        },
        {
            "example_id": 7,
            "group_id": "T4__A4",
            "target_chembl_id": "T4",
            "assay_id": "A4",
            "compound_id": "M7",
            "protein_input_ids": [4, 4, 4, 0],
            "protein_attention_mask": [1, 1, 1, 0],
            "molecule_input_ids": [3, 3, 0, 0, 0],
            "molecule_attention_mask": [1, 1, 0, 0, 0],
            "binary_label": 0,
            "pchembl_value": 4.4,
        },
        {
            "example_id": 8,
            "group_id": "T4__A4",
            "target_chembl_id": "T4",
            "assay_id": "A4",
            "compound_id": "M8",
            "protein_input_ids": [4, 4, 4, 0],
            "protein_attention_mask": [1, 1, 1, 0],
            "molecule_input_ids": [3, 6, 0, 0, 0],
            "molecule_attention_mask": [1, 1, 0, 0, 0],
            "binary_label": 0,
            "pchembl_value": 4.9,
        },
    ]


def test_prepare_tokenized_split_datasets_writes_expected_minimal_columns(tmp_path, monkeypatch):
    split_rows = _split_parquet_rows()
    split_rows["train"][0]["activity_type"] = "Potency"
    for split_name, rows in split_rows.items():
        _write_split_parquet(tmp_path / f"{split_name}.parquet", rows)

    monkeypatch.setattr("reward_model.training.data.load_tokenizer", lambda *_args, **_kwargs: DummyTokenizer())

    artifacts = prepare_tokenized_split_datasets(
        RewardTrainingDataConfig(
            train_parquet_path=str(tmp_path / "train.parquet"),
            val_parquet_path=str(tmp_path / "val.parquet"),
            test_parquet_path=str(tmp_path / "test.parquet"),
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
    split_paths = get_tokenized_split_dataset_paths(artifacts.base_dir)
    train_dataset = load_tokenized_example_dataset(split_paths["train"])
    val_dataset = load_tokenized_example_dataset(split_paths["val"])
    test_dataset = load_tokenized_example_dataset(split_paths["test"])

    assert list(train_dataset.column_names) == list(TOKENIZED_DATASET_COLUMNS)
    assert list(val_dataset.column_names) == list(TOKENIZED_DATASET_COLUMNS)
    assert list(test_dataset.column_names) == list(TOKENIZED_DATASET_COLUMNS)

    assert len(train_dataset) == 5
    assert len(val_dataset) == 2
    assert len(test_dataset) == 2
    assert artifacts.train_groups == 2
    assert artifacts.val_groups == 1
    assert artifacts.test_groups == 1

    assert train_dataset[0]["group_id"] == "T1__A1"
    assert train_dataset[0]["compound_id"] == "M0"
    assert train_dataset[0]["binary_label"] == 0
    assert train_dataset[0]["activity_type"] == "Potency"
    assert val_dataset[0]["activity_type"] == "Unknown"
    assert len(train_dataset[0]["protein_input_ids"]) == 6
    assert len(train_dataset[0]["molecule_input_ids"]) == 8
    assert train_dataset[0]["protein_length"] == 5
    assert train_dataset[0]["molecule_length"] == 6

    assert "assay_group_id" not in train_dataset.column_names
    assert "source" not in train_dataset.column_names
    assert "split" not in train_dataset.column_names


def test_prepare_tokenized_split_datasets_uses_smiles_with_molformer_tokenizer(
    tmp_path,
    monkeypatch,
):
    split_rows = _split_parquet_rows()
    expected_smiles = set()
    for split_name, rows in split_rows.items():
        for index, row in enumerate(rows):
            smiles = f"C{index}O"
            row["smiles"] = smiles
            expected_smiles.add(smiles)
        _write_split_parquet(tmp_path / f"{split_name}.parquet", rows)

    protein_tokenizer = DummyTokenizer()
    molecule_tokenizer = DummyTokenizer(pad_token_id=2)
    tokenizer_loads = []

    def _load_tokenizer(name_or_path, tokenizer_kwargs=None):
        tokenizer_loads.append((name_or_path, tokenizer_kwargs))
        return (
            molecule_tokenizer
            if name_or_path == "ibm/MoLFormer-XL-both-10pct"
            else protein_tokenizer
        )

    monkeypatch.setattr(
        "reward_model.training.data.load_tokenizer",
        _load_tokenizer,
    )

    prepare_tokenized_split_datasets(
        RewardTrainingDataConfig(
            train_parquet_path=str(tmp_path / "train.parquet"),
            val_parquet_path=str(tmp_path / "val.parquet"),
            test_parquet_path=str(tmp_path / "test.parquet"),
            tokenized_dataset_dir=str(tmp_path / "tokenized_smiles"),
            tokenization_batch_size=2,
        ),
        RewardModelConfig(
            protein_model_name_or_path="dummy/protein",
            molecule_model_name_or_path="ibm/MoLFormer-XL-both-10pct",
            molecule_input_representation="smiles",
            molecule_trust_remote_code=True,
            molecule_deterministic_eval=True,
            protein_max_length=16,
            molecule_max_length=16,
        ),
    )

    assert tokenizer_loads[1] == (
        "ibm/MoLFormer-XL-both-10pct",
        {"trust_remote_code": True},
    )
    tokenized_molecule_texts = {
        text
        for call in molecule_tokenizer.calls
        for text in call["texts"]
    }
    assert tokenized_molecule_texts == expected_smiles
    assert not any(text.startswith("[") for text in tokenized_molecule_texts)


def test_prepare_tokenized_split_datasets_rejects_labels_inconsistent_with_threshold(
    tmp_path,
    monkeypatch,
):
    split_rows = _split_parquet_rows()
    split_rows["train"][0]["binary_label"] = 1
    for split_name, rows in split_rows.items():
        _write_split_parquet(tmp_path / f"{split_name}.parquet", rows)
    monkeypatch.setattr(
        "reward_model.training.data.load_tokenizer",
        lambda *_args, **_kwargs: DummyTokenizer(),
    )

    with pytest.raises(ValueError, match="binary_label does not match"):
        prepare_tokenized_split_datasets(
            RewardTrainingDataConfig(
                train_parquet_path=str(tmp_path / "train.parquet"),
                val_parquet_path=str(tmp_path / "val.parquet"),
                test_parquet_path=str(tmp_path / "test.parquet"),
                tokenized_dataset_dir=str(tmp_path / "tokenized_examples"),
            ),
            RewardModelConfig(
                protein_model_name_or_path="dummy/protein",
                molecule_model_name_or_path="dummy/molecule",
                activity_threshold=6.0,
            ),
        )


def test_prepare_tokenized_split_datasets_drops_rows_exceeding_token_limits(tmp_path, monkeypatch):
    split_rows = {
        "train": [
            {
                "source": "Papyrus",
                "assay_id": "A1",
                "target_id": "TARGET_A",
                "target_chembl_id": "T1",
                "protein_sequence": "MKTAA",
                "compound_id": "KEEP",
                "compound_selfies": "[N]",
                "pchembl_value": 6.0,
                "binary_label": 1,
                "assay_group_id": "IGNORED",
                "split": "train",
            },
            {
                "source": "Papyrus",
                "assay_id": "A1",
                "target_id": "TARGET_A",
                "target_chembl_id": "T1",
                "protein_sequence": "MKTAA",
                "compound_id": "DROP_MOL",
                "compound_selfies": "[C][C][C][C]",
                "pchembl_value": 4.0,
                "binary_label": 0,
                "assay_group_id": "IGNORED",
                "split": "train",
            },
            {
                "source": "Papyrus",
                "assay_id": "A2",
                "target_id": "TARGET_B",
                "target_chembl_id": "T2",
                "protein_sequence": "PROTEIN_TOO_LONG",
                "compound_id": "DROP_PROT",
                "compound_selfies": "[O]",
                "pchembl_value": 7.0,
                "binary_label": 1,
                "assay_group_id": "IGNORED",
                "split": "train",
            },
        ],
        "val": [
            {
                "source": "Papyrus",
                "assay_id": "A3",
                "target_id": "TARGET_C",
                "target_chembl_id": "T3",
                "protein_sequence": "TTTTT",
                "compound_id": "VAL_KEEP",
                "compound_selfies": "[S]",
                "pchembl_value": 4.5,
                "binary_label": 0,
                "assay_group_id": "IGNORED",
                "split": "val",
            }
        ],
        "test": [
            {
                "source": "Papyrus",
                "assay_id": "A4",
                "target_id": "TARGET_D",
                "target_chembl_id": "T4",
                "protein_sequence": "CCCCC",
                "compound_id": "TEST_KEEP",
                "compound_selfies": "[F]",
                "pchembl_value": 5.0,
                "binary_label": 0,
                "assay_group_id": "IGNORED",
                "split": "test",
            }
        ],
    }
    for split_name, rows in split_rows.items():
        _write_split_parquet(tmp_path / f"{split_name}.parquet", rows)

    monkeypatch.setattr("reward_model.training.data.load_tokenizer", lambda *_args, **_kwargs: DummyTokenizer())

    artifacts = prepare_tokenized_split_datasets(
        RewardTrainingDataConfig(
            train_parquet_path=str(tmp_path / "train.parquet"),
            val_parquet_path=str(tmp_path / "val.parquet"),
            test_parquet_path=str(tmp_path / "test.parquet"),
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
    split_paths = get_tokenized_split_dataset_paths(artifacts.base_dir)
    train_dataset = load_tokenized_example_dataset(split_paths["train"])

    assert len(train_dataset) == 1
    assert artifacts.train_examples == 1
    assert artifacts.train_groups == 1
    assert train_dataset["compound_id"] == ["KEEP"]
    assert train_dataset["example_id"] == [0]
    assert len(train_dataset[0]["protein_input_ids"]) == 6
    assert len(train_dataset[0]["molecule_input_ids"]) == 8


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

    t2_pairs = [record for record in pair_records if record["group_id"] == "T2__A2"]
    assert len(t2_pairs) == 1
    assert t2_pairs[0]["positive_example_id"] == 4
    assert t2_pairs[0]["negative_example_id"] == 3

    assert not any(record["group_id"] == "T4__A4" for record in pair_records)


def test_save_pair_dataset_from_example_dataset_writes_saved_pairs_with_expected_schema(tmp_path):
    dataset = Dataset.from_list(_tokenized_example_rows())
    pair_artifacts = save_pair_dataset_from_example_dataset(
        dataset,
        str(tmp_path / "train_pairs"),
        split_name="train",
    )
    saved_pair_dataset = load_saved_pair_dataset(pair_artifacts.dataset_path)

    assert list(saved_pair_dataset.column_names) == list(PAIR_DATASET_COLUMNS)
    assert pair_artifacts.num_groups == 4
    assert pair_artifacts.num_groups_with_pairs == 2
    assert pair_artifacts.num_pairs == 4
    assert len(saved_pair_dataset) == 4

    first_pair = dict(saved_pair_dataset[0])
    assert set(first_pair) == set(PAIR_DATASET_COLUMNS)
    assert first_pair["group_id"] in {"T1__A1", "T2__A2"}
    assert first_pair["positive_pchembl"] > first_pair["negative_pchembl"]
    assert is_valid_negative_for_positive(
        first_pair["positive_pchembl"],
        first_pair["negative_pchembl"],
    )


def test_saved_pair_dataset_paths_are_derived_from_tokenized_dataset_dir():
    pair_paths = get_saved_pair_dataset_paths("/tmp/reward_pairs")
    assert pair_paths["train"].endswith("/train_pairs")
    assert pair_paths["val"].endswith("/val_pairs")
    assert pair_paths["test"].endswith("/test_pairs")


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


def test_reward_pair_collator_flattens_pairs_dynamically_and_preserves_duplicate_rows():
    dataset = Dataset.from_list(_tokenized_example_rows())
    pair_dataset = Dataset.from_list([
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
    ])
    pair_dataset = RewardPairDataset(dataset, pair_dataset)
    collator = RewardPairCollator()

    batch = collator([pair_dataset[0], pair_dataset[1]])

    assert batch["protein_input_ids"].shape == (4, 3)
    assert batch["molecule_input_ids"].shape == (4, 2)
    assert batch["activity_labels"].shape == (4,)
    assert batch["positive_indices"].tolist() == [0, 1]
    assert batch["negative_indices"].tolist() == [2, 3]
    assert int(batch["num_pairs"].item()) == 2
    assert batch["group_ids"] == ["T1__A1", "T1__A1"]
    assert torch.equal(batch["protein_input_ids"][0], batch["protein_input_ids"][1])
    assert batch["pchembl_values"].tolist() == [7.0, 7.0, 4.0, 6.0]
    assert torch.equal(
        batch["activity_labels"],
        torch.tensor([1.0, 1.0, 0.0, 1.0]),
    )


def test_reward_pair_collator_can_preserve_legacy_fixed_width_batches():
    dataset = Dataset.from_list(_tokenized_example_rows())
    pair_records = Dataset.from_list(
        [
            {
                "positive_example_id": 2,
                "negative_example_id": 0,
                "group_id": "T1__A1",
                "positive_pchembl": 7.0,
                "negative_pchembl": 4.0,
            }
        ]
    )
    pair_dataset = RewardPairDataset(dataset, pair_records)

    batch = RewardPairCollator(dynamic_padding=False)([pair_dataset[0]])

    assert batch["protein_input_ids"].shape == (2, 4)
    assert batch["molecule_input_ids"].shape == (2, 5)


def test_reward_pair_collator_pads_variable_width_legacy_rows_before_trimming():
    rows = _tokenized_example_rows()[:2]
    rows[0]["protein_input_ids"] = rows[0]["protein_input_ids"][:3]
    rows[0]["protein_attention_mask"] = rows[0]["protein_attention_mask"][:3]
    rows[0]["molecule_input_ids"] = rows[0]["molecule_input_ids"][:2]
    rows[0]["molecule_attention_mask"] = rows[0]["molecule_attention_mask"][:2]
    dataset = Dataset.from_list(rows)
    pair_records = Dataset.from_list(
        [
            {
                "positive_example_id": 1,
                "negative_example_id": 0,
                "group_id": "T1__A1",
                "positive_pchembl": 6.0,
                "negative_pchembl": 4.0,
            }
        ]
    )
    pair_dataset = RewardPairDataset(dataset, pair_records)

    batch = RewardPairCollator(dynamic_padding=True)([pair_dataset[0]])

    assert batch["protein_input_ids"].shape == (2, 3)
    assert batch["molecule_input_ids"].shape == (2, 2)
    assert batch["protein_attention_mask"].all()
    assert batch["molecule_attention_mask"].all()


def test_reward_pair_dataset_derives_bucket_lengths_for_old_and_new_datasets():
    rows = _tokenized_example_rows()[:3]
    old_examples = Dataset.from_list(rows)
    new_examples = Dataset.from_list(
        [
            {
                **row,
                "protein_length": sum(row["protein_attention_mask"]),
                "molecule_length": sum(row["molecule_attention_mask"]),
            }
            for row in rows
        ]
    )
    pair_records = Dataset.from_list(
        [
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
    )

    old_lengths = RewardPairDataset(old_examples, pair_records).get_pair_sequence_lengths()
    new_lengths = RewardPairDataset(new_examples, pair_records).get_pair_sequence_lengths()

    assert list(old_lengths) == [5, 5]
    assert list(new_lengths) == list(old_lengths)
