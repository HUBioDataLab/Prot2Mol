import json

import pytest
from datasets import Dataset, load_from_disk

from reward_model.training.data import get_tokenized_split_dataset_paths
from reward_model.training.overfit import (
    build_overfit_dataset,
    select_overfit_examples,
)


def _tokenized_rows(group_id: str, count: int, *, start: float = 4.0):
    target_id, assay_id = group_id.split("__", maxsplit=1)
    return [
        {
            "example_id": index,
            "group_id": group_id,
            "target_chembl_id": target_id,
            "assay_id": assay_id,
            "compound_id": f"M{index:03d}",
            "pchembl_value": start + 0.05 * index,
            "binary_label": int(start + 0.05 * index >= 6.0),
            "activity_type": "Potency",
            "protein_input_ids": [1, 2, 0],
            "protein_attention_mask": [1, 1, 0],
            "protein_length": 2,
            "molecule_input_ids": [index + 1, 0, 0],
            "molecule_attention_mask": [1, 0, 0],
            "molecule_length": 1,
        }
        for index in range(count)
    ]


def test_select_overfit_examples_keeps_50_unique_untouched_rows():
    rows = _tokenized_rows("T1__A1", 60)
    duplicate = dict(rows[0])
    duplicate["example_id"] = 60
    duplicate["pchembl_value"] = 9.0
    dataset = Dataset.from_list(rows + [duplicate])

    selected, summary = select_overfit_examples(dataset, num_molecules=50)

    assert len(selected) == 50
    assert len(set(selected["compound_id"])) == 50
    assert selected["example_id"] == list(range(50))
    assert summary["group_id"] == "T1__A1"
    assert summary["source_group_rows"] == 61
    assert summary["source_group_unique_molecules"] == 60
    assert summary["pchembl_span"] >= 0.5
    assert summary["valid_ordered_pair_count"] > 0


def test_build_overfit_dataset_replicates_identical_splits(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "overfit"
    source_paths = get_tokenized_split_dataset_paths(str(source_dir))
    Dataset.from_list(_tokenized_rows("T1__A1", 55)).save_to_disk(
        source_paths["train"]
    )

    summary = build_overfit_dataset(
        source_dir=str(source_dir),
        output_dir=str(output_dir),
        num_molecules=50,
    )

    output_paths = get_tokenized_split_dataset_paths(str(output_dir))
    split_payloads = {
        split_name: load_from_disk(path).to_dict()
        for split_name, path in output_paths.items()
    }
    assert split_payloads["train"] == split_payloads["val"]
    assert split_payloads["train"] == split_payloads["test"]
    assert summary["split_rows"] == {"train": 50, "val": 50, "test": 50}
    assert len(set(summary["split_dataset_sha256"].values())) == 1
    with open(output_dir / "overfit_manifest.json", encoding="utf-8") as handle:
        assert json.load(handle) == summary

    with pytest.raises(FileExistsError, match="refusing to replace"):
        build_overfit_dataset(
            source_dir=str(source_dir),
            output_dir=str(output_dir),
            num_molecules=50,
        )


def test_select_overfit_examples_rejects_group_without_50_unique_molecules():
    dataset = Dataset.from_list(_tokenized_rows("T1__A1", 49))

    with pytest.raises(ValueError, match="Could not find 50 unique molecules"):
        select_overfit_examples(dataset, num_molecules=50)
