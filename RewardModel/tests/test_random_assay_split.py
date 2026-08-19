import json

import pandas as pd
import pytest

from reward_model.training import (
    split_target_aware_random_assays,
    write_target_aware_random_assay_split,
)


def _source_frame() -> pd.DataFrame:
    rows = []
    specifications = {
        ("target-a", "AAAA"): ("a1", "a2", "a3", "a4", "a5"),
        ("target-b", "BBBB"): ("b1", "b2"),
        ("target-c", "CCCC"): ("c1",),
    }
    for (target, sequence), assays in specifications.items():
        for assay in assays:
            for compound_index in range(2):
                rows.append(
                    {
                        "target_chembl_id": target,
                        "protein_sequence": sequence,
                        "assay_group_id": f"{target}:{assay}",
                        "compound_id": f"{assay}-compound-{compound_index}",
                        "binary_label": float(compound_index % 2),
                        "activity_type": "IC50",
                        "split": "old-mmseq-split",
                    }
                )
    return pd.DataFrame(rows)


def _assay_assignments(*splits: pd.DataFrame) -> dict[str, str]:
    return {
        str(assay_id): str(split_name)
        for split in splits
        for assay_id, split_name in split[
            ["assay_group_id", "split"]
        ].drop_duplicates().itertuples(index=False)
    }


def test_target_aware_random_split_is_deterministic_and_assay_disjoint():
    source = _source_frame()
    first = split_target_aware_random_assays(
        source,
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=42,
    )
    second = split_target_aware_random_assays(
        source.sample(frac=1.0, random_state=7),
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=42,
    )
    train, validation, test, summary = first

    assert _assay_assignments(train, validation, test) == _assay_assignments(
        *second[:3]
    )
    assert summary == second[3]
    assert len(train) + len(validation) + len(test) == len(source)
    split_assays = [
        set(split["assay_group_id"])
        for split in (train, validation, test)
    ]
    assert split_assays[0].isdisjoint(split_assays[1])
    assert split_assays[0].isdisjoint(split_assays[2])
    assert split_assays[1].isdisjoint(split_assays[2])
    assert validation["target_chembl_id"].isin(train["target_chembl_id"]).all()
    assert test["target_chembl_id"].isin(train["target_chembl_id"]).all()
    assert validation["protein_sequence"].isin(train["protein_sequence"]).all()
    assert test["protein_sequence"].isin(train["protein_sequence"]).all()
    assert "target-c:c1" in split_assays[0]
    assert set(train["split"]) == {"train"}
    assert set(validation["split"]) == {"val"}
    assert set(test["split"]) == {"test"}
    assert summary["split_stats"]["val"]["assays"] == 2
    assert summary["split_stats"]["test"]["assays"] == 2
    assert all(value == 0 for value in summary["leakage_checks"].values())


@pytest.mark.parametrize(
    ("validation_fraction", "test_fraction", "message"),
    [
        (0.0, 0.1, "validation_fraction"),
        (0.1, 0.0, "test_fraction"),
        (0.5, 0.5, "must be < 1"),
    ],
)
def test_target_aware_random_split_rejects_invalid_fractions(
    validation_fraction,
    test_fraction,
    message,
):
    with pytest.raises(ValueError, match=message):
        split_target_aware_random_assays(
            _source_frame(),
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )


def test_target_aware_random_split_requires_non_anchor_holdout_assays():
    source = _source_frame().loc[
        lambda frame: frame["assay_group_id"].isin(
            ["target-b:b1", "target-c:c1"]
        )
    ]
    with pytest.raises(ValueError, match="not enough non-anchor assays"):
        split_target_aware_random_assays(source)


def test_target_aware_random_split_writer_materializes_all_outputs(tmp_path):
    source_path = tmp_path / "all.parquet"
    output_dir = tmp_path / "random_assay"
    _source_frame().to_parquet(source_path, index=False)

    summary = write_target_aware_random_assay_split(
        source_path,
        output_dir,
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=17,
    )

    assert (output_dir / "train.parquet").is_file()
    assert (output_dir / "val.parquet").is_file()
    assert (output_dir / "test.parquet").is_file()
    manifest_path = output_dir / "random_assay_split.json"
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == summary
    with pytest.raises(FileExistsError, match="already exist"):
        write_target_aware_random_assay_split(source_path, output_dir)
