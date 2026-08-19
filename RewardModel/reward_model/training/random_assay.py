from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_RANDOM_ASSAY_COLUMNS = {
    "target_chembl_id",
    "protein_sequence",
    "assay_group_id",
}


def _validate_split_fraction(name: str, value: float) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 < resolved < 1.0:
        raise ValueError(f"{name} must be finite and in (0, 1)")
    return resolved


def _split_statistics(
    frame: pd.DataFrame,
    *,
    source_rows: int,
    source_assays: int,
) -> dict[str, Any]:
    statistics: dict[str, Any] = {
        "rows": int(len(frame)),
        "row_fraction": float(len(frame) / source_rows),
        "assays": int(frame["assay_group_id"].nunique()),
        "assay_fraction": float(
            frame["assay_group_id"].nunique() / source_assays
        ),
        "targets": int(frame["target_chembl_id"].nunique()),
        "protein_sequences": int(frame["protein_sequence"].nunique()),
    }
    if "compound_id" in frame.columns:
        statistics["molecules"] = int(frame["compound_id"].nunique())
    if "binary_label" in frame.columns:
        labels = pd.to_numeric(frame["binary_label"], errors="raise")
        statistics["positives"] = int((labels == 1.0).sum())
        statistics["negatives"] = int((labels == 0.0).sum())
    if "activity_type" in frame.columns:
        statistics["activity_type_rows"] = {
            str(key): int(value)
            for key, value in sorted(
                frame["activity_type"].astype(str).value_counts().items()
            )
        }
    return statistics


def split_target_aware_random_assays(
    frame: pd.DataFrame,
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Randomly split whole assays while retaining every target sequence in train."""
    validation_fraction = _validate_split_fraction(
        "validation_fraction",
        validation_fraction,
    )
    test_fraction = _validate_split_fraction("test_fraction", test_fraction)
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be < 1")
    missing = sorted(REQUIRED_RANDOM_ASSAY_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"source frame is missing columns: {missing}")
    if frame.empty:
        raise ValueError("source frame must not be empty")
    if frame[list(REQUIRED_RANDOM_ASSAY_COLUMNS)].isna().any().any():
        raise ValueError("target, protein sequence, and assay ids must not be null")

    assay_table = frame[
        ["target_chembl_id", "protein_sequence", "assay_group_id"]
    ].copy()
    assay_table = assay_table.astype(str).drop_duplicates()
    assay_identity_counts = assay_table.groupby(
        "assay_group_id",
        sort=False,
    ).size()
    if (assay_identity_counts != 1).any():
        raise ValueError("an assay_group_id maps to multiple target sequences")

    generator = np.random.default_rng(int(seed))
    train_anchor_assays: set[str] = set()
    for _, target_assays in assay_table.groupby(
        ["target_chembl_id", "protein_sequence"],
        sort=True,
    ):
        assay_ids = sorted(target_assays["assay_group_id"].tolist())
        anchor_index = int(generator.integers(0, len(assay_ids)))
        train_anchor_assays.add(assay_ids[anchor_index])

    all_assays = sorted(assay_table["assay_group_id"].tolist())
    eligible_holdout_assays = sorted(set(all_assays) - train_anchor_assays)
    num_assays = len(all_assays)
    validation_assay_count = max(
        1,
        int(math.floor(num_assays * validation_fraction + 0.5)),
    )
    test_assay_count = max(
        1,
        int(math.floor(num_assays * test_fraction + 0.5)),
    )
    requested_holdout_count = validation_assay_count + test_assay_count
    if requested_holdout_count > len(eligible_holdout_assays):
        raise ValueError(
            "not enough non-anchor assays to satisfy the requested validation "
            "and test fractions while retaining every target sequence in train"
        )

    holdout_order = generator.permutation(len(eligible_holdout_assays))
    validation_assays = {
        eligible_holdout_assays[int(index)]
        for index in holdout_order[:validation_assay_count]
    }
    test_assays = {
        eligible_holdout_assays[int(index)]
        for index in holdout_order[
            validation_assay_count:requested_holdout_count
        ]
    }

    assay_keys = frame["assay_group_id"].astype(str)
    validation_mask = assay_keys.isin(validation_assays)
    test_mask = assay_keys.isin(test_assays)
    train_mask = ~(validation_mask | test_mask)
    train = frame.loc[train_mask].copy().reset_index(drop=True)
    validation = frame.loc[validation_mask].copy().reset_index(drop=True)
    test = frame.loc[test_mask].copy().reset_index(drop=True)
    train.loc[:, "split"] = "train"
    validation.loc[:, "split"] = "val"
    test.loc[:, "split"] = "test"

    split_frames = {
        "train": train,
        "val": validation,
        "test": test,
    }
    split_assays = {
        name: set(split["assay_group_id"].astype(str))
        for name, split in split_frames.items()
    }
    if (
        split_assays["train"] & split_assays["val"]
        or split_assays["train"] & split_assays["test"]
        or split_assays["val"] & split_assays["test"]
    ):
        raise RuntimeError("target-aware random split contains assay overlap")
    if sum(len(split) for split in split_frames.values()) != len(frame):
        raise RuntimeError("target-aware random split changed source row cardinality")

    train_targets = set(train["target_chembl_id"].astype(str))
    train_sequences = set(train["protein_sequence"].astype(str))
    for name, split in (("validation", validation), ("test", test)):
        if not set(split["target_chembl_id"].astype(str)).issubset(train_targets):
            raise RuntimeError(f"{name} contains a target absent from train")
        if not set(split["protein_sequence"].astype(str)).issubset(
            train_sequences
        ):
            raise RuntimeError(
                f"{name} contains a protein sequence absent from train"
            )

    source_rows = len(frame)
    summary: dict[str, Any] = {
        "seed": int(seed),
        "requested_validation_assay_fraction": validation_fraction,
        "requested_test_assay_fraction": test_fraction,
        "source_rows": int(source_rows),
        "source_assays": int(num_assays),
        "source_targets": int(frame["target_chembl_id"].nunique()),
        "source_protein_sequences": int(frame["protein_sequence"].nunique()),
        "train_anchor_assays": int(len(train_anchor_assays)),
        "eligible_holdout_assays": int(len(eligible_holdout_assays)),
        "assignment": {
            "method": "seeded_random_whole_assays_with_one_train_anchor_per_target_sequence",
            "assay_count_targets": {
                "validation": int(validation_assay_count),
                "test": int(test_assay_count),
            },
        },
        "split_stats": {
            name: _split_statistics(
                split,
                source_rows=source_rows,
                source_assays=num_assays,
            )
            for name, split in split_frames.items()
        },
        "leakage_checks": {
            "train_val_assay_overlap": 0,
            "train_test_assay_overlap": 0,
            "val_test_assay_overlap": 0,
            "validation_targets_absent_from_train": 0,
            "test_targets_absent_from_train": 0,
            "validation_sequences_absent_from_train": 0,
            "test_sequences_absent_from_train": 0,
        },
    }
    if "compound_id" in frame.columns:
        split_molecules = {
            name: set(split["compound_id"].astype(str))
            for name, split in split_frames.items()
        }
        summary["overlap_diagnostics"] = {
            "train_val_molecules": int(
                len(split_molecules["train"] & split_molecules["val"])
            ),
            "train_test_molecules": int(
                len(split_molecules["train"] & split_molecules["test"])
            ),
            "val_test_molecules": int(
                len(split_molecules["val"] & split_molecules["test"])
            ),
        }
    return train, validation, test, summary


def write_target_aware_random_assay_split(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, Any]:
    resolved_source = Path(source_path).resolve()
    destination = Path(output_dir).resolve()
    if not resolved_source.is_file():
        raise FileNotFoundError(f"source parquet not found: {resolved_source}")

    output_paths = {
        "train": destination / "train.parquet",
        "val": destination / "val.parquet",
        "test": destination / "test.parquet",
        "manifest": destination / "random_assay_split.json",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "target-aware random-assay outputs already exist; pass "
            "overwrite=True to replace: "
            + ", ".join(str(path) for path in existing)
        )

    source = pd.read_parquet(resolved_source)
    train, validation, test, summary = split_target_aware_random_assays(
        source,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    destination.mkdir(parents=True, exist_ok=True)
    temporary_paths = {
        name: path.with_name(path.name + ".tmp")
        for name, path in output_paths.items()
    }
    summary.update(
        {
            "source_parquet": str(resolved_source),
            "train_parquet": str(output_paths["train"]),
            "val_parquet": str(output_paths["val"]),
            "test_parquet": str(output_paths["test"]),
        }
    )
    try:
        train.to_parquet(temporary_paths["train"], index=False)
        validation.to_parquet(temporary_paths["val"], index=False)
        test.to_parquet(temporary_paths["test"], index=False)
        temporary_paths["manifest"].write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        for name in ("train", "val", "test", "manifest"):
            temporary_paths[name].replace(output_paths[name])
    finally:
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)
    return summary
