from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_SEEN_TARGET_COLUMNS = {
    "target_chembl_id",
    "protein_sequence",
    "assay_group_id",
}


def split_seen_target_assay_holdout(
    frame: pd.DataFrame,
    *,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Hold out whole assays while retaining each exact target sequence in train."""
    if not math.isfinite(float(validation_fraction)) or not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be finite and in (0, 1)")
    missing = sorted(REQUIRED_SEEN_TARGET_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"source train frame is missing columns: {missing}")
    if frame.empty:
        raise ValueError("source train frame must not be empty")

    assay_keys = ["target_chembl_id", "protein_sequence", "assay_group_id"]
    assays = frame[assay_keys].drop_duplicates()
    assay_identity_counts = assays.groupby("assay_group_id", sort=False).size()
    if (assay_identity_counts != 1).any():
        raise ValueError("an assay_group_id maps to multiple target sequences")

    generator = np.random.default_rng(int(seed))
    held_out_assays: set[str] = set()
    eligible_target_sequences = 0
    for _, target_assays in assays.groupby(
        ["target_chembl_id", "protein_sequence"],
        sort=True,
    ):
        assay_ids = sorted(target_assays["assay_group_id"].astype(str).tolist())
        if len(assay_ids) < 2:
            continue
        eligible_target_sequences += 1
        holdout_count = min(
            len(assay_ids) - 1,
            max(1, math.ceil(len(assay_ids) * float(validation_fraction))),
        )
        selected = generator.permutation(len(assay_ids))[:holdout_count]
        held_out_assays.update(assay_ids[int(index)] for index in selected)

    if not held_out_assays:
        raise ValueError("no target sequence has at least two assays to split")

    validation_mask = frame["assay_group_id"].astype(str).isin(held_out_assays)
    train = frame.loc[~validation_mask].copy()
    val2 = frame.loc[validation_mask].copy()
    if "split" in train.columns:
        train.loc[:, "split"] = "train"
        val2.loc[:, "split"] = "val2"

    train_assays = set(train["assay_group_id"].astype(str))
    val2_assays = set(val2["assay_group_id"].astype(str))
    train_targets = set(train["target_chembl_id"].astype(str))
    val2_targets = set(val2["target_chembl_id"].astype(str))
    train_sequences = set(train["protein_sequence"].astype(str))
    val2_sequences = set(val2["protein_sequence"].astype(str))
    if train_assays & val2_assays:
        raise RuntimeError("seen-target split contains cross-split assay overlap")
    if not val2_targets.issubset(train_targets):
        raise RuntimeError("seen-target validation contains a target absent from train")
    if not val2_sequences.issubset(train_sequences):
        raise RuntimeError("seen-target validation contains a protein sequence absent from train")
    if len(train) + len(val2) != len(frame):
        raise RuntimeError("seen-target split changed source row cardinality")

    summary: dict[str, Any] = {
        "seed": int(seed),
        "requested_validation_assay_fraction": float(validation_fraction),
        "source_rows": int(len(frame)),
        "train_rows": int(len(train)),
        "val2_rows": int(len(val2)),
        "source_assays": int(frame["assay_group_id"].nunique()),
        "train_assays": int(len(train_assays)),
        "val2_assays": int(len(val2_assays)),
        "eligible_target_sequences": int(eligible_target_sequences),
        "val2_targets": int(len(val2_targets)),
        "val2_protein_sequences": int(len(val2_sequences)),
        "assay_overlap": 0,
        "val2_targets_absent_from_train": 0,
        "val2_protein_sequences_absent_from_train": 0,
    }
    if "protein_cluster_50" in frame.columns:
        train_clusters = set(train["protein_cluster_50"].astype(str))
        val2_clusters = set(val2["protein_cluster_50"].astype(str))
        absent_clusters = val2_clusters.difference(train_clusters)
        if absent_clusters:
            raise RuntimeError(
                "seen-target validation contains a protein cluster absent from train"
            )
        summary.update(
            {
                "val2_protein_clusters": int(len(val2_clusters)),
                "val2_protein_clusters_absent_from_train": 0,
            }
        )
    return train, val2, summary


def write_seen_target_assay_holdout(
    source_train_path: str | Path,
    output_dir: str | Path,
    *,
    validation_fraction: float = 0.1,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_path = Path(source_train_path).resolve()
    destination = Path(output_dir).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source train parquet not found: {source_path}")
    train_path = destination / "train.parquet"
    val2_path = destination / "val2.parquet"
    manifest_path = destination / "seen_target_split.json"
    existing = [path for path in (train_path, val2_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "seen-target outputs already exist; pass overwrite=True to replace: "
            + ", ".join(str(path) for path in existing)
        )

    source = pd.read_parquet(source_path)
    train, val2, summary = split_seen_target_assay_holdout(
        source,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    destination.mkdir(parents=True, exist_ok=True)
    temporary_train = destination / "train.parquet.tmp"
    temporary_val2 = destination / "val2.parquet.tmp"
    temporary_manifest = destination / "seen_target_split.json.tmp"
    train.to_parquet(temporary_train, index=False)
    val2.to_parquet(temporary_val2, index=False)
    summary.update(
        {
            "source_train_parquet": str(source_path),
            "train_parquet": str(train_path),
            "val2_parquet": str(val2_path),
        }
    )
    temporary_manifest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary_train.replace(train_path)
    temporary_val2.replace(val2_path)
    temporary_manifest.replace(manifest_path)
    return summary
