#!/usr/bin/env python3
"""Build the ChEMBL 37 dataset used by the Prot2Mol generator.

The source dataset is the curated ChEMBL binding table.  Only strict positives
(``pchembl_value > threshold``) are retained.  Whole MMseqs50 protein clusters
are assigned to train or validation so homologous proteins cannot cross the
split boundary.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "assay_id",
    "protein_sequence",
    "compound_selfies",
    "smiles",
    "pchembl_value",
}


@dataclass(frozen=True)
class GenerationSplitConfig:
    validation_fraction: float = 0.05
    pchembl_threshold: float = 6.0
    seed: int = 42
    assignment_trials: int = 1_000
    max_protein_residues: int = 1_022
    max_selfies_tokens: int = 254
    deduplicate_protein_molecule_pairs: bool = True

    def validate(self) -> None:
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.assignment_trials < 1:
            raise ValueError("assignment_trials must be positive")
        if self.max_protein_residues < 1 or self.max_selfies_tokens < 1:
            raise ValueError("sequence length limits must be positive")


def choose_validation_clusters(
    cluster_counts: pd.Series,
    validation_fraction: float,
    seed: int,
    trials: int,
) -> set[str]:
    """Choose a diverse, deterministic whole-cluster subset near the row target.

    Each trial walks clusters in a seeded random order until the requested row
    target is reached.  The best candidate minimizes row error first and cluster
    count error second.  Unlike row-level splitting this cannot leak MMseqs50
    clusters, and unlike a largest-first subset it does not collapse validation
    onto a few large protein families.
    """

    counts = cluster_counts.astype("int64").sort_index()
    if len(counts) < 2:
        raise ValueError("At least two protein clusters are required")

    cluster_ids = counts.index.astype(str).to_numpy()
    row_counts = counts.to_numpy()
    target_rows = int(round(int(row_counts.sum()) * validation_fraction))
    target_clusters = max(1, int(round(len(cluster_ids) * validation_fraction)))
    rng = np.random.default_rng(seed)
    best_key = None
    best_indices = None

    for _ in range(trials):
        order = rng.permutation(len(cluster_ids))
        cumulative = np.cumsum(row_counts[order])
        boundary = int(np.searchsorted(cumulative, target_rows, side="left"))
        candidate_sizes = {
            max(1, min(len(order) - 1, boundary)),
            max(1, min(len(order) - 1, boundary + 1)),
        }
        for size in candidate_sizes:
            selected = order[:size]
            selected_rows = int(row_counts[selected].sum())
            largest_share = float(row_counts[selected].max() / max(selected_rows, 1))
            key = (
                abs(selected_rows - target_rows),
                abs(size - target_clusters),
                largest_share,
                tuple(sorted(cluster_ids[selected])),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_indices = selected.copy()

    if best_indices is None:
        raise RuntimeError("Could not assign validation protein clusters")
    return set(cluster_ids[best_indices])


def _split_summary(frame: pd.DataFrame, cluster_column: str) -> dict[str, object]:
    return {
        "rows": int(len(frame)),
        "row_fraction": None,
        "protein_sequences": int(frame["protein_sequence"].nunique()),
        "protein_clusters": int(frame[cluster_column].nunique()),
        "compounds": int(frame["compound_id"].nunique()) if "compound_id" in frame else None,
        "canonical_smiles": int(frame["smiles"].nunique()),
        "pchembl_min": float(frame["pchembl_value"].min()),
        "pchembl_median": float(frame["pchembl_value"].median()),
        "pchembl_max": float(frame["pchembl_value"].max()),
    }


def build_chembl_generation_dataset(
    source_parquet: Path,
    cluster_map_path: Path,
    output_dir: Path,
    config: GenerationSplitConfig | None = None,
) -> dict[str, object]:
    """Filter, split, validate, and write the generator train/validation data."""

    started = perf_counter()
    config = config or GenerationSplitConfig()
    config.validate()

    frame = pd.read_parquet(source_parquet)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"ChEMBL source is missing required columns: {missing}")

    source_rows = len(frame)
    frame["pchembl_value"] = pd.to_numeric(frame["pchembl_value"], errors="coerce")
    strict_positive = frame["pchembl_value"].gt(config.pchembl_threshold)
    valid_sequence = (
        frame["protein_sequence"].notna()
        & frame["protein_sequence"].fillna("").astype(str).str.strip().str.len().gt(0)
    )
    valid_selfies = (
        frame["compound_selfies"].notna()
        & frame["compound_selfies"].fillna("").astype(str).str.strip().str.len().gt(0)
    )
    valid_smiles = (
        frame["smiles"].notna()
        & frame["smiles"].fillna("").astype(str).str.strip().str.len().gt(0)
    )
    protein_within_context = (
        frame["protein_sequence"].fillna("").astype(str).str.len()
        <= config.max_protein_residues
    )
    selfies_token_count = frame["compound_selfies"].fillna("").astype(str).str.count(r"\[")
    molecule_within_context = selfies_token_count.le(config.max_selfies_tokens)
    frame = frame.loc[
        strict_positive
        & valid_sequence
        & valid_selfies
        & valid_smiles
        & protein_within_context
        & molecule_within_context
    ].copy()
    if frame.empty:
        raise ValueError("No rows remain after applying the strict pChEMBL filter")

    retained_before_deduplication = len(frame)
    if config.deduplicate_protein_molecule_pairs:
        pair_columns = ["protein_sequence", "smiles"]
        grouped = frame.groupby(pair_columns, sort=False, dropna=False)
        frame["source_positive_row_count"] = grouped["pchembl_value"].transform("size")
        frame["source_assay_count"] = grouped["assay_id"].transform("nunique")
        frame["pchembl_value"] = grouped["pchembl_value"].transform("median")
        deterministic_columns = pair_columns + [
            column
            for column in ("target_chembl_id", "compound_id", "assay_id")
            if column in frame
        ]
        frame = (
            frame.sort_values(deterministic_columns, kind="stable")
            .drop_duplicates(pair_columns, keep="first")
            .reset_index(drop=True)
        )
        if frame.duplicated(pair_columns).any():
            raise RuntimeError("Protein-molecule pair deduplication failed")

    cluster_map = pd.read_csv(cluster_map_path)
    cluster_column = "protein_cluster_50"
    required_cluster_columns = {"protein_sequence", cluster_column}
    missing_cluster = sorted(required_cluster_columns.difference(cluster_map.columns))
    if missing_cluster:
        raise ValueError(f"Cluster map is missing required columns: {missing_cluster}")
    if cluster_map["protein_sequence"].duplicated().any():
        raise ValueError("Cluster map contains duplicate protein sequences")
    cluster_map[cluster_column] = cluster_map[cluster_column].astype(str)

    frame = frame.merge(
        cluster_map[["protein_sequence", cluster_column]],
        on="protein_sequence",
        how="left",
        validate="many_to_one",
    )
    missing_assignments = int(frame[cluster_column].isna().sum())
    if missing_assignments:
        raise ValueError(f"{missing_assignments:,} retained rows lack an MMseqs50 cluster")

    counts = frame.groupby(cluster_column, sort=True).size()
    validation_clusters = choose_validation_clusters(
        counts,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
        trials=config.assignment_trials,
    )
    frame["split"] = np.where(
        frame[cluster_column].isin(validation_clusters),
        "validation",
        "train",
    )
    train = frame.loc[frame["split"].eq("train")].reset_index(drop=True)
    validation = frame.loc[frame["split"].eq("validation")].reset_index(drop=True)
    if train.empty or validation.empty:
        raise RuntimeError("Cluster assignment produced an empty split")

    train_clusters = set(train[cluster_column])
    validation_clusters_seen = set(validation[cluster_column])
    cluster_overlap = train_clusters & validation_clusters_seen
    sequence_overlap = set(train["protein_sequence"]) & set(validation["protein_sequence"])
    if cluster_overlap or sequence_overlap:
        raise RuntimeError("MMseqs50 validation failed: protein leakage was detected")
    if not train["pchembl_value"].gt(config.pchembl_threshold).all():
        raise RuntimeError("Train split violates the strict pChEMBL threshold")
    if not validation["pchembl_value"].gt(config.pchembl_threshold).all():
        raise RuntimeError("Validation split violates the strict pChEMBL threshold")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.parquet"
    validation_path = output_dir / "validation.parquet"
    cluster_split_path = output_dir / "cluster_split.csv"
    summary_path = output_dir / "summary.json"

    train.to_parquet(train_path, index=False)
    validation.to_parquet(validation_path, index=False)
    (
        frame.groupby([cluster_column, "split"], as_index=False)
        .agg(row_count=("protein_sequence", "size"), protein_count=("protein_sequence", "nunique"))
        .sort_values(["split", cluster_column])
        .to_csv(cluster_split_path, index=False)
    )

    split_stats = {
        "train": _split_summary(train, cluster_column),
        "validation": _split_summary(validation, cluster_column),
    }
    for stats in split_stats.values():
        stats["row_fraction"] = stats["rows"] / len(frame)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_name": "chembl_37_mmseqs50_generation",
        "task": "protein_conditioned_positive_molecule_generation",
        "inputs": {
            "source_parquet": str(source_parquet),
            "cluster_map": str(cluster_map_path),
        },
        "config": {
            **asdict(config),
            "pchembl_operator": ">",
            "cluster_column": cluster_column,
            "cluster_identity_percent": 50,
        },
        "counts": {
            "source_rows": int(source_rows),
            "retained_rows": int(len(frame)),
            "retained_rows_before_pair_deduplication": int(retained_before_deduplication),
            "duplicate_protein_molecule_rows_removed": int(
                retained_before_deduplication - len(frame)
            ),
            "rows_removed_by_filters": int(source_rows - retained_before_deduplication),
            "rows_removed_total_including_pair_deduplication": int(source_rows - len(frame)),
            "filter_failures_nonexclusive": {
                "not_strictly_above_pchembl_threshold": int((~strict_positive).sum()),
                "missing_protein_sequence": int((~valid_sequence).sum()),
                "missing_compound_selfies": int((~valid_selfies).sum()),
                "missing_smiles": int((~valid_smiles).sum()),
                "protein_exceeds_context": int((strict_positive & ~protein_within_context).sum()),
                "selfies_exceeds_context": int((strict_positive & ~molecule_within_context).sum()),
            },
            "splits": split_stats,
        },
        "validation": {
            "cluster_overlap": len(cluster_overlap),
            "protein_sequence_overlap": len(sequence_overlap),
            "missing_cluster_assignments": missing_assignments,
            "all_pchembl_values_strictly_above_threshold": True,
            "duplicate_protein_molecule_pairs": int(
                frame.duplicated(["protein_sequence", "smiles"]).sum()
            ),
            "test_split_written": False,
        },
        "outputs": {
            "train": str(train_path),
            "validation": str(validation_path),
            "cluster_split": str(cluster_split_path),
        },
        "runtime_seconds": round(perf_counter() - started, 2),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build strict-positive ChEMBL37 train/validation data for Prot2Mol."
    )
    parser.add_argument(
        "--source-parquet",
        type=Path,
        default=Path("dataset/processed/chembl_37/protein_cluster_50/all.parquet"),
    )
    parser.add_argument(
        "--cluster-map",
        type=Path,
        default=Path("dataset/processed/chembl_37/protein_cluster_50/protein_cluster_50.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/processed/chembl_37/protein_cluster_50_generation"),
    )
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--pchembl-threshold", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--assignment-trials", type=int, default=1_000)
    parser.add_argument("--max-protein-residues", type=int, default=1_022)
    parser.add_argument("--max-selfies-tokens", type=int, default=254)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_chembl_generation_dataset(
        source_parquet=args.source_parquet,
        cluster_map_path=args.cluster_map,
        output_dir=args.output_dir,
        config=GenerationSplitConfig(
            validation_fraction=args.validation_fraction,
            pchembl_threshold=args.pchembl_threshold,
            seed=args.seed,
            assignment_trials=args.assignment_trials,
            max_protein_residues=args.max_protein_residues,
            max_selfies_tokens=args.max_selfies_tokens,
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
