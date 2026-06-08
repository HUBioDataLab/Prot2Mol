#!/usr/bin/env python3
"""Build a Papyrus positive-only generation dataset.

This creates a train/val split from whole MMseqs protein clusters. The default
uses the existing Papyrus 50% cluster map and targets exactly 10k validation
rows when an exact cluster subset is available.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd


def log(message: str) -> None:
    print(message, flush=True)


def choose_val_clusters(
    cluster_stats: pd.DataFrame,
    target_rows: int,
    selection_mode: str,
) -> tuple[set[str], int]:
    """Find a whole-cluster subset close to target."""
    if selection_mode == "max-diversity":
        return choose_val_clusters_max_diversity(cluster_stats.set_index("cluster")["rows"], target_rows)
    if selection_mode == "min-proteins":
        return choose_val_clusters_min_proteins(cluster_stats, target_rows)
    raise ValueError(f"Unsupported validation selection mode: {selection_mode}")


def choose_val_clusters_max_diversity(cluster_counts: pd.Series, target_rows: int) -> tuple[set[str], int]:
    """Find a whole-cluster subset close to target, preferring more clusters."""
    counts = cluster_counts.sort_values(ascending=True)
    clusters = counts.index.tolist()
    weights = counts.astype(int).tolist()

    prev_sum = [-1] * (target_rows + 1)
    prev_idx = [-1] * (target_rows + 1)
    best_cluster_count = [-1] * (target_rows + 1)
    best_cluster_count[0] = 0

    for idx, weight in enumerate(weights):
        if weight > target_rows:
            continue
        for current in range(target_rows - weight, -1, -1):
            if best_cluster_count[current] >= 0 and best_cluster_count[current] + 1 > best_cluster_count[current + weight]:
                best_cluster_count[current + weight] = best_cluster_count[current] + 1
                prev_sum[current + weight] = current
                prev_idx[current + weight] = idx

    best = target_rows
    while best > 0 and best_cluster_count[best] < 0:
        best -= 1
    if best == 0:
        raise RuntimeError(f"Could not find any non-empty validation cluster subset <= {target_rows}.")

    selected = set()
    current = best
    while current > 0:
        idx = prev_idx[current]
        selected.add(clusters[idx])
        current = prev_sum[current]
    return selected, best


def choose_val_clusters_min_proteins(cluster_stats: pd.DataFrame, target_rows: int) -> tuple[set[str], int]:
    """Find a whole-cluster subset close to target, minimizing protein count."""
    ordered = cluster_stats.sort_values(["rows", "proteins", "cluster"], ascending=[False, True, True])
    records = ordered[["cluster", "rows", "proteins"]].itertuples(index=False, name=None)

    infinity = (10**9, 10**9)
    best_cost = [infinity] * (target_rows + 1)
    prev_sum = [-1] * (target_rows + 1)
    prev_idx = [-1] * (target_rows + 1)
    best_cost[0] = (0, 0)
    rows = []

    for idx, (cluster, row_count, protein_count) in enumerate(records):
        rows.append((cluster, int(row_count), int(protein_count)))
        if row_count > target_rows:
            continue
        for current in range(target_rows - int(row_count), -1, -1):
            if best_cost[current] == infinity:
                continue
            candidate = (best_cost[current][0] + int(protein_count), best_cost[current][1] + 1)
            new_sum = current + int(row_count)
            if candidate < best_cost[new_sum]:
                best_cost[new_sum] = candidate
                prev_sum[new_sum] = current
                prev_idx[new_sum] = idx

    best = target_rows
    while best > 0 and best_cost[best] == infinity:
        best -= 1
    if best == 0:
        raise RuntimeError(f"Could not find any non-empty validation cluster subset <= {target_rows}.")

    selected = set()
    current = best
    while current > 0:
        idx = prev_idx[current]
        selected.add(rows[idx][0])
        current = prev_sum[current]
    return selected, best


def build_papyrus_gen_dataset(
    affinity_path: Path,
    cluster_path: Path,
    output_dir: Path,
    val_rows: int,
    pchembl_threshold: float,
    val_selection: str,
) -> dict:
    started = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"reading affinity: {affinity_path}")
    frame = pd.read_parquet(affinity_path)
    frame["pchembl_value"] = pd.to_numeric(frame["pchembl_value"], errors="coerce")
    frame = frame.loc[frame["pchembl_value"] >= pchembl_threshold].copy()
    frame["binary_label"] = 1

    cluster_frame = pd.read_csv(cluster_path)
    cluster_column_candidates = [column for column in ["protein_cluster_50", "protein_cluster_90"] if column in cluster_frame.columns]
    if not cluster_column_candidates:
        raise ValueError(f"Cluster file lacks protein cluster column: {cluster_path}")
    cluster_column = cluster_column_candidates[0]
    cluster_map = dict(zip(cluster_frame["protein_sequence"], cluster_frame[cluster_column]))
    frame[cluster_column] = frame["protein_sequence"].map(cluster_map)
    missing_clusters = int(frame[cluster_column].isna().sum())
    if missing_clusters:
        raise ValueError(f"{missing_clusters:,} positive rows do not have a cluster assignment.")

    cluster_stats = (
        frame.groupby(cluster_column)
        .agg(rows=("protein_sequence", "size"), proteins=("protein_sequence", "nunique"))
        .rename_axis("cluster")
        .reset_index()
    )
    val_clusters, actual_val_rows = choose_val_clusters(cluster_stats, val_rows, val_selection)
    frame["split"] = "train"
    frame.loc[frame[cluster_column].isin(val_clusters), "split"] = "val"

    train = frame.loc[frame["split"].eq("train")].copy()
    val = frame.loc[frame["split"].eq("val")].copy()
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)

    cluster_split = (
        frame.groupby([cluster_column, "split"])
        .agg(row_count=("binary_label", "size"))
        .reset_index()
    )
    cluster_split_path = output_dir / "cluster_split.csv"
    cluster_split.to_csv(cluster_split_path, index=False)

    def split_stats(split_frame: pd.DataFrame) -> dict:
        return {
            "rows": int(len(split_frame)),
            "binary_label_counts": {
                str(k): int(v) for k, v in split_frame["binary_label"].value_counts().sort_index().to_dict().items()
            },
            "protein_sequences": int(split_frame["protein_sequence"].nunique()),
            "protein_clusters": int(split_frame[cluster_column].nunique()),
            "smiles": int(split_frame["smiles"].nunique()),
            "compounds": int(split_frame["compound_id"].nunique()),
            "pchembl_min": float(split_frame["pchembl_value"].min()) if len(split_frame) else None,
            "pchembl_median": float(split_frame["pchembl_value"].median()) if len(split_frame) else None,
        }

    train_clusters = set(train[cluster_column])
    val_clusters_seen = set(val[cluster_column])
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": "papyrus_gen",
        "task": "positive_only_generation",
        "inputs": {
            "affinity": str(affinity_path),
            "cluster_map": str(cluster_path),
        },
        "config": {
            "cluster_column": cluster_column,
            "cluster_identity_percent": 50 if cluster_column.endswith("_50") else None,
            "pchembl_threshold": pchembl_threshold,
            "requested_val_rows": val_rows,
            "actual_val_rows": actual_val_rows,
            "val_selection": val_selection,
            "negative_rows_filtered_out": "all rows with pchembl_value < threshold",
        },
        "outputs": {
            "train": str(train_path),
            "val": str(val_path),
            "cluster_split": str(cluster_split_path),
        },
        "counts": {
            "total_positive_rows": int(len(frame)),
            "train": split_stats(train),
            "val": split_stats(val),
            "cluster_overlap": int(len(train_clusters & val_clusters_seen)),
        },
        "runtime_seconds": round(perf_counter() - started, 2),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"wrote {train_path} ({len(train):,} rows)")
    log(f"wrote {val_path} ({len(val):,} rows)")
    log(f"wrote {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Papyrus positive-only generation train/val split.")
    parser.add_argument("--affinity", default="dataset/processed/papyrus/papyrus_affinity.parquet")
    parser.add_argument("--cluster-map", default="dataset/processed/papyrus/splits/protein_cluster_50/protein_cluster_50.csv")
    parser.add_argument("--output-dir", default="dataset/processed/papyrus_gen/protein_cluster_50")
    parser.add_argument("--val-rows", type=int, default=10_000)
    parser.add_argument("--val-selection", choices=["min-proteins", "max-diversity"], default="min-proteins")
    parser.add_argument("--pchembl-threshold", type=float, default=6.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_papyrus_gen_dataset(
        affinity_path=Path(args.affinity),
        cluster_path=Path(args.cluster_map),
        output_dir=Path(args.output_dir),
        val_rows=args.val_rows,
        pchembl_threshold=args.pchembl_threshold,
        val_selection=args.val_selection,
    )


if __name__ == "__main__":
    main()
