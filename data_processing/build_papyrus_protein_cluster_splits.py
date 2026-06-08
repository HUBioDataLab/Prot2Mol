#!/usr/bin/env python3
"""Build protein-cluster splits for the prepared Papyrus dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd
import numpy as np

from build_protein_cluster_splits import (
    SplitConfig,
    add_split_columns,
    assign_cluster_splits,
    build_cluster_map,
    build_ranking_pairs,
    collect_sequences,
    load_binary_split_reference,
    log,
    run_mmseqs_cluster,
    write_continuous_split_parquets,
    write_fasta,
    write_split_parquets,
)


def split_leakage(cluster_split: pd.DataFrame) -> dict:
    leakage = {}
    for split_a, split_b in [("train", "val"), ("train", "test"), ("val", "test")]:
        clusters_a = set(cluster_split.loc[cluster_split["split"] == split_a, "protein_cluster_90"])
        clusters_b = set(cluster_split.loc[cluster_split["split"] == split_b, "protein_cluster_90"])
        leakage[f"{split_a}_{split_b}_cluster_overlap"] = int(len(clusters_a & clusters_b))
    return leakage


def summarize_split_file(path: Path, label_column: str | None = None) -> dict:
    frame = pd.read_parquet(path)
    cluster_columns = [column for column in ["protein_cluster_50", "protein_cluster_90"] if column in frame]
    out = {
        "rows": int(len(frame)),
        "proteins": int(frame["protein_sequence"].nunique()) if "protein_sequence" in frame else 0,
        "smiles": int(frame["smiles"].nunique()) if "smiles" in frame else 0,
    }
    if cluster_columns:
        out["clusters"] = int(frame[cluster_columns[0]].nunique())
    if label_column and label_column in frame:
        out["label_counts"] = {
            str(k): int(v) for k, v in frame[label_column].value_counts().sort_index().to_dict().items()
        }
    if "delta_pchembl" in frame and len(frame):
        out["delta_min"] = float(frame["delta_pchembl"].min())
        out["delta_median"] = float(frame["delta_pchembl"].median())
    return out


def cluster_column_name(config: SplitConfig) -> str:
    return f"protein_cluster_{int(round(config.min_seq_id * 100))}"


def rename_cluster_column_in_parquet(path: Path, cluster_column: str) -> None:
    if cluster_column == "protein_cluster_90" or not path.exists():
        return
    frame = pd.read_parquet(path)
    if "protein_cluster_90" in frame.columns and cluster_column not in frame.columns:
        frame = frame.rename(columns={"protein_cluster_90": cluster_column})
        frame.to_parquet(path, index=False)


def rename_cluster_outputs(
    output_dir: Path,
    cluster_path: Path,
    cluster_split_path: Path,
    split_outputs: dict,
    config: SplitConfig,
) -> tuple[Path, Path]:
    cluster_column = cluster_column_name(config)
    if cluster_column == "protein_cluster_90":
        return cluster_path, cluster_split_path

    new_cluster_path = output_dir / f"{cluster_column}.csv"
    new_cluster_split_path = output_dir / "cluster_split.csv"

    cluster_frame = pd.read_csv(cluster_path)
    if "protein_cluster_90" in cluster_frame.columns:
        cluster_frame = cluster_frame.rename(columns={"protein_cluster_90": cluster_column})
    cluster_frame.to_csv(new_cluster_path, index=False)
    if cluster_path != new_cluster_path and cluster_path.exists():
        cluster_path.unlink()

    split_frame = pd.read_csv(cluster_split_path)
    if "protein_cluster_90" in split_frame.columns:
        split_frame = split_frame.rename(columns={"protein_cluster_90": cluster_column})
    split_frame.to_csv(new_cluster_split_path, index=False)

    for result in split_outputs.values():
        for split_path in result.get("outputs", {}).values():
            rename_cluster_column_in_parquet(Path(split_path), cluster_column)

    return new_cluster_path, new_cluster_split_path


def _split_objective(
    val_sums: tuple[float, float, float],
    test_sums: tuple[float, float, float],
    target_rows: float,
    target_pos: float,
    target_neg: float,
) -> float:
    score = 0.0
    for row_count, pos_count, neg_count in [val_sums, test_sums]:
        row_target = max(target_rows, 1.0)
        pos_target = max(target_pos, 1.0)
        neg_target = max(target_neg, 1.0)
        split_score = (
            ((row_count - row_target) / row_target) ** 2
            + 10.0 * ((pos_count - pos_target) / pos_target) ** 2
            + 2.0 * ((neg_count - neg_target) / neg_target) ** 2
        )
        if row_count > row_target * 1.25:
            split_score += ((row_count - row_target * 1.25) / row_target) ** 2 * 10.0
        if row_count < row_target * 0.75:
            split_score += ((row_target * 0.75 - row_count) / row_target) ** 2 * 10.0
        score += split_score
    return float(score)


def assign_cluster_splits_balanced(
    cluster_stats: pd.DataFrame,
    config: SplitConfig,
    trials: int = 2000,
) -> pd.DataFrame:
    """Assign val/test by optimizing row and label balance, train gets the rest."""
    stats = cluster_stats.copy().reset_index(drop=True)
    rows = stats["row_count"].to_numpy(dtype=float)
    pos = stats["pos_count"].to_numpy(dtype=float)
    neg = stats["neg_count"].to_numpy(dtype=float)
    target_rows = rows.sum() * config.val_ratio
    target_pos = pos.sum() * config.val_ratio
    target_neg = neg.sum() * config.val_ratio
    rng = np.random.default_rng(config.random_seed)

    def evaluate(assignments: np.ndarray) -> tuple[float, tuple[float, float, float], tuple[float, float, float]]:
        val_mask = assignments == 1
        test_mask = assignments == 2
        val_sums = (rows[val_mask].sum(), pos[val_mask].sum(), neg[val_mask].sum())
        test_sums = (rows[test_mask].sum(), pos[test_mask].sum(), neg[test_mask].sum())
        return _split_objective(val_sums, test_sums, target_rows, target_pos, target_neg), val_sums, test_sums

    def build_once() -> np.ndarray:
        assignments = np.zeros(len(stats), dtype=np.int8)
        sums = {1: [0.0, 0.0, 0.0], 2: [0.0, 0.0, 0.0]}
        order = np.lexsort((rng.random(len(stats)), -rows))
        top_count = min(200, len(order))
        top = order[:top_count].copy()
        rng.shuffle(top)
        order = np.concatenate([top, order[top_count:]])
        for index in order:
            current = _split_objective(tuple(sums[1]), tuple(sums[2]), target_rows, target_pos, target_neg)
            best_score = current
            best_split = 0
            for split in [1, 2]:
                val_sums = [*sums[1]]
                test_sums = [*sums[2]]
                target = val_sums if split == 1 else test_sums
                target[0] += rows[index]
                target[1] += pos[index]
                target[2] += neg[index]
                score = _split_objective(tuple(val_sums), tuple(test_sums), target_rows, target_pos, target_neg)
                score *= 1.0 + rng.normal(0.0, 0.01)
                if score < best_score:
                    best_score = score
                    best_split = split
            if best_split:
                assignments[index] = best_split
                sums[best_split][0] += rows[index]
                sums[best_split][1] += pos[index]
                sums[best_split][2] += neg[index]
        return assignments

    best_score = float("inf")
    best_assignments: np.ndarray | None = None
    for _ in range(max(1, trials)):
        assignments = build_once()
        score, _, _ = evaluate(assignments)
        if score < best_score:
            best_score = score
            best_assignments = assignments
    assert best_assignments is not None

    split_names = np.where(best_assignments == 1, "val", np.where(best_assignments == 2, "test", "train"))
    stats["split"] = split_names
    stats["assignment_objective"] = best_score
    return stats


def build_papyrus_splits(
    affinity_path: Path,
    binary_path: Path,
    output_dir: Path,
    config: SplitConfig,
    include_affinity_split: bool = True,
    include_ranking_pairs: bool = True,
    assignment_mode: str = "greedy",
    balance_trials: int = 2000,
) -> dict:
    started = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    sequence_inputs = [binary_path]
    if include_affinity_split or include_ranking_pairs:
        sequence_inputs.append(affinity_path)
    sequence_frame = collect_sequences(sequence_inputs, config.chunksize)
    sequence_path = output_dir / "protein_sequences.csv"
    sequence_frame.to_csv(sequence_path, index=False)
    fasta_path = output_dir / "proteins.fasta"
    write_fasta(sequence_frame, fasta_path)

    if config.cluster_mode == "mmseqs":
        cluster_tsv = run_mmseqs_cluster(
            fasta_path=fasta_path,
            output_prefix=output_dir / "mmseqs" / "protein_seq90",
            tmp_dir=output_dir / "mmseqs" / "tmp",
            config=config,
        )
    elif config.cluster_mode == "exact":
        cluster_tsv = None
    else:
        raise ValueError(f"Unsupported cluster mode: {config.cluster_mode}")

    cluster_frame = build_cluster_map(sequence_frame, cluster_tsv)
    cluster_path = output_dir / "protein_cluster_90.csv"
    cluster_frame.to_csv(cluster_path, index=False)
    cluster_map = dict(zip(cluster_frame["protein_sequence"], cluster_frame["protein_cluster_90"]))

    cluster_stats = load_binary_split_reference(binary_path, cluster_map)
    if assignment_mode == "balanced":
        cluster_split = assign_cluster_splits_balanced(cluster_stats, config, balance_trials)
    elif assignment_mode == "greedy":
        cluster_split = assign_cluster_splits(cluster_stats, config)
    else:
        raise ValueError(f"Unsupported assignment mode: {assignment_mode}")
    cluster_split_path = output_dir / "cluster_split.csv"
    cluster_split.to_csv(cluster_split_path, index=False)
    split_map = dict(zip(cluster_split["protein_cluster_90"], cluster_split["split"]))

    split_outputs = {
        "binary_pchembl_threshold": write_split_parquets(
            binary_path,
            output_dir / "binary_pchembl_threshold",
            cluster_map,
            split_map,
        ),
    }
    if include_affinity_split:
        split_outputs["affinity"] = write_continuous_split_parquets(
            affinity_path,
            output_dir / "affinity",
            cluster_map,
            split_map,
            config.chunksize,
        )
    if include_ranking_pairs:
        split_outputs["ranking_pairs"] = build_ranking_pairs(
            affinity_path,
            output_dir / "ranking_pairs",
            cluster_map,
            split_map,
            config,
        )

    cluster_path, cluster_split_path = rename_cluster_outputs(
        output_dir=output_dir,
        cluster_path=cluster_path,
        cluster_split_path=cluster_split_path,
        split_outputs=split_outputs,
        config=config,
    )

    split_dataset_stats = {}
    for view, result in split_outputs.items():
        split_dataset_stats[view] = {}
        for split, split_path in result["outputs"].items():
            split_dataset_stats[view][split] = summarize_split_file(
                Path(split_path),
                label_column="binary_label" if view == "binary_pchembl_threshold" else None,
            )

    output_cluster_column = cluster_column_name(config)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": asdict(config),
        "cluster_identity_percent": int(round(config.min_seq_id * 100)),
        "include_affinity_split": include_affinity_split,
        "include_ranking_pairs": include_ranking_pairs,
        "assignment_mode": assignment_mode,
        "balance_trials": balance_trials if assignment_mode == "balanced" else None,
        "inputs": {
            "affinity": str(affinity_path) if include_affinity_split or include_ranking_pairs else None,
            "binary_pchembl_threshold": str(binary_path),
        },
        "outputs": {
            "protein_sequences": str(sequence_path),
            "protein_fasta": str(fasta_path),
            "protein_cluster_column": output_cluster_column,
            output_cluster_column: str(cluster_path),
            "cluster_split": str(cluster_split_path),
            "split_datasets": split_outputs,
        },
        "counts": {
            "protein_sequences": int(len(sequence_frame)),
            "protein_clusters": int(cluster_frame["protein_cluster_90"].nunique()),
            "cluster_split_counts": cluster_split["split"].value_counts().to_dict(),
            "cluster_row_counts": cluster_split.groupby("split")["row_count"].sum().to_dict(),
            "cluster_positive_counts": cluster_split.groupby("split")["pos_count"].sum().to_dict(),
            "cluster_negative_counts": cluster_split.groupby("split")["neg_count"].sum().to_dict(),
        },
        "split_dataset_stats": split_dataset_stats,
        "leakage_checks": split_leakage(cluster_split),
        "runtime_seconds": round(perf_counter() - started, 2),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"wrote {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Papyrus protein-cluster splits.")
    parser.add_argument("--affinity", default="dataset/processed/papyrus/papyrus_affinity.csv")
    parser.add_argument("--binary", default="dataset/processed/papyrus/papyrus_binary_pchembl6.parquet")
    parser.add_argument("--output-dir", default="dataset/processed/papyrus/splits/protein_cluster_90")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--cluster-mode", choices=("mmseqs", "exact"), default="mmseqs")
    parser.add_argument("--min-seq-id", type=float, default=0.9)
    parser.add_argument("--coverage", type=float, default=0.01)
    parser.add_argument("--cov-mode", type=int, default=0)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--ranking-min-delta", type=float, default=0.5)
    parser.add_argument("--ranking-max-pairs-per-group", type=int, default=200)
    parser.add_argument("--ranking-max-pairs-per-split", type=int, default=500_000)
    parser.add_argument("--ranking-max-train-pairs", type=int, default=None)
    parser.add_argument("--ranking-max-val-pairs", type=int, default=None)
    parser.add_argument("--ranking-max-test-pairs", type=int, default=None)
    parser.add_argument("--allow-ranking-pair-key-duplicates", action="store_true")
    parser.add_argument("--skip-affinity-split", action="store_true")
    parser.add_argument("--skip-ranking-pairs", action="store_true")
    parser.add_argument("--assignment-mode", choices=("greedy", "balanced"), default="greedy")
    parser.add_argument("--balance-trials", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SplitConfig(
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
        min_seq_id=args.min_seq_id,
        coverage=args.coverage,
        cov_mode=args.cov_mode,
        cluster_mode=args.cluster_mode,
        chunksize=args.chunksize,
        ranking_min_delta=args.ranking_min_delta,
        ranking_max_pairs_per_group=args.ranking_max_pairs_per_group,
        ranking_max_pairs_per_split=args.ranking_max_pairs_per_split,
        ranking_max_train_pairs=args.ranking_max_train_pairs,
        ranking_max_val_pairs=args.ranking_max_val_pairs,
        ranking_max_test_pairs=args.ranking_max_test_pairs,
        dedupe_ranking_pair_keys=not args.allow_ranking_pair_key_duplicates,
    )
    build_papyrus_splits(
        affinity_path=Path(args.affinity),
        binary_path=Path(args.binary),
        output_dir=Path(args.output_dir),
        config=config,
        include_affinity_split=not args.skip_affinity_split,
        include_ranking_pairs=not args.skip_ranking_pairs,
        assignment_mode=args.assignment_mode,
        balance_trials=args.balance_trials,
    )


if __name__ == "__main__":
    main()
