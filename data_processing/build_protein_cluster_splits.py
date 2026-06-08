#!/usr/bin/env python3
"""Build Boltz-style protein-cluster train/val/test splits.

This script keeps protein clusters as the primary split unit. Assays are kept
together only when they do not span multiple protein clusters; spanning assays
are reported in the summary because cluster leakage control takes priority.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42
    min_seq_id: float = 0.9
    coverage: float = 0.01
    cov_mode: int = 0
    cluster_mode: str = "mmseqs"
    chunksize: int = 500_000
    ranking_min_delta: float = 0.5
    ranking_max_pairs_per_group: int = 200
    ranking_max_pairs_per_split: int = 500_000
    ranking_max_train_pairs: int | None = None
    ranking_max_val_pairs: int | None = None
    ranking_max_test_pairs: int | None = None
    dedupe_ranking_pair_keys: bool = True


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def normalize_string(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def normalize_sequence(value) -> str:
    return "".join(normalize_string(value).split()).upper()


def stable_hash(value: str, length: int = 16) -> str:
    return hashlib.sha1(normalize_string(value).encode("utf-8")).hexdigest()[:length]


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if path.suffix == ".tsv":
        return pd.read_csv(path, sep="\t", **kwargs)
    return pd.read_csv(path, **kwargs)


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    elif path.suffix == ".tsv":
        frame.to_csv(path, sep="\t", index=False)
    else:
        frame.to_csv(path, index=False)


def collect_sequences(paths: list[Path], chunksize: int) -> pd.DataFrame:
    sequences: set[str] = set()
    for path in paths:
        log(f"collect protein sequences: {path}")
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path, columns=["protein_sequence"])
            sequences.update(frame["protein_sequence"].map(normalize_sequence))
        else:
            for chunk in pd.read_csv(path, usecols=["protein_sequence"], chunksize=chunksize):
                sequences.update(chunk["protein_sequence"].map(normalize_sequence))
    sequences.discard("")
    rows = [
        {"protein_id": f"seq_{idx:06d}", "protein_sequence": sequence}
        for idx, sequence in enumerate(sorted(sequences), start=1)
    ]
    return pd.DataFrame(rows)


def write_fasta(sequence_frame: pd.DataFrame, fasta_path: Path) -> None:
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with fasta_path.open("w") as handle:
        for row in sequence_frame.itertuples(index=False):
            handle.write(f">{row.protein_id}\n")
            sequence = row.protein_sequence
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def run_mmseqs_cluster(
    fasta_path: Path,
    output_prefix: Path,
    tmp_dir: Path,
    config: SplitConfig,
) -> Path:
    if shutil.which("mmseqs") is None:
        raise RuntimeError("mmseqs not found. Install mmseqs or use --cluster-mode exact for debug only.")
    cluster_tsv = output_prefix.with_name(output_prefix.name + "_cluster.tsv")
    if cluster_tsv.exists():
        log(f"reuse existing MMseqs clusters: {cluster_tsv}")
        return cluster_tsv
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "mmseqs",
        "easy-cluster",
        str(fasta_path),
        str(output_prefix),
        str(tmp_dir),
        "--min-seq-id",
        str(config.min_seq_id),
        "--cov-mode",
        str(config.cov_mode),
        "-c",
        str(config.coverage),
    ]
    log("run: " + " ".join(command))
    subprocess.run(command, check=True)
    if not cluster_tsv.exists():
        raise FileNotFoundError(f"MMseqs cluster output not found: {cluster_tsv}")
    return cluster_tsv


def build_cluster_map(sequence_frame: pd.DataFrame, cluster_tsv: Path | None) -> pd.DataFrame:
    if cluster_tsv is None:
        out = sequence_frame.copy()
        out["protein_cluster_90"] = out["protein_id"]
        return out

    cluster_pairs = pd.read_csv(cluster_tsv, sep="\t", header=None, names=["representative_id", "protein_id"])
    out = sequence_frame.merge(cluster_pairs, on="protein_id", how="left")
    missing = out["representative_id"].isna()
    out.loc[missing, "representative_id"] = out.loc[missing, "protein_id"]
    out["protein_cluster_90"] = "cluster_" + out["representative_id"].map(normalize_string)
    return out[["protein_id", "protein_sequence", "protein_cluster_90"]]


def load_binary_split_reference(path: Path, cluster_map: dict[str, str]) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["protein_sequence", "binary_label"])
    frame["protein_sequence"] = frame["protein_sequence"].map(normalize_sequence)
    frame["protein_cluster_90"] = frame["protein_sequence"].map(cluster_map)
    frame = frame.loc[frame["protein_cluster_90"].notna()].copy()
    frame["binary_label"] = pd.to_numeric(frame["binary_label"], errors="coerce").fillna(0).astype(int)
    stats = (
        frame.groupby("protein_cluster_90")["binary_label"]
        .agg(row_count="size", pos_count="sum")
        .reset_index()
    )
    stats["neg_count"] = stats["row_count"] - stats["pos_count"]
    return stats


def assign_cluster_splits(cluster_stats: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    stats = cluster_stats.copy()
    rng = np.random.RandomState(config.random_seed)
    stats["_rand"] = rng.rand(len(stats))
    stats = stats.sort_values(["row_count", "_rand"], ascending=[False, True]).reset_index(drop=True)

    total_rows = int(stats["row_count"].sum())
    total_pos = int(stats["pos_count"].sum())
    total_neg = int(stats["neg_count"].sum())
    split_ratios = {
        "train": max(0.0, 1.0 - config.val_ratio - config.test_ratio),
        "val": config.val_ratio,
        "test": config.test_ratio,
    }
    targets = {
        split: {
            "row_count": total_rows * ratio,
            "pos_count": total_pos * ratio,
            "neg_count": total_neg * ratio,
        }
        for split, ratio in split_ratios.items()
    }
    state = {
        split: {"row_count": 0.0, "pos_count": 0.0, "neg_count": 0.0}
        for split in split_ratios
    }
    weights = {"row_count": 1.0, "pos_count": 4.0, "neg_count": 1.0}

    def score_split(split: str, row) -> float:
        score = 0.0
        for metric, weight in weights.items():
            target = max(targets[split][metric], 1.0)
            value = state[split][metric] + float(getattr(row, metric))
            score += weight * ((value - target) / target) ** 2
            if value > target * 1.15 and split in {"val", "test"}:
                score += weight * ((value - target * 1.15) / target) ** 2 * 10.0
        return score

    assignments = []
    for row in stats.itertuples(index=False):
        split = min(["train", "val", "test"], key=lambda candidate: score_split(candidate, row))
        assignments.append(split)
        for metric in ["row_count", "pos_count", "neg_count"]:
            state[split][metric] += float(getattr(row, metric))

    stats["split"] = assignments
    return stats.drop(columns=["_rand"])


def add_split_columns(frame: pd.DataFrame, cluster_map: dict[str, str], split_map: dict[str, str]) -> pd.DataFrame:
    out = frame.copy()
    out["protein_sequence"] = out["protein_sequence"].map(normalize_sequence)
    out["protein_cluster_90"] = out["protein_sequence"].map(cluster_map)
    out["split"] = out["protein_cluster_90"].map(split_map).fillna("train")
    return out


def write_split_parquets(
    input_path: Path,
    output_dir: Path,
    cluster_map: dict[str, str],
    split_map: dict[str, str],
) -> dict:
    log(f"write split dataset: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = add_split_columns(read_table(input_path), cluster_map, split_map)
    outputs = {}
    counts = {}
    for split, group in frame.groupby("split", sort=True):
        path = output_dir / f"{split}.parquet"
        group.to_parquet(path, index=False)
        outputs[split] = str(path)
        counts[split] = int(len(group))
    return {"outputs": outputs, "counts": counts}


def write_continuous_split_parquets(
    input_path: Path,
    output_dir: Path,
    cluster_map: dict[str, str],
    split_map: dict[str, str],
    chunksize: int,
) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    log(f"write continuous ranking split dataset: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, pq.ParquetWriter] = {}
    outputs = {split: str(output_dir / f"{split}.parquet") for split in ["train", "val", "test"]}
    for path in outputs.values():
        path_obj = Path(path)
        if path_obj.exists():
            path_obj.unlink()
    counts = {"train": 0, "val": 0, "test": 0}
    dtype = {
        "source": "string",
        "assay_id": "string",
        "target_id": "string",
        "target_chembl_id": "string",
        "protein_sequence": "string",
        "compound_id": "string",
        "smiles": "string",
        "activity_type": "string",
        "activity_qualifier": "string",
    }
    try:
        for chunk in pd.read_csv(input_path, chunksize=chunksize, dtype=dtype):
            chunk = add_split_columns(chunk, cluster_map, split_map)
            for split, group in chunk.groupby("split", sort=True):
                path = Path(outputs[split])
                table = pa.Table.from_pandas(group, preserve_index=False)
                if split not in writers:
                    writers[split] = pq.ParquetWriter(path, table.schema)
                writers[split].write_table(table)
                counts[split] += int(len(group))
    finally:
        for writer in writers.values():
            writer.close()

    return {"outputs": outputs, "counts": counts}


def assay_split_violations(path: Path, cluster_map: dict[str, str], split_map: dict[str, str]) -> dict:
    columns = ["assay_group_id", "assay_id", "protein_sequence"]
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path, columns=columns)
    else:
        frame = pd.read_csv(path, usecols=columns)
    frame = add_split_columns(frame, cluster_map, split_map)
    group_split_counts = frame.groupby("assay_group_id")["split"].nunique()
    assay_split_counts = frame.groupby("assay_id")["split"].nunique()
    return {
        "assay_group_ids": int(group_split_counts.size),
        "assay_group_ids_spanning_splits": int((group_split_counts > 1).sum()),
        "assay_ids": int(assay_split_counts.size),
        "assay_ids_spanning_splits": int((assay_split_counts > 1).sum()),
    }


def prepare_ranking_frame(input_path: Path, cluster_map: dict[str, str], split_map: dict[str, str]) -> pd.DataFrame:
    usecols = [
        "source",
        "assay_id",
        "target_id",
        "protein_sequence",
        "compound_id",
        "smiles",
        "activity_type",
        "activity_qualifier",
        "pchembl_value",
    ]
    dtype = {
        "source": "string",
        "assay_id": "string",
        "target_id": "string",
        "protein_sequence": "string",
        "compound_id": "string",
        "smiles": "string",
        "activity_type": "string",
        "activity_qualifier": "string",
    }
    frame = pd.read_csv(input_path, usecols=usecols, dtype=dtype)
    frame["activity_qualifier"] = frame["activity_qualifier"].fillna("=").map(normalize_string)
    frame = frame.loc[frame["activity_qualifier"].eq("=")].copy()
    frame["pchembl_value"] = pd.to_numeric(frame["pchembl_value"], errors="coerce")
    frame = frame.loc[frame["pchembl_value"].notna()].copy()
    frame = add_split_columns(frame, cluster_map, split_map)
    frame["target_id"] = frame["target_id"].map(normalize_string)
    missing_target = frame["target_id"].eq("")
    frame.loc[missing_target, "target_id"] = "SEQ:" + frame.loc[missing_target, "protein_sequence"].map(stable_hash)
    frame["assay_group_id"] = (
        frame["source"].map(normalize_string)
        + ":"
        + frame["assay_id"].map(normalize_string)
        + ":"
        + frame["target_id"].map(normalize_string)
    )
    frame["smiles"] = frame["smiles"].map(normalize_string)
    frame["compound_id"] = frame["compound_id"].map(normalize_string)
    frame = frame.loc[frame["smiles"].ne("")].copy()
    collapsed = (
        frame.groupby(
            [
                "split",
                "source",
                "assay_id",
                "assay_group_id",
                "activity_type",
                "protein_cluster_90",
                "protein_sequence",
                "compound_id",
                "smiles",
            ],
            dropna=False,
        )["pchembl_value"]
        .median()
        .reset_index()
    )
    return collapsed


def sample_pairs_for_group(group: pd.DataFrame, config: SplitConfig, seed_value: str) -> list[dict]:
    group = group.sort_values("pchembl_value").reset_index(drop=True)
    if len(group) < 2:
        return []
    low = group.iloc[0]
    high = group.iloc[-1]
    if high.pchembl_value - low.pchembl_value < config.ranking_min_delta:
        return []

    all_pairs = []
    n = len(group)
    for high_idx in range(n - 1, 0, -1):
        high_row = group.iloc[high_idx]
        eligible = group.iloc[:high_idx]
        eligible = eligible.loc[
            (high_row["pchembl_value"] - eligible["pchembl_value"] >= config.ranking_min_delta)
            & (eligible["smiles"] != high_row["smiles"])
        ]
        for low_row in eligible.itertuples(index=False):
            all_pairs.append((high_row, low_row))
            if len(all_pairs) > config.ranking_max_pairs_per_group * 3:
                break
        if len(all_pairs) > config.ranking_max_pairs_per_group * 3:
            break

    if len(all_pairs) > config.ranking_max_pairs_per_group:
        rng = np.random.RandomState(int(stable_hash(seed_value, 8), 16))
        indices = rng.choice(len(all_pairs), size=config.ranking_max_pairs_per_group, replace=False)
        all_pairs = [all_pairs[index] for index in indices]

    records = []
    first = group.iloc[0]
    for high_row, low_row in all_pairs:
        records.append(
            {
                "split": first["split"],
                "source": first["source"],
                "assay_id": first["assay_id"],
                "assay_group_id": first["assay_group_id"],
                "activity_type": first["activity_type"],
                "protein_cluster_90": first["protein_cluster_90"],
                "protein_sequence": first["protein_sequence"],
                "winner_compound_id": high_row["compound_id"],
                "winner_smiles": high_row["smiles"],
                "winner_pchembl_value": float(high_row["pchembl_value"]),
                "loser_compound_id": low_row.compound_id,
                "loser_smiles": low_row.smiles,
                "loser_pchembl_value": float(low_row.pchembl_value),
                "delta_pchembl": float(high_row["pchembl_value"] - low_row.pchembl_value),
            }
        )
    return records


def ranking_pair_limit(config: SplitConfig, split: str) -> int:
    specific = {
        "train": config.ranking_max_train_pairs,
        "val": config.ranking_max_val_pairs,
        "test": config.ranking_max_test_pairs,
    }.get(split)
    return int(specific if specific is not None else config.ranking_max_pairs_per_split)


def ranking_pair_key(record: dict) -> str:
    return stable_hash(
        "\0".join(
            [
                normalize_string(record["protein_sequence"]),
                normalize_string(record["winner_smiles"]),
                normalize_string(record["loser_smiles"]),
            ]
        ),
        length=24,
    )


def build_ranking_pairs(
    input_path: Path,
    output_dir: Path,
    cluster_map: dict[str, str],
    split_map: dict[str, str],
    config: SplitConfig,
) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    log("prepare exact continuous rows for ranking pairs")
    frame = prepare_ranking_frame(input_path, cluster_map, split_map)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {split: str(output_dir / f"{split}.parquet") for split in ["train", "val", "test"]}
    for path in outputs.values():
        path_obj = Path(path)
        if path_obj.exists():
            path_obj.unlink()

    columns = [
        "split",
        "source",
        "assay_id",
        "assay_group_id",
        "activity_type",
        "protein_cluster_90",
        "protein_sequence",
        "winner_compound_id",
        "winner_smiles",
        "winner_pchembl_value",
        "loser_compound_id",
        "loser_smiles",
        "loser_pchembl_value",
        "delta_pchembl",
    ]
    writers: dict[str, pq.ParquetWriter] = {}
    buffers = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}
    seen_pair_keys = {"train": set(), "val": set(), "test": set()}
    duplicate_pair_keys_skipped = {"train": 0, "val": 0, "test": 0}
    pairable_groups = 0
    capped_groups = 0
    skipped_after_split_cap = {"train": 0, "val": 0, "test": 0}

    def flush(split: str) -> None:
        if not buffers[split]:
            return
        chunk = pd.DataFrame(buffers[split], columns=columns)
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if split not in writers:
            writers[split] = pq.ParquetWriter(outputs[split], table.schema)
        writers[split].write_table(table)
        buffers[split].clear()

    group_cols = ["split", "source", "assay_group_id", "activity_type"]
    try:
        for key, group in frame.groupby(group_cols, sort=False):
            split = key[0]
            split_limit = ranking_pair_limit(config, split)
            if counts[split] >= split_limit:
                skipped_after_split_cap[split] += 1
                continue
            group_records = sample_pairs_for_group(group, config, "|".join(map(str, key)))
            if not group_records:
                continue
            pairable_groups += 1
            if len(group_records) >= config.ranking_max_pairs_per_group:
                capped_groups += 1
            for record in group_records:
                if counts[split] >= split_limit:
                    break
                if config.dedupe_ranking_pair_keys:
                    pair_key = ranking_pair_key(record)
                    if pair_key in seen_pair_keys[split]:
                        duplicate_pair_keys_skipped[split] += 1
                        continue
                    seen_pair_keys[split].add(pair_key)
                buffers[split].append(record)
                counts[split] += 1
                if len(buffers[split]) >= 100_000:
                    flush(split)
    finally:
        for split in ["train", "val", "test"]:
            flush(split)
        for writer in writers.values():
            writer.close()

    for split, path in outputs.items():
        if not Path(path).exists():
            pd.DataFrame(columns=columns).to_parquet(path, index=False)

    pair_key_overlap_checks = {}
    for split_a, split_b in [("train", "val"), ("train", "test"), ("val", "test")]:
        pair_key_overlap_checks[f"{split_a}_{split_b}_pair_key_overlap"] = int(
            len(seen_pair_keys[split_a] & seen_pair_keys[split_b])
        )

    return {
        "outputs": outputs,
        "counts": counts,
        "pairable_groups_sampled": int(pairable_groups),
        "groups_capped_by_max_pairs_per_group": int(capped_groups),
        "groups_skipped_after_split_cap": skipped_after_split_cap,
        "duplicate_pair_keys_skipped": duplicate_pair_keys_skipped,
        "pair_key_overlap_checks": pair_key_overlap_checks,
        "exact_collapsed_rows": int(len(frame)),
        "pair_key_policy": "protein_sequence + winner_smiles + loser_smiles; deduped within each split before writing",
    }


def build_splits(
    binary_all_path: Path,
    binary_screen_path: Path,
    binary_threshold_path: Path,
    ranking_affinity_path: Path,
    output_dir: Path,
    config: SplitConfig,
) -> dict:
    started = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    sequence_frame = collect_sequences(
        [binary_all_path, binary_screen_path, binary_threshold_path, ranking_affinity_path],
        config.chunksize,
    )
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

    cluster_stats = load_binary_split_reference(binary_all_path, cluster_map)
    cluster_split = assign_cluster_splits(cluster_stats, config)
    cluster_split_path = output_dir / "cluster_split.csv"
    cluster_split.to_csv(cluster_split_path, index=False)
    split_map = dict(zip(cluster_split["protein_cluster_90"], cluster_split["split"]))

    split_outputs = {
        "binary_all_source": write_split_parquets(
            binary_all_path,
            output_dir / "binary_all_source",
            cluster_map,
            split_map,
        ),
        "binary_screen_only": write_split_parquets(
            binary_screen_path,
            output_dir / "binary_screen_only",
            cluster_map,
            split_map,
        ),
        "binary_chembl_bindingdb_threshold": write_split_parquets(
            binary_threshold_path,
            output_dir / "binary_chembl_bindingdb_threshold",
            cluster_map,
            split_map,
        ),
        "ranking_affinity": write_continuous_split_parquets(
            ranking_affinity_path,
            output_dir / "ranking_affinity",
            cluster_map,
            split_map,
            config.chunksize,
        ),
    }
    ranking_pairs = build_ranking_pairs(
        ranking_affinity_path,
        output_dir / "ranking_pairs",
        cluster_map,
        split_map,
        config,
    )
    split_outputs["ranking_pairs"] = ranking_pairs

    leakage = {}
    for split_a, split_b in [("train", "val"), ("train", "test"), ("val", "test")]:
        clusters_a = set(cluster_split.loc[cluster_split["split"] == split_a, "protein_cluster_90"])
        clusters_b = set(cluster_split.loc[cluster_split["split"] == split_b, "protein_cluster_90"])
        leakage[f"{split_a}_{split_b}_cluster_overlap"] = int(len(clusters_a & clusters_b))

    assay_violations = {
        "binary_all_source": assay_split_violations(binary_all_path, cluster_map, split_map),
        "binary_screen_only": assay_split_violations(binary_screen_path, cluster_map, split_map),
        "binary_chembl_bindingdb_threshold": assay_split_violations(binary_threshold_path, cluster_map, split_map),
    }

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": asdict(config),
        "inputs": {
            "binary_all_source": str(binary_all_path),
            "binary_screen_only": str(binary_screen_path),
            "binary_chembl_bindingdb_threshold": str(binary_threshold_path),
            "ranking_affinity": str(ranking_affinity_path),
        },
        "outputs": {
            "protein_sequences": str(sequence_path),
            "protein_fasta": str(fasta_path),
            "protein_cluster_90": str(cluster_path),
            "cluster_split": str(cluster_split_path),
            "split_datasets": split_outputs,
        },
        "counts": {
            "protein_sequences": int(len(sequence_frame)),
            "protein_clusters_90": int(cluster_frame["protein_cluster_90"].nunique()),
            "cluster_split_counts": cluster_split["split"].value_counts().to_dict(),
            "cluster_row_counts": cluster_split.groupby("split")["row_count"].sum().to_dict(),
            "cluster_positive_counts": cluster_split.groupby("split")["pos_count"].sum().to_dict(),
            "cluster_negative_counts": cluster_split.groupby("split")["neg_count"].sum().to_dict(),
        },
        "leakage_checks": leakage,
        "assay_split_violations": assay_violations,
        "runtime_seconds": round(perf_counter() - started, 2),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log(f"wrote {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build protein-cluster splits for binary and ranking datasets.")
    parser.add_argument(
        "--binary-all",
        default="dataset/processed/binary_affinity/all_sources_binary/binary_sources_merged.parquet",
    )
    parser.add_argument(
        "--binary-screen",
        default="dataset/processed/binary_affinity/merged/binary_sources_merged.parquet",
    )
    parser.add_argument(
        "--binary-threshold",
        default="dataset/processed/binary_affinity/chembl_bindingdb_threshold/chembl_bindingdb_threshold_binary.parquet",
    )
    parser.add_argument(
        "--ranking-affinity",
        default="dataset/processed/general_affinity/chembl_bindingdb_general_filtered.csv",
    )
    parser.add_argument("--output-dir", default="dataset/processed/splits/protein_cluster_90")
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
    parser.add_argument(
        "--allow-ranking-pair-key-duplicates",
        action="store_true",
        help="Keep duplicate protein_sequence/winner_smiles/loser_smiles pair keys within a split.",
    )
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
    build_splits(
        binary_all_path=Path(args.binary_all),
        binary_screen_path=Path(args.binary_screen),
        binary_threshold_path=Path(args.binary_threshold),
        ranking_affinity_path=Path(args.ranking_affinity),
        output_dir=Path(args.output_dir),
        config=config,
    )


if __name__ == "__main__":
    main()
