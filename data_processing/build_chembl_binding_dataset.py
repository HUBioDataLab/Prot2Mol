#!/usr/bin/env python3
"""Build an assay-aware ChEMBL binding dataset with MMseqs protein splits.

The release, archive URL, and SHA-256 are pinned so the dataset can be rebuilt.
Rows are restricted to binding assays mapped to single-protein targets with a
non-null pChEMBL value. Repeated measurements for the same assay, protein, and
parent molecule are collapsed to their median pChEMBL value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sqlite3
import tarfile
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

import pandas as pd
import numpy as np

try:
    from data_processing.build_protein_cluster_splits import (
        SplitConfig,
        build_cluster_map,
        collect_sequences,
        normalize_sequence,
        run_mmseqs_cluster,
        write_fasta,
    )
except ModuleNotFoundError:  # Direct execution: python data_processing/build_chembl_binding_dataset.py
    from build_protein_cluster_splits import (
        SplitConfig,
        build_cluster_map,
        collect_sequences,
        normalize_sequence,
        run_mmseqs_cluster,
        write_fasta,
    )


CHEMBL_RELEASE = 37
CHEMBL_RELEASE_BASE_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37"
)
CHEMBL_SQLITE_ARCHIVE_URL = f"{CHEMBL_RELEASE_BASE_URL}/chembl_37_sqlite.tar.gz"
CHEMBL_SQLITE_ARCHIVE_SHA256 = (
    "33c203740555f96067710cdfc1c3c55d890660e5908ec5cbf5817492c290d281"
)
DEFAULT_ARCHIVE_PATH = Path("dataset/raw/chembl_37/chembl_37_sqlite.tar.gz")
DEFAULT_RAW_DIR = Path("dataset/raw/chembl_37")
DEFAULT_OUTPUT_DIR = Path("dataset/processed/chembl_37/protein_cluster_50")

BASE_OUTPUT_COLUMNS = [
    "source",
    "assay_id",
    "target_id",
    "target_chembl_id",
    "protein_id",
    "protein_accession",
    "protein_sequence",
    "compound_id",
    "molecule_chembl_id",
    "smiles",
    "pchembl_value",
    "protein_length",
    "compound_selfies",
    "assay_group_id",
    "binary_label",
    "activity_type",
    "measurement_count",
    "confidence_score",
    "depositor_assay_group",
    "assay_type",
    "target_type",
    "standard_relation",
    "chembl_release",
]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def verify_archive(path: Path) -> None:
    actual = sha256_file(path)
    if actual != CHEMBL_SQLITE_ARCHIVE_SHA256:
        raise ValueError(
            f"ChEMBL archive SHA-256 mismatch for {path}: "
            f"expected {CHEMBL_SQLITE_ARCHIVE_SHA256}, got {actual}"
        )


def download_archive(destination: Path) -> Path:
    """Download the pinned archive atomically; an existing file is verified."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        log(f"verify existing archive: {destination}")
        verify_archive(destination)
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        raise FileExistsError(
            f"Partial download already exists: {partial}. Resume it with curl or remove it explicitly."
        )
    log(f"download ChEMBL {CHEMBL_RELEASE}: {CHEMBL_SQLITE_ARCHIVE_URL}")
    with urllib.request.urlopen(CHEMBL_SQLITE_ARCHIVE_URL) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
    verify_archive(partial)
    partial.replace(destination)
    return destination


def extract_sqlite_archive(archive_path: Path, raw_dir: Path) -> Path:
    """Extract only the SQLite database without trusting archive paths."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile() and member.name.lower().endswith((".db", ".sqlite"))
        ]
        if len(members) != 1:
            raise ValueError(
                f"Expected exactly one SQLite database in {archive_path}, found {len(members)}"
            )
        member = members[0]
        output_path = raw_dir / Path(member.name).name
        if output_path.exists():
            log(f"reuse extracted SQLite database: {output_path}")
            return output_path
        source = archive.extractfile(member)
        if source is None:
            raise OSError(f"Could not read {member.name} from {archive_path}")
        log(f"extract {member.name} -> {output_path}")
        with source, output_path.open("xb") as destination:
            shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
    return output_path


def _table_has_column(connection: sqlite3.Connection, table: str, column: str) -> bool:
    return any(row[1] == column for row in connection.execute(f"PRAGMA table_info({table})"))


def build_activity_query(
    *,
    exact_only: bool,
    include_variants: bool,
    include_questionable: bool,
    min_confidence_score: int,
    has_variant_id: bool,
) -> tuple[str, list[object]]:
    clauses = [
        "a.assay_type = 'B'",
        "td.target_type = 'SINGLE PROTEIN'",
        "td.species_group_flag = 0",
        "UPPER(cs.component_type) = 'PROTEIN'",
        "cs.sequence IS NOT NULL",
        "TRIM(cs.sequence) != ''",
        "act.pchembl_value IS NOT NULL",
        "COALESCE(cs_parent.canonical_smiles, cs_raw.canonical_smiles) IS NOT NULL",
    ]
    parameters: list[object] = []
    if exact_only:
        clauses.append("act.standard_relation = '='")
    if not include_variants and has_variant_id:
        clauses.append("a.variant_id IS NULL")
    if not include_questionable:
        clauses.append(
            "(act.data_validity_comment IS NULL OR "
            "act.data_validity_comment = 'Manually validated')"
        )
    if min_confidence_score > 0:
        clauses.append("a.confidence_score >= ?")
        parameters.append(min_confidence_score)

    query = f"""
        SELECT
            a.chembl_id AS assay_id,
            a.assay_group AS depositor_assay_group,
            a.confidence_score AS confidence_score,
            td.chembl_id AS target_chembl_id,
            cs.accession AS protein_accession,
            cs.sequence AS protein_sequence,
            md_parent.chembl_id AS molecule_chembl_id,
            COALESCE(cs_parent.canonical_smiles, cs_raw.canonical_smiles) AS smiles,
            act.standard_type AS activity_type,
            act.standard_relation AS standard_relation,
            act.pchembl_value AS pchembl_value
        FROM activities act
        JOIN assays a ON act.assay_id = a.assay_id
        JOIN target_dictionary td ON a.tid = td.tid
        JOIN (
            SELECT tc_unique.tid, MIN(tc_unique.component_id) AS component_id
            FROM target_components tc_unique
            JOIN component_sequences cs_unique
              ON tc_unique.component_id = cs_unique.component_id
            WHERE UPPER(cs_unique.component_type) = 'PROTEIN'
            GROUP BY tc_unique.tid
            HAVING COUNT(*) = 1
        ) unique_tc ON td.tid = unique_tc.tid
        JOIN component_sequences cs ON unique_tc.component_id = cs.component_id
        JOIN molecule_dictionary md_raw ON act.molregno = md_raw.molregno
        LEFT JOIN molecule_hierarchy mh ON act.molregno = mh.molregno
        JOIN molecule_dictionary md_parent
          ON COALESCE(mh.parent_molregno, act.molregno) = md_parent.molregno
        LEFT JOIN compound_structures cs_parent
          ON md_parent.molregno = cs_parent.molregno
        LEFT JOIN compound_structures cs_raw
          ON md_raw.molregno = cs_raw.molregno
        WHERE {' AND '.join(clauses)}
    """
    return query, parameters


def extract_activity_rows(
    sqlite_path: Path,
    *,
    exact_only: bool = True,
    include_variants: bool = False,
    include_questionable: bool = False,
    min_confidence_score: int = 8,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    connection = sqlite3.connect(f"file:{sqlite_path.resolve()}?mode=ro", uri=True)
    try:
        query, parameters = build_activity_query(
            exact_only=exact_only,
            include_variants=include_variants,
            include_questionable=include_questionable,
            min_confidence_score=min_confidence_score,
            has_variant_id=_table_has_column(connection, "assays", "variant_id"),
        )
        chunks: list[pd.DataFrame] = []
        for index, chunk in enumerate(
            pd.read_sql_query(query, connection, params=parameters, chunksize=chunksize), start=1
        ):
            chunks.append(chunk)
            log(f"extracted chunk {index}: {len(chunk):,} rows")
    finally:
        connection.close()
    if not chunks:
        return pd.DataFrame(
            columns=[
                "assay_id",
                "depositor_assay_group",
                "confidence_score",
                "target_chembl_id",
                "protein_accession",
                "protein_sequence",
                "molecule_chembl_id",
                "smiles",
                "activity_type",
                "standard_relation",
                "pchembl_value",
            ]
        )
    return pd.concat(chunks, ignore_index=True)


def _joined_unique(values: Iterable[object]) -> str:
    return "|".join(sorted({str(value).strip() for value in values if pd.notna(value) and str(value).strip()}))


def encode_selfies(
    smiles_values: Iterable[str],
    encoder: Callable[[str], str] | None = None,
) -> tuple[dict[str, str], int]:
    if encoder is None:
        try:
            import selfies as sf
        except ImportError as exc:
            raise ImportError(
                "selfies is required for drop-in RewardModel parquets; install requirements.txt "
                "or pass --skip-selfies"
            ) from exc
        encoder = sf.encoder

    encoded: dict[str, str] = {}
    failures = 0
    for smiles in sorted(set(smiles_values)):
        try:
            value = encoder(smiles)
        except Exception:
            value = ""
        if value:
            encoded[smiles] = value
        else:
            failures += 1
    return encoded, failures


def curate_activity_rows(
    raw: pd.DataFrame,
    *,
    activity_threshold: float = 6.0,
    add_selfies: bool = True,
    selfies_encoder: Callable[[str], str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = raw.copy()
    raw_rows = len(frame)
    frame["protein_sequence"] = frame["protein_sequence"].map(normalize_sequence)
    for column in [
        "assay_id",
        "target_chembl_id",
        "protein_accession",
        "molecule_chembl_id",
        "smiles",
    ]:
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    frame["pchembl_value"] = pd.to_numeric(frame["pchembl_value"], errors="coerce")
    frame = frame.loc[
        frame["protein_sequence"].ne("")
        & frame["assay_id"].ne("")
        & frame["target_chembl_id"].ne("")
        & frame["molecule_chembl_id"].ne("")
        & frame["smiles"].ne("")
        & frame["pchembl_value"].map(lambda value: pd.notna(value) and math.isfinite(float(value)))
    ].copy()

    group_columns = [
        "assay_id",
        "target_chembl_id",
        "protein_accession",
        "protein_sequence",
        "molecule_chembl_id",
        "smiles",
    ]
    curated = (
        frame.groupby(group_columns, dropna=False, sort=True)
        .agg(
            pchembl_value=("pchembl_value", "median"),
            activity_type=("activity_type", _joined_unique),
            standard_relation=("standard_relation", _joined_unique),
            measurement_count=("pchembl_value", "size"),
            confidence_score=("confidence_score", "max"),
            depositor_assay_group=("depositor_assay_group", _joined_unique),
        )
        .reset_index()
    )

    rows_after_deduplication = len(curated)
    if add_selfies:
        selfies_map, selfies_failures = encode_selfies(curated["smiles"], encoder=selfies_encoder)
        curated["compound_selfies"] = curated["smiles"].map(selfies_map)
        curated = curated.loc[curated["compound_selfies"].notna()].copy()
    else:
        selfies_failures = 0
        curated["compound_selfies"] = curated["smiles"]

    curated.insert(0, "source", "ChEMBL")
    curated["target_id"] = curated["protein_accession"].where(
        curated["protein_accession"].ne(""), curated["target_chembl_id"]
    )
    curated["protein_id"] = curated["target_id"]
    curated["compound_id"] = curated["molecule_chembl_id"]
    curated["protein_length"] = curated["protein_sequence"].str.len().astype("int32")
    curated["assay_group_id"] = (
        "ChEMBL:" + curated["assay_id"] + ":" + curated["target_chembl_id"]
    )
    curated["binary_label"] = curated["pchembl_value"].ge(activity_threshold).astype("int8")
    curated["assay_type"] = "B"
    curated["target_type"] = "SINGLE PROTEIN"
    curated["chembl_release"] = CHEMBL_RELEASE
    curated = curated.loc[:, BASE_OUTPUT_COLUMNS].sort_values(
        ["target_chembl_id", "assay_id", "compound_id"], kind="stable"
    ).reset_index(drop=True)
    stats = {
        "sql_rows": int(raw_rows),
        "valid_rows_before_deduplication": int(len(frame)),
        "curated_rows": int(len(curated)),
        "measurements_collapsed": int(len(frame) - rows_after_deduplication),
        "selfies_failures": int(selfies_failures),
    }
    return curated, stats


def _prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    protected = [output_dir / "summary.json", output_dir / "all.parquet"]
    protected.extend(output_dir / f"{split}.parquet" for split in ("train", "val", "test"))
    existing = [path for path in protected if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing dataset artifacts: {joined}")


def _split_summary(frame: pd.DataFrame) -> dict[str, object]:
    group_sizes = frame.groupby("assay_group_id").size()
    group_unique_values = frame.groupby("assay_group_id")["pchembl_value"].nunique()
    group_label_counts = frame.groupby("assay_group_id")["binary_label"].nunique()
    return {
        "rows": int(len(frame)),
        "row_fraction": None,
        "assays": int(frame["assay_id"].nunique()),
        "assay_groups": int(frame["assay_group_id"].nunique()),
        "targets": int(frame["target_chembl_id"].nunique()),
        "protein_sequences": int(frame["protein_sequence"].nunique()),
        "protein_clusters": int(frame["protein_cluster_50"].nunique()),
        "molecules": int(frame["compound_id"].nunique()),
        "positives": int(frame["binary_label"].sum()),
        "negatives": int((frame["binary_label"] == 0).sum()),
        "groups_with_at_least_two_ligands": int((group_sizes >= 2).sum()),
        "groups_with_at_least_two_pchembl_values": int((group_unique_values >= 2).sum()),
        "groups_with_both_binary_labels": int((group_label_counts >= 2).sum()),
    }


def assign_cluster_splits_balanced(
    cluster_stats: pd.DataFrame,
    config: SplitConfig,
    *,
    swap_trials: int | None = None,
) -> pd.DataFrame:
    """Deterministically balance rows, positives, and negatives by whole cluster.

    The initial largest-first assignment is improved by exact single-cluster
    moves followed by seeded pair swaps. This avoids the strong 72/14/14 skew
    produced by the older per-bin greedy scorer on ChEMBL's heavy-tailed
    cluster sizes.
    """
    stats = cluster_stats.copy().reset_index(drop=True)
    metrics = ["row_count", "pos_count", "neg_count"]
    values = stats.loc[:, metrics].to_numpy(dtype=float)
    split_names = np.array(["train", "val", "test"])
    ratios = np.array(
        [1.0 - config.val_ratio - config.test_ratio, config.val_ratio, config.test_ratio],
        dtype=float,
    )
    if len(stats) < 3:
        raise ValueError("At least three protein clusters are required for train/val/test splitting")
    if (ratios <= 0).any() or not np.isclose(ratios.sum(), 1.0):
        raise ValueError(f"Split ratios must all be positive and sum to one, got {ratios.tolist()}")

    totals = values.sum(axis=0)
    targets = ratios[:, None] * totals[None, :]
    weights = np.ones(3, dtype=float)
    state = np.zeros((3, 3), dtype=float)
    assignments = np.full(len(stats), -1, dtype=np.int8)
    rng = np.random.default_rng(config.random_seed)
    order = np.lexsort((rng.random(len(stats)), -values[:, 0]))

    def partial_objective(current: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.sum(weights * ((current - target) / np.maximum(target, 1.0)) ** 2, axis=-1)

    def objective(current: np.ndarray) -> float:
        return float(np.sum(partial_objective(current, targets)))

    for index in order:
        scores = []
        for destination in range(3):
            candidate = state.copy()
            candidate[destination] += values[index]
            scores.append(objective(candidate))
        destination = int(np.argmin(scores))
        assignments[index] = destination
        state[destination] += values[index]

    for _ in range(10_000):
        best_delta = 0.0
        best_index = -1
        best_destination = -1
        for source in range(3):
            indices = np.flatnonzero(assignments == source)
            candidates = values[indices]
            old_source = partial_objective(state[source], targets[source])
            for destination in range(3):
                if destination == source:
                    continue
                deltas = (
                    partial_objective(state[source] - candidates, targets[source])
                    + partial_objective(state[destination] + candidates, targets[destination])
                    - old_source
                    - partial_objective(state[destination], targets[destination])
                )
                local_index = int(np.argmin(deltas))
                if deltas[local_index] < best_delta:
                    best_delta = float(deltas[local_index])
                    best_index = int(indices[local_index])
                    best_destination = destination
        if best_index < 0:
            break
        source = int(assignments[best_index])
        state[source] -= values[best_index]
        state[best_destination] += values[best_index]
        assignments[best_index] = best_destination

    if swap_trials is None:
        swap_trials = min(1_000_000, max(10_000, len(stats) * 225))
    accepted_swaps = 0
    for _ in range(swap_trials):
        first = int(rng.integers(len(stats)))
        source = int(assignments[first])
        destination = int(rng.integers(2))
        destination += int(destination >= source)
        destination_indices = np.flatnonzero(assignments == destination)
        if not len(destination_indices):
            continue
        second = int(destination_indices[int(rng.integers(len(destination_indices)))])
        old_score = partial_objective(state[source], targets[source]) + partial_objective(
            state[destination], targets[destination]
        )
        delta = values[second] - values[first]
        new_score = partial_objective(state[source] + delta, targets[source]) + partial_objective(
            state[destination] - delta, targets[destination]
        )
        if new_score < old_score:
            state[source] += delta
            state[destination] -= delta
            assignments[first] = destination
            assignments[second] = source
            accepted_swaps += 1

    stats["split"] = split_names[assignments]
    stats["assignment_objective"] = objective(state)
    stats["accepted_pair_swaps"] = accepted_swaps
    stats["swap_trials"] = swap_trials
    return stats


def _assign_and_write_splits(
    frame: pd.DataFrame,
    cluster_frame: pd.DataFrame,
    output_dir: Path,
    config: SplitConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    cluster_map = dict(zip(cluster_frame["protein_sequence"], cluster_frame["protein_cluster_50"]))
    split_frame = frame.copy()
    split_frame["protein_cluster_50"] = split_frame["protein_sequence"].map(cluster_map)
    if split_frame["protein_cluster_50"].isna().any():
        raise ValueError("At least one protein sequence was not assigned to an MMseqs cluster")
    cluster_stats = (
        split_frame.groupby("protein_cluster_50")["binary_label"]
        .agg(row_count="size", pos_count="sum")
        .reset_index()
    )
    cluster_stats["neg_count"] = cluster_stats["row_count"] - cluster_stats["pos_count"]
    cluster_split = assign_cluster_splits_balanced(cluster_stats, config)
    cluster_split_path = output_dir / "cluster_split.csv"
    temporary_cluster_split = Path(str(cluster_split_path) + ".tmp")
    cluster_split.to_csv(temporary_cluster_split, index=False)
    temporary_cluster_split.replace(cluster_split_path)
    split_map = dict(zip(cluster_split["protein_cluster_50"], cluster_split["split"]))
    split_frame["split"] = split_frame["protein_cluster_50"].map(split_map)

    split_stats: dict[str, object] = {}
    total_rows = max(len(split_frame), 1)
    for split in ("train", "val", "test"):
        split_rows = split_frame.loc[split_frame["split"].eq(split)].copy()
        split_path = output_dir / f"{split}.parquet"
        temporary_split_path = Path(str(split_path) + ".tmp")
        split_rows.to_parquet(temporary_split_path, index=False)
        temporary_split_path.replace(split_path)
        stats = _split_summary(split_rows)
        stats["row_fraction"] = len(split_rows) / total_rows
        split_stats[split] = stats

    split_sets = {
        split: {
            "clusters": set(split_frame.loc[split_frame["split"].eq(split), "protein_cluster_50"]),
            "sequences": set(split_frame.loc[split_frame["split"].eq(split), "protein_sequence"]),
            "targets": set(split_frame.loc[split_frame["split"].eq(split), "target_chembl_id"]),
            "assays": set(split_frame.loc[split_frame["split"].eq(split), "assay_id"]),
        }
        for split in ("train", "val", "test")
    }
    leakage: dict[str, int] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        for key in ("clusters", "sequences", "targets", "assays"):
            leakage[f"{left}_{right}_{key}_overlap"] = len(
                split_sets[left][key] & split_sets[right][key]
            )
    return split_frame, {
        "cluster_split": str(cluster_split_path),
        "split_stats": split_stats,
        "leakage_checks": leakage,
        "assignment": {
            "method": "largest_first_local_moves_seeded_pair_swaps",
            "objective": float(cluster_split["assignment_objective"].iloc[0]),
            "accepted_pair_swaps": int(cluster_split["accepted_pair_swaps"].iloc[0]),
            "swap_trials": int(cluster_split["swap_trials"].iloc[0]),
        },
    }


def build_cluster_disjoint_splits(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    config: SplitConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    _prepare_output_dir(output_dir)
    all_input_path = output_dir / "all.parquet"
    frame.to_parquet(all_input_path, index=False)

    sequence_frame = collect_sequences([all_input_path], config.chunksize)
    sequence_path = output_dir / "protein_sequences.csv"
    fasta_path = output_dir / "proteins.fasta"
    sequence_frame.to_csv(sequence_path, index=False)
    write_fasta(sequence_frame, fasta_path)

    if config.cluster_mode == "mmseqs":
        cluster_tsv = run_mmseqs_cluster(
            fasta_path,
            output_dir / "mmseqs" / "protein_seq50",
            output_dir / "mmseqs" / "tmp",
            config,
        )
    elif config.cluster_mode == "exact":
        cluster_tsv = None
    else:
        raise ValueError(f"Unsupported cluster mode: {config.cluster_mode}")

    cluster_frame = build_cluster_map(sequence_frame, cluster_tsv).rename(
        columns={"protein_cluster_90": "protein_cluster_50"}
    )
    cluster_path = output_dir / "protein_cluster_50.csv"
    cluster_frame.to_csv(cluster_path, index=False)
    split_frame, assignment_artifacts = _assign_and_write_splits(
        frame, cluster_frame, output_dir, config
    )
    return split_frame, {
        "protein_sequences": str(sequence_path),
        "protein_fasta": str(fasta_path),
        "protein_cluster_50": str(cluster_path),
        **assignment_artifacts,
    }


def resplit_existing_dataset(output_dir: Path, config: SplitConfig) -> dict[str, object]:
    """Rebalance existing curated rows without repeating SQL or SELFIES work."""
    started = perf_counter()
    all_path = output_dir / "all.parquet"
    cluster_path = output_dir / "protein_cluster_50.csv"
    summary_path = output_dir / "summary.json"
    for path in (all_path, cluster_path, summary_path):
        if not path.exists():
            raise FileNotFoundError(f"Existing dataset artifact not found: {path}")
    frame = pd.read_parquet(all_path)
    cluster_frame = pd.read_csv(cluster_path)
    split_frame, artifacts = _assign_and_write_splits(frame, cluster_frame, output_dir, config)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    activity_threshold = float(summary.get("filters", {}).get("activity_threshold", 6.0))
    validation = validate_dataset(split_frame, activity_threshold=activity_threshold)
    summary["split_config"] = asdict(config)
    summary["outputs"].update(artifacts)
    summary["validation"] = validation
    summary["last_resplit"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(perf_counter() - started, 2),
    }
    temporary_summary = Path(str(summary_path) + ".tmp")
    temporary_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary_summary.replace(summary_path)
    return summary


def validate_dataset(frame: pd.DataFrame, activity_threshold: float = 6.0) -> dict[str, object]:
    required = set(BASE_OUTPUT_COLUMNS) | {"protein_cluster_50", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing output columns: {missing}")
    if frame["pchembl_value"].isna().any() or not frame["pchembl_value"].map(math.isfinite).all():
        raise ValueError("Output contains invalid pChEMBL values")
    if frame["protein_sequence"].eq("").any() or frame["smiles"].eq("").any():
        raise ValueError("Output contains blank protein sequences or SMILES")
    duplicate_count = int(frame.duplicated(["assay_group_id", "compound_id"]).sum())
    if duplicate_count:
        raise ValueError(f"Output contains {duplicate_count} duplicate assay-group/compound rows")
    sequence_counts = frame.groupby("assay_group_id")["protein_sequence"].nunique()
    if (sequence_counts != 1).any():
        raise ValueError("At least one assay group maps to multiple protein sequences")
    expected_labels = frame["pchembl_value"].ge(activity_threshold).astype("int8")
    return {
        "rows_with_nonfinite_pchembl": 0,
        "duplicate_assay_group_compound_rows": duplicate_count,
        "assay_groups_with_multiple_sequences": int((sequence_counts != 1).sum()),
        "binary_label_mismatches": int((frame["binary_label"] != expected_labels).sum()),
    }


def build_dataset(
    sqlite_path: Path,
    output_dir: Path,
    *,
    activity_threshold: float = 6.0,
    exact_only: bool = True,
    include_variants: bool = False,
    include_questionable: bool = False,
    min_confidence_score: int = 8,
    add_selfies: bool = True,
    chunksize: int = 250_000,
    split_config: SplitConfig | None = None,
) -> dict[str, object]:
    started = perf_counter()
    if split_config is None:
        split_config = SplitConfig(
            val_ratio=0.1,
            test_ratio=0.1,
            random_seed=42,
            min_seq_id=0.5,
            coverage=0.01,
            cov_mode=0,
            cluster_mode="mmseqs",
            chunksize=chunksize,
        )
    raw = extract_activity_rows(
        sqlite_path,
        exact_only=exact_only,
        include_variants=include_variants,
        include_questionable=include_questionable,
        min_confidence_score=min_confidence_score,
        chunksize=chunksize,
    )
    curated, curation_stats = curate_activity_rows(
        raw,
        activity_threshold=activity_threshold,
        add_selfies=add_selfies,
    )
    if curated.empty:
        raise ValueError("No ChEMBL rows remained after filtering and curation")
    split_frame, split_artifacts = build_cluster_disjoint_splits(
        curated, output_dir, config=split_config
    )
    validation = validate_dataset(split_frame, activity_threshold=activity_threshold)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "name": "ChEMBL",
            "release": CHEMBL_RELEASE,
            "release_url": CHEMBL_RELEASE_BASE_URL,
            "sqlite_archive_url": CHEMBL_SQLITE_ARCHIVE_URL,
            "sqlite_archive_sha256": CHEMBL_SQLITE_ARCHIVE_SHA256,
            "sqlite_path": str(sqlite_path),
            "license": "CC BY-SA 3.0",
            "required_attribution": (
                "ChEMBL data is from http://www.ebi.ac.uk/chembl - "
                "the version of ChEMBL is chembl_37."
            ),
        },
        "filters": {
            "assay_type": "B",
            "target_type": "SINGLE PROTEIN",
            "species_group_flag": 0,
            "exactly_one_protein_component": True,
            "pchembl_value_not_null": True,
            "exact_standard_relation_only": exact_only,
            "exclude_variants": not include_variants,
            "exclude_questionable_data_validity": not include_questionable,
            "min_confidence_score": min_confidence_score,
            "activity_threshold": activity_threshold,
            "parent_molecule_normalization": True,
            "duplicate_aggregation": "median pChEMBL per assay-protein-parent molecule",
        },
        "split_config": asdict(split_config),
        "curation": curation_stats,
        "outputs": {
            "all": str(output_dir / "all.parquet"),
            "train": str(output_dir / "train.parquet"),
            "val": str(output_dir / "val.parquet"),
            "test": str(output_dir / "test.parquet"),
            **split_artifacts,
        },
        "validation": validation,
        "runtime_seconds": round(perf_counter() - started, 2),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"wrote {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a ChEMBL single-protein binding pChEMBL dataset."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--sqlite-path", type=Path, default=None)
    source.add_argument("--archive-path", type=Path, default=None)
    source.add_argument("--download", action="store_true")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--activity-threshold", type=float, default=6.0)
    parser.add_argument("--min-confidence-score", type=int, default=8)
    parser.add_argument("--include-censored", action="store_true")
    parser.add_argument("--include-variants", action="store_true")
    parser.add_argument("--include-questionable", action="store_true")
    parser.add_argument("--skip-selfies", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--min-seq-id", type=float, default=0.5)
    parser.add_argument("--coverage", type=float, default=0.01)
    parser.add_argument("--cov-mode", type=int, default=0)
    parser.add_argument("--cluster-mode", choices=("mmseqs", "exact"), default="mmseqs")
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument(
        "--resplit-existing",
        action="store_true",
        help="Reuse all.parquet and protein_cluster_50.csv and only rebalance split artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_config = SplitConfig(
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
        min_seq_id=args.min_seq_id,
        coverage=args.coverage,
        cov_mode=args.cov_mode,
        cluster_mode=args.cluster_mode,
        chunksize=args.chunksize,
    )
    if args.resplit_existing:
        summary = resplit_existing_dataset(args.output_dir, split_config)
        print(json.dumps(summary, indent=2))
        return

    if args.sqlite_path is not None:
        sqlite_path = args.sqlite_path
    else:
        archive_path = args.archive_path
        if args.download:
            archive_path = download_archive(DEFAULT_ARCHIVE_PATH)
        if archive_path is None:
            archive_path = DEFAULT_ARCHIVE_PATH
        if not archive_path.exists():
            raise FileNotFoundError(
                f"ChEMBL archive not found: {archive_path}. Pass --download, --archive-path, or --sqlite-path."
            )
        verify_archive(archive_path)
        sqlite_path = extract_sqlite_archive(archive_path, args.raw_dir)

    if not sqlite_path.exists():
        raise FileNotFoundError(f"ChEMBL SQLite database not found: {sqlite_path}")
    summary = build_dataset(
        sqlite_path,
        args.output_dir,
        activity_threshold=args.activity_threshold,
        exact_only=not args.include_censored,
        include_variants=args.include_variants,
        include_questionable=args.include_questionable,
        min_confidence_score=args.min_confidence_score,
        add_selfies=not args.skip_selfies,
        chunksize=args.chunksize,
        split_config=split_config,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
