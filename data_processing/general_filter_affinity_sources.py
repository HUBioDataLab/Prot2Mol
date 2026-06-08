#!/usr/bin/env python3
"""General-filter ChEMBL and BindingDB affinity sources without RDKit.

This intentionally does not run RDKit standardization, PAINS filtering,
heavy-atom filtering, SELFIES encoding, or Boltz assay-signal filters.
It applies only the source/activity filters and writes a merged table.
"""

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

ACTIVITY_TYPES = {"KI", "KD", "IC50", "XC50", "EC50", "AC50"}
SUPPORTED_UNITS_TO_UM = {
    "M": 1_000_000.0,
    "MM": 1_000.0,
    "UM": 1.0,
    "µM": 1.0,
    "ΜM": 1.0,
    "NM": 0.001,
    "PM": 0.000001,
}

OUTPUT_COLUMNS = [
    "source",
    "assay_id",
    "target_id",
    "target_chembl_id",
    "protein_sequence",
    "compound_id",
    "smiles",
    "activity_type",
    "activity_value_uM",
    "activity_qualifier",
    "y_log10_uM",
    "pchembl_value",
]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def normalize_string(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def normalize_sequence(value) -> str:
    return "".join(normalize_string(value).split()).upper()


def normalize_units(value) -> str:
    return normalize_string(value).replace("μ", "µ").upper().replace(" ", "")


def normalize_qualifier(value) -> str:
    qualifier = normalize_string(value)
    if qualifier in {"", "="}:
        return "="
    if qualifier.startswith(">"):
        return ">"
    if qualifier.startswith("<"):
        return "<"
    return qualifier


def stable_hash(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def make_overlap_key(row: pd.Series) -> str:
    value = round(float(row["y_log10_uM"]), 4)
    return "|".join(
        [
            row["protein_sequence"],
            row["smiles"],
            row["activity_type"],
            row["activity_qualifier"],
            str(value),
        ]
    )


def make_exact_key(row: pd.Series) -> str:
    value = round(float(row["y_log10_uM"]), 6)
    return "|".join(
        [
            row["source"],
            row["assay_id"],
            row["target_id"],
            row["protein_sequence"],
            row["smiles"],
            row["activity_type"],
            row["activity_qualifier"],
            str(value),
        ]
    )


def create_database(db_path: Path) -> sqlite3.Connection:
    if db_path.exists():
        db_path.unlink()
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute(
        """
        CREATE TABLE affinity (
            source TEXT,
            assay_id TEXT,
            target_id TEXT,
            target_chembl_id TEXT,
            protein_sequence TEXT,
            compound_id TEXT,
            smiles TEXT,
            activity_type TEXT,
            activity_value_uM REAL,
            activity_qualifier TEXT,
            y_log10_uM REAL,
            pchembl_value REAL,
            exact_key TEXT UNIQUE,
            overlap_key TEXT
        )
        """
    )
    connection.execute("CREATE TABLE chembl_overlap (overlap_key TEXT PRIMARY KEY)")
    return connection


def insert_rows(connection: sqlite3.Connection, frame: pd.DataFrame, source: str) -> tuple[int, int, int]:
    if frame.empty:
        return 0, 0, 0

    inserted = 0
    exact_duplicates = 0
    chembl_overlaps = 0
    records = []
    overlap_records = []
    for _, row in frame.iterrows():
        exact_key = make_exact_key(row)
        overlap_key = make_overlap_key(row)
        if source == "BindingDB":
            exists = connection.execute(
                "SELECT 1 FROM chembl_overlap WHERE overlap_key = ?",
                (overlap_key,),
            ).fetchone()
            if exists:
                chembl_overlaps += 1
                continue
        records.append(tuple(row[column] for column in OUTPUT_COLUMNS) + (exact_key, overlap_key))
        if source == "ChEMBL":
            overlap_records.append((overlap_key,))

    if records:
        before = connection.total_changes
        connection.executemany(
            """
            INSERT OR IGNORE INTO affinity (
                source,
                assay_id,
                target_id,
                target_chembl_id,
                protein_sequence,
                compound_id,
                smiles,
                activity_type,
                activity_value_uM,
                activity_qualifier,
                y_log10_uM,
                pchembl_value,
                exact_key,
                overlap_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        inserted = connection.total_changes - before
        exact_duplicates = len(records) - inserted

    if overlap_records:
        connection.executemany(
            "INSERT OR IGNORE INTO chembl_overlap (overlap_key) VALUES (?)",
            overlap_records,
        )
    connection.commit()
    return inserted, exact_duplicates, chembl_overlaps


def normalize_chembl_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    activity_type = chunk["standard_type"].map(lambda value: normalize_string(value).upper())
    qualifier = chunk["standard_relation"].map(normalize_qualifier)
    values = pd.to_numeric(chunk["standard_value"], errors="coerce")
    units = chunk["standard_units"].map(normalize_units)
    factors = units.map(SUPPORTED_UNITS_TO_UM)
    values_um = values * factors

    frame = pd.DataFrame(
        {
            "source": "ChEMBL",
            "assay_id": chunk["assay_id"].map(normalize_string),
            "target_id": chunk["target_id"].map(normalize_string),
            "target_chembl_id": chunk["target_chembl_id"].map(normalize_string),
            "protein_sequence": chunk["protein_sequence"].map(normalize_sequence),
            "compound_id": chunk["molecule_chembl_id"].map(normalize_string),
            "smiles": chunk["smiles"].map(normalize_string),
            "activity_type": activity_type,
            "activity_value_uM": values_um,
            "activity_qualifier": qualifier,
        }
    )
    keep = frame["activity_type"].isin(ACTIVITY_TYPES)
    keep &= np.isfinite(frame["activity_value_uM"]) & (frame["activity_value_uM"] > 0)
    keep &= frame["activity_qualifier"].isin({"=", ">"})
    keep &= frame["smiles"].ne("")
    keep &= frame["protein_sequence"].ne("")
    frame = frame.loc[keep].copy()
    frame["y_log10_uM"] = np.log10(frame["activity_value_uM"].astype(float))
    frame["pchembl_value"] = 6.0 - frame["y_log10_uM"]
    return frame[OUTPUT_COLUMNS]


def normalize_bindingdb_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    activity_type = chunk["activity_type"].map(lambda value: normalize_string(value).upper())
    qualifier = chunk["activity_qualifier"].map(normalize_qualifier)
    values_um = pd.to_numeric(chunk["activity_value_uM"], errors="coerce")
    chains = pd.to_numeric(chunk.get("num_protein_chains", 1), errors="coerce").fillna(1)

    assay_id = chunk["doi"].map(normalize_string)
    target_id = chunk["target_id"].map(normalize_string)
    smiles = chunk["smiles"].map(normalize_string)
    sequence = chunk["protein_sequence"].map(normalize_sequence)
    compound_id = chunk["compound_id"].map(normalize_string)

    frame = pd.DataFrame(
        {
            "source": "BindingDB",
            "assay_id": assay_id,
            "target_id": target_id,
            "target_chembl_id": "",
            "protein_sequence": sequence,
            "compound_id": compound_id,
            "smiles": smiles,
            "activity_type": activity_type,
            "activity_value_uM": values_um,
            "activity_qualifier": qualifier,
        }
    )
    frame["assay_id"] = frame["assay_id"].where(frame["assay_id"].ne(""), "BindingDB:" + frame["target_id"])
    frame["compound_id"] = frame["compound_id"].where(
        frame["compound_id"].ne(""),
        frame["smiles"].map(lambda value: stable_hash("bdb_cmp", value)),
    )
    keep = frame["activity_type"].isin(ACTIVITY_TYPES)
    keep &= np.isfinite(frame["activity_value_uM"]) & (frame["activity_value_uM"] > 0)
    keep &= frame["activity_qualifier"].isin({"=", ">"})
    keep &= frame["smiles"].ne("")
    keep &= frame["protein_sequence"].ne("")
    keep &= chains <= 1
    frame = frame.loc[keep].copy()
    frame["y_log10_uM"] = np.log10(frame["activity_value_uM"].astype(float))
    frame["pchembl_value"] = 6.0 - frame["y_log10_uM"]
    return frame[OUTPUT_COLUMNS]


def process_source(
    connection: sqlite3.Connection,
    path: Path,
    source: str,
    chunksize: int,
) -> dict:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    normalizer = normalize_chembl_chunk if source == "ChEMBL" else normalize_bindingdb_chunk
    counts = {
        "raw_rows": 0,
        "after_general_filters": 0,
        "inserted_rows": 0,
        "exact_duplicates_removed": 0,
        "bindingdb_chembl_overlaps_removed": 0,
    }
    log(f"Processing {source}: {path}")
    for index, chunk in enumerate(pd.read_csv(path, sep=sep, dtype=str, chunksize=chunksize, low_memory=False), start=1):
        started = perf_counter()
        counts["raw_rows"] += len(chunk)
        frame = normalizer(chunk)
        counts["after_general_filters"] += len(frame)
        inserted, exact_duplicates, chembl_overlaps = insert_rows(connection, frame, source)
        counts["inserted_rows"] += inserted
        counts["exact_duplicates_removed"] += exact_duplicates
        counts["bindingdb_chembl_overlaps_removed"] += chembl_overlaps
        log(
            f"{source} chunk {index}: raw={len(chunk):,}, filtered={len(frame):,}, "
            f"inserted={inserted:,}, exact_dupes={exact_duplicates:,}, "
            f"chembl_overlaps={chembl_overlaps:,}, elapsed={perf_counter() - started:.1f}s"
        )
    return counts


def export_csv(connection: sqlite3.Connection, output_csv: Path, batch_size: int = 100_000) -> int:
    log(f"Exporting merged general-filtered CSV: {output_csv}")
    cursor = connection.execute(
        """
        SELECT source, assay_id, target_id, target_chembl_id, protein_sequence,
               compound_id, smiles, activity_type, activity_value_uM,
               activity_qualifier, y_log10_uM, pchembl_value
        FROM affinity
        ORDER BY source, target_id, assay_id, compound_id
        """
    )
    total = 0
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(OUTPUT_COLUMNS)
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            writer.writerows(rows)
            total += len(rows)
            log(f"Exported {total:,} rows...")
    return total


def collect_sqlite_summary(connection: sqlite3.Connection) -> dict:
    source_counts = {
        source: count
        for source, count in connection.execute(
            "SELECT source, COUNT(*) FROM affinity GROUP BY source"
        )
    }
    activity_counts = {
        activity_type: count
        for activity_type, count in connection.execute(
            "SELECT activity_type, COUNT(*) FROM affinity GROUP BY activity_type"
        )
    }
    row = connection.execute(
        """
        SELECT
            COUNT(DISTINCT target_id),
            COUNT(DISTINCT protein_sequence),
            COUNT(DISTINCT smiles),
            COUNT(DISTINCT assay_id)
        FROM affinity
        """
    ).fetchone()
    return {
        "source_counts": source_counts,
        "activity_type_counts": activity_counts,
        "targets": int(row[0]),
        "protein_sequences": int(row[1]),
        "smiles": int(row[2]),
        "assays": int(row[3]),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="General-filter ChEMBL and BindingDB affinity exports without RDKit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--chembl", required=True)
    parser.add_argument("--bindingdb", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunksize", type=int, default=100_000)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "general_filtered_affinity.sqlite"
    csv_path = output_dir / "chembl_bindingdb_general_filtered.csv"
    summary_path = output_dir / "summary.json"

    started = perf_counter()
    connection = create_database(db_path)
    chembl_counts = process_source(connection, Path(args.chembl), "ChEMBL", args.chunksize)
    bindingdb_counts = process_source(connection, Path(args.bindingdb), "BindingDB", args.chunksize)
    final_rows = export_csv(connection, csv_path)
    sqlite_summary = collect_sqlite_summary(connection)
    connection.close()

    summary = {
        "scope": "general_filters_no_rdkit_no_pains_no_heavy_atom_no_assay_signal_filters",
        "inputs": {
            "chembl": args.chembl,
            "bindingdb": args.bindingdb,
        },
        "outputs": {
            "sqlite": str(db_path),
            "csv": str(csv_path),
            "summary": str(summary_path),
        },
        "counts": {
            "chembl": chembl_counts,
            "bindingdb": bindingdb_counts,
            "final_rows": int(final_rows),
            **sqlite_summary,
        },
        "deduplication": {
            "exact_duplicate_key": "source + assay_id + target_id + protein_sequence + source_smiles + activity_type + qualifier + rounded log10(uM)",
            "bindingdb_chembl_overlap_key": "protein_sequence + source_smiles + activity_type + qualifier + rounded log10(uM)",
            "note": "No RDKit canonicalization was used, so cross-source dedupe only catches exact text-level SMILES matches.",
        },
        "elapsed_seconds": perf_counter() - started,
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
