#!/usr/bin/env python3
"""Lightweight exploratory statistics for Prot2Mol CSV datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

PAIR_KEY_MASK = (1 << 32) - 1


def _stable_digest(value: str) -> bytes:
    return hashlib.blake2b(value.encode("utf-8"), digest_size=16).digest()


def _is_missing(value: str | None) -> bool:
    if value is None:
        return True
    stripped = value.strip()
    return stripped == "" or stripped.lower() == "nan"


def _safe_float(value: str | None) -> float | None:
    if _is_missing(value):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except ValueError:
        return None


def _series_summary(values: Iterable[float | int]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q01": None,
            "q05": None,
            "q25": None,
            "q50": None,
            "q75": None,
            "q95": None,
            "q99": None,
            "max": None,
        }

    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "q01": float(np.quantile(array, 0.01)),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "q50": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def _concentration(counter: Counter[int], total: int) -> dict[str, float | int]:
    counts = sorted(counter.values(), reverse=True)
    result: dict[str, float | int] = {}
    for k in (1, 5, 10, 50):
        top_sum = int(sum(counts[:k]))
        result[f"top_{k}_rows"] = top_sum
        result[f"top_{k}_share"] = float(top_sum / total) if total else 0.0
    return result


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_dataset(csv_path: str | Path, top_n: int = 20) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    csv_path = Path(csv_path)
    row_count = 0
    missing_counts: Counter[str] = Counter()
    pchembl_values: list[float] = []
    protein_lengths_per_row: list[int] = []

    target_index: dict[str, int] = {}
    target_labels: list[str] = []
    target_lengths: list[int] = []
    target_sequence_digest_by_idx: list[bytes] = []
    targets_with_multiple_fastas: set[int] = set()

    compound_index: dict[str, int] = {}
    compound_labels: list[str] = []
    compound_smiles_digest_by_idx: list[bytes] = []
    compounds_with_multiple_smiles: set[int] = set()

    target_row_counts: Counter[int] = Counter()
    compound_row_counts: Counter[int] = Counter()
    pair_stats: dict[int, list[float]] = {}

    unique_fasta_digests: set[bytes] = set()
    unique_smiles_digests: set[bytes] = set()
    unique_selfies_digests: set[bytes] = set()

    columns: list[str] = []

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} does not have a header row.")
        columns = list(reader.fieldnames)

        for row in reader:
            row_count += 1

            for column in columns:
                if _is_missing(row.get(column)):
                    missing_counts[column] += 1

            target_id = (row.get("Target_CHEMBL_ID") or "").strip()
            target_fasta = (row.get("Target_FASTA") or "").strip()
            compound_cid = (row.get("Compound_CID") or "").strip()
            compound_smiles = (row.get("Compound_SMILES") or "").strip()
            compound_selfies = (row.get("Compound_SELFIES") or "").strip()
            compound_key = compound_cid or compound_smiles

            if not _is_missing(target_fasta):
                protein_lengths_per_row.append(len(target_fasta))
                unique_fasta_digests.add(_stable_digest(target_fasta))

            if not _is_missing(compound_smiles):
                unique_smiles_digests.add(_stable_digest(compound_smiles))

            if not _is_missing(compound_selfies):
                unique_selfies_digests.add(_stable_digest(compound_selfies))

            if target_id not in target_index:
                target_index[target_id] = len(target_labels)
                target_labels.append(target_id)
                target_lengths.append(len(target_fasta))
                target_sequence_digest_by_idx.append(_stable_digest(target_fasta))
            else:
                target_idx = target_index[target_id]
                if target_fasta and target_lengths[target_idx] == 0:
                    target_lengths[target_idx] = len(target_fasta)
                    target_sequence_digest_by_idx[target_idx] = _stable_digest(target_fasta)
                elif target_fasta and target_sequence_digest_by_idx[target_idx] != _stable_digest(target_fasta):
                    targets_with_multiple_fastas.add(target_idx)

            if compound_key not in compound_index:
                compound_index[compound_key] = len(compound_labels)
                compound_labels.append(compound_key)
                compound_smiles_digest_by_idx.append(_stable_digest(compound_smiles))
            else:
                compound_idx = compound_index[compound_key]
                if compound_smiles and compound_smiles_digest_by_idx[compound_idx] == _stable_digest(""):
                    compound_smiles_digest_by_idx[compound_idx] = _stable_digest(compound_smiles)
                elif compound_smiles and compound_smiles_digest_by_idx[compound_idx] != _stable_digest(compound_smiles):
                    compounds_with_multiple_smiles.add(compound_idx)

            target_idx = target_index[target_id]
            compound_idx = compound_index[compound_key]

            target_row_counts[target_idx] += 1
            compound_row_counts[compound_idx] += 1

            pchembl_value = _safe_float(row.get("pchembl_value_Median"))
            if pchembl_value is not None:
                pchembl_values.append(pchembl_value)

            pair_key = (target_idx << 32) | compound_idx
            if pair_key not in pair_stats:
                if pchembl_value is None:
                    pair_stats[pair_key] = [1.0, float("nan"), float("nan")]
                else:
                    pair_stats[pair_key] = [1.0, pchembl_value, pchembl_value]
            else:
                pair_stats[pair_key][0] += 1.0
                if pchembl_value is not None:
                    if np.isnan(pair_stats[pair_key][1]) or pchembl_value < pair_stats[pair_key][1]:
                        pair_stats[pair_key][1] = pchembl_value
                    if np.isnan(pair_stats[pair_key][2]) or pchembl_value > pair_stats[pair_key][2]:
                        pair_stats[pair_key][2] = pchembl_value

    unique_compounds_per_target: Counter[int] = Counter()
    unique_targets_per_compound: Counter[int] = Counter()
    rows_per_pair: list[int] = []
    duplicate_pair_spans: list[float] = []
    duplicate_pairs = 0
    rows_in_duplicate_pairs = 0
    duplicate_pairs_with_label_variation = 0

    for pair_key, stats in pair_stats.items():
        count = int(stats[0])
        min_pchembl = stats[1]
        max_pchembl = stats[2]
        target_idx = pair_key >> 32
        compound_idx = pair_key & PAIR_KEY_MASK

        unique_compounds_per_target[target_idx] += 1
        unique_targets_per_compound[compound_idx] += 1
        rows_per_pair.append(count)

        if count > 1:
            duplicate_pairs += 1
            rows_in_duplicate_pairs += count
            if not np.isnan(min_pchembl) and not np.isnan(max_pchembl):
                span = max_pchembl - min_pchembl
                duplicate_pair_spans.append(span)
                if span > 0:
                    duplicate_pairs_with_label_variation += 1

    top_targets = []
    for target_idx, count in target_row_counts.most_common(top_n):
        top_targets.append(
            {
                "Target_CHEMBL_ID": target_labels[target_idx],
                "row_count": count,
                "row_share": count / row_count if row_count else 0.0,
                "unique_compounds": unique_compounds_per_target[target_idx],
                "protein_length": target_lengths[target_idx],
            }
        )

    top_compounds = []
    for compound_idx, count in compound_row_counts.most_common(top_n):
        top_compounds.append(
            {
                "Compound_Key": compound_labels[compound_idx],
                "row_count": count,
                "row_share": count / row_count if row_count else 0.0,
                "unique_targets": unique_targets_per_compound[compound_idx],
            }
        )

    total_targets = len(target_index)
    total_compounds = len(compound_index)
    unique_pair_count = len(pair_stats)
    matrix_cell_count = total_targets * total_compounds

    summary: dict[str, object] = {
        "dataset_path": str(csv_path.resolve()),
        "file_size_bytes": csv_path.stat().st_size,
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns,
        "missing_counts": dict(sorted(missing_counts.items())),
        "unique_counts": {
            "targets_by_chembl_id": total_targets,
            "target_fastas_digest_based": len(unique_fasta_digests),
            "compounds_by_cid_or_smiles": total_compounds,
            "compound_smiles_digest_based": len(unique_smiles_digests),
            "compound_selfies_digest_based": len(unique_selfies_digests),
            "protein_compound_pairs": unique_pair_count,
        },
        "consistency_checks": {
            "targets_with_multiple_fastas": len(targets_with_multiple_fastas),
            "compounds_with_multiple_smiles_for_same_key": len(compounds_with_multiple_smiles),
        },
        "matrix_density": float(unique_pair_count / matrix_cell_count) if matrix_cell_count else 0.0,
        "pchembl": _series_summary(pchembl_values),
        "protein_length_per_row": _series_summary(protein_lengths_per_row),
        "protein_length_per_unique_target": _series_summary(target_lengths),
        "rows_per_target": _series_summary(target_row_counts.values()),
        "rows_per_compound": _series_summary(compound_row_counts.values()),
        "unique_compounds_per_target": _series_summary(unique_compounds_per_target.values()),
        "unique_targets_per_compound": _series_summary(unique_targets_per_compound.values()),
        "rows_per_pair": _series_summary(rows_per_pair),
        "duplicate_pairs": {
            "count": duplicate_pairs,
            "share_of_unique_pairs": float(duplicate_pairs / unique_pair_count) if unique_pair_count else 0.0,
            "rows_in_duplicate_pairs": rows_in_duplicate_pairs,
            "share_of_rows": float(rows_in_duplicate_pairs / row_count) if row_count else 0.0,
            "pairs_with_label_variation": duplicate_pairs_with_label_variation,
            "label_span_across_duplicate_pairs": _series_summary(duplicate_pair_spans),
        },
        "target_row_concentration": _concentration(target_row_counts, row_count),
        "compound_row_concentration": _concentration(compound_row_counts, row_count),
        "singleton_counts": {
            "targets_with_single_row": int(sum(1 for value in target_row_counts.values() if value == 1)),
            "compounds_with_single_row": int(sum(1 for value in compound_row_counts.values() if value == 1)),
            "targets_with_single_unique_compound": int(sum(1 for value in unique_compounds_per_target.values() if value == 1)),
            "compounds_with_single_unique_target": int(sum(1 for value in unique_targets_per_compound.values() if value == 1)),
        },
    }
    return summary, top_targets, top_compounds


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a Prot2Mol CSV dataset.")
    parser.add_argument("csv_path", help="Path to the CSV dataset to analyze.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top targets/compounds to export.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory where summary.json and top tables will be written.",
    )
    args = parser.parse_args()

    summary, top_targets, top_compounds = summarize_dataset(args.csv_path, top_n=args.top_n)

    if args.output_dir is None:
        print(json.dumps(summary, indent=2))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    top_targets_path = args.output_dir / "top_targets.csv"
    top_compounds_path = args.output_dir / "top_compounds.csv"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(
        top_targets_path,
        ["Target_CHEMBL_ID", "row_count", "row_share", "unique_compounds", "protein_length"],
        top_targets,
    )
    _write_csv(
        top_compounds_path,
        ["Compound_Key", "row_count", "row_share", "unique_targets"],
        top_compounds,
    )

    print(f"Saved summary to {summary_path}")
    print(f"Saved top targets to {top_targets_path}")
    print(f"Saved top compounds to {top_compounds_path}")


if __name__ == "__main__":
    main()
