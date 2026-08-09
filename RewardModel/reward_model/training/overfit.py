from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from typing import Any, Dict, Optional, Sequence

from datasets import Dataset, load_from_disk

from .data import (
    get_tokenized_split_dataset_paths,
    is_valid_negative_for_positive,
)


OVERFIT_METADATA_COLUMNS = ("group_id", "compound_id", "pchembl_value")


def _iter_metadata_rows(dataset: Dataset, batch_size: int = 65_536):
    metadata = dataset.select_columns(list(OVERFIT_METADATA_COLUMNS))
    for start in range(0, len(metadata), batch_size):
        stop = min(start + batch_size, len(metadata))
        batch = metadata[start:stop]
        for offset, values in enumerate(
            zip(
                batch["group_id"],
                batch["compound_id"],
                batch["pchembl_value"],
            )
        ):
            yield start + offset, values


def _evenly_spaced_indices(rows: Sequence[tuple[int, float]], count: int) -> list[int]:
    ordered = sorted(rows, key=lambda item: (item[1], item[0]))
    if len(ordered) < count:
        raise ValueError(f"Need {count} rows, but only {len(ordered)} are available")
    if len(ordered) == count:
        return [row_index for row_index, _ in ordered]

    positions = [
        round(position * (len(ordered) - 1) / (count - 1))
        for position in range(count)
    ]
    if len(set(positions)) != count:
        raise RuntimeError("Evenly spaced overfit selection produced duplicate positions")
    return [ordered[position][0] for position in positions]


def select_overfit_examples(
    dataset: Dataset,
    *,
    num_molecules: int = 50,
    group_id: Optional[str] = None,
    min_pchembl_span: float = 0.5,
    candidate_batch_size: int = 32,
) -> tuple[Dataset, Dict[str, Any]]:
    """Select untouched rows for one assay, with one row per molecule."""
    if num_molecules < 3:
        raise ValueError("num_molecules must be >= 3")
    if min_pchembl_span < 0.0:
        raise ValueError("min_pchembl_span must be >= 0")
    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be > 0")
    missing_columns = sorted(
        set(OVERFIT_METADATA_COLUMNS) - set(dataset.column_names)
    )
    if missing_columns:
        raise ValueError(f"Tokenized dataset is missing columns: {missing_columns}")

    group_counts = Counter(
        str(raw_group_id)
        for _, (raw_group_id, _, _) in _iter_metadata_rows(dataset)
    )
    if group_id is not None:
        requested_group_id = str(group_id)
        if requested_group_id not in group_counts:
            raise ValueError(f"Requested group_id is not present: {requested_group_id}")
        candidate_group_ids = [requested_group_id]
    else:
        candidate_group_ids = [
            candidate_group_id
            for candidate_group_id, count in sorted(
                group_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if count >= num_molecules
        ]

    for candidate_start in range(0, len(candidate_group_ids), candidate_batch_size):
        candidate_chunk = candidate_group_ids[
            candidate_start : candidate_start + candidate_batch_size
        ]
        candidate_set = set(candidate_chunk)
        unique_rows_by_group: Dict[str, Dict[str, tuple[int, float]]] = {
            candidate: {} for candidate in candidate_chunk
        }
        for row_index, (raw_group_id, raw_compound_id, raw_pchembl_value) in (
            _iter_metadata_rows(dataset)
        ):
            row_group_id = str(raw_group_id)
            if row_group_id not in candidate_set:
                continue
            pchembl_value = float(raw_pchembl_value)
            if not math.isfinite(pchembl_value):
                continue
            unique_rows_by_group[row_group_id].setdefault(
                str(raw_compound_id),
                (row_index, pchembl_value),
            )

        for candidate_group_id in candidate_chunk:
            unique_rows = list(unique_rows_by_group[candidate_group_id].values())
            if len(unique_rows) < num_molecules:
                continue
            selected_indices = _evenly_spaced_indices(unique_rows, num_molecules)
            selected = dataset.select(selected_indices)
            selected_pchembl = [float(value) for value in selected["pchembl_value"]]
            pchembl_span = max(selected_pchembl) - min(selected_pchembl)
            if pchembl_span < min_pchembl_span:
                continue

            if "example_id" in selected.column_names:
                selected = selected.remove_columns("example_id")
            selected = selected.add_column("example_id", list(range(num_molecules)))
            valid_pair_count = sum(
                1
                for positive_index, positive in enumerate(selected_pchembl)
                for negative_index, negative in enumerate(selected_pchembl)
                if positive_index != negative_index
                and positive > negative
                and is_valid_negative_for_positive(positive, negative)
            )
            summary = {
                "group_id": candidate_group_id,
                "source_group_rows": int(group_counts[candidate_group_id]),
                "source_group_unique_molecules": len(unique_rows),
                "selected_molecules": num_molecules,
                "pchembl_min": min(selected_pchembl),
                "pchembl_max": max(selected_pchembl),
                "pchembl_span": pchembl_span,
                "valid_ordered_pair_count": valid_pair_count,
                "selection": "one untouched row per molecule, evenly spaced by pChEMBL",
            }
            return selected, summary

    requested = f"group {group_id!r}" if group_id is not None else "any assay group"
    raise ValueError(
        f"Could not find {num_molecules} unique molecules with pChEMBL span "
        f">= {min_pchembl_span} in {requested}"
    )


def _dataset_sha256(dataset: Dataset) -> str:
    digest = hashlib.sha256()
    for row in dataset:
        digest.update(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def build_overfit_dataset(
    *,
    source_dir: str,
    output_dir: str,
    num_molecules: int = 50,
    group_id: Optional[str] = None,
    min_pchembl_span: float = 0.5,
) -> Dict[str, Any]:
    """Write the exact same selected assay rows to train, validation, and test."""
    resolved_source_dir = os.path.abspath(source_dir)
    resolved_output_dir = os.path.abspath(output_dir)
    source_train_path = get_tokenized_split_dataset_paths(resolved_source_dir)["train"]
    if not os.path.isdir(source_train_path):
        raise FileNotFoundError(f"Source train examples do not exist: {source_train_path}")
    if os.path.exists(resolved_output_dir):
        raise FileExistsError(
            f"Output directory already exists; refusing to replace it: {resolved_output_dir}"
        )

    source_dataset = load_from_disk(source_train_path)
    selected, selection_summary = select_overfit_examples(
        source_dataset,
        num_molecules=num_molecules,
        group_id=group_id,
        min_pchembl_span=min_pchembl_span,
    )
    dataset_sha256 = _dataset_sha256(selected)
    output_parent = os.path.dirname(resolved_output_dir)
    os.makedirs(output_parent, exist_ok=True)
    temporary_dir = tempfile.mkdtemp(
        prefix=f".{os.path.basename(resolved_output_dir)}.",
        dir=output_parent,
    )
    try:
        temporary_paths = get_tokenized_split_dataset_paths(temporary_dir)
        for split_name in ("train", "val", "test"):
            selected.save_to_disk(temporary_paths[split_name])

        manifest = {
            "source_dir": resolved_source_dir,
            "source_train_examples": source_train_path,
            "output_dir": resolved_output_dir,
            **selection_summary,
            "dataset_sha256": dataset_sha256,
            "split_rows": {
                "train": len(selected),
                "val": len(selected),
                "test": len(selected),
            },
            "split_dataset_sha256": {
                "train": dataset_sha256,
                "val": dataset_sha256,
                "test": dataset_sha256,
            },
            "split_identity": "train, val, and test contain identical rows",
        }
        with open(
            os.path.join(temporary_dir, "overfit_manifest.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_dir, resolved_output_dir)
        return manifest
    except BaseException:
        if os.path.isdir(temporary_dir):
            import shutil

            shutil.rmtree(temporary_dir)
        raise
