from __future__ import annotations

import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping

from ..data_processing.chembl import (
    CURATED_CSV_FILENAME,
    CURATED_PARQUET_FILENAME,
)
def _restore_curated_row_types(row: Dict[str, Any]) -> Dict[str, Any]:
    converted = dict(row)
    for key in ("parent_molregno", "activity_label"):
        if key in converted and converted[key] not in ("", None):
            converted[key] = int(converted[key])
    if "pchembl_value" in converted and converted["pchembl_value"] not in ("", None):
        converted["pchembl_value"] = float(converted["pchembl_value"])
    return converted


def _resolve_curated_path(path: str) -> str:
    candidate = os.path.abspath(path)
    if os.path.isdir(candidate):
        csv_path = os.path.join(candidate, CURATED_CSV_FILENAME)
        if os.path.exists(csv_path):
            return csv_path
        parquet_path = os.path.join(candidate, CURATED_PARQUET_FILENAME)
        if os.path.exists(parquet_path):
            return parquet_path
    return candidate


def load_curated_rows(path: str) -> List[Dict[str, Any]]:
    resolved_path = _resolve_curated_path(path)
    if resolved_path.endswith(".parquet"):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required to load curated parquet files") from exc
        return [_restore_curated_row_types(row) for row in pd.read_parquet(resolved_path).to_dict(orient="records")]

    with open(resolved_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [_restore_curated_row_types(dict(row)) for row in reader]


def group_rows_by_group_id(rows: Iterable[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_id = f"{row['target_chembl_id']}__{row['assay_chembl_id']}"
        grouped[group_id].append(dict(row))
    return dict(grouped)


@dataclass
class RewardDataStore:
    curated_rows: List[Dict[str, Any]]

    @classmethod
    def from_output_dir(cls, output_dir: str) -> "RewardDataStore":
        base_dir = os.path.abspath(output_dir)
        curated_dir = os.path.join(base_dir, "curated")
        return cls(curated_rows=load_curated_rows(curated_dir))

    def curated_groups(self) -> Dict[str, List[Dict[str, Any]]]:
        return group_rows_by_group_id(self.curated_rows)

    def get_curated_group(self, group_id: str) -> List[Dict[str, Any]]:
        return self.curated_groups().get(group_id, [])
