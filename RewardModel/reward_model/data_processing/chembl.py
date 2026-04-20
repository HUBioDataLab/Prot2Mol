from __future__ import annotations

import csv
import gzip
import json
import os
import shutil
import sqlite3
import tarfile
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .config import ChemblPreprocessConfig

CHEMBL_DOWNLOADS_DOC_URL = "https://chembl.gitbook.io/chembl-interface-documentation/downloads"
CHEMBL_FTP_LATEST_DIR = "ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest"
SCHEMA_DOC_URL = (
    "https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/"
    "schema-questions-and-sql-examples"
)
TARGET_DOC_URL = (
    "https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/target-questions"
)
SQL_QUERY_VERSION = "chembl_assay_reward_v2_minimal_curated"

CURATED_CSV_FILENAME = "chembl_assay_rows.csv"
CURATED_PARQUET_FILENAME = "chembl_assay_rows.parquet"
PROVENANCE_FILENAME = "provenance.json"

CURATED_FIELD_ORDER: Tuple[str, ...] = (
    "target_chembl_id",
    "protein_sequence",
    "assay_chembl_id",
    "parent_molregno",
    "molecule_chembl_id",
    "compound_selfies",
    "pchembl_value",
    "activity_label",
)


@dataclass
class PreprocessArtifacts:
    sqlite_path: str
    raw_dir: str
    curated_csv_path: str
    curated_parquet_path: Optional[str]
    provenance_path: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _parse_release_from_filename(path: str) -> Optional[str]:
    basename = os.path.basename(path)
    pieces = basename.replace("-", "_").split("_")
    for index, piece in enumerate(pieces):
        lowered = piece.lower()
        if lowered.startswith("chembl") and lowered != "chembl":
            suffix = piece[len("chembl") :]
            if suffix.isdigit():
                return suffix
        if lowered == "chembl" and index + 1 < len(pieces) and pieces[index + 1].isdigit():
            return pieces[index + 1]
    return None


def _table_has_column(connection: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    cursor = connection.execute(f"PRAGMA table_info({table_name})")
    return any(row[1] == column_name for row in cursor.fetchall())


def _try_encode_selfies(smiles: str) -> Optional[str]:
    if smiles is None:
        return None
    text = str(smiles).strip()
    if not text:
        return None
    try:
        import selfies as sf
    except ImportError as exc:
        raise ImportError(
            "The 'selfies' package is required for ChEMBL preprocessing. "
            "Install it before running Phase 3 preprocessing."
        ) from exc

    try:
        return sf.encoder(text)
    except Exception:
        return None


def _normalize_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _row_to_serializable(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: _normalize_scalar(value) for key, value in row.items()}


def _default_sql_query(config: ChemblPreprocessConfig, include_variant_filter: bool) -> Tuple[str, List[Any]]:
    standard_type_placeholders = ", ".join(["?"] * len(config.standard_types))
    confidence_placeholders = ", ".join(["?"] * len(config.confidence_scores))
    variant_clause = "AND a.variant_id IS NULL" if include_variant_filter else ""
    query = f"""
        SELECT
            a.chembl_id AS assay_chembl_id,
            td.chembl_id AS target_chembl_id,
            cs.sequence AS protein_sequence,
            COALESCE(mh.parent_molregno, act.molregno) AS parent_molregno,
            md_parent.chembl_id AS molecule_chembl_id,
            COALESCE(cs_parent.canonical_smiles, cs_raw.canonical_smiles) AS canonical_smiles,
            act.pchembl_value AS pchembl_value
        FROM activities act
        JOIN assays a
            ON act.assay_id = a.assay_id
        JOIN target_dictionary td
            ON a.tid = td.tid
        JOIN target_components tc
            ON td.tid = tc.tid
        JOIN component_sequences cs
            ON tc.component_id = cs.component_id
        JOIN molecule_dictionary md_raw
            ON act.molregno = md_raw.molregno
        LEFT JOIN molecule_hierarchy mh
            ON act.molregno = mh.molregno
        JOIN molecule_dictionary md_parent
            ON COALESCE(mh.parent_molregno, act.molregno) = md_parent.molregno
        LEFT JOIN compound_structures cs_parent
            ON md_parent.molregno = cs_parent.molregno
        LEFT JOIN compound_structures cs_raw
            ON md_raw.molregno = cs_raw.molregno
        WHERE a.assay_type = ?
          AND td.target_type = ?
          AND a.confidence_score IN ({confidence_placeholders})
          AND td.organism = ?
          AND act.pchembl_value IS NOT NULL
          AND act.standard_relation = '='
          AND act.standard_type IN ({standard_type_placeholders})
          AND (act.data_validity_comment IS NULL OR act.data_validity_comment = 'Manually validated')
          {variant_clause}
    """
    params: List[Any] = [
        config.assay_type,
        config.target_type,
        *config.confidence_scores,
        config.organism,
        *config.standard_types,
    ]
    return query, params


def detect_chembl_release(sqlite_path: str) -> Optional[str]:
    inferred = _parse_release_from_filename(sqlite_path)
    if inferred is not None:
        return inferred
    try:
        connection = sqlite3.connect(sqlite_path)
    except sqlite3.Error:
        return None

    try:
        cursor = connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='version'")
        if cursor.fetchone() is None:
            return None
        version_row = connection.execute("SELECT * FROM version LIMIT 1").fetchone()
        if version_row is None:
            return None
        for value in version_row:
            if isinstance(value, str):
                digits = "".join(character for character in value if character.isdigit())
                if digits:
                    return digits
    finally:
        connection.close()
    return None


def download_chembl_sqlite_archive(
    archive_url: str,
    raw_dir: str,
    overwrite: bool = False,
) -> Tuple[str, Optional[str]]:
    _ensure_dir(raw_dir)
    parsed = urllib.parse.urlparse(archive_url)
    archive_name = os.path.basename(parsed.path) or "chembl_sqlite.tar.gz"
    archive_path = os.path.join(raw_dir, archive_name)

    if not os.path.exists(archive_path) or overwrite:
        with urllib.request.urlopen(archive_url) as response, open(archive_path, "wb") as handle:
            shutil.copyfileobj(response, handle)

    extracted_path = extract_chembl_sqlite_archive(archive_path, raw_dir, overwrite=overwrite)
    return archive_path, extracted_path


def extract_chembl_sqlite_archive(
    archive_path: str,
    output_dir: str,
    overwrite: bool = False,
) -> Optional[str]:
    suffix = "".join(Path(archive_path).suffixes).lower()
    if archive_path.lower().endswith((".sqlite", ".db")):
        return archive_path

    if suffix.endswith(".tar.gz") or suffix.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as archive:
            members = [member for member in archive.getmembers() if member.name.endswith((".sqlite", ".db"))]
            if not members:
                return None
            member = members[0]
            extracted_path = os.path.join(output_dir, os.path.basename(member.name))
            if os.path.exists(extracted_path) and not overwrite:
                return extracted_path
            archive.extract(member, output_dir)
            extracted_member_path = os.path.join(output_dir, member.name)
            if extracted_member_path != extracted_path:
                shutil.move(extracted_member_path, extracted_path)
            return extracted_path

    if suffix.endswith(".gz"):
        extracted_name = os.path.basename(archive_path[: -len(".gz")])
        extracted_path = os.path.join(output_dir, extracted_name)
        if os.path.exists(extracted_path) and not overwrite:
            return extracted_path
        with gzip.open(archive_path, "rb") as compressed, open(extracted_path, "wb") as handle:
            shutil.copyfileobj(compressed, handle)
        return extracted_path

    return None


def extract_chembl_activity_rows(
    sqlite_path: str,
    config: ChemblPreprocessConfig,
) -> Tuple[List[Dict[str, Any]], int]:
    connection = sqlite3.connect(sqlite_path)
    connection.row_factory = sqlite3.Row
    try:
        total_raw_rows = int(connection.execute("SELECT COUNT(*) FROM activities").fetchone()[0])
        include_variant_filter = config.exclude_variants and _table_has_column(connection, "assays", "variant_id")
        query, params = _default_sql_query(config, include_variant_filter=include_variant_filter)
        rows = []
        for raw_row in connection.execute(query, params):
            smiles = raw_row["canonical_smiles"]
            if smiles is None:
                continue
            compound_selfies = _try_encode_selfies(smiles)
            if compound_selfies is None:
                continue
            rows.append(
                {
                    "target_chembl_id": raw_row["target_chembl_id"],
                    "protein_sequence": raw_row["protein_sequence"],
                    "assay_chembl_id": raw_row["assay_chembl_id"],
                    "parent_molregno": int(raw_row["parent_molregno"]),
                    "molecule_chembl_id": raw_row["molecule_chembl_id"],
                    "compound_selfies": compound_selfies,
                    "pchembl_value": float(raw_row["pchembl_value"]),
                }
            )
    finally:
        connection.close()
    return rows, total_raw_rows


def deduplicate_assay_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    config: ChemblPreprocessConfig,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        key = (
            str(row["target_chembl_id"]),
            str(row["assay_chembl_id"]),
            int(row["parent_molregno"]),
        )
        grouped[key].append(row)

    deduplicated_rows: List[Dict[str, Any]] = []
    for grouped_rows in grouped.values():
        example = grouped_rows[0]
        aggregated_pchembl = float(median(float(row["pchembl_value"]) for row in grouped_rows))
        deduplicated_rows.append(
            {
                "target_chembl_id": example["target_chembl_id"],
                "protein_sequence": example["protein_sequence"],
                "assay_chembl_id": example["assay_chembl_id"],
                "parent_molregno": int(example["parent_molregno"]),
                "molecule_chembl_id": example["molecule_chembl_id"],
                "compound_selfies": example["compound_selfies"],
                "pchembl_value": aggregated_pchembl,
                "activity_label": int(aggregated_pchembl >= config.activity_threshold),
            }
        )

    assay_counts = Counter(
        (row["target_chembl_id"], row["assay_chembl_id"])
        for row in deduplicated_rows
    )
    filtered_rows = [
        row
        for row in deduplicated_rows
        if assay_counts[(row["target_chembl_id"], row["assay_chembl_id"])] >= config.min_group_size
    ]
    filtered_rows.sort(
        key=lambda row: (
            row["target_chembl_id"],
            row["assay_chembl_id"],
            row["parent_molregno"],
        )
    )
    return filtered_rows


def save_curated_rows(
    rows: Sequence[Mapping[str, Any]],
    output_dir: str,
    write_parquet: bool = True,
) -> Tuple[str, Optional[str]]:
    curated_dir = _ensure_dir(output_dir)
    csv_path = os.path.join(curated_dir, CURATED_CSV_FILENAME)

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CURATED_FIELD_ORDER))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _row_to_serializable(row).get(key) for key in CURATED_FIELD_ORDER})

    parquet_path: Optional[str] = None
    if write_parquet:
        try:
            import pandas as pd

            parquet_path = os.path.join(curated_dir, CURATED_PARQUET_FILENAME)
            pd.DataFrame(
                [{key: _row_to_serializable(row).get(key) for key in CURATED_FIELD_ORDER} for row in rows]
            ).to_parquet(parquet_path, index=False)
        except Exception:
            parquet_path = None

    return csv_path, parquet_path


def save_json(payload: Mapping[str, Any], path: str) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
    return path


def preprocess_chembl_sqlite(config: ChemblPreprocessConfig) -> PreprocessArtifacts:
    if not config.output_dir:
        raise ValueError("config.output_dir must be provided")

    output_dir = os.path.abspath(config.output_dir)
    raw_dir = _ensure_dir(os.path.join(output_dir, config.raw_dir_name))
    curated_dir = _ensure_dir(os.path.join(output_dir, config.curated_dir_name))

    sqlite_path = config.sqlite_path
    if sqlite_path is None:
        if config.download_url is None:
            raise ValueError("sqlite_path or download_url must be provided")
        _, extracted_path = download_chembl_sqlite_archive(
            archive_url=config.download_url,
            raw_dir=raw_dir,
            overwrite=config.overwrite,
        )
        if extracted_path is None:
            raise FileNotFoundError("Downloaded ChEMBL archive did not contain a SQLite database file")
        sqlite_path = extracted_path

    sqlite_path = os.path.abspath(sqlite_path)
    if not os.path.exists(sqlite_path):
        raise FileNotFoundError(f"ChEMBL SQLite file does not exist: {sqlite_path}")

    filtered_rows, _ = extract_chembl_activity_rows(sqlite_path, config)
    curated_rows = deduplicate_assay_rows(filtered_rows, config)

    provenance = {
        "chembl_release": config.chembl_release or detect_chembl_release(sqlite_path),
        "source_url": config.source_url or CHEMBL_FTP_LATEST_DIR,
        "download_url": config.download_url,
        "download_date": _utc_now_iso() if config.download_url else None,
        "source_sqlite_path": sqlite_path,
        "sql_query_version": SQL_QUERY_VERSION,
        "reference_urls": {
            "downloads": CHEMBL_DOWNLOADS_DOC_URL,
            "schema": SCHEMA_DOC_URL,
            "targets": TARGET_DOC_URL,
        },
    }
    provenance_path = save_json(provenance, os.path.join(raw_dir, PROVENANCE_FILENAME))

    curated_csv_path, curated_parquet_path = save_curated_rows(
        curated_rows,
        curated_dir,
        write_parquet=config.write_parquet,
    )

    return PreprocessArtifacts(
        sqlite_path=sqlite_path,
        raw_dir=raw_dir,
        curated_csv_path=curated_csv_path,
        curated_parquet_path=curated_parquet_path,
        provenance_path=provenance_path,
    )
