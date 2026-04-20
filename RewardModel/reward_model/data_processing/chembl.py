from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import random
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
SQL_QUERY_VERSION = "chembl_assay_reward_v1"
SPLIT_POLICY_NAME = "assay_within_target_80_10_10_train_only_if_lt3_assays"

CURATED_CSV_FILENAME = "chembl_assay_rows.csv"
CURATED_PARQUET_FILENAME = "chembl_assay_rows.parquet"
TOKENIZED_JSONL_FILENAME = "rows.jsonl"
TOKENIZED_DATASET_DIRNAME = "hf_dataset"
TOKENIZED_MANIFEST_FILENAME = "manifest.json"
METADATA_FILENAME = "metadata.json"
PROVENANCE_FILENAME = "provenance.json"
PREPROCESS_CONFIG_FILENAME = "preprocess_config.json"

CURATED_FIELD_ORDER: Tuple[str, ...] = (
    "target_chembl_id",
    "target_pref_name",
    "protein_accession",
    "protein_sequence",
    "assay_chembl_id",
    "assay_id",
    "confidence_score",
    "target_organism",
    "parent_molregno",
    "molecule_chembl_id",
    "canonical_smiles",
    "compound_selfies",
    "pchembl_value",
    "activity_label",
    "n_raw_rows_collapsed",
    "group_id",
    "split",
    "split_seed",
    "split_policy",
    "is_sparse_train_only",
    "raw_activity_ids_json",
    "raw_molregnos_json",
    "raw_molecule_chembl_ids_json",
    "raw_standard_types_json",
)


@dataclass
class PreprocessArtifacts:
    sqlite_path: str
    raw_dir: str
    curated_csv_path: str
    curated_parquet_path: Optional[str]
    tokenized_jsonl_path: str
    tokenized_dataset_path: Optional[str]
    metadata_path: str
    provenance_path: str
    config_path: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


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


def _json_dump_sorted(values: Iterable[Any]) -> str:
    unique = sorted({value for value in values if value is not None})
    return json.dumps(unique)


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
    serializable = {}
    for key, value in row.items():
        if isinstance(value, (list, tuple)):
            serializable[key] = [_normalize_scalar(item) for item in value]
        else:
            serializable[key] = _normalize_scalar(value)
    return serializable


def _split_counts(num_assays: int, ratios: Sequence[float]) -> Tuple[int, int, int]:
    if num_assays < 3:
        return num_assays, 0, 0
    train_ratio, valid_ratio, test_ratio = ratios
    tentative_valid = max(1, int(round(num_assays * valid_ratio)))
    tentative_test = max(1, int(round(num_assays * test_ratio)))

    if tentative_valid + tentative_test >= num_assays:
        overflow = tentative_valid + tentative_test - (num_assays - 1)
        while overflow > 0 and tentative_valid > 1:
            tentative_valid -= 1
            overflow -= 1
        while overflow > 0 and tentative_test > 1:
            tentative_test -= 1
            overflow -= 1
        if overflow > 0:
            tentative_valid = max(0, tentative_valid - overflow)
            overflow = 0
    train_count = num_assays - tentative_valid - tentative_test

    if train_count <= 0:
        train_count = 1
        if tentative_valid >= tentative_test and tentative_valid > 0:
            tentative_valid -= 1
        elif tentative_test > 0:
            tentative_test -= 1
    return train_count, tentative_valid, tentative_test


def _default_sql_query(config: ChemblPreprocessConfig, include_variant_filter: bool) -> Tuple[str, List[Any]]:
    standard_type_placeholders = ", ".join(["?"] * len(config.standard_types))
    confidence_placeholders = ", ".join(["?"] * len(config.confidence_scores))
    variant_clause = "AND a.variant_id IS NULL" if include_variant_filter else ""
    query = f"""
        SELECT
            act.activity_id AS activity_id,
            act.assay_id AS assay_id,
            a.chembl_id AS assay_chembl_id,
            a.confidence_score AS confidence_score,
            td.chembl_id AS target_chembl_id,
            td.pref_name AS target_pref_name,
            td.organism AS target_organism,
            cs.accession AS protein_accession,
            cs.sequence AS protein_sequence,
            act.molregno AS raw_molregno,
            md_raw.chembl_id AS raw_molecule_chembl_id,
            COALESCE(mh.parent_molregno, act.molregno) AS parent_molregno,
            md_parent.chembl_id AS molecule_chembl_id,
            COALESCE(cs_parent.canonical_smiles, cs_raw.canonical_smiles) AS canonical_smiles,
            act.standard_type AS standard_type,
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
                    "activity_id": int(raw_row["activity_id"]),
                    "assay_id": int(raw_row["assay_id"]),
                    "assay_chembl_id": raw_row["assay_chembl_id"],
                    "confidence_score": int(raw_row["confidence_score"]),
                    "target_chembl_id": raw_row["target_chembl_id"],
                    "target_pref_name": raw_row["target_pref_name"],
                    "target_organism": raw_row["target_organism"],
                    "protein_accession": raw_row["protein_accession"],
                    "protein_sequence": raw_row["protein_sequence"],
                    "raw_molregno": int(raw_row["raw_molregno"]),
                    "raw_molecule_chembl_id": raw_row["raw_molecule_chembl_id"],
                    "parent_molregno": int(raw_row["parent_molregno"]),
                    "molecule_chembl_id": raw_row["molecule_chembl_id"],
                    "canonical_smiles": smiles,
                    "compound_selfies": compound_selfies,
                    "standard_type": raw_row["standard_type"],
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
                "target_pref_name": example["target_pref_name"],
                "protein_accession": example["protein_accession"],
                "protein_sequence": example["protein_sequence"],
                "assay_chembl_id": example["assay_chembl_id"],
                "assay_id": int(example["assay_id"]),
                "confidence_score": int(example["confidence_score"]),
                "target_organism": example["target_organism"],
                "parent_molregno": int(example["parent_molregno"]),
                "molecule_chembl_id": example["molecule_chembl_id"],
                "canonical_smiles": example["canonical_smiles"],
                "compound_selfies": example["compound_selfies"],
                "pchembl_value": aggregated_pchembl,
                "activity_label": int(aggregated_pchembl >= config.activity_threshold),
                "n_raw_rows_collapsed": len(grouped_rows),
                "group_id": f"{example['target_chembl_id']}__{example['assay_chembl_id']}",
                "raw_activity_ids_json": _json_dump_sorted(row["activity_id"] for row in grouped_rows),
                "raw_molregnos_json": _json_dump_sorted(row["raw_molregno"] for row in grouped_rows),
                "raw_molecule_chembl_ids_json": _json_dump_sorted(
                    row["raw_molecule_chembl_id"] for row in grouped_rows
                ),
                "raw_standard_types_json": _json_dump_sorted(row["standard_type"] for row in grouped_rows),
            }
        )

    group_counts = Counter(row["group_id"] for row in deduplicated_rows)
    filtered_rows = [
        row
        for row in deduplicated_rows
        if group_counts[row["group_id"]] >= config.min_group_size
    ]
    filtered_rows.sort(
        key=lambda row: (
            row["target_chembl_id"],
            row["assay_chembl_id"],
            row["parent_molregno"],
        )
    )
    return filtered_rows


def assign_assay_based_splits(
    rows: Sequence[Mapping[str, Any]],
    config: ChemblPreprocessConfig,
) -> List[Dict[str, Any]]:
    assays_by_target: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        assays_by_target[str(row["target_chembl_id"])].append(str(row["assay_chembl_id"]))

    split_lookup: Dict[Tuple[str, str], Tuple[str, bool]] = {}
    for target_chembl_id, assay_ids in assays_by_target.items():
        unique_assays = sorted(set(assay_ids))
        if len(unique_assays) < config.min_assays_per_protein_for_holdout:
            for assay_chembl_id in unique_assays:
                split_lookup[(target_chembl_id, assay_chembl_id)] = ("train", True)
            continue

        ordered_assays = list(unique_assays)
        rng = random.Random(_stable_seed(config.split_seed, target_chembl_id))
        rng.shuffle(ordered_assays)

        train_count, valid_count, test_count = _split_counts(len(ordered_assays), config.split_ratios)
        train_assays = ordered_assays[:train_count]
        valid_assays = ordered_assays[train_count : train_count + valid_count]
        test_assays = ordered_assays[train_count + valid_count : train_count + valid_count + test_count]

        for assay_chembl_id in train_assays:
            split_lookup[(target_chembl_id, assay_chembl_id)] = ("train", False)
        for assay_chembl_id in valid_assays:
            split_lookup[(target_chembl_id, assay_chembl_id)] = ("valid", False)
        for assay_chembl_id in test_assays:
            split_lookup[(target_chembl_id, assay_chembl_id)] = ("test", False)

    assigned_rows: List[Dict[str, Any]] = []
    for row in rows:
        split, is_sparse_train_only = split_lookup[(str(row["target_chembl_id"]), str(row["assay_chembl_id"]))]
        assigned_row = dict(row)
        assigned_row["split"] = split
        assigned_row["split_seed"] = int(config.split_seed)
        assigned_row["split_policy"] = SPLIT_POLICY_NAME
        assigned_row["is_sparse_train_only"] = bool(is_sparse_train_only)
        assigned_rows.append(assigned_row)
    return assigned_rows


def tokenize_assay_rows(
    rows: Sequence[Mapping[str, Any]],
    protein_tokenizer: Any,
    molecule_tokenizer: Any,
    config: ChemblPreprocessConfig,
) -> List[Dict[str, Any]]:
    from ..model.encoders import batch_encode_texts

    if not rows:
        return []

    unique_proteins = list(dict.fromkeys(str(row["protein_sequence"]) for row in rows))
    unique_molecules = list(dict.fromkeys(str(row["compound_selfies"]) for row in rows))

    protein_tokens: Dict[str, Dict[str, Any]] = {}
    molecule_tokens: Dict[str, Dict[str, Any]] = {}

    for start in range(0, len(unique_proteins), config.tokenization_batch_size):
        batch = unique_proteins[start : start + config.tokenization_batch_size]
        encoded = batch_encode_texts(
            tokenizer=protein_tokenizer,
            texts=batch,
            max_length=config.protein_max_length,
        )
        for batch_index, sequence in enumerate(batch):
            protein_tokens[sequence] = {
                f"protein_{key}": _normalize_scalar(value[batch_index].tolist())
                for key, value in encoded.items()
            }

    for start in range(0, len(unique_molecules), config.tokenization_batch_size):
        batch = unique_molecules[start : start + config.tokenization_batch_size]
        encoded = batch_encode_texts(
            tokenizer=molecule_tokenizer,
            texts=batch,
            max_length=config.molecule_max_length,
        )
        for batch_index, sequence in enumerate(batch):
            molecule_tokens[sequence] = {
                f"molecule_{key}": _normalize_scalar(value[batch_index].tolist())
                for key, value in encoded.items()
            }

    tokenized_rows: List[Dict[str, Any]] = []
    for row in rows:
        tokenized_row = dict(row)
        tokenized_row.update(protein_tokens[str(row["protein_sequence"])])
        tokenized_row.update(molecule_tokens[str(row["compound_selfies"])])
        tokenized_rows.append(_row_to_serializable(tokenized_row))
    return tokenized_rows


def save_curated_rows(
    rows: Sequence[Mapping[str, Any]],
    output_dir: str,
    write_parquet: bool = True,
) -> Tuple[str, Optional[str]]:
    curated_dir = _ensure_dir(output_dir)
    csv_path = os.path.join(curated_dir, CURATED_CSV_FILENAME)

    fieldnames = list(CURATED_FIELD_ORDER)
    if rows:
        for key in rows[0].keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_to_serializable(row))

    parquet_path: Optional[str] = None
    if write_parquet:
        try:
            import pandas as pd

            parquet_path = os.path.join(curated_dir, CURATED_PARQUET_FILENAME)
            pd.DataFrame([_row_to_serializable(row) for row in rows]).to_parquet(parquet_path, index=False)
        except Exception:
            parquet_path = None

    return csv_path, parquet_path


def save_tokenized_rows(
    rows: Sequence[Mapping[str, Any]],
    output_dir: str,
) -> Tuple[str, Optional[str]]:
    tokenized_dir = _ensure_dir(output_dir)
    jsonl_path = os.path.join(tokenized_dir, TOKENIZED_JSONL_FILENAME)
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_row_to_serializable(row), sort_keys=True))
            handle.write("\n")

    dataset_path: Optional[str] = None
    try:
        from datasets import Dataset, DatasetDict

        split_payload = defaultdict(list)
        for row in rows:
            split_payload[str(row["split"])].append(_row_to_serializable(row))
        dataset_dict = DatasetDict(
            {
                split_name: Dataset.from_list(split_rows)
                for split_name, split_rows in split_payload.items()
                if split_rows
            }
        )
        dataset_path = os.path.join(tokenized_dir, TOKENIZED_DATASET_DIRNAME)
        dataset_dict.save_to_disk(dataset_path)
    except Exception:
        dataset_path = None

    manifest_path = os.path.join(tokenized_dir, TOKENIZED_MANIFEST_FILENAME)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "num_rows": len(rows),
                "columns": sorted(rows[0].keys()) if rows else [],
                "hf_dataset_path": dataset_path,
                "jsonl_path": jsonl_path,
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    return jsonl_path, dataset_path


def build_metadata(
    total_raw_rows: int,
    filtered_rows: Sequence[Mapping[str, Any]],
    curated_rows: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "total_raw_rows": int(total_raw_rows),
        "rows_after_filtering": len(filtered_rows),
        "rows_after_deduplication": len(curated_rows),
        "num_proteins": len({row["target_chembl_id"] for row in curated_rows}),
        "num_assays": len({row["group_id"] for row in curated_rows}),
        "num_sparse_train_only_proteins": len(
            {row["target_chembl_id"] for row in curated_rows if row.get("is_sparse_train_only")}
        ),
        "split_counts": {},
        "sql_query_version": provenance.get("sql_query_version"),
        "chembl_release": provenance.get("chembl_release"),
    }
    for split_name in ("train", "valid", "test"):
        split_rows = [row for row in curated_rows if row.get("split") == split_name]
        metadata["split_counts"][split_name] = {
            "proteins": len({row["target_chembl_id"] for row in split_rows}),
            "assays": len({row["group_id"] for row in split_rows}),
            "rows": len(split_rows),
            "molecules": len({row["molecule_chembl_id"] for row in split_rows}),
        }
    return metadata


def save_json(payload: Mapping[str, Any], path: str) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
    return path


def preprocess_chembl_sqlite(
    config: ChemblPreprocessConfig,
    protein_tokenizer: Any,
    molecule_tokenizer: Any,
) -> PreprocessArtifacts:
    if not config.output_dir:
        raise ValueError("config.output_dir must be provided")

    output_dir = os.path.abspath(config.output_dir)
    raw_dir = _ensure_dir(os.path.join(output_dir, config.raw_dir_name))
    curated_dir = _ensure_dir(os.path.join(output_dir, config.curated_dir_name))
    tokenized_dir = _ensure_dir(os.path.join(output_dir, config.tokenized_dir_name))

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

    filtered_rows, total_raw_rows = extract_chembl_activity_rows(sqlite_path, config)
    curated_rows = assign_assay_based_splits(deduplicate_assay_rows(filtered_rows, config), config)
    tokenized_rows = tokenize_assay_rows(curated_rows, protein_tokenizer, molecule_tokenizer, config)

    config_path = save_json(config.to_dict(), os.path.join(output_dir, PREPROCESS_CONFIG_FILENAME))

    provenance = {
        "chembl_release": config.chembl_release or detect_chembl_release(sqlite_path),
        "source_url": config.source_url or CHEMBL_FTP_LATEST_DIR,
        "download_url": config.download_url,
        "download_date": _utc_now_iso() if config.download_url else None,
        "source_sqlite_path": sqlite_path,
        "sql_query_version": SQL_QUERY_VERSION,
        "preprocessing_config_path": config_path,
        "split_seed": config.split_seed,
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
    tokenized_jsonl_path, tokenized_dataset_path = save_tokenized_rows(tokenized_rows, tokenized_dir)

    metadata = build_metadata(
        total_raw_rows=total_raw_rows,
        filtered_rows=filtered_rows,
        curated_rows=curated_rows,
        provenance=provenance,
    )
    metadata["artifact_paths"] = {
        "curated_csv": curated_csv_path,
        "curated_parquet": curated_parquet_path,
        "tokenized_jsonl": tokenized_jsonl_path,
        "tokenized_dataset": tokenized_dataset_path,
        "provenance": provenance_path,
        "config": config_path,
    }
    metadata_path = save_json(metadata, os.path.join(output_dir, METADATA_FILENAME))

    return PreprocessArtifacts(
        sqlite_path=sqlite_path,
        raw_dir=raw_dir,
        curated_csv_path=curated_csv_path,
        curated_parquet_path=curated_parquet_path,
        tokenized_jsonl_path=tokenized_jsonl_path,
        tokenized_dataset_path=tokenized_dataset_path,
        metadata_path=metadata_path,
        provenance_path=provenance_path,
        config_path=config_path,
    )
