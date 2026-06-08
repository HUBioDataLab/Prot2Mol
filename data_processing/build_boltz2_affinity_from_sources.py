#!/usr/bin/env python3
"""Build Boltz-2-style affinity data directly from public source dumps.

This script fills the gap before ``prepare_boltz2_affinity_dataset.py``:

1. Download ChEMBL SQLite and BindingDB TSV dumps when needed.
2. Extract normalized ChEMBL and BindingDB source CSV/TSV files.
3. Run the curation/deduplication builder to create final training artifacts.

The default ChEMBL version is v34 because that is what the Boltz-2 paper used.
"""

import argparse
import csv
import os
import re
import sqlite3
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.prepare_boltz2_affinity_dataset import (
    Boltz2AffinityConfig,
    prepare_dataset,
    prepare_dataset_chunked,
)

DEFAULT_CHEMBL_VERSION = 34
DEFAULT_CHEMBL_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/"
    "chembl_34/chembl_34_sqlite.tar.gz"
)
DEFAULT_BINDINGDB_URL = (
    "https://www.bindingdb.org/rwd/bind/downloads/BindingDB_All_202504_tsv.zip"
)

BINDINGDB_ACTIVITY_COLUMNS = {
    "Ki": ("Ki (nM)", "ki (nm)"),
    "Kd": ("Kd (nM)", "kd (nm)"),
    "IC50": ("IC50 (nM)", "ic50 (nm)"),
    "EC50": ("EC50 (nM)", "ec50 (nm)"),
}


def download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> Path:
    """Download a file with basic resume support."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    mode = "wb"
    existing_size = destination.stat().st_size if destination.exists() else 0
    if existing_size:
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"

    with requests.get(url, stream=True, headers=headers, timeout=60) as response:
        if response.status_code == 416:
            return destination
        if response.status_code == 200 and existing_size and mode == "ab":
            # Server ignored Range; restart from scratch.
            existing_size = 0
            mode = "wb"
        response.raise_for_status()
        with open(destination, mode) as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
    return destination


def find_chembl_sqlite(raw_dir: Path, chembl_version: int) -> Optional[Path]:
    patterns = [
        f"**/chembl_{chembl_version}.db",
        f"**/chembl_{chembl_version}.sqlite",
        f"**/chembl_{chembl_version}.sqlite3",
    ]
    for pattern in patterns:
        matches = list(raw_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def extract_chembl_sqlite(tar_path: Path, raw_dir: Path, chembl_version: int) -> Path:
    existing = find_chembl_sqlite(raw_dir, chembl_version)
    if existing is not None:
        return existing

    with tarfile.open(tar_path, "r:gz") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile()
            and Path(member.name).name
            in {
                f"chembl_{chembl_version}.db",
                f"chembl_{chembl_version}.sqlite",
                f"chembl_{chembl_version}.sqlite3",
            }
        ]
        if not members:
            raise FileNotFoundError(f"No ChEMBL SQLite DB found inside {tar_path}")
        archive.extract(members[0], path=raw_dir)

    extracted = find_chembl_sqlite(raw_dir, chembl_version)
    if extracted is None:
        raise FileNotFoundError(f"Could not locate extracted ChEMBL {chembl_version} SQLite DB.")
    return extracted


def extract_chembl_affinity(sqlite_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    query = """
    WITH single_component_targets AS (
      SELECT
        tc.tid,
        MIN(cs.accession) AS target_accession,
        MIN(cs.sequence) AS protein_sequence,
        COUNT(DISTINCT tc.component_id) AS component_count
      FROM target_components tc
      JOIN component_sequences cs
        ON cs.component_id = tc.component_id
      WHERE cs.sequence IS NOT NULL
      GROUP BY tc.tid
      HAVING COUNT(DISTINCT tc.component_id) = 1
    )
    SELECT
      'ChEMBL' AS source,
      ass.chembl_id AS assay_id,
      td.chembl_id AS target_id,
      td.chembl_id AS target_chembl_id,
      sct.target_accession AS target_accession,
      sct.protein_sequence AS protein_sequence,
      str.canonical_smiles AS smiles,
      act.standard_type AS standard_type,
      act.standard_value AS standard_value,
      act.standard_units AS standard_units,
      COALESCE(act.standard_relation, '=') AS standard_relation,
      ass.confidence_score AS confidence_score,
      td.target_type AS target_type,
      ass.assay_type AS assay_type,
      0 AS source_unreliable,
      md.chembl_id AS molecule_chembl_id,
      act.activity_id AS activity_id
    FROM activities act
    JOIN assays ass
      ON ass.assay_id = act.assay_id
    JOIN target_dictionary td
      ON td.tid = ass.tid
    JOIN single_component_targets sct
      ON sct.tid = td.tid
    JOIN molecule_dictionary md
      ON md.molregno = act.molregno
    JOIN compound_structures str
      ON str.molregno = act.molregno
    WHERE ass.confidence_score = 9
      AND td.target_type = 'SINGLE PROTEIN'
      AND ass.assay_type IN ('B', 'F')
      AND act.standard_type IN ('Ki', 'Kd', 'IC50', 'XC50', 'EC50', 'AC50')
      AND act.standard_value IS NOT NULL
      AND act.standard_value > 0
      AND act.standard_units IN ('nM', 'uM', 'µM', 'M', 'mM', 'pM')
      AND COALESCE(act.standard_relation, '=') IN ('=', '>')
      AND str.canonical_smiles IS NOT NULL
      AND act.standard_flag = 1
    """
    with sqlite3.connect(sqlite_path) as connection:
        frame = pd.read_sql_query(query, connection)
    frame.to_csv(output_path, index=False)
    return output_path


def _lower_columns(columns: Iterable[str]) -> dict[str, str]:
    return {column.lower(): column for column in columns}


def _find_bindingdb_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lower_to_original = _lower_columns(columns)
    for candidate in candidates:
        found = lower_to_original.get(candidate.lower())
        if found is not None:
            return found
    return None


def _bindingdb_zip_member(path: Path) -> Optional[str]:
    if path.suffix.lower() != ".zip":
        return None
    with zipfile.ZipFile(path) as archive:
        candidates = [
            name
            for name in archive.namelist()
            if name.lower().endswith((".tsv", ".txt"))
            and not name.endswith("/")
        ]
    if not candidates:
        raise FileNotFoundError(f"No TSV/TXT member found in {path}")
    return candidates[0]


def _iter_bindingdb_chunks(path: Path, chunksize: int):
    if path.suffix.lower() == ".zip":
        member = _bindingdb_zip_member(path)
        with zipfile.ZipFile(path) as archive:
            with archive.open(member) as handle:
                yield from pd.read_csv(
                    handle,
                    sep="\t",
                    dtype=str,
                    chunksize=chunksize,
                    low_memory=False,
                )
    else:
        yield from pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            chunksize=chunksize,
            low_memory=False,
        )


def parse_bindingdb_measurement(value) -> tuple[Optional[str], Optional[float]]:
    """Parse BindingDB nM cells such as '>10000', '= 3.2', or '5'."""
    if value is None or pd.isna(value):
        return None, None
    text = str(value).strip()
    if not text:
        return None, None

    qualifier = "="
    if text[0] in {">", "<", "="}:
        qualifier = text[0]
        text = text[1:].strip()
    elif text[:2] in {">=", "<="}:
        qualifier = text[0]
        text = text[2:].strip()

    text = text.replace(",", "")
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if match is None:
        return None, None
    value_nm = float(match.group(0))
    if value_nm <= 0:
        return None, None
    return qualifier, value_nm


def extract_bindingdb_affinity(bindingdb_path: Path, output_path: Path, chunksize: int = 100_000) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False

    for chunk in _iter_bindingdb_chunks(bindingdb_path, chunksize=chunksize):
        columns = list(chunk.columns)
        smiles_col = _find_bindingdb_column(columns, ("Ligand SMILES", "SMILES"))
        sequence_col = _find_bindingdb_column(columns, ("BindingDB Target Chain Sequence", "Target Chain Sequence"))
        chains_col = _find_bindingdb_column(
            columns,
            ("Number of Protein Chains in Target (>1 implies a multichain complex)", "Number of Protein Chains in Target"),
        )
        doi_col = _find_bindingdb_column(columns, ("Article DOI", "DOI"))
        pmid_col = _find_bindingdb_column(columns, ("PMID", "PubMed ID"))
        aid_col = _find_bindingdb_column(columns, ("PubChem AID",))
        monomer_col = _find_bindingdb_column(columns, ("BindingDB MonomerID", "MonomerID"))
        swiss_col = _find_bindingdb_column(columns, ("UniProt (SwissProt) Primary ID of Target Chain",))
        trembl_col = _find_bindingdb_column(columns, ("UniProt (TrEMBL) Primary ID of Target Chain",))
        target_name_col = _find_bindingdb_column(columns, ("Target Name",))

        missing = [
            name
            for name, column in {
                "Ligand SMILES": smiles_col,
                "BindingDB Target Chain Sequence": sequence_col,
            }.items()
            if column is None
        ]
        if missing:
            raise ValueError(f"BindingDB file is missing required columns: {', '.join(missing)}")

        normalized_rows = []
        for _, row in chunk.iterrows():
            smiles = str(row.get(smiles_col, "") or "").strip()
            sequence = str(row.get(sequence_col, "") or "").strip().replace(" ", "").replace("\n", "")
            if not smiles or not sequence:
                continue

            chain_value = str(row.get(chains_col, "") or "").strip() if chains_col else ""
            try:
                num_chains = int(float(chain_value)) if chain_value else 1
            except ValueError:
                num_chains = 1
            if num_chains > 1:
                continue

            doi = str(row.get(doi_col, "") or "").strip() if doi_col else ""
            pmid = str(row.get(pmid_col, "") or "").strip() if pmid_col else ""
            pubchem_aid = str(row.get(aid_col, "") or "").strip() if aid_col else ""
            assay_id = doi or pmid or pubchem_aid or f"bindingdb_target_{abs(hash(sequence))}"

            target_id = ""
            for column in (swiss_col, trembl_col, target_name_col):
                if column:
                    target_id = str(row.get(column, "") or "").strip()
                    if target_id:
                        break
            if not target_id:
                target_id = f"BINDINGDB_TARGET_{abs(hash(sequence))}"

            compound_id = str(row.get(monomer_col, "") or "").strip() if monomer_col else ""
            if compound_id:
                compound_id = f"BindingDB:{compound_id}"

            for activity_type, candidates in BINDINGDB_ACTIVITY_COLUMNS.items():
                value_col = _find_bindingdb_column(columns, candidates)
                if value_col is None:
                    continue
                qualifier, value_nm = parse_bindingdb_measurement(row.get(value_col))
                if qualifier is None or value_nm is None:
                    continue
                normalized_rows.append(
                    {
                        "source": "BindingDB",
                        "doi": assay_id,
                        "target_id": target_id,
                        "protein_sequence": sequence,
                        "smiles": smiles,
                        "activity_type": activity_type,
                        "activity_value_uM": value_nm / 1000.0,
                        "activity_qualifier": qualifier,
                        "num_protein_chains": num_chains,
                        "compound_id": compound_id,
                    }
                )

        if normalized_rows:
            out = pd.DataFrame(normalized_rows)
            out.to_csv(
                output_path,
                sep="\t",
                index=False,
                mode="a" if wrote_header else "w",
                header=not wrote_header,
                quoting=csv.QUOTE_MINIMAL,
            )
            wrote_header = True

    if not wrote_header:
        raise ValueError("No BindingDB affinity rows were extracted.")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download/extract ChEMBL and BindingDB, then build Boltz-2-style affinity data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-dir", default="dataset/raw/boltz2_sources")
    parser.add_argument("--output-dir", default="dataset/processed/boltz2_affinity")
    parser.add_argument("--chembl-version", type=int, default=DEFAULT_CHEMBL_VERSION)
    parser.add_argument("--chembl-url", default=DEFAULT_CHEMBL_URL)
    parser.add_argument("--bindingdb-url", default=DEFAULT_BINDINGDB_URL)
    parser.add_argument("--chembl-sqlite", default=None, help="Use an existing ChEMBL SQLite DB.")
    parser.add_argument("--bindingdb-tsv", default=None, help="Use an existing BindingDB TSV/ZIP.")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-chembl", action="store_true")
    parser.add_argument("--skip-bindingdb", action="store_true")
    parser.add_argument("--bindingdb-chunksize", type=int, default=100_000)
    parser.add_argument("--final-chunk-size", type=int, default=100_000)
    parser.add_argument("--include-censored", action="store_true")
    parser.add_argument("--cluster-map", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    source_files = []

    if not args.skip_chembl:
        if args.chembl_sqlite:
            chembl_sqlite = Path(args.chembl_sqlite)
        else:
            chembl_archive = raw_dir / f"chembl_{args.chembl_version}_sqlite.tar.gz"
            if not args.skip_download and not chembl_archive.exists():
                print(f"Downloading ChEMBL from {args.chembl_url}")
                download_file(args.chembl_url, chembl_archive)
            chembl_sqlite = extract_chembl_sqlite(chembl_archive, raw_dir, args.chembl_version)
        chembl_csv = raw_dir / "chembl_affinity.csv"
        print(f"Extracting ChEMBL affinity rows to {chembl_csv}")
        source_files.append(str(extract_chembl_affinity(chembl_sqlite, chembl_csv)))

    if not args.skip_bindingdb:
        if args.bindingdb_tsv:
            bindingdb_source = Path(args.bindingdb_tsv)
        else:
            bindingdb_source = raw_dir / Path(args.bindingdb_url).name
            if not args.skip_download and not bindingdb_source.exists():
                print(f"Downloading BindingDB from {args.bindingdb_url}")
                download_file(args.bindingdb_url, bindingdb_source)
        bindingdb_tsv = raw_dir / "bindingdb_affinity.tsv"
        print(f"Extracting BindingDB affinity rows to {bindingdb_tsv}")
        source_files.append(
            str(extract_bindingdb_affinity(bindingdb_source, bindingdb_tsv, chunksize=args.bindingdb_chunksize))
        )

    if not source_files:
        raise ValueError("No source files selected. Remove skip flags or provide existing source exports.")

    config = Boltz2AffinityConfig(include_censored=args.include_censored)
    if args.final_chunk_size and args.final_chunk_size > 0:
        summary = prepare_dataset_chunked(
            input_paths=source_files,
            output_dir=args.output_dir,
            config=config,
            cluster_map_path=args.cluster_map,
            chunk_size=args.final_chunk_size,
        )
    else:
        summary = prepare_dataset(
            input_paths=source_files,
            output_dir=args.output_dir,
            config=config,
            cluster_map_path=args.cluster_map,
        )
    print(summary)


if __name__ == "__main__":
    main()
