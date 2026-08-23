#!/usr/bin/env python3
"""Build the SaProt AA+3Di mapping required by FusionDTI GRPO.

The fixed GRPO cohort is read from ``eval_protein_ids`` in the GRPO YAML.
Canonical AlphaFold DB structures are downloaded, Foldseek is used to derive
3Di descriptors, low-confidence AlphaFold positions are masked, and the result
is written to the configured ``structure_aware_path`` Parquet file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prot2mol.io.config import load_yaml_config


ALPHAFOLD_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"


@dataclass(frozen=True)
class ProteinInput:
    accession: str
    sequence: str


def _resolve_config_path(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def load_fixed_cohort(config_path: str | Path) -> tuple[list[ProteinInput], Path]:
    """Load the ordered fixed cohort and output path from a GRPO config."""
    config_path = Path(config_path).expanduser().resolve()
    config = load_yaml_config(str(config_path), section="grpo")
    protein_ids = [str(value).strip() for value in config.get("eval_protein_ids", [])]
    if not protein_ids:
        raise ValueError("GRPO config has no eval_protein_ids fixed cohort")
    if len(set(protein_ids)) != len(protein_ids):
        raise ValueError("GRPO eval_protein_ids contains duplicate accessions")

    train_path_value = config.get("train_parquet_path")
    output_path_value = config.get("structure_aware_path")
    if not train_path_value or not output_path_value:
        raise ValueError(
            "GRPO config must define train_parquet_path and structure_aware_path"
        )

    id_column = str(config.get("protein_id_column", "protein_accession"))
    sequence_column = str(config.get("protein_sequence_column", "protein_sequence"))
    train_path = _resolve_config_path(config_path, str(train_path_value))
    output_path = _resolve_config_path(config_path, str(output_path_value))
    frame = pd.read_parquet(train_path, columns=[id_column, sequence_column])
    frame[id_column] = frame[id_column].astype(str).str.strip()
    frame[sequence_column] = frame[sequence_column].astype(str).str.strip().str.upper()

    sequences_by_accession: dict[str, str] = {}
    for accession in protein_ids:
        values = frame.loc[frame[id_column] == accession, sequence_column].unique()
        if len(values) != 1:
            raise ValueError(
                f"Expected exactly one training sequence for {accession}, found {len(values)}"
            )
        sequence = str(values[0])
        if not sequence:
            raise ValueError(f"Training sequence is empty for {accession}")
        sequences_by_accession[accession] = sequence

    cohort = [
        ProteinInput(accession=accession, sequence=sequences_by_accession[accession])
        for accession in protein_ids
    ]
    return cohort, output_path


def _request_bytes(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Prot2Mol-structure-aware-builder/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_alphafold_entry(accession: str, timeout: float = 60.0) -> dict[str, Any]:
    """Return the canonical AlphaFold DB entry for a UniProt accession."""
    payload = json.loads(
        _request_bytes(ALPHAFOLD_API_URL.format(accession=accession), timeout).decode(
            "utf-8"
        )
    )
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected AlphaFold DB response for {accession}")

    exact = [
        entry
        for entry in payload
        if str(entry.get("uniprotAccession", "")) == accession
        and str(entry.get("entryId", "")) == f"AF-{accession}-F1"
    ]
    if len(exact) != 1:
        raise ValueError(
            f"Expected one canonical AlphaFold DB model for {accession}, found {len(exact)}"
        )
    return exact[0]


def download_structure(entry: dict[str, Any], path: Path, timeout: float = 120.0) -> None:
    """Download an AlphaFold PDB atomically, retaining it as a local cache."""
    if path.exists() and path.stat().st_size > 0:
        return
    pdb_url = str(entry.get("pdbUrl", ""))
    if not pdb_url:
        raise ValueError(f"AlphaFold entry {entry.get('entryId')} has no pdbUrl")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _request_bytes(pdb_url, timeout)
    if not payload.startswith((b"HEADER", b"ATOM", b"TITLE")):
        raise ValueError(f"Downloaded file is not a PDB: {pdb_url}")
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def run_foldseek_descriptor(
    foldseek: str | Path,
    structure_path: Path,
    expected_sequence: str,
) -> tuple[str, str, str]:
    """Return descriptor name, AA sequence, and 3Di sequence for one structure."""
    foldseek = Path(foldseek).expanduser().resolve()
    if not foldseek.is_file():
        raise FileNotFoundError(f"Foldseek executable not found: {foldseek}")

    with tempfile.TemporaryDirectory(prefix="prot2mol-foldseek-") as temporary_dir:
        output_path = Path(temporary_dir) / "descriptor.tsv"
        command = [
            str(foldseek),
            "structureto3didescriptor",
            "-v",
            "0",
            "--threads",
            "1",
            "--chain-name-mode",
            "1",
            str(structure_path),
            str(output_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Foldseek failed for {structure_path.name}: {completed.stderr.strip()}"
            )
        if not output_path.exists():
            raise RuntimeError(f"Foldseek produced no descriptor for {structure_path.name}")
        rows = []
        for line in output_path.read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) >= 3:
                rows.append((fields[0], fields[1].upper(), fields[2].upper()))

    matching = [row for row in rows if row[1] == expected_sequence]
    if len(matching) != 1:
        observed = ", ".join(f"{name}:{len(seq)}" for name, seq, _ in rows)
        raise ValueError(
            f"Foldseek sequence mismatch for {structure_path.name}; expected "
            f"length {len(expected_sequence)}, observed [{observed}]"
        )
    name, aa_sequence, structure_sequence = matching[0]
    if len(aa_sequence) != len(structure_sequence):
        raise ValueError(
            f"Foldseek AA/3Di length mismatch for {structure_path.name}: "
            f"{len(aa_sequence)} != {len(structure_sequence)}"
        )
    return name, aa_sequence, structure_sequence


def extract_ca_plddt(structure_path: str | Path, chain: str = "A") -> list[float]:
    """Extract one AlphaFold pLDDT value per residue from PDB C-alpha atoms."""
    values: list[float] = []
    seen_residues: set[tuple[str, str]] = set()
    with Path(structure_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("ATOM") or len(line) < 66:
                continue
            if line[21].strip() != chain or line[12:16].strip() != "CA":
                continue
            if line[16] not in {" ", "A"}:
                continue
            residue_key = (line[22:26], line[26])
            if residue_key in seen_residues:
                continue
            seen_residues.add(residue_key)
            values.append(float(line[60:66]))
    if not values:
        raise ValueError(f"No C-alpha pLDDT values found in {structure_path} chain {chain}")
    return values


def combine_saprot_sequence(
    aa_sequence: str,
    structure_sequence: str,
    plddt_values: Iterable[float],
    plddt_threshold: float = 70.0,
) -> tuple[str, int]:
    """Interleave AA and lower-case 3Di tokens, masking low-confidence 3Di."""
    plddt_values = list(plddt_values)
    if not (len(aa_sequence) == len(structure_sequence) == len(plddt_values)):
        raise ValueError(
            "AA, 3Di, and pLDDT lengths differ: "
            f"{len(aa_sequence)}, {len(structure_sequence)}, {len(plddt_values)}"
        )
    tokens = []
    masked_positions = 0
    for amino_acid, structure_token, confidence in zip(
        aa_sequence, structure_sequence, plddt_values
    ):
        if confidence < plddt_threshold:
            structure_token = "#"
            masked_positions += 1
        else:
            structure_token = structure_token.lower()
        tokens.append(amino_acid + structure_token)
    return "".join(tokens), masked_positions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_mapping(
    cohort: Iterable[ProteinInput],
    *,
    foldseek: str | Path,
    cache_dir: str | Path,
    plddt_threshold: float = 70.0,
    timeout: float = 120.0,
) -> pd.DataFrame:
    """Download, encode, and validate every protein in an ordered cohort."""
    cache_dir = Path(cache_dir).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for index, protein in enumerate(cohort, start=1):
        print(f"[{index}] {protein.accession}: fetching AlphaFold DB model", flush=True)
        entry = fetch_alphafold_entry(protein.accession, timeout=timeout)
        api_sequence = str(entry.get("sequence", "")).upper()
        if api_sequence != protein.sequence:
            raise ValueError(
                f"AlphaFold/training sequence mismatch for {protein.accession}: "
                f"{len(api_sequence)} != {len(protein.sequence)} residues"
            )

        version = int(entry.get("latestVersion", 0))
        structure_path = cache_dir / f"AF-{protein.accession}-F1-model_v{version}.pdb"
        download_structure(entry, structure_path, timeout=timeout)
        descriptor_name, aa_sequence, structure_sequence = run_foldseek_descriptor(
            foldseek, structure_path, protein.sequence
        )
        plddt_values = extract_ca_plddt(structure_path, chain="A")
        combined_sequence, masked_positions = combine_saprot_sequence(
            aa_sequence,
            structure_sequence,
            plddt_values,
            plddt_threshold=plddt_threshold,
        )
        rows.append(
            {
                "protein_accession": protein.accession,
                "protein_sequence": aa_sequence,
                "structure_aware_sequence": combined_sequence,
                "foldseek_sequence": structure_sequence.lower(),
                "sequence_length": len(aa_sequence),
                "masked_positions": masked_positions,
                "masked_fraction": masked_positions / len(aa_sequence),
                "plddt_threshold": float(plddt_threshold),
                "alphafold_entry_id": str(entry["entryId"]),
                "alphafold_model_version": version,
                "alphafold_global_plddt": float(entry["globalMetricValue"]),
                "structure_source_url": str(entry["pdbUrl"]),
                "structure_sha256": _sha256(structure_path),
                "foldseek_descriptor": descriptor_name,
            }
        )
        print(
            f"    {len(aa_sequence)} residues; {masked_positions} positions masked",
            flush=True,
        )
    return pd.DataFrame(rows)


def write_mapping(frame: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="GRPO YAML with fixed cohort")
    parser.add_argument("--foldseek", required=True, help="Foldseek executable")
    parser.add_argument(
        "--cache-dir",
        default="dataset/cache/alphafold_structures",
        help="Directory retaining downloaded AlphaFold PDB files",
    )
    parser.add_argument("--output", default=None, help="Override configured Parquet output")
    parser.add_argument("--plddt-threshold", type=float, default=70.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cohort, configured_output = load_fixed_cohort(args.config)
    output_path = Path(args.output).expanduser().resolve() if args.output else configured_output
    frame = build_mapping(
        cohort,
        foldseek=args.foldseek,
        cache_dir=args.cache_dir,
        plddt_threshold=args.plddt_threshold,
        timeout=args.timeout,
    )
    write_mapping(frame, output_path)
    print(f"Wrote {len(frame)} structure-aware proteins to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
