from pathlib import Path

import pandas as pd
import pytest
import yaml

from data_processing import build_structure_aware_proteins as builder


def _pdb_ca_line(serial: int, residue: int, plddt: float, chain: str = "A") -> str:
    return (
        f"ATOM  {serial:5d}  CA  ALA {chain}{residue:4d}    "
        f"   0.000   0.000   0.000  1.00{plddt:6.2f}           C  \n"
    )


def test_combine_saprot_sequence_masks_low_confidence_positions():
    combined, masked = builder.combine_saprot_sequence(
        "ACD", "QWE", [90.0, 69.9, 70.0]
    )

    assert combined == "AqC#De"
    assert masked == 1


def test_extract_ca_plddt_reads_one_value_per_residue(tmp_path):
    pdb_path = tmp_path / "model.pdb"
    pdb_path.write_text(
        _pdb_ca_line(1, 1, 91.25)
        + _pdb_ca_line(2, 2, 68.5)
        + _pdb_ca_line(3, 1, 99.0, chain="B"),
        encoding="utf-8",
    )

    assert builder.extract_ca_plddt(pdb_path) == [91.25, 68.5]


def test_fetch_alphafold_entry_selects_canonical_not_isoform(monkeypatch):
    payload = [
        {"uniprotAccession": "P1-2", "entryId": "AF-P1-2-F1"},
        {"uniprotAccession": "P1", "entryId": "AF-P1-F1"},
    ]
    monkeypatch.setattr(
        builder,
        "_request_bytes",
        lambda *args, **kwargs: __import__("json").dumps(payload).encode(),
    )

    assert builder.fetch_alphafold_entry("P1") == payload[1]


def test_load_fixed_cohort_preserves_config_order(tmp_path):
    train = tmp_path / "train.parquet"
    output = tmp_path / "structure_aware.parquet"
    config = tmp_path / "grpo.yaml"
    pd.DataFrame(
        {
            "protein_accession": ["P1", "P2", "P1"],
            "protein_sequence": ["AC", "GT", "AC"],
        }
    ).to_parquet(train, index=False)
    config.write_text(
        yaml.safe_dump(
            {
                "grpo": {
                    "train_parquet_path": train.name,
                    "structure_aware_path": output.name,
                    "protein_id_column": "protein_accession",
                    "protein_sequence_column": "protein_sequence",
                    "eval_protein_ids": ["P2", "P1"],
                }
            }
        ),
        encoding="utf-8",
    )

    cohort, configured_output = builder.load_fixed_cohort(config)

    assert cohort == [
        builder.ProteinInput("P2", "GT"),
        builder.ProteinInput("P1", "AC"),
    ]
    assert configured_output == output


def test_combine_rejects_length_mismatch():
    with pytest.raises(ValueError, match="lengths differ"):
        builder.combine_saprot_sequence("AC", "Q", [90.0, 90.0])
