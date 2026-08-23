import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from prot2mol.inference.produce_molecules import MoleculeGenerator, parse_arguments
from prot2mol.main import COMMANDS


def _config(tmp_path, **overrides):
    values = {
        "model_file": str(tmp_path / "model"),
        "prot_emb_model": "esm2",
        "protein_model_id": None,
        "decoder_type": "gpt2",
        "decoder_model_id": "zjunlp/MolGen-large",
        "n_layer": 1,
        "n_head": 2,
        "n_emb": 8,
        "conditioning_dropout": 0.1,
        "models_base": None,
        "protein_sequence": "MKT",
        "protein_id": None,
        "dataset_path": None,
        "train_reference_limit": 100,
        "num_samples": 3,
        "batch_size": 2,
        "prot_max_length": 16,
        "max_mol_len": 8,
        "temperature": 1.0,
        "top_p": 0.9,
        "output_file": str(tmp_path / "generated.csv"),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _fake_components(self):
    from conftest import DummyBatchTokenizer

    self.mol_tokenizer = DummyBatchTokenizer()
    self.prot_tokenizer = DummyBatchTokenizer()
    self.model = SimpleNamespace(eval=lambda: None)


def test_explicit_sequence_generation_writes_generation_only_output(tmp_path, monkeypatch):
    monkeypatch.setattr(MoleculeGenerator, "_load_components", _fake_components)
    monkeypatch.setattr(
        MoleculeGenerator,
        "_generate_tokens",
        lambda self, sequence: torch.tensor([[1, 3, 2], [1, 3, 2], [1, 3, 2]]),
    )
    generator = MoleculeGenerator(_config(tmp_path))
    frame, metrics = generator.run_generation()

    assert frame["generated_selfies"].tolist() == ["[C]"] * 3
    assert "predicted_pchembl" not in frame
    assert frame["protein_sequence"].eq("MKT").all()
    assert metrics["validity"] == 1.0
    assert metrics["uniqueness"] == pytest.approx(1 / 3)
    assert metrics["generation_time_sec"] >= 0
    assert pd.read_csv(generator.config.output_file).shape[0] == 3
    assert json.loads((tmp_path / "generated.metrics.json").read_text())["generation_time_sec"] >= 0


def test_target_resolution_reads_canonical_validation_parquet(tmp_path, monkeypatch):
    dataset = tmp_path / "generation_data"
    dataset.mkdir()
    pd.DataFrame(
        {
            "target_chembl_id": ["CHEMBL1", "CHEMBL2"],
            "protein_sequence": ["MKT", "GGA"],
            "smiles": ["C", "O"],
        }
    ).to_parquet(dataset / "validation.parquet", index=False)
    pd.DataFrame(
        {"protein_sequence": ["AAA"], "smiles": ["N"]}
    ).to_parquet(dataset / "train.parquet", index=False)
    monkeypatch.setattr(MoleculeGenerator, "_load_components", _fake_components)
    generator = MoleculeGenerator(
        _config(
            tmp_path,
            protein_sequence=None,
            protein_id="CHEMBL1",
            dataset_path=str(dataset),
        )
    )
    sequence, references = generator._resolve_protein_and_references()
    assert sequence == "MKT"
    assert references == ["C"]
    assert generator.train_references == ["N"]


def test_target_resolution_requires_existing_identifier(tmp_path, monkeypatch):
    data = tmp_path / "data.csv"
    pd.DataFrame(
        {"Target_CHEMBL_ID": ["CHEMBL2"], "Target_FASTA": ["GGA"], "Compound_SMILES": ["O"]}
    ).to_csv(data, index=False)
    monkeypatch.setattr(MoleculeGenerator, "_load_components", _fake_components)
    generator = MoleculeGenerator(
        _config(
            tmp_path,
            protein_sequence=None,
            protein_id="CHEMBL1",
            dataset_path=str(data),
        )
    )
    with pytest.raises(ValueError, match="No rows found"):
        generator._resolve_protein_and_references()


def test_cli_and_meta_entrypoint_do_not_expose_affinity_prediction(tmp_path):
    assert set(COMMANDS) == {"train", "generate"}
    with pytest.raises(SystemExit):
        parse_arguments(
            [
                "--model_file",
                str(tmp_path / "model"),
                "--protein_sequence",
                "MKT",
                "--prediction_model_file",
                str(tmp_path / "legacy"),
            ]
        )


def test_generation_cli_rejects_invalid_sampling_values(tmp_path):
    with pytest.raises(ValueError, match="top_p"):
        parse_arguments(
            [
                "--model_file",
                str(tmp_path / "model"),
                "--protein_sequence",
                "MKT",
                "--top_p",
                "1.5",
            ]
        )


def test_saved_config_override_uses_known_generation_fields_only(tmp_path, monkeypatch):
    monkeypatch.setattr(MoleculeGenerator, "_load_components", lambda self: None)
    monkeypatch.setattr(
        "prot2mol.inference.produce_molecules.load_saved_model_config",
        lambda path, logger=None: {
            "decoder_type": "molgen",
            "decoder_model_id": "zjunlp/MolGen-large",
            "max_mol_len": 128,
            "pchembl_tf_hidden_dim": 640,
        },
    )
    generator = MoleculeGenerator(_config(tmp_path))
    generator._apply_saved_config()
    assert generator.config.decoder_type == "molgen"
    assert generator.config.max_mol_len == 128
    assert not hasattr(generator.config, "pchembl_tf_hidden_dim")
