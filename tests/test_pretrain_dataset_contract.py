import json
from types import SimpleNamespace

import pytest
from datasets import Dataset

from prot2mol.training.pretrain import TrainingScript


def _split(cluster, labels=(1, 3, -100, -100)):
    return Dataset.from_dict(
        {
            "protein_cluster_50": [cluster],
            "protein_sequence": [f"SEQ_{cluster}"],
            "prot_input_ids": [[1, 2, 0, 0]],
            "prot_attention_mask": [[1, 1, 0, 0]],
            "labels": [list(labels)],
        }
    )


def _script(train, validation):
    script = TrainingScript.__new__(TrainingScript)
    script.train_data = train
    script.validation_data = validation
    script.mol_tokenizer = SimpleNamespace(__len__=lambda self: 16)
    # Special methods are resolved on the class, not the instance.
    script.mol_tokenizer = type("Tokenizer", (), {"__len__": lambda self: 16})()
    return script


def test_training_contract_accepts_disjoint_right_padded_splits():
    _script(_split("train"), _split("validation"))._validate_tokenized_contract()


def test_training_contract_rejects_cluster_leakage():
    with pytest.raises(ValueError, match="MMseqs50 leakage"):
        _script(_split("same"), _split("same"))._validate_tokenized_contract()


def test_training_contract_rejects_left_or_interior_padding():
    with pytest.raises(ValueError, match="right padded"):
        _script(_split("train", labels=(-100, 1, 3, -100)), _split("validation"))._validate_tokenized_contract()


def test_training_contract_rejects_mismatched_preprocessing_manifest(tmp_path):
    manifest = {
        "config": {
            "prot_emb_model": "esm2",
            "max_mol_len": 256,
            "prot_max_length": 1024,
        },
        "protein_model_id_resolved": "facebook/esm2_t33_650M_UR50D",
        "decoder_model_id_resolved": "wrong/decoder",
        "molecule_padding_side": "right",
        "protein_padding_side": "right",
        "molecule_tokenizer_coverage": {
            "splits": {
                "train": {"unknown_tokens": 0},
                "validation": {"unknown_tokens": 0},
            }
        },
        "protein_tokenizer_coverage": {
            "splits": {
                "train": {"unknown_tokens": 0},
                "validation": {"unknown_tokens": 0},
            }
        },
    }
    (tmp_path / "preprocessing_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    script = TrainingScript.__new__(TrainingScript)
    script.dataset_path = str(tmp_path)
    script.model_config = {
        "prot_emb_model": "esm2",
        "protein_model_id": "facebook/esm2_t33_650M_UR50D",
        "decoder_model_id": "zjunlp/MolGen-large",
        "max_mol_len": 256,
        "prot_max_length": 1024,
    }

    with pytest.raises(ValueError, match="decoder_model_id_resolved"):
        script._load_and_validate_preprocessing_manifest()
