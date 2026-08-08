import pandas as pd
import pytest
import torch
from datasets import Dataset

from conftest import DummyBatchTokenizer
from prot2mol.data.pipeline import (
    extract_smiles_list,
    tokenize_molecule_batch,
    tokenize_protein_batch,
    tokenize_protein_sequences_for_inference,
)


def test_extract_smiles_list_from_dataframe_selfies():
    df = pd.DataFrame({"Compound_SELFIES": ["[C]", "[O]", "", None]})
    smiles = extract_smiles_list(df, drop_invalid=True)
    assert "C" in smiles
    assert "O" in smiles


def test_extract_smiles_list_from_hf_smiles():
    ds = Dataset.from_dict({"Compound_SMILES": ["CCO", "N"]})
    smiles = extract_smiles_list(ds, drop_invalid=False)
    assert smiles == ["CCO", "N"]


def test_tokenize_protein_batch_shapes_and_keys():
    tokenizer = DummyBatchTokenizer()
    out = tokenize_protein_batch(
        batch={"Target_FASTA": ["MKT", "GGAA"]},
        prot_tokenizer=tokenizer,
        prot_emb_model="prot_t5",
        prot_max_length=6,
    )

    assert set(out.keys()) == {"prot_input_ids", "prot_attention_mask"}
    assert out["prot_input_ids"].shape == (2, 6)
    assert out["prot_attention_mask"].shape == (2, 6)
    assert out["prot_input_ids"].dtype == torch.long
    assert out["prot_attention_mask"].dtype == torch.long


def test_tokenize_molecule_batch_masks_pad_and_trains_every_row():
    tokenizer = DummyBatchTokenizer(pad_token_id=0)
    out = tokenize_molecule_batch(
        batch={
            "Compound_SELFIES": ["[C]", "[C][O]"],
        },
        mol_tokenizer=tokenizer,
        max_mol_len=8,
    )

    pad_positions = out["mol_input_ids"] == tokenizer.pad_token_id
    assert torch.all(out["labels"][pad_positions] == -100)
    assert torch.all(out["labels"][~pad_positions] != -100)
    assert "pchembl_values" not in out


def test_tokenize_molecule_batch_ignores_unrelated_affinity_columns():
    tokenizer = DummyBatchTokenizer()
    out = tokenize_molecule_batch(
        batch={
            "Compound_SELFIES": ["[C]", "[O]"],
            "pchembl_value_Median": [1.0, 9.0],
        },
        mol_tokenizer=tokenizer,
        max_mol_len=8,
    )

    assert set(out) == {"mol_input_ids", "mol_attention_mask", "labels"}


def test_inference_tokenizers_return_tensors_on_target_device():
    tokenizer = DummyBatchTokenizer()
    device = torch.device("cpu")

    prot_ids, prot_mask = tokenize_protein_sequences_for_inference(
        sequences=["MKT", "GG"],
        prot_tokenizer=tokenizer,
        prot_emb_model="prot_t5",
        prot_max_length=5,
        device=device,
    )
    assert prot_ids.shape == (2, 5) and prot_mask.shape == (2, 5)
    assert prot_ids.device.type == "cpu"


def test_tokenizers_reject_overlength_inputs_instead_of_truncating():
    tokenizer = DummyBatchTokenizer()

    with pytest.raises(ValueError, match="does not silently truncate"):
        tokenize_protein_sequences_for_inference(
            sequences=["M" * 7],
            prot_tokenizer=tokenizer,
            prot_emb_model="esm2",
            prot_max_length=8,
        )

    with pytest.raises(ValueError, match="rather than truncating the target"):
        tokenize_molecule_batch(
            batch={"compound_selfies": ["[C]" * 7]},
            mol_tokenizer=tokenizer,
            max_mol_len=8,
        )
