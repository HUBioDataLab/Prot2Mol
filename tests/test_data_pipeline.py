import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset

from conftest import DummyBatchTokenizer
from prot2mol.data.pipeline import (
    extract_smiles_list,
    find_molecule_column,
    get_processed_stats_path,
    has_matching_precomputed_split,
    load_processed_stats,
    split_train_eval_dataset,
    tokenize_molecule_batch,
    tokenize_protein_batch,
    tokenize_protein_sequences_for_inference,
    tokenize_selfies_for_inference,
    to_selfies_list,
)


def test_split_train_eval_dataset_random():
    ds = Dataset.from_dict({"x": list(range(100))})
    train, test = split_train_eval_dataset(ds, split_mode="random", split_ratio=0.2, split_seed=7)
    assert len(train) + len(test) == 100
    assert len(test) == 20


def test_split_train_eval_dataset_aid_no_overlap():
    ds = Dataset.from_dict(
        {
            "AID": ["A1", "A1", "A2", "A2", "A3", "A3", "A4", "A4"],
            "x": list(range(8)),
        }
    )
    train, test = split_train_eval_dataset(ds, split_mode="aid", split_ratio=0.25, split_seed=13)
    train_aids = set(train["AID"])
    test_aids = set(test["AID"])
    assert train_aids.isdisjoint(test_aids)
    assert len(test_aids) >= 1


def test_split_train_eval_dataset_boundary_ratios():
    ds = Dataset.from_dict({"x": list(range(4))})

    train, test = split_train_eval_dataset(ds, split_ratio=0.0)
    assert train["x"] == [0, 1, 2, 3]
    assert len(test) == 0

    train, test = split_train_eval_dataset(ds, split_ratio=1.0)
    assert len(train) == 0
    assert test["x"] == [0, 1, 2, 3]


def test_processed_stats_loading_and_validation(tmp_path):
    dataset_path = tmp_path / "binding.data.csv"
    cache_dir = tmp_path / "cache"
    stats_path = get_processed_stats_path(str(dataset_path), cache_dir=str(cache_dir))
    os_path = Path(stats_path)
    os_path.parent.mkdir(parents=True)

    expected = {"eval_split": "aid", "eval_split_ratio": 0.2, "split_seed": 7}
    os_path.write_text(json.dumps(expected))
    assert load_processed_stats(str(dataset_path), cache_dir=str(cache_dir)) == expected

    os_path.write_text("[]")
    assert load_processed_stats(str(dataset_path), cache_dir=str(cache_dir)) is None

    os_path.write_text("not json")
    assert load_processed_stats(str(dataset_path), cache_dir=str(cache_dir)) is None


def test_matching_precomputed_split_requires_exact_metadata():
    dataset = {"train": object(), "test": object()}
    stats = {"eval_split": "aid", "eval_split_ratio": 0.2, "split_seed": 7}

    assert has_matching_precomputed_split(dataset, stats, "aid", 0.2, 7)
    assert not has_matching_precomputed_split(dataset, stats, "random", 0.2, 7)
    assert not has_matching_precomputed_split(dataset, stats, "aid", 0.1, 7)
    assert not has_matching_precomputed_split(dataset, stats, "aid", 0.2, 8)
    assert not has_matching_precomputed_split({"train": object()}, stats, "aid", 0.2, 7)


def test_extract_smiles_list_from_dataframe_selfies():
    df = pd.DataFrame({"Compound_SELFIES": ["[C]", "[O]", "", None]})
    smiles = extract_smiles_list(df, drop_invalid=True)
    assert "C" in smiles
    assert "O" in smiles


def test_extract_smiles_list_from_hf_smiles():
    ds = Dataset.from_dict({"Compound_SMILES": ["CCO", "N"]})
    smiles = extract_smiles_list(ds, drop_invalid=False)
    assert smiles == ["CCO", "N"]


def test_find_molecule_column_and_to_selfies_list():
    col, is_selfies = find_molecule_column(["id", "compound_smiles"])
    assert col == "compound_smiles"
    assert is_selfies is False

    converted = to_selfies_list(["CCO", "not_a_smiles"], is_selfies=False)
    assert converted[0]
    assert converted[1] == "[nop]"

    passthrough = to_selfies_list(["[C]", ""], is_selfies=True)
    assert passthrough == ["[C]", "[nop]"]


def test_to_selfies_list_normalizes_whitespace_for_selfies():
    out = to_selfies_list(
        ["[C] [C] [=C] [Branch1] [C] [O] [C]", "[N]\t[O]\n[C]", "   "],
        is_selfies=True,
    )
    assert out[0] == "[C][C][=C][Branch1][C][O][C]"
    assert out[1] == "[N][O][C]"
    assert out[2] == "[nop]"


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


def test_tokenize_molecule_batch_masks_pad_and_sets_train_flags():
    tokenizer = DummyBatchTokenizer(pad_token_id=0)
    out = tokenize_molecule_batch(
        batch={
            "Compound_SELFIES": ["[C]", "[C][O]"],
            "pchembl_value_Median": [5.0, 7.0],
        },
        mol_tokenizer=tokenizer,
        max_mol_len=8,
        pchembl_mean=6.0,
        pchembl_std=1.0,
        pchembl_threshold=6.0,
        train_pchembl_head=True,
    )

    assert torch.equal(out["train_lm"], torch.tensor([False, True]))
    pad_positions = out["mol_input_ids"] == tokenizer.pad_token_id
    assert torch.all(out["labels"][pad_positions] == -100)
    assert torch.all(out["labels"][~pad_positions] != -100)
    assert torch.allclose(out["pchembl_values"], torch.tensor([-1.0, 1.0]))


def test_tokenize_molecule_batch_without_pchembl_head_defaults():
    tokenizer = DummyBatchTokenizer()
    out = tokenize_molecule_batch(
        batch={
            "Compound_SELFIES": ["[C]", "[O]"],
            "pchembl_value_Median": [1.0, 9.0],
        },
        mol_tokenizer=tokenizer,
        max_mol_len=8,
        pchembl_mean=6.0,
        pchembl_std=2.0,
        pchembl_threshold=6.0,
        train_pchembl_head=False,
    )

    assert torch.equal(out["train_lm"], torch.tensor([True, True]))
    assert torch.allclose(out["pchembl_values"], torch.zeros(2))


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
    mol_ids, mol_mask = tokenize_selfies_for_inference(
        selfies_list=["[C]", "[O]"],
        mol_tokenizer=tokenizer,
        max_mol_len=5,
        device=device,
    )

    assert prot_ids.shape == (2, 5) and prot_mask.shape == (2, 5)
    assert mol_ids.shape == (2, 5) and mol_mask.shape == (2, 5)
    assert prot_ids.device.type == "cpu"
    assert mol_ids.device.type == "cpu"
