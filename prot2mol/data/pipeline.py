import os
from typing import Iterable, Optional, Sequence, Tuple

import selfies as sf
import torch
from datasets import load_from_disk

from ..core.protein_encoders import format_protein_sequences
from ..chem.utils import canonicalize_smiles_list, decode_selfies_list

MOLECULE_SELFIES_HINTS = (
    "selfies",
    "compound_selfies",
    "generated_selfies",
)
MOLECULE_SMILES_HINTS = (
    "smiles",
    "compound_smiles",
    "generated_smiles",
)


def get_processed_data_path(selfies_path: str, cache_dir: Optional[str] = None) -> str:
    """Return HF cached dataset path for a raw CSV path."""
    effective_cache_dir = cache_dir or os.environ.get("DATASETS_CACHE_DIR", "/gpfs/projects/etur29/atabey/datasets")
    dataset_name = os.path.splitext(os.path.basename(selfies_path))[0]
    return os.path.join(effective_cache_dir, dataset_name)


def load_processed_dataset(selfies_path: str, cache_dir: Optional[str] = None):
    """Load preprocessed dataset from disk cache."""
    processed_data_path = get_processed_data_path(selfies_path, cache_dir=cache_dir)
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(f"Preprocessed dataset not found at: {processed_data_path}")
    return load_from_disk(processed_data_path), processed_data_path


def split_train_eval_dataset(
    full_data,
    split_mode: str = "random",
    split_ratio: float = 0.01,
    split_seed: int = 42,
    num_proc: Optional[int] = None,
    logger=None,
):
    """
    Split a tokenized HF dataset into train/eval partitions.
    Supports random split and AID hold-out split.
    """
    if split_mode == "aid":
        if "AID" not in full_data.column_names:
            if logger is not None:
                logger.warning("AID column not found in dataset. Falling back to random split.")
            split = full_data.train_test_split(test_size=split_ratio, seed=split_seed)
            return split["train"], split["test"]

        import numpy as np

        aids = full_data.unique("AID")
        rng = np.random.RandomState(split_seed)
        rng.shuffle(aids)
        n_holdout = max(1, int(len(aids) * split_ratio))
        holdout_aids = set(aids[:n_holdout])
        if logger is not None:
            logger.info(
                "AID hold-out split: %s AIDs held out (%.3f of %s)",
                n_holdout,
                split_ratio,
                len(aids),
            )
        proc = num_proc if (num_proc is None or num_proc >= 1) else None
        test_data = full_data.filter(lambda x: x["AID"] in holdout_aids, num_proc=proc)
        train_data = full_data.filter(lambda x: x["AID"] not in holdout_aids, num_proc=proc)
        return train_data, test_data

    split = full_data.train_test_split(test_size=split_ratio, seed=split_seed)
    return split["train"], split["test"]


def extract_smiles_list(data_source, drop_invalid: bool = False, logger=None):
    """Extract SMILES from pandas/HF/list inputs with optional canonicalization."""
    smiles_values = []
    needs_canonicalization = False
    try:
        if hasattr(data_source, "column_names"):
            if "Compound_SMILES" in data_source.column_names:
                smiles_values = list(data_source["Compound_SMILES"])
            elif "Compound_SELFIES" in data_source.column_names:
                smiles_values = decode_selfies_list(list(data_source["Compound_SELFIES"]))
                needs_canonicalization = True
        elif hasattr(data_source, "columns"):
            if "Compound_SMILES" in data_source.columns:
                smiles_values = data_source["Compound_SMILES"].tolist()
            elif "Compound_SELFIES" in data_source.columns:
                smiles_values = decode_selfies_list(data_source["Compound_SELFIES"].tolist())
                needs_canonicalization = True
        elif isinstance(data_source, list):
            smiles_values = data_source
        elif logger is not None:
            logger.warning("Unsupported data source for SMILES extraction: %s", type(data_source))
    except Exception as exc:
        if logger is not None:
            logger.warning("Failed to extract SMILES: %s", exc)
        smiles_values = []

    if not smiles_values:
        return []

    cleaned_smiles = []
    for entry in smiles_values:
        if isinstance(entry, str):
            stripped = entry.strip()
            if stripped:
                cleaned_smiles.append(stripped)
        elif isinstance(entry, bytes):
            stripped = entry.decode("utf-8").strip()
            if stripped:
                cleaned_smiles.append(stripped)

    if not cleaned_smiles:
        return []

    if needs_canonicalization:
        return canonicalize_smiles_list(cleaned_smiles, drop_invalid=drop_invalid)
    if drop_invalid:
        return [s for s in cleaned_smiles if s]
    return cleaned_smiles


def tokenize_protein_batch(batch, prot_tokenizer, prot_emb_model: str, prot_max_length: int):
    """Tokenize batch of FASTA sequences for training/preprocessing."""
    sequence_examples = format_protein_sequences(batch["Target_FASTA"], prot_emb_model)
    ids = prot_tokenizer.batch_encode_plus(
        sequence_examples,
        add_special_tokens=True,
        truncation=True,
        max_length=prot_max_length,
        padding="max_length",
        return_tensors="pt",
    )
    return {"prot_input_ids": ids["input_ids"], "prot_attention_mask": ids["attention_mask"]}


def tokenize_molecule_batch(
    batch,
    mol_tokenizer,
    max_mol_len: int,
    pchembl_mean: float,
    pchembl_std: float,
    pchembl_threshold: float,
    train_pchembl_head: bool = True,
):
    """Tokenize SELFIES and prepare labels/pChEMBL/train_lm fields."""
    ids = mol_tokenizer.batch_encode_plus(
        batch["Compound_SELFIES"],
        add_special_tokens=True,
        truncation=True,
        max_length=max_mol_len,
        padding="max_length",
        return_tensors="pt",
    )

    labels = ids["input_ids"].clone()
    pad_mask = ids["input_ids"] == mol_tokenizer.pad_token_id
    labels[pad_mask] = -100

    pchembl_values = batch.get("pchembl_value_Median", [0.0] * len(batch["Compound_SELFIES"]))
    if not train_pchembl_head:
        train_lm_flags = [True] * len(pchembl_values)
        normalized_pchembl = [0.0] * len(pchembl_values)
    else:
        train_lm_flags = [val >= pchembl_threshold for val in pchembl_values]
        normalized_pchembl = [
            (val - pchembl_mean) / (pchembl_std + 1e-8) for val in pchembl_values
        ]

    return {
        "mol_input_ids": ids["input_ids"],
        "mol_attention_mask": ids["attention_mask"],
        "labels": labels,
        "pchembl_values": torch.tensor(normalized_pchembl, dtype=torch.float),
        "train_lm": torch.tensor(train_lm_flags, dtype=torch.bool),
    }


def find_molecule_column(columns: Sequence[str]) -> Tuple[Optional[str], bool]:
    """Find first molecule column. Returns (column_name, is_selfies)."""
    lowered = [(col, col.lower()) for col in columns]
    for raw, low in lowered:
        if any(hint in low for hint in MOLECULE_SELFIES_HINTS):
            return raw, True
    for raw, low in lowered:
        if any(hint in low for hint in MOLECULE_SMILES_HINTS):
            return raw, False
    return None, False


def to_selfies_list(
    molecules: Iterable,
    is_selfies: bool,
    invalid_token: str = "[nop]",
):
    """Convert a molecule iterable into a safe SELFIES list."""
    if is_selfies:
        return [
            mol if isinstance(mol, str) and mol.strip() else invalid_token
            for mol in molecules
        ]

    converted = []
    for smiles in molecules:
        try:
            selfi = sf.encoder(smiles)
            converted.append(selfi if selfi else invalid_token)
        except Exception:
            converted.append(invalid_token)
    return converted


def tokenize_protein_sequences_for_inference(
    sequences: Sequence[str],
    prot_tokenizer,
    prot_emb_model: str,
    prot_max_length: int,
    device: Optional[torch.device] = None,
):
    """Batch tokenize protein sequences for inference."""
    formatted_sequences = format_protein_sequences(sequences, prot_emb_model)
    ids = prot_tokenizer.batch_encode_plus(
        formatted_sequences,
        add_special_tokens=True,
        max_length=prot_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = ids["input_ids"]
    attention_mask = ids["attention_mask"]
    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask


def tokenize_selfies_for_inference(
    selfies_list: Sequence[str],
    mol_tokenizer,
    max_mol_len: int,
    device: Optional[torch.device] = None,
):
    """Batch tokenize SELFIES strings for inference."""
    ids = mol_tokenizer.batch_encode_plus(
        selfies_list,
        add_special_tokens=True,
        truncation=True,
        max_length=max_mol_len,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = ids["input_ids"]
    attention_mask = ids["attention_mask"]
    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask
