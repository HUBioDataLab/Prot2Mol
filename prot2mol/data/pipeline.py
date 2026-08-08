import os
from typing import Optional, Sequence

import torch
from datasets import load_from_disk

from ..core.protein_encoders import format_protein_sequences
from ..chem.utils import canonicalize_smiles_list, decode_selfies_list

def load_processed_dataset(dataset_path: str):
    """Load an explicitly preprocessed train/validation DatasetDict."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Preprocessed dataset not found at: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    missing = {"train", "validation"}.difference(dataset.keys())
    if missing:
        raise ValueError(f"Preprocessed dataset is missing splits: {sorted(missing)}")
    if "test" in dataset:
        raise ValueError("Prot2Mol generation data must not contain a test split")
    return dataset


def _column_name(columns: Sequence[str], *candidates: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Missing required column; expected one of {list(candidates)}")


def extract_smiles_list(data_source, drop_invalid: bool = False, logger=None):
    """Extract SMILES from pandas/HF/list inputs with optional canonicalization."""
    smiles_values = []
    needs_canonicalization = False
    try:
        if hasattr(data_source, "column_names"):
            if "smiles" in data_source.column_names:
                smiles_values = list(data_source["smiles"])
            elif "Compound_SMILES" in data_source.column_names:
                smiles_values = list(data_source["Compound_SMILES"])
            elif "compound_selfies" in data_source.column_names:
                smiles_values = decode_selfies_list(list(data_source["compound_selfies"]))
                needs_canonicalization = True
            elif "Compound_SELFIES" in data_source.column_names:
                smiles_values = decode_selfies_list(list(data_source["Compound_SELFIES"]))
                needs_canonicalization = True
        elif hasattr(data_source, "columns"):
            if "smiles" in data_source.columns:
                smiles_values = data_source["smiles"].tolist()
            elif "Compound_SMILES" in data_source.columns:
                smiles_values = data_source["Compound_SMILES"].tolist()
            elif "compound_selfies" in data_source.columns:
                smiles_values = decode_selfies_list(data_source["compound_selfies"].tolist())
                needs_canonicalization = True
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
    protein_column = _column_name(batch.keys(), "protein_sequence", "Target_FASTA")
    sequences = batch[protein_column]
    max_residues = prot_max_length - 2
    if any(len(str(sequence).strip()) > max_residues for sequence in sequences):
        raise ValueError(
            f"Protein sequence exceeds the {max_residues}-residue context; "
            "rebuild or filter the dataset rather than truncating conditioning input."
        )
    sequence_examples = format_protein_sequences(sequences, prot_emb_model)
    ids = prot_tokenizer(
        sequence_examples,
        add_special_tokens=True,
        truncation=False,
        max_length=prot_max_length,
        padding="max_length",
        return_tensors="pt",
    )
    return {"prot_input_ids": ids["input_ids"], "prot_attention_mask": ids["attention_mask"]}


def tokenize_molecule_batch(
    batch,
    mol_tokenizer,
    max_mol_len: int,
):
    """Tokenize SELFIES and prepare molecule language-model labels."""
    molecule_column = _column_name(batch.keys(), "compound_selfies", "Compound_SELFIES")
    selfies_values = batch[molecule_column]
    max_selfies_tokens = max_mol_len - 2
    if any(str(value).count("[") > max_selfies_tokens for value in selfies_values):
        raise ValueError(
            f"SELFIES target exceeds the {max_selfies_tokens}-token context; "
            "rebuild or filter the dataset rather than truncating the target."
        )
    ids = mol_tokenizer(
        selfies_values,
        add_special_tokens=True,
        truncation=False,
        max_length=max_mol_len,
        padding="max_length",
        return_tensors="pt",
    )

    labels = ids["input_ids"].clone()
    pad_mask = ids["input_ids"] == mol_tokenizer.pad_token_id
    labels[pad_mask] = -100

    return {
        "mol_input_ids": ids["input_ids"],
        "mol_attention_mask": ids["attention_mask"],
        "labels": labels,
    }


def tokenize_protein_sequences_for_inference(
    sequences: Sequence[str],
    prot_tokenizer,
    prot_emb_model: str,
    prot_max_length: int,
    device: Optional[torch.device] = None,
):
    """Batch tokenize protein sequences for inference."""
    max_residues = prot_max_length - 2
    if any(len(str(sequence).strip()) > max_residues for sequence in sequences):
        raise ValueError(
            f"Protein sequence exceeds the {max_residues}-residue inference context; "
            "the model does not silently truncate proteins."
        )
    formatted_sequences = format_protein_sequences(sequences, prot_emb_model)
    ids = prot_tokenizer(
        formatted_sequences,
        add_special_tokens=True,
        max_length=prot_max_length,
        padding="max_length",
        truncation=False,
        return_tensors="pt",
    )
    input_ids = ids["input_ids"]
    attention_mask = ids["attention_mask"]
    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask
