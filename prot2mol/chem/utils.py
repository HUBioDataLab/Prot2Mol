"""Small, explicit cheminformatics helpers for generator evaluation."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable

import numpy as np
import pandas as pd
import selfies as sf
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, QED, RDConfig, rdFingerprintGenerator

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402


_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def get_mol(smiles_or_mol):
    """Return a sanitized RDKit molecule, or ``None`` for an invalid value."""

    if isinstance(smiles_or_mol, Chem.Mol):
        return smiles_or_mol
    if not isinstance(smiles_or_mol, str) or not smiles_or_mol.strip():
        return None
    molecule = Chem.MolFromSmiles(smiles_or_mol.strip())
    if molecule is None:
        return None
    try:
        Chem.SanitizeMol(molecule)
    except (ValueError, RuntimeError):
        return None
    return molecule


def canonic_smiles(smiles_or_mol):
    molecule = get_mol(smiles_or_mol)
    return Chem.MolToSmiles(molecule) if molecule is not None else None


def decode_selfies_list(selfies_list: Iterable[object]) -> list[str | None]:
    """Decode SELFIES strings to SMILES while preserving row alignment."""

    decoded: list[str | None] = []
    for value in selfies_list:
        if not isinstance(value, str):
            decoded.append(None)
            continue
        try:
            smiles = sf.decoder("".join(value.split()))
        except sf.DecoderError:
            smiles = None
        decoded.append(smiles or None)
    return decoded


def canonicalize_smiles_list(
    smiles_list: Iterable[object],
    drop_invalid: bool = False,
) -> list[str]:
    """Canonicalize SMILES, optionally retaining invalid rows as empty strings."""

    canonical = [canonic_smiles(value) or "" for value in smiles_list]
    return [value for value in canonical if value] if drop_invalid else canonical


def molecular_property_summary(smiles_list: Iterable[object]) -> dict[str, float]:
    """Summarize QED, synthetic-accessibility score, and logP.

    Invalid inputs are excluded. ``count`` makes the denominator explicit so a
    caller can distinguish an empty/invalid batch from a real batch whose mean
    happens to be zero.
    """

    molecules = [get_mol(value) for value in smiles_list]
    molecules = [molecule for molecule in molecules if molecule is not None]
    values = {
        "qed": [float(QED.qed(molecule)) for molecule in molecules],
        "sas": [float(sascorer.calculateScore(molecule)) for molecule in molecules],
        "logp": [float(Crippen.MolLogP(molecule)) for molecule in molecules],
    }
    summary = {"count": float(len(molecules))}
    for name, property_values in values.items():
        array = np.asarray(property_values, dtype=np.float64)
        summary.update(
            {
                f"{name}_mean": float(array.mean()) if len(array) else 0.0,
                f"{name}_std": float(array.std()) if len(array) else 0.0,
                f"{name}_min": float(array.min()) if len(array) else 0.0,
                f"{name}_max": float(array.max()) if len(array) else 0.0,
            }
        )
    return summary


def _reference_smiles(data_source) -> list[str]:
    if data_source is None:
        return []
    if isinstance(data_source, (list, tuple, set, np.ndarray, pd.Series)):
        return canonicalize_smiles_list(data_source, drop_invalid=True)
    columns = getattr(data_source, "columns", getattr(data_source, "column_names", ()))
    for column in ("smiles", "Compound_SMILES"):
        if column in columns:
            return canonicalize_smiles_list(data_source[column], drop_invalid=True)
    for column in ("compound_selfies", "Compound_SELFIES"):
        if column in columns:
            return canonicalize_smiles_list(
                decode_selfies_list(data_source[column]),
                drop_invalid=True,
            )
    return []


def _fingerprints(molecules: list[Chem.Mol]) -> np.ndarray:
    vectors = np.zeros((len(molecules), 1024), dtype=np.uint8)
    for index, molecule in enumerate(molecules):
        fingerprint = _MORGAN_GENERATOR.GetFingerprint(molecule)
        DataStructs.ConvertToNumpyArray(fingerprint, vectors[index])
    return vectors


def _maximum_tanimoto(
    reference_vectors: np.ndarray,
    query_vectors: np.ndarray,
    batch_size: int = 2_048,
) -> np.ndarray:
    if len(reference_vectors) == 0 or len(query_vectors) == 0:
        return np.zeros(len(query_vectors), dtype=np.float32)
    query = np.asarray(query_vectors, dtype=np.float32)
    query_sums = query.sum(axis=1)
    maxima = np.zeros(len(query), dtype=np.float32)
    for start in range(0, len(reference_vectors), batch_size):
        reference = np.asarray(
            reference_vectors[start : start + batch_size],
            dtype=np.float32,
        )
        intersections = reference @ query.T
        unions = reference.sum(axis=1, keepdims=True) + query_sums[None, :] - intersections
        similarities = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections),
            where=unions > 0,
        )
        maxima = np.maximum(maxima, similarities.max(axis=0))
    return maxima


def _internal_diversity(vectors: np.ndarray) -> float:
    count = len(vectors)
    if count < 2:
        return 0.0
    values = np.asarray(vectors, dtype=np.float32)
    intersections = values @ values.T
    sums = values.sum(axis=1)
    unions = sums[:, None] + sums[None, :] - intersections
    similarities = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections),
        where=unions > 0,
    )
    off_diagonal = similarities[~np.eye(count, dtype=bool)]
    return float(1.0 - off_diagonal.mean())


def metrics_calculation(
    predictions,
    references,
    train_data,
    train_vec=None,
    return_details=False,
):
    """Evaluate decoded SELFIES using valid-molecule denominators.

    Validity is measured over all generations. All remaining molecular metrics
    operate only on valid molecules, so invalid generations are not represented
    as zero fingerprints or counted as molecular duplicates.
    """

    decoded = decode_selfies_list(predictions)
    canonical = canonicalize_smiles_list(decoded, drop_invalid=False)
    valid_indices = [index for index, smiles in enumerate(canonical) if smiles]
    valid_smiles = [canonical[index] for index in valid_indices]
    valid_molecules = [get_mol(smiles) for smiles in valid_smiles]
    valid_molecules = [molecule for molecule in valid_molecules if molecule is not None]

    total_count = len(canonical)
    valid_count = len(valid_molecules)
    unique_valid = set(valid_smiles)
    train_smiles = _reference_smiles(train_data)
    eval_smiles = _reference_smiles(references)
    train_set = set(train_smiles)
    eval_set = set(eval_smiles)

    metrics = {
        "validity": float(valid_count / total_count) if total_count else 0.0,
        "uniqueness": float(len(unique_valid) / valid_count) if valid_count else 0.0,
        "novelty_train": (
            float(len(unique_valid - train_set) / len(unique_valid))
            if unique_valid and train_set
            else 0.0
        ),
        "novelty_eval": (
            float(len(unique_valid - eval_set) / len(unique_valid))
            if unique_valid and eval_set
            else 0.0
        ),
        "intdiv": 0.0,
        "similarity_train": 0.0,
        "similarity_eval": 0.0,
        "sa": 0.0,
        "qed": 0.0,
        "logp": 0.0,
    }

    details = pd.DataFrame(
        {
            "smiles": canonical,
            "similarity_eval": [np.nan] * total_count,
            "similarity_train": [np.nan] * total_count,
            "sa": [np.nan] * total_count,
            "qed": [np.nan] * total_count,
            "logp": [np.nan] * total_count,
        }
    )
    if valid_count:
        generated_vectors = _fingerprints(valid_molecules)
        eval_molecules = [get_mol(value) for value in eval_smiles]
        eval_molecules = [molecule for molecule in eval_molecules if molecule is not None]
        eval_similarities = _maximum_tanimoto(
            _fingerprints(eval_molecules),
            generated_vectors,
        )
        training_vectors = (
            np.asarray(train_vec, dtype=np.uint8)
            if train_vec is not None
            else _fingerprints(
                [molecule for molecule in map(get_mol, train_smiles) if molecule is not None]
            )
        )
        train_similarities = _maximum_tanimoto(training_vectors, generated_vectors)
        sa_values = [float(sascorer.calculateScore(molecule)) for molecule in valid_molecules]
        qed_values = [float(QED.qed(molecule)) for molecule in valid_molecules]
        logp_values = [float(Crippen.MolLogP(molecule)) for molecule in valid_molecules]

        metrics.update(
            {
                "intdiv": _internal_diversity(generated_vectors),
                "similarity_train": float(train_similarities.mean()) if len(training_vectors) else 0.0,
                "similarity_eval": float(eval_similarities.mean()) if len(eval_molecules) else 0.0,
                "sa": float(np.mean(sa_values)),
                "qed": float(np.mean(qed_values)),
                "logp": float(np.mean(logp_values)),
            }
        )
        details.loc[valid_indices, "similarity_eval"] = eval_similarities
        details.loc[valid_indices, "similarity_train"] = train_similarities
        details.loc[valid_indices, "sa"] = sa_values
        details.loc[valid_indices, "qed"] = qed_values
        details.loc[valid_indices, "logp"] = logp_values

    return (metrics, details) if return_details else metrics
