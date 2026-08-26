"""Within-protein molecular diversity shaping for grouped GRPO rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from ..chem.utils import get_mol


@dataclass(frozen=True)
class InternalDiversityResult:
    """Per-sample diversity diagnostics aligned with the original batch."""

    penalty_factor: np.ndarray
    diversity_score: np.ndarray
    combined_tanimoto_similarity: np.ndarray
    mean_tanimoto_similarity: np.ndarray
    max_tanimoto_similarity: np.ndarray
    exact_duplicate: np.ndarray
    comparable: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "diversity_penalty_factor": self.penalty_factor,
            "diversity_score": self.diversity_score,
            "combined_tanimoto_similarity": self.combined_tanimoto_similarity,
            "mean_tanimoto_similarity": self.mean_tanimoto_similarity,
            "max_tanimoto_similarity": self.max_tanimoto_similarity,
            "exact_duplicate": self.exact_duplicate,
            "diversity_comparable": self.comparable,
        }


@dataclass(frozen=True)
class ReferenceSimilarityResult:
    """Per-sample ECFP4 comparisons against target-specific references."""

    canonical_smiles: list[str]
    novel: np.ndarray
    max_tanimoto_similarity: np.ndarray
    max_scaffold_tanimoto_similarity: np.ndarray
    scaffold_comparable: np.ndarray
    reference_count: int
    reference_scaffold_count: int


def _murcko_scaffold_smiles(molecule: Chem.Mol) -> str | None:
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
        scaffold_smiles = Chem.MolToSmiles(scaffold)
    except (RuntimeError, ValueError):
        return None
    return scaffold_smiles or None


def reference_ecfp4_similarity(
    smiles: Sequence[str],
    valid_mask: Sequence[bool],
    reference_smiles: Sequence[str],
    *,
    morgan_bits: int = 2_048,
) -> ReferenceSimilarityResult:
    """Compare valid molecules with target references using radius-2 Morgan FPs.

    Full-molecule and Bemis-Murcko-scaffold similarities are the maximum
    Tanimoto similarity to any target reference. Arrays remain aligned with the
    input; invalid or scaffold-unavailable rows carry zero and are identified
    by the validity/scaffold masks.
    """

    if len(smiles) != len(valid_mask):
        raise ValueError("SMILES and reference-similarity validity masks must align")
    if morgan_bits < 1:
        raise ValueError("morgan_bits must be positive")

    canonical_references: dict[str, Chem.Mol] = {}
    for value in reference_smiles:
        molecule = get_mol(value)
        if molecule is not None:
            canonical_references.setdefault(Chem.MolToSmiles(molecule), molecule)

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=morgan_bits,
    )
    reference_fingerprints = [
        generator.GetFingerprint(molecule)
        for molecule in canonical_references.values()
    ]
    reference_scaffolds: dict[str, Chem.Mol] = {}
    for molecule in canonical_references.values():
        scaffold_smiles = _murcko_scaffold_smiles(molecule)
        if scaffold_smiles is None:
            continue
        scaffold = get_mol(scaffold_smiles)
        if scaffold is not None:
            reference_scaffolds.setdefault(scaffold_smiles, scaffold)
    reference_scaffold_fingerprints = [
        generator.GetFingerprint(molecule)
        for molecule in reference_scaffolds.values()
    ]

    canonical_smiles = [""] * len(smiles)
    novel = np.zeros(len(smiles), dtype=np.float32)
    maximum = np.zeros(len(smiles), dtype=np.float32)
    scaffold_maximum = np.zeros(len(smiles), dtype=np.float32)
    scaffold_comparable = np.zeros(len(smiles), dtype=np.float32)
    reference_set = set(canonical_references)

    for index, is_valid in enumerate(valid_mask):
        if not is_valid:
            continue
        molecule = get_mol(smiles[index])
        if molecule is None:
            raise ValueError(
                "Reference similarity received a molecule marked valid but "
                f"RDKit could not parse sample {index}"
            )
        canonical = Chem.MolToSmiles(molecule)
        canonical_smiles[index] = canonical
        if reference_fingerprints:
            novel[index] = float(canonical not in reference_set)
            maximum[index] = float(
                max(
                    DataStructs.BulkTanimotoSimilarity(
                        generator.GetFingerprint(molecule),
                        reference_fingerprints,
                    )
                )
            )

        scaffold_smiles = _murcko_scaffold_smiles(molecule)
        if scaffold_smiles is None or not reference_scaffold_fingerprints:
            continue
        scaffold = get_mol(scaffold_smiles)
        if scaffold is None:
            continue
        scaffold_maximum[index] = float(
            max(
                DataStructs.BulkTanimotoSimilarity(
                    generator.GetFingerprint(scaffold),
                    reference_scaffold_fingerprints,
                )
            )
        )
        scaffold_comparable[index] = 1.0

    return ReferenceSimilarityResult(
        canonical_smiles=canonical_smiles,
        novel=novel,
        max_tanimoto_similarity=maximum,
        max_scaffold_tanimoto_similarity=scaffold_maximum,
        scaffold_comparable=scaffold_comparable,
        reference_count=len(canonical_references),
        reference_scaffold_count=len(reference_scaffolds),
    )


def internal_diversity_factors(
    smiles: Sequence[str],
    valid_mask: Sequence[bool],
    *,
    group_size: int,
    reward_weight: float = 0.5,
    mean_similarity_weight: float = 0.5,
    duplicate_penalty_factor: float = 0.0,
    morgan_radius: int = 2,
    morgan_bits: int = 2048,
) -> InternalDiversityResult:
    """Calculate continuous per-molecule diversity factors within each group."""

    if len(smiles) != len(valid_mask):
        raise ValueError("SMILES and diversity validity masks must align")
    if group_size < 2 or len(smiles) % group_size:
        raise ValueError("Diversity batches must contain complete groups")
    if not 0.0 <= reward_weight <= 1.0:
        raise ValueError("reward_weight must be in [0, 1]")
    if not 0.0 <= mean_similarity_weight <= 1.0:
        raise ValueError("mean_similarity_weight must be in [0, 1]")
    if not 0.0 <= duplicate_penalty_factor <= 1.0:
        raise ValueError("duplicate_penalty_factor must be in [0, 1]")
    if morgan_radius < 1 or morgan_bits < 8:
        raise ValueError("Morgan fingerprint settings are invalid")

    length = len(smiles)
    penalty = np.zeros(length, dtype=np.float32)
    diversity_score = np.zeros(length, dtype=np.float32)
    combined_similarity = np.zeros(length, dtype=np.float32)
    mean_similarity = np.zeros(length, dtype=np.float32)
    max_similarity = np.zeros(length, dtype=np.float32)
    exact_duplicate = np.zeros(length, dtype=np.float32)
    comparable = np.zeros(length, dtype=np.float32)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=morgan_radius,
        fpSize=morgan_bits,
    )

    for start in range(0, length, group_size):
        stop = start + group_size
        valid_indices = [
            index for index in range(start, stop) if bool(valid_mask[index])
        ]
        if not valid_indices:
            continue
        canonical_counts: dict[str, int] = {}
        fingerprints = []
        canonical_smiles = []
        for index in valid_indices:
            molecule = get_mol(smiles[index])
            if molecule is None:
                raise ValueError(
                    "Diversity shaping received a molecule marked valid but RDKit "
                    f"could not parse sample {index}"
                )
            fingerprints.append(generator.GetFingerprint(molecule))
            canonical = Chem.MolToSmiles(molecule)
            canonical_smiles.append(canonical)
            canonical_counts[canonical] = canonical_counts.get(canonical, 0) + 1
        for index, canonical in zip(valid_indices, canonical_smiles):
            exact_duplicate[index] = float(canonical_counts[canonical] > 1)

        if len(valid_indices) == 1:
            penalty[valid_indices[0]] = 1.0
            diversity_score[valid_indices[0]] = 1.0
            continue

        count = len(valid_indices)
        similarities = np.eye(count, dtype=np.float64)
        for left in range(count):
            for right in range(left + 1, count):
                value = float(
                    DataStructs.TanimotoSimilarity(
                        fingerprints[left],
                        fingerprints[right],
                    )
                )
                similarities[left, right] = value
                similarities[right, left] = value

        for offset, index in enumerate(valid_indices):
            others = np.delete(similarities[offset], offset)
            sample_mean = float(others.mean())
            sample_max = float(others.max())
            sample_combined = (
                mean_similarity_weight * sample_mean
                + (1.0 - mean_similarity_weight) * sample_max
            )
            sample_diversity = 1.0 - sample_combined
            sample_factor = (
                (1.0 - reward_weight) + reward_weight * sample_diversity
            )
            if exact_duplicate[index]:
                sample_factor = duplicate_penalty_factor
            penalty[index] = sample_factor
            diversity_score[index] = sample_diversity
            combined_similarity[index] = sample_combined
            mean_similarity[index] = sample_mean
            max_similarity[index] = sample_max
            comparable[index] = 1.0

    return InternalDiversityResult(
        penalty_factor=penalty,
        diversity_score=diversity_score,
        combined_tanimoto_similarity=combined_similarity,
        mean_tanimoto_similarity=mean_similarity,
        max_tanimoto_similarity=max_similarity,
        exact_duplicate=exact_duplicate,
        comparable=comparable,
    )


def scaffold_diversity_summary(
    smiles: Sequence[str],
    valid_mask: Sequence[bool],
) -> dict[str, float]:
    """Summarize non-empty Bemis-Murcko scaffolds among valid molecules."""

    if len(smiles) != len(valid_mask):
        raise ValueError("SMILES and scaffold validity masks must align")
    valid_count = 0
    scaffolds: list[str] = []
    for index, is_valid in enumerate(valid_mask):
        if not is_valid:
            continue
        molecule = get_mol(smiles[index])
        if molecule is None:
            raise ValueError(
                "Scaffold metrics received a molecule marked valid but RDKit "
                f"could not parse sample {index}"
            )
        valid_count += 1
        scaffold_smiles = _murcko_scaffold_smiles(molecule)
        # RDKit can parse some unusual organometallic molecules but fail later
        # while updating the property cache for Murcko extraction. The molecule
        # still contributes to other metrics; only its scaffold is unavailable.
        if scaffold_smiles is None:
            continue
        if scaffold_smiles:
            scaffolds.append(scaffold_smiles)
    return {
        "scaffold_available_fraction": (
            len(scaffolds) / valid_count if valid_count else 0.0
        ),
        "scaffold_unique_fraction": (
            len(set(scaffolds)) / len(scaffolds) if scaffolds else 0.0
        ),
    }
