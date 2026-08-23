"""Within-protein molecular diversity shaping for grouped GRPO rollouts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from ..chem.utils import get_mol


@dataclass(frozen=True)
class InternalDiversityResult:
    """Per-sample diversity diagnostics aligned with the original batch."""

    penalty_factor: np.ndarray
    mean_tanimoto_similarity: np.ndarray
    max_tanimoto_similarity: np.ndarray
    similarity_excess: np.ndarray
    violation: np.ndarray
    exact_duplicate: np.ndarray
    comparable: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "diversity_penalty_factor": self.penalty_factor,
            "mean_tanimoto_similarity": self.mean_tanimoto_similarity,
            "max_tanimoto_similarity": self.max_tanimoto_similarity,
            "diversity_similarity_excess": self.similarity_excess,
            "diversity_violation": self.violation,
            "exact_duplicate": self.exact_duplicate,
            "diversity_comparable": self.comparable,
        }


def internal_diversity_factors(
    smiles: Sequence[str],
    valid_mask: Sequence[bool],
    *,
    group_size: int,
    similarity_threshold: float = 0.4,
    similarity_softness: float = 0.2,
    morgan_radius: int = 2,
    morgan_bits: int = 2048,
) -> InternalDiversityResult:
    """Calculate mean-pairwise-Tanimoto soft penalties within each group."""

    if len(smiles) != len(valid_mask):
        raise ValueError("SMILES and diversity validity masks must align")
    if group_size < 2 or len(smiles) % group_size:
        raise ValueError("Diversity batches must contain complete groups")
    if not 0.0 <= similarity_threshold < 1.0:
        raise ValueError("similarity_threshold must be in [0, 1)")
    if similarity_softness <= 0.0:
        raise ValueError("similarity_softness must be positive")
    if morgan_radius < 1 or morgan_bits < 8:
        raise ValueError("Morgan fingerprint settings are invalid")

    length = len(smiles)
    penalty = np.zeros(length, dtype=np.float32)
    mean_similarity = np.zeros(length, dtype=np.float32)
    max_similarity = np.zeros(length, dtype=np.float32)
    excess = np.zeros(length, dtype=np.float32)
    violation = np.zeros(length, dtype=np.float32)
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
        for index in valid_indices:
            molecule = get_mol(smiles[index])
            if molecule is None:
                raise ValueError(
                    "Diversity shaping received a molecule marked valid but RDKit "
                    f"could not parse sample {index}"
                )
            fingerprints.append(generator.GetFingerprint(molecule))
            canonical_counts[smiles[index]] = canonical_counts.get(smiles[index], 0) + 1
        for index in valid_indices:
            exact_duplicate[index] = float(canonical_counts[smiles[index]] > 1)

        if len(valid_indices) == 1:
            penalty[valid_indices[0]] = 1.0
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
            sample_excess = max(
                0.0,
                (sample_mean - similarity_threshold) / similarity_softness,
            )
            penalty[index] = math.exp(-0.5 * sample_excess * sample_excess)
            mean_similarity[index] = sample_mean
            max_similarity[index] = sample_max
            excess[index] = sample_excess
            violation[index] = float(sample_mean > similarity_threshold)
            comparable[index] = 1.0

    return InternalDiversityResult(
        penalty_factor=penalty,
        mean_tanimoto_similarity=mean_similarity,
        max_tanimoto_similarity=max_similarity,
        similarity_excess=excess,
        violation=violation,
        exact_duplicate=exact_duplicate,
        comparable=comparable,
    )
