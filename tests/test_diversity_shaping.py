import numpy as np
import pytest
from rdkit import Chem

from prot2mol.rewards import diversity
from prot2mol.rewards.diversity import (
    internal_diversity_factors,
    scaffold_diversity_summary,
)


def test_identical_group_receives_zero_duplicate_reward():
    result = internal_diversity_factors(
        ["CC"] * 8,
        [True] * 8,
        group_size=8,
        morgan_radius=2,
        morgan_bits=2048,
    )

    assert result.mean_tanimoto_similarity == pytest.approx(np.ones(8))
    assert result.max_tanimoto_similarity == pytest.approx(np.ones(8))
    assert result.combined_tanimoto_similarity == pytest.approx(np.ones(8))
    assert result.diversity_score == pytest.approx(np.zeros(8))
    assert result.penalty_factor == pytest.approx(np.zeros(8))
    assert result.exact_duplicate == pytest.approx(np.ones(8))
    assert result.comparable == pytest.approx(np.ones(8))


def test_diversity_is_group_local_and_single_valid_sample_is_neutral():
    result = internal_diversity_factors(
        ["CC", "CC", "N", ""],
        [True, True, True, False],
        group_size=2,
    )

    assert result.penalty_factor[:2] == pytest.approx([0.0, 0.0])
    assert result.penalty_factor[2] == pytest.approx(1.0)
    assert result.comparable[2] == pytest.approx(0.0)
    assert result.exact_duplicate[2] == pytest.approx(0.0)
    assert result.penalty_factor[3] == pytest.approx(0.0)


def test_diversity_factor_continuously_combines_mean_and_nearest_similarity():
    result = internal_diversity_factors(
        ["CC", "CO"],
        [True, True],
        group_size=2,
        reward_weight=0.5,
        mean_similarity_weight=0.5,
    )

    assert result.mean_tanimoto_similarity == pytest.approx(
        result.max_tanimoto_similarity
    )
    assert result.combined_tanimoto_similarity == pytest.approx(
        result.mean_tanimoto_similarity
    )
    assert result.diversity_score == pytest.approx(
        1.0 - result.combined_tanimoto_similarity
    )
    assert result.penalty_factor == pytest.approx(
        0.5 + 0.5 * result.diversity_score
    )


def test_scaffold_metrics_ignore_acyclic_molecules_and_invalid_rows():
    summary = scaffold_diversity_summary(
        ["c1ccccc1", "Cc1ccccc1", "CC", ""],
        [True, True, True, False],
    )

    assert summary["scaffold_available_fraction"] == pytest.approx(2.0 / 3.0)
    assert summary["scaffold_unique_fraction"] == pytest.approx(0.5)


def test_scaffold_metrics_treat_rdkit_extraction_failures_as_unavailable(
    monkeypatch,
):
    def fail_scaffold_extraction(_molecule):
        raise Chem.rdchem.AtomValenceException("unsupported scaffold valence")

    monkeypatch.setattr(
        diversity.MurckoScaffold,
        "GetScaffoldForMol",
        fail_scaffold_extraction,
    )

    summary = scaffold_diversity_summary(["c1ccccc1"], [True])

    assert summary["scaffold_available_fraction"] == pytest.approx(0.0)
    assert summary["scaffold_unique_fraction"] == pytest.approx(0.0)


def test_diversity_rejects_misaligned_or_incomplete_groups():
    with pytest.raises(ValueError, match="must align"):
        internal_diversity_factors(["CC"], [], group_size=2)
    with pytest.raises(ValueError, match="complete groups"):
        internal_diversity_factors(["CC", "CO", "CN"], [True] * 3, group_size=2)
    with pytest.raises(ValueError, match="marked valid"):
        internal_diversity_factors(["not-smiles", "CC"], [True, True], group_size=2)
    with pytest.raises(ValueError, match="reward_weight"):
        internal_diversity_factors(["CC", "CO"], [True, True], group_size=2, reward_weight=1.1)
