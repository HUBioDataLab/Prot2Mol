import math

import numpy as np
import pytest

from prot2mol.rewards.diversity import internal_diversity_factors


def test_identical_group_receives_expected_soft_diversity_penalty():
    result = internal_diversity_factors(
        ["CC"] * 8,
        [True] * 8,
        group_size=8,
        similarity_threshold=0.4,
        similarity_softness=0.2,
        morgan_radius=2,
        morgan_bits=2048,
    )

    assert result.mean_tanimoto_similarity == pytest.approx(np.ones(8))
    assert result.max_tanimoto_similarity == pytest.approx(np.ones(8))
    assert result.similarity_excess == pytest.approx(np.full(8, 3.0))
    assert result.penalty_factor == pytest.approx(
        np.full(8, math.exp(-4.5))
    )
    assert result.violation == pytest.approx(np.ones(8))
    assert result.exact_duplicate == pytest.approx(np.ones(8))
    assert result.comparable == pytest.approx(np.ones(8))


def test_diversity_is_group_local_and_single_valid_sample_is_neutral():
    result = internal_diversity_factors(
        ["CC", "CC", "N", ""],
        [True, True, True, False],
        group_size=2,
    )

    assert result.penalty_factor[:2] == pytest.approx(
        [math.exp(-4.5), math.exp(-4.5)]
    )
    assert result.penalty_factor[2] == pytest.approx(1.0)
    assert result.comparable[2] == pytest.approx(0.0)
    assert result.exact_duplicate[2] == pytest.approx(0.0)
    assert result.penalty_factor[3] == pytest.approx(0.0)


def test_diversity_rejects_misaligned_or_incomplete_groups():
    with pytest.raises(ValueError, match="must align"):
        internal_diversity_factors(["CC"], [], group_size=2)
    with pytest.raises(ValueError, match="complete groups"):
        internal_diversity_factors(["CC", "CO", "CN"], [True] * 3, group_size=2)
    with pytest.raises(ValueError, match="marked valid"):
        internal_diversity_factors(["not-smiles", "CC"], [True, True], group_size=2)
