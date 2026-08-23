"""Opt-in test of the complete published FusionDTI artifact stack.

Enable with ``PROT2MOL_RUN_FUSIONDTI_LIVE=1``. The first run downloads the
large SaProt and SELFormer encoder checkpoints.
"""

import os

import pytest
import torch

from prot2mol.rewards import FusionDTIActivityScorer


@pytest.mark.skipif(
    os.environ.get("PROT2MOL_RUN_FUSIONDTI_LIVE") != "1",
    reason="set PROT2MOL_RUN_FUSIONDTI_LIVE=1 to load published encoders",
)
def test_published_fusiondti_stack_scores_structure_aware_protein_and_selfies():
    scorer = FusionDTIActivityScorer.from_pretrained(
        dataset="BindingDB",
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=2,
        max_length=32,
    )

    probabilities = scorer(
        ["MdEvLp", "MdEvLp"],
        ["[C]", "[C][O]"],
    )

    assert probabilities.shape == (2,)
    assert torch.isfinite(probabilities).all()
    assert probabilities.ge(0.0).all()
    assert probabilities.le(1.0).all()
    assert probabilities.std(unbiased=False) > 0.0
    assert scorer.training is False
    assert all(not parameter.requires_grad for parameter in scorer.parameters())
