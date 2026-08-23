import math

import pytest
import torch

from prot2mol.rewards import property_shaping
from prot2mol.rewards.property_shaping import (
    TargetActivePropertyStats,
    TargetPropertyShapedActivityScorer,
)


class FakeActivityScorer(torch.nn.Module):
    protein_representation = "structure_aware"
    molecule_representation = "selfies"

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, proteins, molecules):
        self.calls.append((list(proteins), list(molecules)))
        return torch.tensor([0.8, 0.6], dtype=torch.float32)


def test_property_shaping_uses_soft_z_excess_and_zeroes_invalid(monkeypatch):
    monkeypatch.setattr(
        property_shaping,
        "molecular_property_rows",
        lambda smiles: [
            {"qed": 0.5, "logp": 3.0, "sas": 3.0},
            {"qed": 0.5, "logp": 6.0, "sas": 5.0},
            {"qed": None, "logp": None, "sas": None},
        ],
    )
    base = FakeActivityScorer()
    scorer = TargetPropertyShapedActivityScorer(
        base,
        {
            "Aa": TargetActivePropertyStats(
                active_count=100,
                logp_mean=3.0,
                logp_std=1.0,
                sas_mean=3.0,
                sas_std=0.5,
            )
        },
        allowed_sigma=2.0,
        penalty_strength=0.5,
    )

    rewards = scorer(["Aa", "Aa", "Aa"], ["[C]", "[O]", "invalid"])
    diagnostics = scorer.last_diagnostics()

    assert rewards.tolist() == pytest.approx(
        [0.8, 0.6 * math.exp(-2.5), 0.0]
    )
    assert base.calls == [(["Aa", "Aa"], ["[C]", "[O]"])]
    assert diagnostics["logp_excess_z"].tolist() == pytest.approx([0.0, 1.0, 0.0])
    assert diagnostics["sas_excess_z"].tolist() == pytest.approx([0.0, 2.0, 0.0])
    assert diagnostics["property_penalty_factor"].tolist() == pytest.approx(
        [1.0, math.exp(-2.5), 0.0]
    )
    assert diagnostics["property_valid"].tolist() == [1.0, 1.0, 0.0]
    assert scorer.training is False


def test_property_shaping_requires_target_statistics():
    scorer = TargetPropertyShapedActivityScorer(
        FakeActivityScorer(),
        {
            "Aa": TargetActivePropertyStats(
                active_count=2,
                logp_mean=2.0,
                logp_std=1.0,
                sas_mean=3.0,
                sas_std=1.0,
            )
        },
    )

    with pytest.raises(ValueError, match="Missing target"):
        scorer(["Bb"], ["[C]"])
