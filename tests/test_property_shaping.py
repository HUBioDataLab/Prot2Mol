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
            {
                "qed": 0.5,
                "logp": 3.0,
                "sas": 3.0,
                "heavy_atom_count": 20.0,
            },
            {
                "qed": 0.5,
                "logp": 6.0,
                "sas": 5.0,
                "heavy_atom_count": 40.0,
            },
            {
                "qed": None,
                "logp": None,
                "sas": None,
                "heavy_atom_count": None,
            },
        ],
    )
    base = FakeActivityScorer()
    scorer = TargetPropertyShapedActivityScorer(
        base,
        {
            "Aa": TargetActivePropertyStats(
                active_count=100,
                qed_mean=0.5,
                qed_std=0.1,
                qed_lower_bound=0.3,
                logp_mean=3.0,
                logp_std=1.0,
                sas_mean=3.0,
                sas_std=0.5,
                heavy_atom_mean=20.0,
                heavy_atom_std=5.0,
            )
        },
        allowed_sigma=2.0,
        penalty_strength=0.5,
        heavy_atom_penalty_weight=0.15,
    )

    rewards = scorer(["Aa", "Aa", "Aa"], ["[C]", "[O]", "invalid"])
    diagnostics = scorer.last_diagnostics()

    assert rewards.tolist() == pytest.approx(
        [0.8, 0.6 * math.exp(-2.5) * (0.85 + 0.15 * math.exp(-2.0)), 0.0]
    )
    assert base.calls == [(["Aa", "Aa"], ["[C]", "[O]"])]
    assert diagnostics["logp_excess_z"].tolist() == pytest.approx([0.0, 1.0, 0.0])
    assert diagnostics["sas_excess_z"].tolist() == pytest.approx([0.0, 2.0, 0.0])
    assert diagnostics["heavy_atom_excess_z"].tolist() == pytest.approx(
        [0.0, 2.0, 0.0]
    )
    assert diagnostics["property_penalty_factor"].tolist() == pytest.approx(
        [1.0, math.exp(-2.5) * (0.85 + 0.15 * math.exp(-2.0)), 0.0]
    )
    assert diagnostics["property_valid"].tolist() == [1.0, 1.0, 0.0]
    assert scorer.training is False


def test_property_shaping_requires_target_statistics():
    scorer = TargetPropertyShapedActivityScorer(
        FakeActivityScorer(),
        {
            "Aa": TargetActivePropertyStats(
                active_count=2,
                qed_mean=0.5,
                qed_std=0.1,
                qed_lower_bound=0.3,
                logp_mean=2.0,
                logp_std=1.0,
                sas_mean=3.0,
                sas_std=1.0,
                heavy_atom_mean=10.0,
                heavy_atom_std=0.0,
            )
        },
    )

    with pytest.raises(ValueError, match="Missing target"):
        scorer(["Bb"], ["[C]"])

    with pytest.raises(ValueError, match="activity_threshold_bonus_weight"):
        TargetPropertyShapedActivityScorer(
            FakeActivityScorer(),
            scorer.property_stats,
            activity_threshold_bonus_weight=1.1,
        )


def test_property_shaping_can_reward_crossing_activity_threshold(monkeypatch):
    monkeypatch.setattr(
        property_shaping,
        "molecular_property_rows",
        lambda smiles: [
            {"qed": 0.5, "logp": 2.0, "sas": 3.0, "heavy_atom_count": 20.0},
            {"qed": 0.5, "logp": 2.0, "sas": 3.0, "heavy_atom_count": 20.0},
        ],
    )
    base = FakeActivityScorer()
    base.forward = lambda proteins, molecules: torch.tensor([0.49, 0.51])
    scorer = TargetPropertyShapedActivityScorer(
        base,
        {
            "Aa": TargetActivePropertyStats(
                active_count=10,
                qed_mean=0.5,
                qed_std=0.1,
                qed_lower_bound=0.3,
                logp_mean=2.0,
                logp_std=1.0,
                sas_mean=3.0,
                sas_std=1.0,
                heavy_atom_mean=20.0,
                heavy_atom_std=2.0,
            )
        },
        activity_probability_threshold=0.5,
        activity_threshold_bonus_weight=0.5,
    )

    rewards = scorer(["Aa", "Aa"], ["[C]", "[O]"])
    diagnostics = scorer.last_diagnostics()

    assert rewards.tolist() == pytest.approx([0.245, 0.755])
    assert diagnostics["activity_probability"].tolist() == pytest.approx(
        [0.49, 0.51]
    )
    assert diagnostics["activity_optimization_reward"].tolist() == pytest.approx(
        [0.245, 0.755]
    )


def test_qed_penalty_uses_target_lower_bound_and_updates_band(monkeypatch):
    monkeypatch.setattr(
        property_shaping,
        "molecular_property_rows",
        lambda smiles: [
            {"qed": 0.2, "logp": 2.0, "sas": 3.0, "heavy_atom_count": 20.0},
            {"qed": 0.3, "logp": 2.0, "sas": 3.0, "heavy_atom_count": 20.0},
        ],
    )
    base = FakeActivityScorer()
    scorer = TargetPropertyShapedActivityScorer(
        base,
        {
            "Aa": TargetActivePropertyStats(
                active_count=10,
                qed_mean=0.5,
                qed_std=0.1,
                qed_lower_bound=0.3,
                logp_mean=2.0,
                logp_std=1.0,
                sas_mean=3.0,
                sas_std=1.0,
                heavy_atom_mean=20.0,
                heavy_atom_std=2.0,
            )
        },
        qed_penalty_strength=2.0,
    )

    rewards = scorer(["Aa", "Aa"], ["[C]", "[O]"])
    diagnostics = scorer.last_diagnostics()

    assert rewards.tolist() == pytest.approx([0.8 * math.exp(-2.0), 0.6])
    assert diagnostics["qed_penalty_factor"].tolist() == pytest.approx(
        [math.exp(-2.0), 1.0]
    )
    assert diagnostics["qed_deficit_z"].tolist() == pytest.approx([1.0, 0.0])
    assert diagnostics["qed_violation"].tolist() == [1.0, 0.0]
    assert diagnostics["property_band_eligible"].tolist() == [0.0, 1.0]


def test_activity_threshold_bonus_can_require_target_property_bands(monkeypatch):
    monkeypatch.setattr(
        property_shaping,
        "molecular_property_rows",
        lambda smiles: [
            {"qed": 0.5, "logp": 2.0, "sas": 3.0, "heavy_atom_count": 20.0},
            {"qed": 0.4, "logp": 2.0, "sas": 6.0, "heavy_atom_count": 20.0},
        ],
    )
    base = FakeActivityScorer()
    base.forward = lambda proteins, molecules: torch.tensor([0.6, 0.6])
    scorer = TargetPropertyShapedActivityScorer(
        base,
        {
            "Aa": TargetActivePropertyStats(
                active_count=10,
                qed_mean=0.5,
                qed_std=0.1,
                qed_lower_bound=0.3,
                logp_mean=2.0,
                logp_std=1.0,
                sas_mean=3.0,
                sas_std=1.0,
                heavy_atom_mean=20.0,
                heavy_atom_std=2.0,
            )
        },
        activity_probability_threshold=0.5,
        activity_threshold_bonus_weight=0.9,
        activity_threshold_bonus_requires_property_band=True,
    )

    rewards = scorer(["Aa", "Aa"], ["[C]", "[O]"])
    diagnostics = scorer.last_diagnostics()

    assert diagnostics["property_band_eligible"].tolist() == [1.0, 0.0]
    assert diagnostics["activity_threshold_bonus_eligible"].tolist() == [1.0, 0.0]
    assert diagnostics["activity_optimization_reward"].tolist() == pytest.approx(
        [0.96, 0.06]
    )
    assert rewards.tolist() == pytest.approx([0.96, 0.06 * math.exp(-0.5)])


def test_zero_variance_heavy_atom_reference_uses_one_atom_soft_scale(monkeypatch):
    monkeypatch.setattr(
        property_shaping,
        "molecular_property_rows",
        lambda smiles: [
            {
                "qed": 0.5,
                "logp": 2.0,
                "sas": 3.0,
                "heavy_atom_count": 12.0,
            }
        ],
    )
    scorer = TargetPropertyShapedActivityScorer(
        FakeActivityScorer(),
        {
            "Aa": TargetActivePropertyStats(
                active_count=2,
                qed_mean=0.5,
                qed_std=0.1,
                qed_lower_bound=0.3,
                logp_mean=2.0,
                logp_std=1.0,
                sas_mean=3.0,
                sas_std=1.0,
                heavy_atom_mean=10.0,
                heavy_atom_std=0.0,
            )
        },
        heavy_atom_penalty_weight=0.2,
    )
    scorer.activity_scorer.forward = lambda proteins, molecules: torch.tensor([0.8])

    reward = scorer(["Aa"], ["[C]"])

    expected_factor = 0.8 + 0.2 * math.exp(-2.0)
    assert reward.item() == pytest.approx(0.8 * expected_factor)
    assert scorer.last_diagnostics()["heavy_atom_excess_z"].item() == 2.0
