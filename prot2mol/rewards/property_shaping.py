"""Target-conditional molecular-property shaping for activity rewards."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import selfies as sf
import torch
import torch.nn as nn

from ..chem.utils import canonic_smiles, molecular_property_rows


@dataclass(frozen=True)
class TargetActivePropertyStats:
    """Property distribution of one target's unique training-set actives."""

    active_count: int
    logp_mean: float
    logp_std: float
    sas_mean: float
    sas_std: float
    heavy_atom_mean: float
    heavy_atom_std: float

    def __post_init__(self) -> None:
        values = (
            self.logp_mean,
            self.logp_std,
            self.sas_mean,
            self.sas_std,
            self.heavy_atom_mean,
            self.heavy_atom_std,
        )
        if self.active_count < 2:
            raise ValueError("Property shaping requires at least two active molecules")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Target active-property statistics must be finite")
        if self.logp_std <= 0.0 or self.sas_std <= 0.0:
            raise ValueError(
                "Target active-property standard deviations must be positive"
            )
        if self.heavy_atom_std < 0.0:
            raise ValueError("Heavy-atom standard deviation must be nonnegative")

    def as_dict(self, *, allowed_sigma: float) -> dict[str, float | int]:
        return {
            "training_active_property_reference_count": self.active_count,
            "training_active_logp_mean": self.logp_mean,
            "training_active_logp_std": self.logp_std,
            "training_active_logp_lower": self.logp_mean
            - allowed_sigma * self.logp_std,
            "training_active_logp_upper": self.logp_mean
            + allowed_sigma * self.logp_std,
            "training_active_sas_mean": self.sas_mean,
            "training_active_sas_std": self.sas_std,
            "training_active_sas_upper": self.sas_mean
            + allowed_sigma * self.sas_std,
            "training_active_heavy_atom_mean": self.heavy_atom_mean,
            "training_active_heavy_atom_std": self.heavy_atom_std,
            "training_active_heavy_atom_lower": max(
                0.0,
                self.heavy_atom_mean - allowed_sigma * self.heavy_atom_std,
            ),
            "training_active_heavy_atom_upper": self.heavy_atom_mean
            + allowed_sigma * self.heavy_atom_std,
        }


class TargetPropertyShapedActivityScorer(nn.Module):
    """Multiply activity probability by target-specific LogP and SAS gates."""

    def __init__(
        self,
        activity_scorer: nn.Module,
        property_stats: Mapping[str, TargetActivePropertyStats],
        *,
        allowed_sigma: float = 2.0,
        penalty_strength: float = 0.5,
        heavy_atom_penalty_weight: float = 0.15,
        activity_probability_threshold: float = 0.5,
        activity_threshold_bonus_weight: float = 0.0,
        activity_threshold_bonus_requires_property_band: bool = False,
    ):
        super().__init__()
        if allowed_sigma <= 0.0:
            raise ValueError("allowed_sigma must be positive")
        if penalty_strength <= 0.0:
            raise ValueError("penalty_strength must be positive")
        if not 0.0 <= heavy_atom_penalty_weight <= 1.0:
            raise ValueError("heavy_atom_penalty_weight must be in [0, 1]")
        if not 0.0 < activity_probability_threshold < 1.0:
            raise ValueError("activity_probability_threshold must be in (0, 1)")
        if not 0.0 <= activity_threshold_bonus_weight <= 1.0:
            raise ValueError("activity_threshold_bonus_weight must be in [0, 1]")
        if not property_stats:
            raise ValueError("Property shaping requires target property statistics")
        self.activity_scorer = activity_scorer
        self.property_stats = dict(property_stats)
        self.allowed_sigma = float(allowed_sigma)
        self.penalty_strength = float(penalty_strength)
        self.heavy_atom_penalty_weight = float(heavy_atom_penalty_weight)
        self.activity_probability_threshold = float(
            activity_probability_threshold
        )
        self.activity_threshold_bonus_weight = float(
            activity_threshold_bonus_weight
        )
        self.activity_threshold_bonus_requires_property_band = bool(
            activity_threshold_bonus_requires_property_band
        )
        self.protein_representation = getattr(
            activity_scorer,
            "protein_representation",
            "sequence",
        )
        self.molecule_representation = getattr(
            activity_scorer,
            "molecule_representation",
            "smiles",
        )
        self._last_diagnostics: dict[str, torch.Tensor] = {}
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "TargetPropertyShapedActivityScorer":
        del mode
        super().train(False)
        return self

    def last_diagnostics(self) -> dict[str, torch.Tensor]:
        """Return detached per-sample components from the immediately prior call."""

        return {
            name: values.detach().float().cpu().clone()
            for name, values in self._last_diagnostics.items()
        }

    def _smiles(self, molecule_sequences: Sequence[str]) -> list[str]:
        if self.molecule_representation == "smiles":
            return [canonic_smiles(value) or "" for value in molecule_sequences]
        if self.molecule_representation != "selfies":
            raise ValueError(
                "Property shaping supports SMILES or SELFIES molecule rewards; "
                f"got {self.molecule_representation!r}"
            )
        smiles = []
        for value in molecule_sequences:
            try:
                decoded = sf.decoder(value) if value else ""
            except Exception:
                decoded = ""
            smiles.append(canonic_smiles(decoded) or "")
        return smiles

    def forward(
        self,
        protein_sequences: Sequence[str],
        molecule_sequences: Sequence[str],
    ) -> torch.Tensor:
        if len(protein_sequences) != len(molecule_sequences):
            raise ValueError("Protein and molecule batches must align")
        if not protein_sequences:
            self._last_diagnostics = {}
            return torch.empty(0, dtype=torch.float32)
        missing = sorted(set(protein_sequences).difference(self.property_stats))
        if missing:
            raise ValueError(
                "Missing target active-property statistics for reward sequences: "
                f"{missing[:5]}"
            )

        properties = molecular_property_rows(self._smiles(molecule_sequences))
        property_valid = [
            row["logp"] is not None
            and row["sas"] is not None
            and row["heavy_atom_count"] is not None
            for row in properties
        ]
        valid_indices = [index for index, valid in enumerate(property_valid) if valid]
        activity = torch.zeros(len(protein_sequences), dtype=torch.float32)
        if valid_indices:
            with torch.inference_mode():
                valid_activity = torch.as_tensor(
                    self.activity_scorer(
                        [protein_sequences[index] for index in valid_indices],
                        [molecule_sequences[index] for index in valid_indices],
                    ),
                    dtype=torch.float32,
                ).reshape(-1).cpu()
            if valid_activity.numel() != len(valid_indices):
                raise ValueError("Activity scorer returned the wrong number of scores")
            if (
                not torch.isfinite(valid_activity).all()
                or valid_activity.lt(0.0).any()
                or valid_activity.gt(1.0).any()
            ):
                raise ValueError(
                    "Activity scorer must return finite probabilities in [0, 1]"
                )
            activity[torch.tensor(valid_indices)] = valid_activity

        diagnostics = {
            "activity_probability": [],
            "activity_optimization_reward": [],
            "property_band_eligible": [],
            "activity_threshold_bonus_eligible": [],
            "logp_penalty_factor": [],
            "sas_penalty_factor": [],
            "heavy_atom_penalty_factor": [],
            "property_penalty_factor": [],
            "logp_excess_z": [],
            "sas_excess_z": [],
            "heavy_atom_excess_z": [],
            "logp_violation": [],
            "sas_violation": [],
            "heavy_atom_violation": [],
            "heavy_atom_count": [],
            "property_valid": [],
            "property_shaped_reward": [],
        }
        rewards = []
        for index, (protein, row) in enumerate(zip(protein_sequences, properties)):
            stats = self.property_stats[protein]
            logp = row["logp"]
            sas = row["sas"]
            heavy_atom_count = row["heavy_atom_count"]
            if logp is None or sas is None or heavy_atom_count is None:
                logp_excess = 0.0
                sas_excess = 0.0
                heavy_atom_excess = 0.0
                logp_factor = 0.0
                sas_factor = 0.0
                heavy_atom_factor = 0.0
            else:
                logp_excess = max(
                    0.0,
                    abs(float(logp) - stats.logp_mean) / stats.logp_std
                    - self.allowed_sigma,
                )
                sas_excess = max(
                    0.0,
                    (float(sas) - stats.sas_mean) / stats.sas_std
                    - self.allowed_sigma,
                )
                heavy_atom_scale = max(stats.heavy_atom_std, 1.0)
                heavy_atom_excess = max(
                    0.0,
                    abs(float(heavy_atom_count) - stats.heavy_atom_mean)
                    / heavy_atom_scale
                    - (
                        self.allowed_sigma * stats.heavy_atom_std
                        / heavy_atom_scale
                    ),
                )
                logp_factor = math.exp(
                    -self.penalty_strength * logp_excess * logp_excess
                )
                sas_factor = math.exp(
                    -self.penalty_strength * sas_excess * sas_excess
                )
                heavy_atom_gate = math.exp(
                    -0.5 * heavy_atom_excess * heavy_atom_excess
                )
                heavy_atom_factor = 1.0 - self.heavy_atom_penalty_weight * (
                    1.0 - heavy_atom_gate
                )
            property_factor = logp_factor * sas_factor * heavy_atom_factor
            property_band_eligible = float(
                property_valid[index]
                and logp_excess == 0.0
                and sas_excess == 0.0
                and heavy_atom_excess == 0.0
            )
            activity_probability = float(activity[index])
            threshold_bonus = float(
                activity_probability >= self.activity_probability_threshold
            )
            threshold_bonus_eligible = threshold_bonus * (
                property_band_eligible
                if self.activity_threshold_bonus_requires_property_band
                else 1.0
            )
            activity_optimization_reward = (
                (1.0 - self.activity_threshold_bonus_weight)
                * activity_probability
                + self.activity_threshold_bonus_weight * threshold_bonus_eligible
            )
            shaped_reward = activity_optimization_reward * property_factor
            rewards.append(shaped_reward)
            diagnostics["activity_probability"].append(activity_probability)
            diagnostics["activity_optimization_reward"].append(
                activity_optimization_reward
            )
            diagnostics["property_band_eligible"].append(
                property_band_eligible
            )
            diagnostics["activity_threshold_bonus_eligible"].append(
                threshold_bonus_eligible
            )
            diagnostics["logp_penalty_factor"].append(logp_factor)
            diagnostics["sas_penalty_factor"].append(sas_factor)
            diagnostics["heavy_atom_penalty_factor"].append(heavy_atom_factor)
            diagnostics["property_penalty_factor"].append(property_factor)
            diagnostics["logp_excess_z"].append(logp_excess)
            diagnostics["sas_excess_z"].append(sas_excess)
            diagnostics["heavy_atom_excess_z"].append(heavy_atom_excess)
            diagnostics["logp_violation"].append(float(logp_excess > 0.0))
            diagnostics["sas_violation"].append(float(sas_excess > 0.0))
            diagnostics["heavy_atom_violation"].append(
                float(heavy_atom_excess > 0.0)
            )
            diagnostics["heavy_atom_count"].append(
                float(heavy_atom_count) if heavy_atom_count is not None else 0.0
            )
            diagnostics["property_valid"].append(float(property_valid[index]))
            diagnostics["property_shaped_reward"].append(shaped_reward)

        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        if not torch.isfinite(reward_tensor).all():
            raise RuntimeError("Property-shaped activity reward became non-finite")
        self._last_diagnostics = {
            name: torch.tensor(values, dtype=torch.float32)
            for name, values in diagnostics.items()
        }
        return reward_tensor
