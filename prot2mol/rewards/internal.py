"""Frozen activity scorer backed by Prot2Mol's standalone RewardModel."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn


class InternalRewardModelActivityScorer(nn.Module):
    """Expose a saved RewardModel's binary activity probability in batches."""

    protein_representation = "sequence"

    def __init__(self, reward_model: nn.Module, *, batch_size: int = 8):
        super().__init__()
        if batch_size < 1:
            raise ValueError("Internal reward batch_size must be positive")
        representation = getattr(
            getattr(reward_model, "config", None),
            "molecule_input_representation",
            None,
        )
        if representation not in {"smiles", "selfies"}:
            raise ValueError(
                "Internal RewardModel must declare a SMILES or SELFIES input "
                f"representation; got {representation!r}"
            )
        self.reward_model = reward_model
        self.batch_size = int(batch_size)
        self.molecule_representation = str(representation)
        self._last_activity_probability = torch.empty(0, dtype=torch.float32)
        self.reward_model.requires_grad_(False)
        self.reward_model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        device: torch.device,
        batch_size: int = 8,
    ) -> "InternalRewardModelActivityScorer":
        try:
            from RewardModel.reward_model.model import load_reward_model
        except ImportError as exc:  # pragma: no cover - installation error path
            raise ImportError(
                "Prot2Mol's internal RewardModel package is unavailable"
            ) from exc

        model = load_reward_model(
            str(Path(model_path).expanduser().resolve()),
            device=device,
            strict=True,
            config_overrides={
                "freeze_protein_encoder": True,
                "freeze_molecule_encoder": True,
            },
        )
        return cls(model, batch_size=batch_size)

    def train(self, mode: bool = True) -> "InternalRewardModelActivityScorer":
        del mode
        super().train(False)
        self.reward_model.eval()
        return self

    def last_diagnostics(self) -> dict[str, torch.Tensor]:
        """Return activity probabilities from the most recent scoring call."""

        return {
            "activity_probability": self._last_activity_probability.detach().clone()
        }

    def forward(
        self,
        protein_sequences: Sequence[str],
        molecule_sequences: Sequence[str],
    ) -> torch.Tensor:
        if len(protein_sequences) != len(molecule_sequences):
            raise ValueError("Protein and molecule batches must align")
        if not protein_sequences:
            self._last_activity_probability = torch.empty(0, dtype=torch.float32)
            return self._last_activity_probability.clone()

        probabilities = []
        with torch.inference_mode():
            for start in range(0, len(protein_sequences), self.batch_size):
                outputs = self.reward_model.score_pairs(
                    protein_sequences=protein_sequences[
                        start : start + self.batch_size
                    ],
                    molecule_sequences=molecule_sequences[
                        start : start + self.batch_size
                    ],
                )
                batch = outputs.activity_probability.detach().float().reshape(-1)
                expected = min(
                    self.batch_size,
                    len(protein_sequences) - start,
                )
                if batch.numel() != expected:
                    raise ValueError(
                        "Internal RewardModel returned the wrong number of scores"
                    )
                if (
                    not torch.isfinite(batch).all()
                    or batch.lt(0.0).any()
                    or batch.gt(1.0).any()
                ):
                    raise ValueError(
                        "Internal RewardModel must return finite probabilities in [0, 1]"
                    )
                probabilities.append(batch.cpu())
        self._last_activity_probability = torch.cat(probabilities)
        return self._last_activity_probability.clone()
