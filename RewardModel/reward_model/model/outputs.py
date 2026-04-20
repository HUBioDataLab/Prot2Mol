from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Iterable, Optional, Tuple

import torch


@dataclass
class RewardModelOutput:
    ranking_score: torch.Tensor
    activity_logits: torch.Tensor
    activity_probability: torch.Tensor
    joint_embedding: torch.Tensor
    pair_loss: Optional[torch.Tensor] = None
    classification_loss: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    protein_token_embeddings: Optional[torch.Tensor] = None
    molecule_token_embeddings: Optional[torch.Tensor] = None
    fused_protein_tokens: Optional[torch.Tensor] = None
    fused_molecule_tokens: Optional[torch.Tensor] = None
    protein_attention_mask: Optional[torch.Tensor] = None
    molecule_attention_mask: Optional[torch.Tensor] = None

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return getattr(self, key, None) is not None

    def keys(self) -> Iterable[str]:
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                yield field.name

    def items(self):
        for key in self.keys():
            yield key, getattr(self, key)

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {key: value for key, value in self.items()}

    def to_tuple(self) -> Tuple[torch.Tensor, ...]:
        return tuple(value for _, value in self.items())
