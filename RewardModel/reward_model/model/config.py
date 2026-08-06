from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional

from .losses import DEFAULT_RANKING_AFFINITY_MARGIN


_VALID_POOLING_TYPES = {"cls", "mean", "mean_all_tok"}
_VALID_FUSION_ATTENTION_BACKENDS = {"manual", "sdpa"}


@dataclass(eq=True)
class RewardModelConfig:
    protein_model_name_or_path: str = "facebook/esm2_t12_35M_UR50D"
    molecule_model_name_or_path: str = "HUBioDataLab/SELFormer"
    protein_tokenizer_name_or_path: Optional[str] = None
    molecule_tokenizer_name_or_path: Optional[str] = None
    protein_hidden_size: Optional[int] = None
    molecule_hidden_size: Optional[int] = None
    protein_max_length: int = 1024
    molecule_max_length: int = 512
    fusion_hidden_dim: int = 512
    fusion_num_heads: int = 8
    fusion_attention_backend: str = "manual"
    dropout: float = 0.1
    pooling_type: str = "mean"
    activity_threshold: float = 6.0
    ranking_loss_weight: float = 1.0
    classification_loss_weight: float = 1.0
    ranking_temperature: float = 1.0
    ranking_affinity_margin: float = DEFAULT_RANKING_AFFINITY_MARGIN
    ranking_min_pchembl_span: float = 0.5
    bce_pos_weight: float = 1.0
    deduplicate_protein_inputs: bool = True
    deduplicate_molecule_inputs: bool = True
    freeze_protein_encoder: bool = False
    freeze_molecule_encoder: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.protein_model_name_or_path:
            raise ValueError("protein_model_name_or_path must be provided")
        if not self.molecule_model_name_or_path:
            raise ValueError("molecule_model_name_or_path must be provided")
        if self.fusion_hidden_dim <= 0:
            raise ValueError("fusion_hidden_dim must be > 0")
        if self.fusion_num_heads <= 0:
            raise ValueError("fusion_num_heads must be > 0")
        if self.fusion_hidden_dim % self.fusion_num_heads != 0:
            raise ValueError(
                f"fusion_hidden_dim ({self.fusion_hidden_dim}) must be divisible by "
                f"fusion_num_heads ({self.fusion_num_heads})"
            )
        if self.fusion_attention_backend not in _VALID_FUSION_ATTENTION_BACKENDS:
            raise ValueError(
                "fusion_attention_backend must be one of "
                f"{sorted(_VALID_FUSION_ATTENTION_BACKENDS)}"
            )
        if self.pooling_type not in _VALID_POOLING_TYPES:
            raise ValueError(
                f"Unsupported pooling_type: {self.pooling_type}. "
                f"Expected one of {sorted(_VALID_POOLING_TYPES)}"
            )
        if self.protein_max_length <= 0:
            raise ValueError("protein_max_length must be > 0")
        if self.molecule_max_length <= 0:
            raise ValueError("molecule_max_length must be > 0")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")
        if self.bce_pos_weight <= 0.0:
            raise ValueError("bce_pos_weight must be > 0")
        if self.ranking_loss_weight < 0.0:
            raise ValueError("ranking_loss_weight must be >= 0")
        if self.classification_loss_weight < 0.0:
            raise ValueError("classification_loss_weight must be >= 0")
        if self.ranking_temperature <= 0.0:
            raise ValueError("ranking_temperature must be > 0")
        if self.ranking_affinity_margin < 0.0 or not math.isfinite(
            float(self.ranking_affinity_margin)
        ):
            raise ValueError("ranking_affinity_margin must be finite and >= 0")
        if self.ranking_min_pchembl_span < 0.0:
            raise ValueError("ranking_min_pchembl_span must be >= 0")
        if not isinstance(self.deduplicate_protein_inputs, bool):
            raise ValueError("deduplicate_protein_inputs must be a boolean")
        if not isinstance(self.deduplicate_molecule_inputs, bool):
            raise ValueError("deduplicate_molecule_inputs must be a boolean")
        if not isinstance(self.freeze_protein_encoder, bool):
            raise ValueError("freeze_protein_encoder must be a boolean")
        if not isinstance(self.freeze_molecule_encoder, bool):
            raise ValueError("freeze_molecule_encoder must be a boolean")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RewardModelConfig":
        normalized = dict(payload)
        legacy_pair_weight = normalized.pop("pair_loss_weight", None)
        if legacy_pair_weight is not None and "ranking_loss_weight" not in normalized:
            normalized["ranking_loss_weight"] = legacy_pair_weight
        return cls(**normalized)

    def save_json(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
        return path

    @classmethod
    def load_json(cls, path: str) -> "RewardModelConfig":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Config JSON at {path} must contain a mapping")
        return cls.from_dict(payload)
