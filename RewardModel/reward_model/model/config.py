from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional

from .losses import (
    DEFAULT_CONTRASTIVE_ACTIVE_THRESHOLD,
    DEFAULT_RANKING_AFFINITY_MARGIN,
)


_VALID_POOLING_TYPES = {"cls", "mean", "mean_all_tok"}
_VALID_FUSION_ATTENTION_BACKENDS = {"manual", "sdpa"}
_VALID_PAIR_SCORING_MODES = {"cosine", "mlp", "scaled_cosine"}
_VALID_MOLECULE_INPUT_REPRESENTATIONS = {"selfies", "smiles"}


@dataclass(eq=True)
class RewardModelConfig:
    protein_model_name_or_path: str = "facebook/esm2_t12_35M_UR50D"
    molecule_model_name_or_path: str = "HUBioDataLab/SELFormer"
    protein_tokenizer_name_or_path: Optional[str] = None
    molecule_tokenizer_name_or_path: Optional[str] = None
    molecule_input_representation: str = "selfies"
    molecule_trust_remote_code: bool = False
    molecule_deterministic_eval: bool = False
    molecule_hidden_dropout_prob: Optional[float] = None
    molecule_attention_probs_dropout_prob: Optional[float] = None
    protein_hidden_size: Optional[int] = None
    molecule_hidden_size: Optional[int] = None
    protein_max_length: int = 1024
    molecule_max_length: int = 512
    fusion_hidden_dim: int = 512
    fusion_num_heads: int = 8
    fusion_attention_backend: str = "manual"
    fusion_residual: bool = False
    dropout: float = 0.1
    pooling_type: str = "mean"
    pair_scoring_mode: str = "mlp"
    cosine_scale_init: float = 13.0
    cosine_scale_max: float = 100.0
    cosine_classification_bias_init: float = 0.0
    activity_threshold: float = 6.0
    ranking_loss_weight: float = 1.0
    contrastive_loss_weight: float = 0.0
    contrastive_active_threshold: float = DEFAULT_CONTRASTIVE_ACTIVE_THRESHOLD
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
        if (
            self.molecule_input_representation
            not in _VALID_MOLECULE_INPUT_REPRESENTATIONS
        ):
            raise ValueError(
                "molecule_input_representation must be one of "
                f"{sorted(_VALID_MOLECULE_INPUT_REPRESENTATIONS)}"
            )
        if not isinstance(self.molecule_trust_remote_code, bool):
            raise ValueError("molecule_trust_remote_code must be a boolean")
        if not isinstance(self.molecule_deterministic_eval, bool):
            raise ValueError("molecule_deterministic_eval must be a boolean")
        for field_name in (
            "molecule_hidden_dropout_prob",
            "molecule_attention_probs_dropout_prob",
        ):
            value = getattr(self, field_name)
            if value is not None and (
                not math.isfinite(float(value)) or not 0.0 <= float(value) < 1.0
            ):
                raise ValueError(
                    f"{field_name} must be None or finite and in [0.0, 1.0)"
                )
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
        if not isinstance(self.fusion_residual, bool):
            raise ValueError("fusion_residual must be a boolean")
        if self.pooling_type not in _VALID_POOLING_TYPES:
            raise ValueError(
                f"Unsupported pooling_type: {self.pooling_type}. "
                f"Expected one of {sorted(_VALID_POOLING_TYPES)}"
            )
        if self.pair_scoring_mode not in _VALID_PAIR_SCORING_MODES:
            raise ValueError(
                "pair_scoring_mode must be one of "
                f"{sorted(_VALID_PAIR_SCORING_MODES)}"
            )
        if self.cosine_scale_init <= 0.0 or not math.isfinite(
            float(self.cosine_scale_init)
        ):
            raise ValueError("cosine_scale_init must be finite and > 0")
        if self.cosine_scale_max <= 0.0 or not math.isfinite(
            float(self.cosine_scale_max)
        ):
            raise ValueError("cosine_scale_max must be finite and > 0")
        if self.cosine_scale_init > self.cosine_scale_max:
            raise ValueError("cosine_scale_init must be <= cosine_scale_max")
        if not math.isfinite(float(self.cosine_classification_bias_init)):
            raise ValueError("cosine_classification_bias_init must be finite")
        if self.protein_max_length <= 0:
            raise ValueError("protein_max_length must be > 0")
        if self.molecule_max_length <= 0:
            raise ValueError("molecule_max_length must be > 0")
        if not math.isfinite(float(self.dropout)) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be finite and in [0.0, 1.0)")
        if self.bce_pos_weight <= 0.0:
            raise ValueError("bce_pos_weight must be > 0")
        if self.ranking_loss_weight < 0.0:
            raise ValueError("ranking_loss_weight must be >= 0")
        if self.contrastive_loss_weight < 0.0 or not math.isfinite(
            float(self.contrastive_loss_weight)
        ):
            raise ValueError("contrastive_loss_weight must be finite and >= 0")
        if not math.isfinite(float(self.contrastive_active_threshold)):
            raise ValueError("contrastive_active_threshold must be finite")
        if (
            self.contrastive_loss_weight > 0.0
            and self.pair_scoring_mode != "cosine"
        ):
            raise ValueError(
                "contrastive_loss_weight > 0 requires pair_scoring_mode='cosine' "
                "so ranking and contrastive learning share one normalized score matrix"
            )
        if (
            self.contrastive_loss_weight > 0.0
            and not self.deduplicate_protein_inputs
        ):
            raise ValueError(
                "contrastive_loss_weight > 0 requires deduplicate_protein_inputs=true "
                "so each assay uses one protein embedding"
            )
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
