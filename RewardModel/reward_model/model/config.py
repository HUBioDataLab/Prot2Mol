from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from .losses import (
    DEFAULT_CONTRASTIVE_ACTIVE_THRESHOLD,
    DEFAULT_RANKING_AFFINITY_MARGIN,
)


_VALID_POOLING_TYPES = {"cls", "mean", "mean_all_tok"}
_VALID_FUSION_ATTENTION_BACKENDS = {"manual", "sdpa"}
_VALID_PAIR_SCORING_MODES = {
    "cosine",
    "fusion_contrastive",
    "mlp",
    "mlp_contrastive",
    "scaled_cosine",
}
# Modes whose forward pass produces the normalized embedding pair the
# in-batch retrieval objective needs.
_CONTRASTIVE_PAIR_SCORING_MODES = {
    "cosine",
    "fusion_contrastive",
    "mlp_contrastive",
}
_VALID_PROJECTION_TYPES = {"linear", "nonlinear"}
_VALID_MOLECULE_INPUT_REPRESENTATIONS = {"selfies", "smiles"}


@dataclass(eq=True)
class RewardModelConfig:
    protein_model_name_or_path: str = "facebook/esm2_t12_35M_UR50D"
    molecule_model_name_or_path: str = "HUBioDataLab/SELFormer"
    protein_tokenizer_name_or_path: Optional[str] = None
    molecule_tokenizer_name_or_path: Optional[str] = None
    protein_hidden_dropout_prob: Optional[float] = None
    protein_attention_probs_dropout_prob: Optional[float] = None
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
    projection_type: str = "linear"
    fusion_num_heads: int = 8
    # Number of stacked TokenFusion blocks. Each block carries its own
    # attention parameters and its own residual, so layer N re-fuses the
    # post-residual output of layer N-1 rather than the pre-fusion
    # projections. One block reproduces the historical architecture.
    fusion_num_layers: int = 1
    # Per-attention output projection (W_O). Standard multi-head attention mixes
    # the concatenated heads through a learned matrix; without it each head is
    # confined to its own fixed slice of the output. Enabling this gives the
    # self- and cross-attention of each stream its OWN W_O, applied before the
    # two are averaged, so heads can be weighted differently for the two
    # directions. Four Linear(d, d) per fusion block.
    fusion_output_projection: bool = False
    # Position-wise feed-forward sublayer width, as a multiple of
    # fusion_hidden_dim. 0 disables it. When > 0 each stream gets its own
    # Linear(d, m*d) -> GELU -> Linear(m*d, d) with its own residual and
    # LayerNorm, applied after the attention sublayer.
    fusion_ffn_multiplier: int = 0
    fusion_attention_backend: str = "manual"
    fusion_residual: bool = False
    dropout: float = 0.1
    pooling_type: str = "mean"
    protein_pooling_type: Optional[str] = None
    molecule_pooling_type: Optional[str] = None
    pair_scoring_mode: str = "mlp"
    # Explicit deployed/evaluation scorer for loss-subset experiments. The
    # historical default is unchanged. Alternative readouts require R=0 and
    # a trained corresponding objective; an untrained R head must not select
    # the checkpoint of a B-only or C-only model.
    ranking_readout: str = "ranking_head"
    cosine_scale_init: float = 13.0
    cosine_scale_max: float = 100.0
    cosine_classification_bias_init: float = 0.0
    activity_threshold: float = 6.0
    ranking_loss_weight: float = 1.0
    contrastive_loss_weight: float = 0.0
    contrastive_active_threshold: float = DEFAULT_CONTRASTIVE_ACTIVE_THRESHOLD
    contrastive_strict_active_only: bool = False
    contrastive_mask_policy: str = "reference"
    classification_loss_weight: float = 1.0
    cosine_classification_mlp: bool = False
    # Hidden widths of the ``mlp`` scoring-mode heads. Defaults reproduce the
    # historical hard-coded stack; a shorter tuple such as ``(128,)`` gives a
    # head of the same size as the fusion-contrastive classification head.
    ranking_head_hidden_dims: Sequence[int] = (2048, 1024, 512, 256, 128)
    classification_head_hidden_dims: Sequence[int] = (2048, 1024, 512, 256, 128)
    cosine_marginal_biases: bool = False
    # Apply LigUnity's fixed cosine score scale in ``cosine`` mode. LigUnity
    # declares logit_scale as a parameter but uses it as ``.exp().detach()``, so
    # it never receives gradient and stays at exp(log 13) = 13 for the whole
    # run. Off by default, which keeps plain ``cosine`` scoring the raw cosine
    # and every checkpoint trained that way loadable unchanged.
    cosine_fixed_score_scale: bool = False
    ranking_temperature: float = 1.0
    ranking_objective: str = "ligunity_listwise"
    ranking_normalize_per_group: bool = False
    ranking_balance_assays: bool = False
    ranking_position_weighting: str = "top"
    ranking_pair_target_temperature: Optional[float] = None
    ranking_affinity_margin: float = DEFAULT_RANKING_AFFINITY_MARGIN
    ranking_min_pchembl_span: float = 0.5
    bce_pos_weight: float = 1.0
    deduplicate_protein_inputs: bool = True
    deduplicate_molecule_inputs: bool = True
    freeze_protein_encoder: bool = False
    freeze_molecule_encoder: bool = False
    protein_unfrozen_last_layers: Optional[int] = None
    molecule_unfrozen_last_layers: Optional[int] = None
    encoder_lora_rank: int = 0
    encoder_lora_alpha: float = 16.0
    learn_contrastive_scale: bool = True
    contrastive_detach_encoders: bool = False

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
            "protein_hidden_dropout_prob",
            "protein_attention_probs_dropout_prob",
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
        for name in ("ranking_head_hidden_dims", "classification_head_hidden_dims"):
            dims = tuple(getattr(self, name))
            if not dims:
                raise ValueError(f"{name} must contain at least one hidden width")
            if any(int(dim) <= 0 for dim in dims):
                raise ValueError(f"{name} entries must all be > 0")
            setattr(self, name, dims)
        if self.projection_type not in _VALID_PROJECTION_TYPES:
            raise ValueError(
                "projection_type must be one of "
                f"{sorted(_VALID_PROJECTION_TYPES)}"
            )
        if self.fusion_num_heads <= 0:
            raise ValueError("fusion_num_heads must be > 0")
        if self.fusion_num_layers <= 0:
            raise ValueError("fusion_num_layers must be > 0")
        if not isinstance(self.fusion_output_projection, bool):
            raise ValueError("fusion_output_projection must be a boolean")
        if (
            not isinstance(self.fusion_ffn_multiplier, int)
            or isinstance(self.fusion_ffn_multiplier, bool)
            or self.fusion_ffn_multiplier < 0
        ):
            raise ValueError("fusion_ffn_multiplier must be an integer >= 0")
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
        for field_name in ("protein_pooling_type", "molecule_pooling_type"):
            value = getattr(self, field_name)
            if value is not None and value not in _VALID_POOLING_TYPES:
                raise ValueError(
                    f"Unsupported {field_name}: {value}. "
                    f"Expected None or one of {sorted(_VALID_POOLING_TYPES)}"
                )
        if self.pair_scoring_mode not in _VALID_PAIR_SCORING_MODES:
            raise ValueError(
                "pair_scoring_mode must be one of "
                f"{sorted(_VALID_PAIR_SCORING_MODES)}"
            )
        if self.fusion_num_layers > 1 and self.pair_scoring_mode == "cosine":
            raise ValueError(
                "fusion_num_layers > 1 requires a fusion-based pair_scoring_mode; "
                "'cosine' pools and projects without any token fusion"
            )
        if self.pair_scoring_mode == "cosine" and (
            self.fusion_output_projection or self.fusion_ffn_multiplier > 0
        ):
            raise ValueError(
                "fusion_output_projection and fusion_ffn_multiplier require a "
                "fusion-based pair_scoring_mode; 'cosine' has no token fusion"
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
        if not isinstance(self.contrastive_strict_active_only, bool):
            raise ValueError("contrastive_strict_active_only must be a boolean")
        if (
            self.contrastive_loss_weight > 0.0
            and self.pair_scoring_mode not in _CONTRASTIVE_PAIR_SCORING_MODES
        ):
            raise ValueError(
                "contrastive_loss_weight > 0 requires pair_scoring_mode to be "
                "'cosine' or 'fusion_contrastive'"
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
        if not isinstance(self.cosine_classification_mlp, bool):
            raise ValueError("cosine_classification_mlp must be a boolean")
        if not isinstance(self.cosine_fixed_score_scale, bool):
            raise ValueError("cosine_fixed_score_scale must be a boolean")
        if self.cosine_fixed_score_scale and self.pair_scoring_mode != "cosine":
            raise ValueError(
                "cosine_fixed_score_scale only applies to pair_scoring_mode="
                "'cosine'; the other cosine-based modes already scale their score"
            )
        if self.cosine_classification_mlp and self.pair_scoring_mode != "cosine":
            raise ValueError(
                "cosine_classification_mlp=true requires pair_scoring_mode='cosine'"
            )
        if not isinstance(self.cosine_marginal_biases, bool):
            raise ValueError("cosine_marginal_biases must be a boolean")
        if self.cosine_marginal_biases and self.pair_scoring_mode != "cosine":
            raise ValueError(
                "cosine_marginal_biases=true requires pair_scoring_mode='cosine'"
            )
        if self.cosine_marginal_biases and self.cosine_classification_mlp:
            raise ValueError(
                "cosine_marginal_biases and cosine_classification_mlp are mutually exclusive"
            )
        if self.ranking_temperature <= 0.0:
            raise ValueError("ranking_temperature must be > 0")
        if self.ranking_affinity_margin < 0.0 or not math.isfinite(
            float(self.ranking_affinity_margin)
        ):
            raise ValueError("ranking_affinity_margin must be finite and >= 0")
        if self.ranking_min_pchembl_span < 0.0:
            raise ValueError("ranking_min_pchembl_span must be >= 0")
        if self.ranking_objective not in {"ligunity_listwise", "pairwise_logistic"}:
            raise ValueError("Unknown ranking_objective")
        if not isinstance(self.ranking_normalize_per_group, bool):
            raise ValueError("ranking_normalize_per_group must be boolean")
        if not isinstance(self.ranking_balance_assays, bool):
            raise ValueError("ranking_balance_assays must be boolean")
        if self.ranking_position_weighting not in {"top", "uniform"}:
            raise ValueError("Unknown ranking_position_weighting")
        if self.ranking_pair_target_temperature is not None:
            if (not math.isfinite(float(self.ranking_pair_target_temperature))
                    or self.ranking_pair_target_temperature <= 0):
                raise ValueError("ranking_pair_target_temperature must be finite and positive")
            if self.ranking_objective != "pairwise_logistic" or self.ranking_affinity_margin != 0:
                raise ValueError("Soft targets require pairwise_logistic and explicit zero affinity margin")
        if not isinstance(self.deduplicate_protein_inputs, bool):
            raise ValueError("deduplicate_protein_inputs must be a boolean")
        if not isinstance(self.deduplicate_molecule_inputs, bool):
            raise ValueError("deduplicate_molecule_inputs must be a boolean")
        if not isinstance(self.freeze_protein_encoder, bool):
            raise ValueError("freeze_protein_encoder must be a boolean")
        if not isinstance(self.freeze_molecule_encoder, bool):
            raise ValueError("freeze_molecule_encoder must be a boolean")
        if not isinstance(self.encoder_lora_rank, int) or isinstance(self.encoder_lora_rank, bool) or self.encoder_lora_rank < 0:
            raise ValueError("encoder_lora_rank must be a nonnegative integer")
        if not math.isfinite(float(self.encoder_lora_alpha)) or self.encoder_lora_alpha <= 0:
            raise ValueError("encoder_lora_alpha must be finite and positive")
        for side in ("protein", "molecule"):
            layers = getattr(self, f"{side}_unfrozen_last_layers")
            if layers is not None and (not isinstance(layers, int) or isinstance(layers, bool) or layers <= 0):
                raise ValueError(f"{side}_unfrozen_last_layers must be a positive integer or None")
            if sum((getattr(self, f"freeze_{side}_encoder"), layers is not None, self.encoder_lora_rank > 0)) > 1:
                raise ValueError(f"Choose one adaptation strategy for the {side} encoder")
        for name in ("learn_contrastive_scale", "contrastive_detach_encoders"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.contrastive_mask_policy not in {"none", "reference", "training_known"}:
            raise ValueError("Unknown contrastive_mask_policy")
        if self.contrastive_mask_policy == "training_known" and not self.contrastive_strict_active_only:
            raise ValueError("Training-known masking requires strict active-only contrastive supervision")
        if self.contrastive_detach_encoders and self.pair_scoring_mode not in {"mlp_contrastive", "fusion_contrastive"}:
            raise ValueError("contrastive_detach_encoders requires a separate contrastive projection pathway")
        if not self.learn_contrastive_scale and self.pair_scoring_mode not in {"mlp_contrastive", "fusion_contrastive"}:
            raise ValueError("A fixed contrastive scale requires a separate contrastive pathway")
        if self.ranking_readout not in {"ranking_head", "activity_logit", "cosine"}:
            raise ValueError("Unknown ranking_readout")
        if self.ranking_readout != "ranking_head":
            if self.pair_scoring_mode != "mlp_contrastive" or self.ranking_loss_weight != 0:
                raise ValueError("Alternative ranking readouts require mlp_contrastive and zero ranking loss weight")
            if self.ranking_readout == "activity_logit" and self.classification_loss_weight <= 0:
                raise ValueError("activity_logit readout requires a trained classification objective")
            if self.ranking_readout == "cosine" and self.contrastive_loss_weight <= 0:
                raise ValueError("cosine readout requires a trained contrastive objective")

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
