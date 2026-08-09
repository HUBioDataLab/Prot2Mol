from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RewardModelConfig
from .encoders import LoadedEncoder, batch_encode_texts, encode_tokens, load_encoder_bundle
from .fusion import RewardMLPHead, TokenFusion, masked_pool, normalize_mask
from .losses import (
    ligunity_bidirectional_contrastive_loss,
    ligunity_listwise_loss,
)
from .outputs import RewardModelOutput


class RewardModel(nn.Module):
    """Standalone reward model for protein-molecule pair scoring."""

    def __init__(
        self,
        config: RewardModelConfig | Dict[str, Any],
        protein_bundle: Optional[LoadedEncoder] = None,
        molecule_bundle: Optional[LoadedEncoder] = None,
    ):
        super().__init__()
        self._config = config if isinstance(config, RewardModelConfig) else RewardModelConfig.from_dict(config)

        if protein_bundle is None:
            protein_bundle = load_encoder_bundle(
                name_or_path=self._config.protein_model_name_or_path,
                tokenizer_name_or_path=self._config.protein_tokenizer_name_or_path,
            )
        if molecule_bundle is None:
            molecule_model_kwargs: Dict[str, Any] = {
                "trust_remote_code": self._config.molecule_trust_remote_code,
            }
            if self._config.molecule_deterministic_eval:
                molecule_model_kwargs["deterministic_eval"] = True
            molecule_bundle = load_encoder_bundle(
                name_or_path=self._config.molecule_model_name_or_path,
                tokenizer_name_or_path=self._config.molecule_tokenizer_name_or_path,
                tokenizer_kwargs={
                    "trust_remote_code": self._config.molecule_trust_remote_code,
                },
                model_kwargs=molecule_model_kwargs,
            )

        self.protein_tokenizer = protein_bundle.tokenizer
        self.molecule_tokenizer = molecule_bundle.tokenizer
        self.protein_encoder = protein_bundle.model
        self.molecule_encoder = molecule_bundle.model
        self.protein_encoder.requires_grad_(not self._config.freeze_protein_encoder)
        self.molecule_encoder.requires_grad_(not self._config.freeze_molecule_encoder)
        if self._config.freeze_protein_encoder:
            self.protein_encoder.eval()
        if self._config.freeze_molecule_encoder:
            self.molecule_encoder.eval()

        self._config.protein_hidden_size = self._config.protein_hidden_size or protein_bundle.hidden_size
        self._config.molecule_hidden_size = self._config.molecule_hidden_size or molecule_bundle.hidden_size
        self._config.validate()

        self.protein_projection = nn.Linear(self._config.protein_hidden_size, self._config.fusion_hidden_dim)
        self.molecule_projection = nn.Linear(self._config.molecule_hidden_size, self._config.fusion_hidden_dim)
        if self._config.pair_scoring_mode == "cosine":
            # The simple cosine ablation is intentionally only
            # encoder -> projection -> pooling -> L2 normalization. Keep these
            # attributes for a stable public interface, but do not instantiate
            # any of the non-encoder processing used by the other modes.
            self.protein_norm = None
            self.molecule_norm = None
            self.projection_dropout = None
            self.fusion = None
        else:
            self.protein_norm = nn.LayerNorm(self._config.fusion_hidden_dim)
            self.molecule_norm = nn.LayerNorm(self._config.fusion_hidden_dim)
            self.projection_dropout = nn.Dropout(self._config.dropout)
            self.fusion = TokenFusion(
                hidden_dim=self._config.fusion_hidden_dim,
                num_heads=self._config.fusion_num_heads,
                attention_backend=self._config.fusion_attention_backend,
                residual=self._config.fusion_residual,
                dropout=self._config.dropout,
            )

        if self._config.pair_scoring_mode == "mlp":
            head_input_dim = self._config.fusion_hidden_dim * 2
            ranking_hidden_dims = (2048, 1024, 512, 256, 128)
            classification_hidden_dims = (2048, 1024, 512, 256, 128)
            self.ranking_head = RewardMLPHead(
                input_dim=head_input_dim,
                hidden_dims=ranking_hidden_dims,
                dropout=self._config.dropout,
            )
            self.classification_head = RewardMLPHead(
                input_dim=head_input_dim,
                hidden_dims=classification_hidden_dims,
                dropout=self._config.dropout,
            )
            self.logit_scale = None
            self.classification_logit_bias = None
        elif self._config.pair_scoring_mode == "scaled_cosine":
            # LigUnity initializes its log-space cosine scale at log(13). We
            # preserve that reference point, but allow both of our objectives
            # to optimize the shared positive scale instead of detaching it.
            self.ranking_head = None
            self.classification_head = None
            self.logit_scale = nn.Parameter(
                torch.tensor(math.log(self._config.cosine_scale_init))
            )
            self.classification_logit_bias = nn.Parameter(
                torch.tensor(self._config.cosine_classification_bias_init)
            )
        else:
            self.ranking_head = None
            self.classification_head = None
            self.logit_scale = None
            self.classification_logit_bias = None

    @property
    def config(self) -> RewardModelConfig:
        return self._config

    def train(self, mode: bool = True):
        super().train(mode)
        if self._config.freeze_protein_encoder:
            self.protein_encoder.eval()
        if self._config.freeze_molecule_encoder:
            self.molecule_encoder.eval()
        return self

    def encode_protein(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return encode_tokens(self.protein_encoder, input_ids=input_ids, attention_mask=attention_mask)

    def encode_molecule(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return encode_tokens(self.molecule_encoder, input_ids=input_ids, attention_mask=attention_mask)

    @staticmethod
    def _deduplicated_encode(
        encode_fn,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        *,
        enabled: bool,
    ) -> torch.Tensor:
        """Encode identical rows once while preserving the original batch and gradients."""
        if not enabled or input_ids.size(0) <= 1:
            return encode_fn(input_ids=input_ids, attention_mask=attention_mask)

        if attention_mask is None:
            unique_keys, inverse_indices = torch.unique(
                input_ids,
                dim=0,
                return_inverse=True,
            )
            unique_input_ids = unique_keys
            unique_attention_mask = None
        else:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            row_keys = torch.cat(
                [input_ids, attention_mask.to(dtype=input_ids.dtype)],
                dim=1,
            )
            unique_keys, inverse_indices = torch.unique(
                row_keys,
                dim=0,
                return_inverse=True,
            )
            sequence_length = input_ids.size(1)
            unique_input_ids = unique_keys[:, :sequence_length]
            unique_attention_mask = unique_keys[:, sequence_length:].to(
                dtype=attention_mask.dtype
            )

        if unique_input_ids.size(0) == input_ids.size(0):
            return encode_fn(input_ids=input_ids, attention_mask=attention_mask)

        unique_tokens = encode_fn(
            input_ids=unique_input_ids,
            attention_mask=unique_attention_mask,
        )
        return unique_tokens.index_select(0, inverse_indices)

    def _model_device(self) -> torch.device:
        return next(self.parameters()).device

    def tokenize_proteins(
        self,
        sequences: Sequence[str],
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        return batch_encode_texts(
            tokenizer=self.protein_tokenizer,
            texts=sequences,
            max_length=self._config.protein_max_length,
            device=device if device is not None else self._model_device(),
        )

    def tokenize_molecules(
        self,
        molecules: Sequence[str],
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        return batch_encode_texts(
            tokenizer=self.molecule_tokenizer,
            texts=molecules,
            max_length=self._config.molecule_max_length,
            device=device if device is not None else self._model_device(),
        )

    def _project_tokens(
        self,
        protein_tokens: torch.Tensor,
        molecule_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.protein_norm is None
            or self.molecule_norm is None
            or self.projection_dropout is None
        ):
            raise RuntimeError(
                "Projection normalization and dropout are not initialized for "
                f"pair_scoring_mode={self._config.pair_scoring_mode!r}"
            )
        protein_tokens = self.projection_dropout(
            self.protein_norm(self.protein_projection(protein_tokens))
        )
        molecule_tokens = self.projection_dropout(
            self.molecule_norm(self.molecule_projection(molecule_tokens))
        )
        return protein_tokens, molecule_tokens

    def _compute_ranking_loss(
        self,
        ranking_score: torch.Tensor,
        pchembl_values: torch.Tensor,
        ranking_group_ids: torch.Tensor,
    ) -> torch.Tensor:
        return ligunity_listwise_loss(
            ranking_score,
            pchembl_values,
            ranking_group_ids,
            temperature=self._config.ranking_temperature,
            # The dataset applies this threshold to the complete assay before
            # constructing non-overlapping sublists. Reapplying it to each
            # random sublist would silently discard valid assay opportunities.
            min_pchembl_span=0.0,
            affinity_margin=self._config.ranking_affinity_margin,
        )

    def _compute_contrastive_loss(
        self,
        normalized_protein_embedding: torch.Tensor,
        normalized_molecule_embedding: torch.Tensor,
        pchembl_values: torch.Tensor,
        ranking_group_ids: torch.Tensor,
        contrastive_target_ids: torch.Tensor,
        contrastive_molecule_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build LigUnity's shared assay-by-ligand score matrix from pair rows."""
        group_ids = ranking_group_ids.reshape(-1).to(
            device=normalized_protein_embedding.device,
            dtype=torch.long,
        )
        pchembl = pchembl_values.reshape(-1).to(
            device=normalized_protein_embedding.device,
        )
        target_ids = contrastive_target_ids.reshape(-1).to(
            device=normalized_protein_embedding.device,
            dtype=torch.long,
        )
        molecule_ids = contrastive_molecule_ids.reshape(-1).to(
            device=normalized_protein_embedding.device,
            dtype=torch.long,
        )
        num_pairs = normalized_protein_embedding.size(0)
        if normalized_molecule_embedding.size(0) != num_pairs:
            raise ValueError("normalized protein and molecule batches must align")
        if not all(
            values.numel() == num_pairs
            for values in (group_ids, pchembl, target_ids, molecule_ids)
        ):
            raise ValueError(
                "contrastive metadata and embeddings must contain the same number of pairs"
            )

        valid_indices = torch.nonzero(group_ids >= 0, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            zero = normalized_protein_embedding.sum() * 0.0
            return zero, zero, zero

        valid_group_ids = group_ids.index_select(0, valid_indices)
        unique_group_ids, ligand_group_ids = torch.unique(
            valid_group_ids,
            sorted=True,
            return_inverse=True,
        )
        num_groups = unique_group_ids.numel()
        group_membership = F.one_hot(
            ligand_group_ids,
            num_classes=num_groups,
        ).transpose(0, 1).bool()
        representative_positions = group_membership.long().argmax(dim=1)
        representative_indices = valid_indices.index_select(
            0,
            representative_positions,
        )
        valid_target_ids = target_ids.index_select(0, valid_indices)
        group_target_ids = valid_target_ids.index_select(
            0,
            representative_positions,
        )
        if not torch.equal(
            group_target_ids.index_select(0, ligand_group_ids),
            valid_target_ids,
        ):
            raise ValueError(
                "every contrastive assay group must contain exactly one target identity"
            )

        group_protein_embeddings = normalized_protein_embedding.index_select(
            0,
            representative_indices,
        )
        ligand_molecule_embeddings = normalized_molecule_embedding.index_select(
            0,
            valid_indices,
        )
        # Ranking divides its diagonal scores by this same temperature. The
        # matrix therefore supplies identical scaled cosine scores to both
        # objectives, matching LigUnity's shared-score construction.
        contrastive_scores = torch.matmul(
            group_protein_embeddings.float(),
            ligand_molecule_embeddings.float().transpose(0, 1),
        ) / float(self._config.ranking_temperature)
        return ligunity_bidirectional_contrastive_loss(
            contrastive_scores,
            pchembl.index_select(0, valid_indices),
            ligand_group_ids,
            group_target_ids,
            molecule_ids.index_select(0, valid_indices),
            active_threshold=self._config.contrastive_active_threshold,
        )

    def _compute_legacy_pair_ranking_loss(
        self,
        ranking_score: torch.Tensor,
        positive_indices: torch.Tensor,
        negative_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Map legacy pairs to independent two-item LigUnity lists."""
        if positive_indices.shape != negative_indices.shape:
            raise ValueError("positive_indices and negative_indices must have the same shape")
        if positive_indices.numel() == 0:
            raise ValueError("positive_indices and negative_indices must contain at least one pair")
        positive_scores = ranking_score.index_select(0, positive_indices.long())
        negative_scores = ranking_score.index_select(0, negative_indices.long())
        pair_scores = torch.stack([positive_scores, negative_scores], dim=1).reshape(-1)
        pair_targets = torch.tensor(
            [1.0, 0.0],
            device=pair_scores.device,
            dtype=pair_scores.dtype,
        ).repeat(positive_scores.numel())
        pair_group_ids = torch.arange(
            positive_scores.numel(),
            device=pair_scores.device,
            dtype=torch.long,
        ).repeat_interleave(2)
        return ligunity_listwise_loss(
            pair_scores,
            pair_targets,
            pair_group_ids,
            temperature=self._config.ranking_temperature,
            min_pchembl_span=0.0,
            min_list_size=2,
            # Legacy pairs carry only positive/negative ordering labels, not
            # measured affinity differences to which a pChEMBL margin applies.
            affinity_margin=0.0,
        )

    def _compute_classification_loss(
        self,
        activity_logits: torch.Tensor,
        activity_labels: torch.Tensor,
    ) -> torch.Tensor:
        pos_weight = torch.tensor(
            self._config.bce_pos_weight,
            dtype=activity_logits.dtype,
            device=activity_logits.device,
        )
        return F.binary_cross_entropy_with_logits(
            activity_logits,
            activity_labels.to(dtype=activity_logits.dtype),
            pos_weight=pos_weight,
        )

    def _score_pooled_pair(
        self,
        pooled_protein: torch.Tensor,
        pooled_molecule: torch.Tensor,
        joint_embedding: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        if self._config.pair_scoring_mode == "mlp":
            if self.ranking_head is None or self.classification_head is None:
                raise RuntimeError("MLP scoring heads are not initialized")
            return (
                self.ranking_head(joint_embedding),
                self.classification_head(joint_embedding),
                None,
                None,
                None,
                None,
            )

        normalized_protein = F.normalize(
            pooled_protein, p=2, dim=-1, eps=1e-6
        )
        normalized_molecule = F.normalize(
            pooled_molecule, p=2, dim=-1, eps=1e-6
        )
        cosine_similarity = (normalized_protein * normalized_molecule).sum(dim=-1)

        if self._config.pair_scoring_mode == "cosine":
            # activity_logits mirrors the score only to preserve the output
            # contract used by evaluation. The ranking-only config assigns no
            # classification loss to it.
            return (
                cosine_similarity,
                cosine_similarity,
                cosine_similarity,
                None,
                normalized_protein,
                normalized_molecule,
            )

        if self.logit_scale is None or self.classification_logit_bias is None:
            raise RuntimeError("Scaled-cosine parameters are not initialized")
        score_scale = self.logit_scale.clamp(
            max=math.log(self._config.cosine_scale_max)
        ).exp()
        ranking_score = score_scale * cosine_similarity
        activity_logits = ranking_score + self.classification_logit_bias
        return (
            ranking_score,
            activity_logits,
            cosine_similarity,
            score_scale,
            normalized_protein,
            normalized_molecule,
        )

    def forward(
        self,
        protein_input_ids: torch.Tensor,
        molecule_input_ids: torch.Tensor,
        protein_attention_mask: Optional[torch.Tensor] = None,
        molecule_attention_mask: Optional[torch.Tensor] = None,
        activity_labels: Optional[torch.Tensor] = None,
        pchembl_values: Optional[torch.Tensor] = None,
        ranking_group_ids: Optional[torch.Tensor] = None,
        contrastive_target_ids: Optional[torch.Tensor] = None,
        contrastive_molecule_ids: Optional[torch.Tensor] = None,
        positive_indices: Optional[torch.Tensor] = None,
        negative_indices: Optional[torch.Tensor] = None,
        return_token_embeddings: bool = False,
        return_dict: bool = True,
    ) -> RewardModelOutput | tuple[torch.Tensor, ...]:
        protein_tokens = self._deduplicated_encode(
            self.encode_protein,
            protein_input_ids,
            protein_attention_mask,
            enabled=self._config.deduplicate_protein_inputs,
        )
        molecule_tokens = self._deduplicated_encode(
            self.encode_molecule,
            molecule_input_ids,
            molecule_attention_mask,
            enabled=self._config.deduplicate_molecule_inputs,
        )

        protein_mask = normalize_mask(protein_attention_mask, protein_tokens)
        molecule_mask = normalize_mask(molecule_attention_mask, molecule_tokens)

        if self._config.pair_scoring_mode == "cosine":
            protein_tokens = self.protein_projection(protein_tokens)
            molecule_tokens = self.molecule_projection(molecule_tokens)
            fused_protein = None
            fused_molecule = None
            pooled_protein = masked_pool(
                protein_tokens,
                protein_mask,
                self._config.pooling_type,
            )
            pooled_molecule = masked_pool(
                molecule_tokens,
                molecule_mask,
                self._config.pooling_type,
            )
        else:
            protein_tokens, molecule_tokens = self._project_tokens(
                protein_tokens,
                molecule_tokens,
            )
            if self.fusion is None:
                raise RuntimeError("Token fusion is not initialized")
            fused_protein, fused_molecule = self.fusion(
                protein_tokens=protein_tokens,
                molecule_tokens=molecule_tokens,
                protein_mask=protein_mask,
                molecule_mask=molecule_mask,
            )
            pooled_protein = masked_pool(
                fused_protein,
                protein_mask,
                self._config.pooling_type,
            )
            pooled_molecule = masked_pool(
                fused_molecule,
                molecule_mask,
                self._config.pooling_type,
            )
        joint_embedding = torch.cat([pooled_protein, pooled_molecule], dim=-1)

        (
            ranking_score,
            activity_logits,
            cosine_similarity,
            score_scale,
            normalized_protein_embedding,
            normalized_molecule_embedding,
        ) = self._score_pooled_pair(
            pooled_protein,
            pooled_molecule,
            joint_embedding,
        )
        if self._config.pair_scoring_mode == "cosine":
            if (
                normalized_protein_embedding is None
                or normalized_molecule_embedding is None
            ):
                raise RuntimeError("Cosine embeddings were not normalized")
            joint_embedding = torch.cat(
                [normalized_protein_embedding, normalized_molecule_embedding],
                dim=-1,
            )
        activity_probability = torch.sigmoid(activity_logits)

        ranking_loss = None
        contrastive_loss = None
        contrastive_protein_to_molecule_loss = None
        contrastive_molecule_to_protein_loss = None
        classification_loss = None
        total_loss = None

        if (
            activity_labels is not None
            and self._config.classification_loss_weight > 0.0
        ):
            classification_loss = self._compute_classification_loss(activity_logits, activity_labels)
            total_loss = classification_loss * self._config.classification_loss_weight

        if self._config.contrastive_loss_weight > 0.0:
            if pchembl_values is None or ranking_group_ids is None:
                raise ValueError(
                    "contrastive learning requires pchembl_values and ranking_group_ids"
                )
            if contrastive_target_ids is None or contrastive_molecule_ids is None:
                raise ValueError(
                    "contrastive learning requires target and molecule identity ids"
                )
            if (
                normalized_protein_embedding is None
                or normalized_molecule_embedding is None
            ):
                raise RuntimeError(
                    "contrastive learning requires normalized protein and molecule embeddings"
                )
            (
                contrastive_loss,
                contrastive_protein_to_molecule_loss,
                contrastive_molecule_to_protein_loss,
            ) = self._compute_contrastive_loss(
                normalized_protein_embedding,
                normalized_molecule_embedding,
                pchembl_values,
                ranking_group_ids,
                contrastive_target_ids,
                contrastive_molecule_ids,
            )
            weighted_contrastive = (
                contrastive_loss * self._config.contrastive_loss_weight
            )
            total_loss = (
                weighted_contrastive
                if total_loss is None
                else total_loss + weighted_contrastive
            )

        if pchembl_values is not None or ranking_group_ids is not None:
            if pchembl_values is None or ranking_group_ids is None:
                raise ValueError("pchembl_values and ranking_group_ids must be provided together")
            if positive_indices is not None or negative_indices is not None:
                raise ValueError(
                    "Use either listwise ranking inputs or legacy pair indices, not both"
                )
            ranking_loss = self._compute_ranking_loss(
                ranking_score,
                pchembl_values,
                ranking_group_ids,
            )
        elif positive_indices is not None or negative_indices is not None:
            if positive_indices is None or negative_indices is None:
                raise ValueError("positive_indices and negative_indices must be provided together")
            ranking_loss = self._compute_legacy_pair_ranking_loss(
                ranking_score,
                positive_indices,
                negative_indices,
            )

        if ranking_loss is not None:
            weighted_ranking = ranking_loss * self._config.ranking_loss_weight
            total_loss = weighted_ranking if total_loss is None else total_loss + weighted_ranking

        outputs = RewardModelOutput(
            ranking_score=ranking_score,
            activity_logits=activity_logits,
            activity_probability=activity_probability,
            joint_embedding=joint_embedding,
            ranking_loss=ranking_loss,
            contrastive_loss=contrastive_loss,
            contrastive_protein_to_molecule_loss=(
                contrastive_protein_to_molecule_loss
            ),
            contrastive_molecule_to_protein_loss=(
                contrastive_molecule_to_protein_loss
            ),
            classification_loss=classification_loss,
            loss=total_loss,
            protein_token_embeddings=protein_tokens if return_token_embeddings else None,
            molecule_token_embeddings=molecule_tokens if return_token_embeddings else None,
            fused_protein_tokens=fused_protein if return_token_embeddings else None,
            fused_molecule_tokens=fused_molecule if return_token_embeddings else None,
            protein_attention_mask=protein_mask if return_token_embeddings else None,
            molecule_attention_mask=molecule_mask if return_token_embeddings else None,
            cosine_similarity=cosine_similarity,
            score_scale=score_scale,
            classification_logit_bias=self.classification_logit_bias,
            normalized_protein_embedding=normalized_protein_embedding,
            normalized_molecule_embedding=normalized_molecule_embedding,
        )
        return outputs if return_dict else outputs.to_tuple()

    def num_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters())

    def num_trainable_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def score_pairs(
        self,
        protein_sequences: Sequence[str],
        molecule_sequences: Sequence[str],
        activity_labels: Optional[torch.Tensor] = None,
        pchembl_values: Optional[torch.Tensor] = None,
        ranking_group_ids: Optional[torch.Tensor] = None,
        contrastive_target_ids: Optional[torch.Tensor] = None,
        contrastive_molecule_ids: Optional[torch.Tensor] = None,
        positive_indices: Optional[torch.Tensor] = None,
        negative_indices: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        return_token_embeddings: bool = False,
        return_dict: bool = True,
    ) -> RewardModelOutput | tuple[torch.Tensor, ...]:
        if len(protein_sequences) != len(molecule_sequences):
            raise ValueError("protein_sequences and molecule_sequences must have the same length")

        target_device = device if device is not None else self._model_device()
        protein_batch = self.tokenize_proteins(protein_sequences, device=target_device)
        molecule_batch = self.tokenize_molecules(molecule_sequences, device=target_device)

        if activity_labels is not None:
            activity_labels = activity_labels.to(target_device)
        if pchembl_values is not None:
            pchembl_values = pchembl_values.to(target_device)
        if ranking_group_ids is not None:
            ranking_group_ids = ranking_group_ids.to(target_device)
        if contrastive_target_ids is not None:
            contrastive_target_ids = contrastive_target_ids.to(target_device)
        if contrastive_molecule_ids is not None:
            contrastive_molecule_ids = contrastive_molecule_ids.to(target_device)
        if positive_indices is not None:
            positive_indices = positive_indices.to(target_device)
        if negative_indices is not None:
            negative_indices = negative_indices.to(target_device)

        return self(
            protein_input_ids=protein_batch["input_ids"],
            protein_attention_mask=protein_batch.get("attention_mask"),
            molecule_input_ids=molecule_batch["input_ids"],
            molecule_attention_mask=molecule_batch.get("attention_mask"),
            activity_labels=activity_labels,
            pchembl_values=pchembl_values,
            ranking_group_ids=ranking_group_ids,
            contrastive_target_ids=contrastive_target_ids,
            contrastive_molecule_ids=contrastive_molecule_ids,
            positive_indices=positive_indices,
            negative_indices=negative_indices,
            return_token_embeddings=return_token_embeddings,
            return_dict=return_dict,
        )
