from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RewardModelConfig
from .encoders import LoadedEncoder, batch_encode_texts, encode_tokens, load_encoder_bundle
from .fusion import RewardMLPHead, TokenFusion, masked_pool, normalize_mask
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
            molecule_bundle = load_encoder_bundle(
                name_or_path=self._config.molecule_model_name_or_path,
                tokenizer_name_or_path=self._config.molecule_tokenizer_name_or_path,
            )

        self.protein_tokenizer = protein_bundle.tokenizer
        self.molecule_tokenizer = molecule_bundle.tokenizer
        self.protein_encoder = protein_bundle.model
        self.molecule_encoder = molecule_bundle.model

        self._config.protein_hidden_size = self._config.protein_hidden_size or protein_bundle.hidden_size
        self._config.molecule_hidden_size = self._config.molecule_hidden_size or molecule_bundle.hidden_size
        self._config.validate()

        self.protein_projection = nn.Linear(self._config.protein_hidden_size, self._config.fusion_hidden_dim)
        self.molecule_projection = nn.Linear(self._config.molecule_hidden_size, self._config.fusion_hidden_dim)
        self.protein_norm = nn.LayerNorm(self._config.fusion_hidden_dim)
        self.molecule_norm = nn.LayerNorm(self._config.fusion_hidden_dim)

        self.fusion = TokenFusion(
            hidden_dim=self._config.fusion_hidden_dim,
            num_heads=self._config.fusion_num_heads,
        )
        head_input_dim = self._config.fusion_hidden_dim * 2
        self.ranking_head = RewardMLPHead(
            input_dim=head_input_dim,
            hidden_dim=self._config.fusion_hidden_dim,
            dropout=self._config.dropout,
        )
        self.classification_head = RewardMLPHead(
            input_dim=head_input_dim,
            hidden_dim=self._config.fusion_hidden_dim,
            dropout=self._config.dropout,
        )

    @property
    def config(self) -> RewardModelConfig:
        return self._config

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
        protein_tokens = self.protein_norm(self.protein_projection(protein_tokens))
        molecule_tokens = self.molecule_norm(self.molecule_projection(molecule_tokens))
        return protein_tokens, molecule_tokens

    def _compute_pair_loss(
        self,
        ranking_score: torch.Tensor,
        positive_indices: torch.Tensor,
        negative_indices: torch.Tensor,
    ) -> torch.Tensor:
        if positive_indices.shape != negative_indices.shape:
            raise ValueError("positive_indices and negative_indices must have the same shape")
        if positive_indices.numel() == 0:
            raise ValueError("positive_indices and negative_indices must contain at least one pair")
        diffs = ranking_score[positive_indices.long()] - ranking_score[negative_indices.long()]
        return -F.logsigmoid(diffs).mean()

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

    def forward(
        self,
        protein_input_ids: torch.Tensor,
        molecule_input_ids: torch.Tensor,
        protein_attention_mask: Optional[torch.Tensor] = None,
        molecule_attention_mask: Optional[torch.Tensor] = None,
        activity_labels: Optional[torch.Tensor] = None,
        positive_indices: Optional[torch.Tensor] = None,
        negative_indices: Optional[torch.Tensor] = None,
        return_token_embeddings: bool = False,
        return_dict: bool = True,
    ) -> RewardModelOutput | tuple[torch.Tensor, ...]:
        protein_tokens = self.encode_protein(
            input_ids=protein_input_ids,
            attention_mask=protein_attention_mask,
        )
        molecule_tokens = self.encode_molecule(
            input_ids=molecule_input_ids,
            attention_mask=molecule_attention_mask,
        )

        protein_mask = normalize_mask(protein_attention_mask, protein_tokens)
        molecule_mask = normalize_mask(molecule_attention_mask, molecule_tokens)

        protein_tokens, molecule_tokens = self._project_tokens(protein_tokens, molecule_tokens)
        fused_protein, fused_molecule = self.fusion(
            protein_tokens=protein_tokens,
            molecule_tokens=molecule_tokens,
            protein_mask=protein_mask,
            molecule_mask=molecule_mask,
        )

        pooled_protein = masked_pool(fused_protein, protein_mask, self._config.pooling_type)
        pooled_molecule = masked_pool(fused_molecule, molecule_mask, self._config.pooling_type)
        joint_embedding = torch.cat([pooled_protein, pooled_molecule], dim=-1)

        ranking_score = self.ranking_head(joint_embedding)
        activity_logits = self.classification_head(joint_embedding)
        activity_probability = torch.sigmoid(activity_logits)

        pair_loss = None
        classification_loss = None
        total_loss = None

        if activity_labels is not None:
            classification_loss = self._compute_classification_loss(activity_logits, activity_labels)
            total_loss = classification_loss * self._config.classification_loss_weight

        if positive_indices is not None or negative_indices is not None:
            if positive_indices is None or negative_indices is None:
                raise ValueError("positive_indices and negative_indices must be provided together")
            pair_loss = self._compute_pair_loss(ranking_score, positive_indices, negative_indices)
            weighted_pair = pair_loss * self._config.pair_loss_weight
            total_loss = weighted_pair if total_loss is None else total_loss + weighted_pair

        outputs = RewardModelOutput(
            ranking_score=ranking_score,
            activity_logits=activity_logits,
            activity_probability=activity_probability,
            joint_embedding=joint_embedding,
            pair_loss=pair_loss,
            classification_loss=classification_loss,
            loss=total_loss,
            protein_token_embeddings=protein_tokens if return_token_embeddings else None,
            molecule_token_embeddings=molecule_tokens if return_token_embeddings else None,
            fused_protein_tokens=fused_protein if return_token_embeddings else None,
            fused_molecule_tokens=fused_molecule if return_token_embeddings else None,
            protein_attention_mask=protein_mask if return_token_embeddings else None,
            molecule_attention_mask=molecule_mask if return_token_embeddings else None,
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
            positive_indices=positive_indices,
            negative_indices=negative_indices,
            return_token_embeddings=return_token_embeddings,
            return_dict=return_dict,
        )
