from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_mask(mask: Optional[torch.Tensor], tokens: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones(tokens.size(0), tokens.size(1), device=tokens.device, dtype=torch.bool)
    return mask.to(device=tokens.device).bool()


def masked_pool(tokens: torch.Tensor, mask: torch.Tensor, pooling_type: str) -> torch.Tensor:
    mask = normalize_mask(mask, tokens)
    if pooling_type == "cls":
        return tokens[:, 0]
    if pooling_type == "mean_all_tok":
        return tokens.mean(dim=1)

    weights = mask.unsqueeze(-1).to(tokens.dtype)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return (tokens * weights).sum(dim=1) / denom


class TokenFusion(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        attention_backend: str = "manual",
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_size = hidden_dim // num_heads
        self.attention_backend = attention_backend
        if self.attention_backend not in {"manual", "sdpa"}:
            raise ValueError("attention_backend must be 'manual' or 'sdpa'")

        self.query_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_p = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.query_m = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_m = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_m = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def _apply_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, self.head_size)

    def _masked_softmax(
        self,
        logits: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_pairs = row_mask.unsqueeze(2).unsqueeze(-1) & col_mask.unsqueeze(1).unsqueeze(-1)
        mask_value = torch.finfo(logits.dtype).min
        masked_logits = torch.where(valid_pairs, logits, torch.full_like(logits, mask_value))
        alpha = torch.softmax(masked_logits, dim=2)
        return torch.where(valid_pairs, alpha, torch.zeros_like(alpha))

    def _sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        valid_pairs = row_mask[:, None, :, None] & col_mask[:, None, None, :]
        fallback_column = torch.zeros_like(valid_pairs)
        fallback_column[..., 0] = True
        valid_pairs = valid_pairs | (
            ~valid_pairs.any(dim=-1, keepdim=True) & fallback_column
        )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=valid_pairs,
            dropout_p=0.0,
            is_causal=False,
        )
        return attended.transpose(1, 2).flatten(-2)

    def forward(
        self,
        protein_tokens: torch.Tensor,
        molecule_tokens: torch.Tensor,
        protein_mask: torch.Tensor,
        molecule_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        protein_mask = normalize_mask(protein_mask, protein_tokens)
        molecule_mask = normalize_mask(molecule_mask, molecule_tokens)

        protein_q = self._apply_heads(self.query_p(protein_tokens))
        protein_k = self._apply_heads(self.key_p(protein_tokens))
        protein_v = self._apply_heads(self.value_p(protein_tokens))

        molecule_q = self._apply_heads(self.query_m(molecule_tokens))
        molecule_k = self._apply_heads(self.key_m(molecule_tokens))
        molecule_v = self._apply_heads(self.value_m(molecule_tokens))

        if self.attention_backend == "sdpa":
            fused_protein = (
                self._sdpa(
                    protein_q,
                    protein_k,
                    protein_v,
                    protein_mask,
                    protein_mask,
                )
                + self._sdpa(
                    protein_q,
                    molecule_k,
                    molecule_v,
                    protein_mask,
                    molecule_mask,
                )
            ) / 2.0
            fused_molecule = (
                self._sdpa(
                    molecule_q,
                    protein_k,
                    protein_v,
                    molecule_mask,
                    protein_mask,
                )
                + self._sdpa(
                    molecule_q,
                    molecule_k,
                    molecule_v,
                    molecule_mask,
                    molecule_mask,
                )
            ) / 2.0
        else:
            logits_pp = torch.einsum(
                "blhd,bkhd->blkh", protein_q, protein_k
            ) / math.sqrt(self.head_size)
            logits_pm = torch.einsum(
                "blhd,bkhd->blkh", protein_q, molecule_k
            ) / math.sqrt(self.head_size)
            logits_mp = torch.einsum(
                "blhd,bkhd->blkh", molecule_q, protein_k
            ) / math.sqrt(self.head_size)
            logits_mm = torch.einsum(
                "blhd,bkhd->blkh", molecule_q, molecule_k
            ) / math.sqrt(self.head_size)

            alpha_pp = self._masked_softmax(logits_pp, protein_mask, protein_mask)
            alpha_pm = self._masked_softmax(logits_pm, protein_mask, molecule_mask)
            alpha_mp = self._masked_softmax(logits_mp, molecule_mask, protein_mask)
            alpha_mm = self._masked_softmax(logits_mm, molecule_mask, molecule_mask)

            fused_protein = (
                torch.einsum("blkh,bkhd->blhd", alpha_pp, protein_v).flatten(-2)
                + torch.einsum("blkh,bkhd->blhd", alpha_pm, molecule_v).flatten(-2)
            ) / 2.0
            fused_molecule = (
                torch.einsum("blkh,bkhd->blhd", alpha_mp, protein_v).flatten(-2)
                + torch.einsum("blkh,bkhd->blhd", alpha_mm, molecule_v).flatten(-2)
            ) / 2.0

        fused_protein = fused_protein * protein_mask.unsqueeze(-1).to(fused_protein.dtype)
        fused_molecule = fused_molecule * molecule_mask.unsqueeze(-1).to(fused_molecule.dtype)
        return fused_protein, fused_molecule


class RewardMLPHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.ln1 = nn.LayerNorm(hidden_dims[0])
        self.hidden_layers = nn.ModuleList(
            nn.Linear(input_dim, output_dim)
            for input_dim, output_dim in zip(hidden_dims, hidden_dims[1:])
        )
        self.hidden_norms = nn.ModuleList(
            nn.LayerNorm(hidden_dim) for hidden_dim in hidden_dims[1:]
        )
        self.fc2 = nn.Linear(hidden_dims[-1], 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.ln1(F.gelu(self.fc1(x))))
        for layer, norm in zip(self.hidden_layers, self.hidden_norms):
            x = self.dropout(norm(F.gelu(layer(x))))
        return self.fc2(x).squeeze(-1)
