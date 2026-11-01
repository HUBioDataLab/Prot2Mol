"""Explicit encoder adaptation controls for the paired-ranking experiments."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class LowRankLinear(nn.Module):
    """Frozen affine map plus alpha/r times a trainable low-rank update.

    Uses the reference LoRA initialization convention: Kaiming-uniform A and
    zero B. We keep the update separate in both train and eval modes so that
    checkpoint loading and repeated eval calls never merge weights in place.
    https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
    """
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank <= 0 or rank > min(base.in_features, base.out_features):
            raise ValueError("LoRA rank must fit both affine dimensions")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        self.base = base.requires_grad_(False)
        self.rank, self.scaling = rank, float(alpha) / rank
        self.in_features, self.out_features = base.in_features, base.out_features
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, inputs):
        return self.base(inputs) + F.linear(F.linear(inputs, self.lora_A), self.lora_B) * self.scaling


def configure_encoder(encoder: nn.Module, *, frozen: bool, last_layers: int | None,
                      lora_rank: int, lora_alpha: float):
    """Return frozen submodules and a serializable adaptation description.

    Call under a forked RNG so adapter initialization cannot change the common
    randomly initialized projection/fusion/head weights across ablation arms.
    """
    if sum((frozen, last_layers is not None, lora_rank > 0)) > 1:
        raise ValueError("Choose one encoder adaptation strategy")
    encoder.requires_grad_(not frozen)
    keep_eval = []
    report = {"mode": "frozen" if frozen else "full", "adapted_modules": []}
    if frozen:
        keep_eval = [encoder]
    elif last_layers is not None:
        blocks = getattr(getattr(encoder, "encoder", None), "layer", None)
        if not isinstance(blocks, nn.ModuleList) or not 0 < last_layers <= len(blocks):
            raise ValueError("Partial unfreezing requires a supported encoder.layer stack and a valid layer count")
        encoder.requires_grad_(False)
        for number in range(len(blocks) - last_layers, len(blocks)):
            blocks[number].requires_grad_(True)
            report["adapted_modules"].append(f"encoder.layer.{number}")
        final_norm = getattr(encoder.encoder, "emb_layer_norm_after", None)
        if isinstance(final_norm, nn.Module):
            final_norm.requires_grad_(True)
            report["adapted_modules"].append("encoder.emb_layer_norm_after")
        keep_eval = list(blocks[:-last_layers])
        embeddings = getattr(encoder, "embeddings", None)
        if isinstance(embeddings, nn.Module):
            keep_eval.append(embeddings)
        report.update(mode="partial", last_layers=last_layers,
                      dropout_policy="frozen embeddings and lower blocks eval; adapted blocks train")
    elif lora_rank:
        encoder.requires_grad_(False)
        candidates = [(name, layer) for name, layer in encoder.named_modules()
                      if isinstance(layer, nn.Linear) and name.endswith((".attention.self.query", ".attention.self.value"))]
        if not candidates:
            raise ValueError("No supported attention query/value matrices found for LoRA")
        for name, layer in candidates:
            parent_name, leaf = name.rsplit(".", 1)
            setattr(encoder.get_submodule(parent_name), leaf, LowRankLinear(layer, lora_rank, lora_alpha))
            report["adapted_modules"].append(name)
        report.update(mode="lora", rank=lora_rank, alpha=lora_alpha,
                      dropout_policy="original encoder dropout follows train/eval mode; no extra adapter dropout")
    for module in keep_eval:
        module.eval()
    report["total_parameters"] = sum(p.numel() for p in encoder.parameters())
    report["trainable_parameters"] = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    return tuple(keep_eval), report
