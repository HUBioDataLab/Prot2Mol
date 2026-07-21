from __future__ import annotations

import math

import torch


def _validate_listwise_inputs(
    ranking_scores: torch.Tensor,
    pchembl_values: torch.Tensor,
    ranking_group_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = ranking_scores.reshape(-1)
    if not scores.is_floating_point():
        raise ValueError("ranking_scores must be floating point")
    targets = pchembl_values.reshape(-1).to(device=scores.device)
    group_ids = ranking_group_ids.reshape(-1).to(device=scores.device, dtype=torch.long)
    if scores.shape != targets.shape or scores.shape != group_ids.shape:
        raise ValueError(
            "ranking_scores, pchembl_values, and ranking_group_ids must have the same shape"
        )
    if not torch.isfinite(scores).all():
        raise ValueError("ranking_scores must contain only finite values")
    if not torch.isfinite(targets).all():
        raise ValueError("pchembl_values must contain only finite values")
    return scores, targets, group_ids


def _tie_averaged_position_weight(
    *,
    first_position: int,
    tie_count: int,
    list_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return LigUnity's top-decay weight without ordering equal-affinity ligands."""
    positions = torch.arange(
        first_position,
        first_position + tie_count,
        device=device,
        dtype=dtype,
    )
    weights = 1.0 / (math.sqrt(list_size) * torch.log(positions + 1.0))
    return weights.mean()


def ligunity_listwise_loss(
    ranking_scores: torch.Tensor,
    pchembl_values: torch.Tensor,
    ranking_group_ids: torch.Tensor,
    *,
    temperature: float = 1.0,
    min_pchembl_span: float = 0.5,
) -> torch.Tensor:
    """Compute LigUnity's weighted Plackett-Luce ranking objective.

    Group ids below zero mark classification-only observations. For a unique
    affinity ordering this follows equations 2, 4, and 5 of Feng et al. (2025),
    "Hierarchical affinity landscape navigation through learning a shared
    pocket-ligand space". Equal-affinity ligands share the mean positional
    weight of their occupied ranks, making the objective permutation invariant
    within ties rather than imposing an arbitrary order.
    """
    if temperature <= 0.0 or not math.isfinite(float(temperature)):
        raise ValueError("temperature must be finite and > 0")
    if min_pchembl_span < 0.0 or not math.isfinite(float(min_pchembl_span)):
        raise ValueError("min_pchembl_span must be finite and >= 0")

    scores, targets, group_ids = _validate_listwise_inputs(
        ranking_scores,
        pchembl_values,
        ranking_group_ids,
    )
    computation_dtype = (
        torch.float32
        if scores.dtype in (torch.float16, torch.bfloat16)
        else scores.dtype
    )
    scaled_scores = scores.to(dtype=computation_dtype) / float(temperature)
    targets = targets.to(dtype=computation_dtype)
    list_losses: list[torch.Tensor] = []

    valid_indices = torch.nonzero(group_ids >= 0, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        return scores.sum() * 0.0
    group_order = torch.argsort(
        group_ids.index_select(0, valid_indices),
        stable=True,
    )
    grouped_indices = valid_indices.index_select(0, group_order)
    sorted_group_ids = group_ids.index_select(0, grouped_indices)
    _, group_counts = torch.unique_consecutive(
        sorted_group_ids,
        return_counts=True,
    )

    group_start = 0
    for group_count_tensor in group_counts:
        group_count = int(group_count_tensor.item())
        group_indices = grouped_indices[group_start : group_start + group_count]
        group_start += group_count
        group_scores = scaled_scores.index_select(0, group_indices)
        group_targets = targets.index_select(0, group_indices)
        list_size = int(group_scores.numel())
        if list_size < 2:
            continue
        if float((group_targets.max() - group_targets.min()).detach().item()) < min_pchembl_span:
            continue

        affinity_order = torch.argsort(group_targets, descending=True, stable=True)
        ordered_targets = group_targets.index_select(0, affinity_order)
        ordered_scores = group_scores.index_select(0, affinity_order)
        suffix_log_denominators = torch.logcumsumexp(
            ordered_scores.flip(0),
            dim=0,
        ).flip(0)
        _, tie_counts = torch.unique_consecutive(
            ordered_targets,
            return_counts=True,
        )
        first_position = 1
        group_loss = group_scores.sum() * 0.0
        tie_start = 0
        for tie_count_tensor in tie_counts:
            tie_count = int(tie_count_tensor.item())
            log_denominator = suffix_log_denominators[tie_start]
            position_weight = _tie_averaged_position_weight(
                first_position=first_position,
                tie_count=tie_count,
                list_size=list_size,
                device=group_scores.device,
                dtype=group_scores.dtype,
            )
            group_loss = group_loss + position_weight * (
                tie_count * log_denominator
                - ordered_scores[tie_start : tie_start + tie_count].sum()
            )
            first_position += tie_count
            tie_start += tie_count
        list_losses.append(group_loss)

    if not list_losses:
        return scores.sum() * 0.0
    return torch.stack(list_losses).mean()
