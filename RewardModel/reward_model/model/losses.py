from __future__ import annotations

import math

import torch
import torch.nn.functional as F


MIN_LISTWISE_LIGANDS = 3
DEFAULT_RANKING_AFFINITY_MARGIN = math.log10(3.0)
DEFAULT_CONTRASTIVE_ACTIVE_THRESHOLD = 5.0
CONTRASTIVE_MASK_VALUE = -1e9


def _validate_contrastive_inputs(
    contrastive_scores: torch.Tensor,
    pchembl_values: torch.Tensor,
    ligand_group_ids: torch.Tensor,
    group_target_ids: torch.Tensor,
    molecule_identity_ids: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if contrastive_scores.ndim != 2:
        raise ValueError("contrastive_scores must have shape [num_groups, num_ligands]")
    if not contrastive_scores.is_floating_point():
        raise ValueError("contrastive_scores must be floating point")
    if not torch.isfinite(contrastive_scores).all():
        raise ValueError("contrastive_scores must contain only finite values")

    num_groups, num_ligands = contrastive_scores.shape
    if num_groups <= 0 or num_ligands <= 0:
        raise ValueError("contrastive_scores must contain at least one group and ligand")

    device = contrastive_scores.device
    targets = pchembl_values.reshape(-1).to(device=device)
    owners = ligand_group_ids.reshape(-1).to(device=device, dtype=torch.long)
    target_ids = group_target_ids.reshape(-1).to(device=device, dtype=torch.long)
    molecule_ids = molecule_identity_ids.reshape(-1).to(
        device=device,
        dtype=torch.long,
    )
    if targets.numel() != num_ligands:
        raise ValueError("pchembl_values must contain one value per ligand")
    if owners.numel() != num_ligands:
        raise ValueError("ligand_group_ids must contain one value per ligand")
    if target_ids.numel() != num_groups:
        raise ValueError("group_target_ids must contain one value per group")
    if molecule_ids.numel() != num_ligands:
        raise ValueError("molecule_identity_ids must contain one value per ligand")
    if not torch.isfinite(targets).all():
        raise ValueError("pchembl_values must contain only finite values")
    if (owners < 0).any() or (owners >= num_groups).any():
        raise ValueError("ligand_group_ids must be in [0, num_groups)")
    present_groups = torch.unique(owners, sorted=True)
    expected_groups = torch.arange(num_groups, device=device, dtype=torch.long)
    if not torch.equal(present_groups, expected_groups):
        raise ValueError("every contrastive group must own at least one ligand")

    computation_dtype = (
        torch.float32
        if contrastive_scores.dtype in (torch.float16, torch.bfloat16)
        else contrastive_scores.dtype
    )
    return (
        contrastive_scores.to(dtype=computation_dtype),
        targets.to(dtype=computation_dtype),
        owners,
        target_ids,
        molecule_ids,
    )


def ligunity_bidirectional_contrastive_loss(
    contrastive_scores: torch.Tensor,
    pchembl_values: torch.Tensor,
    ligand_group_ids: torch.Tensor,
    group_target_ids: torch.Tensor,
    molecule_identity_ids: torch.Tensor,
    *,
    active_threshold: float = DEFAULT_CONTRASTIVE_ACTIVE_THRESHOLD,
    strict_active_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror LigUnity's masked bidirectional in-batch retrieval objective.

    ``contrastive_scores`` is the shared, already-scaled cosine matrix with one
    protein/assay row and one column per sampled ligand. For protein-to-molecule
    retrieval, each measured active ligand competes against ligands from other
    targets while the other ligands from its own assay are hidden. By default,
    this follows LigUnity's released code: multi-ligand assay observations below
    the active threshold are skipped only in this direction, while
    molecule-to-protein retrieval uses every sampled ligand. When
    ``strict_active_only`` is enabled, both directions retain only observations
    whose pChEMBL is strictly greater than ``active_threshold``.

    Cross-assay entries are masked when they have the same target identity or
    duplicate a molecule measured in the query assay. Directional losses use
    LigUnity's ``1 / sqrt(num_assay_ligands)`` weighting and are averaged over
    assay rows, matching its trainer-level sample-size normalization.

    Returns ``(total, protein_to_molecule, molecule_to_protein)`` where total is
    the sum of the two directional losses before the external objective weight.
    """
    if not math.isfinite(float(active_threshold)):
        raise ValueError("active_threshold must be finite")
    if not isinstance(strict_active_only, bool):
        raise ValueError("strict_active_only must be a boolean")

    (
        scores,
        targets,
        owners,
        target_ids,
        molecule_ids,
    ) = _validate_contrastive_inputs(
        contrastive_scores,
        pchembl_values,
        ligand_group_ids,
        group_target_ids,
        molecule_identity_ids,
    )
    num_groups, num_ligands = scores.shape

    # LigUnity masks false negatives before applying either retrieval direction.
    group_membership = F.one_hot(
        owners,
        num_classes=num_groups,
    ).transpose(0, 1).bool()
    ligand_target_ids = target_ids.index_select(0, owners)
    same_target = target_ids[:, None] == ligand_target_ids[None, :]
    same_molecule = molecule_ids[:, None] == molecule_ids[None, :]
    duplicate_molecule = (
        group_membership.to(dtype=scores.dtype)
        @ same_molecule.to(dtype=scores.dtype)
    ) > 0
    false_negative_mask = (~group_membership) & (
        same_target | duplicate_molecule
    )
    masked_scores = scores.masked_fill(
        false_negative_mask,
        CONTRASTIVE_MASK_VALUE,
    )

    ligand_positions = torch.arange(
        num_ligands,
        device=scores.device,
        dtype=torch.long,
    )
    group_sizes = torch.bincount(
        owners,
        minlength=num_groups,
    ).to(dtype=scores.dtype)
    ligand_weights = group_sizes.rsqrt().index_select(0, owners)

    # Select each ligand's owning protein row, then hide the other ligands from
    # that same assay. This vectorizes LigUnity's per-positive retrieval loop.
    protein_to_molecule_logits = masked_scores.index_select(0, owners)
    same_owner = owners[:, None] == owners[None, :]
    same_owner.fill_diagonal_(False)
    protein_to_molecule_logits = protein_to_molecule_logits.masked_fill(
        same_owner,
        CONTRASTIVE_MASK_VALUE,
    )
    per_ligand_protein_to_molecule = F.cross_entropy(
        protein_to_molecule_logits,
        ligand_positions,
        reduction="none",
    )
    if strict_active_only:
        eligible_protein_to_molecule = targets > float(active_threshold)
    else:
        eligible_protein_to_molecule = (
            group_sizes.index_select(0, owners) == 1
        ) | (targets >= float(active_threshold))
    protein_to_molecule = (
        per_ligand_protein_to_molecule[eligible_protein_to_molecule]
        * ligand_weights[eligible_protein_to_molecule]
    ).sum() / num_groups

    per_ligand_molecule_to_protein = F.cross_entropy(
        masked_scores.transpose(0, 1),
        owners,
        reduction="none",
    )
    if strict_active_only:
        molecule_to_protein = (
            per_ligand_molecule_to_protein[eligible_protein_to_molecule]
            * ligand_weights[eligible_protein_to_molecule]
        ).sum() / num_groups
    else:
        molecule_to_protein = (
            per_ligand_molecule_to_protein * ligand_weights
        ).sum() / num_groups
    return (
        protein_to_molecule + molecule_to_protein,
        protein_to_molecule,
        molecule_to_protein,
    )


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
    min_list_size: int = MIN_LISTWISE_LIGANDS,
    affinity_margin: float = DEFAULT_RANKING_AFFINITY_MARGIN,
) -> torch.Tensor:
    """Compute LigUnity's weighted Plackett-Luce ranking objective.

    Group ids below zero mark classification-only observations. For a unique
    affinity ordering this follows equations 2, 4, and 5 of Feng et al. (2025),
    "Hierarchical affinity landscape navigation through learning a shared
    pocket-ligand space". Equal-affinity ligands share the mean positional
    weight of their occupied ranks, making the objective permutation invariant
    within ties rather than imposing an arbitrary order. By default, groups
    with fewer than three ligands are classification-only, matching LigUnity's
    listwise training boundary. At each selection step, ligands within the
    affinity margin are omitted from the denominator. The default margin is
    log10(3) pChEMBL units, so only ligands measured as more than threefold
    weaker compete with the selected ligand.
    """
    if temperature <= 0.0 or not math.isfinite(float(temperature)):
        raise ValueError("temperature must be finite and > 0")
    if min_pchembl_span < 0.0 or not math.isfinite(float(min_pchembl_span)):
        raise ValueError("min_pchembl_span must be finite and >= 0")
    if min_list_size < 2:
        raise ValueError("min_list_size must be >= 2")
    if affinity_margin < 0.0 or not math.isfinite(float(affinity_margin)):
        raise ValueError("affinity_margin must be finite and >= 0")

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
        if list_size < min_list_size:
            continue
        if float((group_targets.max() - group_targets.min()).detach().item()) < min_pchembl_span:
            continue

        affinity_order = torch.argsort(group_targets, descending=True, stable=True)
        ordered_targets = group_targets.index_select(0, affinity_order)
        ordered_scores = group_scores.index_select(0, affinity_order)
        _, tie_counts = torch.unique_consecutive(
            ordered_targets,
            return_counts=True,
        )
        first_position = 1
        group_loss = group_scores.sum() * 0.0
        tie_start = 0
        for tie_count_tensor in tie_counts:
            tie_count = int(tie_count_tensor.item())
            tie_end = tie_start + tie_count
            tie_target = ordered_targets[tie_start]
            tie_scores = ordered_scores[tie_start:tie_end]
            weaker_scores = ordered_scores[
                ordered_targets < tie_target - float(affinity_margin)
            ]
            position_weight = _tie_averaged_position_weight(
                first_position=first_position,
                tie_count=tie_count,
                list_size=list_size,
                device=group_scores.device,
                dtype=group_scores.dtype,
            )
            if weaker_scores.numel() > 0:
                weaker_logsumexp = torch.logsumexp(weaker_scores, dim=0)
                selection_losses = (
                    torch.logaddexp(tie_scores, weaker_logsumexp) - tie_scores
                )
                group_loss = group_loss + position_weight * selection_losses.sum()
            first_position += tie_count
            tie_start = tie_end
        list_losses.append(group_loss)

    if not list_losses:
        return scores.sum() * 0.0
    return torch.stack(list_losses).mean()
