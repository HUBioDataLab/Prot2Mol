from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data import RewardPairCollator, RewardPairDataset
from ..model.losses import (
    DEFAULT_RANKING_AFFINITY_MARGIN,
    MIN_LISTWISE_LIGANDS,
    ligunity_bidirectional_contrastive_loss,
    ligunity_listwise_loss,
)

ASSAY_SPEARMAN_LOG_FILENAME = "eval_assay_spearman.jsonl"


def build_complete_coverage_ranking_partitions(
    *,
    pchembl_values: torch.Tensor,
    ranking_group_ids: torch.Tensor,
    max_list_size: int = 16,
    num_partitions: int = 3,
    seed: int = 42,
    min_pchembl_span: float = 0.5,
) -> list[torch.Tensor]:
    """Build fixed balanced ranking lists that cover every eligible row once.

    Each returned tensor describes one complete validation partition. Negative
    ids mark assays that are ineligible for listwise ranking. Eligible assays
    are shuffled deterministically and split into balanced lists whose sizes
    differ by at most one, avoiding unusable one- or two-item remainders.
    """
    targets = pchembl_values.reshape(-1).detach().cpu().float()
    assay_ids = ranking_group_ids.reshape(-1).detach().cpu().long()
    if targets.shape != assay_ids.shape:
        raise ValueError("pchembl_values and ranking_group_ids must have the same shape")
    if not torch.isfinite(targets).all():
        raise ValueError("pchembl_values must contain only finite values")
    if (assay_ids < 0).any():
        raise ValueError("ranking_group_ids must be non-negative for evaluation")
    if max_list_size < 5:
        raise ValueError("max_list_size must be >= 5 for complete balanced coverage")
    if num_partitions <= 0:
        raise ValueError("num_partitions must be > 0")
    if min_pchembl_span < 0.0 or not math.isfinite(float(min_pchembl_span)):
        raise ValueError("min_pchembl_span must be finite and >= 0")

    partitions = [torch.full_like(assay_ids, -1) for _ in range(num_partitions)]
    next_list_ids = [0 for _ in range(num_partitions)]
    max_seed = (1 << 63) - 1

    for assay_id_tensor in torch.unique(assay_ids, sorted=True):
        assay_id = int(assay_id_tensor.item())
        assay_indices = torch.nonzero(
            assay_ids == assay_id,
            as_tuple=False,
        ).flatten()
        assay_targets = targets.index_select(0, assay_indices)
        assay_size = int(assay_indices.numel())
        if assay_size < MIN_LISTWISE_LIGANDS:
            continue
        if float((assay_targets.max() - assay_targets.min()).item()) < float(
            min_pchembl_span
        ):
            continue

        list_count = math.ceil(assay_size / max_list_size)
        base_size, larger_list_count = divmod(assay_size, list_count)
        list_sizes = [base_size + 1] * larger_list_count + [base_size] * (
            list_count - larger_list_count
        )
        if min(list_sizes) < MIN_LISTWISE_LIGANDS:
            raise ValueError(
                "max_list_size cannot cover this assay without a ranking list "
                f"smaller than {MIN_LISTWISE_LIGANDS}"
            )

        for partition_index, partition_ids in enumerate(partitions):
            generator = torch.Generator()
            partition_seed = (
                int(seed)
                + 1_000_003 * (partition_index + 1)
                + 97_409 * (assay_id + 1)
            ) % max_seed
            generator.manual_seed(partition_seed)
            order = torch.randperm(assay_size, generator=generator)
            shuffled_indices = assay_indices.index_select(0, order)
            cursor = 0
            for list_size in list_sizes:
                list_indices = shuffled_indices[cursor : cursor + list_size]
                partition_ids[list_indices] = next_list_ids[partition_index]
                next_list_ids[partition_index] += 1
                cursor += list_size
            if cursor != assay_size:
                raise RuntimeError("validation partition did not cover the complete assay")

    return partitions


def build_complete_coverage_contrastive_partitions(
    *,
    contrastive_group_ids: torch.Tensor,
    max_list_size: int = 16,
    num_partitions: int = 3,
    seed: int = 42,
) -> list[torch.Tensor]:
    """Build deterministic bounded lists covering every assay observation.

    Unlike ranking partitions, contrastive partitions deliberately retain
    singleton, two-ligand, and narrow-affinity-span assays. This is the data
    boundary used by LigUnity's retrieval objective.
    """
    assay_ids = contrastive_group_ids.reshape(-1).detach().cpu().long()
    if (assay_ids < 0).any():
        raise ValueError(
            "contrastive_group_ids must be non-negative for evaluation"
        )
    if max_list_size <= 0:
        raise ValueError("max_list_size must be > 0")
    if num_partitions <= 0:
        raise ValueError("num_partitions must be > 0")

    partitions = [torch.full_like(assay_ids, -1) for _ in range(num_partitions)]
    next_list_ids = [0 for _ in range(num_partitions)]
    max_seed = (1 << 63) - 1
    for assay_id_tensor in torch.unique(assay_ids, sorted=True):
        assay_id = int(assay_id_tensor.item())
        assay_indices = torch.nonzero(
            assay_ids == assay_id,
            as_tuple=False,
        ).flatten()
        assay_size = int(assay_indices.numel())
        list_count = math.ceil(assay_size / max_list_size)
        base_size, larger_list_count = divmod(assay_size, list_count)
        list_sizes = [base_size + 1] * larger_list_count + [base_size] * (
            list_count - larger_list_count
        )
        for partition_index, partition_ids in enumerate(partitions):
            generator = torch.Generator()
            partition_seed = (
                int(seed)
                + 1_000_003 * (partition_index + 1)
                + 97_409 * (assay_id + 1)
            ) % max_seed
            generator.manual_seed(partition_seed)
            shuffled_indices = assay_indices.index_select(
                0,
                torch.randperm(assay_size, generator=generator),
            )
            cursor = 0
            for list_size in list_sizes:
                list_indices = shuffled_indices[cursor : cursor + list_size]
                partition_ids[list_indices] = next_list_ids[partition_index]
                next_list_ids[partition_index] += 1
                cursor += list_size
            if cursor != assay_size:
                raise RuntimeError(
                    "contrastive partition did not cover the complete assay"
                )
    return partitions


def compute_contrastive_evaluation_loss(
    *,
    normalized_protein_embeddings: torch.Tensor,
    normalized_molecule_embeddings: torch.Tensor,
    pchembl_values: torch.Tensor,
    contrastive_group_ids: torch.Tensor,
    target_identity_ids: torch.Tensor,
    molecule_identity_ids: torch.Tensor,
    temperature: float,
    active_threshold: float,
    assay_batch_size: int,
    ranking_max_ligands: int = 16,
    ranking_num_partitions: int = 3,
    ranking_partition_seed: int = 42,
    ranking_min_pchembl_span: float = 0.5,
) -> torch.Tensor:
    """Compute deterministic full-coverage LigUnity validation loss.

    Every assay is partitioned into bounded lists, including singleton,
    two-ligand, and narrow-affinity-span assays that are ineligible for ranking.
    Lists are deterministically shuffled into local batches so each contrastive
    matrix has the same assay-list capacity as one training-device microbatch.
    Losses are weighted by assay-list count before averaging across
    complete-coverage partitions.
    """
    protein_embeddings = (
        normalized_protein_embeddings.detach().cpu().float()
    )
    molecule_embeddings = (
        normalized_molecule_embeddings.detach().cpu().float()
    )
    pchembl = pchembl_values.reshape(-1).detach().cpu().float()
    assay_ids = contrastive_group_ids.reshape(-1).detach().cpu().long()
    target_ids = target_identity_ids.reshape(-1).detach().cpu().long()
    molecule_ids = molecule_identity_ids.reshape(-1).detach().cpu().long()

    if protein_embeddings.ndim != 2 or molecule_embeddings.ndim != 2:
        raise ValueError("normalized evaluation embeddings must be rank-2")
    if protein_embeddings.shape != molecule_embeddings.shape:
        raise ValueError("normalized protein and molecule embeddings must align")
    num_examples = protein_embeddings.size(0)
    if not all(
        values.numel() == num_examples
        for values in (pchembl, assay_ids, target_ids, molecule_ids)
    ):
        raise ValueError("contrastive evaluation inputs must contain the same rows")
    if not (
        torch.isfinite(protein_embeddings).all()
        and torch.isfinite(molecule_embeddings).all()
        and torch.isfinite(pchembl).all()
    ):
        raise ValueError("contrastive evaluation inputs must be finite")
    if (target_ids < 0).any() or (molecule_ids < 0).any():
        raise ValueError("contrastive evaluation identity ids must be non-negative")
    if temperature <= 0.0 or not math.isfinite(float(temperature)):
        raise ValueError("temperature must be finite and > 0")
    if not math.isfinite(float(active_threshold)):
        raise ValueError("active_threshold must be finite")
    if assay_batch_size <= 0:
        raise ValueError("assay_batch_size must be > 0")

    partitions = build_complete_coverage_contrastive_partitions(
        contrastive_group_ids=assay_ids,
        max_list_size=ranking_max_ligands,
        num_partitions=ranking_num_partitions,
        seed=ranking_partition_seed,
    )
    partition_losses: list[torch.Tensor] = []
    max_seed = (1 << 63) - 1
    for partition_index, partition_ids in enumerate(partitions):
        ranked_indices = torch.nonzero(
            partition_ids >= 0,
            as_tuple=False,
        ).flatten()
        if ranked_indices.numel() == 0:
            continue
        ranked_list_ids = partition_ids.index_select(0, ranked_indices)
        num_lists = int(ranked_list_ids.max().item()) + 1
        list_counts = torch.bincount(
            ranked_list_ids,
            minlength=num_lists,
        )
        list_order = torch.argsort(ranked_list_ids, stable=True)
        sorted_row_indices = ranked_indices.index_select(0, list_order)
        list_offsets = torch.cat(
            (
                torch.zeros(1, dtype=torch.long),
                list_counts.cumsum(dim=0),
            )
        )

        generator = torch.Generator()
        generator.manual_seed(
            (
                int(ranking_partition_seed)
                + 2_000_003 * (partition_index + 1)
            )
            % max_seed
        )
        shuffled_lists = torch.randperm(num_lists, generator=generator)
        weighted_partition_loss = torch.zeros((), dtype=torch.float32)
        partition_list_count = 0
        for start in range(0, num_lists, assay_batch_size):
            batch_list_ids = shuffled_lists[start : start + assay_batch_size]
            batch_members = [
                sorted_row_indices[
                    list_offsets[list_id] : list_offsets[list_id + 1]
                ]
                for list_id in batch_list_ids.tolist()
            ]
            selected_indices = torch.cat(batch_members)
            group_sizes = torch.tensor(
                [members.numel() for members in batch_members],
                dtype=torch.long,
            )
            ligand_group_ids = torch.repeat_interleave(
                torch.arange(len(batch_members), dtype=torch.long),
                group_sizes,
            )
            representative_indices = torch.tensor(
                [int(members[0].item()) for members in batch_members],
                dtype=torch.long,
            )
            group_target_ids = target_ids.index_select(
                0,
                representative_indices,
            )
            selected_target_ids = target_ids.index_select(0, selected_indices)
            if not torch.equal(
                group_target_ids.index_select(0, ligand_group_ids),
                selected_target_ids,
            ):
                raise ValueError(
                    "every contrastive evaluation list must contain one target identity"
                )

            contrastive_scores = torch.matmul(
                protein_embeddings.index_select(0, representative_indices),
                molecule_embeddings.index_select(0, selected_indices).transpose(0, 1),
            ) / float(temperature)
            batch_loss, _, _ = ligunity_bidirectional_contrastive_loss(
                contrastive_scores,
                pchembl.index_select(0, selected_indices),
                ligand_group_ids,
                group_target_ids,
                molecule_ids.index_select(0, selected_indices),
                active_threshold=active_threshold,
            )
            batch_list_count = len(batch_members)
            weighted_partition_loss += batch_loss * batch_list_count
            partition_list_count += batch_list_count
        if partition_list_count != num_lists:
            raise RuntimeError("contrastive evaluation did not cover every assay list")
        partition_losses.append(weighted_partition_loss / partition_list_count)

    if not partition_losses:
        return torch.zeros((), dtype=torch.float32)
    return torch.stack(partition_losses).mean()


def compute_ranking_score_diagnostics(
    *,
    ranking_scores: torch.Tensor,
    pchembl_values: torch.Tensor,
    ranking_group_ids: torch.Tensor,
    cosine_similarities: torch.Tensor | None = None,
    temperature: float = 1.0,
    affinity_margin: float = DEFAULT_RANKING_AFFINITY_MARGIN,
) -> Dict[str, float]:
    """Describe cosine and margin-eligible ranking geometry.

    These metrics are observational only: they do not transform scores or
    participate in the loss. Groups with negative ids are ignored.
    """
    scores = ranking_scores.reshape(-1).detach().cpu().float()
    targets = pchembl_values.reshape(-1).detach().cpu().float()
    group_ids = ranking_group_ids.reshape(-1).detach().cpu().long()
    if not (scores.shape == targets.shape == group_ids.shape):
        raise ValueError("ranking diagnostic tensors must have the same shape")
    if not torch.isfinite(scores).all() or not torch.isfinite(targets).all():
        raise ValueError("ranking diagnostic scores and targets must be finite")
    cosines = None
    if cosine_similarities is not None:
        cosines = cosine_similarities.reshape(-1).detach().cpu().float()
        if cosines.shape != scores.shape:
            raise ValueError(
                "cosine_similarities must have the same shape as ranking_scores"
            )
        if not torch.isfinite(cosines).all():
            raise ValueError("cosine_similarities must be finite")
        if ((cosines < -1.0001) | (cosines > 1.0001)).any():
            raise ValueError("cosine_similarities must be in [-1, 1]")
    if temperature <= 0.0 or not math.isfinite(float(temperature)):
        raise ValueError("temperature must be finite and > 0")
    if affinity_margin < 0.0 or not math.isfinite(float(affinity_margin)):
        raise ValueError("affinity_margin must be finite and >= 0")

    ranked_mask = group_ids >= 0
    if not ranked_mask.any():
        return {}

    metrics: Dict[str, float] = {}
    if cosines is not None:
        ranked_cosines = cosines[ranked_mask]
        cosine_quantiles = torch.quantile(
            ranked_cosines,
            torch.tensor([0.01, 0.99], dtype=ranked_cosines.dtype),
        )
        metrics.update(
            {
                "ranking_cosine_mean": float(ranked_cosines.mean().item()),
                "ranking_cosine_std": float(
                    ranked_cosines.std(unbiased=False).item()
                ),
                "ranking_cosine_p01": float(cosine_quantiles[0].item()),
                "ranking_cosine_p99": float(cosine_quantiles[1].item()),
            }
        )

    list_entropies: list[torch.Tensor] = []
    signed_pair_gaps: list[torch.Tensor] = []
    for group_id in torch.unique(group_ids[ranked_mask], sorted=True):
        group_mask = group_ids == group_id
        group_scores = scores[group_mask]
        group_targets = targets[group_mask]
        if group_scores.numel() >= 2:
            probabilities = torch.softmax(group_scores / float(temperature), dim=0)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
            list_entropies.append(entropy / math.log(group_scores.numel()))

            upper_triangle = torch.triu(
                torch.ones(
                    (group_scores.numel(), group_scores.numel()),
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            target_differences = group_targets[:, None] - group_targets[None, :]
            eligible_pairs = upper_triangle & (
                target_differences.abs() > float(affinity_margin)
            )
            score_differences = group_scores[:, None] - group_scores[None, :]
            signed_pair_gaps.append(
                (score_differences * target_differences.sign())[eligible_pairs]
            )

    if list_entropies:
        entropies = torch.stack(list_entropies)
        metrics["ranking_list_normalized_entropy"] = float(entropies.mean().item())

    nonempty_gaps = [gaps for gaps in signed_pair_gaps if gaps.numel() > 0]
    if nonempty_gaps:
        gaps = torch.cat(nonempty_gaps)
        metrics.update(
            {
                "ranking_margin_pair_accuracy": float(
                    ((gaps > 0).float() + 0.5 * (gaps == 0).float()).mean().item()
                ),
                "ranking_margin_pair_gap_p50": float(
                    torch.quantile(gaps, 0.5).item()
                ),
                "ranking_margin_pair_count": float(gaps.numel()),
            }
        )
    return metrics


def compute_classification_metrics(
    probabilities: Sequence[float],
    labels: Sequence[float],
    *,
    threshold: float = 0.5,
) -> Dict[str, float]:
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    predicted_labels = (probabilities_array >= threshold).astype(np.int64)
    metrics = {
        "eval_mcc": float(matthews_corrcoef(labels_array, predicted_labels)),
        "eval_f1": float(f1_score(labels_array, predicted_labels, zero_division=0)),
        "eval_precision": float(
            precision_score(labels_array, predicted_labels, zero_division=0)
        ),
        "eval_recall": float(recall_score(labels_array, predicted_labels, zero_division=0)),
        "eval_accuracy": float(accuracy_score(labels_array, predicted_labels)),
    }
    metrics["eval_roc_auc"] = (
        float(roc_auc_score(labels_array, probabilities_array))
        if len(np.unique(labels_array)) >= 2
        else float("nan")
    )
    return metrics


def compute_joint_evaluation_metrics(
    *,
    activity_logits: torch.Tensor,
    ranking_scores: torch.Tensor,
    activity_labels: torch.Tensor,
    pchembl_values: torch.Tensor,
    ranking_group_ids: torch.Tensor,
    group_id_names: Sequence[str],
    activity_types: Sequence[str] | None = None,
    protein_shuffled_scores: torch.Tensor | None = None,
    classification_loss_weight: float,
    ranking_loss_weight: float,
    bce_pos_weight: float,
    ranking_temperature: float,
    ranking_affinity_margin: float,
    ranking_min_pchembl_span: float,
    ranking_max_ligands: int = 16,
    ranking_num_partitions: int = 3,
    ranking_partition_seed: int = 42,
    ranking_score_diagnostics: bool = False,
    cosine_similarities: torch.Tensor | None = None,
    metrics_profile: str = "full",
    contrastive_loss: torch.Tensor | float | None = None,
    contrastive_loss_weight: float = 0.0,
) -> tuple[Dict[str, float], list[Dict[str, float | int | str]]]:
    """Compute deterministic full-dataset losses and metrics from one scoring pass."""
    logits = activity_logits.reshape(-1).detach().cpu().float()
    scores = ranking_scores.reshape(-1).detach().cpu().float()
    labels = activity_labels.reshape(-1).detach().cpu().float()
    pchembl = pchembl_values.reshape(-1).detach().cpu().float()
    group_indices = ranking_group_ids.reshape(-1).detach().cpu().long()
    if not (
        logits.numel()
        == scores.numel()
        == labels.numel()
        == pchembl.numel()
        == group_indices.numel()
    ):
        raise ValueError("evaluation tensors must contain the same number of observations")
    if activity_types is not None and len(activity_types) != logits.numel():
        raise ValueError("activity_types must align with evaluation observations")
    shuffled_scores = None
    if protein_shuffled_scores is not None:
        shuffled_scores = protein_shuffled_scores.reshape(-1).detach().cpu().float()
        if shuffled_scores.shape != scores.shape:
            raise ValueError(
                "protein_shuffled_scores must have the same shape as ranking_scores"
            )
        if not torch.isfinite(shuffled_scores).all():
            raise ValueError("protein_shuffled_scores must contain only finite values")

    if metrics_profile == "ranking":
        if classification_loss_weight != 0.0:
            raise ValueError(
                "metrics_profile=ranking requires classification_loss_weight=0.0"
            )
        classification_loss = torch.zeros((), dtype=logits.dtype)
    else:
        pos_weight = torch.tensor(float(bce_pos_weight), dtype=logits.dtype)
        classification_loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=pos_weight,
        )
    ranking_partitions = build_complete_coverage_ranking_partitions(
        pchembl_values=pchembl,
        ranking_group_ids=group_indices,
        max_list_size=ranking_max_ligands,
        num_partitions=ranking_num_partitions,
        seed=ranking_partition_seed,
        min_pchembl_span=ranking_min_pchembl_span,
    )
    partition_losses = [
        ligunity_listwise_loss(
            scores,
            pchembl,
            partition_group_ids,
            temperature=ranking_temperature,
            # Eligibility is determined from the complete assay before it is
            # partitioned, matching the training dataset's sampling contract.
            min_pchembl_span=0.0,
            affinity_margin=ranking_affinity_margin,
        )
        for partition_group_ids in ranking_partitions
    ]
    ranking_loss = torch.stack(partition_losses).mean()
    if contrastive_loss_weight < 0.0 or not math.isfinite(
        float(contrastive_loss_weight)
    ):
        raise ValueError("contrastive_loss_weight must be finite and >= 0")
    has_contrastive_loss = contrastive_loss is not None
    if contrastive_loss is None:
        if contrastive_loss_weight > 0.0:
            raise ValueError(
                "contrastive_loss is required when contrastive_loss_weight > 0"
            )
        resolved_contrastive_loss = torch.zeros((), dtype=ranking_loss.dtype)
    else:
        resolved_contrastive_loss = torch.as_tensor(
            contrastive_loss,
            dtype=ranking_loss.dtype,
        ).reshape(())
        if not torch.isfinite(resolved_contrastive_loss):
            raise ValueError("contrastive_loss must be finite")
    total_loss = (
        float(classification_loss_weight) * classification_loss
        + float(ranking_loss_weight) * ranking_loss
        + float(contrastive_loss_weight) * resolved_contrastive_loss
    )

    string_group_ids = [group_id_names[int(index)] for index in group_indices.tolist()]
    spearman_metrics, assay_records = _compute_groupwise_spearman_with_records(
        group_ids=string_group_ids,
        ranking_scores=scores.tolist(),
        pchembl_values=pchembl.tolist(),
        min_group_size=MIN_LISTWISE_LIGANDS,
        min_pchembl_span=ranking_min_pchembl_span,
    )
    if metrics_profile == "ranking":
        metrics = {
            "eval_loss": float(total_loss.item()),
            "eval_ranking_loss": float(ranking_loss.item()),
            "eval_spearman": spearman_metrics["eval_spearman"],
            "eval_pearson": _compute_weighted_groupwise_pearson(
                group_ids=string_group_ids,
                ranking_scores=scores.tolist(),
                pchembl_values=pchembl.tolist(),
                min_group_size=MIN_LISTWISE_LIGANDS,
                min_pchembl_span=ranking_min_pchembl_span,
            ),
        }
        if has_contrastive_loss:
            metrics["eval_contrastive_loss"] = float(
                resolved_contrastive_loss.item()
            )
    elif metrics_profile == "full":
        probabilities = torch.sigmoid(logits).numpy()
        metrics = compute_classification_metrics(probabilities, labels.numpy())
        metrics.update(spearman_metrics)
        if activity_types is not None:
            metrics.update(
                compute_activity_type_metrics(
                    probabilities=probabilities,
                    labels=labels.numpy(),
                    activity_types=activity_types,
                    group_ids=string_group_ids,
                    ranking_scores=scores.tolist(),
                    pchembl_values=pchembl.tolist(),
                    min_pchembl_span=ranking_min_pchembl_span,
                )
            )
        if shuffled_scores is not None:
            metrics.update(
                compute_protein_shuffle_sensitivity(
                    group_ids=string_group_ids,
                    baseline_scores=scores.tolist(),
                    shuffled_scores=shuffled_scores.tolist(),
                    pchembl_values=pchembl.tolist(),
                    min_pchembl_span=ranking_min_pchembl_span,
                )
            )
        metrics.update(
            {
                "eval_loss": float(total_loss.item()),
                "eval_total_loss": float(total_loss.item()),
                "eval_classification_loss": float(classification_loss.item()),
                "eval_ranking_loss": float(ranking_loss.item()),
                "eval_num_examples": float(logits.numel()),
                "eval_num_ranking_groups": float(len(assay_records)),
                "eval_num_ranking_lists": float(
                    int(ranking_partitions[0].max().item()) + 1
                    if (ranking_partitions[0] >= 0).any()
                    else 0
                ),
                "eval_num_ranked_examples": float(
                    (ranking_partitions[0] >= 0).sum().item()
                ),
                "eval_ranking_partitions": float(len(ranking_partitions)),
            }
        )
        if has_contrastive_loss:
            metrics["eval_contrastive_loss"] = float(
                resolved_contrastive_loss.item()
            )
    else:
        raise ValueError("metrics_profile must be 'full' or 'ranking'")

    if ranking_score_diagnostics:
        diagnostic_metrics = compute_ranking_score_diagnostics(
            ranking_scores=scores,
            pchembl_values=pchembl,
            # The first deterministic partition covers every eligible row and
            # keeps diagnostics bounded to the same maximum list size as loss.
            ranking_group_ids=ranking_partitions[0],
            cosine_similarities=cosine_similarities,
            temperature=ranking_temperature,
            affinity_margin=ranking_affinity_margin,
        )
        if metrics_profile == "ranking":
            if "ranking_cosine_std" in diagnostic_metrics:
                metrics["eval_cosine_std"] = diagnostic_metrics[
                    "ranking_cosine_std"
                ]
            if "ranking_margin_pair_accuracy" in diagnostic_metrics:
                metrics["eval_pair_accuracy"] = diagnostic_metrics[
                    "ranking_margin_pair_accuracy"
                ]
        else:
            metrics.update(
                {f"eval_{key}": value for key, value in diagnostic_metrics.items()}
            )
    return metrics, assay_records


def compute_pairwise_accuracy(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
) -> float:
    positive_array = np.asarray(positive_scores, dtype=np.float64)
    negative_array = np.asarray(negative_scores, dtype=np.float64)
    return float(np.mean(positive_array > negative_array))


def compute_groupwise_spearman(
    *,
    group_ids: Sequence[str],
    ranking_scores: Sequence[float],
    pchembl_values: Sequence[float],
    min_group_size: int = 3,
) -> Dict[str, float]:
    metrics, _ = _compute_groupwise_spearman_with_records(
        group_ids=group_ids,
        ranking_scores=ranking_scores,
        pchembl_values=pchembl_values,
        min_group_size=min_group_size,
    )
    return metrics


def compute_activity_type_metrics(
    *,
    probabilities: Sequence[float],
    labels: Sequence[float],
    activity_types: Sequence[str],
    group_ids: Sequence[str],
    ranking_scores: Sequence[float],
    pchembl_values: Sequence[float],
    min_pchembl_span: float = 0.5,
) -> Dict[str, float]:
    """Report Potency/qHTS and non-Potency performance separately."""
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    activity_array = np.asarray([str(value) for value in activity_types], dtype=object)
    group_array = np.asarray(group_ids, dtype=object)
    score_array = np.asarray(ranking_scores, dtype=np.float64)
    pchembl_array = np.asarray(pchembl_values, dtype=np.float64)
    expected_size = probabilities_array.size
    if not all(
        values.size == expected_size
        for values in (
            labels_array,
            activity_array,
            group_array,
            score_array,
            pchembl_array,
        )
    ):
        raise ValueError("activity-type metric inputs must have the same length")

    potency_mask = np.asarray(
        [
            any(
                token.strip().casefold() == "potency"
                or "qhts" in token.strip().casefold()
                for token in value.split("|")
            )
            for value in activity_array
        ],
        dtype=bool,
    )
    metrics: Dict[str, float] = {}
    for name, mask in (
        ("potency_qhts", potency_mask),
        ("non_potency", ~potency_mask),
    ):
        count = int(mask.sum())
        metrics[f"eval_{name}_num_examples"] = float(count)
        if count == 0:
            continue
        metrics[f"eval_{name}_positive_fraction"] = float(labels_array[mask].mean())
        classification = compute_classification_metrics(
            probabilities_array[mask],
            labels_array[mask],
        )
        metrics.update(
            {
                key.replace("eval_", f"eval_{name}_", 1): value
                for key, value in classification.items()
            }
        )
        ranking, _ = _compute_groupwise_spearman_with_records(
            group_ids=group_array[mask].tolist(),
            ranking_scores=score_array[mask].tolist(),
            pchembl_values=pchembl_array[mask].tolist(),
            min_group_size=MIN_LISTWISE_LIGANDS,
            min_pchembl_span=min_pchembl_span,
        )
        metrics.update(
            {
                key.replace("eval_", f"eval_{name}_", 1): value
                for key, value in ranking.items()
            }
        )
    return metrics


def compute_protein_shuffle_sensitivity(
    *,
    group_ids: Sequence[str],
    baseline_scores: Sequence[float],
    shuffled_scores: Sequence[float],
    pchembl_values: Sequence[float],
    min_pchembl_span: float = 0.5,
) -> Dict[str, float]:
    """Measure whether changing the protein changes within-assay rankings."""
    baseline_metrics, baseline_records = _compute_groupwise_spearman_with_records(
        group_ids=group_ids,
        ranking_scores=baseline_scores,
        pchembl_values=pchembl_values,
        min_group_size=MIN_LISTWISE_LIGANDS,
        min_pchembl_span=min_pchembl_span,
    )
    shuffled_metrics, _ = _compute_groupwise_spearman_with_records(
        group_ids=group_ids,
        ranking_scores=shuffled_scores,
        pchembl_values=pchembl_values,
        min_group_size=MIN_LISTWISE_LIGANDS,
        min_pchembl_span=min_pchembl_span,
    )

    metrics = {
        "eval_protein_shuffled_macro_spearman": shuffled_metrics[
            "eval_macro_spearman"
        ],
        "eval_protein_shuffled_weighted_spearman": shuffled_metrics[
            "eval_weighted_spearman"
        ],
        "eval_protein_shuffle_macro_spearman_drop": baseline_metrics[
            "eval_macro_spearman"
        ]
        - shuffled_metrics["eval_macro_spearman"],
        "eval_protein_shuffle_weighted_spearman_drop": baseline_metrics[
            "eval_weighted_spearman"
        ]
        - shuffled_metrics["eval_weighted_spearman"],
    }

    grouped_baseline: Dict[str, list[float]] = defaultdict(list)
    grouped_shuffled: Dict[str, list[float]] = defaultdict(list)
    for group_id, baseline, shuffled in zip(
        group_ids, baseline_scores, shuffled_scores
    ):
        grouped_baseline[str(group_id)].append(float(baseline))
        grouped_shuffled[str(group_id)].append(float(shuffled))
    eligible_groups = {str(record["group_id"]) for record in baseline_records}
    stability_records: list[tuple[float, int]] = []
    for group_id in sorted(eligible_groups):
        baseline = grouped_baseline[group_id]
        shuffled = grouped_shuffled[group_id]
        if len(set(baseline)) < 2 or len(set(shuffled)) < 2:
            stability = 0.0
        else:
            result = spearmanr(baseline, shuffled)
            stability = float(getattr(result, "statistic", result[0]))
            if not math.isfinite(stability):
                stability = 0.0
        stability_records.append((stability, len(baseline)))
    if stability_records:
        metrics["eval_protein_shuffle_macro_rank_stability"] = float(
            np.mean([value for value, _ in stability_records])
        )
        metrics["eval_protein_shuffle_weighted_rank_stability"] = float(
            sum(value * size for value, size in stability_records)
            / sum(size for _, size in stability_records)
        )
    else:
        metrics["eval_protein_shuffle_macro_rank_stability"] = float("nan")
        metrics["eval_protein_shuffle_weighted_rank_stability"] = float("nan")
    return metrics


def _split_group_id(group_id: str) -> tuple[str, str]:
    target_chembl_id, assay_id = group_id.split("__", 1)
    return target_chembl_id, assay_id


def _compute_weighted_groupwise_pearson(
    *,
    group_ids: Sequence[str],
    ranking_scores: Sequence[float],
    pchembl_values: Sequence[float],
    min_group_size: int = 3,
    min_pchembl_span: float = 0.0,
) -> float:
    return compute_groupwise_rank_correlations(
        group_ids=group_ids,
        ranking_scores=ranking_scores,
        pchembl_values=pchembl_values,
        min_group_size=min_group_size,
        min_pchembl_span=min_pchembl_span,
    )["pearson"]


def compute_groupwise_rank_correlations(
    *,
    group_ids: Sequence[str | int],
    ranking_scores: Sequence[float],
    pchembl_values: Sequence[float],
    min_group_size: int = 3,
    min_pchembl_span: float = 0.0,
) -> Dict[str, float]:
    """Return example-weighted Spearman and Pearson across ranking groups."""
    grouped_scores: Dict[str, list[float]] = defaultdict(list)
    grouped_pchembl: Dict[str, list[float]] = defaultdict(list)
    for group_id, score, pchembl in zip(group_ids, ranking_scores, pchembl_values):
        grouped_scores[str(group_id)].append(float(score))
        grouped_pchembl[str(group_id)].append(float(pchembl))

    correlations: list[tuple[float, float, int]] = []
    for group_id in sorted(grouped_scores):
        scores = grouped_scores[group_id]
        pchembls = grouped_pchembl[group_id]
        if len(scores) < min_group_size:
            continue
        if len(set(pchembls)) == 1:
            continue
        if max(pchembls) - min(pchembls) < min_pchembl_span:
            continue
        if len(set(scores)) == 1:
            spearman = float("nan")
            pearson = float("nan")
        else:
            spearman_result = spearmanr(scores, pchembls)
            pearson_result = pearsonr(scores, pchembls)
            spearman = float(
                getattr(spearman_result, "statistic", spearman_result[0])
            )
            pearson = float(
                getattr(pearson_result, "statistic", pearson_result[0])
            )
        correlations.append((spearman, pearson, len(scores)))

    if not correlations:
        return {"spearman": float("nan"), "pearson": float("nan")}
    total_size = sum(size for _, _, size in correlations)
    return {
        "spearman": float(
            sum(spearman * size for spearman, _, size in correlations)
            / total_size
        ),
        "pearson": float(
            sum(pearson * size for _, pearson, size in correlations)
            / total_size
        ),
    }


def _compute_groupwise_spearman_with_records(
    *,
    group_ids: Sequence[str],
    ranking_scores: Sequence[float],
    pchembl_values: Sequence[float],
    min_group_size: int = 3,
    min_pchembl_span: float = 0.0,
) -> tuple[Dict[str, float], list[Dict[str, float | int | str]]]:
    grouped_scores: Dict[str, list[float]] = defaultdict(list)
    grouped_pchembl: Dict[str, list[float]] = defaultdict(list)
    for group_id, score, pchembl in zip(group_ids, ranking_scores, pchembl_values):
        grouped_scores[str(group_id)].append(float(score))
        grouped_pchembl[str(group_id)].append(float(pchembl))

    assay_records: list[Dict[str, float | int | str]] = []
    for group_id in sorted(grouped_scores):
        scores = grouped_scores[group_id]
        pchembls = grouped_pchembl[group_id]
        if len(scores) < min_group_size:
            continue
        if len(set(pchembls)) == 1:
            continue
        if max(pchembls) - min(pchembls) < min_pchembl_span:
            continue
        result = spearmanr(scores, pchembls)
        target_chembl_id, assay_id = _split_group_id(group_id)
        assay_records.append(
            {
                "group_id": group_id,
                "target_chembl_id": target_chembl_id,
                "assay_id": assay_id,
                "num_examples": len(scores),
                "spearman": float(getattr(result, "statistic", result[0])),
            }
        )

    if not assay_records:
        return (
            {
                "eval_spearman": float("nan"),
                "eval_weighted_spearman": float("nan"),
                "eval_macro_spearman": float("nan"),
                "eval_spearman_num_groups": 0.0,
            },
            assay_records,
        )

    weighted_sum = sum(
        float(record["spearman"]) * int(record["num_examples"])
        for record in assay_records
    )
    total_weight = sum(int(record["num_examples"]) for record in assay_records)
    weighted_spearman = float(weighted_sum / total_weight)
    macro_spearman = float(
        np.mean([float(record["spearman"]) for record in assay_records])
    )
    return (
        {
            # Backward-compatible alias; explicit names remove ambiguity.
            "eval_spearman": weighted_spearman,
            "eval_weighted_spearman": weighted_spearman,
            "eval_macro_spearman": macro_spearman,
            "eval_spearman_num_groups": float(len(assay_records)),
        },
        assay_records,
    )


def _append_assay_spearman_log(
    trainer,
    metrics: Mapping[str, float],
    assay_records: Sequence[Mapping[str, float | int | str]],
    *,
    metric_key_prefix: str,
) -> None:
    if not trainer.is_world_process_zero():
        return

    os.makedirs(trainer.args.output_dir, exist_ok=True)
    filename = (
        ASSAY_SPEARMAN_LOG_FILENAME
        if metric_key_prefix == "eval"
        else f"{metric_key_prefix}_assay_spearman.jsonl"
    )
    path = os.path.join(trainer.args.output_dir, filename)
    macro_spearman = metrics.get(f"{metric_key_prefix}_macro_spearman")
    if macro_spearman is None:
        macro_spearman = (
            float(np.mean([float(record["spearman"]) for record in assay_records]))
            if assay_records
            else float("nan")
        )
    num_eligible_groups = metrics.get(
        f"{metric_key_prefix}_spearman_num_groups",
        float(len(assay_records)),
    )
    record = {
        "global_step": int(trainer.state.global_step),
        "epoch": trainer.state.epoch,
        "weighted_spearman": metrics[f"{metric_key_prefix}_spearman"],
        "macro_spearman": macro_spearman,
        "num_eligible_groups": num_eligible_groups,
        "assays": list(assay_records),
    }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _with_metric_prefix(metrics: Mapping[str, float], metric_key_prefix: str) -> Dict[str, float]:
    if metric_key_prefix == "eval":
        return dict(metrics)
    return {
        key.replace("eval_", f"{metric_key_prefix}_", 1): value
        for key, value in metrics.items()
    }


def _collate_example_rows(
    rows: Sequence[Mapping[str, Any]],
    collator: RewardPairCollator,
) -> Dict[str, torch.Tensor]:
    return collator.collate_example_tokens(rows)


def _score_example_dataset(
    trainer,
    model,
    example_dataset,
    *,
    batch_size: int,
) -> Dict[str, Sequence[float] | Sequence[str]]:
    ranking_scores: list[float] = []
    activity_probabilities: list[float] = []
    labels: list[float] = []
    group_ids: list[str] = []
    pchembl_values: list[float] = []
    collator = (
        trainer.data_collator
        if isinstance(trainer.data_collator, RewardPairCollator)
        else RewardPairCollator()
    )

    for start in range(0, len(example_dataset), batch_size):
        rows = [dict(example_dataset[index]) for index in range(start, min(start + batch_size, len(example_dataset)))]
        model_inputs = trainer._prepare_inputs(_collate_example_rows(rows, collator))
        with torch.no_grad():
            outputs = model(return_dict=True, **model_inputs)
        ranking_scores.extend(outputs.ranking_score.detach().cpu().tolist())
        activity_probabilities.extend(outputs.activity_probability.detach().cpu().tolist())
        labels.extend(float(row["binary_label"]) for row in rows)
        group_ids.extend(str(row["group_id"]) for row in rows)
        pchembl_values.extend(float(row["pchembl_value"]) for row in rows)

    return {
        "ranking_scores": ranking_scores,
        "activity_probabilities": activity_probabilities,
        "labels": labels,
        "group_ids": group_ids,
        "pchembl_values": pchembl_values,
    }


def _score_pair_dataset(
    trainer,
    model,
    eval_dataset: RewardPairDataset,
    *,
    batch_size: int,
) -> Dict[str, float]:
    collator = (
        trainer.data_collator
        if isinstance(trainer.data_collator, RewardPairCollator)
        else RewardPairCollator()
    )
    positive_scores: list[float] = []
    negative_scores: list[float] = []

    for start in range(0, len(eval_dataset), batch_size):
        features = [
            eval_dataset[index]
            for index in range(start, min(start + batch_size, len(eval_dataset)))
        ]
        batch = collator(features)
        model_inputs = trainer._prepare_inputs(trainer._build_model_inputs(batch))
        with torch.no_grad():
            outputs = model(**model_inputs)
        ranking_score = outputs.ranking_score.detach().cpu()
        positive_indices = batch["positive_indices"]
        negative_indices = batch["negative_indices"]
        positive_scores.extend(ranking_score[positive_indices].tolist())
        negative_scores.extend(ranking_score[negative_indices].tolist())

    return {
        "eval_pairwise_accuracy": compute_pairwise_accuracy(positive_scores, negative_scores),
    }


def compute_reward_model_eval_metrics(
    trainer,
    model,
    eval_dataset: RewardPairDataset,
    *,
    metric_key_prefix: str = "eval",
    pairwise_accuracy: float | None = None,
) -> Dict[str, float]:
    batch_size = int(trainer.args.per_device_eval_batch_size)
    example_metrics_source = _score_example_dataset(
        trainer,
        model,
        eval_dataset.example_dataset,
        batch_size=batch_size,
    )
    metrics = compute_classification_metrics(
        example_metrics_source["activity_probabilities"],
        example_metrics_source["labels"],
        threshold=0.5,
    )
    spearman_metrics, assay_records = _compute_groupwise_spearman_with_records(
        group_ids=example_metrics_source["group_ids"],
        ranking_scores=example_metrics_source["ranking_scores"],
        pchembl_values=example_metrics_source["pchembl_values"],
        min_group_size=3,
    )
    metrics.update(spearman_metrics)
    if pairwise_accuracy is None:
        metrics.update(
            _score_pair_dataset(
                trainer,
                model,
                eval_dataset,
                batch_size=batch_size,
            )
        )
    else:
        metrics["eval_pairwise_accuracy"] = float(pairwise_accuracy)
    metrics = _with_metric_prefix(metrics, metric_key_prefix)
    _append_assay_spearman_log(
        trainer,
        metrics,
        assay_records,
        metric_key_prefix=metric_key_prefix,
    )
    return metrics
