from __future__ import annotations

import json
import math
import os
import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from transformers.trainer_pt_utils import get_parameter_names

from ..training import (
    RewardAssayListCollator,
    RewardAssayListDataset,
    get_tokenized_split_dataset_paths,
    load_tokenized_example_dataset,
    validate_tokenized_split_cardinality,
)
from .ranking_head import select_complete_assays


CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")
MODULE_PREFIXES = {
    "protein_encoder": "protein_encoder.",
    "molecule_encoder": "molecule_encoder.",
    "protein_projection": "protein_projection.",
    "molecule_projection": "molecule_projection.",
}
MODEL_INPUT_KEYS = (
    "protein_input_ids",
    "protein_attention_mask",
    "molecule_input_ids",
    "molecule_attention_mask",
    "activity_labels",
    "pchembl_values",
    "ranking_group_ids",
    "contrastive_group_ids",
    "contrastive_target_ids",
    "contrastive_molecule_ids",
)


@dataclass(frozen=True)
class CheckpointPair:
    best: str
    last: str


def _checkpoint_step(path: str) -> int:
    match = CHECKPOINT_PATTERN.match(os.path.basename(os.path.abspath(path)))
    if match is None:
        raise ValueError(f"Checkpoint directory must use checkpoint-N naming: {path}")
    return int(match.group(1))


def _checkpoint_directories(run_dir: str) -> list[str]:
    resolved = os.path.abspath(run_dir)
    if not os.path.isdir(resolved):
        raise FileNotFoundError(f"Run directory does not exist: {resolved}")
    checkpoints = []
    for name in os.listdir(resolved):
        path = os.path.join(resolved, name)
        if os.path.isdir(path) and CHECKPOINT_PATTERN.match(name):
            checkpoints.append(path)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-N directories found under {resolved}")
    return sorted(checkpoints, key=_checkpoint_step)


def _resolve_recorded_checkpoint(value: str, *, run_dir: str) -> str:
    recorded = os.path.abspath(value)
    if os.path.isdir(recorded):
        return recorded
    relocated = os.path.join(os.path.abspath(run_dir), os.path.basename(recorded))
    if os.path.isdir(relocated):
        return relocated
    raise FileNotFoundError(
        "trainer_state.json points to a best checkpoint that cannot be found: "
        f"{value}"
    )


def resolve_checkpoint_pair(
    *,
    run_dir: str | None = None,
    best_checkpoint: str | None = None,
    last_checkpoint: str | None = None,
) -> CheckpointPair:
    """Resolve explicit checkpoints or discover retained best and latest ones."""
    if best_checkpoint is not None or last_checkpoint is not None:
        if best_checkpoint is None or last_checkpoint is None:
            raise ValueError(
                "best_checkpoint and last_checkpoint must be provided together"
            )
        best = os.path.abspath(best_checkpoint)
        last = os.path.abspath(last_checkpoint)
        for label, path in (("best", best), ("last", last)):
            if not os.path.isdir(path):
                raise FileNotFoundError(f"{label} checkpoint does not exist: {path}")
        return CheckpointPair(best=best, last=last)

    if run_dir is None:
        raise ValueError(
            "Provide run_dir or both best_checkpoint and last_checkpoint"
        )
    checkpoints = _checkpoint_directories(run_dir)
    last = checkpoints[-1]
    state_candidates = [
        os.path.join(last, "trainer_state.json"),
        os.path.join(os.path.abspath(run_dir), "trainer_state.json"),
        *(os.path.join(path, "trainer_state.json") for path in reversed(checkpoints)),
    ]
    best_value = None
    inspected = set()
    for state_path in state_candidates:
        if state_path in inspected or not os.path.isfile(state_path):
            continue
        inspected.add(state_path)
        with open(state_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        candidate = payload.get("best_model_checkpoint")
        if candidate:
            best_value = str(candidate)
            break
    if best_value is None:
        raise FileNotFoundError(
            "Could not find best_model_checkpoint in retained trainer_state.json files"
        )
    return CheckpointPair(
        best=_resolve_recorded_checkpoint(best_value, run_dir=run_dir),
        last=last,
    )


def prepare_diagnostic_feature_batches(
    training_config,
    *,
    split: str,
    max_assays: int,
    batch_items: int,
    max_batches: int,
    seed: int,
    sampling_epoch: int = 0,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """Select and sample a fixed set of assay-list features for replay."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    if max_assays <= 0 or batch_items <= 0 or max_batches <= 0:
        raise ValueError("max_assays, batch_items, and max_batches must be > 0")
    dataset_path = get_tokenized_split_dataset_paths(
        training_config.data.tokenized_dataset_dir
    )[split]
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"Tokenized {split} dataset does not exist: {dataset_path}"
        )
    examples = load_tokenized_example_dataset(dataset_path)
    model_config = getattr(training_config, "model", None)
    if model_config is not None:
        validate_tokenized_split_cardinality(
            training_config.data,
            model_config,
            {split: examples},
        )
    selected, assay_ids = select_complete_assays(
        examples,
        max_assays=max_assays,
        seed=seed,
        min_size=3,
        min_pchembl_span=training_config.data.ranking_min_pchembl_span,
    )
    if not assay_ids:
        raise ValueError(f"No eligible assays selected from {split}")
    sampled = RewardAssayListDataset(
        selected,
        seed=seed,
        ranking_max_ligands=training_config.data.ranking_max_ligands,
        ranking_opportunity_divisor=training_config.data.ranking_opportunity_divisor,
        ranking_min_pchembl_span=training_config.data.ranking_min_pchembl_span,
        max_classification_only_per_item=0,
        include_all_assays_for_contrastive=True,
    )
    sampled.set_epoch(sampling_epoch)
    item_count = min(len(sampled), batch_items * max_batches)
    features = [sampled[index] for index in range(item_count)]
    feature_batches = [
        features[start : start + batch_items]
        for start in range(0, len(features), batch_items)
    ]
    metadata = {
        "split": split,
        "seed": int(seed),
        "sampling_epoch": int(sampling_epoch),
        "selected_assays": len(assay_ids),
        "selected_examples": len(selected),
        "sampled_dataset_items": len(sampled),
        "analyzed_dataset_items": item_count,
        "analyzed_batches": len(feature_batches),
        "batch_items": int(batch_items),
        "assay_ids": list(assay_ids),
    }
    return feature_batches, metadata


def _autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if precision != "bf16":
        raise ValueError("precision must be fp32 or bf16")
    if device.type not in {"cuda", "cpu"}:
        raise ValueError(f"bf16 autocast is unsupported for device type {device.type}")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA device does not support bf16")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _tensor_gradient_norm(
    loss: torch.Tensor | None,
    tensors: Sequence[torch.Tensor | None],
    *,
    weight: float = 1.0,
) -> float:
    active_tensors = [tensor for tensor in tensors if tensor is not None]
    if loss is None or not active_tensors or weight == 0.0:
        return 0.0
    gradients = torch.autograd.grad(
        loss * float(weight),
        active_tensors,
        retain_graph=True,
        allow_unused=True,
    )
    total = torch.zeros((), device=loss.device, dtype=torch.float32)
    for gradient in gradients:
        if gradient is not None:
            total += gradient.detach().float().square().sum()
    return float(total.sqrt().cpu().item())


def _parameter_norm(parameters: Iterable[torch.nn.Parameter], *, gradient: bool) -> float:
    total = None
    for parameter in parameters:
        value = parameter.grad if gradient else parameter.detach()
        if value is None:
            continue
        square = value.detach().float().square().sum()
        total = square if total is None else total + square
    return 0.0 if total is None else float(total.sqrt().cpu().item())


def _finite_parameter_gradients(parameters: Iterable[torch.nn.Parameter]) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
    )


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "count": int(array.size),
            "finite_count": 0,
            "min": None,
            "p01": None,
            "p05": None,
            "median": None,
            "mean": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "min": float(finite.min()),
        "p01": float(np.quantile(finite, 0.01)),
        "p05": float(np.quantile(finite, 0.05)),
        "median": float(np.median(finite)),
        "mean": float(finite.mean()),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)),
        "max": float(finite.max()),
    }


def _append_tensor(series: dict[str, list[float]], key: str, value: torch.Tensor) -> None:
    series.setdefault(key, []).extend(
        value.detach().float().reshape(-1).cpu().tolist()
    )


def _batch_model_inputs(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: batch[key].to(device)
        for key in MODEL_INPUT_KEYS
        if key in batch
    } | {"return_dict": True}


def _projection_capture_hooks(model, captures: dict[str, torch.Tensor]):
    hooks = []
    for modality in ("protein", "molecule"):
        projection = getattr(model, f"{modality}_projection")

        def capture_projection(_module, _inputs, output, *, name=modality):
            captures[f"{name}_encoder_pooled"] = _inputs[0]
            captures[f"{name}_raw"] = output

        hooks.append(projection.register_forward_hook(capture_projection))
        linear1 = getattr(projection, "linear1", None)
        if linear1 is not None:

            def capture_linear1(_module, _inputs, output, *, name=modality):
                captures[f"{name}_linear1"] = output

            hooks.append(linear1.register_forward_hook(capture_linear1))
        activation = getattr(projection, "activation", None)
        if activation is not None:

            def capture_activation(_module, _inputs, output, *, name=modality):
                captures[f"{name}_relu"] = output

            hooks.append(activation.register_forward_hook(capture_activation))
    return hooks


def analyze_embedding_matrix(
    embeddings: torch.Tensor,
    identity_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Measure directional and centered diversity in one representation matrix."""
    values = embeddings.detach().float().cpu()
    if values.ndim != 2:
        raise ValueError("embedding matrix must have shape [num_vectors, dimension]")
    num_vectors, dimension = values.shape
    if num_vectors == 0 or dimension == 0:
        raise ValueError("embedding matrix must be non-empty")
    if identity_ids is not None and len(identity_ids) != num_vectors:
        raise ValueError("identity_ids must align with the embedding matrix")
    finite_rows = torch.isfinite(values).all(dim=1)
    finite_identity_ids = (
        [
            str(identity_ids[index])
            for index in torch.nonzero(finite_rows, as_tuple=False).flatten().tolist()
        ]
        if identity_ids is not None
        else None
    )
    values = values[finite_rows]
    if values.size(0) == 0:
        return {
            "num_vectors": int(num_vectors),
            "finite_vectors": 0,
            "dimension": int(dimension),
        }

    centroid = values.mean(dim=0)
    centered = values - centroid
    per_dimension_variance = centered.square().mean(dim=0)
    variance_trace = float(per_dimension_variance.sum().item())
    rms_radius = float(centered.square().sum(dim=1).mean().sqrt().item())
    centroid_norm = float(centroid.norm().item())
    norms = values.norm(dim=1)
    normalized = torch.nn.functional.normalize(values, p=2, dim=1, eps=1e-12)
    normalized_sum = normalized.sum(dim=0)
    finite_count = int(values.size(0))
    if finite_count > 1:
        pairwise_cosine_mean = float(
            (
                normalized_sum.square().sum() - finite_count
            ).div(finite_count * (finite_count - 1)).item()
        )
    else:
        pairwise_cosine_mean = None

    exact_unique = int(torch.unique(values, dim=0).size(0))
    rounded_normalized = torch.round(normalized * 10_000).to(torch.int32)
    rounded_unique = int(torch.unique(rounded_normalized, dim=0).size(0))

    # The non-zero eigenvalues of X X^T equal squared singular values of X.
    # Using the smaller Gram matrix keeps the analysis bounded when the encoder
    # width exceeds the number of diagnostic examples.
    gram = centered @ centered.transpose(0, 1)
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    eigenvalue_sum = eigenvalues.sum()
    if float(eigenvalue_sum.item()) > 0.0:
        probabilities = eigenvalues / eigenvalue_sum
        positive = probabilities > 0
        effective_rank = float(
            (-(probabilities[positive] * probabilities[positive].log()).sum())
            .exp()
            .item()
        )
        largest = float(eigenvalues.max().item())
        stable_rank = float(eigenvalue_sum.item()) / largest if largest > 0 else 0.0
        descending = eigenvalues.flip(0)
        explained = descending.cumsum(0) / eigenvalue_sum

        def explained_at(count: int) -> float:
            return float(explained[min(count, explained.numel()) - 1].item())

        top_explained = {
            "top1": explained_at(1),
            "top5": explained_at(5),
            "top10": explained_at(10),
        }
    else:
        effective_rank = 0.0
        stable_rank = 0.0
        top_explained = {"top1": 0.0, "top5": 0.0, "top10": 0.0}

    maximum_variance = float(per_dimension_variance.max().item())
    active_variance_threshold = max(1.0e-12, maximum_variance * 1.0e-8)
    result = {
        "num_vectors": int(num_vectors),
        "finite_vectors": finite_count,
        "dimension": int(dimension),
        "norm": _distribution(norms.tolist()),
        "per_dimension_variance": _distribution(
            per_dimension_variance.tolist()
        ),
        "variance_trace": variance_trace,
        "rms_radius": rms_radius,
        "centroid_norm": centroid_norm,
        "radius_to_centroid_norm": (
            rms_radius / centroid_norm if centroid_norm > 0.0 else None
        ),
        "mean_pairwise_cosine": pairwise_cosine_mean,
        "exact_unique_ratio": exact_unique / finite_count,
        "normalized_unique_ratio_1e4": rounded_unique / finite_count,
        "active_variance_dimension_fraction": float(
            (per_dimension_variance > active_variance_threshold)
            .float()
            .mean()
            .item()
        ),
        "effective_rank_centered": effective_rank,
        "stable_rank_centered": stable_rank,
        "explained_variance_fraction": top_explained,
    }
    if finite_identity_ids is not None:
        first_indices: dict[str, int] = {}
        identity_members: dict[str, list[int]] = {}
        for row_index, identity in enumerate(finite_identity_ids):
            first_indices.setdefault(identity, row_index)
            identity_members.setdefault(identity, []).append(row_index)
        unique_indices = torch.tensor(
            list(first_indices.values()),
            dtype=torch.long,
        )
        repeated_residual_squares = []
        for members in identity_members.values():
            if len(members) <= 1:
                continue
            member_values = values.index_select(
                0,
                torch.tensor(members, dtype=torch.long),
            )
            residuals = member_values - member_values.mean(dim=0)
            repeated_residual_squares.extend(
                residuals.square().sum(dim=1).tolist()
            )
        result["identity_count"] = len(first_indices)
        result["identity_deduplicated"] = analyze_embedding_matrix(
            values.index_select(0, unique_indices)
        )
        result["within_identity_rms_radius"] = (
            math.sqrt(float(np.mean(repeated_residual_squares)))
            if repeated_residual_squares
            else 0.0
        )
    return result


def _flatten_feature_rows(
    features: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [row for feature in features for row in feature.get("rows", [])]


def analyze_contrastive_retrieval(
    normalized_protein: torch.Tensor,
    normalized_molecule: torch.Tensor,
    batch: Mapping[str, Any],
    *,
    temperature: float,
    active_threshold: float,
    row_metadata: Sequence[Mapping[str, Any]] | None = None,
    worst_query_limit: int = 20,
) -> dict[str, Any]:
    """Evaluate the exact masked retrieval candidates used by the loss."""
    if temperature <= 0.0 or not math.isfinite(float(temperature)):
        raise ValueError("temperature must be finite and > 0")
    group_ids = batch["contrastive_group_ids"].reshape(-1).to(
        device=normalized_protein.device,
        dtype=torch.long,
    )
    pchembl = batch["pchembl_values"].reshape(-1).to(
        device=normalized_protein.device,
        dtype=torch.float32,
    )
    target_ids = batch["contrastive_target_ids"].reshape(-1).to(
        device=normalized_protein.device,
        dtype=torch.long,
    )
    molecule_ids = batch["contrastive_molecule_ids"].reshape(-1).to(
        device=normalized_protein.device,
        dtype=torch.long,
    )
    valid_indices = torch.nonzero(group_ids >= 0, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        raise ValueError("retrieval analysis requires contrastive examples")
    valid_group_ids = group_ids.index_select(0, valid_indices)
    _, owners = torch.unique(valid_group_ids, sorted=True, return_inverse=True)
    num_groups = int(owners.max().item()) + 1
    membership = torch.nn.functional.one_hot(
        owners,
        num_classes=num_groups,
    ).transpose(0, 1).bool()
    representative_positions = membership.long().argmax(dim=1)
    representative_indices = valid_indices.index_select(
        0,
        representative_positions,
    )
    valid_targets = target_ids.index_select(0, valid_indices)
    group_targets = valid_targets.index_select(0, representative_positions)
    valid_molecules = molecule_ids.index_select(0, valid_indices)
    group_proteins = normalized_protein.index_select(
        0,
        representative_indices,
    ).float()
    ligand_molecules = normalized_molecule.index_select(0, valid_indices).float()
    cosine_matrix = group_proteins @ ligand_molecules.transpose(0, 1)
    score_matrix = cosine_matrix / float(temperature)

    ligand_target_ids = group_targets.index_select(0, owners)
    same_target = group_targets[:, None] == ligand_target_ids[None, :]
    same_molecule = valid_molecules[:, None] == valid_molecules[None, :]
    duplicate_molecule = (
        membership.to(dtype=score_matrix.dtype)
        @ same_molecule.to(dtype=score_matrix.dtype)
    ) > 0
    false_negative_mask = (~membership) & (same_target | duplicate_molecule)
    masked_scores = score_matrix.masked_fill(false_negative_mask, -torch.inf)
    ligand_positions = torch.arange(
        valid_indices.numel(),
        device=score_matrix.device,
        dtype=torch.long,
    )
    group_sizes = torch.bincount(owners, minlength=num_groups)
    positive_scores = score_matrix[owners, ligand_positions]

    p2m_negative_mask = ~(owners[:, None] == owners[None, :])
    p2m_negative_mask &= ~false_negative_mask.index_select(0, owners)
    p2m_candidate_scores = score_matrix.index_select(0, owners).masked_fill(
        ~p2m_negative_mask,
        -torch.inf,
    )
    p2m_hardest, p2m_hardest_indices = p2m_candidate_scores.max(dim=1)
    p2m_has_negative = torch.isfinite(p2m_hardest)
    p2m_eligible = (
        (group_sizes.index_select(0, owners) == 1)
        | (pchembl.index_select(0, valid_indices) >= float(active_threshold))
    ) & p2m_has_negative
    p2m_margin = positive_scores - p2m_hardest

    m2p_negative_mask = torch.ones_like(masked_scores, dtype=torch.bool)
    m2p_negative_mask[owners, ligand_positions] = False
    m2p_negative_mask &= ~false_negative_mask
    m2p_candidate_scores = score_matrix.masked_fill(~m2p_negative_mask, -torch.inf)
    m2p_hardest, m2p_hardest_groups = m2p_candidate_scores.max(dim=0)
    m2p_has_negative = torch.isfinite(m2p_hardest)
    m2p_margin = positive_scores - m2p_hardest

    metadata = list(row_metadata or [])
    valid_rows = [
        metadata[int(index)] if int(index) < len(metadata) else {}
        for index in valid_indices.detach().cpu().tolist()
    ]
    representative_rows = [
        valid_rows[int(index)]
        for index in representative_positions.detach().cpu().tolist()
    ]

    def direction_report(
        *,
        eligible: torch.Tensor,
        hardest_scores: torch.Tensor,
        margins: torch.Tensor,
        hardest_indices: torch.Tensor,
        direction: str,
    ) -> dict[str, Any]:
        indices = torch.nonzero(eligible, as_tuple=False).flatten()
        selected_positive = positive_scores.index_select(0, indices)
        selected_hardest = hardest_scores.index_select(0, indices)
        selected_margins = margins.index_select(0, indices)
        correct = selected_margins > 0.0
        failures = []
        worst_order = torch.argsort(selected_margins)[:worst_query_limit]
        for local_index in worst_order.detach().cpu().tolist():
            ligand_index = int(indices[local_index].item())
            hardest_index = int(hardest_indices[ligand_index].item())
            ligand_row = valid_rows[ligand_index]
            if direction == "protein_to_molecule":
                query_row = representative_rows[int(owners[ligand_index].item())]
                predicted_row = valid_rows[hardest_index]
            else:
                query_row = ligand_row
                predicted_row = representative_rows[hardest_index]
            failures.append(
                {
                    "query": str(
                        (
                            query_row.get("group_id")
                            if direction == "protein_to_molecule"
                            else query_row.get("compound_id")
                        )
                        or ligand_index
                    ),
                    "positive": str(
                        ligand_row.get("compound_id")
                        if direction == "protein_to_molecule"
                        else ligand_row.get("group_id")
                    ),
                    "hardest_negative": str(
                        predicted_row.get("compound_id")
                        if direction == "protein_to_molecule"
                        else predicted_row.get("group_id")
                    ),
                    "pchembl": float(
                        pchembl.index_select(0, valid_indices)[ligand_index]
                        .detach()
                        .cpu()
                        .item()
                    ),
                    "positive_score": float(selected_positive[local_index].item()),
                    "hardest_negative_score": float(
                        selected_hardest[local_index].item()
                    ),
                    "margin": float(selected_margins[local_index].item()),
                    "cosine_margin": float(
                        selected_margins[local_index].item() * temperature
                    ),
                }
            )
        return {
            "queries": int(indices.numel()),
            "strict_top1_correct": int(correct.sum().item()),
            "strict_top1_accuracy": (
                float(correct.float().mean().item())
                if indices.numel() > 0
                else None
            ),
            "positive_score": _distribution(selected_positive.detach().cpu().tolist()),
            "hardest_negative_score": _distribution(
                selected_hardest.detach().cpu().tolist()
            ),
            "margin": _distribution(selected_margins.detach().cpu().tolist()),
            "cosine_margin": _distribution(
                (selected_margins * float(temperature)).detach().cpu().tolist()
            ),
            "worst_queries": failures,
        }

    return {
        "scope": "replayed_training_microbatch",
        "num_groups": num_groups,
        "num_ligands": int(valid_indices.numel()),
        "score_matrix_shape": [num_groups, int(valid_indices.numel())],
        "score_matrix": {
            "min": float(score_matrix.min().item()),
            "max": float(score_matrix.max().item()),
            "mean": float(score_matrix.mean().item()),
            "std": float(score_matrix.std(unbiased=False).item()),
        },
        "protein_to_molecule": direction_report(
            eligible=p2m_eligible,
            hardest_scores=p2m_hardest,
            margins=p2m_margin,
            hardest_indices=p2m_hardest_indices,
            direction="protein_to_molecule",
        ),
        "molecule_to_protein": direction_report(
            eligible=m2p_has_negative,
            hardest_scores=m2p_hardest,
            margins=m2p_margin,
            hardest_indices=m2p_hardest_groups,
            direction="molecule_to_protein",
        ),
    }


def aggregate_contrastive_retrieval(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate exact per-microbatch retrieval results without changing scope."""
    result: dict[str, Any] = {
        "scope": "aggregate_of_replayed_training_microbatches",
        "microbatches": len(reports),
        "num_groups": sum(int(report["num_groups"]) for report in reports),
        "num_ligands": sum(int(report["num_ligands"]) for report in reports),
    }
    for direction in ("protein_to_molecule", "molecule_to_protein"):
        queries = sum(int(report[direction]["queries"]) for report in reports)
        correct = sum(
            int(report[direction]["strict_top1_correct"])
            for report in reports
        )
        mean_margin_numerator = sum(
            int(report[direction]["queries"])
            * float(report[direction]["margin"]["mean"] or 0.0)
            for report in reports
        )
        worst_queries = [
            {"batch_index": report.get("batch_index"), **query}
            for report in reports
            for query in report[direction]["worst_queries"]
        ]
        worst_queries.sort(key=lambda query: float(query["margin"]))
        result[direction] = {
            "queries": queries,
            "strict_top1_correct": correct,
            "strict_top1_accuracy": correct / queries if queries > 0 else None,
            "mean_margin": (
                mean_margin_numerator / queries if queries > 0 else None
            ),
            "mean_cosine_margin": (
                sum(
                    int(report[direction]["queries"])
                    * float(
                        report[direction]["cosine_margin"]["mean"] or 0.0
                    )
                    for report in reports
                )
                / queries
                if queries > 0
                else None
            ),
            "worst_queries": worst_queries[:20],
        }
    return result


def analyze_model_stability(
    model,
    feature_batches: Sequence[Sequence[Mapping[str, Any]]],
    *,
    device: torch.device,
    precision: str,
    mode: str,
    seed: int,
    label: str,
) -> dict[str, Any]:
    """Replay fixed batches and attribute checkpoint gradient amplification."""
    if mode not in {"eval", "train"}:
        raise ValueError("mode must be eval or train")
    if not feature_batches:
        raise ValueError("feature_batches must not be empty")
    collator = RewardAssayListCollator(
        dynamic_padding=True,
        protein_pad_token_id=getattr(model.protein_tokenizer, "pad_token_id", 0),
        molecule_pad_token_id=getattr(model.molecule_tokenizer, "pad_token_id", 0),
    )
    modules = {
        "protein_encoder": model.protein_encoder,
        "molecule_encoder": model.molecule_encoder,
        "protein_projection": model.protein_projection,
        "molecule_projection": model.molecule_projection,
    }
    module_parameter_norms = {
        name: _parameter_norm(module.parameters(), gradient=False)
        for name, module in modules.items()
    }
    series: dict[str, list[float]] = {}
    boundary_matrices: dict[str, list[torch.Tensor]] = {}
    boundary_identity_ids: dict[str, list[str]] = {
        "protein": [],
        "molecule": [],
    }
    retrieval_reports: list[dict[str, Any]] = []
    batch_records: list[dict[str, Any]] = []
    nonfinite_batches = 0
    was_training = model.training
    model.train(mode == "train")
    try:
        for batch_index, features in enumerate(feature_batches):
            torch.manual_seed(seed + batch_index)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed + batch_index)
            model.zero_grad(set_to_none=True)
            captures: dict[str, torch.Tensor] = {}
            hooks = _projection_capture_hooks(model, captures)
            try:
                batch = collator(features)
                model_inputs = _batch_model_inputs(batch, device)
                with _autocast_context(device, precision):
                    outputs = model(**model_inputs)
            finally:
                for hook in hooks:
                    hook.remove()

            required = {
                "loss": outputs.loss,
                "protein_raw": captures.get("protein_raw"),
                "molecule_raw": captures.get("molecule_raw"),
                "normalized_protein": outputs.normalized_protein_embedding,
                "normalized_molecule": outputs.normalized_molecule_embedding,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise RuntimeError(
                    f"Checkpoint analysis requires cosine outputs and loss; missing {missing}"
                )
            raw_embeddings = [captures["protein_raw"], captures["molecule_raw"]]
            normalized_embeddings = [
                outputs.normalized_protein_embedding,
                outputs.normalized_molecule_embedding,
            ]
            ranking_weight = float(model.config.ranking_loss_weight)
            contrastive_weight = float(model.config.contrastive_loss_weight)
            objective_gradients = {
                "ranking_normalized_embedding_grad_norm": _tensor_gradient_norm(
                    outputs.ranking_loss,
                    normalized_embeddings,
                    weight=ranking_weight,
                ),
                "contrastive_normalized_embedding_grad_norm": _tensor_gradient_norm(
                    outputs.contrastive_loss,
                    normalized_embeddings,
                    weight=contrastive_weight,
                ),
                "ranking_raw_projection_grad_norm": _tensor_gradient_norm(
                    outputs.ranking_loss,
                    raw_embeddings,
                    weight=ranking_weight,
                ),
                "contrastive_raw_projection_grad_norm": _tensor_gradient_norm(
                    outputs.contrastive_loss,
                    raw_embeddings,
                    weight=contrastive_weight,
                ),
                "total_raw_projection_grad_norm": _tensor_gradient_norm(
                    outputs.loss,
                    raw_embeddings,
                ),
            }
            outputs.loss.backward()
            total_gradient_norm = _parameter_norm(model.parameters(), gradient=True)
            module_gradient_norms = {
                name: _parameter_norm(module.parameters(), gradient=True)
                for name, module in modules.items()
            }
            gradients_finite = _finite_parameter_gradients(model.parameters())
            output_tensors = [
                outputs.loss,
                outputs.ranking_loss,
                outputs.contrastive_loss,
                outputs.ranking_score,
                *raw_embeddings,
                *normalized_embeddings,
            ]
            outputs_finite = all(
                value is None or bool(torch.isfinite(value).all())
                for value in output_tensors
            )
            if not gradients_finite or not outputs_finite:
                nonfinite_batches += 1

            protein_raw_norm = captures["protein_raw"].detach().float().norm(dim=-1)
            molecule_raw_norm = captures["molecule_raw"].detach().float().norm(dim=-1)
            protein_normalized_norm = (
                outputs.normalized_protein_embedding.detach().float().norm(dim=-1)
            )
            molecule_normalized_norm = (
                outputs.normalized_molecule_embedding.detach().float().norm(dim=-1)
            )
            cosine = outputs.cosine_similarity.detach().float()
            _append_tensor(series, "protein_raw_norm", protein_raw_norm)
            _append_tensor(series, "molecule_raw_norm", molecule_raw_norm)
            _append_tensor(series, "protein_normalized_norm", protein_normalized_norm)
            _append_tensor(series, "molecule_normalized_norm", molecule_normalized_norm)
            _append_tensor(series, "cosine", cosine)
            for capture_name in (
                "protein_encoder_pooled",
                "protein_linear1",
                "protein_relu",
                "protein_raw",
                "molecule_encoder_pooled",
                "molecule_linear1",
                "molecule_relu",
                "molecule_raw",
            ):
                captured = captures.get(capture_name)
                if captured is not None:
                    boundary_matrices.setdefault(capture_name, []).append(
                        captured.detach().float().cpu()
                    )
            boundary_matrices.setdefault("protein_normalized", []).append(
                outputs.normalized_protein_embedding.detach().float().cpu()
            )
            boundary_matrices.setdefault("molecule_normalized", []).append(
                outputs.normalized_molecule_embedding.detach().float().cpu()
            )
            retrieval = analyze_contrastive_retrieval(
                outputs.normalized_protein_embedding,
                outputs.normalized_molecule_embedding,
                batch,
                temperature=float(model.config.ranking_temperature),
                active_threshold=float(model.config.contrastive_active_threshold),
                row_metadata=_flatten_feature_rows(features),
            )
            retrieval["batch_index"] = int(batch_index)
            retrieval_reports.append(retrieval)
            feature_rows = _flatten_feature_rows(features)
            boundary_identity_ids["protein"].extend(
                str(row.get("target_chembl_id", row.get("group_id", index)))
                for index, row in enumerate(feature_rows)
            )
            boundary_identity_ids["molecule"].extend(
                str(row.get("compound_id", index))
                for index, row in enumerate(feature_rows)
            )

            relu_metrics: dict[str, float] = {}
            for modality in ("protein", "molecule"):
                activation = captures.get(f"{modality}_relu")
                if activation is None:
                    continue
                active = activation.detach() > 0
                relu_metrics[f"{modality}_relu_positive_fraction"] = float(
                    active.float().mean().cpu().item()
                )
                relu_metrics[f"{modality}_relu_dead_row_fraction"] = float(
                    (~active.any(dim=-1)).float().mean().cpu().item()
                )

            record = {
                "label": label,
                "mode": mode,
                "batch_index": int(batch_index),
                "num_examples": int(batch["num_examples"].item()),
                "num_contrastive_lists": int(
                    batch["num_contrastive_lists"].item()
                ),
                "num_ranking_lists": int(batch["num_ranking_lists"].item()),
                "protein_token_width": int(batch["protein_input_ids"].shape[1]),
                "molecule_token_width": int(batch["molecule_input_ids"].shape[1]),
                "loss": float(outputs.loss.detach().float().cpu().item()),
                "ranking_loss": (
                    None
                    if outputs.ranking_loss is None
                    else float(outputs.ranking_loss.detach().float().cpu().item())
                ),
                "contrastive_loss": (
                    None
                    if outputs.contrastive_loss is None
                    else float(outputs.contrastive_loss.detach().float().cpu().item())
                ),
                "cosine_std": float(cosine.std(unbiased=False).cpu().item()),
                "protein_raw_norm_min": float(protein_raw_norm.min().cpu().item()),
                "protein_raw_norm_mean": float(protein_raw_norm.mean().cpu().item()),
                "molecule_raw_norm_min": float(molecule_raw_norm.min().cpu().item()),
                "molecule_raw_norm_mean": float(molecule_raw_norm.mean().cpu().item()),
                "total_parameter_grad_norm": total_gradient_norm,
                "clip_scale_at_norm_1": (
                    min(1.0, 1.0 / total_gradient_norm)
                    if total_gradient_norm > 0.0
                    else 1.0
                ),
                "outputs_finite": bool(outputs_finite),
                "gradients_finite": bool(gradients_finite),
                **objective_gradients,
                **{
                    f"{name}_grad_norm": value
                    for name, value in module_gradient_norms.items()
                },
                **relu_metrics,
            }
            batch_records.append(record)
            for key, value in record.items():
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and key
                    not in {
                        "batch_index",
                        "num_examples",
                        "num_contrastive_lists",
                        "num_ranking_lists",
                        "protein_token_width",
                        "molecule_token_width",
                    }
                ):
                    series.setdefault(key, []).append(float(value))
    finally:
        model.train(was_training)
        model.zero_grad(set_to_none=True)

    return {
        "label": label,
        "mode": mode,
        "num_batches": len(batch_records),
        "num_examples": sum(record["num_examples"] for record in batch_records),
        "nonfinite_batches": int(nonfinite_batches),
        "module_parameter_norms": module_parameter_norms,
        "representations": {
            name: analyze_embedding_matrix(
                torch.cat(matrices, dim=0),
                boundary_identity_ids[
                    "protein" if name.startswith("protein_") else "molecule"
                ],
            )
            for name, matrices in sorted(boundary_matrices.items())
        },
        "contrastive_retrieval": {
            "aggregate": aggregate_contrastive_retrieval(retrieval_reports),
            "microbatches": retrieval_reports,
        },
        "distributions": {
            key: _distribution(values) for key, values in sorted(series.items())
        },
        "batches": batch_records,
    }


def snapshot_parameters(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def _parameter_snapshot_metrics(
    best: Mapping[str, torch.Tensor],
    last: Mapping[str, torch.Tensor],
    names: Sequence[str],
) -> dict[str, float | int | None]:
    best_square = 0.0
    last_square = 0.0
    delta_square = 0.0
    dot = 0.0
    max_abs_delta = 0.0
    num_parameters = 0
    for name in names:
        best_value = best[name].double()
        last_value = last[name].double()
        delta = last_value - best_value
        best_square += float(best_value.square().sum().item())
        last_square += float(last_value.square().sum().item())
        delta_square += float(delta.square().sum().item())
        dot += float((best_value * last_value).sum().item())
        max_abs_delta = max(max_abs_delta, float(delta.abs().max().item()))
        num_parameters += best_value.numel()
    best_norm = math.sqrt(best_square)
    last_norm = math.sqrt(last_square)
    delta_norm = math.sqrt(delta_square)
    denominator = best_norm * last_norm
    return {
        "num_parameters": int(num_parameters),
        "best_norm": best_norm,
        "last_norm": last_norm,
        "delta_norm": delta_norm,
        "relative_delta_to_best": (
            delta_norm / best_norm if best_norm > 0.0 else None
        ),
        "parameter_cosine": dot / denominator if denominator > 0.0 else None,
        "max_abs_delta": max_abs_delta,
    }


def compare_parameter_snapshots(
    best: Mapping[str, torch.Tensor],
    last: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, float | int | None]]:
    if set(best) != set(last):
        missing_best = sorted(set(last) - set(best))
        missing_last = sorted(set(best) - set(last))
        raise ValueError(
            "Checkpoint parameter names differ: "
            f"missing_best={missing_best[:5]}, missing_last={missing_last[:5]}"
        )
    groups = {"all": "", **MODULE_PREFIXES}
    result: dict[str, dict[str, float | int | None]] = {}
    for group_name, prefix in groups.items():
        names = [name for name in best if name.startswith(prefix)]
        if not names:
            continue
        result[group_name] = _parameter_snapshot_metrics(best, last, names)
    return result


def _encoder_parameter_region(parameter_name: str, prefix: str) -> str:
    remainder = parameter_name[len(prefix) :]
    if remainder.startswith("embeddings."):
        return "embeddings"
    layer_match = re.match(r"encoder\.layer\.(\d+)\.", remainder)
    if layer_match is not None:
        return f"encoder_layer_{int(layer_match.group(1)):02d}"
    if remainder.startswith("encoder.emb_layer_norm_after."):
        return "encoder_final_norm"
    if remainder.startswith("pooler."):
        return "pooler"
    first_component = remainder.split(".", 1)[0]
    return f"other_{first_component}"


def compare_encoder_layer_snapshots(
    best: Mapping[str, torch.Tensor],
    last: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Attribute encoder drift to embeddings, individual layers, and norms."""
    if set(best) != set(last):
        raise ValueError("Checkpoint parameter names differ")
    result = {}
    for encoder_name in ("protein_encoder", "molecule_encoder"):
        prefix = f"{encoder_name}."
        regions: dict[str, list[str]] = {}
        for parameter_name in best:
            if parameter_name.startswith(prefix):
                region = _encoder_parameter_region(parameter_name, prefix)
                regions.setdefault(region, []).append(parameter_name)
        region_metrics = {
            region: _parameter_snapshot_metrics(best, last, names)
            for region, names in sorted(regions.items())
        }
        result[encoder_name] = {
            "regions": region_metrics,
            "largest_relative_drift": sorted(
                (
                    {
                        "region": region,
                        **metrics,
                    }
                    for region, metrics in region_metrics.items()
                ),
                key=lambda record: float(
                    record.get("relative_delta_to_best") or 0.0
                ),
                reverse=True,
            ),
            "largest_absolute_drift": sorted(
                (
                    {
                        "region": region,
                        **metrics,
                    }
                    for region, metrics in region_metrics.items()
                ),
                key=lambda record: float(record.get("delta_norm") or 0.0),
                reverse=True,
            ),
        }
    return result


def _optimizer_parameter_groups(
    model,
    training_config,
) -> list[list[str]]:
    """Reproduce the exact parameter grouping order used by RewardModelTrainer."""
    training = training_config.training
    protein_encoder_learning_rate = getattr(
        training,
        "protein_encoder_learning_rate",
        None,
    )
    molecule_encoder_learning_rate = getattr(
        training,
        "molecule_encoder_learning_rate",
        None,
    )
    forbidden_name_patterns = [
        r"bias",
        r"layernorm",
        r"rmsnorm",
        r"(?:^|\.)norm(?:$|\.)",
        r"_norm(?:$|\.)",
    ]
    decay_parameters = set(
        get_parameter_names(
            model,
            [torch.nn.LayerNorm],
            forbidden_name_patterns,
        )
    )
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if (
        training.encoder_learning_rate is None
        and protein_encoder_learning_rate is None
        and molecule_encoder_learning_rate is None
        and training.projection_learning_rate is None
    ):
        return [
            [name for name, _ in trainable if name in decay_parameters],
            [name for name, _ in trainable if name not in decay_parameters],
        ]

    grouped: dict[tuple[float, float], list[str]] = {}
    for name, _ in trainable:
        if name.startswith("protein_encoder."):
            learning_rate = float(
                protein_encoder_learning_rate
                or training.encoder_learning_rate
                or training.learning_rate
            )
        elif name.startswith("molecule_encoder."):
            learning_rate = float(
                molecule_encoder_learning_rate
                or training.encoder_learning_rate
                or training.learning_rate
            )
        elif name.startswith(("protein_projection.", "molecule_projection.")):
            learning_rate = float(
                training.projection_learning_rate or training.learning_rate
            )
        else:
            learning_rate = float(training.learning_rate)
        weight_decay = (
            float(training.weight_decay) if name in decay_parameters else 0.0
        )
        grouped.setdefault((learning_rate, weight_decay), []).append(name)
    return list(grouped.values())


def _optimizer_state_parameter_names(
    optimizer_state: Mapping[str, Any],
    model,
    training_config,
) -> dict[Any, str]:
    saved_groups = optimizer_state.get("param_groups")
    if not isinstance(saved_groups, list):
        raise ValueError("optimizer.pt does not contain param_groups")
    expected_groups = _optimizer_parameter_groups(model, training_config)
    if len(saved_groups) != len(expected_groups):
        raise ValueError(
            "Optimizer parameter group count does not match the configured trainer: "
            f"saved={len(saved_groups)}, expected={len(expected_groups)}"
        )
    result = {}
    for group_index, (saved_group, expected_names) in enumerate(
        zip(saved_groups, expected_groups)
    ):
        saved_ids = saved_group.get("params", [])
        if len(saved_ids) != len(expected_names):
            raise ValueError(
                "Optimizer parameter group size does not match the configured model: "
                f"group={group_index}, saved={len(saved_ids)}, "
                f"expected={len(expected_names)}"
            )
        result.update(zip(saved_ids, expected_names))
    return result


def _optimizer_module_name(parameter_name: str) -> str:
    for module_name, prefix in MODULE_PREFIXES.items():
        if parameter_name.startswith(prefix):
            return module_name
    return "other"


def _empty_optimizer_accumulator() -> dict[str, Any]:
    return {
        "parameter_states": 0,
        "tensor_elements": 0,
        "nonfinite_tensors": 0,
        "exp_avg_square_sum": 0.0,
        "exp_avg_sq_sum": 0.0,
        "exp_avg_sq_max": 0.0,
        "normalized_moment_square_sum": 0.0,
        "normalized_moment_max_abs": 0.0,
        "steps": [],
        "parameter_exp_avg_l2": [],
        "parameter_exp_avg_sq_max": [],
        "parameter_normalized_moment_l2": [],
    }


def _finalize_optimizer_accumulator(accumulator: Mapping[str, Any]) -> dict[str, Any]:
    elements = int(accumulator["tensor_elements"])
    return {
        "parameter_states": int(accumulator["parameter_states"]),
        "tensor_elements": elements,
        "nonfinite_tensors": int(accumulator["nonfinite_tensors"]),
        "step": _distribution(accumulator["steps"]),
        "exp_avg_l2": math.sqrt(float(accumulator["exp_avg_square_sum"])),
        "exp_avg_sq_mean": (
            float(accumulator["exp_avg_sq_sum"]) / elements
            if elements > 0
            else None
        ),
        "exp_avg_sq_max": (
            float(accumulator["exp_avg_sq_max"]) if elements > 0 else None
        ),
        "max_stored_rms": (
            math.sqrt(float(accumulator["exp_avg_sq_max"]))
            if elements > 0
            else None
        ),
        "normalized_moment_l2": math.sqrt(
            float(accumulator["normalized_moment_square_sum"])
        ),
        "normalized_moment_max_abs": (
            float(accumulator["normalized_moment_max_abs"])
            if elements > 0
            else None
        ),
        "per_parameter_exp_avg_l2": _distribution(
            accumulator["parameter_exp_avg_l2"]
        ),
        "per_parameter_exp_avg_sq_max": _distribution(
            accumulator["parameter_exp_avg_sq_max"]
        ),
        "per_parameter_normalized_moment_l2": _distribution(
            accumulator["parameter_normalized_moment_l2"]
        ),
    }


def summarize_optimizer_state(
    optimizer_state: Mapping[str, Any],
    parameter_names: Mapping[Any, str],
    *,
    adam_epsilon: float = 1.0e-8,
    top_k: int = 20,
) -> dict[str, Any]:
    """Summarize Adam moments without materializing flattened model-size arrays."""
    raw_state = optimizer_state.get("state")
    if not isinstance(raw_state, Mapping):
        raise ValueError("optimizer.pt does not contain an optimizer state mapping")
    accumulators = {"all": _empty_optimizer_accumulator()}
    parameter_records = []
    unmapped_state_ids = []
    for state_id, state in raw_state.items():
        parameter_name = parameter_names.get(state_id)
        if parameter_name is None:
            unmapped_state_ids.append(state_id)
            continue
        if not isinstance(state, Mapping):
            continue
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        if not isinstance(exp_avg, torch.Tensor) or not isinstance(
            exp_avg_sq, torch.Tensor
        ):
            continue
        module_name = _optimizer_module_name(parameter_name)
        accumulator_targets = [
            accumulators["all"],
            accumulators.setdefault(module_name, _empty_optimizer_accumulator()),
        ]
        first_moment = exp_avg.detach().float()
        second_moment = exp_avg_sq.detach().float()
        finite = bool(torch.isfinite(first_moment).all()) and bool(
            torch.isfinite(second_moment).all()
        )
        if not finite:
            first_moment = torch.nan_to_num(first_moment)
            second_moment = torch.nan_to_num(second_moment)
        normalized_moment = first_moment / (
            second_moment.clamp_min(0.0).sqrt() + float(adam_epsilon)
        )
        exp_avg_square_sum = float(
            first_moment.square().sum(dtype=torch.float64).item()
        )
        exp_avg_sq_sum = float(second_moment.sum(dtype=torch.float64).item())
        exp_avg_sq_max = float(second_moment.max().item())
        normalized_square_sum = float(
            normalized_moment.square().sum(dtype=torch.float64).item()
        )
        normalized_max_abs = float(normalized_moment.abs().max().item())
        exp_avg_l2 = math.sqrt(exp_avg_square_sum)
        normalized_l2 = math.sqrt(normalized_square_sum)
        step = state.get("step", 0.0)
        if isinstance(step, torch.Tensor):
            step = float(step.detach().cpu().item())
        else:
            step = float(step)
        for accumulator in accumulator_targets:
            accumulator["parameter_states"] += 1
            accumulator["tensor_elements"] += first_moment.numel()
            accumulator["nonfinite_tensors"] += int(not finite)
            accumulator["exp_avg_square_sum"] += exp_avg_square_sum
            accumulator["exp_avg_sq_sum"] += exp_avg_sq_sum
            accumulator["exp_avg_sq_max"] = max(
                accumulator["exp_avg_sq_max"], exp_avg_sq_max
            )
            accumulator["normalized_moment_square_sum"] += normalized_square_sum
            accumulator["normalized_moment_max_abs"] = max(
                accumulator["normalized_moment_max_abs"], normalized_max_abs
            )
            accumulator["steps"].append(step)
            accumulator["parameter_exp_avg_l2"].append(exp_avg_l2)
            accumulator["parameter_exp_avg_sq_max"].append(exp_avg_sq_max)
            accumulator["parameter_normalized_moment_l2"].append(normalized_l2)
        parameter_records.append(
            {
                "name": parameter_name,
                "module": module_name,
                "step": step,
                "elements": int(first_moment.numel()),
                "exp_avg_l2": exp_avg_l2,
                "exp_avg_sq_max": exp_avg_sq_max,
                "max_stored_rms": math.sqrt(max(0.0, exp_avg_sq_max)),
                "normalized_moment_l2": normalized_l2,
                "normalized_moment_max_abs": normalized_max_abs,
                "finite": finite,
            }
        )
    return {
        "mapped_parameter_ids": len(parameter_names),
        "state_entries": len(raw_state),
        "unmapped_state_ids": [str(value) for value in unmapped_state_ids],
        "modules": {
            name: _finalize_optimizer_accumulator(accumulator)
            for name, accumulator in accumulators.items()
        },
        "top_exp_avg_sq": sorted(
            parameter_records,
            key=lambda record: record["exp_avg_sq_max"],
            reverse=True,
        )[:top_k],
        "top_exp_avg_l2": sorted(
            parameter_records,
            key=lambda record: record["exp_avg_l2"],
            reverse=True,
        )[:top_k],
        "top_normalized_moment_l2": sorted(
            parameter_records,
            key=lambda record: record["normalized_moment_l2"],
            reverse=True,
        )[:top_k],
    }


def analyze_checkpoint_optimizer_state(
    checkpoint: str,
    model,
    training_config,
) -> dict[str, Any]:
    optimizer_path = os.path.join(checkpoint, "optimizer.pt")
    if not os.path.isfile(optimizer_path):
        return {
            "available": False,
            "path": optimizer_path,
            "reason": "optimizer.pt is not present in this checkpoint",
        }
    optimizer_state = torch.load(
        optimizer_path,
        map_location="cpu",
        weights_only=True,
    )
    parameter_names = _optimizer_state_parameter_names(
        optimizer_state,
        model,
        training_config,
    )
    return {
        "available": True,
        "path": optimizer_path,
        **summarize_optimizer_state(optimizer_state, parameter_names),
    }


def compare_optimizer_state_reports(
    best: Mapping[str, Any],
    last: Mapping[str, Any],
) -> dict[str, Any]:
    if not best.get("available") or not last.get("available"):
        return {
            "available": False,
            "best_available": bool(best.get("available")),
            "last_available": bool(last.get("available")),
        }
    metrics = {}
    modules = sorted(set(best.get("modules", {})) | set(last.get("modules", {})))
    for module_name in modules:
        best_module = best.get("modules", {}).get(module_name, {})
        last_module = last.get("modules", {}).get(module_name, {})
        metrics[module_name] = {}
        for metric_name in (
            "exp_avg_l2",
            "exp_avg_sq_mean",
            "exp_avg_sq_max",
            "max_stored_rms",
            "normalized_moment_l2",
            "normalized_moment_max_abs",
        ):
            best_value = best_module.get(metric_name)
            last_value = last_module.get(metric_name)
            metrics[module_name][metric_name] = {
                "best": best_value,
                "last": last_value,
                "last_to_best_ratio": (
                    float(last_value) / float(best_value)
                    if best_value not in {None, 0.0} and last_value is not None
                    else None
                ),
            }
    second_moment_ratios = [
        module_metrics["exp_avg_sq_max"]["last_to_best_ratio"]
        for module_name, module_metrics in metrics.items()
        if module_name != "all"
        and module_metrics["exp_avg_sq_max"]["last_to_best_ratio"] is not None
    ]
    return {
        "available": True,
        "modules": metrics,
        "flags": {
            "stored_second_moment_growth": bool(
                second_moment_ratios and max(second_moment_ratios) > 10.0
            ),
            "nonfinite_optimizer_state": bool(
                any(
                    int(module.get("nonfinite_tensors", 0)) > 0
                    for report in (best, last)
                    for module in report.get("modules", {}).values()
                )
            ),
        },
    }


def compare_representation_reports(
    best: Mapping[str, Any],
    last: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare diversity and retrieval behavior at every captured boundary."""
    best_representations = best.get("representations", {})
    last_representations = last.get("representations", {})
    boundaries = sorted(set(best_representations) & set(last_representations))
    representation_metrics = {}
    for boundary in boundaries:
        best_boundary = best_representations[boundary]
        last_boundary = last_representations[boundary]
        boundary_metrics = {}
        scalar_values = {
            "norm_median": (
                best_boundary.get("norm", {}).get("median"),
                last_boundary.get("norm", {}).get("median"),
            ),
            **{
                key: (best_boundary.get(key), last_boundary.get(key))
                for key in (
                    "variance_trace",
                    "rms_radius",
                    "centroid_norm",
                    "radius_to_centroid_norm",
                    "mean_pairwise_cosine",
                    "exact_unique_ratio",
                    "normalized_unique_ratio_1e4",
                    "active_variance_dimension_fraction",
                    "effective_rank_centered",
                    "stable_rank_centered",
                )
            },
            **{
                f"identity_{key}": (
                    best_boundary.get("identity_deduplicated", {}).get(key),
                    last_boundary.get("identity_deduplicated", {}).get(key),
                )
                for key in (
                    "variance_trace",
                    "rms_radius",
                    "mean_pairwise_cosine",
                    "effective_rank_centered",
                    "stable_rank_centered",
                )
            },
        }
        for metric_name, (best_value, last_value) in scalar_values.items():
            boundary_metrics[metric_name] = {
                "best": best_value,
                "last": last_value,
                "last_minus_best": (
                    float(last_value) - float(best_value)
                    if best_value is not None and last_value is not None
                    else None
                ),
                "last_to_best_ratio": (
                    float(last_value) / float(best_value)
                    if best_value not in {None, 0.0} and last_value is not None
                    else None
                ),
            }
        representation_metrics[boundary] = boundary_metrics

    best_retrieval = best.get("contrastive_retrieval", {}).get("aggregate", {})
    last_retrieval = last.get("contrastive_retrieval", {}).get("aggregate", {})
    retrieval_metrics = {}
    for direction in ("protein_to_molecule", "molecule_to_protein"):
        retrieval_metrics[direction] = {}
        for metric_name in (
            "strict_top1_accuracy",
            "mean_margin",
            "mean_cosine_margin",
        ):
            best_value = best_retrieval.get(direction, {}).get(metric_name)
            last_value = last_retrieval.get(direction, {}).get(metric_name)
            retrieval_metrics[direction][metric_name] = {
                "best": best_value,
                "last": last_value,
                "last_minus_best": (
                    float(last_value) - float(best_value)
                    if best_value is not None and last_value is not None
                    else None
                ),
                "last_to_best_ratio": (
                    float(last_value) / float(best_value)
                    if best_value not in {None, 0.0} and last_value is not None
                    else None
                ),
            }

    def boundary_ratio(boundary: str, metric: str) -> float | None:
        return (
            representation_metrics.get(boundary, {})
            .get(metric, {})
            .get("last_to_best_ratio")
        )

    def preferred_variance_ratio(boundary: str) -> float | None:
        identity_ratio = boundary_ratio(boundary, "identity_variance_trace")
        return (
            identity_ratio
            if identity_ratio is not None
            else boundary_ratio(boundary, "variance_trace")
        )

    accuracy_deltas = [
        metrics["strict_top1_accuracy"]["last_minus_best"]
        for metrics in retrieval_metrics.values()
        if metrics["strict_top1_accuracy"]["last_minus_best"] is not None
    ]
    molecule_norm_ratio = boundary_ratio("molecule_raw", "norm_median")
    molecule_variance_ratios = [
        preferred_variance_ratio(boundary)
        for boundary in (
            "molecule_encoder_pooled",
            "molecule_linear1",
            "molecule_relu",
            "molecule_raw",
            "molecule_normalized",
        )
    ]
    molecule_variance_ratios = [
        value for value in molecule_variance_ratios if value is not None
    ]
    return {
        "mode": best.get("mode"),
        "representations": representation_metrics,
        "contrastive_retrieval": retrieval_metrics,
        "flags": {
            "molecule_projection_norm_explosion": bool(
                molecule_norm_ratio is not None and molecule_norm_ratio > 3.0
            ),
            "molecule_boundary_variance_collapse": bool(
                molecule_variance_ratios
                and min(molecule_variance_ratios) < 0.1
            ),
            "contrastive_retrieval_accuracy_collapse": bool(
                accuracy_deltas and min(accuracy_deltas) < -0.2
            ),
        },
    }


def compare_stability_reports(
    best: Mapping[str, Any],
    last: Mapping[str, Any],
) -> dict[str, Any]:
    def median(report: Mapping[str, Any], key: str) -> float | None:
        value = report.get("distributions", {}).get(key, {}).get("median")
        return None if value is None else float(value)

    keys = (
        "loss",
        "ranking_loss",
        "contrastive_loss",
        "cosine_std",
        "protein_raw_norm",
        "molecule_raw_norm",
        "total_parameter_grad_norm",
        "protein_encoder_grad_norm",
        "molecule_encoder_grad_norm",
        "protein_projection_grad_norm",
        "molecule_projection_grad_norm",
        "ranking_normalized_embedding_grad_norm",
        "contrastive_normalized_embedding_grad_norm",
        "ranking_raw_projection_grad_norm",
        "contrastive_raw_projection_grad_norm",
    )
    metrics = {}
    for key in keys:
        best_value = median(best, key)
        last_value = median(last, key)
        metrics[key] = {
            "best_median": best_value,
            "last_median": last_value,
            "last_to_best_ratio": (
                last_value / best_value
                if best_value not in {None, 0.0} and last_value is not None
                else None
            ),
        }
    raw_ratios = [
        metrics[key]["last_to_best_ratio"]
        for key in ("protein_raw_norm", "molecule_raw_norm")
        if metrics[key]["last_to_best_ratio"] is not None
    ]
    gradient_ratio = metrics["total_parameter_grad_norm"]["last_to_best_ratio"]
    module_gradient_ratios = [
        metrics[key]["last_to_best_ratio"]
        for key in (
            "protein_encoder_grad_norm",
            "molecule_encoder_grad_norm",
            "protein_projection_grad_norm",
            "molecule_projection_grad_norm",
        )
        if metrics[key]["last_to_best_ratio"] is not None
    ]
    cosine_ratio = metrics["cosine_std"]["last_to_best_ratio"]
    return {
        "mode": best.get("mode"),
        "metrics": metrics,
        "flags": {
            "raw_projection_norm_collapse": bool(
                raw_ratios and min(raw_ratios) < 0.1
            ),
            "raw_projection_norm_explosion": bool(
                raw_ratios and max(raw_ratios) > 3.0
            ),
            "parameter_gradient_explosion": bool(
                gradient_ratio is not None and gradient_ratio > 10.0
            ),
            "parameter_gradient_starvation": bool(
                gradient_ratio is not None and gradient_ratio < 0.1
            ),
            "module_gradient_starvation": bool(
                module_gradient_ratios and min(module_gradient_ratios) < 0.1
            ),
            "score_diversity_collapse": bool(
                cosine_ratio is not None and cosine_ratio < 0.25
            ),
            "score_diversity_contraction": bool(
                cosine_ratio is not None and cosine_ratio < 0.5
            ),
            "nonfinite_values": bool(
                int(best.get("nonfinite_batches", 0)) > 0
                or int(last.get("nonfinite_batches", 0)) > 0
            ),
        },
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
