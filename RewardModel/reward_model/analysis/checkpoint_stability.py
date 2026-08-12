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
            captures[f"{name}_raw"] = output

        hooks.append(projection.register_forward_hook(capture_projection))
        activation = getattr(projection, "activation", None)
        if activation is not None:

            def capture_activation(_module, _inputs, output, *, name=modality):
                captures[f"{name}_relu"] = output

            hooks.append(activation.register_forward_hook(capture_activation))
    return hooks


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
        result[group_name] = {
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
    return result


def _optimizer_parameter_groups(
    model,
    training_config,
) -> list[list[str]]:
    """Reproduce the exact parameter grouping order used by RewardModelTrainer."""
    training = training_config.training
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
        and training.projection_learning_rate is None
    ):
        return [
            [name for name, _ in trainable if name in decay_parameters],
            [name for name, _ in trainable if name not in decay_parameters],
        ]

    grouped: dict[tuple[float, float], list[str]] = {}
    for name, _ in trainable:
        if name.startswith(("protein_encoder.", "molecule_encoder.")):
            learning_rate = float(
                training.encoder_learning_rate or training.learning_rate
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
    cosine_ratio = metrics["cosine_std"]["last_to_best_ratio"]
    return {
        "mode": best.get("mode"),
        "metrics": metrics,
        "flags": {
            "raw_projection_norm_collapse": bool(
                raw_ratios and min(raw_ratios) < 0.1
            ),
            "parameter_gradient_explosion": bool(
                gradient_ratio is not None and gradient_ratio > 10.0
            ),
            "score_diversity_collapse": bool(
                cosine_ratio is not None and cosine_ratio < 0.25
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
