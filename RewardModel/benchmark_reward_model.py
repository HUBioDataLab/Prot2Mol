from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from reward_model.model import RewardModel
from reward_model.training import (
    LengthBucketSampler,
    RewardPairCollator,
    RewardPairDataset,
    get_saved_pair_dataset_paths,
    get_tokenized_split_dataset_paths,
    load_reward_training_config,
    load_saved_pair_dataset,
    load_tokenized_example_dataset,
)


@dataclass(frozen=True)
class BenchmarkVariant:
    dynamic_padding: bool
    length_bucketing: bool
    deduplicate_inputs: bool
    fusion_attention_backend: str = "manual"


VARIANTS = {
    "baseline": BenchmarkVariant(False, False, False),
    "dynamic_padding": BenchmarkVariant(True, False, False),
    "dynamic_padding_dedup": BenchmarkVariant(True, False, True),
    "optimized": BenchmarkVariant(True, True, True),
    "optimized_sdpa": BenchmarkVariant(True, True, True, "sdpa"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark RewardModel training microsteps with reproducible legacy and "
            "optimized data paths. The caller must reserve and safety-check the GPU first."
        )
    )
    parser.add_argument(
        "--config",
        default=os.path.join("configs", "reward_train_small.yaml"),
        help="RewardModel training YAML used for model and dataset paths",
    )
    parser.add_argument("--batch-size", type=int, choices=(4, 8), default=4)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=8)
    parser.add_argument("--correctness-pairs", type=int, default=2)
    parser.add_argument(
        "--synthetic-pairs",
        type=int,
        default=0,
        help=(
            "Use this many deterministic in-memory ranking pairs instead of loading "
            "tokenized datasets from the config (0 loads the real dataset)"
        ),
    )
    parser.add_argument("--num-workers", type=int, choices=range(0, 11), default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="fp16",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANTS),
        default=("baseline", "optimized"),
    )
    parser.add_argument(
        "--allow-nonzero-visible-device",
        action="store_true",
        help="Allow use outside the assigned physical GPU 0 (disabled for shared-server runs)",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    if args.measure_steps <= 0:
        raise ValueError("measure_steps must be > 0")
    if args.correctness_pairs <= 0:
        raise ValueError("correctness_pairs must be > 0")
    if args.synthetic_pairs < 0:
        raise ValueError("synthetic_pairs must be >= 0")
    if not args.allow_nonzero_visible_device and os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError(
            "Shared-server safety check failed: CUDA_VISIBLE_DEVICES must be exactly '0'"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Shared-server safety check failed: exactly one CUDA device must be visible"
        )
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The visible GPU/PyTorch runtime does not support BF16")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return float(ordered[max(0, index)])


def _tensor_batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    model_keys = (
        "protein_input_ids",
        "protein_attention_mask",
        "molecule_input_ids",
        "molecule_attention_mask",
        "activity_labels",
        "positive_indices",
        "negative_indices",
    )
    return {
        key: batch[key].to(device=device, non_blocking=True)
        for key in model_keys
    }


def _unique_row_count(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> int:
    keys = torch.cat([input_ids, attention_mask.to(dtype=input_ids.dtype)], dim=1)
    return int(torch.unique(keys, dim=0).size(0))


def _batch_shape_stats(batch: Mapping[str, Any]) -> Dict[str, float]:
    protein_ids = batch["protein_input_ids"]
    protein_mask = batch["protein_attention_mask"]
    molecule_ids = batch["molecule_input_ids"]
    molecule_mask = batch["molecule_attention_mask"]
    examples = int(protein_ids.size(0))
    return {
        "examples": float(examples),
        "protein_width": float(protein_ids.size(1)),
        "molecule_width": float(molecule_ids.size(1)),
        "active_protein_tokens": float(protein_mask.sum().item()),
        "active_molecule_tokens": float(molecule_mask.sum().item()),
        "unique_proteins": float(_unique_row_count(protein_ids, protein_mask)),
        "unique_molecules": float(_unique_row_count(molecule_ids, molecule_mask)),
    }


def _mean_stats(records: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not records:
        return {}
    return {
        key: float(statistics.mean(record[key] for record in records))
        for key in records[0]
    }


def _make_collator(model: RewardModel, *, dynamic_padding: bool) -> RewardPairCollator:
    return RewardPairCollator(
        dynamic_padding=dynamic_padding,
        protein_pad_token_id=getattr(model.protein_tokenizer, "pad_token_id", 0),
        molecule_pad_token_id=getattr(model.molecule_tokenizer, "pad_token_id", 0),
    )


def _make_dataloader(
    dataset: RewardPairDataset,
    model: RewardModel,
    variant: BenchmarkVariant,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader:
    if variant.length_bucketing:
        sampler = LengthBucketSampler(
            dataset.get_pair_sequence_lengths(),
            batch_size=batch_size,
            bucket_size_multiplier=50,
            seed=seed,
        )
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)
        sampler = RandomSampler(dataset, generator=generator)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=_make_collator(model, dynamic_padding=variant.dynamic_padding),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )


def _autocast_settings(precision: str) -> tuple[bool, torch.dtype]:
    if precision == "fp16":
        return True, torch.float16
    if precision == "bf16":
        return True, torch.bfloat16
    return False, torch.float32


def _all_gradients_finite(model: torch.nn.Module) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
    )


def _gradient_norm(model: torch.nn.Module) -> float:
    squared_norm = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            norm = parameter.grad.detach().float().norm().item()
            squared_norm += norm * norm
    return float(math.sqrt(squared_norm))


def _run_variant(
    *,
    name: str,
    variant: BenchmarkVariant,
    model: RewardModel,
    dataset: RewardPairDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    _seed_everything(args.seed)
    model.config.deduplicate_protein_inputs = variant.deduplicate_inputs
    model.config.deduplicate_molecule_inputs = variant.deduplicate_inputs
    model.config.fusion_attention_backend = variant.fusion_attention_backend
    model.fusion.attention_backend = variant.fusion_attention_backend
    model.train()

    loader_setup_start = time.perf_counter()
    dataloader = _make_dataloader(
        dataset,
        model,
        variant,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    loader_setup_seconds = time.perf_counter() - loader_setup_start
    iterator: Iterable[Mapping[str, Any]] = iter(dataloader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.0)
    amp_enabled, amp_dtype = _autocast_settings(args.precision)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    step_times_ms: list[float] = []
    data_wait_times_ms: list[float] = []
    losses: list[float] = []
    shape_records: list[Dict[str, float]] = []
    total_steps = args.warmup_steps + args.measure_steps

    for step_index in range(total_steps):
        data_start = time.perf_counter()
        try:
            cpu_batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            cpu_batch = next(iterator)
        data_wait_ms = (time.perf_counter() - data_start) * 1000.0

        torch.cuda.synchronize(device)
        step_start = time.perf_counter()
        batch = _tensor_batch_to_device(cpu_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            outputs = model(return_dict=True, **batch)
        if outputs.loss is None:
            raise RuntimeError("RewardModel did not produce a training loss")
        scaler.scale(outputs.loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(device)
        step_ms = (time.perf_counter() - step_start) * 1000.0

        if step_index == args.warmup_steps - 1:
            torch.cuda.reset_peak_memory_stats(device)
        if step_index >= args.warmup_steps:
            step_times_ms.append(step_ms)
            data_wait_times_ms.append(data_wait_ms)
            losses.append(float(outputs.loss.detach().float().item()))
            shape_records.append(_batch_shape_stats(cpu_batch))

    total_step_seconds = sum(step_times_ms) / 1000.0
    total_pipeline_seconds = (
        sum(step_times_ms) + sum(data_wait_times_ms)
    ) / 1000.0
    measured_pairs = args.batch_size * args.measure_steps
    result = {
        "variant": name,
        "settings": {
            "dynamic_padding": variant.dynamic_padding,
            "length_bucketing": variant.length_bucketing,
            "deduplicate_inputs": variant.deduplicate_inputs,
            "fusion_attention_backend": variant.fusion_attention_backend,
        },
        "loader_setup_seconds": loader_setup_seconds,
        "step_ms_mean": float(statistics.mean(step_times_ms)),
        "step_ms_median": float(statistics.median(step_times_ms)),
        "step_ms_p95": _percentile(step_times_ms, 0.95),
        "data_wait_ms_mean": float(statistics.mean(data_wait_times_ms)),
        "pairs_per_second_compute": measured_pairs / total_step_seconds,
        "pairs_per_second_pipeline": measured_pairs / total_pipeline_seconds,
        "examples_per_second_pipeline": (2 * measured_pairs) / total_pipeline_seconds,
        "peak_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "peak_memory_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
        "loss_mean": float(statistics.mean(losses)),
        "loss_min": float(min(losses)),
        "loss_max": float(max(losses)),
        "losses_finite": all(math.isfinite(loss) for loss in losses),
        "gradients_finite": _all_gradients_finite(model),
        "gradient_norm": _gradient_norm(model),
        "mean_batch": _mean_stats(shape_records),
    }
    print(json.dumps({"completed_variant": name, "result": result}), flush=True)
    del optimizer, scaler, dataloader, iterator, batch, cpu_batch, outputs
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return result


def _gradient_signature(model: torch.nn.Module) -> Dict[str, tuple[float, float]]:
    signature: Dict[str, tuple[float, float]] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            gradient = parameter.grad.detach().float()
            signature[name] = (
                float(gradient.norm().item()),
                float(gradient.sum().item()),
            )
    return signature


def _correctness_pass(
    model: RewardModel,
    batch: Mapping[str, torch.Tensor],
    *,
    deduplicate: bool,
    fusion_attention_backend: str,
) -> Dict[str, Any]:
    model.config.deduplicate_protein_inputs = deduplicate
    model.config.deduplicate_molecule_inputs = deduplicate
    model.config.fusion_attention_backend = fusion_attention_backend
    model.fusion.attention_backend = fusion_attention_backend
    model.zero_grad(set_to_none=True)
    outputs = model(return_dict=True, **batch)
    if outputs.loss is None:
        raise RuntimeError("RewardModel did not produce a correctness loss")
    outputs.loss.backward()
    result = {
        "ranking_score": outputs.ranking_score.detach().float().cpu(),
        "activity_logits": outputs.activity_logits.detach().float().cpu(),
        "loss": outputs.loss.detach().float().cpu(),
        "gradient_signature": _gradient_signature(model),
        "gradients_finite": _all_gradients_finite(model),
    }
    model.zero_grad(set_to_none=True)
    return result


def _max_gradient_signature_errors(
    baseline: Mapping[str, tuple[float, float]],
    optimized: Mapping[str, tuple[float, float]],
) -> Dict[str, float]:
    if baseline.keys() != optimized.keys():
        missing = sorted(baseline.keys() ^ optimized.keys())
        raise RuntimeError(f"Gradient parameter sets differ: {missing}")
    max_norm_absolute = 0.0
    max_norm_relative = 0.0
    max_sum_absolute = 0.0
    for name in baseline:
        baseline_norm, baseline_sum = baseline[name]
        optimized_norm, optimized_sum = optimized[name]
        norm_absolute = abs(baseline_norm - optimized_norm)
        norm_relative = norm_absolute / max(abs(baseline_norm), abs(optimized_norm), 1e-12)
        max_norm_absolute = max(max_norm_absolute, norm_absolute)
        max_norm_relative = max(max_norm_relative, norm_relative)
        max_sum_absolute = max(max_sum_absolute, abs(baseline_sum - optimized_sum))
    return {
        "gradient_norm_max_absolute_error": max_norm_absolute,
        "gradient_norm_max_relative_error": max_norm_relative,
        "gradient_sum_max_absolute_error": max_sum_absolute,
    }


def _gradient_norms_close(
    baseline: Mapping[str, tuple[float, float]],
    optimized: Mapping[str, tuple[float, float]],
) -> bool:
    if baseline.keys() != optimized.keys():
        return False
    return all(
        math.isclose(
            baseline[name][0],
            optimized[name][0],
            rel_tol=1e-4,
            abs_tol=1e-6,
        )
        for name in baseline
    )


def _compare_correctness_passes(
    baseline: Mapping[str, Any],
    optimized: Mapping[str, Any],
) -> Dict[str, Any]:
    score_error = float(
        (baseline["ranking_score"] - optimized["ranking_score"]).abs().max().item()
    )
    logit_error = float(
        (baseline["activity_logits"] - optimized["activity_logits"]).abs().max().item()
    )
    loss_error = float((baseline["loss"] - optimized["loss"]).abs().item())
    gradient_errors = _max_gradient_signature_errors(
        baseline["gradient_signature"],
        optimized["gradient_signature"],
    )
    forward_close = (
        torch.allclose(
            baseline["ranking_score"],
            optimized["ranking_score"],
            atol=1e-5,
            rtol=1e-4,
        )
        and torch.allclose(
            baseline["activity_logits"],
            optimized["activity_logits"],
            atol=1e-5,
            rtol=1e-4,
        )
        and torch.allclose(
            baseline["loss"],
            optimized["loss"],
            atol=1e-5,
            rtol=1e-4,
        )
    )
    gradients_close = _gradient_norms_close(
        baseline["gradient_signature"],
        optimized["gradient_signature"],
    )
    return {
        "passed": bool(
            forward_close
            and gradients_close
            and baseline["gradients_finite"]
            and optimized["gradients_finite"]
        ),
        "forward_close": bool(forward_close),
        "gradients_close": bool(gradients_close),
        "baseline_gradients_finite": baseline["gradients_finite"],
        "optimized_gradients_finite": optimized["gradients_finite"],
        "ranking_score_max_absolute_error": score_error,
        "activity_logit_max_absolute_error": logit_error,
        "loss_absolute_error": loss_error,
        **gradient_errors,
    }


def _run_correctness_check(
    *,
    model: RewardModel,
    dataset: RewardPairDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    if args.correctness_pairs > len(dataset):
        raise ValueError("correctness_pairs exceeds the available pair dataset")
    features = [dataset[index] for index in range(args.correctness_pairs)]
    fixed_cpu = _make_collator(model, dynamic_padding=False)(features)
    dynamic_cpu = _make_collator(model, dynamic_padding=True)(features)
    fixed_batch = _tensor_batch_to_device(fixed_cpu, device)
    dynamic_batch = _tensor_batch_to_device(dynamic_cpu, device)

    model.eval()
    _seed_everything(args.seed)
    baseline = _correctness_pass(
        model,
        fixed_batch,
        deduplicate=False,
        fusion_attention_backend="manual",
    )
    _seed_everything(args.seed)
    optimized = _correctness_pass(
        model,
        dynamic_batch,
        deduplicate=True,
        fusion_attention_backend="manual",
    )
    result = {
        **_compare_correctness_passes(baseline, optimized),
        "fixed_shapes": {
            "protein": list(fixed_cpu["protein_input_ids"].shape),
            "molecule": list(fixed_cpu["molecule_input_ids"].shape),
        },
        "dynamic_shapes": {
            "protein": list(dynamic_cpu["protein_input_ids"].shape),
            "molecule": list(dynamic_cpu["molecule_input_ids"].shape),
        },
    }
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return result


def _run_fusion_correctness_check(
    *,
    model: RewardModel,
    dataset: RewardPairDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    features = [dataset[index] for index in range(args.correctness_pairs)]
    dynamic_cpu = _make_collator(model, dynamic_padding=True)(features)
    dynamic_batch = _tensor_batch_to_device(dynamic_cpu, device)

    model.eval()
    _seed_everything(args.seed)
    manual = _correctness_pass(
        model,
        dynamic_batch,
        deduplicate=True,
        fusion_attention_backend="manual",
    )
    _seed_everything(args.seed)
    sdpa = _correctness_pass(
        model,
        dynamic_batch,
        deduplicate=True,
        fusion_attention_backend="sdpa",
    )
    result = {
        **_compare_correctness_passes(manual, sdpa),
        "shapes": {
            "protein": list(dynamic_cpu["protein_input_ids"].shape),
            "molecule": list(dynamic_cpu["molecule_input_ids"].shape),
        },
    }
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return result


def _random_token_row(
    *,
    active_length: int,
    max_length: int,
    vocab_size: int,
    pad_token_id: int,
    generator: torch.Generator,
) -> tuple[list[int], list[int]]:
    if active_length <= 0 or active_length > max_length:
        raise ValueError("active_length must be in [1, max_length]")
    if vocab_size <= 1:
        raise ValueError("vocab_size must be > 1")
    active = torch.randint(
        low=0,
        high=vocab_size,
        size=(active_length,),
        generator=generator,
        dtype=torch.long,
    )
    replacement_id = (pad_token_id + 1) % vocab_size
    active = torch.where(active == pad_token_id, replacement_id, active)
    padding_length = max_length - active_length
    return (
        active.tolist() + [pad_token_id] * padding_length,
        [1] * active_length + [0] * padding_length,
    )


def _build_synthetic_pair_dataset(
    model: RewardModel,
    *,
    num_pairs: int,
    seed: int,
) -> RewardPairDataset:
    from datasets import Dataset

    if num_pairs <= 0:
        raise ValueError("num_pairs must be > 0")
    protein_max_length = int(model.config.protein_max_length)
    molecule_max_length = int(model.config.molecule_max_length)
    protein_lengths = sorted(
        {
            max(4, protein_max_length // 4),
            max(4, protein_max_length // 2),
            max(4, (3 * protein_max_length) // 4),
            protein_max_length,
        }
    )
    molecule_lengths = sorted(
        {
            max(4, molecule_max_length // 4),
            max(4, molecule_max_length // 2),
            max(4, (3 * molecule_max_length) // 4),
            molecule_max_length,
        }
    )
    protein_vocab_size = int(model.protein_encoder.config.vocab_size)
    molecule_vocab_size = int(model.molecule_encoder.config.vocab_size)
    protein_pad_token_id = int(getattr(model.protein_tokenizer, "pad_token_id", 0) or 0)
    molecule_pad_token_id = int(getattr(model.molecule_tokenizer, "pad_token_id", 0) or 0)
    generator = torch.Generator()
    generator.manual_seed(seed)

    molecule_library_size = max(8, min(32, num_pairs))
    molecule_library: list[tuple[list[int], list[int], int]] = []
    for molecule_index in range(molecule_library_size):
        molecule_length = molecule_lengths[molecule_index % len(molecule_lengths)]
        molecule_ids, molecule_mask = _random_token_row(
            active_length=molecule_length,
            max_length=molecule_max_length,
            vocab_size=molecule_vocab_size,
            pad_token_id=molecule_pad_token_id,
            generator=generator,
        )
        molecule_library.append((molecule_ids, molecule_mask, molecule_length))

    example_columns: Dict[str, list[Any]] = {
        "example_id": [],
        "group_id": [],
        "target_chembl_id": [],
        "assay_id": [],
        "compound_id": [],
        "pchembl_value": [],
        "binary_label": [],
        "protein_input_ids": [],
        "protein_attention_mask": [],
        "protein_length": [],
        "molecule_input_ids": [],
        "molecule_attention_mask": [],
        "molecule_length": [],
    }
    pair_columns: Dict[str, list[Any]] = {
        "positive_example_id": [],
        "negative_example_id": [],
        "group_id": [],
        "positive_pchembl": [],
        "negative_pchembl": [],
    }

    for pair_index in range(num_pairs):
        protein_length = protein_lengths[pair_index % len(protein_lengths)]
        protein_ids, protein_mask = _random_token_row(
            active_length=protein_length,
            max_length=protein_max_length,
            vocab_size=protein_vocab_size,
            pad_token_id=protein_pad_token_id,
            generator=generator,
        )
        group_id = f"SYNTH_TARGET_{pair_index}__SYNTH_ASSAY_{pair_index}"
        positive_id = pair_index * 2
        negative_id = positive_id + 1
        pair_columns["positive_example_id"].append(positive_id)
        pair_columns["negative_example_id"].append(negative_id)
        pair_columns["group_id"].append(group_id)
        pair_columns["positive_pchembl"].append(8.0)
        pair_columns["negative_pchembl"].append(5.0)

        for polarity, example_id, pchembl, label, library_offset in (
            ("positive", positive_id, 8.0, 1, 0),
            ("negative", negative_id, 5.0, 0, 1),
        ):
            molecule_index = (pair_index * 2 + library_offset) % molecule_library_size
            molecule_ids, molecule_mask, molecule_length = molecule_library[molecule_index]
            example_columns["example_id"].append(example_id)
            example_columns["group_id"].append(group_id)
            example_columns["target_chembl_id"].append(f"SYNTH_TARGET_{pair_index}")
            example_columns["assay_id"].append(f"SYNTH_ASSAY_{pair_index}")
            example_columns["compound_id"].append(
                f"SYNTH_COMPOUND_{molecule_index}_{polarity}"
            )
            example_columns["pchembl_value"].append(pchembl)
            example_columns["binary_label"].append(label)
            example_columns["protein_input_ids"].append(list(protein_ids))
            example_columns["protein_attention_mask"].append(list(protein_mask))
            example_columns["protein_length"].append(protein_length)
            example_columns["molecule_input_ids"].append(list(molecule_ids))
            example_columns["molecule_attention_mask"].append(list(molecule_mask))
            example_columns["molecule_length"].append(molecule_length)

    return RewardPairDataset(
        Dataset.from_dict(example_columns),
        Dataset.from_dict(pair_columns),
    )


def _load_dataset(config):
    example_paths = get_tokenized_split_dataset_paths(config.data.tokenized_dataset_dir)
    pair_paths = get_saved_pair_dataset_paths(config.data.tokenized_dataset_dir)
    train_examples = load_tokenized_example_dataset(example_paths["train"])
    train_pairs = load_saved_pair_dataset(pair_paths["train"])
    return RewardPairDataset(train_examples, train_pairs)


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    _seed_everything(args.seed)
    device = torch.device("cuda", 0)
    config = load_reward_training_config(args.config)
    model = RewardModel(config.model).to(device)
    dataset = (
        _build_synthetic_pair_dataset(
            model,
            num_pairs=args.synthetic_pairs,
            seed=args.seed,
        )
        if args.synthetic_pairs
        else _load_dataset(config)
    )
    if len(dataset) < args.batch_size:
        raise RuntimeError(
            f"Pair dataset has {len(dataset)} rows, fewer than batch size {args.batch_size}"
        )

    report: Dict[str, Any] = {
        "benchmark": {
            "config": os.path.abspath(args.config),
            "seed": args.seed,
            "batch_size_pairs": args.batch_size,
            "effective_examples_per_batch": args.batch_size * 2,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "num_workers": args.num_workers,
            "precision": args.precision,
            "dataset_pairs": len(dataset),
            "dataset_examples": len(dataset.example_dataset),
            "synthetic_dataset": bool(args.synthetic_pairs),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
        }
    }
    report["correctness"] = _run_correctness_check(
        model=model,
        dataset=dataset,
        args=args,
        device=device,
    )
    print(json.dumps({"correctness": report["correctness"]}), flush=True)
    if not report["correctness"]["passed"]:
        raise RuntimeError("Baseline-versus-optimized correctness check failed")
    report["fusion_correctness"] = _run_fusion_correctness_check(
        model=model,
        dataset=dataset,
        args=args,
        device=device,
    )
    print(json.dumps({"fusion_correctness": report["fusion_correctness"]}), flush=True)
    if not report["fusion_correctness"]["passed"]:
        raise RuntimeError("Manual-versus-SDPA fusion correctness check failed")

    variants: Dict[str, Any] = {}
    for name in args.variants:
        variants[name] = _run_variant(
            name=name,
            variant=VARIANTS[name],
            model=model,
            dataset=dataset,
            args=args,
            device=device,
        )
    report["variants"] = variants
    if "baseline" in variants:
        baseline_step = variants["baseline"]["step_ms_mean"]
        baseline_memory = variants["baseline"]["peak_memory_allocated_gib"]
        report["relative_to_baseline"] = {
            name: {
                "speedup": baseline_step / result["step_ms_mean"],
                "step_time_reduction_percent": 100.0
                * (baseline_step - result["step_ms_mean"])
                / baseline_step,
                "peak_memory_reduction_gib": baseline_memory
                - result["peak_memory_allocated_gib"],
                "peak_memory_reduction_percent": 100.0
                * (baseline_memory - result["peak_memory_allocated_gib"])
                / baseline_memory,
            }
            for name, result in variants.items()
        }
    if "optimized" in variants and "optimized_sdpa" in variants:
        manual_result = variants["optimized"]
        sdpa_result = variants["optimized_sdpa"]
        report["sdpa_relative_to_manual"] = {
            "speedup": manual_result["step_ms_mean"] / sdpa_result["step_ms_mean"],
            "step_time_reduction_percent": 100.0
            * (manual_result["step_ms_mean"] - sdpa_result["step_ms_mean"])
            / manual_result["step_ms_mean"],
            "peak_memory_reduction_gib": manual_result["peak_memory_allocated_gib"]
            - sdpa_result["peak_memory_allocated_gib"],
            "peak_memory_reduction_percent": 100.0
            * (
                manual_result["peak_memory_allocated_gib"]
                - sdpa_result["peak_memory_allocated_gib"]
            )
            / manual_result["peak_memory_allocated_gib"],
        }
    print("BENCHMARK_REPORT=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
