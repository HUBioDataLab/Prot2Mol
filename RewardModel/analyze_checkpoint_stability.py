#!/usr/bin/env python3
"""Compare retained best and last RewardModel checkpoints for collapse signals."""

from __future__ import annotations

import argparse
import json
import os

import torch

from reward_model.analysis.checkpoint_stability import (
    analyze_checkpoint_optimizer_state,
    analyze_model_stability,
    compare_optimizer_state_reports,
    compare_encoder_layer_snapshots,
    compare_parameter_snapshots,
    compare_representation_reports,
    compare_stability_reports,
    json_safe,
    prepare_diagnostic_feature_batches,
    resolve_checkpoint_pair,
    snapshot_parameters,
)
from reward_model.analysis.cli import resolve_device
from reward_model.model import load_reward_model
from reward_model.training import load_reward_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run-dir",
        help="Training output directory containing retained checkpoint-N folders",
    )
    source.add_argument(
        "--best-checkpoint",
        help="Explicit best checkpoint; requires --last-checkpoint",
    )
    parser.add_argument("--last-checkpoint")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--max-assays", type=int, default=96)
    parser.add_argument("--batch-items", type=int, default=12)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--sampling-epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("eval", "train"),
        default=("eval", "train"),
    )
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="checkpoint_stability_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.best_checkpoint is not None and args.last_checkpoint is None:
        raise ValueError("--best-checkpoint requires --last-checkpoint")
    checkpoints = resolve_checkpoint_pair(
        run_dir=args.run_dir,
        best_checkpoint=args.best_checkpoint,
        last_checkpoint=args.last_checkpoint,
    )
    training_config = load_reward_training_config(args.config)
    feature_batches, selection = prepare_diagnostic_feature_batches(
        training_config,
        split=args.split,
        max_assays=args.max_assays,
        batch_items=args.batch_items,
        max_batches=args.max_batches,
        seed=args.seed,
        sampling_epoch=args.sampling_epoch,
    )
    device = resolve_device(args.device)
    checkpoint_reports = {}
    best_parameters = None
    parameter_drift = None
    parameter_layer_drift = None
    for label, checkpoint in (("best", checkpoints.best), ("last", checkpoints.last)):
        model = load_reward_model(checkpoint, device=device)
        mode_reports = {}
        for mode in args.modes:
            mode_reports[mode] = analyze_model_stability(
                model,
                feature_batches,
                device=device,
                precision=args.precision,
                mode=mode,
                seed=args.seed,
                label=label,
            )
        checkpoint_reports[label] = {
            "path": checkpoint,
            "modes": mode_reports,
            "optimizer": analyze_checkpoint_optimizer_state(
                checkpoint,
                model,
                training_config,
            ),
        }
        parameters = snapshot_parameters(model)
        if label == "best":
            best_parameters = parameters
        else:
            if best_parameters is None:
                raise RuntimeError("Best checkpoint parameters were not captured")
            parameter_drift = compare_parameter_snapshots(
                best_parameters,
                parameters,
            )
            parameter_layer_drift = compare_encoder_layer_snapshots(
                best_parameters,
                parameters,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    comparisons = {
        mode: compare_stability_reports(
            checkpoint_reports["best"]["modes"][mode],
            checkpoint_reports["last"]["modes"][mode],
        )
        for mode in args.modes
    }
    representation_comparisons = {
        mode: compare_representation_reports(
            checkpoint_reports["best"]["modes"][mode],
            checkpoint_reports["last"]["modes"][mode],
        )
        for mode in args.modes
    }
    optimizer_comparison = compare_optimizer_state_reports(
        checkpoint_reports["best"]["optimizer"],
        checkpoint_reports["last"]["optimizer"],
    )
    report = json_safe(
        {
            "config": os.path.abspath(args.config),
            "device": str(device),
            "precision": args.precision,
            "checkpoints": {
                "best": checkpoints.best,
                "last": checkpoints.last,
            },
            "selection": selection,
            "checkpoint_reports": checkpoint_reports,
            "comparisons": comparisons,
            "representation_comparisons": representation_comparisons,
            "optimizer_comparison": optimizer_comparison,
            "parameter_drift": parameter_drift,
            "parameter_layer_drift": parameter_layer_drift,
        }
    )
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
    print(
        json.dumps(
            {
                "output": output_path,
                "checkpoints": report["checkpoints"],
                "selection": {
                    key: report["selection"][key]
                    for key in (
                        "split",
                        "selected_assays",
                        "selected_examples",
                        "analyzed_batches",
                    )
                },
                "stability_flags": {
                    mode: comparison["flags"]
                    for mode, comparison in report["comparisons"].items()
                },
                "representation_flags": {
                    mode: comparison["flags"]
                    for mode, comparison in report[
                        "representation_comparisons"
                    ].items()
                },
                "contrastive_retrieval": {
                    mode: comparison["contrastive_retrieval"]
                    for mode, comparison in report[
                        "representation_comparisons"
                    ].items()
                },
                "optimizer_flags": report["optimizer_comparison"].get(
                    "flags", {}
                ),
                "last_optimizer_top_exp_avg_sq": report["checkpoint_reports"]
                ["last"]["optimizer"].get("top_exp_avg_sq", [])[:5],
                "parameter_drift": report["parameter_drift"],
                "largest_molecule_encoder_layer_drift": report[
                    "parameter_layer_drift"
                ]["molecule_encoder"]["largest_absolute_drift"][:5],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
