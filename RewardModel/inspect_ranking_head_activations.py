#!/usr/bin/env python3
"""Check ranking/classification MLP activations for collapse or non-finite values."""

from __future__ import annotations

import argparse
import json
import os

from reward_model.analysis.cli import (
    load_model_and_training_config,
    load_selected_split,
    resolve_device,
)
from reward_model.analysis.ranking_head import capture_head_activations


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/reward_train.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--assay-manifest")
    parser.add_argument("--max-assays", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_assays < 0:
        parser.error("--max-assays must be >= 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    return args


def main() -> None:
    args = _parse_args()
    device = resolve_device(args.device)
    model, config = load_model_and_training_config(
        checkpoint=args.checkpoint,
        config_path=args.config,
        device=device,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, f"{args.split}_assays.json")
    dataset, _ = load_selected_split(
        training_config=config,
        split=args.split,
        max_assays=args.max_assays or None,
        seed=args.seed,
        manifest_path=args.assay_manifest,
        output_manifest_path=manifest_path,
    )
    predictions, activations = capture_head_activations(
        model,
        dataset,
        batch_size=args.batch_size,
        device=device,
    )
    predictions_path = os.path.abspath(
        os.path.join(args.output_dir, f"{args.split}_activation_predictions.parquet")
    )
    activations_path = os.path.abspath(
        os.path.join(args.output_dir, f"{args.split}_head_activations.csv")
    )
    predictions.to_parquet(predictions_path, index=False)
    activations.to_csv(activations_path, index=False)
    summary = {
        "split": args.split,
        "num_examples": len(predictions),
        "num_modules": len(activations),
        "total_nonfinite_activations": int(activations["nonfinite_count"].sum()),
        "predictions": predictions_path,
        "activations": activations_path,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
