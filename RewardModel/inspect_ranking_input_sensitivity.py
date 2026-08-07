#!/usr/bin/env python3
"""Inspect whether ranking predictions depend on proteins, ligands, or both."""

from __future__ import annotations

import argparse
import json
import os

from reward_model.analysis.cli import (
    load_model_and_training_config,
    load_selected_split,
    resolve_device,
)
from reward_model.analysis.ranking_head import (
    run_input_sensitivity,
    write_sensitivity_analysis,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/reward_train.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--assay-manifest")
    parser.add_argument("--max-assays", type=int, default=50)
    parser.add_argument("--num-shuffles", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_assays < 0:
        parser.error("--max-assays must be >= 0")
    if args.num_shuffles <= 0:
        parser.error("--num-shuffles must be > 0")
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
    output_manifest = os.path.join(args.output_dir, f"{args.split}_assays.json")
    dataset, _ = load_selected_split(
        training_config=config,
        split=args.split,
        max_assays=args.max_assays or None,
        seed=args.seed,
        manifest_path=args.assay_manifest,
        output_manifest_path=output_manifest,
    )
    rows, by_repeat, summary = run_input_sensitivity(
        model,
        dataset,
        split=args.split,
        batch_size=args.batch_size,
        device=device,
        num_shuffles=args.num_shuffles,
        seed=args.seed,
    )
    paths = write_sensitivity_analysis(
        args.output_dir,
        split=args.split,
        sensitivity_rows=rows,
        by_repeat=by_repeat,
        summary=summary,
    )
    print(json.dumps({"metrics": summary, "artifacts": paths}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
