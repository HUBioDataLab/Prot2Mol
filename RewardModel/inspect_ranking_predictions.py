#!/usr/bin/env python3
"""Dump and summarize per-molecule ranking-head predictions by assay."""

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
    analyze_predictions,
    score_dataset_pairs,
    write_prediction_analysis,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/reward_train.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-train-assays",
        type=int,
        default=100,
        help="Use 0 for every eligible train assay",
    )
    parser.add_argument(
        "--max-val-assays",
        type=int,
        default=0,
        help="Use 0 for every eligible validation assay",
    )
    parser.add_argument(
        "--max-test-assays",
        type=int,
        default=0,
        help="Use 0 for every eligible test assay",
    )
    args = parser.parse_args()
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
    all_summaries = {}
    for split in args.splits:
        configured_max = getattr(args, f"max_{split}_assays")
        max_assays = configured_max or None
        manifest_path = os.path.join(args.output_dir, f"{split}_assays.json")
        dataset, _ = load_selected_split(
            training_config=config,
            split=split,
            max_assays=max_assays,
            seed=args.seed,
            output_manifest_path=manifest_path,
        )
        scored = score_dataset_pairs(
            model,
            dataset,
            batch_size=args.batch_size,
            device=device,
        )
        predictions, assay_summary, split_summary = analyze_predictions(
            scored,
            split=split,
            affinity_margin=model.config.ranking_affinity_margin,
            temperature=model.config.ranking_temperature,
        )
        paths = write_prediction_analysis(
            args.output_dir,
            predictions=predictions,
            assay_summary=assay_summary,
            split_summary=split_summary,
        )
        all_summaries[split] = {"metrics": split_summary, "artifacts": paths}
        print(json.dumps(all_summaries[split], indent=2, sort_keys=True))

    with open(
        os.path.join(args.output_dir, "prediction_analysis_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(all_summaries, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
