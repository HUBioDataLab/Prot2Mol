#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from reward_model.training import write_seen_target_assay_holdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic seen-target validation split by holding out "
            "whole assays while retaining each exact target sequence in train."
        )
    )
    parser.add_argument("--source-train", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_seen_target_assay_holdout(
        args.source_train,
        args.output_dir,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
