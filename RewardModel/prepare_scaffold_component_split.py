#!/usr/bin/env python
"""Create an assay-preserving, scaffold-disjoint train/validation/test split.

Runs directly on the unsplit source parquet, the same input the random-assay
splitter takes, so the two split strategies are alternatives rather than stages.

    python prepare_scaffold_component_split.py \
        --source ../dataset/processed/chembl_37/random_assay_source/all.parquet \
        --output-dir ../dataset/processed/chembl_37/scaffold_component_split \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reward_model.training.scaffold_component_split import (  # noqa: E402
    write_scaffold_component_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic scaffold-disjoint train/validation/test split "
            "whose atomic unit is a connected component of the assay-scaffold graph, "
            "keeping every assay intact."
        )
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-proc", type=int, default=8,
                        help="processes used for Bemis-Murcko scaffold generation")
    parser.add_argument("--balance-restarts", type=int, default=64,
                        help="seeded restarts of the validation/test balancing greedy")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_scaffold_component_split(
        args.source,
        args.output_dir,
        seed=args.seed,
        num_proc=args.num_proc,
        balance_restarts=args.balance_restarts,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
