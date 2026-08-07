#!/usr/bin/env python3
"""Build an activity-balanced split from existing ChEMBL MMseqs50 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from data_processing.build_chembl_binding_dataset import (
        DEFAULT_ACTIVITY_BALANCED_OUTPUT_DIR,
        DEFAULT_ACTIVITY_BALANCE_ANCHOR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_REWARD_PROTEIN_MAX_RESIDUES,
        build_activity_balanced_resplit,
    )
    from data_processing.build_protein_cluster_splits import SplitConfig
except ModuleNotFoundError:  # Direct execution from data_processing/
    from build_chembl_binding_dataset import (
        DEFAULT_ACTIVITY_BALANCED_OUTPUT_DIR,
        DEFAULT_ACTIVITY_BALANCE_ANCHOR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_REWARD_PROTEIN_MAX_RESIDUES,
        build_activity_balanced_resplit,
    )
    from build_protein_cluster_splits import SplitConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse an existing ChEMBL MMseqs50 cluster map and create a new "
            "activity-balanced train/validation/test split."
        )
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ACTIVITY_BALANCED_OUTPUT_DIR,
    )
    parser.add_argument(
        "--anchor-activity-type",
        default=DEFAULT_ACTIVITY_BALANCE_ANCHOR,
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--reward-protein-max-residues",
        type=int,
        default=DEFAULT_REWARD_PROTEIN_MAX_RESIDUES,
        help=(
            "Co-balance activity rows that survive the reward model protein "
            "token limit (1022 residues plus two ESM special tokens by default)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SplitConfig(
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
        min_seq_id=0.5,
        coverage=0.01,
        cov_mode=0,
        cluster_mode="mmseqs",
    )
    summary = build_activity_balanced_resplit(
        args.source_dir,
        args.output_dir,
        config,
        anchor_activity_type=args.anchor_activity_type,
        reward_protein_max_residues=args.reward_protein_max_residues,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
