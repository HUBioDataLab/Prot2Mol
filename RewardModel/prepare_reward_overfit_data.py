#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from reward_model.training.overfit import build_overfit_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select one 50-molecule assay list from tokenized training data and "
            "copy it identically to train, validation, and test."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-molecules", type=int, default=50)
    parser.add_argument(
        "--group-id",
        default=None,
        help="Optional exact target__assay group; otherwise choose deterministically",
    )
    parser.add_argument("--min-pchembl-span", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_overfit_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        num_molecules=args.num_molecules,
        group_id=args.group_id,
        min_pchembl_span=args.min_pchembl_span,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
