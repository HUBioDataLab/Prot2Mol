#!/usr/bin/env python
"""Write a copy of a split dataset with pair-median harmonised activity labels.

Runs AFTER the split, so every assay stays in the split it was already assigned
to. Only ``binary_label`` changes; ``pchembl_value`` is preserved exactly.

    python prepare_harmonized_labels.py \
        --source-dir ../dataset/processed/chembl_37/target_aware_random_assay_activity_balanced \
        --output-dir ../dataset/processed/chembl_37/target_aware_random_assay_activity_balanced_median
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reward_model.training.label_harmonization import (  # noqa: E402
    DEFAULT_ACTIVITY_THRESHOLD,
    DEFAULT_PAIR_KEY,
    harmonize_labels_by_pair_median,
)

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", required=True,
                        help="directory holding train.parquet / val.parquet / test.parquet")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--activity-threshold", type=float, default=DEFAULT_ACTIVITY_THRESHOLD)
    parser.add_argument("--pair-key", nargs="+", default=list(DEFAULT_PAIR_KEY))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = os.path.abspath(args.source_dir)
    destination = os.path.abspath(args.output_dir)
    if os.path.abspath(source) == destination:
        raise SystemExit("--output-dir must differ from --source-dir; the source is never modified")

    frames = {}
    for split in SPLITS:
        path = os.path.join(source, f"{split}.parquet")
        if not os.path.isfile(path):
            raise SystemExit(f"missing {path}")
        frames[split] = pd.read_parquet(path)
        print(f"  read {split:5s} {len(frames[split]):>9,} rows", flush=True)

    harmonized, summary = harmonize_labels_by_pair_median(
        frames,
        activity_threshold=args.activity_threshold,
        pair_key=tuple(args.pair_key),
    )

    os.makedirs(destination, exist_ok=True)
    written = []
    for split, frame in harmonized.items():
        path = os.path.join(destination, f"{split}.parquet")
        if os.path.exists(path) and not args.overwrite:
            raise SystemExit(f"{path} exists; pass --overwrite to replace")
        temporary = path + ".tmp"
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
        written.append(path)

    summary["source_dir"] = source
    summary["output_dir"] = destination
    manifest = os.path.join(destination, "label_harmonization.json")
    with open(manifest, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print()
    print(f"  conflicting pairs resolved : {summary['pairs_conflicting']:,}")
    print(f"  rows relabelled            : {summary['rows_relabelled']:,}"
          f"  (1->0 {summary['rows_positive_to_negative']:,},"
          f"  0->1 {summary['rows_negative_to_positive']:,})")
    for split, info in summary["splits"].items():
        print(f"  {split:5s} {info['rows']:>9,} rows  relabelled {info['rows_relabelled']:>7,}"
              f"  pos {info['positive_rate_before']:.3f} -> {info['positive_rate_after']:.3f}")
    print(f"  contradictions remaining   : {summary['pairs_with_contradictions_after']}")
    print(f"\n  wrote {destination}")


if __name__ == "__main__":
    main()
