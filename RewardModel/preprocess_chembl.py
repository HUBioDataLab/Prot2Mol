#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

from reward_model.data_processing import ChemblPreprocessConfig, preprocess_chembl_sqlite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess ChEMBL into assay-based reward-model artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sqlite-path", type=str, default=None, help="Path to an extracted ChEMBL SQLite file")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for the Phase 3 artifacts")
    parser.add_argument(
        "--download-url",
        type=str,
        default=None,
        help="Optional ChEMBL SQLite archive URL. Used when --sqlite-path is not provided.",
    )
    parser.add_argument("--chembl-release", type=str, default=None, help="Optional ChEMBL release label")
    parser.add_argument("--activity-threshold", type=float, default=6.0, help="Binary activity threshold on pChEMBL")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite downloaded/existing derived artifacts")
    parser.add_argument("--no-parquet", action="store_true", help="Skip optional parquet export")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ChemblPreprocessConfig(
        sqlite_path=args.sqlite_path,
        output_dir=os.path.abspath(args.output_dir),
        download_url=args.download_url,
        chembl_release=args.chembl_release,
        activity_threshold=args.activity_threshold,
        overwrite=args.overwrite,
        write_parquet=not args.no_parquet,
    )

    artifacts = preprocess_chembl_sqlite(config=config)

    summary = {
        "sqlite_path": artifacts.sqlite_path,
        "curated_csv_path": artifacts.curated_csv_path,
        "curated_parquet_path": artifacts.curated_parquet_path,
        "provenance_path": artifacts.provenance_path,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
