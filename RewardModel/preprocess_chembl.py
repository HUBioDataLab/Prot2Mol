#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

from reward_model.data_processing import ChemblPreprocessConfig, preprocess_chembl_sqlite
from reward_model.model import load_tokenizer


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
    parser.add_argument("--split-seed", type=int, default=42, help="Deterministic seed for assay splits")
    parser.add_argument("--protein-max-length", type=int, default=1024, help="Protein tokenizer max length")
    parser.add_argument("--molecule-max-length", type=int, default=512, help="Molecule tokenizer max length")
    parser.add_argument(
        "--tokenization-batch-size",
        type=int,
        default=256,
        help="Batch size for tokenizer calls during row export",
    )
    parser.add_argument(
        "--protein-model-name-or-path",
        type=str,
        default="facebook/esm2_t12_35M_UR50D",
        help="Protein tokenizer/model name or local path",
    )
    parser.add_argument(
        "--molecule-model-name-or-path",
        type=str,
        default="HUBioDataLab/SELFormer",
        help="Molecule tokenizer/model name or local path",
    )
    parser.add_argument(
        "--protein-tokenizer-name-or-path",
        type=str,
        default=None,
        help="Optional tokenizer override for the protein encoder",
    )
    parser.add_argument(
        "--molecule-tokenizer-name-or-path",
        type=str,
        default=None,
        help="Optional tokenizer override for the molecule encoder",
    )
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
        protein_model_name_or_path=args.protein_model_name_or_path,
        molecule_model_name_or_path=args.molecule_model_name_or_path,
        protein_tokenizer_name_or_path=args.protein_tokenizer_name_or_path,
        molecule_tokenizer_name_or_path=args.molecule_tokenizer_name_or_path,
        protein_max_length=args.protein_max_length,
        molecule_max_length=args.molecule_max_length,
        tokenization_batch_size=args.tokenization_batch_size,
        activity_threshold=args.activity_threshold,
        split_seed=args.split_seed,
        overwrite=args.overwrite,
        write_parquet=not args.no_parquet,
    )

    protein_tokenizer = load_tokenizer(
        config.protein_tokenizer_name_or_path or config.protein_model_name_or_path,
    )
    molecule_tokenizer = load_tokenizer(
        config.molecule_tokenizer_name_or_path or config.molecule_model_name_or_path,
    )

    artifacts = preprocess_chembl_sqlite(
        config=config,
        protein_tokenizer=protein_tokenizer,
        molecule_tokenizer=molecule_tokenizer,
    )

    summary = {
        "sqlite_path": artifacts.sqlite_path,
        "curated_csv_path": artifacts.curated_csv_path,
        "curated_parquet_path": artifacts.curated_parquet_path,
        "tokenized_jsonl_path": artifacts.tokenized_jsonl_path,
        "tokenized_dataset_path": artifacts.tokenized_dataset_path,
        "metadata_path": artifacts.metadata_path,
        "provenance_path": artifacts.provenance_path,
        "config_path": artifacts.config_path,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
