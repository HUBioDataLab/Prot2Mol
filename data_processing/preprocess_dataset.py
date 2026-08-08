#!/usr/bin/env python3
"""Tokenize the pre-split ChEMBL generation dataset once for training."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prot2mol.core.protein_encoders import (
    get_protein_tokenizer,
    resolve_protein_model_id,
)
from prot2mol.data.pipeline import tokenize_molecule_batch, tokenize_protein_batch
from prot2mol.io.hf_utils import load_molgen_tokenizer


@dataclass(frozen=True)
class PreprocessingConfig:
    input_dir: Path
    output_dir: Path
    prot_emb_model: str = "esm2"
    protein_model_id: str | None = None
    decoder_model_id: str = "zjunlp/MolGen-large"
    max_mol_len: int = 256
    prot_max_length: int = 1024
    num_proc: int = 8
    batch_size: int = 1_000
    datasets_cache_dir: Path | None = None
    overwrite: bool = False

    def validate(self) -> None:
        if self.max_mol_len < 3 or self.prot_max_length < 3:
            raise ValueError("Token contexts must leave room for content and special tokens")
        if self.num_proc < 1 or self.batch_size < 1:
            raise ValueError("num_proc and batch_size must be positive")


class DatasetPreprocessor:
    """Convert raw train/validation Parquets into one HF DatasetDict."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.config.validate()
        self.logger = logging.getLogger(__name__)
        self.mol_tokenizer = load_molgen_tokenizer(
            padding_side="right",
            model_id=config.decoder_model_id,
        )
        self.prot_tokenizer = get_protein_tokenizer(
            config.prot_emb_model,
            model_id=config.protein_model_id,
        )
        self.prot_tokenizer.padding_side = "right"

    def _source_paths(self) -> dict[str, str]:
        paths = {
            "train": self.config.input_dir / "train.parquet",
            "validation": self.config.input_dir / "validation.parquet",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing generation split files: {missing}")
        if (self.config.input_dir / "test.parquet").exists():
            raise ValueError("Generation dataset unexpectedly contains test.parquet")
        return {split: str(path) for split, path in paths.items()}

    @staticmethod
    def _validate_columns(dataset) -> None:
        required = {"protein_sequence", "compound_selfies", "smiles", "protein_cluster_50"}
        for split in ("train", "validation"):
            missing = sorted(required.difference(dataset[split].column_names))
            if missing:
                raise ValueError(f"{split} split is missing columns: {missing}")

    def _validate_context_lengths(self, dataset) -> None:
        max_protein = self.config.prot_max_length - 2
        max_selfies = self.config.max_mol_len - 2
        for split in ("train", "validation"):
            protein_too_long = sum(
                len(str(sequence)) > max_protein
                for sequence in dataset[split]["protein_sequence"]
            )
            selfies_too_long = sum(
                str(value).count("[") > max_selfies
                for value in dataset[split]["compound_selfies"]
            )
            if protein_too_long or selfies_too_long:
                raise ValueError(
                    f"{split} exceeds configured contexts: proteins={protein_too_long}, "
                    f"molecules={selfies_too_long}. Rebuild the raw generation split "
                    "with matching length limits instead of truncating targets."
                )

    def _molecule_tokenizer_coverage(self, dataset) -> dict[str, object]:
        """Prove that MolGen can represent every target without truncation or UNKs."""

        unknown_token_id = getattr(self.mol_tokenizer, "unk_token_id", None)
        coverage: dict[str, object] = {
            "tokenizer": getattr(
                self.mol_tokenizer,
                "name_or_path",
                self.config.decoder_model_id,
            ),
            "vocabulary_size": int(len(self.mol_tokenizer)),
            "unknown_token_id": unknown_token_id,
            "splits": {},
        }
        for split in ("train", "validation"):
            active_tokens = 0
            unknown_tokens = 0
            unknown_sequences = 0
            max_encoded_length = 0
            max_token_id = -1
            for batch in dataset[split].iter(batch_size=10_000):
                token_ids = np.asarray(batch["mol_input_ids"], dtype=np.int64)
                attention_mask = np.asarray(batch["mol_attention_mask"], dtype=bool)
                active_tokens += int(attention_mask.sum())
                lengths = attention_mask.sum(axis=1)
                max_encoded_length = max(max_encoded_length, int(lengths.max(initial=0)))
                if attention_mask.any():
                    max_token_id = max(max_token_id, int(token_ids[attention_mask].max()))
                if unknown_token_id is not None:
                    unknown_mask = (token_ids == int(unknown_token_id)) & attention_mask
                    unknown_tokens += int(unknown_mask.sum())
                    unknown_sequences += int(unknown_mask.any(axis=1).sum())

            split_coverage = {
                "sequences": int(len(dataset[split])),
                "active_tokens": active_tokens,
                "unknown_tokens": unknown_tokens,
                "unknown_sequences": unknown_sequences,
                "max_encoded_length": max_encoded_length,
                "max_token_id": max_token_id,
            }
            coverage["splits"][split] = split_coverage
            if unknown_tokens:
                raise ValueError(
                    f"MolGen tokenizer produced {unknown_tokens:,} unknown tokens in "
                    f"{unknown_sequences:,} {split} targets."
                )
            if max_encoded_length > self.config.max_mol_len:
                raise ValueError(
                    f"{split} contains an encoded molecule of length {max_encoded_length}, "
                    f"above max_mol_len={self.config.max_mol_len}."
                )
        return coverage

    def _protein_tokenizer_coverage(self, dataset) -> dict[str, object]:
        unknown_token_id = getattr(self.prot_tokenizer, "unk_token_id", None)
        coverage: dict[str, object] = {
            "tokenizer": getattr(
                self.prot_tokenizer,
                "name_or_path",
                resolve_protein_model_id(
                    self.config.prot_emb_model,
                    self.config.protein_model_id,
                ),
            ),
            "vocabulary_size": int(len(self.prot_tokenizer)),
            "unknown_token_id": unknown_token_id,
            "splits": {},
        }
        for split in ("train", "validation"):
            active_tokens = 0
            unknown_tokens = 0
            unknown_sequences = 0
            max_encoded_length = 0
            max_token_id = -1
            for batch in dataset[split].iter(batch_size=10_000):
                token_ids = np.asarray(batch["prot_input_ids"], dtype=np.int64)
                attention_mask = np.asarray(batch["prot_attention_mask"], dtype=bool)
                active_tokens += int(attention_mask.sum())
                lengths = attention_mask.sum(axis=1)
                max_encoded_length = max(max_encoded_length, int(lengths.max(initial=0)))
                if attention_mask.any():
                    max_token_id = max(max_token_id, int(token_ids[attention_mask].max()))
                if unknown_token_id is not None:
                    unknown_mask = (token_ids == int(unknown_token_id)) & attention_mask
                    unknown_tokens += int(unknown_mask.sum())
                    unknown_sequences += int(unknown_mask.any(axis=1).sum())
            coverage["splits"][split] = {
                "sequences": int(len(dataset[split])),
                "active_tokens": active_tokens,
                "unknown_tokens": unknown_tokens,
                "unknown_sequences": unknown_sequences,
                "max_encoded_length": max_encoded_length,
                "max_token_id": max_token_id,
            }
            if unknown_tokens:
                raise ValueError(
                    f"Protein tokenizer produced {unknown_tokens:,} unknown tokens in "
                    f"{unknown_sequences:,} {split} sequences."
                )
            if max_encoded_length > self.config.prot_max_length:
                raise ValueError(
                    f"{split} contains an encoded protein of length {max_encoded_length}, "
                    f"above prot_max_length={self.config.prot_max_length}."
                )
        return coverage

    def preprocess(self):
        output_dir = self.config.output_dir
        if output_dir.exists():
            if not self.config.overwrite:
                raise FileExistsError(
                    f"Output already exists: {output_dir}. Pass --overwrite to replace it."
                )
            shutil.rmtree(output_dir)

        source_paths = self._source_paths()
        datasets_cache = self.config.datasets_cache_dir or (
            self.config.output_dir.parent / ".datasets_cache"
        )
        datasets_cache.mkdir(parents=True, exist_ok=True)
        dataset = load_dataset(
            "parquet",
            data_files=source_paths,
            cache_dir=str(datasets_cache),
        )
        self._validate_columns(dataset)
        self._validate_context_lengths(dataset)
        num_proc = self.config.num_proc if self.config.num_proc > 1 else None
        dataset = dataset.map(
            lambda batch: tokenize_protein_batch(
                batch,
                prot_tokenizer=self.prot_tokenizer,
                prot_emb_model=self.config.prot_emb_model,
                prot_max_length=self.config.prot_max_length,
            ),
            batched=True,
            num_proc=num_proc,
            batch_size=self.config.batch_size,
            desc="Tokenizing proteins",
        )
        dataset = dataset.map(
            lambda batch: tokenize_molecule_batch(
                batch,
                mol_tokenizer=self.mol_tokenizer,
                max_mol_len=self.config.max_mol_len,
            ),
            batched=True,
            num_proc=num_proc,
            batch_size=self.config.batch_size,
            desc="Tokenizing molecules",
        )
        protein_coverage = self._protein_tokenizer_coverage(dataset)
        molecule_coverage = self._molecule_tokenizer_coverage(dataset)
        dataset = dataset.remove_columns(["mol_input_ids", "mol_attention_mask"])
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_dir))

        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": {
                **asdict(self.config),
                "input_dir": str(self.config.input_dir),
                "output_dir": str(self.config.output_dir),
            },
            "splits": {split: int(len(dataset[split])) for split in dataset},
            "molecule_padding_side": self.mol_tokenizer.padding_side,
            "protein_padding_side": self.prot_tokenizer.padding_side,
            "protein_model_id_resolved": resolve_protein_model_id(
                self.config.prot_emb_model,
                self.config.protein_model_id,
            ),
            "decoder_model_id_resolved": self.config.decoder_model_id,
            "molecule_token_ids": {
                "pad": self.mol_tokenizer.pad_token_id,
                "bos": self.mol_tokenizer.bos_token_id,
                "eos": self.mol_tokenizer.eos_token_id,
            },
            "molecule_tokenizer_coverage": molecule_coverage,
            "protein_tokenizer_coverage": protein_coverage,
        }
        (output_dir / "preprocessing_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str),
            encoding="utf-8",
        )
        return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("dataset/processed/chembl_37/protein_cluster_50_generation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/cache/prot2mol/chembl_37_mmseqs50_generation_esm2"),
    )
    parser.add_argument("--prot-emb-model", choices=["esm2", "prot_t5"], default="esm2")
    parser.add_argument("--protein-model-id", default=None)
    parser.add_argument("--decoder-model-id", default="zjunlp/MolGen-large")
    parser.add_argument("--max-mol-len", type=int, default=256)
    parser.add_argument("--prot-max-length", type=int, default=1024)
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--datasets-cache-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = DatasetPreprocessor(
        PreprocessingConfig(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            prot_emb_model=args.prot_emb_model,
            protein_model_id=args.protein_model_id,
            decoder_model_id=args.decoder_model_id,
            max_mol_len=args.max_mol_len,
            prot_max_length=args.prot_max_length,
            num_proc=args.num_proc,
            batch_size=args.batch_size,
            datasets_cache_dir=args.datasets_cache_dir,
            overwrite=args.overwrite,
        )
    ).preprocess()
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
