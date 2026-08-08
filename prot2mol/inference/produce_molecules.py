#!/usr/bin/env python3
"""Generate SELFIES for an explicit protein sequence or ChEMBL target."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import selfies as sf
import torch
from tqdm import tqdm

from prot2mol.chem.utils import canonicalize_smiles_list, metrics_calculation
from prot2mol.core.protein_encoders import get_protein_tokenizer
from prot2mol.data.pipeline import extract_smiles_list, tokenize_protein_sequences_for_inference
from prot2mol.io.config import parse_args_with_config
from prot2mol.io.hf_utils import (
    load_molgen_tokenizer,
    load_prot2mol_inference_model,
    load_saved_model_config,
)


class MoleculeGenerator:
    def __init__(self, config: argparse.Namespace):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_references: List[str] = []
        self._load_components()

    def _apply_saved_config(self) -> None:
        for key, value in load_saved_model_config(self.config.model_file, self.logger).items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def _load_components(self) -> None:
        self._apply_saved_config()
        self.mol_tokenizer = load_molgen_tokenizer(
            models_base=self.config.models_base,
            padding_side="right",
            model_id=self.config.decoder_model_id,
        )
        self.prot_tokenizer = get_protein_tokenizer(
            self.config.prot_emb_model,
            model_id=self.config.protein_model_id,
        )
        self.model = load_prot2mol_inference_model(
            model_path=self.config.model_file,
            device=self.device,
            mol_tokenizer=self.mol_tokenizer,
            prot_emb_model=self.config.prot_emb_model,
            protein_model_id=self.config.protein_model_id,
            decoder_model_id=self.config.decoder_model_id,
            conditioning_dropout=self.config.conditioning_dropout,
            max_mol_len=self.config.max_mol_len,
            prot_max_length=self.config.prot_max_length,
            strict=True,
            logger=self.logger,
        )
        self.model.eval()

    @staticmethod
    def _read_frame(path: Path) -> pd.DataFrame:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        raise ValueError(f"Unsupported dataset file: {path}")

    @staticmethod
    def _first_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
        return next((column for column in candidates if column in frame), None)

    def _resolve_protein_and_references(self) -> tuple[str, List[str]]:
        if self.config.protein_sequence:
            return str(self.config.protein_sequence), []
        if not self.config.dataset_path or not self.config.protein_id:
            raise ValueError("Provide --protein_sequence, or both --dataset_path and --protein_id")

        source = Path(self.config.dataset_path).expanduser()
        if source.is_dir():
            validation_path = source / "validation.parquet"
            if not validation_path.exists():
                raise FileNotFoundError(f"Missing {validation_path}")
            frame = pd.read_parquet(validation_path)
            train_path = source / "train.parquet"
            if train_path.exists() and self.config.train_reference_limit > 0:
                train = pd.read_parquet(train_path).head(self.config.train_reference_limit)
                self.train_references = extract_smiles_list(train, drop_invalid=True)
        elif source.is_file():
            frame = self._read_frame(source)
        else:
            raise FileNotFoundError(f"Dataset not found: {source}")

        id_column = self._first_column(
            frame,
            ("target_chembl_id", "protein_accession", "protein_id", "Target_CHEMBL_ID"),
        )
        sequence_column = self._first_column(frame, ("protein_sequence", "Target_FASTA"))
        if id_column is None or sequence_column is None:
            raise ValueError("Dataset lacks target identifier or protein sequence columns")
        target = frame.loc[frame[id_column].astype(str).eq(str(self.config.protein_id))]
        if target.empty:
            raise ValueError(f"No rows found for protein {self.config.protein_id}")
        sequence_values = target[sequence_column].dropna().astype(str).unique()
        if len(sequence_values) != 1:
            raise ValueError(f"Protein id maps to {len(sequence_values)} distinct sequences")
        return sequence_values[0], extract_smiles_list(target, drop_invalid=True)

    def _generate_tokens(self, protein_sequence: str) -> torch.Tensor:
        prot_ids, prot_mask = tokenize_protein_sequences_for_inference(
            [protein_sequence],
            prot_tokenizer=self.prot_tokenizer,
            prot_emb_model=self.config.prot_emb_model,
            prot_max_length=self.config.prot_max_length,
            device=self.device,
        )
        generated = []
        with torch.no_grad():
            embeddings = self.model.encode_protein(prot_ids, prot_mask)
            for start in tqdm(range(0, self.config.num_samples, self.config.batch_size)):
                size = min(self.config.batch_size, self.config.num_samples - start)
                generated.append(
                    self.model.generate_from_protein_embeddings(
                        embeddings.repeat(size, 1, 1),
                        prot_mask.repeat(size, 1),
                        max_length=self.config.max_mol_len,
                        do_sample=True,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                    ).cpu()
                )
        return torch.cat(generated)

    def run_generation(self):
        started = time.perf_counter()
        protein_sequence, references = self._resolve_protein_and_references()
        token_ids = self._generate_tokens(protein_sequence)
        generated_selfies = [
            self.mol_tokenizer.decode(tokens, skip_special_tokens=True).replace(" ", "")
            for tokens in token_ids
        ]
        generated = pd.DataFrame(
            {
                "generated_selfies": generated_selfies,
                "generated_smiles": [self._decode(value) for value in generated_selfies],
                "protein_id": self.config.protein_id,
                "protein_sequence": protein_sequence,
                "model": Path(self.config.model_file).name,
            }
        )
        metrics: Dict[str, float] = metrics_calculation(
            predictions=generated_selfies,
            references=canonicalize_smiles_list(references, drop_invalid=True),
            train_data=self.train_references,
        )
        metrics["generation_time_sec"] = time.perf_counter() - started

        output = Path(self.config.output_file)
        output.parent.mkdir(parents=True, exist_ok=True)
        generated.to_csv(output, index=False)
        output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return generated, metrics

    @staticmethod
    def _decode(value: str) -> str:
        try:
            return sf.decoder(value) or ""
        except Exception:
            return ""


def parse_arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--prot_emb_model", default="esm2", choices=["prot_t5", "esm2"])
    parser.add_argument("--protein_model_id", default=None)
    parser.add_argument("--decoder_model_id", default="zjunlp/MolGen-large")
    parser.add_argument("--conditioning_dropout", type=float, default=0.1)
    parser.add_argument("--models_base", default=None)
    parser.add_argument("--protein_sequence", default=None)
    parser.add_argument("--protein_id", default=None)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--train_reference_limit", type=int, default=10_000)
    parser.add_argument("--num_samples", type=int, default=1_000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--prot_max_length", type=int, default=1024)
    parser.add_argument("--max_mol_len", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--output_file", default="./generated_molecules.csv")
    config = parse_args_with_config(parser, section="generate", argv=argv)
    if config.num_samples < 1 or config.batch_size < 1:
        raise ValueError("num_samples and batch_size must be positive")
    if config.prot_max_length < 3 or config.max_mol_len < 3:
        raise ValueError("Token contexts must leave room for content and special tokens")
    if config.temperature <= 0 or not 0 < config.top_p <= 1:
        raise ValueError("temperature must be positive and top_p must be in (0, 1]")
    if config.train_reference_limit < 0:
        raise ValueError("train_reference_limit cannot be negative")
    return config


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    generator = MoleculeGenerator(parse_arguments())
    generated, metrics = generator.run_generation()
    print(f"Generated {len(generated)} molecules")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
