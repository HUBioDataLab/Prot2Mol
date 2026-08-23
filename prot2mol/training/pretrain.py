"""End-to-end Prot2Mol training entrypoint."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.distributed import destroy_process_group

from prot2mol.chem import generate_morgan_fingerprints_parallel
from prot2mol.core.model import create_prot2mol_model
from prot2mol.core.protein_encoders import (
    get_protein_tokenizer,
    resolve_protein_model_id,
)
from prot2mol.data.pipeline import (
    extract_smiles_list,
    load_processed_dataset,
    tokenize_protein_sequences_for_inference,
)
from prot2mol.io.hf_utils import (
    load_molgen_tokenizer,
    prepare_prot2mol_state_dict,
)
from prot2mol.training.distributed import resolve_distributed_context
from prot2mol.training.entry import (
    create_run_name,
    parse_arguments,
    setup_logging,
    validate_and_process_paths,
)
from prot2mol.training.metrics import (
    compute_conditional_generation_metrics,
    compute_generation_metrics,
)
from prot2mol.training.training_runner import TrainingRunner


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DIR", os.path.abspath("./wandb"))


class TrainingScript:
    """Own model construction, split validation, and bounded final evaluation."""

    def __init__(self, config, pretrain_save_to, dataset_name, run_name, distributed_context):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.dataset_path = config.dataset_path
        self.pretrain_save_to = pretrain_save_to
        self.dataset_name = dataset_name
        self.run_name = run_name
        self.distributed_context = distributed_context
        self.local_rank = distributed_context.local_rank
        self.global_rank = distributed_context.global_rank

        self.model_config = {
            "prot_emb_model": config.prot_emb_model,
            "protein_model_id": config.protein_model_id,
            "decoder_type": config.decoder_type,
            "decoder_model_id": config.decoder_model_id,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_emb": config.n_emb,
            "conditioning_dropout": config.conditioning_dropout,
            "max_mol_len": config.max_mol_len,
            "prot_max_length": config.prot_max_length,
            "train_encoder_model": config.train_encoder_model,
            "train_projection_model": config.train_projection_model,
            "train_decoder_model": config.train_decoder_model,
        }
        self.training_config = {
            "train_batch_size": config.train_batch_size,
            "valid_batch_size": config.valid_batch_size,
            "epochs": config.epoch,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "dataloader_num_workers": config.dataloader_num_workers,
            "resume_from_checkpoint": config.resume_from_checkpoint,
            "load_pretrained_model": config.load_pretrained_model,
            "precision": config.precision,
            "max_grad_norm": config.max_grad_norm,
            "logging_steps": config.logging_steps,
            "split_seed": config.split_seed,
            "generation_eval_proteins": config.generation_eval_proteins,
            "generation_samples_per_protein": config.generation_samples_per_protein,
            "generation_train_reference_limit": config.generation_train_reference_limit,
        }

        self.mol_tokenizer = load_molgen_tokenizer(
            padding_side="right",
            model_id=config.decoder_model_id,
        )
        self.prot_tokenizer = get_protein_tokenizer(
            config.prot_emb_model,
            model_id=config.protein_model_id,
        )
        self.prot_tokenizer.padding_side = "right"
        model_config = {**self.model_config, "mol_tokenizer": self.mol_tokenizer}
        self.model = create_prot2mol_model(model_config)
        self.training_runner = TrainingRunner(self.local_rank, self.global_rank, self.logger)
        self.train_data = None
        self.validation_data = None
        self.generation_sequences = []
        self.eval_reference_smiles = []
        self.eval_reference_smiles_by_sequence = {}
        self.train_reference_smiles = []
        self.training_vec = None
        self.trainer = None

    def _load_and_validate_preprocessing_manifest(self) -> dict:
        manifest_path = Path(self.dataset_path) / "preprocessing_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing preprocessing contract: {manifest_path}. "
                "Re-run data_processing/preprocess_dataset.py."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        preprocessed = manifest.get("config", {})
        expected_protein_id = resolve_protein_model_id(
            self.model_config["prot_emb_model"],
            self.model_config["protein_model_id"],
        )
        expected = {
            "prot_emb_model": self.model_config["prot_emb_model"],
            "protein_model_id_resolved": expected_protein_id,
            "decoder_model_id_resolved": self.model_config["decoder_model_id"],
            "max_mol_len": int(self.model_config["max_mol_len"]),
            "prot_max_length": int(self.model_config["prot_max_length"]),
            "molecule_padding_side": "right",
            "protein_padding_side": "right",
        }
        actual = {
            "prot_emb_model": preprocessed.get("prot_emb_model"),
            "protein_model_id_resolved": manifest.get("protein_model_id_resolved"),
            "decoder_model_id_resolved": manifest.get("decoder_model_id_resolved"),
            "max_mol_len": preprocessed.get("max_mol_len"),
            "prot_max_length": preprocessed.get("prot_max_length"),
            "molecule_padding_side": manifest.get("molecule_padding_side"),
            "protein_padding_side": manifest.get("protein_padding_side"),
        }
        mismatches = {
            key: {"expected": value, "actual": actual[key]}
            for key, value in expected.items()
            if actual[key] != value
        }
        for coverage_name in (
            "molecule_tokenizer_coverage",
            "protein_tokenizer_coverage",
        ):
            coverage = manifest.get(coverage_name, {}).get("splits", {})
            for split in ("train", "validation"):
                if coverage.get(split, {}).get("unknown_tokens") != 0:
                    mismatches[f"{coverage_name}.{split}.unknown_tokens"] = {
                        "expected": 0,
                        "actual": coverage.get(split, {}).get("unknown_tokens"),
                    }
        if mismatches:
            raise ValueError(
                "Preprocessed dataset does not match the training tokenization contract: "
                f"{mismatches}"
            )
        return manifest

    def ddp_setup(self):
        self.training_runner.ddp_setup()

    @staticmethod
    def _is_right_padded(mask) -> bool:
        seen_padding = False
        for value in mask:
            if int(value) == 0:
                seen_padding = True
            elif seen_padding:
                return False
        return True

    def _validate_tokenized_contract(self) -> None:
        train_clusters = set(self.train_data.unique("protein_cluster_50"))
        validation_clusters = set(self.validation_data.unique("protein_cluster_50"))
        overlap = train_clusters & validation_clusters
        if overlap:
            raise ValueError(f"MMseqs50 leakage in tokenized data: {len(overlap)} clusters")

        expected = {
            "prot_input_ids",
            "prot_attention_mask",
            "labels",
        }
        for name, split in (("train", self.train_data), ("validation", self.validation_data)):
            missing = sorted(expected.difference(split.column_names))
            if missing:
                raise ValueError(f"{name} split lacks tokenized fields: {missing}")
            sample = split.select(range(min(128, len(split))))
            if not all(self._is_right_padded(row) for row in sample["prot_attention_mask"]):
                raise ValueError(f"{name}.prot_attention_mask is not right padded")
            if not all(
                self._is_right_padded([0 if value == -100 else 1 for value in row])
                for row in sample["labels"]
            ):
                raise ValueError(f"{name}.labels is not right padded")
            max_token_id = max(
                value
                for row in sample["labels"]
                for value in row
                if value != -100
            )
            if max_token_id >= len(self.mol_tokenizer):
                raise ValueError(
                    f"{name} contains molecule token {max_token_id}, outside vocabulary {len(self.mol_tokenizer)}"
                )

    def _select_generation_panel(self) -> None:
        limit = max(0, int(self.training_config["generation_eval_proteins"]))
        if limit == 0:
            return
        sequences = set(str(value) for value in self.validation_data["protein_sequence"])
        seed = self.training_config["split_seed"]
        ordered = sorted(
            sequences,
            key=lambda sequence: hashlib.sha256(f"{seed}:{sequence}".encode()).hexdigest(),
        )
        self.generation_sequences = ordered[:limit]
        selected = set(self.generation_sequences)
        panel_indices = [
            index
            for index, sequence in enumerate(self.validation_data["protein_sequence"])
            if str(sequence) in selected
        ]
        panel = self.validation_data.select(panel_indices)
        self.eval_reference_smiles = extract_smiles_list(panel, drop_invalid=True, logger=self.logger)
        for sequence in self.generation_sequences:
            sequence_indices = [
                index
                for index, value in enumerate(panel["protein_sequence"])
                if str(value) == sequence
            ]
            self.eval_reference_smiles_by_sequence[sequence] = extract_smiles_list(
                panel.select(sequence_indices),
                drop_invalid=True,
                logger=self.logger,
            )

        train_limit = max(0, int(self.training_config["generation_train_reference_limit"]))
        if train_limit:
            count = min(train_limit, len(self.train_data))
            indices = np.linspace(0, len(self.train_data) - 1, num=count, dtype=int)
            train_sample = self.train_data.select(indices.tolist())
            self.train_reference_smiles = extract_smiles_list(
                train_sample,
                drop_invalid=True,
                logger=self.logger,
            )
            if self.train_reference_smiles:
                self.training_vec = generate_morgan_fingerprints_parallel(
                    smiles=self.train_reference_smiles,
                    radius=2,
                    nBits=1024,
                    n_jobs=None,
                )

    def _prepare_datasets(self) -> None:
        preprocessing_manifest = self._load_and_validate_preprocessing_manifest()
        dataset = load_processed_dataset(self.dataset_path)
        self.train_data = dataset["train"]
        self.validation_data = dataset["validation"]
        self._validate_tokenized_contract()
        expected_split_sizes = preprocessing_manifest.get("splits", {})
        actual_split_sizes = {
            "train": len(self.train_data),
            "validation": len(self.validation_data),
        }
        if expected_split_sizes != actual_split_sizes:
            raise ValueError(
                "Tokenized dataset sizes differ from preprocessing manifest: "
                f"expected={expected_split_sizes}, actual={actual_split_sizes}"
            )
        if self.global_rank == 0:
            self._select_generation_panel()

        dataset_stats = {
            "dataset_name": self.dataset_name,
            "dataset_source_path": str(Path(self.dataset_path).resolve()),
            "dataset_total_samples": len(self.train_data) + len(self.validation_data),
            "train_samples": len(self.train_data),
            "eval_samples": len(self.validation_data),
            "train_unique_proteins": len(self.train_data.unique("protein_sequence")),
            "eval_unique_proteins": len(self.validation_data.unique("protein_sequence")),
            "generation_eval_proteins": len(self.generation_sequences),
            "generation_samples_per_protein": self.training_config[
                "generation_samples_per_protein"
            ],
            "split_seed": self.training_config["split_seed"],
        }
        self.model._config.update(dataset_stats)
        self.logger.info("Dataset stats: %s", dataset_stats)

    def _load_pretrained_weights(self, model_path: str) -> None:
        path = Path(model_path)
        if path.is_dir():
            candidates = [path / "pytorch_model.bin", path / "model.safetensors"]
            checkpoint = next((candidate for candidate in candidates if candidate.exists()), None)
        else:
            checkpoint = path
        if checkpoint is None or not checkpoint.exists():
            raise FileNotFoundError(f"No model weights found at {model_path}")
        if checkpoint.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(checkpoint))
        else:
            state = torch.load(str(checkpoint), map_location="cpu")
        state = prepare_prot2mol_state_dict(
            state,
            decoder_type=self.model.decoder_type,
        )
        self.model.load_state_dict(state, strict=True)

    def compute_generation_eval_metrics(self):
        if self.global_rank != 0 or not self.generation_sequences:
            return {}
        model = getattr(self.trainer.model, "module", self.trainer.model)
        device = next(model.parameters()).device
        samples_per_protein = self.training_config["generation_samples_per_protein"]
        batches = {}
        cuda_devices = [device.index] if device.type == "cuda" and device.index is not None else []
        with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(self.training_config["split_seed"])
            for sequence in self.generation_sequences:
                prot_ids, prot_mask = tokenize_protein_sequences_for_inference(
                    [sequence],
                    prot_tokenizer=self.prot_tokenizer,
                    prot_emb_model=self.model_config["prot_emb_model"],
                    prot_max_length=self.model_config["prot_max_length"],
                    device=device,
                )
                embeddings = model.encode_protein(prot_ids, prot_mask)
                generated = model.generate_from_protein_embeddings(
                    embeddings.repeat(samples_per_protein, 1, 1),
                    prot_mask.repeat(samples_per_protein, 1),
                    max_length=self.model_config["max_mol_len"],
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.9,
                )
                batches[sequence] = generated.cpu().numpy()
        token_ids = np.concatenate(list(batches.values()), axis=0)
        metrics = compute_generation_metrics(
            generated_token_ids=token_ids,
            mol_tokenizer=self.mol_tokenizer,
            eval_reference_smiles=self.eval_reference_smiles,
            train_smiles_list=self.train_reference_smiles,
            training_vec=self.training_vec,
        )
        metrics.update(
            compute_conditional_generation_metrics(
                generated_token_ids_by_protein=batches,
                mol_tokenizer=self.mol_tokenizer,
                reference_smiles_by_protein=self.eval_reference_smiles_by_sequence,
            )
        )
        metrics["gen_count"] = int(len(token_ids))
        metrics["gen_protein_count"] = int(len(self.generation_sequences))
        return metrics

    def model_training(self):
        if self.training_config["load_pretrained_model"]:
            self._load_pretrained_weights(self.training_config["load_pretrained_model"])
        self._prepare_datasets()

        if self.global_rank == 0:
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "prot2mol"),
                name=self.run_name,
                config={**self.model_config, **self.training_config},
            )
        try:
            self.trainer = self.training_runner.create_trainer(
                model=self.model,
                train_dataset=self.train_data,
                eval_dataset=self.validation_data,
                compute_generation_metrics=self.compute_generation_eval_metrics,
                run_name=self.run_name,
                output_dir=self.pretrain_save_to,
                training_config=self.training_config,
            )
            return self.training_runner.run(
                self.trainer,
                self.training_config,
                self.pretrain_save_to,
            )
        finally:
            if self.global_rank == 0 and wandb.run is not None:
                wandb.finish()


def main():
    config = parse_arguments()
    context = resolve_distributed_context(config.training_mode)
    setup_logging(config.log_level, rank=context.global_rank)
    dataset_name = validate_and_process_paths(config)
    run_name = create_run_name(config, dataset_name)
    save_dir = os.path.join(config.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    script = TrainingScript(config, save_dir, dataset_name, run_name, context)
    try:
        if context.is_distributed:
            script.ddp_setup()
        script.model_training()
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            destroy_process_group()


if __name__ == "__main__":
    main()
