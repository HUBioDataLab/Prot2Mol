# Standard library imports
from collections import Counter
import logging
import os
import sys
from typing import Dict

from torch.distributed import destroy_process_group

# Third-party library imports
import numpy as np
import torch
import wandb

# Local application imports
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from prot2mol.data.pipeline import (
    attach_metric_group_ids,
    attach_split_targets,
    extract_smiles_list,
    get_processed_data_path,
    has_matching_precomputed_split,
    load_processed_dataset,
    load_processed_stats,
    split_train_eval_dataset,
    tokenize_protein_sequences_for_inference,
)
from prot2mol.chem import generate_morgan_fingerprints_parallel
from prot2mol.io.hf_utils import (
    filter_legacy_pchembl_head_state,
    load_molgen_tokenizer,
    load_saved_model_config,
)
from prot2mol.core.model import create_prot2mol_model
from prot2mol.core.protein_encoders import get_protein_tokenizer
from prot2mol.training.entry import (
    create_run_name,
    parse_arguments,
    setup_logging,
    validate_and_process_paths,
)
from prot2mol.training.distributed import resolve_distributed_context
from prot2mol.training.metrics import (
    compute_generation_metrics,
    compute_pchembl_metrics,
    preprocess_logits_for_metrics as preprocess_logits_for_metrics_fn,
)
from prot2mol.training.training_runner import TrainingRunner

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DIR"] = "/gpfs/projects/etur29/atabey/"


class TrainingScript:
    """Trainer orchestrator for the Prot2Mol model."""

    def __init__(self, config, selfies_path, pretrain_save_to, dataset_name, run_name, distributed_context):
        self.logger = logging.getLogger(__name__)
        self.distributed_context = distributed_context
        self.local_rank = distributed_context.local_rank
        self.global_rank = distributed_context.global_rank

        # Organize configurations into logical groups
        self.model_config = {
            "prot_emb_model": config.prot_emb_model,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_emb": config.n_emb,
            "max_mol_len": config.max_mol_len,
            "prot_max_length": config.prot_max_length,
            "train_encoder_model": config.train_encoder_model,
            "train_decoder_model": config.train_decoder_model,
            "train_pchembl_head": config.train_pchembl_head,
            "training_stage": config.training_stage,
            "pchembl_huber_delta": config.pchembl_huber_delta,
            "stop_pchembl_gradients": config.stop_pchembl_gradients,
            "pchembl_tf_hidden_dim": config.pchembl_tf_hidden_dim,
            "pchembl_tf_num_heads": config.pchembl_tf_num_heads,
            "pchembl_tf_group_size": config.pchembl_tf_group_size,
            "pchembl_tf_agg_mode": config.pchembl_tf_agg_mode,
            "pchembl_tf_dropout": config.pchembl_tf_dropout,
        }
        self.training_config = {
            "train_batch_size": config.train_batch_size,
            "valid_batch_size": config.valid_batch_size,
            "epochs": config.epoch,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "dataloader_num_workers": config.dataloader_num_workers,
            "pchembl_huber_delta": config.pchembl_huber_delta,
            "resume_from_checkpoint": config.resume_from_checkpoint,
            "load_pretrained_model": config.load_pretrained_model,
            "ignore_mismatched_optimizer": config.ignore_mismatched_optimizer,
            "training_mode": config.training_mode,
            "training_stage": config.training_stage,
            "eval_split": config.eval_split,
            "eval_split_ratio": config.eval_split_ratio,
            "split_seed": config.split_seed,
        }
        self._validate_training_stage()

        self.selfies_path = selfies_path
        self.pretrain_save_to = pretrain_save_to
        self.dataset_name = dataset_name
        self.run_name = run_name
        self.train_smiles_list = []
        self.eval_reference_smiles = []
        self.dataset_stats: Dict[str, object] = {}
        self.trainer = None
        self.training_vec = None
        self.pchembl_mean = 0.0
        self.pchembl_std = 1.0
        self.pchembl_threshold = 6.0

        if self.training_config["resume_from_checkpoint"]:
            self.logger.info(
                "Checkpoint resume mode enabled: %s",
                self.training_config["resume_from_checkpoint"],
            )

        if self.training_config["load_pretrained_model"]:
            self.logger.info(
                "Will load pretrained model weights from: %s",
                self.training_config["load_pretrained_model"],
            )
            if self.training_config["resume_from_checkpoint"]:
                raise ValueError(
                    "Cannot use both --resume_from_checkpoint and --load_pretrained_model. "
                    "Use --resume_from_checkpoint to continue training with same dataset, "
                    "or --load_pretrained_model to fine-tune on a new dataset."
                )

        self._init_tokenizers()
        self._init_models()
        self.training_runner = TrainingRunner(
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            logger=self.logger,
        )

        self.logger.info("Model parameter count: %s", f"{self.model.num_parameters():,}")

    def _extract_smiles_list(self, data_source, drop_invalid=False):
        """Extract canonical SMILES using shared data-pipeline helpers."""
        return extract_smiles_list(data_source, drop_invalid=drop_invalid, logger=self.logger)

    def _validate_training_stage(self):
        stage = self.training_config.get("training_stage", self.model_config["training_stage"])
        if stage in {"pchembl_only", "multitask"} and not self.model_config["train_pchembl_head"]:
            raise ValueError(
                f"training_stage={stage} requires --train_pchembl_head to be enabled."
            )
        if stage in {"lm_only", "multitask"} and not (
            self.model_config["train_encoder_model"] or self.model_config["train_decoder_model"]
        ):
            raise ValueError(
                f"training_stage={stage} requires at least one of encoder/decoder to be trainable."
            )

    def _stage_has_lm(self) -> bool:
        return self.training_config.get("training_stage", self.model_config["training_stage"]) in {"lm_only", "multitask"}

    def _stage_has_pchembl(self) -> bool:
        return self.training_config.get("training_stage", self.model_config["training_stage"]) in {"pchembl_only", "multitask"}

    def _apply_split_targets(self, dataset_split):
        self.logger.info(
            "Attaching split targets for stage=%s on %s samples",
            self.training_config["training_stage"],
            len(dataset_split),
        )
        return attach_split_targets(
            dataset_split=dataset_split,
            training_stage=self.training_config["training_stage"],
            pchembl_mean=self.pchembl_mean,
            pchembl_std=self.pchembl_std,
            pchembl_threshold=self.pchembl_threshold,
        )

    def _set_split_normalization(self, pchembl_mean: float, pchembl_std: float, pchembl_threshold: float):
        self.pchembl_mean = float(pchembl_mean)
        self.pchembl_std = float(pchembl_std)
        self.pchembl_threshold = float(pchembl_threshold)
        pchembl_huber_delta = self.training_config.get(
            "pchembl_huber_delta",
            self.model_config["pchembl_huber_delta"],
        )
        normalized_delta = (
            pchembl_huber_delta / self.pchembl_std
            if self.pchembl_std > 0
            else pchembl_huber_delta
        )
        self.model._config["pchembl_huber_delta"] = normalized_delta
        self.model._config["pchembl_mean"] = self.pchembl_mean
        self.model._config["pchembl_std"] = self.pchembl_std
        self.model._config["pchembl_threshold"] = self.pchembl_threshold
        self.model_config["pchembl_huber_delta"] = normalized_delta

        self.logger.info(
            "Train-split pChEMBL stats: mean=%.4f std=%.4f threshold=%.1f delta_norm=%.4f",
            self.pchembl_mean,
            self.pchembl_std,
            self.pchembl_threshold,
            normalized_delta,
        )

    def _compute_split_normalization(self):
        if not self._stage_has_pchembl():
            self.pchembl_mean = 0.0
            self.pchembl_std = 1.0
            self.pchembl_threshold = 6.0
            self.model_config["pchembl_huber_delta"] = self.model._config["pchembl_huber_delta"]
            self.model._config["pchembl_mean"] = self.pchembl_mean
            self.model._config["pchembl_std"] = self.pchembl_std
            self.model._config["pchembl_threshold"] = self.pchembl_threshold
            return

        values = np.asarray(self.train_data["pchembl_value_Median"], dtype=np.float32)
        values = values[~np.isnan(values)]
        if values.size == 0:
            raise ValueError("Training split does not contain valid pChEMBL values.")

        self._set_split_normalization(
            pchembl_mean=float(values.mean()),
            pchembl_std=float(values.std(ddof=0)),
            pchembl_threshold=6.0,
        )

    def _build_train_vectors(self):
        if not self.train_smiles_list:
            self.training_vec = None
            return
        self.logger.info(
            "Generating train-split Morgan fingerprints for %s molecules",
            len(self.train_smiles_list),
        )
        self.training_vec = generate_morgan_fingerprints_parallel(
            smiles=self.train_smiles_list,
            radius=2,
            nBits=1024,
            n_jobs=None,
        )

    def _count_unique_proteins(self, dataset_split) -> int:
        if hasattr(dataset_split, "column_names"):
            if "Target_CHEMBL_ID" in dataset_split.column_names:
                return len(set(str(v) for v in dataset_split["Target_CHEMBL_ID"]))
            if "Target_FASTA" in dataset_split.column_names:
                return len(set(str(v) for v in dataset_split["Target_FASTA"]))
        return 0

    def _count_train_lm_positive_samples(self) -> int:
        stage = self.training_config["training_stage"]
        if stage == "lm_only":
            return int(len(self.train_data))
        if stage == "pchembl_only":
            return 0
        if "train_lm" in getattr(self.train_data, "column_names", []):
            return int(sum(bool(flag) for flag in self.train_data["train_lm"]))
        if "pchembl_value_Median" not in getattr(self.train_data, "column_names", []):
            return 0
        values = np.asarray(self.train_data["pchembl_value_Median"], dtype=np.float32)
        valid_values = values[~np.isnan(values)]
        return int(np.sum(valid_values >= self.pchembl_threshold))

    def _update_dataset_stats(self, total_samples: int):
        self.dataset_stats = {
            "dataset_name": self.dataset_name,
            "dataset_source_path": os.path.abspath(self.selfies_path),
            "dataset_total_samples": int(total_samples),
            "train_samples": int(len(self.train_data)),
            "eval_samples": int(len(self.test_data)),
            "train_lm_positive_samples": self._count_train_lm_positive_samples(),
            "train_unique_proteins": self._count_unique_proteins(self.train_data),
            "eval_unique_proteins": self._count_unique_proteins(self.test_data),
            "train_unique_molecules": len(set(self.train_smiles_list)),
            "eval_unique_molecules": len(set(self.eval_reference_smiles)),
            "eval_split": self.training_config["eval_split"],
            "eval_split_ratio": float(self.training_config["eval_split_ratio"]),
            "split_seed": int(self.training_config["split_seed"]),
            "pchembl_mean": float(self.pchembl_mean),
            "pchembl_std": float(self.pchembl_std),
            "pchembl_threshold": float(self.pchembl_threshold),
        }
        self.model._config.update(self.dataset_stats)
        self.model_config.update(self.dataset_stats)

        self.logger.info(
            "Dataset stats: total=%s train=%s eval=%s train_unique_proteins=%s eval_unique_proteins=%s "
            "train_unique_molecules=%s eval_unique_molecules=%s train_lm_positive=%s",
            self.dataset_stats["dataset_total_samples"],
            self.dataset_stats["train_samples"],
            self.dataset_stats["eval_samples"],
            self.dataset_stats["train_unique_proteins"],
            self.dataset_stats["eval_unique_proteins"],
            self.dataset_stats["train_unique_molecules"],
            self.dataset_stats["eval_unique_molecules"],
            self.dataset_stats["train_lm_positive_samples"],
        )

    def _init_tokenizers(self):
        """Initialize tokenizers for proteins and molecules."""
        self.logger.info("Initializing tokenizers...")
        self.mol_tokenizer = load_molgen_tokenizer(padding_side="left")
        self.prot_tokenizer = get_protein_tokenizer(self.model_config["prot_emb_model"])

    def _init_models(self):
        """Initialize the unified Prot2Mol model."""
        self.logger.info("Initializing unified Prot2Mol model...")
        model_config_with_tokenizer = self.model_config.copy()
        model_config_with_tokenizer["mol_tokenizer"] = self.mol_tokenizer
        self.model = create_prot2mol_model(model_config_with_tokenizer)

    def _load_pretrained_weights(self, pretrained_model_path):
        """Load pretrained model weights for fine-tuning only."""
        import glob

        self.logger.info("Loading pretrained weights from: %s", pretrained_model_path)

        if os.path.exists(os.path.join(pretrained_model_path, "pytorch_model.bin")):
            model_file = os.path.join(pretrained_model_path, "pytorch_model.bin")
        elif os.path.exists(os.path.join(pretrained_model_path, "model.safetensors")):
            model_file = os.path.join(pretrained_model_path, "model.safetensors")
        else:
            checkpoint_pattern = os.path.join(pretrained_model_path, "checkpoint-*", "pytorch_model.bin")
            checkpoints = glob.glob(checkpoint_pattern)
            if not checkpoints:
                checkpoint_pattern = os.path.join(pretrained_model_path, "checkpoint-*", "model.safetensors")
                checkpoints = glob.glob(checkpoint_pattern)

            if not checkpoints:
                if os.path.exists(os.path.join(pretrained_model_path, "pytorch_model.bin")):
                    model_file = os.path.join(pretrained_model_path, "pytorch_model.bin")
                else:
                    raise FileNotFoundError(
                        f"Could not find model weights in {pretrained_model_path}. "
                        "Expected pytorch_model.bin or model.safetensors"
                    )
            else:
                model_file = sorted(checkpoints)[-1]

        self.logger.info("Loading model weights from: %s", model_file)

        if model_file.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(model_file)
        else:
            state_dict = torch.load(model_file, map_location="cpu")

        saved_model_config = load_saved_model_config(pretrained_model_path, logger=self.logger)
        state_dict = filter_legacy_pchembl_head_state(state_dict, saved_model_config, logger=self.logger)

        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            self.logger.warning("Missing keys when loading pretrained weights: %s", missing_keys)
        if unexpected_keys:
            self.logger.warning("Unexpected keys when loading pretrained weights: %s", unexpected_keys)

        self.logger.info("Successfully loaded pretrained weights!")
        self.logger.info("Training will start from epoch 0 with fresh optimizer state.")

    def _prepare_datasets(self):
        cache_dir = os.environ.get("DATASETS_CACHE_DIR", "/gpfs/projects/etur29/atabey/datasets")
        processed_data_path = get_processed_data_path(self.selfies_path, cache_dir=cache_dir)

        if not os.path.exists(processed_data_path):
            error_msg = (
                f"\n{'=' * 80}\n"
                "ERROR: Preprocessed dataset not found!\n"
                f"{'=' * 80}\n"
                f"Expected location: {processed_data_path}\n\n"
                "Please run preprocessing BEFORE training:\n\n"
                "  sbatch preprocess_job.sh\n\n"
                "Or manually:\n\n"
                "  python preprocess_dataset.py \\\n"
                f"    --selfies_path {self.selfies_path} \\\n"
                f"    --prot_emb_model {self.model_config['prot_emb_model']} \\\n"
                f"    --max_mol_len {self.model_config['max_mol_len']} \\\n"
                f"    --prot_max_length {self.model_config['prot_max_length']}\n\n"
                "After preprocessing completes, rerun training.\n"
                f"{'=' * 80}\n"
            )
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        self.logger.info(
            "Rank %s: Loading pre-processed dataset from %s",
            self.global_rank,
            processed_data_path,
        )
        dataset, _ = load_processed_dataset(self.selfies_path, cache_dir=cache_dir)
        split_ratio = self.training_config.get("eval_split_ratio", 0.01)
        split_mode = self.training_config.get("eval_split", "random")
        split_seed = self.training_config.get("split_seed", 42)
        stats = load_processed_stats(self.selfies_path, cache_dir=cache_dir)
        if has_matching_precomputed_split(dataset, stats, split_mode=split_mode, split_ratio=split_ratio, split_seed=split_seed):
            self.logger.info("Using precomputed train/test splits from cache.")
            self.train_data = attach_metric_group_ids(dataset["train"])
            self.test_data = attach_metric_group_ids(dataset["test"])
            self._set_split_normalization(
                pchembl_mean=stats["pchembl_mean"],
                pchembl_std=stats["pchembl_std"],
                pchembl_threshold=stats.get("pchembl_threshold", 6.0),
            )
            total_samples = int(stats.get("dataset_total_samples", len(self.train_data) + len(self.test_data)))
        elif stats is not None and stats.get("split_preprocessed") and hasattr(dataset, "keys") and "test" in dataset:
            raise ValueError(
                "Cached dataset contains precomputed train/test splits that do not match the requested "
                f"split configuration (cached: mode={stats.get('eval_split')} ratio={stats.get('eval_split_ratio')} "
                f"seed={stats.get('split_seed')}; requested: mode={split_mode} ratio={split_ratio} seed={split_seed}). "
                "Re-run preprocessing with the requested split settings."
            )
        else:
            self.logger.info("Splitting dataset into train and test sets from legacy full cache...")
            self.logger.info("Legacy cache detected; re-run preprocessing to avoid startup split materialization.")
            full_data = dataset["train"]
            self.train_data, self.test_data = split_train_eval_dataset(
                full_data=full_data,
                split_mode=split_mode,
                split_ratio=split_ratio,
                split_seed=split_seed,
                num_proc=self.training_config.get("dataloader_num_workers"),
                logger=self.logger,
            )
            self._compute_split_normalization()
            self.train_data = self._apply_split_targets(self.train_data)
            self.test_data = self._apply_split_targets(self.test_data)
            total_samples = len(full_data)

        self.logger.info(
            "Dataset split: %s train, %s test samples",
            len(self.train_data),
            len(self.test_data),
        )

        if self.global_rank == 0:
            self.logger.info("Caching canonical SMILES for metrics...")
            self.train_smiles_list = self._extract_smiles_list(self.train_data, drop_invalid=True)
            self.eval_reference_smiles = self._extract_smiles_list(self.test_data, drop_invalid=True)

            if not self.train_smiles_list:
                self.logger.warning("Training data does not contain valid SMILES entries")
            if not self.eval_reference_smiles:
                self.logger.warning("Evaluation data does not contain valid SMILES entries")
            self._build_train_vectors()
        else:
            self.train_smiles_list = []
            self.eval_reference_smiles = []
            self.training_vec = None

        self._update_dataset_stats(total_samples=total_samples)

    def ddp_setup(self):
        """Initialize DDP with proper error handling and device setup."""
        self.training_runner.ddp_setup()

    def preprocess_logits_for_metrics(self, logits, labels):
        return preprocess_logits_for_metrics_fn(logits, labels, logger=self.logger)

    def compute_generation_eval_metrics(self, eval_dataset=None):
        """Compute real generation metrics on the evaluation proteins."""
        if not self._stage_has_lm():
            return {}
        if self.global_rank != 0:
            return {}

        target_dataset = eval_dataset or self.test_data
        if target_dataset is None or len(target_dataset) == 0:
            return {}
        if "Target_FASTA" not in target_dataset.column_names:
            self.logger.warning("Evaluation dataset missing Target_FASTA; skipping generation metrics.")
            return {}

        model = getattr(self.trainer.model, "module", self.trainer.model)
        model_device = next(model.parameters()).device
        sequences = list(target_dataset["Target_FASTA"])
        sequence_counts = Counter(str(sequence) for sequence in sequences)
        generated_batches = []
        batch_size = self.training_config["valid_batch_size"]
        generation_kwargs = {
            "max_length": self.model_config["max_mol_len"],
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.9,
            "pad_token_id": 1,
            "bos_token_id": 1,
            "eos_token_id": self.mol_tokenizer.eos_token_id,
        }

        cuda_devices = [model_device.index] if model_device.type == "cuda" and model_device.index is not None else []
        with torch.no_grad():
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(self.training_config["split_seed"])
                for sequence, count in sequence_counts.items():
                    prot_input_ids, prot_attention_mask = tokenize_protein_sequences_for_inference(
                        sequences=[sequence],
                        prot_tokenizer=self.prot_tokenizer,
                        prot_emb_model=self.model_config["prot_emb_model"],
                        prot_max_length=self.model_config["prot_max_length"],
                        device=model_device,
                    )
                    protein_embeddings = model.encode_protein(prot_input_ids, prot_attention_mask)
                    remaining = count
                    while remaining > 0:
                        current_batch = min(batch_size, remaining)
                        generated = model.generate_from_protein_embeddings(
                            protein_embeddings=protein_embeddings.repeat(current_batch, 1, 1),
                            prot_attention_mask=prot_attention_mask.repeat(current_batch, 1),
                            **generation_kwargs,
                        )
                        generated_batches.append(generated.detach().cpu())
                        remaining -= current_batch

        if not generated_batches:
            return {}

        generated_token_ids = torch.cat(generated_batches, dim=0).numpy()
        metrics = compute_generation_metrics(
            generated_token_ids=generated_token_ids,
            mol_tokenizer=self.mol_tokenizer,
            eval_reference_smiles=self.eval_reference_smiles,
            train_smiles_list=self.train_smiles_list,
            training_vec=self.training_vec,
            logger=self.logger,
        )
        metrics["gen_count"] = int(generated_token_ids.shape[0])
        return metrics

    def _compute_pchembl_metrics(self, pchembl_predictions, pchembl_targets, group_ids=None):
        return compute_pchembl_metrics(
            pchembl_predictions=pchembl_predictions,
            pchembl_targets=pchembl_targets,
            pchembl_mean=self.pchembl_mean,
            pchembl_std=self.pchembl_std,
            group_ids=group_ids,
            inputs_are_normalized=True,
            logger=self.logger,
        )

    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics for pChEMBL prediction."""
        try:
            metrics = {}

            if self.trainer is not None:
                pchembl_preds, pchembl_targets, group_ids = self.trainer.get_pchembl_predictions()
                if pchembl_preds is not None and pchembl_targets is not None:
                    pchembl_metrics = self._compute_pchembl_metrics(pchembl_preds, pchembl_targets, group_ids)
                    metrics.update(pchembl_metrics)
                    self.trainer.clear_pchembl_predictions()

        except Exception as exc:
            self.logger.error("Rank %s: Error computing metrics: %s", self.global_rank, str(exc))
            import traceback

            self.logger.error("Traceback: %s", traceback.format_exc())
            metrics = {}

        return metrics

    def model_training(self):
        """Execute the model training process."""
        self.logger.info("Starting training process for run: %s", self.run_name)

        try:
            self.model.update_trainable_components(
                trainable_encoder=self.model_config["train_encoder_model"],
                trainable_decoder=self.model_config["train_decoder_model"],
                trainable_pchembl_head=self.model_config["train_pchembl_head"],
            )

            if self.training_config["load_pretrained_model"]:
                self._load_pretrained_weights(self.training_config["load_pretrained_model"])

            self._prepare_datasets()

            if self.global_rank == 0:
                try:
                    wandb.init(
                        project="prot2mol",
                        name=self.run_name,
                        config={
                            **self.model_config,
                            **self.training_config,
                            "dataset_name": self.run_name.split("_")[0],
                            "global_rank": self.global_rank,
                            "local_rank": self.local_rank,
                            "world_size": self.distributed_context.world_size,
                            "effective_training_mode": self.distributed_context.effective_mode,
                        },
                    )
                    self.logger.info("Wandb initialized successfully on rank 0")
                except Exception as exc:
                    self.logger.warning("Failed to initialize wandb on rank 0: %s", exc)

            self.model.train()
            self.trainer = self.training_runner.create_trainer(
                model=self.model,
                train_dataset=self.train_data,
                eval_dataset=self.test_data,
                compute_metrics=self.compute_metrics,
                compute_generation_metrics=self.compute_generation_eval_metrics,
                preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
                run_name=self.run_name,
                output_dir=self.pretrain_save_to,
                training_config=self.training_config,
                model_config=self.model_config,
            )
            eval_results = self.training_runner.run(
                trainer=self.trainer,
                training_config=self.training_config,
                output_dir=self.pretrain_save_to,
            )

            if self.global_rank == 0:
                try:
                    wandb.finish()
                    self.logger.info("Wandb session finished successfully")
                except Exception as exc:
                    self.logger.warning("Failed to finish wandb session: %s", exc)

            return eval_results

        except Exception as exc:
            self.logger.error("Error during model training: %s", str(exc), exc_info=True)
            if self.global_rank == 0:
                try:
                    wandb.finish()
                    self.logger.info("Wandb session finished due to error")
                except Exception as wandb_error:
                    self.logger.warning(
                        "Failed to finish wandb session during error cleanup: %s",
                        wandb_error,
                    )
            raise


def main():
    """Main entry point for training."""
    config = parse_arguments()

    setup_logging(config.log_level)
    logger = logging.getLogger(__name__)
    logger.info("Starting Prot2Mol training")
    distributed_context = resolve_distributed_context(config.training_mode)
    logger.info("Distributed context: %s", distributed_context.to_log_fields())

    dataset_name = validate_and_process_paths(config)
    run_name = create_run_name(config, dataset_name)

    save_dir = os.path.join(config.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    trainer = TrainingScript(
        config=config,
        selfies_path=config.selfies_path,
        pretrain_save_to=save_dir,
        dataset_name=dataset_name,
        run_name=run_name,
        distributed_context=distributed_context,
    )
    try:
        if distributed_context.is_distributed:
            trainer.ddp_setup()
        else:
            logger.info("Running non-distributed training mode: %s", distributed_context.effective_mode)
        trainer.model_training()
    finally:
        if (
            distributed_context.is_distributed
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            destroy_process_group()


if __name__ == "__main__":
    main()
