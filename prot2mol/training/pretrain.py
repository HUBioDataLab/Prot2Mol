# Standard library imports
import logging
import os
import sys

from torch.distributed import destroy_process_group

# Third-party library imports
import torch
import wandb

# Local application imports
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from prot2mol.data.pipeline import (
    extract_smiles_list,
    get_processed_data_path,
    load_processed_dataset,
    split_train_eval_dataset,
)
from prot2mol.io.hf_utils import load_molgen_tokenizer
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
    compute_lm_metrics,
    compute_pchembl_metrics,
    preprocess_logits_for_metrics as preprocess_logits_for_metrics_fn,
)
from prot2mol.training.normalization_service import NormalizationService
from prot2mol.training.training_runner import TrainingRunner
from prot2mol.training.vector_service import VectorService

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
            "pchembl_huber_delta": config.pchembl_huber_delta,
            "stop_pchembl_gradients": config.stop_pchembl_gradients,
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
            "ignore_mismatched_optimizer": config.ignore_mismatched_optimizer,
            "training_mode": config.training_mode,
            "eval_split": config.eval_split,
            "eval_split_ratio": config.eval_split_ratio,
            "split_seed": config.split_seed,
        }

        self.selfies_path = selfies_path
        self.pretrain_save_to = pretrain_save_to
        self.dataset_name = dataset_name
        self.run_name = run_name
        self.train_smiles_list = []
        self.eval_reference_smiles = []
        self.trainer = None

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

        norm_stats = NormalizationService(
            selfies_path=self.selfies_path,
            train_pchembl_head=self.model_config["train_pchembl_head"],
            pchembl_huber_delta_raw=config.pchembl_huber_delta,
            logger=self.logger,
        ).load_or_compute()
        self.pchembl_mean = norm_stats.pchembl_mean
        self.pchembl_std = norm_stats.pchembl_std
        self.pchembl_threshold = norm_stats.pchembl_threshold
        self.model_config["pchembl_huber_delta"] = norm_stats.pchembl_huber_delta_norm

        self.training_vec = VectorService(selfies_path=self.selfies_path, logger=self.logger).load_or_generate()

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

        self.logger.info("Splitting dataset into train and test sets...")
        full_data = dataset["train"]
        split_ratio = self.training_config.get("eval_split_ratio", 0.01)
        split_mode = self.training_config.get("eval_split", "random")
        self.train_data, self.test_data = split_train_eval_dataset(
            full_data=full_data,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=self.training_config.get("split_seed", 42),
            num_proc=self.training_config.get("dataloader_num_workers"),
            logger=self.logger,
        )

        self.logger.info(
            "Dataset split: %s train, %s test samples",
            len(self.train_data),
            len(self.test_data),
        )

        self.logger.info("Caching canonical SMILES for metrics...")
        self.train_smiles_list = self._extract_smiles_list(self.train_data, drop_invalid=True)
        self.eval_reference_smiles = self._extract_smiles_list(self.test_data, drop_invalid=True)

        if not self.train_smiles_list:
            self.logger.warning("Training data does not contain valid SMILES entries")
        if not self.eval_reference_smiles:
            self.logger.warning("Evaluation data does not contain valid SMILES entries")

    def ddp_setup(self):
        """Initialize DDP with proper error handling and device setup."""
        self.training_runner.ddp_setup()

    def preprocess_logits_for_metrics(self, logits, labels):
        return preprocess_logits_for_metrics_fn(logits, labels, logger=self.logger)

    def _compute_lm_metrics(self, predictions, labels):
        return compute_lm_metrics(
            predictions=predictions,
            labels=labels,
            mol_tokenizer=self.mol_tokenizer,
            eval_reference_smiles=getattr(self, "eval_reference_smiles", None),
            train_smiles_list=self.train_smiles_list,
            training_vec=self.training_vec,
            global_rank=self.global_rank,
            logger=self.logger,
        )

    def _compute_pchembl_metrics(self, pchembl_predictions, pchembl_targets, group_ids=None):
        return compute_pchembl_metrics(
            pchembl_predictions=pchembl_predictions,
            pchembl_targets=pchembl_targets,
            pchembl_mean=self.pchembl_mean,
            pchembl_std=self.pchembl_std,
            group_ids=group_ids,
            logger=self.logger,
        )

    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics for language modeling and pChEMBL prediction."""
        try:
            predictions = eval_pred.predictions
            labels = eval_pred.label_ids
            metrics = {}

            if predictions is None:
                self.logger.warning(
                    "Rank %s: Predictions is None, skipping metrics computation",
                    self.global_rank,
                )
                return {}

            if predictions is not None and labels is not None:
                if hasattr(predictions, "cpu"):
                    predictions_np = predictions.cpu().numpy()
                elif hasattr(predictions, "numpy"):
                    predictions_np = predictions.numpy()
                else:
                    predictions_np = predictions

                lm_metrics = self._compute_lm_metrics(predictions_np, labels)
                if lm_metrics:
                    metrics.update(lm_metrics)
                else:
                    self.logger.info(
                        "Rank %s: No LM metrics (batch had no positive samples)",
                        self.global_rank,
                    )

            if self.trainer is not None:
                pchembl_preds, pchembl_targets, group_ids = self.trainer.get_pchembl_predictions()
                if pchembl_preds is not None and pchembl_targets is not None:
                    if predictions is not None:
                        lm_sample_count = len(predictions_np) if "predictions_np" in locals() else predictions.shape[0]
                        pchembl_sample_count = len(pchembl_preds)
                        if lm_sample_count != pchembl_sample_count:
                            pchembl_preds = pchembl_preds[:lm_sample_count]
                            pchembl_targets = pchembl_targets[:lm_sample_count]
                            if group_ids is not None:
                                group_ids = group_ids[:lm_sample_count]

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
