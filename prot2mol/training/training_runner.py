"""TrainingArguments and lifecycle orchestration."""

from __future__ import annotations

import inspect
import os

import torch
from torch.distributed import init_process_group
from transformers import TrainingArguments

from .distributed import REQUIRED_DISTRIBUTED_ENV_VARS
from .trainer import Prot2MolTrainer


class TrainingRunner:
    def __init__(self, local_rank: int, global_rank: int, logger=None):
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.logger = logger

    def ddp_setup(self):
        missing = [name for name in REQUIRED_DISTRIBUTED_ENV_VARS if not os.environ.get(name)]
        if missing:
            raise RuntimeError(f"Incomplete distributed environment: {', '.join(missing)}")
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed Prot2Mol training requires CUDA")
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(self.local_rank)
        init_process_group(backend="nccl", rank=self.global_rank, world_size=world_size)

    @staticmethod
    def _precision_flags(precision: str) -> tuple[bool, bool]:
        if precision == "fp32":
            return False, False
        if precision == "bf16":
            return True, False
        if precision == "fp16":
            return False, True
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return True, False
        if torch.cuda.is_available():
            return False, True
        return False, False

    def create_trainer(
        self,
        model,
        train_dataset,
        eval_dataset,
        compute_generation_metrics,
        run_name: str,
        output_dir: str,
        training_config: dict,
    ):
        return Prot2MolTrainer(
            model=model,
            args=self._create_training_args(run_name, output_dir, training_config),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_generation_metrics=compute_generation_metrics,
        )

    def run(self, trainer, training_config: dict, output_dir: str):
        checkpoint = training_config.get("resume_from_checkpoint")
        trainer.train(resume_from_checkpoint=checkpoint or None)
        # Trainer restores the best checkpoint before train() returns. Evaluate
        # that exact model, then run the bounded autoregressive panel once.
        metrics = trainer.evaluate(run_generation_metrics=True)
        trainer.save_model(output_dir)
        return metrics

    def _create_training_args(self, run_name: str, output_dir: str, config: dict):
        bf16, fp16 = self._precision_flags(config.get("precision", "auto"))
        kwargs = {
            "run_name": run_name,
            "output_dir": output_dir,
            "overwrite_output_dir": config.get("resume_from_checkpoint") is None,
            "save_strategy": "epoch",
            "num_train_epochs": config["epochs"],
            "learning_rate": config["learning_rate"],
            "weight_decay": config["weight_decay"],
            "per_device_train_batch_size": config["train_batch_size"],
            "per_device_eval_batch_size": config["valid_batch_size"],
            "gradient_accumulation_steps": config["gradient_accumulation_steps"],
            "max_grad_norm": config["max_grad_norm"],
            "save_total_limit": 2,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "logging_steps": config["logging_steps"],
            "dataloader_num_workers": config["dataloader_num_workers"],
            "bf16": bf16,
            "fp16": fp16,
            "remove_unused_columns": False,
            "report_to": ["wandb"],
        }
        distributed = self.local_rank != -1 or int(os.environ.get("WORLD_SIZE", "1")) > 1
        if distributed:
            kwargs.update(
                {
                    "local_rank": self.local_rank,
                    "ddp_backend": "nccl",
                    "ddp_find_unused_parameters": False,
                }
            )
        parameters = inspect.signature(TrainingArguments.__init__).parameters
        kwargs = {key: value for key, value in kwargs.items() if key in parameters}
        kwargs["eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"] = "epoch"
        return TrainingArguments(**kwargs)
