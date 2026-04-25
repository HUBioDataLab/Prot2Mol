import os
import inspect

import torch
from torch.distributed import init_process_group
from transformers import TrainingArguments

from ..io.hf_utils import save_model_config
from .trainer import GPT2_w_crs_attn_Trainer


class TrainingRunner:
    """Orchestrate DDP setup and HuggingFace Trainer lifecycle for training runs."""

    def __init__(self, local_rank: int, global_rank: int, logger=None):
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.logger = logger

    def ddp_setup(self):
        """Initialize DDP with explicit rank/device validation."""
        try:
            world_size = int(os.environ["WORLD_SIZE"])
            torch.cuda.set_device(self.local_rank)
            init_process_group(backend="nccl", rank=self.global_rank, world_size=world_size)

            if self.logger is not None:
                self.logger.info(
                    "Initialized DDP with rank %s/%s on device %s",
                    self.global_rank,
                    world_size,
                    self.local_rank,
                )

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available but DDP is being initialized")

            device = torch.cuda.current_device()
            memory_total = torch.cuda.get_device_properties(device).total_memory / 1e9
            if self.logger is not None:
                self.logger.info(
                    "Rank %s: Using GPU %s with %.1fGB memory",
                    self.global_rank,
                    device,
                    memory_total,
                )
        except Exception as exc:
            if self.logger is not None:
                self.logger.error("Failed to initialize DDP: %s", exc)
            raise

    def create_trainer(
        self,
        model,
        train_dataset,
        eval_dataset,
        compute_metrics,
        compute_generation_metrics,
        preprocess_logits_for_metrics,
        run_name: str,
        output_dir: str,
        training_config: dict,
        model_config: dict,
    ):
        """Create configured HF Trainer for current run."""
        training_args = self._create_training_args(run_name, output_dir, training_config)
        training_stage = training_config.get("training_stage", model_config.get("training_stage", "multitask"))

        if self.logger is not None:
            self.logger.info("Training stage: %s", training_stage)
            self.logger.info("Initializing trainer...")

        trainer = GPT2_w_crs_attn_Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            compute_generation_metrics=compute_generation_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            training_stage=training_stage,
            ignore_mismatched_optimizer=training_config.get("ignore_mismatched_optimizer", False),
        )

        if self.logger is not None:
            self.logger.info(
                "Building trainer on device: %s with %s GPUs",
                training_args.device,
                training_args.n_gpu,
            )

        return trainer

    def run(self, trainer, training_config: dict, output_dir: str):
        """Execute training and model save."""
        resume_from_checkpoint = training_config.get("resume_from_checkpoint")
        if resume_from_checkpoint:
            if self.logger is not None:
                self.logger.info("Resuming training from checkpoint: %s", resume_from_checkpoint)
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            if self.logger is not None:
                self.logger.info("Starting training from scratch...")
            trainer.train()

        if self.logger is not None:
            self.logger.info("Training finished successfully")

        if self.logger is not None:
            self.logger.info("Saving model to %s", output_dir)
        trainer.save_model(output_dir)
        model_holder = getattr(trainer.model, "module", trainer.model)
        if model_holder is None:
            model_wrapped = getattr(trainer, "model_wrapped", None)
            model_holder = getattr(model_wrapped, "module", model_wrapped)
        model_config = getattr(model_holder, "_config", None)
        should_save_config = getattr(trainer, "is_world_process_zero", lambda: True)()
        if should_save_config and model_config is not None:
            save_model_config(output_dir, model_config, logger=self.logger)
        if self.logger is not None:
            self.logger.info("Model saved successfully")

        eval_results = {}
        for entry in reversed(trainer.state.log_history):
            entry_eval = {key: value for key, value in entry.items() if key.startswith("eval_")}
            if entry_eval:
                eval_results.update(entry_eval)
            elif eval_results:
                break
        return eval_results

    def _create_training_args(self, run_name: str, output_dir: str, training_config: dict):
        overwrite_output = training_config.get("resume_from_checkpoint") is None
        args_kwargs = dict(
            run_name=run_name,
            output_dir=output_dir,
            overwrite_output_dir=overwrite_output,
            save_strategy="epoch",
            num_train_epochs=training_config["epochs"],
            learning_rate=training_config["learning_rate"],
            weight_decay=training_config["weight_decay"],
            per_device_train_batch_size=training_config["train_batch_size"],
            per_device_eval_batch_size=training_config["valid_batch_size"],
            gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            disable_tqdm=True,
            logging_steps=1,
            dataloader_num_workers=training_config["dataloader_num_workers"],
            fp16=True,
            remove_unused_columns=False,
            include_inputs_for_metrics=False,
            save_safetensors=False,
        )
        is_distributed = self.local_rank != -1 or int(os.environ.get("WORLD_SIZE", "1")) > 1
        if is_distributed:
            args_kwargs["local_rank"] = self.local_rank
            args_kwargs["ddp_backend"] = "nccl"
            args_kwargs["ddp_find_unused_parameters"] = True
        init_params = inspect.signature(TrainingArguments.__init__).parameters
        if "evaluation_strategy" in init_params:
            args_kwargs["evaluation_strategy"] = "epoch"
        else:
            args_kwargs["eval_strategy"] = "epoch"
        return TrainingArguments(**args_kwargs)
