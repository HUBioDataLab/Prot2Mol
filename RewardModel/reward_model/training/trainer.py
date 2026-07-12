from __future__ import annotations

import inspect
import os
from typing import Any, Dict, Optional

import torch
from transformers import Trainer, TrainingArguments
from transformers.trainer import TRAINING_ARGS_NAME

from ..model import save_reward_model
from .config import RewardTrainerConfig
from .data import RewardPairDataset
from .evaluation import compute_reward_model_eval_metrics


REQUIRED_DISTRIBUTED_ENV_VARS = (
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
)


def _safe_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _missing_env_vars(names):
    return [name for name in names if os.environ.get(name) in (None, "")]


def _validate_training_mode_environment(config: RewardTrainerConfig) -> None:
    mode = config.training_mode
    world_size = max(1, _safe_int(os.environ.get("WORLD_SIZE"), 1))
    default_local_world_size = world_size if world_size > 1 else 1
    local_world_size = max(
        1,
        _safe_int(os.environ.get("LOCAL_WORLD_SIZE"), default_local_world_size),
    )
    global_rank = _safe_int(os.environ.get("RANK"), -1)
    local_rank = _safe_int(os.environ.get("LOCAL_RANK"), -1)
    requires_distributed_env = mode in {"multi_gpu", "multi_node"} or (
        mode == "auto" and world_size > 1
    )
    if requires_distributed_env:
        missing = _missing_env_vars(REQUIRED_DISTRIBUTED_ENV_VARS)
        if missing:
            raise ValueError(
                "Distributed launch environment is incomplete; missing "
                f"{', '.join(missing)}. Use torchrun/sbatch launcher wiring or "
                "run with training_mode=single_gpu."
            )
        if global_rank < 0 or global_rank >= world_size:
            raise ValueError(
                f"Invalid distributed rank: RANK={global_rank}, WORLD_SIZE={world_size}"
            )
        if local_rank < 0 or local_rank >= local_world_size:
            raise ValueError(
                f"Invalid local rank: LOCAL_RANK={local_rank}, "
                f"LOCAL_WORLD_SIZE={local_world_size}"
            )

    if mode == "single_gpu" and world_size > 1:
        raise ValueError(
            "training_mode=single_gpu requires WORLD_SIZE=1. "
            "Unset distributed launcher variables or choose multi_gpu/multi_node."
        )
    if mode == "multi_gpu":
        if world_size <= 1:
            raise ValueError("training_mode=multi_gpu requires WORLD_SIZE>1")
        if world_size != local_world_size:
            raise ValueError(
                "training_mode=multi_gpu expects all ranks on one node. "
                "Use multi_node for multi-node launches."
            )
    if mode == "multi_node" and world_size <= local_world_size:
        raise ValueError("training_mode=multi_node requires more than one node")


class RewardModelTrainer(Trainer):
    def __init__(self, *args, val2_eval_dataset=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.val2_eval_dataset = val2_eval_dataset
        self._reset_train_component_accumulator()
        self._reset_eval_component_accumulator()

    def _reset_train_component_accumulator(self) -> None:
        self._train_component_sums = {
            "pair_loss": 0.0,
            "classification_loss": 0.0,
            "total_loss": 0.0,
            "num_pairs": 0.0,
        }
        self._train_component_count = 0

    def _reset_eval_component_accumulator(self) -> None:
        self._eval_component_sums = {
            "pair_loss": 0.0,
            "classification_loss": 0.0,
            "total_loss": 0.0,
            "num_pairs": 0.0,
        }
        self._eval_component_count = 0

    @staticmethod
    def _to_scalar(value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, torch.Tensor):
            return float(value.detach().float().mean().cpu().item())
        return float(value)

    def _record_train_components(
        self,
        pair_loss: Optional[torch.Tensor],
        classification_loss: Optional[torch.Tensor],
        total_loss: torch.Tensor,
        num_pairs: Any,
    ) -> None:
        self._train_component_sums["pair_loss"] += self._to_scalar(pair_loss)
        self._train_component_sums["classification_loss"] += self._to_scalar(classification_loss)
        self._train_component_sums["total_loss"] += self._to_scalar(total_loss)
        self._train_component_sums["num_pairs"] += self._to_scalar(num_pairs)
        self._train_component_count += 1

    def _record_eval_components(
        self,
        pair_loss: Optional[torch.Tensor],
        classification_loss: Optional[torch.Tensor],
        total_loss: torch.Tensor,
        num_pairs: Any,
    ) -> None:
        self._eval_component_sums["pair_loss"] += self._to_scalar(pair_loss)
        self._eval_component_sums["classification_loss"] += self._to_scalar(classification_loss)
        self._eval_component_sums["total_loss"] += self._to_scalar(total_loss)
        self._eval_component_sums["num_pairs"] += self._to_scalar(num_pairs)
        self._eval_component_count += 1

    def _consume_train_component_logs(self) -> Dict[str, float]:
        if self._train_component_count == 0:
            return {}
        denom = float(self._train_component_count)
        logs = {
            "pair_loss": self._train_component_sums["pair_loss"] / denom,
            "classification_loss": self._train_component_sums["classification_loss"] / denom,
            "total_loss": self._train_component_sums["total_loss"] / denom,
            "num_pairs": self._train_component_sums["num_pairs"] / denom,
        }
        self._reset_train_component_accumulator()
        return logs

    def _consume_eval_component_logs(self, metric_key_prefix: str) -> Dict[str, float]:
        if self._eval_component_count == 0:
            return {}
        denom = float(self._eval_component_count)
        logs = {
            f"{metric_key_prefix}_pair_loss": self._eval_component_sums["pair_loss"] / denom,
            f"{metric_key_prefix}_classification_loss": self._eval_component_sums["classification_loss"] / denom,
            f"{metric_key_prefix}_total_loss": self._eval_component_sums["total_loss"] / denom,
            f"{metric_key_prefix}_num_pairs": self._eval_component_sums["num_pairs"] / denom,
        }
        self._reset_eval_component_accumulator()
        return logs

    def _build_model_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "protein_input_ids": inputs["protein_input_ids"],
            "protein_attention_mask": inputs["protein_attention_mask"],
            "molecule_input_ids": inputs["molecule_input_ids"],
            "molecule_attention_mask": inputs["molecule_attention_mask"],
            "activity_labels": inputs["activity_labels"],
            "positive_indices": inputs["positive_indices"],
            "negative_indices": inputs["negative_indices"],
            "return_dict": True,
        }

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
        model_inputs = self._build_model_inputs(inputs)
        outputs = model(**model_inputs)
        if outputs.loss is None:
            raise RuntimeError("RewardModel did not return a scalar loss")

        if getattr(model, "training", False):
            self._record_train_components(
                pair_loss=outputs.pair_loss,
                classification_loss=outputs.classification_loss,
                total_loss=outputs.loss,
                num_pairs=inputs.get("num_pairs"),
            )

        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            model_inputs = self._build_model_inputs(inputs)
            outputs = model(**model_inputs)

        if outputs.loss is None:
            raise RuntimeError("RewardModel did not return a scalar loss during evaluation")

        loss = outputs.loss.detach()
        self._record_eval_components(
            pair_loss=outputs.pair_loss,
            classification_loss=outputs.classification_loss,
            total_loss=outputs.loss,
            num_pairs=inputs.get("num_pairs"),
        )

        if prediction_loss_only or self.compute_metrics is None:
            return (loss, None, None)

        logits = outputs.activity_logits.detach()
        labels = inputs["activity_labels"].detach()
        return (loss, logits, labels)

    def _evaluate_reward_dataset(
        self,
        eval_dataset,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        self._reset_eval_component_accumulator()
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        active_eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if not isinstance(active_eval_dataset, RewardPairDataset):
            raise TypeError("RewardModelTrainer expects eval_dataset to be a RewardPairDataset")
        component_metrics = self._consume_eval_component_logs(metric_key_prefix)
        reward_metrics = compute_reward_model_eval_metrics(
            self,
            self.model,
            active_eval_dataset,
            metric_key_prefix=metric_key_prefix,
        )
        metrics.update(component_metrics)
        metrics.update(reward_metrics)
        extra_metrics = {**component_metrics, **reward_metrics}
        if extra_metrics:
            self.log(extra_metrics)
        return metrics

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        metrics = self._evaluate_reward_dataset(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        if eval_dataset is None and metric_key_prefix == "eval" and self.val2_eval_dataset is not None:
            metrics.update(
                self._evaluate_reward_dataset(
                    eval_dataset=self.val2_eval_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix="eval_val2",
                )
            )
        return metrics

    def log(self, logs, start_time=None):
        logs = dict(logs)
        if "loss" in logs:
            logs.update(self._consume_train_component_logs())
        return super().log(logs, start_time=start_time)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if not self.args.should_save:
            return

        save_dir = output_dir or self.args.output_dir
        os.makedirs(save_dir, exist_ok=True)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        save_reward_model(model_to_save, save_dir)
        torch.save(self.args, os.path.join(save_dir, TRAINING_ARGS_NAME))

def create_training_arguments(config: RewardTrainerConfig) -> TrainingArguments:
    _validate_training_mode_environment(config)

    args_kwargs = dict(
        output_dir=os.path.abspath(config.output_dir),
        run_name=os.path.basename(os.path.abspath(config.output_dir)),
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        dataloader_num_workers=config.dataloader_num_workers,
        seed=config.seed,
        fp16=config.fp16,
        save_total_limit=config.save_total_limit,
        remove_unused_columns=False,
        disable_tqdm=True,
        report_to=config.report_to,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    schedule_strategy = "steps" if config.eval_steps is not None else "epoch"
    args_kwargs["save_strategy"] = schedule_strategy
    if config.eval_steps is not None:
        args_kwargs["eval_steps"] = config.eval_steps
        args_kwargs["save_steps"] = config.eval_steps

    init_params = inspect.signature(TrainingArguments.__init__).parameters
    args_kwargs = {
        key: value
        for key, value in args_kwargs.items()
        if key in init_params
    }
    if "evaluation_strategy" in init_params:
        args_kwargs["evaluation_strategy"] = schedule_strategy
    elif "eval_strategy" in init_params:
        args_kwargs["eval_strategy"] = schedule_strategy
    try:
        return TrainingArguments(**args_kwargs)
    except ImportError as exc:
        raise ImportError(
            "RewardModel training via Hugging Face Trainer requires the 'accelerate' package. "
            "Install accelerate>=1.1.0 in the reward_model environment."
        ) from exc
