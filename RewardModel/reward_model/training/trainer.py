from __future__ import annotations

import inspect
import os
from typing import Any, Dict, Optional

import torch
from transformers import Trainer, TrainingArguments
from transformers.trainer import TRAINING_ARGS_NAME

from ..model import save_reward_model
from .config import RewardTrainerConfig


class RewardModelTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_grad_norm = 0.0
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
            return float(value.detach().mean().cpu().item())
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
            "grad_norm": self._last_grad_norm,
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

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        self._last_grad_norm = self._compute_grad_norm(model)
        return loss

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

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        self._reset_eval_component_accumulator()
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        component_metrics = self._consume_eval_component_logs(metric_key_prefix)
        metrics.update(component_metrics)
        if component_metrics:
            self.log(component_metrics)
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
        save_reward_model(
            model_to_save,
            save_dir,
            use_safetensors=self.args.save_safetensors,
        )
        torch.save(self.args, os.path.join(save_dir, TRAINING_ARGS_NAME))

    @staticmethod
    def _compute_grad_norm(model) -> float:
        total = 0.0
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            total += float(torch.sum(grad * grad).item())
        return total ** 0.5


def create_training_arguments(config: RewardTrainerConfig) -> TrainingArguments:
    args_kwargs = dict(
        output_dir=os.path.abspath(config.output_dir),
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
        save_safetensors=config.save_safetensors,
        save_total_limit=config.save_total_limit,
        remove_unused_columns=False,
        disable_tqdm=True,
        report_to=[],
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    init_params = inspect.signature(TrainingArguments.__init__).parameters
    args_kwargs = {
        key: value
        for key, value in args_kwargs.items()
        if key in init_params
    }
    if "evaluation_strategy" in init_params:
        args_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in init_params:
        args_kwargs["eval_strategy"] = "epoch"
    try:
        return TrainingArguments(**args_kwargs)
    except ImportError as exc:
        raise ImportError(
            "RewardModel training via Hugging Face Trainer requires the 'accelerate' package. "
            "Install accelerate>=1.1.0 in the reward_model environment."
        ) from exc
