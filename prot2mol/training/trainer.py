"""Hugging Face Trainer integration for Prot2Mol."""

from __future__ import annotations

import math
import os
import time

import torch
from transformers import Trainer
from transformers.trainer import TRAINING_ARGS_NAME, WEIGHTS_NAME

from ..io.hf_utils import save_model_config


MODEL_INPUT_KEYS = (
    "prot_input_ids",
    "prot_attention_mask",
    "labels",
)


def generation_data_collator(features):
    """Collate tensor fields and deliberately ignore raw provenance strings."""

    batch = {}
    for key in MODEL_INPUT_KEYS:
        values = [feature[key] for feature in features]
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values)
        else:
            batch[key] = torch.tensor(values, dtype=torch.long)
    return batch


class Prot2MolTrainer(Trainer):
    """Loss, checkpoint, perplexity, and bounded final-generation handling."""

    def __init__(self, *args, compute_generation_metrics=None, **kwargs):
        kwargs.setdefault("data_collator", generation_data_collator)
        super().__init__(*args, **kwargs)
        self.compute_generation_metrics = compute_generation_metrics
        self._lm_loss_sum = 0.0
        self._lm_loss_count = 0

    @staticmethod
    def _build_model_inputs(inputs):
        return {key: inputs[key] for key in MODEL_INPUT_KEYS}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**self._build_model_inputs(inputs))
        loss = outputs.get("loss")
        if loss is None:
            raise RuntimeError("Prot2Mol did not return a language-modeling loss")
        if model.training:
            self._lm_loss_sum += float(loss.detach().cpu())
            self._lm_loss_count += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        logs = dict(logs)
        if "loss" in logs and self._lm_loss_count:
            logs["lm_loss"] = self._lm_loss_sum / self._lm_loss_count
            self._lm_loss_sum = 0.0
            self._lm_loss_count = 0
        try:
            return super().log(logs, start_time=start_time)
        except TypeError:
            return super().log(logs)

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
        run_generation_metrics: bool = False,
    ):
        started = time.perf_counter()
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        loss = metrics.get(f"{metric_key_prefix}_loss")
        if loss is not None:
            metrics[f"{metric_key_prefix}_perplexity"] = (
                float(math.exp(loss)) if loss < 100 else float("inf")
            )

        if run_generation_metrics and self.compute_generation_metrics is not None:
            distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
            if distributed:
                torch.distributed.barrier()
            generated = self.compute_generation_metrics()
            if distributed:
                torch.distributed.barrier()
            metrics.update(
                {f"{metric_key_prefix}_{key}": value for key, value in generated.items()}
            )

        metrics[f"{metric_key_prefix}_evaluation_time_sec"] = time.perf_counter() - started
        self.log({key: value for key, value in metrics.items() if key.startswith(f"{metric_key_prefix}_")})
        return metrics

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        destination = output_dir or self.args.output_dir
        model = getattr(self.model, "module", self.model)
        model_config = getattr(model, "_config", None)
        if model_config is not None and self.is_world_process_zero():
            save_model_config(destination, model_config)

    def _save(self, output_dir=None, state_dict=None):
        """Save shared BART weights safely for this non-PreTrainedModel wrapper."""
        destination = output_dir or self.args.output_dir
        os.makedirs(destination, exist_ok=True)
        state_dict = state_dict or self.model.state_dict()
        torch.save(state_dict, os.path.join(destination, WEIGHTS_NAME))
        torch.save(self.args, os.path.join(destination, TRAINING_ARGS_NAME))
