from __future__ import annotations

import inspect
import os
from array import array
from typing import Any, Dict, Iterator, Optional, Sequence

import torch
from torch.utils.data import Sampler
from transformers import Trainer, TrainingArguments
from transformers.trainer import TRAINING_ARGS_NAME

from ..model import save_reward_model
from .config import RewardTrainerConfig
from .data import RewardAssayListDataset, RewardEvaluationDataset, RewardPairDataset
from .evaluation import (
    _append_assay_spearman_log,
    _with_metric_prefix,
    compute_joint_evaluation_metrics,
)


REQUIRED_DISTRIBUTED_ENV_VARS = (
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
)


class LengthBucketSampler(Sampler[int]):
    """Shuffle globally, then sort bounded pools into length-homogeneous batches."""

    def __init__(
        self,
        lengths: Sequence[int],
        *,
        batch_size: int,
        bucket_size_multiplier: int,
        seed: int,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if bucket_size_multiplier <= 0:
            raise ValueError("bucket_size_multiplier must be > 0")
        self.lengths = (
            lengths
            if isinstance(lengths, array)
            else array("I", (int(length) for length in lengths))
        )
        self.batch_size = int(batch_size)
        self.bucket_size = self.batch_size * int(bucket_size_multiplier)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.lengths)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        shuffled_indices = torch.randperm(
            len(self.lengths),
            generator=generator,
            dtype=torch.int32,
        )

        for start in range(0, len(shuffled_indices), self.bucket_size):
            bucket = shuffled_indices[start : start + self.bucket_size].tolist()
            bucket.sort(key=self.lengths.__getitem__, reverse=True)
            yield from bucket


class AssayListEpochSampler(Sampler[tuple[int, int]]):
    """Regenerate dynamic assay lists and order them deterministically each epoch."""

    def __init__(
        self,
        dataset: RewardAssayListDataset,
        *,
        seed: int,
        length_bucketing: bool,
        batch_size: int,
        bucket_size_multiplier: int,
    ):
        self.dataset = dataset
        self.seed = int(seed)
        self.length_bucketing = bool(length_bucketing)
        self.batch_size = int(batch_size)
        self.bucket_size = self.batch_size * int(bucket_size_multiplier)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        # Update before DataLoader workers are started so non-persistent workers
        # inherit the prepared bundles instead of rebuilding them independently.
        self.dataset.set_epoch(self.epoch)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        self.dataset.set_epoch(self.epoch)
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        order = torch.randperm(
            len(self.dataset),
            generator=generator,
            dtype=torch.int32,
        ).tolist()
        if not self.length_bucketing:
            yield from ((self.epoch, index) for index in order)
            return

        lengths = self.dataset.get_item_sequence_lengths()
        for start in range(0, len(order), self.bucket_size):
            bucket = order[start : start + self.bucket_size]
            bucket.sort(key=lengths.__getitem__, reverse=True)
            yield from ((self.epoch, index) for index in bucket)


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

    def _reset_train_component_accumulator(self) -> None:
        self._train_component_sums = {
            "ranking_loss": 0.0,
            "classification_loss": 0.0,
            "total_loss": 0.0,
            "num_examples": 0.0,
            "num_ranking_lists": 0.0,
            "num_ranked_examples": 0.0,
        }
        self._train_component_count = 0

    def _get_train_sampler(self, train_dataset=None):
        active_train_dataset = train_dataset if train_dataset is not None else self.train_dataset
        if isinstance(active_train_dataset, RewardAssayListDataset):
            data_seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
            return AssayListEpochSampler(
                active_train_dataset,
                seed=int(data_seed),
                length_bucketing=bool(
                    getattr(self.args, "reward_length_bucketing", False)
                ),
                batch_size=int(
                    getattr(self, "_train_batch_size", self.args.train_batch_size)
                ),
                bucket_size_multiplier=int(
                    getattr(self.args, "length_bucket_size_multiplier", 50)
                ),
            )
        if (
            getattr(self.args, "reward_length_bucketing", False)
            and isinstance(active_train_dataset, RewardPairDataset)
        ):
            data_seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
            return LengthBucketSampler(
                active_train_dataset.get_pair_sequence_lengths(),
                batch_size=int(getattr(self, "_train_batch_size", self.args.train_batch_size)),
                bucket_size_multiplier=int(
                    getattr(self.args, "length_bucket_size_multiplier", 50)
                ),
                seed=int(data_seed),
            )

        parent_sampler = super()._get_train_sampler
        if "train_dataset" in inspect.signature(parent_sampler).parameters:
            return parent_sampler(train_dataset)
        return parent_sampler()

    @staticmethod
    def _to_scalar(value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, torch.Tensor):
            return float(value.detach().float().mean().cpu().item())
        return float(value)

    def _record_train_components(
        self,
        ranking_loss: Optional[torch.Tensor],
        classification_loss: Optional[torch.Tensor],
        total_loss: torch.Tensor,
        num_examples: Any,
        num_ranking_lists: Any,
        num_ranked_examples: Any,
    ) -> None:
        self._train_component_sums["ranking_loss"] += self._to_scalar(ranking_loss)
        self._train_component_sums["classification_loss"] += self._to_scalar(classification_loss)
        self._train_component_sums["total_loss"] += self._to_scalar(total_loss)
        self._train_component_sums["num_examples"] += self._to_scalar(num_examples)
        self._train_component_sums["num_ranking_lists"] += self._to_scalar(
            num_ranking_lists
        )
        self._train_component_sums["num_ranked_examples"] += self._to_scalar(
            num_ranked_examples
        )
        self._train_component_count += 1

    def _consume_train_component_logs(self) -> Dict[str, float]:
        if self._train_component_count == 0:
            return {}
        denom = float(self._train_component_count)
        logs = {
            "ranking_loss": self._train_component_sums["ranking_loss"] / denom,
            "classification_loss": self._train_component_sums["classification_loss"] / denom,
            "total_loss": self._train_component_sums["total_loss"] / denom,
            "num_examples": self._train_component_sums["num_examples"] / denom,
            "num_ranking_lists": self._train_component_sums["num_ranking_lists"] / denom,
            "num_ranked_examples": self._train_component_sums["num_ranked_examples"] / denom,
        }
        self._reset_train_component_accumulator()
        return logs

    def _build_model_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_inputs = {
            "protein_input_ids": inputs["protein_input_ids"],
            "protein_attention_mask": inputs["protein_attention_mask"],
            "molecule_input_ids": inputs["molecule_input_ids"],
            "molecule_attention_mask": inputs["molecule_attention_mask"],
            "activity_labels": inputs["activity_labels"],
            "return_dict": True,
        }
        if "pchembl_values" in inputs and "ranking_group_ids" in inputs:
            model_inputs["pchembl_values"] = inputs["pchembl_values"]
            model_inputs["ranking_group_ids"] = inputs["ranking_group_ids"]
        elif "positive_indices" in inputs and "negative_indices" in inputs:
            model_inputs["positive_indices"] = inputs["positive_indices"]
            model_inputs["negative_indices"] = inputs["negative_indices"]
        return model_inputs

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
        model_inputs = self._build_model_inputs(inputs)
        outputs = model(**model_inputs)
        if outputs.loss is None:
            raise RuntimeError("RewardModel did not return a scalar loss")

        if getattr(model, "training", False):
            self._record_train_components(
                ranking_loss=outputs.ranking_loss,
                classification_loss=outputs.classification_loss,
                total_loss=outputs.loss,
                num_examples=inputs.get("num_examples"),
                num_ranking_lists=inputs.get("num_ranking_lists"),
                num_ranked_examples=inputs.get("num_ranked_examples"),
            )

        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def _evaluate_reward_dataset(
        self,
        eval_dataset,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        active_eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if not isinstance(active_eval_dataset, RewardEvaluationDataset):
            raise TypeError(
                "RewardModelTrainer expects eval_dataset to be a RewardEvaluationDataset"
            )
        dataloader = self.get_eval_dataloader(active_eval_dataset)
        model = self.model_wrapped if self.model_wrapped is not None else self.model
        was_training = model.training
        model.eval()

        gathered_logits: list[torch.Tensor] = []
        gathered_scores: list[torch.Tensor] = []
        gathered_labels: list[torch.Tensor] = []
        gathered_pchembl: list[torch.Tensor] = []
        gathered_groups: list[torch.Tensor] = []
        for batch in dataloader:
            batch = self._prepare_inputs(batch)
            with torch.no_grad(), self.compute_loss_context_manager():
                outputs = model(**self._build_model_inputs(batch))
            gathered = self.accelerator.gather_for_metrics(
                (
                    outputs.activity_logits.detach(),
                    outputs.ranking_score.detach(),
                    batch["activity_labels"].detach(),
                    batch["pchembl_values"].detach(),
                    batch["evaluation_group_indices"].detach(),
                )
            )
            logits, scores, labels, pchembl, group_indices = gathered
            gathered_logits.append(logits.cpu())
            gathered_scores.append(scores.cpu())
            gathered_labels.append(labels.cpu())
            gathered_pchembl.append(pchembl.cpu())
            gathered_groups.append(group_indices.cpu())

        if was_training:
            model.train()
        if not gathered_logits:
            raise ValueError("Evaluation dataset produced no observations")
        activity_logits = torch.cat(gathered_logits)
        ranking_scores = torch.cat(gathered_scores)
        activity_labels = torch.cat(gathered_labels)
        pchembl_values = torch.cat(gathered_pchembl)
        ranking_group_ids = torch.cat(gathered_groups)
        if (ranking_group_ids < 0).any():
            raise RuntimeError("Evaluation observations must have stable assay group ids")

        model_config = self.model.config
        metrics, assay_records = compute_joint_evaluation_metrics(
            activity_logits=activity_logits,
            ranking_scores=ranking_scores,
            activity_labels=activity_labels,
            pchembl_values=pchembl_values,
            ranking_group_ids=ranking_group_ids,
            group_id_names=active_eval_dataset.group_ids,
            classification_loss_weight=model_config.classification_loss_weight,
            ranking_loss_weight=model_config.ranking_loss_weight,
            bce_pos_weight=model_config.bce_pos_weight,
            ranking_temperature=model_config.ranking_temperature,
            ranking_min_pchembl_span=active_eval_dataset.ranking_min_pchembl_span
            if hasattr(active_eval_dataset, "ranking_min_pchembl_span")
            else model_config.ranking_min_pchembl_span,
        )
        metrics = _with_metric_prefix(metrics, metric_key_prefix)
        _append_assay_spearman_log(
            self,
            metrics,
            assay_records,
            metric_key_prefix=metric_key_prefix,
        )
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args,
            self.state,
            self.control,
            metrics,
        )
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
        optim=config.optim,
        save_safetensors=config.save_safetensors,
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
        training_args = TrainingArguments(**args_kwargs)
        training_args.reward_length_bucketing = config.length_bucketing
        training_args.length_bucket_size_multiplier = config.length_bucket_size_multiplier
        return training_args
    except ImportError as exc:
        raise ImportError(
            "RewardModel training via Hugging Face Trainer requires the 'accelerate' package. "
            "Install accelerate>=1.1.0 in the reward_model environment."
        ) from exc
