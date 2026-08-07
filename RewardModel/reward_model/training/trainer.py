from __future__ import annotations

import inspect
import json
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
    compute_classification_metrics,
    compute_joint_evaluation_metrics,
    compute_ranking_score_diagnostics,
)


RANKING_SCORE_DIAGNOSTICS_LOG_FILENAME = "ranking_score_diagnostics.jsonl"

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
            "ranking_loss_sum": 0.0,
            "classification_correct": 0.0,
            "classification_count": 0.0,
            "ranking_pairwise_correct": 0.0,
            "ranking_pairwise_count": 0.0,
        }
        self._train_component_count = 0
        self._train_classification_probabilities: list[float] = []
        self._train_classification_labels: list[float] = []
        self._train_ranking_diagnostic_sums: Dict[str, float] = {}
        self._train_ranking_diagnostic_counts: Dict[str, int] = {}

    def _ranking_score_diagnostics_enabled(self) -> bool:
        return bool(
            getattr(
                getattr(self, "args", None),
                "reward_ranking_score_diagnostics",
                False,
            )
        )

    def _ranking_score_diagnostic_scales(self) -> tuple[float, ...]:
        return tuple(
            getattr(
                getattr(self, "args", None),
                "reward_ranking_score_diagnostic_scales",
                (3.0, 5.0, 13.0),
            )
        )

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
        activity_logits: torch.Tensor,
        activity_labels: torch.Tensor,
        ranking_score: torch.Tensor,
        pchembl_values: Optional[torch.Tensor],
        ranking_group_ids: Optional[torch.Tensor],
    ) -> None:
        ranking_loss_value = self._to_scalar(ranking_loss)
        num_ranking_lists_value = self._to_scalar(num_ranking_lists)
        self._train_component_sums["ranking_loss"] += ranking_loss_value
        self._train_component_sums["classification_loss"] += self._to_scalar(classification_loss)
        self._train_component_sums["total_loss"] += self._to_scalar(total_loss)
        self._train_component_sums["num_examples"] += self._to_scalar(num_examples)
        self._train_component_sums["num_ranking_lists"] += num_ranking_lists_value
        self._train_component_sums["num_ranked_examples"] += self._to_scalar(
            num_ranked_examples
        )
        self._train_component_sums["ranking_loss_sum"] += (
            ranking_loss_value * num_ranking_lists_value
        )

        with torch.no_grad():
            logits = activity_logits.detach().reshape(-1)
            labels = activity_labels.detach().to(device=logits.device).reshape(-1)
            self._train_component_sums["classification_correct"] += float(
                ((logits >= 0.0) == (labels >= 0.5)).sum().item()
            )
            self._train_component_sums["classification_count"] += float(
                labels.numel()
            )
            self._train_classification_probabilities.extend(
                torch.sigmoid(logits.float()).cpu().tolist()
            )
            self._train_classification_labels.extend(labels.float().cpu().tolist())

            if pchembl_values is not None and ranking_group_ids is not None:
                scores = ranking_score.detach().reshape(-1)
                targets = pchembl_values.detach().to(device=scores.device).reshape(-1)
                group_ids = ranking_group_ids.detach().to(device=scores.device).reshape(-1)
                same_group = (
                    (group_ids[:, None] == group_ids[None, :])
                    & (group_ids[:, None] >= 0)
                )
                comparable = (
                    torch.triu(same_group, diagonal=1)
                    & (targets[:, None] != targets[None, :])
                )
                comparison = (
                    (scores[:, None] - scores[None, :])
                    * (targets[:, None] - targets[None, :])
                )
                self._train_component_sums["ranking_pairwise_correct"] += float(
                    (
                        (comparison[comparable] > 0).float().sum()
                        + 0.5 * (comparison[comparable] == 0).float().sum()
                    ).item()
                )
                self._train_component_sums["ranking_pairwise_count"] += float(
                    comparable.sum().item()
                )
                if self._ranking_score_diagnostics_enabled():
                    diagnostics = compute_ranking_score_diagnostics(
                        ranking_scores=scores,
                        pchembl_values=targets,
                        ranking_group_ids=group_ids,
                        temperature=float(self.model.config.ranking_temperature),
                        affinity_margin=float(
                            self.model.config.ranking_affinity_margin
                        ),
                        saturation_scales=self._ranking_score_diagnostic_scales(),
                    )
                    for key, value in diagnostics.items():
                        self._train_ranking_diagnostic_sums[key] = (
                            self._train_ranking_diagnostic_sums.get(key, 0.0)
                            + float(value)
                        )
                        self._train_ranking_diagnostic_counts[key] = (
                            self._train_ranking_diagnostic_counts.get(key, 0) + 1
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
        ranked_example_count = self._train_component_sums["num_ranked_examples"]
        if ranked_example_count > 0.0:
            logs["ranking_loss_per_ranked_example"] = (
                self._train_component_sums["ranking_loss_sum"]
                / ranked_example_count
            )
        classification_count = self._train_component_sums["classification_count"]
        if classification_count > 0.0:
            logs["classification_accuracy"] = (
                self._train_component_sums["classification_correct"]
                / classification_count
            )
            classification_metrics = compute_classification_metrics(
                self._train_classification_probabilities,
                self._train_classification_labels,
            )
            logs.update(
                {
                    "classification_mcc": classification_metrics["eval_mcc"],
                    "classification_f1": classification_metrics["eval_f1"],
                    "classification_auroc": classification_metrics["eval_roc_auc"],
                }
            )
        ranking_pairwise_count = self._train_component_sums["ranking_pairwise_count"]
        if ranking_pairwise_count > 0.0:
            logs["ranking_pairwise_accuracy"] = (
                self._train_component_sums["ranking_pairwise_correct"]
                / ranking_pairwise_count
            )
        logs.update(
            {
                key: total / self._train_ranking_diagnostic_counts[key]
                for key, total in self._train_ranking_diagnostic_sums.items()
                if self._train_ranking_diagnostic_counts.get(key, 0) > 0
            }
        )
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
                activity_logits=outputs.activity_logits,
                activity_labels=inputs["activity_labels"],
                ranking_score=outputs.ranking_score,
                pchembl_values=inputs.get("pchembl_values"),
                ranking_group_ids=inputs.get("ranking_group_ids"),
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
        gathered_example_indices: list[torch.Tensor] = []
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
                    batch["evaluation_example_indices"].detach(),
                )
            )
            logits, scores, labels, pchembl, group_indices, example_indices = gathered
            gathered_logits.append(logits.cpu())
            gathered_scores.append(scores.cpu())
            gathered_labels.append(labels.cpu())
            gathered_pchembl.append(pchembl.cpu())
            gathered_groups.append(group_indices.cpu())
            gathered_example_indices.append(example_indices.cpu())

        if was_training:
            model.train()
        if not gathered_logits:
            raise ValueError("Evaluation dataset produced no observations")
        activity_logits = torch.cat(gathered_logits)
        ranking_scores = torch.cat(gathered_scores)
        activity_labels = torch.cat(gathered_labels)
        pchembl_values = torch.cat(gathered_pchembl)
        ranking_group_ids = torch.cat(gathered_groups)
        evaluation_example_indices = torch.cat(gathered_example_indices)
        if (ranking_group_ids < 0).any():
            raise RuntimeError("Evaluation observations must have stable assay group ids")
        if (evaluation_example_indices < 0).any():
            raise RuntimeError("Evaluation observations must have stable example indices")
        if (
            evaluation_example_indices.numel() != len(active_eval_dataset)
            or torch.unique(evaluation_example_indices).numel()
            != len(active_eval_dataset)
        ):
            raise RuntimeError("Evaluation must score every example exactly once")
        stable_order = torch.argsort(evaluation_example_indices, stable=True)
        activity_logits = activity_logits.index_select(0, stable_order)
        ranking_scores = ranking_scores.index_select(0, stable_order)
        activity_labels = activity_labels.index_select(0, stable_order)
        pchembl_values = pchembl_values.index_select(0, stable_order)
        ranking_group_ids = ranking_group_ids.index_select(0, stable_order)

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
            ranking_affinity_margin=model_config.ranking_affinity_margin,
            ranking_min_pchembl_span=active_eval_dataset.ranking_min_pchembl_span
            if hasattr(active_eval_dataset, "ranking_min_pchembl_span")
            else model_config.ranking_min_pchembl_span,
            ranking_max_ligands=getattr(
                active_eval_dataset,
                "ranking_max_ligands",
                16,
            ),
            ranking_num_partitions=getattr(
                active_eval_dataset,
                "ranking_num_partitions",
                3,
            ),
            ranking_partition_seed=getattr(
                active_eval_dataset,
                "ranking_partition_seed",
                int(self.args.seed),
            ),
            ranking_score_diagnostics=self._ranking_score_diagnostics_enabled(),
            ranking_score_diagnostic_scales=self._ranking_score_diagnostic_scales(),
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
        self._append_ranking_score_diagnostics_log(logs)
        parent_log = super().log
        if "start_time" in inspect.signature(parent_log).parameters:
            return parent_log(logs, start_time=start_time)
        return parent_log(logs)

    def _append_ranking_score_diagnostics_log(self, logs: Dict[str, Any]) -> None:
        if not self._ranking_score_diagnostics_enabled():
            return
        if not self.is_world_process_zero():
            return
        diagnostic_metrics = {
            key: float(value)
            for key, value in logs.items()
            if (
                "ranking_score_" in key
                or "ranking_margin_pair_" in key
                or "ranking_list_" in key
                or key in {"loss", "grad_norm", "ranking_loss", "total_loss"}
                or key.endswith("_ranking_loss")
                or key.endswith("_spearman")
            )
            and isinstance(value, (int, float))
        }
        if not diagnostic_metrics:
            return

        split = "train"
        if any(key.startswith("eval_val2_") for key in diagnostic_metrics):
            split = "eval_val2"
        elif any(key.startswith("eval_") for key in diagnostic_metrics):
            split = "eval"
        os.makedirs(self.args.output_dir, exist_ok=True)
        path = os.path.join(
            self.args.output_dir,
            RANKING_SCORE_DIAGNOSTICS_LOG_FILENAME,
        )
        record = {
            "global_step": int(self.state.global_step),
            "epoch": self.state.epoch,
            "split": split,
            "metrics": diagnostic_metrics,
        }
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

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
    if config.max_steps is not None:
        args_kwargs["max_steps"] = config.max_steps

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
    if "eval_strategy" in init_params:
        args_kwargs["eval_strategy"] = schedule_strategy
    elif "evaluation_strategy" in init_params:
        args_kwargs["evaluation_strategy"] = schedule_strategy
    try:
        training_args = TrainingArguments(**args_kwargs)
        training_args.reward_length_bucketing = config.length_bucketing
        training_args.length_bucket_size_multiplier = config.length_bucket_size_multiplier
        training_args.reward_ranking_score_diagnostics = (
            config.ranking_score_diagnostics
        )
        training_args.reward_ranking_score_diagnostic_scales = (
            config.ranking_score_diagnostic_scales
        )
        return training_args
    except ImportError as exc:
        raise ImportError(
            "RewardModel training via Hugging Face Trainer requires the 'accelerate' package. "
            "Install accelerate>=1.1.0 in the reward_model environment."
        ) from exc
