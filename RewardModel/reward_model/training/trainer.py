from __future__ import annotations

import inspect
import json
import math
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
    compute_contrastive_evaluation_loss,
    compute_groupwise_rank_correlations,
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


def _encode_identity_ids(values: Sequence[Any]) -> torch.Tensor:
    identities = [str(value) for value in values]
    identity_to_id = {
        identity: index for index, identity in enumerate(sorted(set(identities)))
    }
    return torch.tensor(
        [identity_to_id[identity] for identity in identities],
        dtype=torch.long,
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
    def __init__(
        self,
        *args,
        val2_eval_dataset=None,
        fixed_train_eval_dataset=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.val2_eval_dataset = val2_eval_dataset
        self.fixed_train_eval_dataset = fixed_train_eval_dataset
        self._last_objective_gradient_diagnostic_step: int | None = None
        self._pending_objective_gradient_logs: Dict[str, float] = {}
        self._reset_train_component_accumulator()

    def _reset_train_component_accumulator(self) -> None:
        self._train_component_sums = {
            "ranking_loss": 0.0,
            "contrastive_loss": 0.0,
            "classification_loss": 0.0,
            "num_examples": 0.0,
            "num_contrastive_lists": 0.0,
            "num_contrastive_examples": 0.0,
            "num_ranking_lists": 0.0,
            "num_ranked_examples": 0.0,
            "classification_correct": 0.0,
            "classification_count": 0.0,
        }
        self._train_component_count = 0
        self._train_classification_probabilities: list[float] = []
        self._train_classification_labels: list[float] = []
        self._train_ranking_diagnostic_sums: Dict[str, float] = {}
        self._train_ranking_diagnostic_counts: Dict[str, int] = {}
        self._train_ranking_window_scores: list[torch.Tensor] = []
        self._train_ranking_window_targets: list[torch.Tensor] = []
        self._train_ranking_window_group_ids: list[torch.Tensor] = []
        self._train_ranking_window_cosines: list[torch.Tensor] = []
        self._train_ranking_group_offset = 0

    @staticmethod
    def _component_learning_rate(
        parameter_name: str,
        *,
        default_learning_rate: float,
        encoder_learning_rate: Optional[float],
        projection_learning_rate: Optional[float],
    ) -> float:
        if parameter_name.startswith(("protein_encoder.", "molecule_encoder.")):
            return float(encoder_learning_rate or default_learning_rate)
        if parameter_name.startswith(
            ("protein_projection.", "molecule_projection.")
        ):
            return float(projection_learning_rate or default_learning_rate)
        return float(default_learning_rate)

    def create_optimizer(self, model=None):
        """Create AdamW-style parameter groups for controlled LR ablations."""
        encoder_learning_rate = getattr(
            self.args,
            "reward_encoder_learning_rate",
            None,
        )
        projection_learning_rate = getattr(
            self.args,
            "reward_projection_learning_rate",
            None,
        )
        if encoder_learning_rate is None and projection_learning_rate is None:
            parent_create_optimizer = super().create_optimizer
            if "model" in inspect.signature(parent_create_optimizer).parameters:
                return parent_create_optimizer(model=model)
            return parent_create_optimizer()

        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model if model is None else model
        decay_parameters = self.get_decay_parameter_names(opt_model)
        grouped_parameters: Dict[tuple[float, float], list[torch.nn.Parameter]] = {}
        for parameter_name, parameter in opt_model.named_parameters():
            if not parameter.requires_grad:
                continue
            learning_rate = self._component_learning_rate(
                parameter_name,
                default_learning_rate=float(self.args.learning_rate),
                encoder_learning_rate=encoder_learning_rate,
                projection_learning_rate=projection_learning_rate,
            )
            weight_decay = (
                float(self.args.weight_decay)
                if parameter_name in decay_parameters
                else 0.0
            )
            grouped_parameters.setdefault((learning_rate, weight_decay), []).append(
                parameter
            )

        optimizer_grouped_parameters = [
            {
                "params": parameters,
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
            for (learning_rate, weight_decay), parameters in (
                grouped_parameters.items()
            )
        ]
        if self.optimizer_cls_and_kwargs is not None:
            optimizer_cls, raw_optimizer_kwargs = self.optimizer_cls_and_kwargs
        else:
            optimizer_cls, raw_optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                self.args,
                opt_model,
            )
        optimizer_kwargs = dict(raw_optimizer_kwargs)
        unsupported_overrides = {
            key
            for key in ("params", "model", "optimizer_dict")
            if key in optimizer_kwargs
        }
        if unsupported_overrides:
            raise ValueError(
                "encoder/projection learning rates are not compatible with optimizer "
                f"parameter overrides: {sorted(unsupported_overrides)}"
            )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def _ranking_score_diagnostics_enabled(self) -> bool:
        return bool(
            getattr(
                getattr(self, "args", None),
                "reward_ranking_score_diagnostics",
                False,
            )
        )

    def _ranking_metrics_profile_enabled(self) -> bool:
        return (
            getattr(
                getattr(self, "args", None),
                "reward_metrics_profile",
                "full",
            )
            == "ranking"
        )

    def _record_objective_gradient_diagnostics(self, outputs) -> None:
        """Measure effective objective gradients at the shared embedding boundary."""
        interval = int(
            getattr(
                getattr(self, "args", None),
                "reward_objective_gradient_diagnostics_steps",
                0,
            )
        )
        if interval <= 0:
            return
        # compute_loss runs before Trainer increments global_step. Associate the
        # diagnostic with the optimizer step this batch is about to complete so
        # an interval of 100 appears in the step-100 log, not step 110.
        diagnostic_step = int(self.state.global_step) + 1
        if diagnostic_step % interval != 0:
            return
        if self._last_objective_gradient_diagnostic_step == diagnostic_step:
            return
        if outputs.ranking_loss is None or outputs.contrastive_loss is None:
            return
        embedding_tensors = (
            outputs.normalized_protein_embedding,
            outputs.normalized_molecule_embedding,
        )
        if any(tensor is None for tensor in embedding_tensors):
            return

        config = self.model.config
        ranking_objective = (
            outputs.ranking_loss * float(config.ranking_loss_weight)
        )
        contrastive_objective = (
            outputs.contrastive_loss * float(config.contrastive_loss_weight)
        )
        ranking_gradients = torch.autograd.grad(
            ranking_objective,
            embedding_tensors,
            retain_graph=True,
            allow_unused=True,
        )
        contrastive_gradients = torch.autograd.grad(
            contrastive_objective,
            embedding_tensors,
            retain_graph=True,
            allow_unused=True,
        )
        reference = next(
            tensor for tensor in embedding_tensors if tensor is not None
        )
        ranking_square = torch.zeros((), device=reference.device, dtype=torch.float32)
        contrastive_square = torch.zeros_like(ranking_square)
        gradient_dot = torch.zeros_like(ranking_square)
        for ranking_gradient, contrastive_gradient in zip(
            ranking_gradients,
            contrastive_gradients,
        ):
            if ranking_gradient is not None:
                ranking_gradient = ranking_gradient.detach().float()
                ranking_square += ranking_gradient.square().sum()
            if contrastive_gradient is not None:
                contrastive_gradient = contrastive_gradient.detach().float()
                contrastive_square += contrastive_gradient.square().sum()
            if ranking_gradient is not None and contrastive_gradient is not None:
                gradient_dot += (ranking_gradient * contrastive_gradient).sum()
        ranking_norm = ranking_square.sqrt()
        contrastive_norm = contrastive_square.sqrt()
        denominator = ranking_norm * contrastive_norm
        gradient_cosine = torch.where(
            denominator > 0,
            gradient_dot / denominator,
            torch.zeros_like(denominator),
        )
        diagnostics = torch.stack(
            [ranking_norm, contrastive_norm, gradient_cosine]
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                diagnostics,
                op=torch.distributed.ReduceOp.SUM,
            )
            diagnostics /= torch.distributed.get_world_size()
        values = diagnostics.cpu().tolist()
        self._pending_objective_gradient_logs = {
            "ranking_embedding_grad_norm": float(values[0]),
            "contrastive_embedding_grad_norm": float(values[1]),
            "ranking_contrastive_embedding_grad_cosine": float(values[2]),
        }
        self._last_objective_gradient_diagnostic_step = diagnostic_step

    def _current_scaled_cosine_logs(self) -> Dict[str, float]:
        model = getattr(self, "model", None)
        if model is None:
            return {}
        if hasattr(self, "accelerator"):
            model = self.accelerator.unwrap_model(model)
        config = getattr(model, "config", None)
        if getattr(config, "pair_scoring_mode", "mlp") != "scaled_cosine":
            return {}
        logit_scale = getattr(model, "logit_scale", None)
        classification_bias = getattr(model, "classification_logit_bias", None)
        if logit_scale is None or classification_bias is None:
            return {}
        with torch.no_grad():
            scale = logit_scale.detach().float().clamp(
                max=math.log(float(config.cosine_scale_max))
            ).exp()
            return {
                "cosine_scale": float(scale.cpu().item()),
                "classification_logit_bias": float(
                    classification_bias.detach().float().cpu().item()
                ),
            }

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
        num_examples: Any,
        num_contrastive_lists: Any,
        num_contrastive_examples: Any,
        num_ranking_lists: Any,
        num_ranked_examples: Any,
        activity_logits: torch.Tensor,
        activity_labels: torch.Tensor,
        ranking_score: torch.Tensor,
        pchembl_values: Optional[torch.Tensor],
        ranking_group_ids: Optional[torch.Tensor],
        cosine_similarity: Optional[torch.Tensor] = None,
        contrastive_loss: Optional[torch.Tensor] = None,
    ) -> None:
        ranking_loss_value = self._to_scalar(ranking_loss)
        num_ranking_lists_value = self._to_scalar(num_ranking_lists)
        self._train_component_sums["ranking_loss"] += ranking_loss_value
        self._train_component_sums["contrastive_loss"] += self._to_scalar(
            contrastive_loss
        )
        if not self._ranking_metrics_profile_enabled():
            self._train_component_sums["classification_loss"] += self._to_scalar(
                classification_loss
            )
        self._train_component_sums["num_examples"] += self._to_scalar(num_examples)
        self._train_component_sums["num_contrastive_lists"] += self._to_scalar(
            num_contrastive_lists
        )
        self._train_component_sums["num_contrastive_examples"] += self._to_scalar(
            num_contrastive_examples
        )
        self._train_component_sums["num_ranking_lists"] += num_ranking_lists_value
        self._train_component_sums["num_ranked_examples"] += self._to_scalar(
            num_ranked_examples
        )
        with torch.no_grad():
            logits = activity_logits.detach().reshape(-1)
            labels = activity_labels.detach().to(device=logits.device).reshape(-1)
            if not self._ranking_metrics_profile_enabled():
                self._train_component_sums["classification_correct"] += float(
                    ((logits >= 0.0) == (labels >= 0.5)).sum().item()
                )
                self._train_component_sums["classification_count"] += float(
                    labels.numel()
                )
                self._train_classification_probabilities.extend(
                    torch.sigmoid(logits.float()).cpu().tolist()
                )
                self._train_classification_labels.extend(
                    labels.float().cpu().tolist()
                )

            if pchembl_values is not None and ranking_group_ids is not None:
                scores = ranking_score.detach().reshape(-1)
                targets = pchembl_values.detach().to(device=scores.device).reshape(-1)
                group_ids = ranking_group_ids.detach().to(device=scores.device).reshape(-1)
                if self._ranking_metrics_profile_enabled():
                    ranked_mask = group_ids >= 0
                    if ranked_mask.any():
                        ranked_group_ids = group_ids[ranked_mask]
                        _, remapped_group_ids = torch.unique(
                            ranked_group_ids,
                            sorted=True,
                            return_inverse=True,
                        )
                        remapped_group_ids = (
                            remapped_group_ids + self._train_ranking_group_offset
                        )
                        self._train_ranking_group_offset += int(
                            remapped_group_ids.max().item()
                            - remapped_group_ids.min().item()
                            + 1
                        )
                        self._train_ranking_window_scores.append(
                            scores[ranked_mask].float().cpu()
                        )
                        self._train_ranking_window_targets.append(
                            targets[ranked_mask].float().cpu()
                        )
                        self._train_ranking_window_group_ids.append(
                            remapped_group_ids.long().cpu()
                        )
                        if cosine_similarity is not None:
                            cosines = cosine_similarity.detach().reshape(-1)
                            self._train_ranking_window_cosines.append(
                                cosines[ranked_mask].float().cpu()
                            )
                elif self._ranking_score_diagnostics_enabled():
                    diagnostics = compute_ranking_score_diagnostics(
                        ranking_scores=scores,
                        pchembl_values=targets,
                        ranking_group_ids=group_ids,
                        cosine_similarities=cosine_similarity,
                        temperature=float(self.model.config.ranking_temperature),
                        affinity_margin=float(
                            self.model.config.ranking_affinity_margin
                        ),
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
        if self._ranking_metrics_profile_enabled():
            logs = {}
            if float(self.model.config.contrastive_loss_weight) > 0.0:
                logs.update(
                    {
                        "ranking_loss": (
                            self._train_component_sums["ranking_loss"] / denom
                        ),
                        "contrastive_loss": (
                            self._train_component_sums["contrastive_loss"] / denom
                        ),
                    }
                )
            if self._train_ranking_window_scores:
                scores = torch.cat(self._train_ranking_window_scores)
                targets = torch.cat(self._train_ranking_window_targets)
                group_ids = torch.cat(self._train_ranking_window_group_ids)
                diagnostics = compute_ranking_score_diagnostics(
                    ranking_scores=scores,
                    pchembl_values=targets,
                    ranking_group_ids=group_ids,
                    temperature=float(self.model.config.ranking_temperature),
                    affinity_margin=float(
                        self.model.config.ranking_affinity_margin
                    ),
                )
                if "ranking_margin_pair_accuracy" in diagnostics:
                    logs["pair_accuracy"] = diagnostics[
                        "ranking_margin_pair_accuracy"
                    ]
                logs.update(
                    compute_groupwise_rank_correlations(
                        group_ids=group_ids.tolist(),
                        ranking_scores=scores.tolist(),
                        pchembl_values=targets.tolist(),
                    )
                )
            if self._train_ranking_window_cosines:
                logs["cosine_std"] = float(
                    torch.cat(self._train_ranking_window_cosines)
                    .std(unbiased=False)
                    .item()
                )
            self._reset_train_component_accumulator()
            return logs

        logs = {
            "ranking_loss": self._train_component_sums["ranking_loss"] / denom,
            "contrastive_loss": self._train_component_sums["contrastive_loss"] / denom,
            "classification_loss": self._train_component_sums["classification_loss"] / denom,
            "num_examples": self._train_component_sums["num_examples"] / denom,
            "num_contrastive_lists": self._train_component_sums[
                "num_contrastive_lists"
            ] / denom,
            "num_contrastive_examples": self._train_component_sums[
                "num_contrastive_examples"
            ] / denom,
            "num_ranking_lists": self._train_component_sums["num_ranking_lists"] / denom,
            "num_ranked_examples": self._train_component_sums["num_ranked_examples"] / denom,
        }
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
            if "contrastive_group_ids" in inputs:
                model_inputs["contrastive_group_ids"] = inputs[
                    "contrastive_group_ids"
                ]
            if "contrastive_target_ids" in inputs:
                model_inputs["contrastive_target_ids"] = inputs[
                    "contrastive_target_ids"
                ]
            if "contrastive_molecule_ids" in inputs:
                model_inputs["contrastive_molecule_ids"] = inputs[
                    "contrastive_molecule_ids"
                ]
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
            self._record_objective_gradient_diagnostics(outputs)
            self._record_train_components(
                ranking_loss=outputs.ranking_loss,
                classification_loss=outputs.classification_loss,
                num_examples=inputs.get("num_examples"),
                num_contrastive_lists=inputs.get("num_contrastive_lists"),
                num_contrastive_examples=inputs.get(
                    "num_contrastive_examples"
                ),
                num_ranking_lists=inputs.get("num_ranking_lists"),
                num_ranked_examples=inputs.get("num_ranked_examples"),
                activity_logits=outputs.activity_logits,
                activity_labels=inputs["activity_labels"],
                ranking_score=outputs.ranking_score,
                pchembl_values=inputs.get("pchembl_values"),
                ranking_group_ids=inputs.get("ranking_group_ids"),
                cosine_similarity=outputs.cosine_similarity,
                contrastive_loss=outputs.contrastive_loss,
            )

        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def _evaluate_reward_dataset(
        self,
        eval_dataset,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
        notify_callbacks: bool = True,
    ) -> Dict[str, float]:
        active_eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if not isinstance(active_eval_dataset, RewardEvaluationDataset):
            raise TypeError(
                "RewardModelTrainer expects eval_dataset to be a RewardEvaluationDataset"
            )
        dataloader = self.get_eval_dataloader(active_eval_dataset)
        model = self.model_wrapped if self.model_wrapped is not None else self.model
        model_config = self.model.config
        contrastive_evaluation_enabled = (
            float(model_config.contrastive_loss_weight) > 0.0
        )
        was_training = model.training
        model.eval()

        gathered_logits: list[torch.Tensor] = []
        gathered_scores: list[torch.Tensor] = []
        gathered_labels: list[torch.Tensor] = []
        gathered_pchembl: list[torch.Tensor] = []
        gathered_groups: list[torch.Tensor] = []
        gathered_example_indices: list[torch.Tensor] = []
        gathered_cosines: list[torch.Tensor] = []
        gathered_protein_embeddings: list[torch.Tensor] = []
        gathered_molecule_embeddings: list[torch.Tensor] = []
        gathered_protein_shuffled_scores: list[torch.Tensor] = []
        for batch in dataloader:
            batch = self._prepare_inputs(batch)
            with torch.no_grad(), self.compute_loss_context_manager():
                outputs = model(**self._build_model_inputs(batch))
            gathered_values = [
                outputs.activity_logits.detach(),
                outputs.ranking_score.detach(),
                batch["activity_labels"].detach(),
                batch["pchembl_values"].detach(),
                batch["evaluation_group_indices"].detach(),
                batch["evaluation_example_indices"].detach(),
            ]
            if outputs.cosine_similarity is not None:
                gathered_values.append(outputs.cosine_similarity.detach())
            protein_embedding_index = None
            molecule_embedding_index = None
            if contrastive_evaluation_enabled:
                if (
                    outputs.normalized_protein_embedding is None
                    or outputs.normalized_molecule_embedding is None
                ):
                    raise RuntimeError(
                        "Contrastive evaluation requires normalized model embeddings"
                    )
                protein_embedding_index = len(gathered_values)
                gathered_values.append(
                    outputs.normalized_protein_embedding.detach()
                )
                molecule_embedding_index = len(gathered_values)
                gathered_values.append(
                    outputs.normalized_molecule_embedding.detach()
                )
            gathered = self.accelerator.gather_for_metrics(tuple(gathered_values))
            logits, scores, labels, pchembl, group_indices, example_indices = gathered[:6]
            gathered_logits.append(logits.cpu())
            gathered_scores.append(scores.cpu())
            gathered_labels.append(labels.cpu())
            gathered_pchembl.append(pchembl.cpu())
            gathered_groups.append(group_indices.cpu())
            gathered_example_indices.append(example_indices.cpu())
            if outputs.cosine_similarity is not None:
                gathered_cosines.append(gathered[6].cpu())
            if protein_embedding_index is not None:
                gathered_protein_embeddings.append(
                    gathered[protein_embedding_index].cpu()
                )
                gathered_molecule_embeddings.append(
                    gathered[molecule_embedding_index].cpu()
                )
            if (
                bool(
                    getattr(
                        self.args,
                        "reward_protein_shuffle_sensitivity",
                        True,
                    )
                )
                and "protein_shuffled_input_ids" in batch
            ):
                with torch.no_grad(), self.compute_loss_context_manager():
                    shuffled_outputs = model(
                        protein_input_ids=batch["protein_shuffled_input_ids"],
                        protein_attention_mask=batch[
                            "protein_shuffled_attention_mask"
                        ],
                        molecule_input_ids=batch["molecule_input_ids"],
                        molecule_attention_mask=batch["molecule_attention_mask"],
                        return_dict=True,
                    )
                shuffled_scores = self.accelerator.gather_for_metrics(
                    shuffled_outputs.ranking_score.detach()
                )
                gathered_protein_shuffled_scores.append(shuffled_scores.cpu())

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
        cosine_similarities = (
            torch.cat(gathered_cosines) if gathered_cosines else None
        )
        normalized_protein_embeddings = (
            torch.cat(gathered_protein_embeddings)
            if gathered_protein_embeddings
            else None
        )
        normalized_molecule_embeddings = (
            torch.cat(gathered_molecule_embeddings)
            if gathered_molecule_embeddings
            else None
        )
        protein_shuffled_scores = (
            torch.cat(gathered_protein_shuffled_scores)
            if gathered_protein_shuffled_scores
            else None
        )
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
        if cosine_similarities is not None:
            cosine_similarities = cosine_similarities.index_select(0, stable_order)
        if normalized_protein_embeddings is not None:
            normalized_protein_embeddings = normalized_protein_embeddings.index_select(
                0,
                stable_order,
            )
            normalized_molecule_embeddings = normalized_molecule_embeddings.index_select(
                0,
                stable_order,
            )
        if protein_shuffled_scores is not None:
            protein_shuffled_scores = protein_shuffled_scores.index_select(
                0, stable_order
            )

        contrastive_loss = None
        if contrastive_evaluation_enabled:
            if (
                normalized_protein_embeddings is None
                or normalized_molecule_embeddings is None
            ):
                raise RuntimeError("Evaluation did not gather contrastive embeddings")
            required_identity_columns = {"target_chembl_id", "compound_id"}
            missing_identity_columns = required_identity_columns.difference(
                active_eval_dataset.example_dataset.column_names
            )
            if missing_identity_columns:
                raise ValueError(
                    "Contrastive evaluation requires identity columns: "
                    f"{sorted(missing_identity_columns)}"
                )
            contrastive_loss = compute_contrastive_evaluation_loss(
                normalized_protein_embeddings=normalized_protein_embeddings,
                normalized_molecule_embeddings=normalized_molecule_embeddings,
                pchembl_values=pchembl_values,
                contrastive_group_ids=ranking_group_ids,
                target_identity_ids=_encode_identity_ids(
                    active_eval_dataset.example_dataset["target_chembl_id"]
                ),
                molecule_identity_ids=_encode_identity_ids(
                    active_eval_dataset.example_dataset["compound_id"]
                ),
                temperature=model_config.ranking_temperature,
                active_threshold=model_config.contrastive_active_threshold,
                assay_batch_size=int(
                    getattr(
                        self,
                        "_train_batch_size",
                        self.args.per_device_train_batch_size,
                    )
                ),
                ranking_max_ligands=active_eval_dataset.ranking_max_ligands,
                ranking_num_partitions=active_eval_dataset.ranking_num_partitions,
                ranking_partition_seed=active_eval_dataset.ranking_partition_seed,
                ranking_min_pchembl_span=(
                    active_eval_dataset.ranking_min_pchembl_span
                ),
            )
        ranking_metrics_profile = self._ranking_metrics_profile_enabled()
        metrics, assay_records = compute_joint_evaluation_metrics(
            activity_logits=activity_logits,
            ranking_scores=ranking_scores,
            activity_labels=activity_labels,
            pchembl_values=pchembl_values,
            ranking_group_ids=ranking_group_ids,
            group_id_names=active_eval_dataset.group_ids,
            activity_types=(
                None
                if ranking_metrics_profile
                else (
                    list(active_eval_dataset.example_dataset["activity_type"])
                    if "activity_type"
                    in active_eval_dataset.example_dataset.column_names
                    else ["Unknown"] * len(active_eval_dataset)
                )
            ),
            protein_shuffled_scores=protein_shuffled_scores,
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
            cosine_similarities=cosine_similarities,
            metrics_profile=getattr(
                self.args,
                "reward_metrics_profile",
                "full",
            ),
            contrastive_loss=contrastive_loss,
            contrastive_loss_weight=model_config.contrastive_loss_weight,
        )
        metrics.update(
            {
                f"eval_{key}": value
                for key, value in self._current_scaled_cosine_logs().items()
            }
        )
        metrics = _with_metric_prefix(metrics, metric_key_prefix)
        _append_assay_spearman_log(
            self,
            metrics,
            assay_records,
            metric_key_prefix=metric_key_prefix,
        )
        self.log(metrics)
        if notify_callbacks:
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
        if (
            eval_dataset is None
            and metric_key_prefix == "eval"
            and self.fixed_train_eval_dataset is not None
        ):
            metrics.update(
                self._evaluate_reward_dataset(
                    eval_dataset=self.fixed_train_eval_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix="train_eval",
                    notify_callbacks=False,
                )
            )
        return metrics

    def log(self, logs, start_time=None):
        logs = dict(logs)
        if "loss" in logs:
            logs.update(self._consume_train_component_logs())
            logs.update(self._current_scaled_cosine_logs())
            logs.update(
                getattr(self, "_pending_objective_gradient_logs", {})
            )
            self._pending_objective_gradient_logs = {}
        if self._ranking_metrics_profile_enabled():
            ranking_metric_suffixes = {
                "loss",
                "spearman",
                "pearson",
                "cosine_std",
                "pair_accuracy",
            }
            logs = {
                key: value
                for key, value in logs.items()
                if key
                in {
                    "loss",
                    "grad_norm",
                    "learning_rate",
                    "epoch",
                    "cosine_std",
                    "pair_accuracy",
                    "spearman",
                    "pearson",
                    "ranking_loss",
                    "contrastive_loss",
                    "train_loss",
                    "ranking_embedding_grad_norm",
                    "contrastive_embedding_grad_norm",
                    "ranking_contrastive_embedding_grad_cosine",
                }
                or (
                    key.startswith(("eval_", "test_", "train_eval_"))
                    and any(
                        key.endswith(f"_{suffix}")
                        for suffix in ranking_metric_suffixes
                    )
                )
            }
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
                "ranking_cosine_" in key
                or "cosine_scale" in key
                or "classification_logit_bias" in key
                or "ranking_margin_pair_" in key
                or "ranking_list_" in key
                or "embedding_grad_" in key
                or "embedding_grad" in key
                or key in {"cosine_std", "eval_cosine_std", "eval_pair_accuracy"}
                or key in {"pair_accuracy", "spearman", "pearson"}
                or key
                in {
                    "loss",
                    "grad_norm",
                    "ranking_loss",
                    "contrastive_loss",
                    "total_loss",
                }
                or key.endswith("_loss")
                or key.endswith("_spearman")
            )
            and isinstance(value, (int, float))
        }
        if not diagnostic_metrics:
            return

        split = "train"
        if any(key.startswith("eval_val2_") for key in diagnostic_metrics):
            split = "eval_val2"
        elif any(key.startswith("train_eval_") for key in diagnostic_metrics):
            split = "train_eval"
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
        warmup_ratio=config.warmup_ratio,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        dataloader_num_workers=config.dataloader_num_workers,
        seed=config.seed,
        fp16=config.fp16,
        bf16=config.bf16,
        optim=config.optim,
        save_safetensors=config.save_safetensors,
        save_total_limit=config.save_total_limit,
        remove_unused_columns=False,
        disable_tqdm=True,
        report_to=config.report_to,
        load_best_model_at_end=True,
        metric_for_best_model="eval_spearman",
        greater_is_better=True,
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
        training_args.reward_objective_gradient_diagnostics_steps = (
            config.objective_gradient_diagnostics_steps
        )
        training_args.reward_metrics_profile = config.metrics_profile
        training_args.reward_protein_shuffle_sensitivity = (
            config.protein_shuffle_sensitivity
        )
        training_args.reward_encoder_learning_rate = config.encoder_learning_rate
        training_args.reward_projection_learning_rate = (
            config.projection_learning_rate
        )
        return training_args
    except ImportError as exc:
        raise ImportError(
            "RewardModel training via Hugging Face Trainer requires the 'accelerate' package. "
            "Install accelerate>=1.1.0 in the reward_model environment."
        ) from exc
