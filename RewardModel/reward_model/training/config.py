from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import yaml

from ..model import RewardModelConfig

VALID_TRAINING_MODES = ("auto", "single_gpu", "multi_gpu", "multi_node")


def _resolve_path(path: str, base_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


@dataclass(eq=True)
class RewardTrainingDataConfig:
    train_parquet_path: str
    val_parquet_path: str
    test_parquet_path: str
    tokenized_dataset_dir: str
    val2_tokenized_dataset_dir: Optional[str] = None
    tokenization_batch_size: int = 64
    ranking_max_ligands: int = 16
    ranking_opportunity_divisor: int = 32
    evaluation_ranking_partitions: int = 3
    ranking_min_pchembl_span: float = 0.5
    max_classification_only_per_item: int = 16

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.train_parquet_path:
            raise ValueError("train_parquet_path must be provided")
        if not self.val_parquet_path:
            raise ValueError("val_parquet_path must be provided")
        if not self.test_parquet_path:
            raise ValueError("test_parquet_path must be provided")
        if not self.tokenized_dataset_dir:
            raise ValueError("tokenized_dataset_dir must be provided")
        if self.tokenization_batch_size <= 0:
            raise ValueError("tokenization_batch_size must be > 0")
        if self.ranking_max_ligands < 5:
            raise ValueError("ranking_max_ligands must be >= 5")
        if self.ranking_opportunity_divisor < self.ranking_max_ligands:
            raise ValueError(
                "ranking_opportunity_divisor must be >= ranking_max_ligands"
            )
        if self.evaluation_ranking_partitions <= 0:
            raise ValueError("evaluation_ranking_partitions must be > 0")
        if self.ranking_min_pchembl_span < 0.0:
            raise ValueError("ranking_min_pchembl_span must be >= 0")
        if self.max_classification_only_per_item <= 0:
            raise ValueError("max_classification_only_per_item must be > 0")


@dataclass(eq=True)
class RewardTrainerConfig:
    output_dir: str
    num_train_epochs: float = 1.0
    max_steps: Optional[int] = None
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    eval_steps: Optional[int] = None
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    dataloader_num_workers: int = 0
    dynamic_padding: bool = True
    length_bucketing: bool = True
    length_bucket_size_multiplier: int = 50
    seed: int = 42
    fp16: bool = False
    optim: str = "adamw_torch"
    save_safetensors: bool = False
    save_total_limit: int = 2
    training_mode: str = "single_gpu"
    report_to: Optional[Any] = None
    ranking_score_diagnostics: bool = False
    ranking_score_diagnostic_scales: tuple[float, ...] = (3.0, 5.0, 13.0)

    def __post_init__(self) -> None:
        if self.report_to is None:
            self.report_to = []
        elif isinstance(self.report_to, str):
            self.report_to = [self.report_to]
        else:
            self.report_to = list(self.report_to)
        self.ranking_score_diagnostic_scales = tuple(
            float(scale) for scale in self.ranking_score_diagnostic_scales
        )
        self.validate()

    def validate(self) -> None:
        if not self.output_dir:
            raise ValueError("output_dir must be provided")
        if self.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be > 0")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be > 0 when provided")
        if self.per_device_train_batch_size <= 0:
            raise ValueError("per_device_train_batch_size must be > 0")
        if self.per_device_eval_batch_size <= 0:
            raise ValueError("per_device_eval_batch_size must be > 0")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be > 0")
        if self.eval_steps is not None and self.eval_steps <= 0:
            raise ValueError("eval_steps must be > 0 when provided")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be >= 0")
        if self.max_grad_norm < 0.0:
            raise ValueError("max_grad_norm must be >= 0")
        if self.logging_steps <= 0:
            raise ValueError("logging_steps must be > 0")
        if self.dataloader_num_workers < 0 or self.dataloader_num_workers > 10:
            raise ValueError("dataloader_num_workers must be in [0, 10]")
        if not isinstance(self.optim, str) or not self.optim:
            raise ValueError("optim must be a non-empty string")
        if not isinstance(self.dynamic_padding, bool):
            raise ValueError("dynamic_padding must be a boolean")
        if not isinstance(self.length_bucketing, bool):
            raise ValueError("length_bucketing must be a boolean")
        if self.length_bucket_size_multiplier <= 0:
            raise ValueError("length_bucket_size_multiplier must be > 0")
        if self.save_total_limit <= 0:
            raise ValueError("save_total_limit must be > 0")
        if self.training_mode not in VALID_TRAINING_MODES:
            raise ValueError(
                f"training_mode must be one of {', '.join(VALID_TRAINING_MODES)}"
            )
        if not isinstance(self.ranking_score_diagnostics, bool):
            raise ValueError("ranking_score_diagnostics must be a boolean")
        if not self.ranking_score_diagnostic_scales:
            raise ValueError("ranking_score_diagnostic_scales must not be empty")
        if any(
            scale <= 0.0 or not math.isfinite(scale)
            for scale in self.ranking_score_diagnostic_scales
        ):
            raise ValueError(
                "ranking_score_diagnostic_scales must contain finite values > 0"
            )


@dataclass(eq=True)
class RewardTrainingConfigBundle:
    model: RewardModelConfig
    data: RewardTrainingDataConfig
    training: RewardTrainerConfig


def load_reward_training_config(config_path: str) -> RewardTrainingConfigBundle:
    resolved_path = os.path.abspath(config_path)
    with open(resolved_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Training config at {resolved_path} must contain a mapping")

    if "model" not in payload or "data" not in payload or "training" not in payload:
        raise ValueError(
            "Training config must contain 'model', 'data', and 'training' sections"
        )

    base_dir = os.path.dirname(resolved_path)
    model_config = RewardModelConfig.from_dict(dict(payload["model"]))

    data_section: Dict[str, Any] = dict(payload["data"])
    data_section["train_parquet_path"] = _resolve_path(data_section["train_parquet_path"], base_dir)
    data_section["val_parquet_path"] = _resolve_path(data_section["val_parquet_path"], base_dir)
    data_section["test_parquet_path"] = _resolve_path(data_section["test_parquet_path"], base_dir)
    data_section["tokenized_dataset_dir"] = _resolve_path(data_section["tokenized_dataset_dir"], base_dir)
    if data_section.get("val2_tokenized_dataset_dir") is not None:
        data_section["val2_tokenized_dataset_dir"] = _resolve_path(
            data_section["val2_tokenized_dataset_dir"],
            base_dir,
        )
    data_config = RewardTrainingDataConfig(**data_section)

    training_section: Dict[str, Any] = dict(payload["training"])
    training_section["output_dir"] = _resolve_path(training_section["output_dir"], base_dir)
    training_config = RewardTrainerConfig(**training_section)

    return RewardTrainingConfigBundle(
        model=model_config,
        data=data_config,
        training=training_config,
    )
