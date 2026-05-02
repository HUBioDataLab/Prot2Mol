from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping

import yaml

from ..model import RewardModelConfig


def _resolve_path(path: str, base_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


@dataclass(eq=True)
class RewardTrainingDataConfig:
    curated_data_path: str
    tokenized_dataset_dir: str
    tokenization_batch_size: int = 64
    eval_split_ratio: float = 0.05
    split_seed: int = 42

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.curated_data_path:
            raise ValueError("curated_data_path must be provided")
        if not self.tokenized_dataset_dir:
            raise ValueError("tokenized_dataset_dir must be provided")
        if self.tokenization_batch_size <= 0:
            raise ValueError("tokenization_batch_size must be > 0")
        if self.eval_split_ratio <= 0.0 or self.eval_split_ratio >= 1.0:
            raise ValueError("eval_split_ratio must be in (0.0, 1.0)")
        if self.split_seed < 0:
            raise ValueError("split_seed must be >= 0")


@dataclass(eq=True)
class RewardTrainerConfig:
    output_dir: str
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    dataloader_num_workers: int = 0
    seed: int = 42
    fp16: bool = False
    save_safetensors: bool = False
    save_total_limit: int = 2

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.output_dir:
            raise ValueError("output_dir must be provided")
        if self.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be > 0")
        if self.per_device_train_batch_size <= 0:
            raise ValueError("per_device_train_batch_size must be > 0")
        if self.per_device_eval_batch_size <= 0:
            raise ValueError("per_device_eval_batch_size must be > 0")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be > 0")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be >= 0")
        if self.max_grad_norm < 0.0:
            raise ValueError("max_grad_norm must be >= 0")
        if self.logging_steps <= 0:
            raise ValueError("logging_steps must be > 0")
        if self.dataloader_num_workers < 0:
            raise ValueError("dataloader_num_workers must be >= 0")
        if self.save_total_limit <= 0:
            raise ValueError("save_total_limit must be > 0")


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
    data_section["curated_data_path"] = _resolve_path(data_section["curated_data_path"], base_dir)
    data_section["tokenized_dataset_dir"] = _resolve_path(data_section["tokenized_dataset_dir"], base_dir)
    data_config = RewardTrainingDataConfig(**data_section)

    training_section: Dict[str, Any] = dict(payload["training"])
    training_section["output_dir"] = _resolve_path(training_section["output_dir"], base_dir)
    training_config = RewardTrainerConfig(**training_section)

    return RewardTrainingConfigBundle(
        model=model_config,
        data=data_config,
        training=training_config,
    )
