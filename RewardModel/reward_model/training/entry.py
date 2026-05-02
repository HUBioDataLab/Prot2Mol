from __future__ import annotations

import os
from typing import Any, Dict

from .config import RewardTrainingConfigBundle, load_reward_training_config
from .data import (
    RewardPairCollator,
    RewardPairDataset,
    build_pair_records,
    load_tokenized_example_dataset,
    prepare_tokenized_example_dataset,
    split_tokenized_examples_by_group,
)
from .trainer import RewardModelTrainer, create_training_arguments
from ..model import RewardModel


def prepare_training_examples_from_config(config_path: str) -> Dict[str, Any]:
    config = load_reward_training_config(config_path)
    artifacts = prepare_tokenized_example_dataset(config.data, config.model)
    return {
        "dataset_path": artifacts.dataset_path,
        "num_examples": artifacts.num_examples,
        "num_groups": artifacts.num_groups,
    }


def train_reward_model_from_config(config_path: str) -> Dict[str, Any]:
    config = load_reward_training_config(config_path)
    dataset_path = os.path.abspath(config.data.tokenized_dataset_dir)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Tokenized example dataset not found at {dataset_path}. "
            "Run prepare_reward_training_data.py first."
        )

    tokenized_examples = load_tokenized_example_dataset(dataset_path)
    train_examples, eval_examples, split_stats = split_tokenized_examples_by_group(
        tokenized_examples,
        eval_split_ratio=config.data.eval_split_ratio,
        split_seed=config.data.split_seed,
    )

    train_pairs, train_pair_stats = build_pair_records(train_examples)
    eval_pairs, eval_pair_stats = build_pair_records(eval_examples)
    if not train_pairs:
        raise ValueError("Temporary training split produced zero valid ranking pairs")
    if not eval_pairs:
        raise ValueError("Temporary evaluation split produced zero valid ranking pairs")

    train_dataset = RewardPairDataset(train_examples, train_pairs)
    eval_dataset = RewardPairDataset(eval_examples, eval_pairs)
    collator = RewardPairCollator()

    trainer = RewardModelTrainer(
        model=RewardModel(config.model),
        args=create_training_arguments(config.training),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(config.training.output_dir)
    eval_metrics = trainer.evaluate()

    return {
        "train_examples": split_stats.train_examples,
        "eval_examples": split_stats.eval_examples,
        "train_groups": split_stats.train_groups,
        "eval_groups": split_stats.eval_groups,
        "train_pairs": train_pair_stats.num_pairs,
        "eval_pairs": eval_pair_stats.num_pairs,
        "output_dir": os.path.abspath(config.training.output_dir),
        "eval_metrics": eval_metrics,
    }
