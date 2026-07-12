from __future__ import annotations

import os
import time
from typing import Any, Dict

from .config import RewardTrainingConfigBundle, load_reward_training_config
from .data import (
    RewardPairCollator,
    RewardPairDataset,
    get_saved_pair_dataset_paths,
    get_tokenized_split_dataset_paths,
    load_saved_pair_dataset,
    load_tokenized_example_dataset,
    prepare_tokenized_split_datasets,
    save_pair_dataset_from_example_dataset,
)
from .trainer import RewardModelTrainer, create_training_arguments
from ..model import RewardModel


def _split_dataset_path(tokenized_dataset_dir: str, split_name: str, dataset_kind: str) -> str:
    return os.path.join(os.path.abspath(tokenized_dataset_dir), f"{split_name}_{dataset_kind}")


def prepare_training_examples_from_config(config_path: str) -> Dict[str, Any]:
    config = load_reward_training_config(config_path)
    artifacts = prepare_tokenized_split_datasets(config.data, config.model)
    return {
        "tokenized_dataset_dir": artifacts.base_dir,
        "train_dataset_path": artifacts.train_dataset_path,
        "val_dataset_path": artifacts.val_dataset_path,
        "test_dataset_path": artifacts.test_dataset_path,
        "train_examples": artifacts.train_examples,
        "val_examples": artifacts.val_examples,
        "test_examples": artifacts.test_examples,
        "train_groups": artifacts.train_groups,
        "val_groups": artifacts.val_groups,
        "test_groups": artifacts.test_groups,
    }


def prepare_pair_datasets_from_config(config_path: str) -> Dict[str, Any]:
    config = load_reward_training_config(config_path)
    example_paths = get_tokenized_split_dataset_paths(config.data.tokenized_dataset_dir)
    pair_paths = get_saved_pair_dataset_paths(config.data.tokenized_dataset_dir)

    missing_paths = [
        path
        for path in (example_paths["train"], example_paths["val"], example_paths["test"])
        if not os.path.exists(path)
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Tokenized split datasets not found. "
            f"Missing: {missing_paths}. Run prepare_reward_training_data.py first."
        )

    summaries: Dict[str, Any] = {
        "pair_dataset_dir": os.path.abspath(config.data.tokenized_dataset_dir),
    }
    for split_name in ("train", "val", "test"):
        example_dataset = load_tokenized_example_dataset(example_paths[split_name])
        start_time = time.perf_counter()
        artifacts = save_pair_dataset_from_example_dataset(
            example_dataset,
            pair_paths[split_name],
            split_name=split_name,
        )
        elapsed = time.perf_counter() - start_time
        if split_name in {"train", "val"} and artifacts.num_pairs == 0:
            raise ValueError(f"{split_name.capitalize()} split produced zero valid ranking pairs")

        summaries[f"{split_name}_pair_dataset_path"] = artifacts.dataset_path
        summaries[f"{split_name}_examples"] = len(example_dataset)
        summaries[f"{split_name}_groups"] = artifacts.num_groups
        summaries[f"{split_name}_groups_with_pairs"] = artifacts.num_groups_with_pairs
        summaries[f"{split_name}_pairs"] = artifacts.num_pairs
        summaries[f"{split_name}_pair_build_seconds"] = round(elapsed, 4)

    return summaries


def train_reward_model_from_config(config_path: str) -> Dict[str, Any]:
    config = load_reward_training_config(config_path)
    example_paths = get_tokenized_split_dataset_paths(config.data.tokenized_dataset_dir)
    pair_paths = get_saved_pair_dataset_paths(config.data.tokenized_dataset_dir)

    missing_example_paths = [
        path
        for path in (example_paths["train"], example_paths["val"], example_paths["test"])
        if not os.path.exists(path)
    ]
    if missing_example_paths:
        raise FileNotFoundError(
            "Tokenized split datasets not found. "
            f"Missing: {missing_example_paths}. Run prepare_reward_training_data.py first."
        )

    missing_pair_paths = [
        path
        for path in (pair_paths["train"], pair_paths["val"])
        if not os.path.exists(path)
    ]
    if missing_pair_paths:
        raise FileNotFoundError(
            "Saved pair datasets not found. "
            f"Missing: {missing_pair_paths}. Run prepare_reward_pair_datasets.py first."
        )

    train_examples = load_tokenized_example_dataset(example_paths["train"])
    val_examples = load_tokenized_example_dataset(example_paths["val"])
    test_examples = load_tokenized_example_dataset(example_paths["test"])
    train_pairs = load_saved_pair_dataset(pair_paths["train"])
    eval_pairs = load_saved_pair_dataset(pair_paths["val"])

    train_dataset = RewardPairDataset(train_examples, train_pairs)
    eval_dataset = RewardPairDataset(val_examples, eval_pairs)
    val2_eval_dataset = None
    val2_examples = None
    val2_pairs = None
    if config.data.val2_tokenized_dataset_dir is not None:
        val2_examples_path = _split_dataset_path(
            config.data.val2_tokenized_dataset_dir,
            "val2",
            "examples",
        )
        val2_pairs_path = _split_dataset_path(
            config.data.val2_tokenized_dataset_dir,
            "val2",
            "pairs",
        )
        missing_val2_paths = [
            path
            for path in (val2_examples_path, val2_pairs_path)
            if not os.path.exists(path)
        ]
        if missing_val2_paths:
            raise FileNotFoundError(
                "Semi-seen val2 datasets not found. "
                f"Missing: {missing_val2_paths}."
            )
        val2_examples = load_tokenized_example_dataset(val2_examples_path)
        val2_pairs = load_saved_pair_dataset(val2_pairs_path)
        val2_eval_dataset = RewardPairDataset(val2_examples, val2_pairs)
    collator = RewardPairCollator()

    trainer = RewardModelTrainer(
        model=RewardModel(config.model),
        args=create_training_arguments(config.training),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        val2_eval_dataset=val2_eval_dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(config.training.output_dir)
    eval_metrics = trainer.evaluate()

    return {
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "test_examples": len(test_examples),
        "train_groups": len(set(train_examples["group_id"])),
        "val_groups": len(set(val_examples["group_id"])),
        "test_groups": len(set(test_examples["group_id"])),
        "train_pairs": len(train_pairs),
        "val_pairs": len(eval_pairs),
        "output_dir": os.path.abspath(config.training.output_dir),
        "eval_metrics": eval_metrics,
        **(
            {}
            if val2_eval_dataset is None
            else {
                "val2_examples": len(val2_examples),
                "val2_pairs": len(val2_pairs),
            }
        ),
    }
