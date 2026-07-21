from __future__ import annotations

import os
from typing import Any, Dict

from .config import RewardTrainingConfigBundle, load_reward_training_config
from .data import (
    RewardAssayListCollator,
    RewardAssayListDataset,
    RewardEvaluationDataset,
    get_tokenized_split_dataset_paths,
    load_tokenized_example_dataset,
    prepare_tokenized_split_datasets,
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
    """Summarize dynamic listwise datasets; no pair table is materialized."""
    config = load_reward_training_config(config_path)
    example_paths = get_tokenized_split_dataset_paths(config.data.tokenized_dataset_dir)

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
        "dataset_dir": os.path.abspath(config.data.tokenized_dataset_dir),
        "materialized_pair_tables": False,
    }
    for split_name in ("train", "val", "test"):
        example_dataset = load_tokenized_example_dataset(example_paths[split_name])
        assay_dataset = RewardAssayListDataset(
            example_dataset,
            seed=config.training.seed,
            ranking_max_ligands=config.data.ranking_max_ligands,
            ranking_opportunity_divisor=config.data.ranking_opportunity_divisor,
            ranking_min_pchembl_span=config.data.ranking_min_pchembl_span,
            max_classification_only_per_item=(
                config.data.max_classification_only_per_item
            ),
        )
        stats = assay_dataset.stats
        if split_name in {"train", "val"} and stats.num_ranking_lists == 0:
            raise ValueError(
                f"{split_name.capitalize()} split produced zero eligible ranking lists"
            )
        summaries[f"{split_name}_examples"] = stats.num_examples
        summaries[f"{split_name}_groups"] = stats.num_assays
        summaries[f"{split_name}_eligible_groups"] = stats.num_eligible_assays
        summaries[f"{split_name}_ranking_lists"] = stats.num_ranking_lists
        summaries[f"{split_name}_ranked_examples"] = stats.num_ranked_examples
        summaries[f"{split_name}_classification_only_examples"] = (
            stats.num_classification_only_examples
        )

    return summaries


def train_reward_model_from_config(config_path: str) -> Dict[str, Any]:
    config = load_reward_training_config(config_path)
    example_paths = get_tokenized_split_dataset_paths(config.data.tokenized_dataset_dir)

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

    train_examples = load_tokenized_example_dataset(example_paths["train"])
    val_examples = load_tokenized_example_dataset(example_paths["val"])
    test_examples = load_tokenized_example_dataset(example_paths["test"])
    if config.model.ranking_min_pchembl_span != config.data.ranking_min_pchembl_span:
        raise ValueError(
            "model.ranking_min_pchembl_span and data.ranking_min_pchembl_span must match"
    )
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    item_count_multiple = (
        world_size * config.training.per_device_train_batch_size
        if world_size > 1
        else 1
    )
    train_dataset = RewardAssayListDataset(
        train_examples,
        seed=config.training.seed,
        ranking_max_ligands=config.data.ranking_max_ligands,
        ranking_opportunity_divisor=config.data.ranking_opportunity_divisor,
        ranking_min_pchembl_span=config.data.ranking_min_pchembl_span,
        max_classification_only_per_item=config.data.max_classification_only_per_item,
        item_count_multiple=item_count_multiple,
    )
    eval_dataset = RewardEvaluationDataset(
        val_examples,
        ranking_min_pchembl_span=config.data.ranking_min_pchembl_span,
    )
    val2_eval_dataset = None
    val2_examples = None
    if config.data.val2_tokenized_dataset_dir is not None:
        val2_examples_path = _split_dataset_path(
            config.data.val2_tokenized_dataset_dir,
            "val2",
            "examples",
        )
        missing_val2_paths = [
            path
            for path in (val2_examples_path,)
            if not os.path.exists(path)
        ]
        if missing_val2_paths:
            raise FileNotFoundError(
                "Semi-seen val2 datasets not found. "
                f"Missing: {missing_val2_paths}."
            )
        val2_examples = load_tokenized_example_dataset(val2_examples_path)
        val2_eval_dataset = RewardEvaluationDataset(
            val2_examples,
            ranking_min_pchembl_span=config.data.ranking_min_pchembl_span,
        )
    model = RewardModel(config.model)
    collator = RewardAssayListCollator(
        dynamic_padding=config.training.dynamic_padding,
        protein_pad_token_id=getattr(
            getattr(model, "protein_tokenizer", None),
            "pad_token_id",
            0,
        ),
        molecule_pad_token_id=getattr(
            getattr(model, "molecule_tokenizer", None),
            "pad_token_id",
            0,
        ),
    )

    trainer = RewardModelTrainer(
        model=model,
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
        "train_ranking_lists": train_dataset.stats.num_ranking_lists,
        "train_ranked_examples": train_dataset.stats.num_ranked_examples,
        "train_classification_examples": train_dataset.stats.num_examples,
        "output_dir": os.path.abspath(config.training.output_dir),
        "eval_metrics": eval_metrics,
        **(
            {}
            if val2_eval_dataset is None
            else {
                "val2_examples": len(val2_examples),
            }
        ),
    }
