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
    select_fixed_ranking_evaluation_subset,
    validate_tokenized_split_cardinality,
)
from .trainer import RewardModelTrainer, create_training_arguments
from ..model import (
    RewardModel,
    RewardModelConfig,
    load_reward_model,
    load_reward_model_config,
)


_WARM_START_ARCHITECTURE_FIELDS = (
    "protein_model_name_or_path",
    "molecule_model_name_or_path",
    "molecule_input_representation",
    "molecule_trust_remote_code",
    "molecule_deterministic_eval",
    "protein_hidden_size",
    "molecule_hidden_size",
    "fusion_hidden_dim",
    "projection_type",
    "fusion_num_heads",
    "fusion_residual",
    "pooling_type",
    "protein_pooling_type",
    "molecule_pooling_type",
    "pair_scoring_mode",
    "cosine_classification_mlp",
    "cosine_marginal_biases",
)


def _split_dataset_path(tokenized_dataset_dir: str, split_name: str, dataset_kind: str) -> str:
    return os.path.join(os.path.abspath(tokenized_dataset_dir), f"{split_name}_{dataset_kind}")


def _resolve_warm_start_path(
    init_from_checkpoint: str | None,
    output_dir: str,
) -> str | None:
    if init_from_checkpoint is None:
        return None
    checkpoint_path = os.path.realpath(os.path.abspath(init_from_checkpoint))
    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(
            f"Warm-start checkpoint directory does not exist: {checkpoint_path}"
        )
    resolved_output_dir = os.path.realpath(os.path.abspath(output_dir))
    common_path = os.path.commonpath([checkpoint_path, resolved_output_dir])
    if common_path in {checkpoint_path, resolved_output_dir}:
        raise ValueError(
            "Warm-start checkpoint and output directory must be separate, "
            "non-nested directories so phase one cannot be overwritten"
        )
    return checkpoint_path


def _validate_warm_start_architecture(
    checkpoint_config: RewardModelConfig,
    target_config: RewardModelConfig,
) -> None:
    mismatches = []
    for field_name in _WARM_START_ARCHITECTURE_FIELDS:
        checkpoint_value = getattr(checkpoint_config, field_name)
        target_value = getattr(target_config, field_name)
        if target_value is None or checkpoint_value is None:
            continue
        if checkpoint_value != target_value:
            mismatches.append(
                f"{field_name}: checkpoint={checkpoint_value!r}, "
                f"target={target_value!r}"
            )
    if mismatches:
        raise ValueError(
            "Warm-start checkpoint architecture is incompatible with the "
            "target config: " + "; ".join(mismatches)
        )


def _initialize_training_model(
    model_config: RewardModelConfig,
    init_from_checkpoint: str | None,
) -> RewardModel:
    if init_from_checkpoint is None:
        return RewardModel(model_config)
    checkpoint_config = load_reward_model_config(init_from_checkpoint)
    _validate_warm_start_architecture(checkpoint_config, model_config)
    return load_reward_model(
        init_from_checkpoint,
        strict=True,
        config_overrides=model_config.to_dict(),
    )


def prepare_training_examples_from_config(config_path: str) -> Dict[str, Any]:
    config = load_reward_training_config(config_path)
    artifacts = prepare_tokenized_split_datasets(config.data, config.model)
    summary = {
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
    if artifacts.val2_dataset_path is not None:
        summary.update(
            {
                "val2_dataset_path": artifacts.val2_dataset_path,
                "val2_examples": artifacts.val2_examples,
                "val2_groups": artifacts.val2_groups,
            }
        )
    return summary


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
        validate_tokenized_split_cardinality(
            config.data,
            config.model,
            {split_name: example_dataset},
        )
        assay_dataset = RewardAssayListDataset(
            example_dataset,
            seed=config.training.seed,
            ranking_max_ligands=config.data.ranking_max_ligands,
            ranking_opportunity_divisor=config.data.ranking_opportunity_divisor,
            ranking_min_pchembl_span=config.data.ranking_min_pchembl_span,
            max_classification_only_per_item=(
                config.data.max_classification_only_per_item
            ),
            include_all_assays_for_contrastive=(
                config.model.contrastive_loss_weight > 0.0
            ),
        )
        stats = assay_dataset.stats
        if (
            split_name in {"train", "val"}
            and config.model.ranking_loss_weight > 0.0
            and stats.num_ranking_lists == 0
        ):
            raise ValueError(
                f"{split_name.capitalize()} split produced zero eligible ranking lists"
            )
        summaries[f"{split_name}_examples"] = stats.num_examples
        summaries[f"{split_name}_groups"] = stats.num_assays
        summaries[f"{split_name}_eligible_groups"] = stats.num_eligible_assays
        summaries[f"{split_name}_contrastive_lists"] = (
            stats.num_contrastive_lists
        )
        summaries[f"{split_name}_contrastive_examples"] = (
            stats.num_contrastive_examples
        )
        summaries[f"{split_name}_ranking_lists"] = stats.num_ranking_lists
        summaries[f"{split_name}_ranked_examples"] = stats.num_ranked_examples
        summaries[f"{split_name}_classification_only_examples"] = (
            stats.num_classification_only_examples
        )

    return summaries


def train_reward_model_from_config(
    config_path: str,
    init_from_checkpoint: str | None = None,
) -> Dict[str, Any]:
    config = load_reward_training_config(config_path)
    warm_start_path = _resolve_warm_start_path(
        init_from_checkpoint,
        config.training.output_dir,
    )
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
    validate_tokenized_split_cardinality(
        config.data,
        config.model,
        {
            "train": train_examples,
            "val": val_examples,
            "test": test_examples,
        },
    )
    if config.training.metrics_profile == "full":
        for split_name, split_examples in (
            ("val", val_examples),
            ("test", test_examples),
        ):
            if "activity_type" not in split_examples.column_names:
                raise ValueError(
                    f"Tokenized {split_name} examples do not contain activity_type. "
                    "Rerun prepare_reward_training_data.py so Potency/qHTS metrics "
                    "are computed from the current split parquet files."
                )
    elif config.model.classification_loss_weight != 0.0:
        raise ValueError(
            "metrics_profile=ranking requires classification_loss_weight=0.0"
        )
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
        include_all_assays_for_contrastive=(
            config.model.contrastive_loss_weight > 0.0
        ),
        item_count_multiple=item_count_multiple,
    )
    eval_dataset = RewardEvaluationDataset(
        val_examples,
        ranking_min_pchembl_span=config.data.ranking_min_pchembl_span,
        ranking_max_ligands=config.data.ranking_max_ligands,
        ranking_num_partitions=config.data.evaluation_ranking_partitions,
        ranking_partition_seed=config.training.seed,
        protein_shuffle_sensitivity=config.training.protein_shuffle_sensitivity,
    )
    test_eval_dataset = RewardEvaluationDataset(
        test_examples,
        ranking_min_pchembl_span=config.data.ranking_min_pchembl_span,
        ranking_max_ligands=config.data.ranking_max_ligands,
        ranking_num_partitions=config.data.evaluation_ranking_partitions,
        ranking_partition_seed=config.training.seed,
        protein_shuffle_sensitivity=config.training.protein_shuffle_sensitivity,
    )
    val2_eval_dataset = None
    val2_examples = None
    val2_tokenized_dataset_dir = (
        config.data.val2_tokenized_dataset_dir
        or (
            config.data.tokenized_dataset_dir
            if config.data.val2_parquet_path is not None
            else None
        )
    )
    if val2_tokenized_dataset_dir is not None:
        val2_examples_path = _split_dataset_path(
            val2_tokenized_dataset_dir,
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
        validate_tokenized_split_cardinality(
            config.data,
            config.model,
            {"val2": val2_examples},
        )
        if (
            config.training.metrics_profile == "full"
            and config.data.val2_parquet_path is not None
            and "activity_type" not in val2_examples.column_names
        ):
            raise ValueError(
                "Tokenized val2 examples do not contain activity_type. "
                "Rerun prepare_reward_training_data.py."
            )
        val2_eval_dataset = RewardEvaluationDataset(
            val2_examples,
            ranking_min_pchembl_span=config.data.ranking_min_pchembl_span,
            ranking_max_ligands=config.data.ranking_max_ligands,
            ranking_num_partitions=config.data.evaluation_ranking_partitions,
            ranking_partition_seed=config.training.seed,
            protein_shuffle_sensitivity=config.training.protein_shuffle_sensitivity,
        )
    fixed_train_eval_dataset = None
    fixed_train_eval_examples = None
    if config.training.fixed_train_eval_assays > 0:
        fixed_train_eval_examples = select_fixed_ranking_evaluation_subset(
            train_examples,
            num_assays=config.training.fixed_train_eval_assays,
            seed=config.training.seed,
            min_pchembl_span=config.data.ranking_min_pchembl_span,
        )
        fixed_train_eval_dataset = RewardEvaluationDataset(
            fixed_train_eval_examples,
            ranking_min_pchembl_span=config.data.ranking_min_pchembl_span,
            ranking_max_ligands=config.data.ranking_max_ligands,
            ranking_num_partitions=config.data.evaluation_ranking_partitions,
            ranking_partition_seed=config.training.seed,
            protein_shuffle_sensitivity=False,
        )
    model = _initialize_training_model(config.model, warm_start_path)
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
        fixed_train_eval_dataset=fixed_train_eval_dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(config.training.output_dir)
    eval_metrics = trainer.evaluate()
    test_metrics = trainer.evaluate(
        eval_dataset=test_eval_dataset,
        metric_key_prefix="test",
    )

    return {
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "test_examples": len(test_examples),
        "train_groups": len(set(train_examples["group_id"])),
        "val_groups": len(set(val_examples["group_id"])),
        "test_groups": len(set(test_examples["group_id"])),
        "train_ranking_lists": train_dataset.stats.num_ranking_lists,
        "train_ranked_examples": train_dataset.stats.num_ranked_examples,
        "train_contrastive_lists": train_dataset.stats.num_contrastive_lists,
        "train_contrastive_examples": (
            train_dataset.stats.num_contrastive_examples
        ),
        "train_classification_examples": train_dataset.stats.num_examples,
        "output_dir": os.path.abspath(config.training.output_dir),
        "init_from_checkpoint": warm_start_path,
        "optimizer_state_restored": False,
        "eval_metrics": eval_metrics,
        "test_metrics": test_metrics,
        **(
            {}
            if fixed_train_eval_examples is None
            else {
                "fixed_train_eval_examples": len(fixed_train_eval_examples),
                "fixed_train_eval_assays": len(
                    set(fixed_train_eval_examples["group_id"])
                ),
            }
        ),
        **(
            {}
            if val2_eval_dataset is None
            else {
                "val2_examples": len(val2_examples),
            }
        ),
    }
