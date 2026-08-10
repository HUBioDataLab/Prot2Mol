import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from transformers import Trainer

from conftest import DummyEncoder, DummyTokenizer
from reward_model.model import LoadedEncoder, RewardModel, RewardModelConfig, load_reward_model
from reward_model.training import (
    RewardAssayListCollator,
    RewardAssayListDataset,
    RewardEvaluationDataset,
    RewardPairCollator,
    RewardPairDataset,
    RewardModelTrainer,
    RewardTrainerConfig,
    build_pair_records,
    compute_activity_type_metrics,
    compute_classification_metrics,
    compute_groupwise_spearman,
    compute_joint_evaluation_metrics,
    compute_pairwise_accuracy,
    compute_protein_shuffle_sensitivity,
    compute_ranking_score_diagnostics,
    create_training_arguments,
    get_saved_pair_dataset_paths,
    get_tokenized_split_dataset_paths,
    load_reward_training_config,
    load_saved_pair_dataset,
    prepare_pair_datasets_from_config,
    save_pair_dataset_from_example_dataset,
)
from reward_model.training.entry import train_reward_model_from_config
from reward_model.training.entry import (
    _resolve_warm_start_path,
    _validate_warm_start_architecture,
)
from reward_model.training.trainer import LengthBucketSampler
from train_reward_model import parse_args


def _dummy_bundles():
    protein_bundle = LoadedEncoder(
        name_or_path="protein/dummy",
        tokenizer=DummyTokenizer(),
        model=DummyEncoder(hidden_size=6),
        hidden_size=6,
    )
    molecule_bundle = LoadedEncoder(
        name_or_path="molecule/dummy",
        tokenizer=DummyTokenizer(),
        model=DummyEncoder(hidden_size=8),
        hidden_size=8,
    )
    return protein_bundle, molecule_bundle


def _pair_ready_examples():
    return Dataset.from_list(
        [
            {
                "example_id": 0,
                "group_id": "T1__A1",
                "target_chembl_id": "T1",
                "assay_id": "A1",
                "compound_id": "M0",
                "protein_input_ids": [1, 1, 0],
                "protein_attention_mask": [1, 1, 0],
                "molecule_input_ids": [7, 8, 0, 0],
                "molecule_attention_mask": [1, 1, 0, 0],
                "binary_label": 0,
                "pchembl_value": 4.0,
            },
            {
                "example_id": 1,
                "group_id": "T1__A1",
                "target_chembl_id": "T1",
                "assay_id": "A1",
                "compound_id": "M1",
                "protein_input_ids": [1, 1, 0],
                "protein_attention_mask": [1, 1, 0],
                "molecule_input_ids": [8, 8, 0, 0],
                "molecule_attention_mask": [1, 1, 0, 0],
                "binary_label": 1,
                "pchembl_value": 7.0,
            },
            {
                "example_id": 2,
                "group_id": "T2__A2",
                "target_chembl_id": "T2",
                "assay_id": "A2",
                "compound_id": "M2",
                "protein_input_ids": [2, 2, 0],
                "protein_attention_mask": [1, 1, 0],
                "molecule_input_ids": [3, 4, 0, 0],
                "molecule_attention_mask": [1, 1, 0, 0],
                "binary_label": 0,
                "pchembl_value": 5.0,
            },
            {
                "example_id": 3,
                "group_id": "T2__A2",
                "target_chembl_id": "T2",
                "assay_id": "A2",
                "compound_id": "M3",
                "protein_input_ids": [2, 2, 0],
                "protein_attention_mask": [1, 1, 0],
                "molecule_input_ids": [5, 6, 0, 0],
                "molecule_attention_mask": [1, 1, 0, 0],
                "binary_label": 1,
                "pchembl_value": 8.0,
            },
        ]
    )


def _metric_ready_examples():
    return Dataset.from_list(
        [
            {
                "example_id": 0,
                "group_id": "T1__A1",
                "target_chembl_id": "T1",
                "assay_id": "A1",
                "compound_id": "M0",
                "protein_input_ids": [1, 1, 0],
                "protein_attention_mask": [1, 1, 0],
                "molecule_input_ids": [7, 8, 0, 0],
                "molecule_attention_mask": [1, 1, 0, 0],
                "binary_label": 0,
                "pchembl_value": 4.0,
            },
            {
                "example_id": 1,
                "group_id": "T1__A1",
                "target_chembl_id": "T1",
                "assay_id": "A1",
                "compound_id": "M1",
                "protein_input_ids": [1, 1, 0],
                "protein_attention_mask": [1, 1, 0],
                "molecule_input_ids": [8, 8, 0, 0],
                "molecule_attention_mask": [1, 1, 0, 0],
                "binary_label": 0,
                "pchembl_value": 6.0,
            },
            {
                "example_id": 2,
                "group_id": "T1__A1",
                "target_chembl_id": "T1",
                "assay_id": "A1",
                "compound_id": "M2",
                "protein_input_ids": [1, 1, 0],
                "protein_attention_mask": [1, 1, 0],
                "molecule_input_ids": [9, 9, 0, 0],
                "molecule_attention_mask": [1, 1, 0, 0],
                "binary_label": 1,
                "pchembl_value": 8.0,
            },
            {
                "example_id": 3,
                "group_id": "T2__A2",
                "target_chembl_id": "T2",
                "assay_id": "A2",
                "compound_id": "M3",
                "protein_input_ids": [2, 2, 0],
                "protein_attention_mask": [1, 1, 0],
                "molecule_input_ids": [3, 4, 0, 0],
                "molecule_attention_mask": [1, 1, 0, 0],
                "binary_label": 0,
                "pchembl_value": 4.5,
            },
            {
                "example_id": 4,
                "group_id": "T2__A2",
                "target_chembl_id": "T2",
                "assay_id": "A2",
                "compound_id": "M4",
                "protein_input_ids": [2, 2, 0],
                "protein_attention_mask": [1, 1, 0],
                "molecule_input_ids": [4, 5, 0, 0],
                "molecule_attention_mask": [1, 1, 0, 0],
                "binary_label": 1,
                "pchembl_value": 7.0,
            },
            {
                "example_id": 5,
                "group_id": "T2__A2",
                "target_chembl_id": "T2",
                "assay_id": "A2",
                "compound_id": "M5",
                "protein_input_ids": [2, 2, 0],
                "protein_attention_mask": [1, 1, 0],
                "molecule_input_ids": [5, 6, 0, 0],
                "molecule_attention_mask": [1, 1, 0, 0],
                "binary_label": 1,
                "pchembl_value": 8.5,
            },
        ]
    )


def test_metric_helpers_compute_expected_values():
    classification_metrics = compute_classification_metrics(
        probabilities=[0.1, 0.4, 0.8, 0.9],
        labels=[0, 0, 1, 1],
    )
    assert classification_metrics["eval_mcc"] == pytest.approx(1.0)
    assert classification_metrics["eval_f1"] == pytest.approx(1.0)
    assert classification_metrics["eval_roc_auc"] == pytest.approx(1.0)
    assert classification_metrics["eval_precision"] == pytest.approx(1.0)
    assert classification_metrics["eval_recall"] == pytest.approx(1.0)
    assert classification_metrics["eval_accuracy"] == pytest.approx(1.0)

    pairwise_accuracy = compute_pairwise_accuracy(
        positive_scores=[2.0, 1.5, 0.0],
        negative_scores=[1.0, 2.0, -1.0],
    )
    assert pairwise_accuracy == pytest.approx(2.0 / 3.0)

    spearman_metrics = compute_groupwise_spearman(
        group_ids=[
            "T1__A1",
            "T1__A1",
            "T1__A1",
            "T2__A2",
            "T2__A2",
            "T2__A2",
            "T2__A2",
            "T3__A3",
            "T3__A3",
            "T3__A3",
        ],
        ranking_scores=[1.0, 2.0, 3.0, 1.0, 2.0, 4.0, 3.0, 4.0, 5.0, 6.0],
        pchembl_values=[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 1.0, 1.0],
        min_group_size=3,
    )
    assert spearman_metrics["eval_spearman"] == pytest.approx((3.0 * 1.0 + 4.0 * 0.8) / 7.0)
    assert spearman_metrics["eval_weighted_spearman"] == pytest.approx(
        spearman_metrics["eval_spearman"]
    )
    assert spearman_metrics["eval_macro_spearman"] == pytest.approx(0.9)
    assert spearman_metrics["eval_spearman_num_groups"] == pytest.approx(2.0)


def test_activity_type_metrics_separate_potency_and_non_potency_assays():
    metrics = compute_activity_type_metrics(
        probabilities=[0.1, 0.8, 0.9, 0.8, 0.4, 0.2],
        labels=[0, 1, 1, 1, 0, 0],
        activity_types=["Potency", "Potency", "Potency", "Binding", "Binding", "Binding"],
        group_ids=["T1__A1"] * 3 + ["T2__A2"] * 3,
        ranking_scores=[1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
        pchembl_values=[4.0, 5.0, 6.0, 4.0, 5.0, 6.0],
        min_pchembl_span=0.5,
    )

    assert metrics["eval_potency_qhts_num_examples"] == 3
    assert metrics["eval_non_potency_num_examples"] == 3
    assert metrics["eval_potency_qhts_macro_spearman"] == pytest.approx(1.0)
    assert metrics["eval_non_potency_macro_spearman"] == pytest.approx(-1.0)


def test_protein_shuffle_sensitivity_reports_rank_collapse():
    metrics = compute_protein_shuffle_sensitivity(
        group_ids=["T1__A1"] * 3 + ["T2__A2"] * 4,
        baseline_scores=[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0],
        shuffled_scores=[3.0, 2.0, 1.0, 4.0, 3.0, 2.0, 1.0],
        pchembl_values=[4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 7.0],
        min_pchembl_span=0.5,
    )

    assert metrics["eval_protein_shuffled_macro_spearman"] == pytest.approx(-1.0)
    assert metrics["eval_protein_shuffle_macro_spearman_drop"] == pytest.approx(2.0)
    assert metrics["eval_protein_shuffle_macro_rank_stability"] == pytest.approx(-1.0)


def test_ranking_score_diagnostics_measure_cosine_margin_and_entropy():
    metrics = compute_ranking_score_diagnostics(
        ranking_scores=torch.tensor([3.0, 1.0, -2.0, 100.0]),
        cosine_similarities=torch.tensor([0.3, 0.1, -0.2, 0.9]),
        pchembl_values=torch.tensor([8.0, 7.0, 6.0, 1.0]),
        ranking_group_ids=torch.tensor([0, 0, 0, -1]),
        temperature=1.0,
        affinity_margin=0.5,
    )

    assert metrics["ranking_cosine_mean"] == pytest.approx(0.2 / 3.0)
    assert metrics["ranking_cosine_std"] == pytest.approx(
        torch.tensor([0.3, 0.1, -0.2]).std(unbiased=False).item()
    )
    assert metrics["ranking_cosine_p01"] < metrics["ranking_cosine_p99"]
    assert metrics["ranking_margin_pair_accuracy"] == pytest.approx(1.0)
    assert metrics["ranking_margin_pair_gap_p50"] == pytest.approx(3.0)
    assert metrics["ranking_margin_pair_count"] == pytest.approx(3.0)
    assert 0.0 < metrics["ranking_list_normalized_entropy"] < 1.0
    assert not any("tanh" in key for key in metrics)
    assert not any(key.startswith("ranking_score_") for key in metrics)


def test_ranking_score_diagnostics_exclude_pairs_inside_affinity_margin():
    metrics = compute_ranking_score_diagnostics(
        ranking_scores=torch.tensor([3.0, -100.0, -2.0]),
        pchembl_values=torch.tensor([8.0, 7.8, 6.0]),
        ranking_group_ids=torch.tensor([0, 0, 0]),
        affinity_margin=0.5,
    )

    assert metrics["ranking_margin_pair_count"] == pytest.approx(2.0)
    assert metrics["ranking_margin_pair_accuracy"] == pytest.approx(0.5)


def test_ranking_metrics_profile_returns_only_decision_metrics():
    metrics, assay_records = compute_joint_evaluation_metrics(
        activity_logits=torch.tensor([-0.4, -0.2, 0.0, 0.2, 0.4]),
        ranking_scores=torch.tensor([-0.4, -0.2, 0.0, 0.2, 0.4]),
        activity_labels=torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]),
        pchembl_values=torch.tensor([5.0, 6.0, 7.0, 8.0, 9.0]),
        ranking_group_ids=torch.zeros(5, dtype=torch.long),
        group_id_names=["T1__A1"],
        classification_loss_weight=0.0,
        ranking_loss_weight=0.5,
        bce_pos_weight=1.0,
        ranking_temperature=1.0,
        ranking_affinity_margin=0.5,
        ranking_min_pchembl_span=0.5,
        ranking_score_diagnostics=True,
        cosine_similarities=torch.tensor([-0.4, -0.2, 0.0, 0.2, 0.4]),
        metrics_profile="ranking",
        contrastive_loss=torch.tensor(2.0),
        contrastive_loss_weight=0.5,
    )

    assert set(metrics) == {
        "eval_loss",
        "eval_ranking_loss",
        "eval_contrastive_loss",
        "eval_spearman",
        "eval_pearson",
        "eval_cosine_std",
        "eval_pair_accuracy",
    }
    assert metrics["eval_spearman"] == pytest.approx(1.0)
    assert metrics["eval_pearson"] == pytest.approx(1.0)
    assert metrics["eval_pair_accuracy"] == pytest.approx(1.0)
    assert metrics["eval_loss"] == pytest.approx(
        0.5 * metrics["eval_ranking_loss"]
        + 0.5 * metrics["eval_contrastive_loss"]
    )
    assert assay_records


def test_reward_model_trainer_runs_and_saves_checkpoint(tmp_path, monkeypatch):
    pytest.importorskip("accelerate")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("WANDB_MODE", "disabled")

    protein_bundle, molecule_bundle = _dummy_bundles()
    examples = _metric_ready_examples()
    train_dataset = RewardAssayListDataset(
        examples,
        seed=42,
        max_classification_only_per_item=4,
        item_count_multiple=2,
    )
    eval_dataset = RewardEvaluationDataset(examples)

    model = RewardModel(
        config=RewardModelConfig(
            protein_model_name_or_path="protein/dummy",
            molecule_model_name_or_path="molecule/dummy",
            fusion_hidden_dim=10,
            fusion_num_heads=2,
            pair_scoring_mode="scaled_cosine",
            dropout=0.0,
        ),
        protein_bundle=protein_bundle,
        molecule_bundle=molecule_bundle,
    )
    trainer = RewardModelTrainer(
        model=model,
        args=create_training_arguments(
            RewardTrainerConfig(
                output_dir=str(tmp_path / "trainer_output"),
                num_train_epochs=10,
                max_steps=1,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                logging_steps=1,
                fp16=False,
                ranking_score_diagnostics=True,
            )
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        val2_eval_dataset=eval_dataset,
        data_collator=RewardAssayListCollator(),
    )

    train_result = trainer.train()

    eval_metrics = trainer.evaluate()
    save_dir = tmp_path / "saved_model"
    trainer.save_model(str(save_dir))

    assert train_result.training_loss >= 0.0
    assert trainer.state.global_step == 1
    assert "eval_ranking_loss" in eval_metrics
    assert "eval_classification_loss" in eval_metrics
    assert eval_metrics["eval_num_examples"] == len(examples)
    assert "eval_mcc" in eval_metrics
    assert "eval_f1" in eval_metrics
    assert "eval_roc_auc" in eval_metrics
    assert "eval_precision" in eval_metrics
    assert "eval_recall" in eval_metrics
    assert "eval_accuracy" in eval_metrics
    assert "eval_spearman" in eval_metrics
    assert "eval_spearman_num_groups" in eval_metrics
    assert "eval_ranking_cosine_std" in eval_metrics
    assert "eval_cosine_scale" in eval_metrics
    assert "eval_classification_logit_bias" in eval_metrics
    assert "eval_ranking_list_normalized_entropy" in eval_metrics
    assert "eval_ranking_margin_pair_accuracy" in eval_metrics
    assert "eval_val2_loss" in eval_metrics
    assert "eval_val2_ranking_loss" in eval_metrics
    assert "eval_val2_classification_loss" in eval_metrics
    assert "eval_val2_mcc" in eval_metrics
    assert "eval_val2_f1" in eval_metrics
    assert "eval_val2_roc_auc" in eval_metrics
    assert "eval_val2_precision" in eval_metrics
    assert "eval_val2_recall" in eval_metrics
    assert "eval_val2_accuracy" in eval_metrics
    assert "eval_val2_spearman" in eval_metrics
    assert "eval_val2_spearman_num_groups" in eval_metrics
    assert "eval_val2_ranking_cosine_std" in eval_metrics
    assert "eval_val2_cosine_scale" in eval_metrics
    assay_log_path = tmp_path / "trainer_output" / "eval_assay_spearman.jsonl"
    val2_assay_log_path = tmp_path / "trainer_output" / "eval_val2_assay_spearman.jsonl"
    assay_log_records = [
        json.loads(line)
        for line in assay_log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(assay_log_records) >= 1
    latest_assay_log = assay_log_records[-1]
    assert latest_assay_log["global_step"] == trainer.state.global_step
    if math.isnan(eval_metrics["eval_spearman"]):
        assert math.isnan(latest_assay_log["weighted_spearman"])
    else:
        assert latest_assay_log["weighted_spearman"] == pytest.approx(
            eval_metrics["eval_spearman"]
        )
    assert latest_assay_log["num_eligible_groups"] == pytest.approx(
        eval_metrics["eval_spearman_num_groups"]
    )
    assert latest_assay_log["assays"]
    assert {
        "group_id",
        "target_chembl_id",
        "assay_id",
        "num_examples",
        "spearman",
    }.issubset(latest_assay_log["assays"][0])
    val2_assay_log_records = [
        json.loads(line)
        for line in val2_assay_log_path.read_text(encoding="utf-8").splitlines()
    ]
    if math.isnan(eval_metrics["eval_val2_spearman"]):
        assert math.isnan(val2_assay_log_records[-1]["weighted_spearman"])
    else:
        assert val2_assay_log_records[-1]["weighted_spearman"] == pytest.approx(
            eval_metrics["eval_val2_spearman"]
        )
    training_logs = [
        entry for entry in trainer.state.log_history if "ranking_loss" in entry
    ]
    assert training_logs
    assert {
        "classification_accuracy",
        "classification_mcc",
        "classification_f1",
        "classification_auroc",
        "ranking_cosine_std",
        "cosine_scale",
        "classification_logit_bias",
        "ranking_list_normalized_entropy",
        "ranking_margin_pair_accuracy",
    }.issubset(training_logs[-1])
    diagnostic_log_path = (
        tmp_path / "trainer_output" / "ranking_score_diagnostics.jsonl"
    )
    diagnostic_records = [
        json.loads(line)
        for line in diagnostic_log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert {record["split"] for record in diagnostic_records} >= {
        "train",
        "eval",
        "eval_val2",
    }
    train_diagnostic_record = next(
        record for record in diagnostic_records if record["split"] == "train"
    )
    eval_diagnostic_record = next(
        record for record in diagnostic_records if record["split"] == "eval"
    )
    assert "ranking_loss" in train_diagnostic_record["metrics"]
    assert "eval_spearman" in eval_diagnostic_record["metrics"]
    assert os.path.exists(save_dir / "pytorch_model.bin")
    assert os.path.exists(save_dir / "config.json")

    reloaded = load_reward_model(
        str(save_dir),
        protein_bundle=protein_bundle,
        molecule_bundle=molecule_bundle,
    )
    assert reloaded.config.fusion_hidden_dim == 10


def test_contrastive_reward_trainer_runs_one_step_and_logs_both_losses(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("accelerate")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("WANDB_MODE", "disabled")

    protein_bundle, molecule_bundle = _dummy_bundles()
    examples = _metric_ready_examples()
    train_dataset = RewardAssayListDataset(
        examples,
        seed=42,
        max_classification_only_per_item=0,
    )
    eval_dataset = RewardEvaluationDataset(
        examples,
        protein_shuffle_sensitivity=False,
    )
    model = RewardModel(
        config=RewardModelConfig(
            protein_model_name_or_path="protein/dummy",
            molecule_model_name_or_path="molecule/dummy",
            fusion_hidden_dim=10,
            fusion_num_heads=2,
            pair_scoring_mode="cosine",
            ranking_temperature=0.1,
            ranking_loss_weight=0.5,
            contrastive_loss_weight=0.5,
            classification_loss_weight=0.0,
            dropout=0.0,
        ),
        protein_bundle=protein_bundle,
        molecule_bundle=molecule_bundle,
    )
    trainer = RewardModelTrainer(
        model=model,
        args=create_training_arguments(
            RewardTrainerConfig(
                output_dir=str(tmp_path / "contrastive_trainer_output"),
                num_train_epochs=10,
                max_steps=1,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                logging_steps=1,
                fp16=False,
                metrics_profile="ranking",
                ranking_score_diagnostics=True,
                protein_shuffle_sensitivity=False,
            )
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=RewardAssayListCollator(),
    )

    result = trainer.train()
    eval_metrics = trainer.evaluate()

    assert math.isfinite(result.training_loss)
    assert trainer.state.global_step == 1
    objective_logs = [
        entry
        for entry in trainer.state.log_history
        if "contrastive_loss" in entry
    ]
    assert objective_logs
    assert math.isfinite(objective_logs[-1]["ranking_loss"])
    assert objective_logs[-1]["contrastive_loss"] > 0.0
    assert set(eval_metrics) >= {
        "eval_loss",
        "eval_ranking_loss",
        "eval_contrastive_loss",
        "eval_spearman",
        "eval_pearson",
        "eval_pair_accuracy",
    }
    assert eval_metrics["eval_loss"] == pytest.approx(
        0.5 * eval_metrics["eval_ranking_loss"]
        + 0.5 * eval_metrics["eval_contrastive_loss"]
    )


def test_create_training_arguments_uses_step_based_schedule_when_eval_steps_is_set(tmp_path):
    args = create_training_arguments(
        RewardTrainerConfig(
            output_dir=str(tmp_path / "trainer_output"),
            eval_steps=25,
            max_steps=1000,
            fp16=False,
        )
    )

    def _strategy_value(value):
        return value.value if hasattr(value, "value") else value

    assert _strategy_value(args.save_strategy) == "steps"
    assert args.eval_steps == 25
    assert args.save_steps == 25
    assert args.max_steps == 1000
    assert args.metric_for_best_model == "eval_spearman"
    assert args.greater_is_better is True
    if hasattr(args, "evaluation_strategy"):
        assert _strategy_value(args.evaluation_strategy) == "steps"
    if hasattr(args, "eval_strategy"):
        assert _strategy_value(args.eval_strategy) == "steps"


def test_create_training_arguments_defaults_to_no_reporters(tmp_path):
    args = create_training_arguments(
        RewardTrainerConfig(
            output_dir=str(tmp_path / "trainer_output"),
            fp16=False,
        )
    )

    assert "wandb" not in list(args.report_to)


def test_create_training_arguments_preserves_component_learning_rates(tmp_path):
    args = create_training_arguments(
        RewardTrainerConfig(
            output_dir=str(tmp_path / "trainer_output"),
            encoder_learning_rate=1.0e-5,
            projection_learning_rate=1.0e-3,
        )
    )

    assert args.reward_encoder_learning_rate == pytest.approx(1.0e-5)
    assert args.reward_projection_learning_rate == pytest.approx(1.0e-3)


def test_reward_trainer_builds_encoder_and_projection_lr_groups(tmp_path):
    class ComponentModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.protein_encoder = torch.nn.Linear(2, 2)
            self.molecule_encoder = torch.nn.Linear(2, 2)
            self.protein_projection = torch.nn.Linear(2, 2)
            self.molecule_projection = torch.nn.Linear(2, 2)

    model = ComponentModel()
    trainer = RewardModelTrainer(
        model=model,
        eval_dataset=[0],
        args=create_training_arguments(
            RewardTrainerConfig(
                output_dir=str(tmp_path / "trainer_output"),
                learning_rate=1.0e-5,
                encoder_learning_rate=1.0e-5,
                projection_learning_rate=1.0e-3,
                optim="adamw_torch",
            )
        ),
    )

    optimizer = trainer.create_optimizer()
    lr_by_parameter_id = {
        id(parameter): group["lr"]
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    for name, parameter in model.named_parameters():
        expected = 1.0e-5 if "encoder" in name else 1.0e-3
        assert lr_by_parameter_id[id(parameter)] == pytest.approx(expected)


def test_reward_trainer_log_supports_transformers_without_start_time(monkeypatch):
    captured = {}

    def _legacy_log(self, logs):
        captured.update(logs)
        return "logged"

    monkeypatch.setattr(Trainer, "log", _legacy_log)
    trainer = object.__new__(RewardModelTrainer)

    assert trainer.log({"eval_loss": 0.25}, start_time=123.0) == "logged"
    assert captured == {"eval_loss": 0.25}


def test_ranking_metrics_profile_filters_reporter_only_fields(monkeypatch):
    captured = {}

    def _legacy_log(self, logs):
        captured.update(logs)
        return "logged"

    monkeypatch.setattr(Trainer, "log", _legacy_log)
    trainer = object.__new__(RewardModelTrainer)
    trainer.args = SimpleNamespace(reward_metrics_profile="ranking")
    trainer.model = torch.nn.Module()
    trainer.model.config = RewardModelConfig(pair_scoring_mode="cosine")
    trainer._consume_train_component_logs = lambda: {
        "cosine_std": 0.12,
        "pair_accuracy": 0.75,
        "spearman": 0.3,
        "pearson": 0.4,
    }
    trainer._append_ranking_score_diagnostics_log = lambda logs: None

    result = trainer.log(
        {
            "loss": 0.25,
            "grad_norm": 1.5,
            "learning_rate": 1.0e-5,
            "epoch": 0.1,
            "num_examples": 48,
            "train_runtime": 20.0,
        }
    )

    assert result == "logged"
    assert captured == {
        "loss": 0.25,
        "grad_norm": 1.5,
        "learning_rate": 1.0e-5,
        "epoch": 0.1,
        "cosine_std": 0.12,
        "pair_accuracy": 0.75,
        "spearman": 0.3,
        "pearson": 0.4,
    }


def test_reward_trainer_logs_scaled_cosine_parameters(monkeypatch):
    captured = {}

    def _legacy_log(self, logs):
        captured.update(logs)
        return "logged"

    monkeypatch.setattr(Trainer, "log", _legacy_log)
    trainer = object.__new__(RewardModelTrainer)
    trainer.model = torch.nn.Module()
    trainer.model.config = RewardModelConfig(
        pair_scoring_mode="scaled_cosine",
        cosine_scale_init=13.0,
        cosine_classification_bias_init=-0.5,
    )
    trainer.model.logit_scale = torch.nn.Parameter(torch.tensor(math.log(13.0)))
    trainer.model.classification_logit_bias = torch.nn.Parameter(torch.tensor(-0.5))
    trainer._consume_train_component_logs = lambda: {}
    trainer._append_ranking_score_diagnostics_log = lambda logs: None

    assert trainer.log({"loss": 0.25}) == "logged"
    assert captured["cosine_scale"] == pytest.approx(13.0)
    assert captured["classification_logit_bias"] == pytest.approx(-0.5)


def test_training_metrics_are_count_weighted_and_ranking_ties_are_excluded():
    trainer = object.__new__(RewardModelTrainer)
    trainer._reset_train_component_accumulator()
    trainer._record_train_components(
        ranking_loss=torch.tensor(3.0),
        classification_loss=torch.tensor(0.6),
        num_examples=torch.tensor(3),
        num_ranking_lists=torch.tensor(1),
        num_ranked_examples=torch.tensor(3),
        activity_logits=torch.tensor([1.0, -1.0, 1.0]),
        activity_labels=torch.tensor([1.0, 0.0, 0.0]),
        ranking_score=torch.tensor([3.0, 2.0, 1.0]),
        pchembl_values=torch.tensor([8.0, 7.0, 6.0]),
        ranking_group_ids=torch.tensor([0, 0, 0]),
    )
    trainer._record_train_components(
        ranking_loss=torch.tensor(2.0),
        classification_loss=torch.tensor(0.4),
        num_examples=torch.tensor(3),
        num_ranking_lists=torch.tensor(1),
        num_ranked_examples=torch.tensor(3),
        activity_logits=torch.tensor([-1.0, 1.0, -1.0]),
        activity_labels=torch.tensor([1.0, 1.0, 0.0]),
        ranking_score=torch.tensor([0.0, 1.0, 2.0]),
        pchembl_values=torch.tensor([8.0, 7.0, 7.0]),
        ranking_group_ids=torch.tensor([0, 0, 0]),
    )

    logs = trainer._consume_train_component_logs()

    assert logs["classification_accuracy"] == pytest.approx(4.0 / 6.0)
    assert logs["classification_mcc"] == pytest.approx(1.0 / 3.0)
    assert logs["classification_f1"] == pytest.approx(2.0 / 3.0)
    assert logs["classification_auroc"] == pytest.approx(2.0 / 3.0)
    assert logs["ranking_loss"] == pytest.approx(2.5)
    assert "total_loss" not in logs
    assert "ranking_pairwise_accuracy" not in logs
    assert "ranking_loss_per_ranked_example" not in logs


def test_ranking_metrics_profile_keeps_training_logs_slim():
    trainer = object.__new__(RewardModelTrainer)
    trainer.args = SimpleNamespace(
        reward_metrics_profile="ranking",
        reward_ranking_score_diagnostics=True,
    )
    trainer.model = torch.nn.Module()
    trainer.model.config = RewardModelConfig(
        pair_scoring_mode="cosine",
        classification_loss_weight=0.0,
    )
    trainer._reset_train_component_accumulator()
    trainer._record_train_components(
        ranking_loss=torch.tensor(1.25),
        classification_loss=torch.tensor(0.75),
        num_examples=torch.tensor(3),
        num_ranking_lists=torch.tensor(1),
        num_ranked_examples=torch.tensor(3),
        activity_logits=torch.tensor([-0.3, 0.0, 0.3]),
        activity_labels=torch.tensor([0.0, 0.0, 1.0]),
        ranking_score=torch.tensor([-0.3, 0.0, 0.3]),
        pchembl_values=torch.tensor([5.0, 6.0, 7.0]),
        ranking_group_ids=torch.tensor([0, 0, 0]),
        cosine_similarity=torch.tensor([-0.3, 0.0, 0.3]),
    )
    trainer._record_train_components(
        ranking_loss=torch.tensor(1.5),
        classification_loss=torch.tensor(0.5),
        num_examples=torch.tensor(5),
        num_ranking_lists=torch.tensor(1),
        num_ranked_examples=torch.tensor(5),
        activity_logits=torch.tensor([0.5, 0.25, 0.0, -0.25, -0.5]),
        activity_labels=torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]),
        ranking_score=torch.tensor([0.5, 0.25, 0.0, -0.25, -0.5]),
        pchembl_values=torch.tensor([5.0, 6.0, 7.0, 8.0, 9.0]),
        ranking_group_ids=torch.zeros(5, dtype=torch.long),
        cosine_similarity=torch.tensor([0.5, 0.25, 0.0, -0.25, -0.5]),
    )

    logs = trainer._consume_train_component_logs()

    assert set(logs) == {"cosine_std", "pair_accuracy", "spearman", "pearson"}
    assert logs["cosine_std"] == pytest.approx(
        torch.tensor([-0.3, 0.0, 0.3, 0.5, 0.25, 0.0, -0.25, -0.5])
        .std(unbiased=False)
        .item()
    )
    assert logs["pair_accuracy"] == pytest.approx(3.0 / 13.0)
    assert logs["spearman"] == pytest.approx(-0.25)
    assert logs["pearson"] == pytest.approx(-0.25)


def test_contrastive_ranking_profile_adds_only_objective_component_losses():
    trainer = object.__new__(RewardModelTrainer)
    trainer.args = SimpleNamespace(
        reward_metrics_profile="ranking",
        reward_ranking_score_diagnostics=True,
    )
    trainer.model = torch.nn.Module()
    trainer.model.config = RewardModelConfig(
        pair_scoring_mode="cosine",
        ranking_loss_weight=0.5,
        contrastive_loss_weight=0.5,
        classification_loss_weight=0.0,
    )
    trainer._reset_train_component_accumulator()
    trainer._record_train_components(
        ranking_loss=torch.tensor(1.25),
        contrastive_loss=torch.tensor(2.75),
        classification_loss=None,
        num_examples=torch.tensor(3),
        num_ranking_lists=torch.tensor(1),
        num_ranked_examples=torch.tensor(3),
        activity_logits=torch.tensor([-0.3, 0.0, 0.3]),
        activity_labels=torch.tensor([0.0, 0.0, 1.0]),
        ranking_score=torch.tensor([-0.3, 0.0, 0.3]),
        pchembl_values=torch.tensor([5.0, 6.0, 7.0]),
        ranking_group_ids=torch.tensor([0, 0, 0]),
        cosine_similarity=torch.tensor([-0.3, 0.0, 0.3]),
    )

    logs = trainer._consume_train_component_logs()

    assert set(logs) == {
        "ranking_loss",
        "contrastive_loss",
        "cosine_std",
        "pair_accuracy",
        "spearman",
        "pearson",
    }
    assert logs["ranking_loss"] == pytest.approx(1.25)
    assert logs["contrastive_loss"] == pytest.approx(2.75)


def test_create_training_arguments_supports_fused_adamw(tmp_path):
    args = create_training_arguments(
        RewardTrainerConfig(
            output_dir=str(tmp_path / "trainer_output"),
            optim="adamw_torch_fused",
            fp16=False,
        )
    )

    optim_value = args.optim.value if hasattr(args.optim, "value") else str(args.optim)
    assert optim_value == "adamw_torch_fused"


def test_trainer_config_enforces_shared_server_worker_limit(tmp_path):
    with pytest.raises(ValueError, match=r"\[0, 10\]"):
        RewardTrainerConfig(
            output_dir=str(tmp_path / "trainer_output"),
            dataloader_num_workers=11,
        )


def test_reward_trainer_config_rejects_multiple_mixed_precision_modes(tmp_path):
    with pytest.raises(ValueError, match="fp16 and bf16 cannot both be enabled"):
        RewardTrainerConfig(
            output_dir=str(tmp_path / "trainer_output"),
            fp16=True,
            bf16=True,
        )


@pytest.mark.parametrize("max_steps", [0, -1])
def test_reward_trainer_config_rejects_invalid_max_steps(tmp_path, max_steps):
    with pytest.raises(ValueError, match="max_steps must be > 0"):
        RewardTrainerConfig(
            output_dir=str(tmp_path / "trainer_output"),
            max_steps=max_steps,
        )


def test_create_training_arguments_enables_length_bucketing_by_default(tmp_path):
    args = create_training_arguments(
        RewardTrainerConfig(
            output_dir=str(tmp_path / "trainer_output"),
            length_bucket_size_multiplier=7,
            fp16=False,
        )
    )

    assert args.reward_length_bucketing is True
    assert args.length_bucket_size_multiplier == 7


def test_length_bucket_sampler_groups_similar_lengths_and_changes_by_epoch():
    lengths = [1, 2, 3, 4, 100, 101, 102, 103]
    fully_grouped_sampler = LengthBucketSampler(
        lengths,
        batch_size=2,
        bucket_size_multiplier=4,
        seed=11,
    )

    first_epoch = list(fully_grouped_sampler)
    batches = [first_epoch[start : start + 2] for start in range(0, len(first_epoch), 2)]
    assert sorted(first_epoch) == list(range(len(lengths)))
    assert all(
        max(lengths[index] for index in batch) - min(lengths[index] for index in batch) <= 1
        for batch in batches
    )

    epoch_sampler = LengthBucketSampler(
        lengths,
        batch_size=2,
        bucket_size_multiplier=2,
        seed=11,
    )
    epoch_zero = list(epoch_sampler)
    epoch_sampler.set_epoch(1)
    assert list(epoch_sampler) != epoch_zero


def test_create_training_arguments_rejects_distributed_env_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")

    with pytest.raises(ValueError, match="training_mode=single_gpu"):
        create_training_arguments(
            RewardTrainerConfig(
                output_dir=str(tmp_path / "trainer_output"),
                fp16=False,
            )
        )


def test_reward_training_arguments_rejects_incomplete_multi_gpu_env(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    monkeypatch.delenv("MASTER_PORT", raising=False)

    with pytest.raises(ValueError, match="Distributed launch environment is incomplete"):
        create_training_arguments(
            RewardTrainerConfig(
                output_dir=str(tmp_path / "trainer_output"),
                fp16=False,
                training_mode="multi_gpu",
            )
        )


def test_reward_training_arguments_rejects_invalid_distributed_rank(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")

    with pytest.raises(ValueError, match="Invalid distributed rank"):
        create_training_arguments(
            RewardTrainerConfig(
                output_dir=str(tmp_path / "trainer_output"),
                fp16=False,
                training_mode="multi_gpu",
            )
        )


def test_reward_trainer_config_normalizes_report_to_string(tmp_path):
    config = RewardTrainerConfig(
        output_dir=str(tmp_path / "trainer_output"),
        report_to="wandb",
    )

    assert config.report_to == ["wandb"]


def test_warm_start_path_requires_existing_disjoint_directory(tmp_path):
    checkpoint = tmp_path / "phase1" / "checkpoint-1000"
    checkpoint.mkdir(parents=True)
    output = tmp_path / "phase2"

    assert _resolve_warm_start_path(str(checkpoint), str(output)) == str(
        checkpoint.resolve()
    )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _resolve_warm_start_path(str(tmp_path / "missing"), str(output))
    with pytest.raises(ValueError, match="separate, non-nested"):
        _resolve_warm_start_path(str(checkpoint), str(tmp_path / "phase1"))
    with pytest.raises(ValueError, match="separate, non-nested"):
        _resolve_warm_start_path(str(checkpoint), str(checkpoint / "phase2"))


def test_warm_start_architecture_allows_freeze_change_but_rejects_shape_change():
    checkpoint_config = RewardModelConfig(
        fusion_hidden_dim=512,
        fusion_num_heads=8,
        fusion_residual=True,
        pooling_type="mean",
        pair_scoring_mode="scaled_cosine",
        freeze_protein_encoder=True,
        freeze_molecule_encoder=True,
    )
    target_config = RewardModelConfig.from_dict(
        {
            **checkpoint_config.to_dict(),
            "freeze_protein_encoder": False,
            "freeze_molecule_encoder": False,
        }
    )

    _validate_warm_start_architecture(checkpoint_config, target_config)

    incompatible = RewardModelConfig.from_dict(
        {
            **target_config.to_dict(),
            "fusion_hidden_dim": 256,
            "fusion_num_heads": 4,
        }
    )
    with pytest.raises(ValueError, match="fusion_hidden_dim"):
        _validate_warm_start_architecture(checkpoint_config, incompatible)

    simple_cosine = RewardModelConfig.from_dict(
        {
            **target_config.to_dict(),
            "pair_scoring_mode": "cosine",
        }
    )
    with pytest.raises(ValueError, match="pair_scoring_mode"):
        _validate_warm_start_architecture(checkpoint_config, simple_cosine)


def test_train_cli_accepts_weight_only_warm_start(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_reward_model.py",
            "--config",
            "phase2.yaml",
            "--init-from-checkpoint",
            "checkpoint-1000",
        ],
    )

    args = parse_args()

    assert args.config == "phase2.yaml"
    assert args.init_from_checkpoint == "checkpoint-1000"


def test_unfrozen_phase_two_config_preserves_architecture_and_reduces_memory_batch():
    config_path = Path(__file__).parents[1] / "configs" / "reward_train_unfrozen.yaml"
    config = load_reward_training_config(str(config_path))

    assert config.model.pair_scoring_mode == "scaled_cosine"
    assert config.model.fusion_hidden_dim == 512
    assert config.model.fusion_num_heads == 8
    assert config.model.fusion_residual is True
    assert config.model.freeze_protein_encoder is False
    assert config.model.freeze_molecule_encoder is False
    assert config.model.ranking_loss_weight == pytest.approx(1.0)
    assert config.model.classification_loss_weight == pytest.approx(0.0)
    assert config.data.max_classification_only_per_item == 0
    assert config.training.per_device_train_batch_size == 12
    assert config.training.gradient_accumulation_steps == 4
    assert config.training.max_steps == 10_000
    assert config.training.eval_steps == 500
    assert config.training.learning_rate == pytest.approx(1.0e-5)
    assert config.training.dataloader_num_workers == 4
    assert "ranking_only_unfrozen_10000_steps" in config.training.output_dir


def test_simple_cosine_config_is_ranking_only_without_fusion_settings():
    config_path = (
        Path(__file__).parents[1] / "configs" / "reward_train_simple_cosine.yaml"
    )
    config = load_reward_training_config(str(config_path))

    assert config.model.pair_scoring_mode == "cosine"
    assert config.model.pooling_type == "mean"
    assert config.model.fusion_hidden_dim == 512
    assert config.model.dropout == pytest.approx(0.0)
    assert config.model.freeze_protein_encoder is False
    assert config.model.freeze_molecule_encoder is False
    assert config.model.ranking_loss_weight == pytest.approx(1.0)
    assert config.model.classification_loss_weight == pytest.approx(0.0)
    assert config.data.max_classification_only_per_item == 0
    assert config.training.dataloader_num_workers == 4
    assert config.training.training_mode == "single_gpu"
    assert config.training.metrics_profile == "ranking"
    assert config.training.ranking_score_diagnostics is True
    assert config.training.protein_shuffle_sensitivity is False
    assert "simple_cosine_ranking_only_unfrozen" in config.training.output_dir


def test_simple_cosine_scale10_config_uses_full_data_and_successful_overfit_settings():
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "reward_train_simple_cosine_scale10.yaml"
    )
    config = load_reward_training_config(str(config_path))

    assert config.model.protein_model_name_or_path == (
        "facebook/esm2_t12_35M_UR50D"
    )
    assert config.model.molecule_model_name_or_path == "HUBioDataLab/SELFormer"
    assert config.model.molecule_input_representation == "selfies"
    assert config.model.pair_scoring_mode == "cosine"
    assert config.model.ranking_temperature == pytest.approx(0.1)
    assert config.model.classification_loss_weight == pytest.approx(0.0)
    assert config.model.freeze_protein_encoder is False
    assert config.model.freeze_molecule_encoder is False
    assert config.data.tokenized_dataset_dir.endswith(
        "chembl_37_mmseqs50_activity_balanced"
    )
    assert "overfit_50" not in config.data.tokenized_dataset_dir
    assert config.data.ranking_max_ligands == 16
    assert config.data.ranking_opportunity_divisor == 32
    assert config.data.evaluation_ranking_partitions == 3
    assert config.data.max_classification_only_per_item == 0
    assert config.training.encoder_learning_rate == pytest.approx(1.0e-5)
    assert config.training.projection_learning_rate == pytest.approx(1.0e-3)
    assert config.training.max_grad_norm == pytest.approx(10.0)
    assert config.training.dataloader_num_workers == 8
    assert config.training.fp16 is False
    assert config.training.bf16 is True
    assert config.training.training_mode == "single_gpu"
    assert config.training.metrics_profile == "ranking"
    assert "simple_cosine_scale10" in config.training.output_dir


def test_scale10_contrastive_config_mirrors_ligunity_without_classification():
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "reward_train_simple_cosine_scale10_contrastive.yaml"
    )
    config = load_reward_training_config(str(config_path))

    assert config.model.protein_model_name_or_path == (
        "facebook/esm2_t12_35M_UR50D"
    )
    assert config.model.molecule_model_name_or_path == "HUBioDataLab/SELFormer"
    assert config.model.pair_scoring_mode == "cosine"
    assert config.model.ranking_temperature == pytest.approx(0.1)
    assert config.model.ranking_loss_weight == pytest.approx(0.5)
    assert config.model.contrastive_loss_weight == pytest.approx(0.5)
    assert config.model.contrastive_active_threshold == pytest.approx(5.0)
    assert config.model.classification_loss_weight == pytest.approx(0.0)
    assert config.model.deduplicate_protein_inputs is True
    assert config.model.freeze_protein_encoder is False
    assert config.model.freeze_molecule_encoder is False
    assert config.data.max_classification_only_per_item == 0
    assert config.training.per_device_train_batch_size == 12
    assert config.training.gradient_accumulation_steps == 4
    assert config.training.dataloader_num_workers == 8
    assert config.training.metrics_profile == "ranking"
    assert "scale10_contrastive_ranking" in config.training.output_dir


def test_scale10_contrastive_lr1e4_4gpu_config_is_controlled_experiment():
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "reward_train_simple_cosine_scale10_contrastive_lr1e4_4gpu.yaml"
    )
    config = load_reward_training_config(str(config_path))

    assert config.model.molecule_model_name_or_path == "HUBioDataLab/SELFormer"
    assert config.model.protein_hidden_dropout_prob == pytest.approx(0.15)
    assert config.model.protein_attention_probs_dropout_prob == pytest.approx(0.15)
    assert config.model.molecule_hidden_dropout_prob == pytest.approx(0.15)
    assert config.model.molecule_attention_probs_dropout_prob == pytest.approx(0.15)
    assert config.model.dropout == pytest.approx(0.15)
    assert config.model.projection_type == "nonlinear"
    assert config.model.fusion_hidden_dim == 128
    assert config.model.ranking_temperature == pytest.approx(0.1)
    assert config.model.ranking_loss_weight == pytest.approx(0.5)
    assert config.model.contrastive_loss_weight == pytest.approx(0.5)
    assert config.model.classification_loss_weight == pytest.approx(0.0)
    assert config.model.freeze_protein_encoder is False
    assert config.model.freeze_molecule_encoder is False
    assert config.training.learning_rate == pytest.approx(1.0e-4)
    assert config.training.encoder_learning_rate == pytest.approx(1.0e-4)
    assert config.training.projection_learning_rate == pytest.approx(1.0e-3)
    assert config.training.per_device_train_batch_size == 16
    assert config.training.gradient_accumulation_steps == 1
    assert config.training.num_train_epochs == pytest.approx(100.0)
    assert config.training.max_steps is None
    assert config.training.eval_steps is None
    assert config.training.training_mode == "multi_gpu"
    assert config.training.bf16 is True
    assert config.training.dataloader_num_workers == 4
    assert (
        "nonlinear128_lr1e4_batch16_dropout015_4gpu_100_epochs"
        in config.training.output_dir
    )


def test_overfit_grid_covers_all_lr_clip_and_temperature_combinations():
    config_dir = Path(__file__).parents[1] / "configs" / "overfit_grid"
    config_paths = sorted(config_dir.glob("*.yaml"))
    combinations = set()

    assert len(config_paths) == 8
    for config_path in config_paths:
        config = load_reward_training_config(str(config_path))
        assert config.model.protein_model_name_or_path == (
            "facebook/esm2_t12_35M_UR50D"
        )
        assert config.model.molecule_model_name_or_path == "HUBioDataLab/SELFormer"
        assert config.model.pair_scoring_mode == "cosine"
        assert config.model.classification_loss_weight == pytest.approx(0.0)
        assert config.model.freeze_protein_encoder is False
        assert config.model.freeze_molecule_encoder is False
        assert config.data.ranking_max_ligands == 50
        assert config.data.ranking_opportunity_divisor == 50
        assert config.data.evaluation_ranking_partitions == 1
        assert config.data.max_classification_only_per_item == 0
        assert config.training.bf16 is True
        assert config.training.dataloader_num_workers == 8
        combinations.add(
            (
                config.training.projection_learning_rate,
                config.training.max_grad_norm,
                config.model.ranking_temperature,
            )
        )

    assert combinations == {
        (projection_lr, clip_norm, temperature)
        for projection_lr in (1.0e-5, 1.0e-3)
        for clip_norm in (1.0, 10.0)
        for temperature in (1.0, 0.1)
    }


def test_molformer_simple_cosine_config_uses_smiles_and_separate_cache():
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "reward_train_simple_cosine_molformer.yaml"
    )
    config = load_reward_training_config(str(config_path))

    assert config.model.molecule_model_name_or_path == (
        "ibm/MoLFormer-XL-both-10pct"
    )
    assert config.model.molecule_input_representation == "smiles"
    assert config.model.molecule_trust_remote_code is True
    assert config.model.molecule_deterministic_eval is True
    assert config.model.molecule_max_length == 202
    assert config.model.pair_scoring_mode == "cosine"
    assert config.model.classification_loss_weight == pytest.approx(0.0)
    assert config.data.tokenization_num_proc == 8
    assert config.training.dataloader_num_workers == 8
    assert config.training.fp16 is False
    assert config.training.bf16 is True
    assert "molformer_smiles" in config.data.tokenized_dataset_dir
    assert "molformer_smiles" in config.training.output_dir
    assert config.training.metrics_profile == "ranking"


def test_selformer_esm2_150m_config_reuses_esm_tokenized_cache():
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "reward_train_simple_cosine_selformer_esm2_150m.yaml"
    )
    config = load_reward_training_config(str(config_path))

    assert config.model.protein_model_name_or_path == (
        "facebook/esm2_t30_150M_UR50D"
    )
    assert config.model.molecule_model_name_or_path == "HUBioDataLab/SELFormer"
    assert config.model.molecule_input_representation == "selfies"
    assert config.model.freeze_protein_encoder is False
    assert config.model.freeze_molecule_encoder is False
    assert config.model.pair_scoring_mode == "cosine"
    assert config.model.classification_loss_weight == pytest.approx(0.0)
    assert config.data.tokenization_num_proc == 8
    assert config.data.tokenized_dataset_dir.endswith(
        "chembl_37_mmseqs50_activity_balanced"
    )
    assert config.training.dataloader_num_workers == 8
    assert config.training.fp16 is False
    assert config.training.bf16 is True
    assert "selformer_esm2_150m" in config.training.output_dir
    assert config.training.metrics_profile == "ranking"


def test_prepare_pair_datasets_from_config_summarizes_without_materializing_pairs(tmp_path):
    split_paths = get_tokenized_split_dataset_paths(str(tmp_path / "tokenized"))
    pair_paths = get_saved_pair_dataset_paths(str(tmp_path / "tokenized"))

    listwise_ready_examples = _metric_ready_examples()
    train_examples = listwise_ready_examples.select([0, 1, 2])
    val_examples = listwise_ready_examples.select([3, 4, 5])
    test_examples = Dataset.from_list(
        [
            {
                "example_id": 0,
                "group_id": "T3__A3",
                "target_chembl_id": "T3",
                "assay_id": "A3",
                "compound_id": "M4",
                "pchembl_value": 5.0,
                "binary_label": 0,
                "protein_input_ids": [7, 7, 0],
                "protein_attention_mask": [1, 1, 0],
                "molecule_input_ids": [8, 8, 0, 0],
                "molecule_attention_mask": [1, 1, 0, 0],
            }
        ]
    )

    train_examples.save_to_disk(split_paths["train"])
    val_examples.save_to_disk(split_paths["val"])
    test_examples.save_to_disk(split_paths["test"])

    config_path = tmp_path / "reward_train.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  protein_model_name_or_path: protein/dummy",
                "  molecule_model_name_or_path: molecule/dummy",
                "data:",
                f"  train_parquet_path: {tmp_path / 'unused_train.parquet'}",
                f"  val_parquet_path: {tmp_path / 'unused_val.parquet'}",
                f"  test_parquet_path: {tmp_path / 'unused_test.parquet'}",
                f"  tokenized_dataset_dir: {tmp_path / 'tokenized'}",
                "  tokenization_batch_size: 2",
                "training:",
                f"  output_dir: {tmp_path / 'trainer_output'}",
            ]
        ),
        encoding="utf-8",
    )

    summary = prepare_pair_datasets_from_config(str(config_path))

    assert not os.path.exists(pair_paths["train"])
    assert not os.path.exists(pair_paths["val"])
    assert not os.path.exists(pair_paths["test"])
    assert summary["materialized_pair_tables"] is False
    assert summary["train_ranking_lists"] == 1
    assert summary["val_ranking_lists"] == 1
    assert summary["test_ranking_lists"] == 0


def test_train_reward_model_from_config_trains_on_train_and_final_evaluates_val_and_test(tmp_path, monkeypatch):
    split_paths = get_tokenized_split_dataset_paths(str(tmp_path / "tokenized"))

    train_examples = Dataset.from_list(
        [
            {
                "example_id": 0,
                "group_id": "T1__A1",
                "target_chembl_id": "T1",
                "assay_id": "A1",
                "compound_id": "M0",
                "pchembl_value": 4.0,
                "binary_label": 0,
                "protein_input_ids": [1, 1, 0],
                "protein_attention_mask": [1, 1, 0],
                "molecule_input_ids": [2, 2, 0, 0],
                "molecule_attention_mask": [1, 1, 0, 0],
            },
            {
                "example_id": 1,
                "group_id": "T1__A1",
                "target_chembl_id": "T1",
                "assay_id": "A1",
                "compound_id": "M1",
                "pchembl_value": 7.0,
                "binary_label": 1,
                "protein_input_ids": [1, 1, 0],
                "protein_attention_mask": [1, 1, 0],
                "molecule_input_ids": [3, 3, 0, 0],
                "molecule_attention_mask": [1, 1, 0, 0],
            },
            {
                "example_id": 2,
                "group_id": "T1__A1",
                "target_chembl_id": "T1",
                "assay_id": "A1",
                "compound_id": "M2",
                "pchembl_value": 5.5,
                "binary_label": 0,
                "protein_input_ids": [1, 1, 0],
                "protein_attention_mask": [1, 1, 0],
                "molecule_input_ids": [4, 4, 0, 0],
                "molecule_attention_mask": [1, 1, 0, 0],
            },
        ]
    )
    val_examples = Dataset.from_list(
        [
            {
                "example_id": 0,
                "group_id": "T2__A2",
                "target_chembl_id": "T2",
                "assay_id": "A2",
                "compound_id": "M2",
                "pchembl_value": 4.0,
                "binary_label": 0,
                "protein_input_ids": [4, 4, 0],
                "protein_attention_mask": [1, 1, 0],
                "molecule_input_ids": [5, 5, 0, 0],
                "molecule_attention_mask": [1, 1, 0, 0],
            },
            {
                "example_id": 1,
                "group_id": "T2__A2",
                "target_chembl_id": "T2",
                "assay_id": "A2",
                "compound_id": "M3",
                "pchembl_value": 8.0,
                "binary_label": 1,
                "protein_input_ids": [4, 4, 0],
                "protein_attention_mask": [1, 1, 0],
                "molecule_input_ids": [6, 6, 0, 0],
                "molecule_attention_mask": [1, 1, 0, 0],
            },
            {
                "example_id": 2,
                "group_id": "T2__A2",
                "target_chembl_id": "T2",
                "assay_id": "A2",
                "compound_id": "M4",
                "pchembl_value": 6.0,
                "binary_label": 1,
                "protein_input_ids": [4, 4, 0],
                "protein_attention_mask": [1, 1, 0],
                "molecule_input_ids": [7, 7, 0, 0],
                "molecule_attention_mask": [1, 1, 0, 0],
            },
        ]
    )
    val_examples = val_examples.add_column(
        "activity_type", ["Binding"] * len(val_examples)
    )
    test_examples = Dataset.from_list(
        [
            {
                "example_id": 0,
                "group_id": "T3__A3",
                "target_chembl_id": "T3",
                "assay_id": "A3",
                "compound_id": "M4",
                "pchembl_value": 5.0,
                "binary_label": 0,
                "protein_input_ids": [7, 7, 0],
                "protein_attention_mask": [1, 1, 0],
                "molecule_input_ids": [8, 8, 0, 0],
                "molecule_attention_mask": [1, 1, 0, 0],
            }
        ]
    )
    test_examples = test_examples.add_column(
        "activity_type", ["Potency"] * len(test_examples)
    )

    train_examples.save_to_disk(split_paths["train"])
    val_examples.save_to_disk(split_paths["val"])
    test_examples.save_to_disk(split_paths["test"])

    config_path = tmp_path / "reward_train.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  protein_model_name_or_path: protein/dummy",
                "  molecule_model_name_or_path: molecule/dummy",
                "  fusion_hidden_dim: 8",
                "  fusion_num_heads: 2",
                "  dropout: 0.0",
                "data:",
                f"  train_parquet_path: {tmp_path / 'unused_train.parquet'}",
                f"  val_parquet_path: {tmp_path / 'unused_val.parquet'}",
                f"  test_parquet_path: {tmp_path / 'unused_test.parquet'}",
                f"  tokenized_dataset_dir: {tmp_path / 'tokenized'}",
                "  tokenization_batch_size: 2",
                "training:",
                f"  output_dir: {tmp_path / 'trainer_output'}",
                "  num_train_epochs: 1",
                "  per_device_train_batch_size: 2",
                "  per_device_eval_batch_size: 2",
                "  logging_steps: 1",
                "  fp16: false",
            ]
        ),
        encoding="utf-8",
    )

    captured = {}
    warm_start = tmp_path / "phase1" / "checkpoint-1000"
    warm_start.mkdir(parents=True)

    class _FakeTrainer:
        def __init__(
            self,
            model,
            args,
            train_dataset,
            eval_dataset,
            data_collator,
            val2_eval_dataset=None,
        ):
            captured["train_dataset_len"] = len(train_dataset)
            captured["eval_dataset_len"] = len(eval_dataset)
            captured["train_dataset"] = train_dataset
            captured["eval_dataset"] = eval_dataset
            captured["val2_eval_dataset"] = val2_eval_dataset
            captured["data_collator"] = data_collator

        def train(self):
            captured["train_called"] = True
            return None

        def save_model(self, output_dir):
            captured["save_model_output_dir"] = output_dir

        def evaluate(self, eval_dataset=None, metric_key_prefix="eval"):
            captured.setdefault("evaluate_calls", []).append(
                (eval_dataset, metric_key_prefix)
            )
            return {f"{metric_key_prefix}_loss": 0.5}

    monkeypatch.setattr("reward_model.training.entry.RewardModelTrainer", _FakeTrainer)

    def _fake_initialize_training_model(config, init_from_checkpoint):
        captured["model_config"] = config
        captured["init_from_checkpoint"] = init_from_checkpoint
        return object()

    monkeypatch.setattr(
        "reward_model.training.entry._initialize_training_model",
        _fake_initialize_training_model,
    )
    monkeypatch.setattr("reward_model.training.entry.create_training_arguments", lambda config: object())

    summary = train_reward_model_from_config(
        str(config_path),
        init_from_checkpoint=str(warm_start),
    )

    assert captured["train_called"] is True
    assert len(captured["evaluate_calls"]) == 2
    assert captured["evaluate_calls"][0] == (None, "eval")
    test_eval_dataset, test_prefix = captured["evaluate_calls"][1]
    assert len(test_eval_dataset) == 1
    assert test_prefix == "test"
    assert captured["train_dataset_len"] == 1
    assert captured["eval_dataset_len"] == 3
    assert captured["eval_dataset"].ranking_max_ligands == 16
    assert captured["eval_dataset"].ranking_num_partitions == 3
    assert captured["eval_dataset"].ranking_partition_seed == 42
    assert captured["val2_eval_dataset"] is None
    assert captured["init_from_checkpoint"] == str(warm_start.resolve())
    assert isinstance(captured["data_collator"], RewardAssayListCollator)
    assert summary["train_examples"] == 3
    assert summary["val_examples"] == 3
    assert summary["test_examples"] == 1
    assert summary["train_ranking_lists"] == 1
    assert summary["train_ranked_examples"] == 3
    assert summary["train_classification_examples"] == 3
    assert summary["init_from_checkpoint"] == str(warm_start.resolve())
    assert summary["optimizer_state_restored"] is False
    assert summary["test_metrics"]["test_loss"] == pytest.approx(0.5)


def test_train_reward_model_from_config_loads_optional_val2_dataset(tmp_path, monkeypatch):
    split_paths = get_tokenized_split_dataset_paths(str(tmp_path / "tokenized"))

    train_examples = _pair_ready_examples().select([0, 1])
    val_examples = _pair_ready_examples().select([2, 3])
    test_examples = _pair_ready_examples().select([0])
    val_examples = val_examples.add_column(
        "activity_type", ["Binding"] * len(val_examples)
    )
    test_examples = test_examples.add_column(
        "activity_type", ["Potency"] * len(test_examples)
    )
    val2_examples = Dataset.from_list(
        [
            {
                "example_id": 0,
                "group_id": "T4__A4",
                "target_chembl_id": "T4",
                "assay_id": "A4",
                "compound_id": "M0",
                "pchembl_value": 4.0,
                "binary_label": 0,
                "protein_input_ids": [1, 1, 0],
                "protein_attention_mask": [1, 1, 0],
                "molecule_input_ids": [2, 2, 0, 0],
                "molecule_attention_mask": [1, 1, 0, 0],
            },
            {
                "example_id": 1,
                "group_id": "T4__A4",
                "target_chembl_id": "T4",
                "assay_id": "A4",
                "compound_id": "M1",
                "pchembl_value": 7.0,
                "binary_label": 1,
                "protein_input_ids": [1, 1, 0],
                "protein_attention_mask": [1, 1, 0],
                "molecule_input_ids": [3, 3, 0, 0],
                "molecule_attention_mask": [1, 1, 0, 0],
            },
        ]
    )

    train_examples.save_to_disk(split_paths["train"])
    val_examples.save_to_disk(split_paths["val"])
    test_examples.save_to_disk(split_paths["test"])
    val2_examples.save_to_disk(tmp_path / "tokenized" / "val2_examples")

    config_path = tmp_path / "reward_train.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  protein_model_name_or_path: protein/dummy",
                "  molecule_model_name_or_path: molecule/dummy",
                "  fusion_hidden_dim: 8",
                "  fusion_num_heads: 2",
                "  dropout: 0.0",
                "data:",
                f"  train_parquet_path: {tmp_path / 'unused_train.parquet'}",
                f"  val_parquet_path: {tmp_path / 'unused_val.parquet'}",
                f"  test_parquet_path: {tmp_path / 'unused_test.parquet'}",
                f"  tokenized_dataset_dir: {tmp_path / 'tokenized'}",
                f"  val2_tokenized_dataset_dir: {tmp_path / 'tokenized'}",
                "  tokenization_batch_size: 2",
                "training:",
                f"  output_dir: {tmp_path / 'trainer_output'}",
                "  num_train_epochs: 1",
                "  per_device_train_batch_size: 2",
                "  per_device_eval_batch_size: 2",
                "  logging_steps: 1",
                "  fp16: false",
            ]
        ),
        encoding="utf-8",
    )

    captured = {}

    class _FakeTrainer:
        def __init__(
            self,
            model,
            args,
            train_dataset,
            eval_dataset,
            data_collator,
            val2_eval_dataset=None,
        ):
            captured["val2_eval_dataset_len"] = len(val2_eval_dataset)

        def train(self):
            return None

        def save_model(self, output_dir):
            captured["save_model_output_dir"] = output_dir

        def evaluate(self, eval_dataset=None, metric_key_prefix="eval"):
            if metric_key_prefix == "test":
                return {"test_loss": 0.6}
            return {"eval_loss": 0.5, "eval_val2_loss": 0.4}

    monkeypatch.setattr("reward_model.training.entry.RewardModelTrainer", _FakeTrainer)
    monkeypatch.setattr("reward_model.training.entry.RewardModel", lambda config: object())
    monkeypatch.setattr("reward_model.training.entry.create_training_arguments", lambda config: object())

    summary = train_reward_model_from_config(str(config_path))

    assert captured["val2_eval_dataset_len"] == 2
    assert summary["val2_examples"] == 2
    assert summary["eval_metrics"]["eval_val2_loss"] == pytest.approx(0.4)
    assert summary["test_metrics"]["test_loss"] == pytest.approx(0.6)
