import json
import os

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
    compute_classification_metrics,
    compute_groupwise_spearman,
    compute_pairwise_accuracy,
    compute_ranking_score_diagnostics,
    create_training_arguments,
    get_saved_pair_dataset_paths,
    get_tokenized_split_dataset_paths,
    load_saved_pair_dataset,
    prepare_pair_datasets_from_config,
    save_pair_dataset_from_example_dataset,
)
from reward_model.training.entry import train_reward_model_from_config
from reward_model.training.trainer import LengthBucketSampler


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
    assert spearman_metrics["eval_spearman_num_groups"] == pytest.approx(2.0)


def test_ranking_score_diagnostics_measure_scale_margin_and_saturation():
    metrics = compute_ranking_score_diagnostics(
        ranking_scores=torch.tensor([3.0, 1.0, -2.0, 100.0]),
        pchembl_values=torch.tensor([8.0, 7.0, 6.0, 1.0]),
        ranking_group_ids=torch.tensor([0, 0, 0, -1]),
        temperature=1.0,
        affinity_margin=0.5,
        saturation_scales=(1.0, 5.0),
    )

    assert metrics["ranking_score_num_examples"] == pytest.approx(3.0)
    assert metrics["ranking_score_mean"] == pytest.approx(2.0 / 3.0)
    assert metrics["ranking_score_std"] == pytest.approx(
        torch.tensor([3.0, 1.0, -2.0]).std(unbiased=False).item()
    )
    assert metrics["ranking_margin_pair_accuracy"] == pytest.approx(1.0)
    assert metrics["ranking_margin_pair_gap_p50"] == pytest.approx(3.0)
    assert metrics["ranking_margin_pair_count"] == pytest.approx(3.0)
    assert 0.0 < metrics["ranking_list_normalized_entropy"] < 1.0
    assert metrics["ranking_score_tanh_1_saturation_fraction"] == pytest.approx(
        2.0 / 3.0
    )
    assert metrics["ranking_score_tanh_5_saturation_fraction"] == pytest.approx(0.0)


def test_ranking_score_diagnostics_exclude_pairs_inside_affinity_margin():
    metrics = compute_ranking_score_diagnostics(
        ranking_scores=torch.tensor([3.0, -100.0, -2.0]),
        pchembl_values=torch.tensor([8.0, 7.8, 6.0]),
        ranking_group_ids=torch.tensor([0, 0, 0]),
        affinity_margin=0.5,
    )

    assert metrics["ranking_margin_pair_count"] == pytest.approx(2.0)
    assert metrics["ranking_margin_pair_accuracy"] == pytest.approx(0.5)


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
                num_train_epochs=1,
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
    assert "eval_ranking_score_std" in eval_metrics
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
    assert "eval_val2_ranking_score_std" in eval_metrics
    assay_log_path = tmp_path / "trainer_output" / "eval_assay_spearman.jsonl"
    val2_assay_log_path = tmp_path / "trainer_output" / "eval_val2_assay_spearman.jsonl"
    assay_log_records = [
        json.loads(line)
        for line in assay_log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(assay_log_records) >= 1
    latest_assay_log = assay_log_records[-1]
    assert latest_assay_log["global_step"] == trainer.state.global_step
    assert latest_assay_log["weighted_spearman"] == pytest.approx(eval_metrics["eval_spearman"])
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
        "ranking_pairwise_accuracy",
        "ranking_loss_per_ranked_example",
        "ranking_score_std",
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


def test_create_training_arguments_uses_step_based_schedule_when_eval_steps_is_set(tmp_path):
    args = create_training_arguments(
        RewardTrainerConfig(
            output_dir=str(tmp_path / "trainer_output"),
            eval_steps=25,
            fp16=False,
        )
    )

    def _strategy_value(value):
        return value.value if hasattr(value, "value") else value

    assert _strategy_value(args.save_strategy) == "steps"
    assert args.eval_steps == 25
    assert args.save_steps == 25
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


def test_reward_trainer_log_supports_transformers_without_start_time(monkeypatch):
    captured = {}

    def _legacy_log(self, logs):
        captured.update(logs)
        return "logged"

    monkeypatch.setattr(Trainer, "log", _legacy_log)
    trainer = object.__new__(RewardModelTrainer)

    assert trainer.log({"eval_loss": 0.25}, start_time=123.0) == "logged"
    assert captured == {"eval_loss": 0.25}


def test_training_metrics_are_count_weighted_and_ranking_ties_are_excluded():
    trainer = object.__new__(RewardModelTrainer)
    trainer._reset_train_component_accumulator()
    trainer._record_train_components(
        ranking_loss=torch.tensor(3.0),
        classification_loss=torch.tensor(0.6),
        total_loss=torch.tensor(3.6),
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
        total_loss=torch.tensor(2.4),
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
    assert logs["ranking_pairwise_accuracy"] == pytest.approx(3.0 / 5.0)
    assert logs["ranking_loss_per_ranked_example"] == pytest.approx(5.0 / 6.0)
    assert logs["ranking_loss"] == pytest.approx(2.5)


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


def test_train_reward_model_from_config_uses_train_and_val_splits_only(tmp_path, monkeypatch):
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

        def evaluate(self):
            captured["evaluate_called"] = True
            return {"eval_loss": 0.5}

    monkeypatch.setattr("reward_model.training.entry.RewardModelTrainer", _FakeTrainer)
    monkeypatch.setattr("reward_model.training.entry.RewardModel", lambda config: object())
    monkeypatch.setattr("reward_model.training.entry.create_training_arguments", lambda config: object())

    summary = train_reward_model_from_config(str(config_path))

    assert captured["train_called"] is True
    assert captured["evaluate_called"] is True
    assert captured["train_dataset_len"] == 1
    assert captured["eval_dataset_len"] == 3
    assert captured["eval_dataset"].ranking_max_ligands == 16
    assert captured["eval_dataset"].ranking_num_partitions == 3
    assert captured["eval_dataset"].ranking_partition_seed == 42
    assert captured["val2_eval_dataset"] is None
    assert isinstance(captured["data_collator"], RewardAssayListCollator)
    assert summary["train_examples"] == 3
    assert summary["val_examples"] == 3
    assert summary["test_examples"] == 1
    assert summary["train_ranking_lists"] == 1
    assert summary["train_ranked_examples"] == 3
    assert summary["train_classification_examples"] == 3


def test_train_reward_model_from_config_loads_optional_val2_dataset(tmp_path, monkeypatch):
    split_paths = get_tokenized_split_dataset_paths(str(tmp_path / "tokenized"))

    train_examples = _pair_ready_examples().select([0, 1])
    val_examples = _pair_ready_examples().select([2, 3])
    test_examples = _pair_ready_examples().select([0])
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

        def evaluate(self):
            return {"eval_loss": 0.5, "eval_val2_loss": 0.4}

    monkeypatch.setattr("reward_model.training.entry.RewardModelTrainer", _FakeTrainer)
    monkeypatch.setattr("reward_model.training.entry.RewardModel", lambda config: object())
    monkeypatch.setattr("reward_model.training.entry.create_training_arguments", lambda config: object())

    summary = train_reward_model_from_config(str(config_path))

    assert captured["val2_eval_dataset_len"] == 2
    assert summary["val2_examples"] == 2
    assert summary["eval_metrics"]["eval_val2_loss"] == pytest.approx(0.4)
