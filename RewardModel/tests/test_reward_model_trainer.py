import json
import os

import pytest
from datasets import Dataset

from conftest import DummyEncoder, DummyTokenizer
from reward_model.model import LoadedEncoder, RewardModel, RewardModelConfig, load_reward_model
from reward_model.training import (
    RewardPairCollator,
    RewardPairDataset,
    RewardModelTrainer,
    RewardTrainerConfig,
    build_pair_records,
    compute_classification_metrics,
    compute_groupwise_spearman,
    compute_pairwise_accuracy,
    create_training_arguments,
    get_saved_pair_dataset_paths,
    get_tokenized_split_dataset_paths,
    load_saved_pair_dataset,
    prepare_pair_datasets_from_config,
    save_pair_dataset_from_example_dataset,
)
from reward_model.training.entry import train_reward_model_from_config


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


def test_reward_model_trainer_runs_and_saves_checkpoint(tmp_path, monkeypatch):
    pytest.importorskip("accelerate")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("WANDB_MODE", "disabled")

    protein_bundle, molecule_bundle = _dummy_bundles()
    examples = _metric_ready_examples()
    pair_records, _ = build_pair_records(examples)
    pair_dataset = Dataset.from_list(pair_records)
    train_dataset = RewardPairDataset(examples, pair_dataset)
    eval_dataset = RewardPairDataset(examples, pair_dataset)

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
            )
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        val2_eval_dataset=eval_dataset,
        data_collator=RewardPairCollator(),
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    save_dir = tmp_path / "saved_model"
    trainer.save_model(str(save_dir))

    assert train_result.training_loss >= 0.0
    assert "eval_pair_loss" in eval_metrics
    assert "eval_classification_loss" in eval_metrics
    assert "eval_num_pairs" in eval_metrics
    assert "eval_mcc" in eval_metrics
    assert "eval_f1" in eval_metrics
    assert "eval_roc_auc" in eval_metrics
    assert "eval_precision" in eval_metrics
    assert "eval_recall" in eval_metrics
    assert "eval_accuracy" in eval_metrics
    assert "eval_pairwise_accuracy" in eval_metrics
    assert "eval_spearman" in eval_metrics
    assert "eval_spearman_num_groups" in eval_metrics
    assert "eval_val2_loss" in eval_metrics
    assert "eval_val2_pair_loss" in eval_metrics
    assert "eval_val2_classification_loss" in eval_metrics
    assert "eval_val2_pairwise_accuracy" in eval_metrics
    assert "eval_val2_mcc" in eval_metrics
    assert "eval_val2_f1" in eval_metrics
    assert "eval_val2_roc_auc" in eval_metrics
    assert "eval_val2_precision" in eval_metrics
    assert "eval_val2_recall" in eval_metrics
    assert "eval_val2_accuracy" in eval_metrics
    assert "eval_val2_spearman" in eval_metrics
    assert "eval_val2_spearman_num_groups" in eval_metrics
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
    assert any("pair_loss" in entry for entry in trainer.state.log_history)
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


def test_prepare_pair_datasets_from_config_saves_train_val_and_test_pair_datasets(tmp_path):
    split_paths = get_tokenized_split_dataset_paths(str(tmp_path / "tokenized"))
    pair_paths = get_saved_pair_dataset_paths(str(tmp_path / "tokenized"))

    pair_ready_examples = _pair_ready_examples()
    train_examples = pair_ready_examples.select([0, 1])
    val_examples = pair_ready_examples.select([2, 3])
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

    assert os.path.exists(pair_paths["train"])
    assert os.path.exists(pair_paths["val"])
    assert os.path.exists(pair_paths["test"])
    assert summary["train_pairs"] == 1
    assert summary["val_pairs"] == 1
    assert summary["test_pairs"] == 0

    train_pairs = load_saved_pair_dataset(pair_paths["train"])
    val_pairs = load_saved_pair_dataset(pair_paths["val"])
    test_pairs = load_saved_pair_dataset(pair_paths["test"])
    assert len(train_pairs) == 1
    assert len(val_pairs) == 1
    assert len(test_pairs) == 0


def test_train_reward_model_from_config_uses_train_and_val_splits_only(tmp_path, monkeypatch):
    split_paths = get_tokenized_split_dataset_paths(str(tmp_path / "tokenized"))
    pair_paths = get_saved_pair_dataset_paths(str(tmp_path / "tokenized"))

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
    save_pair_dataset_from_example_dataset(train_examples, pair_paths["train"], split_name="train")
    save_pair_dataset_from_example_dataset(val_examples, pair_paths["val"], split_name="val")

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
    assert captured["eval_dataset_len"] == 1
    assert captured["val2_eval_dataset"] is None
    assert isinstance(captured["data_collator"], RewardPairCollator)
    assert summary["train_examples"] == 2
    assert summary["val_examples"] == 2
    assert summary["test_examples"] == 1
    assert summary["train_pairs"] == 1
    assert summary["val_pairs"] == 1


def test_train_reward_model_from_config_loads_optional_val2_dataset(tmp_path, monkeypatch):
    split_paths = get_tokenized_split_dataset_paths(str(tmp_path / "tokenized"))
    pair_paths = get_saved_pair_dataset_paths(str(tmp_path / "tokenized"))

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
    save_pair_dataset_from_example_dataset(train_examples, pair_paths["train"], split_name="train")
    save_pair_dataset_from_example_dataset(val_examples, pair_paths["val"], split_name="val")
    save_pair_dataset_from_example_dataset(
        val2_examples,
        str(tmp_path / "tokenized" / "val2_pairs"),
        split_name="val2",
    )

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

    assert captured["val2_eval_dataset_len"] == 1
    assert summary["val2_examples"] == 2
    assert summary["val2_pairs"] == 1
    assert summary["eval_metrics"]["eval_val2_loss"] == pytest.approx(0.4)


def test_train_reward_model_from_config_requires_saved_pair_datasets(tmp_path):
    split_paths = get_tokenized_split_dataset_paths(str(tmp_path / "tokenized"))

    pair_ready_examples = _pair_ready_examples()
    train_examples = pair_ready_examples.select([0, 1])
    val_examples = pair_ready_examples.select([2, 3])
    test_examples = pair_ready_examples.select([0])
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

    with pytest.raises(FileNotFoundError, match="prepare_reward_pair_datasets.py"):
        train_reward_model_from_config(str(config_path))
