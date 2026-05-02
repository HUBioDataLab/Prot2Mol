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
    create_training_arguments,
)


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
                "group_id": "G1",
                "target_chembl_id": "T1",
                "assay_chembl_id": "A1",
                "molecule_chembl_id": "M0",
                "protein_input_ids": [1, 1, 0],
                "protein_attention_mask": [1, 1, 0],
                "molecule_input_ids": [7, 8, 0, 0],
                "molecule_attention_mask": [1, 1, 0, 0],
                "activity_label": 0,
                "pchembl_value": 4.0,
            },
            {
                "example_id": 1,
                "group_id": "G1",
                "target_chembl_id": "T1",
                "assay_chembl_id": "A1",
                "molecule_chembl_id": "M1",
                "protein_input_ids": [1, 1, 0],
                "protein_attention_mask": [1, 1, 0],
                "molecule_input_ids": [8, 8, 0, 0],
                "molecule_attention_mask": [1, 1, 0, 0],
                "activity_label": 1,
                "pchembl_value": 7.0,
            },
            {
                "example_id": 2,
                "group_id": "G2",
                "target_chembl_id": "T2",
                "assay_chembl_id": "A2",
                "molecule_chembl_id": "M2",
                "protein_input_ids": [2, 2, 0],
                "protein_attention_mask": [1, 1, 0],
                "molecule_input_ids": [3, 4, 0, 0],
                "molecule_attention_mask": [1, 1, 0, 0],
                "activity_label": 0,
                "pchembl_value": 5.0,
            },
            {
                "example_id": 3,
                "group_id": "G2",
                "target_chembl_id": "T2",
                "assay_chembl_id": "A2",
                "molecule_chembl_id": "M3",
                "protein_input_ids": [2, 2, 0],
                "protein_attention_mask": [1, 1, 0],
                "molecule_input_ids": [5, 6, 0, 0],
                "molecule_attention_mask": [1, 1, 0, 0],
                "activity_label": 1,
                "pchembl_value": 8.0,
            },
        ]
    )


def test_reward_model_trainer_runs_and_saves_checkpoint(tmp_path):
    pytest.importorskip("accelerate")

    protein_bundle, molecule_bundle = _dummy_bundles()
    examples = _pair_ready_examples()
    pair_records, _ = build_pair_records(examples)
    train_dataset = RewardPairDataset(examples, pair_records)
    eval_dataset = RewardPairDataset(examples, pair_records)

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
    assert any("pair_loss" in entry for entry in trainer.state.log_history)
    assert os.path.exists(save_dir / "pytorch_model.bin")
    assert os.path.exists(save_dir / "config.json")

    reloaded = load_reward_model(
        str(save_dir),
        protein_bundle=protein_bundle,
        molecule_bundle=molecule_bundle,
    )
    assert reloaded.config.fusion_hidden_dim == 10
