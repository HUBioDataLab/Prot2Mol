import json
import math

import numpy as np
import pytest
import torch
from datasets import Dataset

from reward_model.model.losses import ligunity_listwise_loss
from reward_model.training import (
    RewardAssayListCollator,
    RewardAssayListDataset,
    RewardEvaluationDataset,
    RewardModelTrainer,
    RewardTrainerConfig,
    create_training_arguments,
)
from reward_model.training.evaluation import build_complete_coverage_ranking_partitions
from test_paired_generalization_adaptation import batch, model


def unequal_assay_examples():
    rows = []
    for assay, count in enumerate((9, 3, 1)):
        for index in range(count):
            value = (8.0, 6.5, 4.0)[index % 3]
            rows.append(dict(
                example_id=len(rows), group_id=f"p{assay}__a{assay}",
                target_chembl_id=f"p{assay}", assay_id=f"a{assay}",
                compound_id=f"m{len(rows)}", pchembl_value=value,
                binary_label=int(value > 6),
                protein_input_ids=[3 + assay, 6 + assay, 1],
                protein_attention_mask=[1, 1, 0],
                molecule_input_ids=[3 + index, 4 + index, 1],
                molecule_attention_mask=[1, 1, 0],
            ))
    return Dataset.from_list(rows)


@pytest.mark.parametrize("controls", [
    {},
    dict(ranking_normalize_per_group=True),
    dict(ranking_normalize_per_group=True, ranking_balance_assays=True),
    dict(ranking_objective="pairwise_logistic", ranking_normalize_per_group=True),
])
def test_prediction_only_forward_preserves_scores_and_embeddings(controls):
    fitted = model(**controls).eval()
    data = batch()
    data.update(ranking_group_weights=torch.tensor([.5] * 6 + [2.] * 3),
                ranking_batch_item_count=3)
    builder = object.__new__(RewardModelTrainer)
    with torch.no_grad():
        supervised = fitted(**builder._build_model_inputs(data))
        predicted = fitted(**builder._build_model_inputs(data, include_supervision=False))
    assert supervised.loss is not None and torch.isfinite(supervised.loss)
    assert predicted.loss is None
    for field in ("ranking_score", "activity_logits", "cosine_similarity", "score_scale",
                  "normalized_protein_embedding", "normalized_molecule_embedding"):
        torch.testing.assert_close(getattr(predicted, field), getattr(supervised, field),
                                   rtol=0, atol=0)


@pytest.mark.parametrize("eval_batch_size", [2, 7])
def test_balanced_training_crosses_evaluation_and_saves_native_checkpoint(
    tmp_path, monkeypatch, eval_batch_size,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    examples = unequal_assay_examples()
    training = RewardAssayListDataset(
        examples, seed=42, ranking_max_ligands=5,
        include_all_assays_for_contrastive=True, balance_ranking_assays=True,
    )
    evaluation = RewardEvaluationDataset(examples, ranking_max_ligands=5)
    fitted = model(ranking_normalize_per_group=True, ranking_balance_assays=True)
    trainer = RewardModelTrainer(
        model=fitted,
        args=create_training_arguments(RewardTrainerConfig(
            output_dir=str(tmp_path), num_train_epochs=1,
            per_device_train_batch_size=2, per_device_eval_batch_size=eval_batch_size,
            logging_steps=1, report_to=[], export_evaluation_predictions=True,
            metric_for_best_model="eval_macro_spearman", greater_is_better=True,
        )),
        train_dataset=training, eval_dataset=evaluation,
        fixed_train_eval_dataset=evaluation, data_collator=RewardAssayListCollator(),
    )
    assert trainer.args.device.type == "cpu"
    result = trainer.train()
    assert math.isfinite(result.training_loss)
    assert trainer.state.epoch == 1
    native_rows = [row for row in trainer.state.log_history if "eval_macro_spearman" in row]
    assert len(native_rows) == 1
    assert native_rows[0]["epoch"] == 1
    checkpoint = tmp_path / f"checkpoint-{trainer.state.global_step}"
    assert (checkpoint / "trainer_state.json").is_file()
    assert (checkpoint / "pytorch_model.bin").is_file()

    directory = tmp_path / "prediction_exports" / "trajectory"
    for prefix in ("eval", "train_eval"):
        stem = f"{prefix}_step{trainer.state.global_step:08d}"
        manifest = json.loads((directory / f"{stem}.json").read_text())
        metrics = manifest["metrics"]
        with np.load(directory / f"{stem}.npz", allow_pickle=False) as payload:
            assert np.array_equal(payload["row_index"], np.arange(len(examples)))
            scores = torch.from_numpy(payload["ranking_score"].copy())
            labels = torch.from_numpy(payload["pchembl_value"].copy())
            assays = torch.from_numpy(payload["evaluation_group_index"].copy())
        partitions = build_complete_coverage_ranking_partitions(
            pchembl_values=labels, ranking_group_ids=assays, max_list_size=5,
            num_partitions=3, seed=42, min_pchembl_span=.5,
        )
        # Independent reduction: mean lists within each original assay, then
        # equal-weight the two eligible assays, regardless of prediction batches.
        partition_losses = []
        for partition in partitions:
            per_assay = {}
            for list_id in partition.unique().tolist():
                if list_id < 0:
                    continue
                mask = partition == list_id
                owners = assays[mask].unique().tolist()
                assert len(owners) == 1
                loss = ligunity_listwise_loss(
                    scores[mask], labels[mask], torch.zeros(int(mask.sum()), dtype=torch.long),
                    normalize_per_group=True, min_pchembl_span=0,
                )
                per_assay.setdefault(owners[0], []).append(loss)
            assert len(per_assay) == 2
            partition_losses.append(torch.stack([
                torch.stack(losses).mean() for losses in per_assay.values()
            ]).mean())
        expected = float(torch.stack(partition_losses).mean())
        assert metrics[f"{prefix}_ranking_loss"] == pytest.approx(expected, abs=1e-6)
        assert metrics[f"{prefix}_num_examples"] == len(examples)
        assert metrics[f"{prefix}_contrastive_loss"] > 0
        assert metrics[f"{prefix}_loss"] == pytest.approx(
            .25 * expected + .25 * metrics[f"{prefix}_classification_loss"]
            + metrics[f"{prefix}_contrastive_loss"], abs=1e-6,
        )
