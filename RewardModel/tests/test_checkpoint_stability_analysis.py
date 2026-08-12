from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset

from conftest import DummyEncoder, DummyTokenizer
from reward_model.analysis.checkpoint_stability import (
    _optimizer_state_parameter_names,
    analyze_contrastive_retrieval,
    analyze_embedding_matrix,
    analyze_model_stability,
    compare_encoder_layer_snapshots,
    compare_optimizer_state_reports,
    compare_parameter_snapshots,
    compare_representation_reports,
    compare_stability_reports,
    prepare_diagnostic_feature_batches,
    resolve_checkpoint_pair,
    snapshot_parameters,
    summarize_optimizer_state,
)
from reward_model.model import LoadedEncoder, RewardModel, RewardModelConfig
from reward_model.training import RewardModelTrainer, RewardTrainerConfig
from reward_model.training.trainer import create_training_arguments


def _build_cosine_model() -> RewardModel:
    return RewardModel(
        config=RewardModelConfig(
            protein_model_name_or_path="protein/dummy",
            molecule_model_name_or_path="molecule/dummy",
            fusion_hidden_dim=10,
            fusion_num_heads=2,
            projection_type="nonlinear",
            pooling_type="cls",
            pair_scoring_mode="cosine",
            ranking_loss_weight=0.5,
            contrastive_loss_weight=0.5,
            classification_loss_weight=0.0,
            ranking_temperature=0.1,
            dropout=0.0,
        ),
        protein_bundle=LoadedEncoder(
            name_or_path="protein/dummy",
            tokenizer=DummyTokenizer(),
            model=DummyEncoder(hidden_size=6),
            hidden_size=6,
        ),
        molecule_bundle=LoadedEncoder(
            name_or_path="molecule/dummy",
            tokenizer=DummyTokenizer(),
            model=DummyEncoder(hidden_size=8),
            hidden_size=8,
        ),
    )


def _row(
    *,
    protein_token: int,
    molecule_token: int,
    target: str,
    assay: str,
    compound: str,
    pchembl: float,
) -> dict:
    return {
        "example_id": molecule_token,
        "group_id": f"{target}::{assay}",
        "target_chembl_id": target,
        "assay_id": assay,
        "compound_id": compound,
        "protein_input_ids": [protein_token, 2, 0],
        "protein_attention_mask": [1, 1, 0],
        "molecule_input_ids": [molecule_token, 3, 0],
        "molecule_attention_mask": [1, 1, 0],
        "binary_label": float(pchembl >= 6.0),
        "pchembl_value": pchembl,
    }


def _feature_batches() -> list[list[dict]]:
    features = []
    for group_index, target in enumerate(("T1", "T2", "T3")):
        rows = [
            _row(
                protein_token=group_index + 1,
                molecule_token=group_index * 3 + offset + 1,
                target=target,
                assay=f"A{group_index}",
                compound=f"C{group_index}_{offset}",
                pchembl=value,
            )
            for offset, value in enumerate((5.0, 6.0, 7.0))
        ]
        features.append(
            {
                "rows": rows,
                "example_indices": list(range(group_index * 3, group_index * 3 + 3)),
                "contrastive_group_sizes": [3],
                "contrastive_assay_ids": [f"{target}::A{group_index}"],
                "ranking_group_flags": [True],
                "ranking_group_sizes": [3],
                "ranking_assay_ids": [f"{target}::A{group_index}"],
            }
        )
    return [features]


def test_resolve_checkpoint_pair_uses_latest_state_and_relocates_best(tmp_path):
    run_dir = tmp_path / "run"
    best = run_dir / "checkpoint-10"
    last = run_dir / "checkpoint-20"
    best.mkdir(parents=True)
    last.mkdir()
    (last / "trainer_state.json").write_text(
        json.dumps(
            {
                "best_model_checkpoint": "/old/server/output/checkpoint-10",
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_checkpoint_pair(run_dir=str(run_dir))

    assert resolved.best == str(best.resolve())
    assert resolved.last == str(last.resolve())


def test_analyze_model_stability_reports_projection_and_module_gradients():
    torch.manual_seed(7)
    model = _build_cosine_model()

    report = analyze_model_stability(
        model,
        _feature_batches(),
        device=torch.device("cpu"),
        precision="fp32",
        mode="eval",
        seed=42,
        label="checkpoint",
    )

    assert report["num_batches"] == 1
    assert report["num_examples"] == 9
    assert report["nonfinite_batches"] == 0
    assert report["distributions"]["protein_raw_norm"]["min"] > 0.0
    assert report["distributions"]["molecule_raw_norm"]["min"] > 0.0
    assert report["distributions"]["cosine"]["finite_count"] == 9
    batch = report["batches"][0]
    assert batch["num_contrastive_lists"] == 3
    assert batch["num_ranking_lists"] == 3
    assert batch["ranking_normalized_embedding_grad_norm"] > 0.0
    assert batch["contrastive_normalized_embedding_grad_norm"] > 0.0
    assert batch["ranking_raw_projection_grad_norm"] > 0.0
    assert batch["contrastive_raw_projection_grad_norm"] > 0.0
    assert batch["protein_encoder_grad_norm"] > 0.0
    assert batch["molecule_encoder_grad_norm"] > 0.0
    assert batch["protein_projection_grad_norm"] > 0.0
    assert batch["molecule_projection_grad_norm"] > 0.0
    assert 0.0 < batch["protein_relu_positive_fraction"] < 1.0
    assert 0.0 < batch["molecule_relu_positive_fraction"] < 1.0
    assert "molecule_encoder_pooled" in report["representations"]
    assert "molecule_linear1" in report["representations"]
    assert "molecule_relu" in report["representations"]
    assert "molecule_raw" in report["representations"]
    assert "molecule_normalized" in report["representations"]
    retrieval = report["contrastive_retrieval"]["aggregate"]
    assert retrieval["num_groups"] == 3
    assert retrieval["num_ligands"] == 9
    assert retrieval["protein_to_molecule"]["queries"] == 9
    assert retrieval["molecule_to_protein"]["queries"] == 9


def test_prepare_diagnostic_feature_batches_is_seeded_and_bounded(tmp_path):
    rows = [
        row
        for feature in _feature_batches()[0]
        for row in feature["rows"]
    ]
    tokenized_dir = tmp_path / "tokenized"
    Dataset.from_list(rows).save_to_disk(str(tokenized_dir / "train_examples"))
    config = SimpleNamespace(
        data=SimpleNamespace(
            tokenized_dataset_dir=str(tokenized_dir),
            ranking_min_pchembl_span=0.5,
            ranking_max_ligands=16,
            ranking_opportunity_divisor=32,
        )
    )

    first, first_metadata = prepare_diagnostic_feature_batches(
        config,
        split="train",
        max_assays=3,
        batch_items=2,
        max_batches=1,
        seed=42,
    )
    second, second_metadata = prepare_diagnostic_feature_batches(
        config,
        split="train",
        max_assays=3,
        batch_items=2,
        max_batches=1,
        seed=42,
    )

    assert first == second
    assert first_metadata == second_metadata
    assert first_metadata["selected_assays"] == 3
    assert first_metadata["analyzed_batches"] == 1
    assert first_metadata["analyzed_dataset_items"] == 2


def test_small_projection_norm_is_detected_as_gradient_amplification():
    torch.manual_seed(11)
    best_model = _build_cosine_model()
    last_model = copy.deepcopy(best_model)
    with torch.no_grad():
        for projection in (
            last_model.protein_projection,
            last_model.molecule_projection,
        ):
            projection.linear2.weight.mul_(1.0e-4)
            projection.linear2.bias.mul_(1.0e-4)

    kwargs = {
        "feature_batches": _feature_batches(),
        "device": torch.device("cpu"),
        "precision": "fp32",
        "mode": "eval",
        "seed": 42,
    }
    best_report = analyze_model_stability(best_model, label="best", **kwargs)
    last_report = analyze_model_stability(last_model, label="last", **kwargs)
    comparison = compare_stability_reports(best_report, last_report)

    assert comparison["flags"]["raw_projection_norm_collapse"] is True
    assert comparison["flags"]["parameter_gradient_explosion"] is True
    assert comparison["metrics"]["protein_raw_norm"][
        "last_to_best_ratio"
    ] == pytest.approx(1.0e-4, rel=1.0e-3)
    assert comparison["metrics"]["molecule_raw_norm"][
        "last_to_best_ratio"
    ] == pytest.approx(1.0e-4, rel=1.0e-3)


def test_compare_parameter_snapshots_attributes_module_drift():
    torch.manual_seed(13)
    best_model = _build_cosine_model()
    last_model = copy.deepcopy(best_model)
    with torch.no_grad():
        last_model.protein_projection.linear2.bias.add_(0.25)

    drift = compare_parameter_snapshots(
        snapshot_parameters(best_model),
        snapshot_parameters(last_model),
    )

    assert drift["protein_projection"]["delta_norm"] > 0.0
    assert drift["protein_projection"]["relative_delta_to_best"] > 0.0
    assert drift["protein_encoder"]["delta_norm"] == pytest.approx(0.0)
    assert drift["molecule_encoder"]["delta_norm"] == pytest.approx(0.0)
    assert drift["molecule_projection"]["delta_norm"] == pytest.approx(0.0)


def test_optimizer_state_summary_attributes_moments_to_modules():
    state = {
        "state": {
            0: {
                "step": torch.tensor(12.0),
                "exp_avg": torch.tensor([3.0, 4.0]),
                "exp_avg_sq": torch.tensor([1.0, 9.0]),
            },
            1: {
                "step": torch.tensor(12.0),
                "exp_avg": torch.tensor([0.5]),
                "exp_avg_sq": torch.tensor([0.25]),
            },
        },
        "param_groups": [{"params": [0, 1]}],
    }

    report = summarize_optimizer_state(
        state,
        {
            0: "protein_projection.linear2.weight",
            1: "molecule_encoder.layer.weight",
        },
    )

    assert report["state_entries"] == 2
    assert report["modules"]["all"]["parameter_states"] == 2
    assert report["modules"]["protein_projection"]["exp_avg_l2"] == pytest.approx(
        5.0
    )
    assert report["modules"]["protein_projection"]["exp_avg_sq_max"] == 9.0
    assert report["modules"]["molecule_encoder"]["max_stored_rms"] == 0.5
    assert report["top_exp_avg_sq"][0]["name"] == (
        "protein_projection.linear2.weight"
    )


def test_optimizer_comparison_flags_stored_second_moment_growth():
    def report(value: float) -> dict:
        return {
            "available": True,
            "modules": {
                "all": {
                    "exp_avg_l2": 1.0,
                    "exp_avg_sq_mean": value,
                    "exp_avg_sq_max": value,
                    "max_stored_rms": value**0.5,
                    "normalized_moment_l2": 1.0,
                    "normalized_moment_max_abs": 1.0,
                    "nonfinite_tensors": 0,
                },
                "protein_projection": {
                    "exp_avg_l2": 1.0,
                    "exp_avg_sq_mean": value,
                    "exp_avg_sq_max": value,
                    "max_stored_rms": value**0.5,
                    "normalized_moment_l2": 1.0,
                    "normalized_moment_max_abs": 1.0,
                    "nonfinite_tensors": 0,
                },
            },
        }

    comparison = compare_optimizer_state_reports(report(1.0), report(25.0))

    assert comparison["available"] is True
    assert comparison["flags"]["stored_second_moment_growth"] is True
    assert comparison["modules"]["protein_projection"]["exp_avg_sq_max"][
        "last_to_best_ratio"
    ] == 25.0


def test_optimizer_parameter_mapping_matches_trainer_group_order(tmp_path):
    model = _build_cosine_model()
    trainer_config = RewardTrainerConfig(
        output_dir=str(tmp_path / "output"),
        learning_rate=1.0e-4,
        protein_encoder_learning_rate=1.0e-5,
        molecule_encoder_learning_rate=3.0e-6,
        projection_learning_rate=1.0e-3,
        optim="adamw_torch",
    )
    trainer = RewardModelTrainer(
        model=model,
        eval_dataset=[0],
        args=create_training_arguments(trainer_config),
    )
    optimizer = trainer.create_optimizer()
    optimizer_state = optimizer.state_dict()

    mapped = _optimizer_state_parameter_names(
        optimizer_state,
        model,
        SimpleNamespace(training=trainer_config),
    )
    name_by_object_id = {
        id(parameter): name for name, parameter in model.named_parameters()
    }
    actual_names = [
        name_by_object_id[id(parameter)]
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    mapped_names = [
        mapped[state_id]
        for group in optimizer_state["param_groups"]
        for state_id in group["params"]
    ]

    assert mapped_names == actual_names


def test_embedding_matrix_exposes_directional_collapse():
    diverse = analyze_embedding_matrix(torch.eye(4))
    collapsed = analyze_embedding_matrix(
        torch.tensor([[3.0, 4.0]]).repeat(4, 1),
        identity_ids=["A", "A", "B", "B"],
    )

    assert diverse["effective_rank_centered"] > 2.0
    assert diverse["mean_pairwise_cosine"] == pytest.approx(0.0)
    assert collapsed["effective_rank_centered"] == pytest.approx(0.0)
    assert collapsed["variance_trace"] == pytest.approx(0.0)
    assert collapsed["mean_pairwise_cosine"] == pytest.approx(1.0)
    assert collapsed["exact_unique_ratio"] == pytest.approx(0.25)
    assert collapsed["identity_count"] == 2
    assert collapsed["identity_deduplicated"]["num_vectors"] == 2
    assert collapsed["within_identity_rms_radius"] == pytest.approx(0.0)


def test_contrastive_retrieval_reports_strict_top1_and_margin():
    protein = torch.eye(2)
    molecule = torch.eye(2)
    batch = {
        "contrastive_group_ids": torch.tensor([0, 1]),
        "pchembl_values": torch.tensor([7.0, 7.0]),
        "contrastive_target_ids": torch.tensor([0, 1]),
        "contrastive_molecule_ids": torch.tensor([0, 1]),
    }
    rows = [
        {"group_id": "A0", "compound_id": "C0"},
        {"group_id": "A1", "compound_id": "C1"},
    ]

    perfect = analyze_contrastive_retrieval(
        protein,
        molecule,
        batch,
        temperature=0.1,
        active_threshold=5.0,
        row_metadata=rows,
    )
    reversed_report = analyze_contrastive_retrieval(
        protein,
        molecule.flip(0),
        batch,
        temperature=0.1,
        active_threshold=5.0,
        row_metadata=rows,
    )

    for direction in ("protein_to_molecule", "molecule_to_protein"):
        assert perfect[direction]["strict_top1_accuracy"] == pytest.approx(1.0)
        assert perfect[direction]["margin"]["mean"] == pytest.approx(10.0)
        assert reversed_report[direction]["strict_top1_accuracy"] == pytest.approx(
            0.0
        )
        assert reversed_report[direction]["margin"]["mean"] == pytest.approx(
            -10.0
        )


def test_encoder_layer_drift_is_attributed_to_exact_layer():
    best = {
        "molecule_encoder.embeddings.weight": torch.ones(2),
        "molecule_encoder.encoder.layer.0.attention.weight": torch.ones(2),
        "molecule_encoder.encoder.layer.1.attention.weight": torch.ones(2),
        "protein_encoder.encoder.layer.0.attention.weight": torch.ones(2),
    }
    last = {name: value.clone() for name, value in best.items()}
    last["molecule_encoder.encoder.layer.1.attention.weight"].add_(4.0)

    report = compare_encoder_layer_snapshots(best, last)

    molecule = report["molecule_encoder"]
    assert molecule["regions"]["encoder_layer_00"]["delta_norm"] == 0.0
    assert molecule["regions"]["encoder_layer_01"]["delta_norm"] > 0.0
    assert molecule["largest_absolute_drift"][0]["region"] == "encoder_layer_01"
    assert report["protein_encoder"]["regions"]["encoder_layer_00"][
        "delta_norm"
    ] == 0.0


def test_representation_comparison_flags_norm_and_variance_collapse():
    def report(norm: float, variance: float, accuracy: float) -> dict:
        boundary = {
            "norm": {"median": norm},
            "variance_trace": variance,
            "rms_radius": variance**0.5,
            "centroid_norm": norm,
            "radius_to_centroid_norm": variance**0.5 / norm,
            "mean_pairwise_cosine": 0.5,
            "exact_unique_ratio": 1.0,
            "normalized_unique_ratio_1e4": 1.0,
            "active_variance_dimension_fraction": 1.0,
            "effective_rank_centered": 4.0,
            "stable_rank_centered": 2.0,
            "identity_deduplicated": {
                "variance_trace": variance,
                "rms_radius": variance**0.5,
                "mean_pairwise_cosine": 0.5,
                "effective_rank_centered": 4.0,
                "stable_rank_centered": 2.0,
            },
        }
        retrieval = {
            direction: {
                "strict_top1_accuracy": accuracy,
                "mean_margin": accuracy - 0.5,
                "mean_cosine_margin": accuracy - 0.5,
            }
            for direction in ("protein_to_molecule", "molecule_to_protein")
        }
        return {
            "mode": "eval",
            "representations": {"molecule_raw": boundary},
            "contrastive_retrieval": {"aggregate": retrieval},
        }

    comparison = compare_representation_reports(
        report(10.0, 5.0, 0.8),
        report(100.0, 0.01, 0.1),
    )

    assert comparison["flags"]["molecule_projection_norm_explosion"] is True
    assert comparison["flags"]["molecule_boundary_variance_collapse"] is True
    assert comparison["flags"]["contrastive_retrieval_accuracy_collapse"] is True
