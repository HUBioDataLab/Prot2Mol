import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from datasets import Dataset

from compare_ranking_checkpoints import compare_runs
from conftest import DummyEncoder, DummyTokenizer
from reward_model.analysis.ranking_head import (
    analyze_predictions,
    capture_head_activations,
    load_assay_manifest,
    margin_pair_counts,
    run_fusion_cosine_sensitivity,
    run_input_sensitivity,
    score_fusion_cosines,
    score_dataset_pairs,
    select_complete_assays,
    write_assay_manifest,
    write_fusion_cosine_analysis,
    write_prediction_analysis,
    write_sensitivity_analysis,
)
from reward_model.model import LoadedEncoder, RewardModel, RewardModelConfig


def _analysis_dataset() -> Dataset:
    rows = []
    example_id = 0
    group_specs = {
        "T1__A1": (1, [4.0, 6.0, 8.0]),
        "T2__A2": (9, [5.0, 7.0, 9.0]),
        "T3__A3": (2, [4.5, 6.5, 8.5]),
    }
    for group_id, (protein_token, targets) in group_specs.items():
        target_id, assay_id = group_id.split("__")
        for ligand_offset, target in enumerate(targets, start=1):
            rows.append(
                {
                    "example_id": example_id,
                    "group_id": group_id,
                    "target_chembl_id": target_id,
                    "assay_id": assay_id,
                    "compound_id": f"M{example_id}",
                    "pchembl_value": target,
                    "binary_label": int(target >= 6.0),
                    "protein_input_ids": [protein_token, protein_token, 0],
                    "protein_attention_mask": [1, 1, 0],
                    "molecule_input_ids": [ligand_offset, ligand_offset + 1, 0],
                    "molecule_attention_mask": [1, 1, 0],
                    "protein_length": 2,
                    "molecule_length": 2,
                }
            )
            example_id += 1
    rows.extend(
        [
            {
                "example_id": example_id,
                "group_id": "T4__A4",
                "target_chembl_id": "T4",
                "assay_id": "A4",
                "compound_id": f"M{example_id}",
                "pchembl_value": 5.0 + offset,
                "binary_label": int(offset == 1),
                "protein_input_ids": [4, 4, 0],
                "protein_attention_mask": [1, 1, 0],
                "molecule_input_ids": [offset + 1, offset + 2, 0],
                "molecule_attention_mask": [1, 1, 0],
                "protein_length": 2,
                "molecule_length": 2,
            }
            for offset in range(2)
        ]
    )
    return Dataset.from_list(rows)


def _dummy_model(*, fusion_residual: bool = False) -> RewardModel:
    torch.manual_seed(7)
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
    return RewardModel(
        RewardModelConfig(
            protein_model_name_or_path="protein/dummy",
            molecule_model_name_or_path="molecule/dummy",
            fusion_hidden_dim=10,
            fusion_num_heads=2,
            fusion_residual=fusion_residual,
            dropout=0.0,
        ),
        protein_bundle=protein_bundle,
        molecule_bundle=molecule_bundle,
    )


class _FormulaModel(torch.nn.Module):
    def __init__(self, *, interaction: bool):
        super().__init__()
        self.interaction = interaction
        self.protein_tokenizer = DummyTokenizer()
        self.molecule_tokenizer = DummyTokenizer()

    def forward(
        self,
        protein_input_ids,
        molecule_input_ids,
        protein_attention_mask,
        molecule_attention_mask,
        return_dict=True,
    ):
        protein = (
            protein_input_ids.float() * protein_attention_mask.float()
        ).sum(dim=1) / protein_attention_mask.sum(dim=1)
        molecule = (
            molecule_input_ids.float() * molecule_attention_mask.float()
        ).sum(dim=1) / molecule_attention_mask.sum(dim=1)
        ranking_score = (protein - 5.0) * molecule if self.interaction else molecule
        activity_logits = molecule - 2.5
        return SimpleNamespace(
            ranking_score=ranking_score,
            activity_logits=activity_logits,
            activity_probability=torch.sigmoid(activity_logits),
            joint_embedding=torch.stack([protein, molecule], dim=1),
        )


class _FusionCosineFormulaModel(torch.nn.Module):
    def __init__(self, *, collapse_after_fusion: bool):
        super().__init__()
        self.collapse_after_fusion = collapse_after_fusion
        self.protein_tokenizer = DummyTokenizer()
        self.molecule_tokenizer = DummyTokenizer()
        self.config = SimpleNamespace(pooling_type="mean", fusion_residual=True)

    def forward(
        self,
        protein_input_ids,
        molecule_input_ids,
        protein_attention_mask,
        molecule_attention_mask,
        return_token_embeddings=False,
        return_dict=True,
    ):
        protein_values = protein_input_ids.float()
        molecule_values = molecule_input_ids.float()
        protein_tokens = torch.stack(
            [protein_values, torch.ones_like(protein_values)], dim=-1
        )
        molecule_tokens = torch.stack(
            [molecule_values, torch.ones_like(molecule_values)], dim=-1
        )
        if self.collapse_after_fusion:
            fused_protein = torch.ones_like(protein_tokens)
            fused_molecule = torch.ones_like(molecule_tokens)
        else:
            fused_protein = protein_tokens
            fused_molecule = molecule_tokens
        return SimpleNamespace(
            protein_token_embeddings=(protein_tokens if return_token_embeddings else None),
            molecule_token_embeddings=(molecule_tokens if return_token_embeddings else None),
            fused_protein_tokens=(fused_protein if return_token_embeddings else None),
            fused_molecule_tokens=(fused_molecule if return_token_embeddings else None),
            protein_attention_mask=protein_attention_mask.bool(),
            molecule_attention_mask=molecule_attention_mask.bool(),
        )


def _brute_margin_pair_counts(scores, targets, margin):
    correct = 0.0
    count = 0
    for left in range(len(scores)):
        for right in range(left + 1, len(scores)):
            target_delta = targets[left] - targets[right]
            if abs(target_delta) <= margin:
                continue
            score_delta = scores[left] - scores[right]
            comparison = score_delta * target_delta
            correct += float(comparison > 0) + 0.5 * float(comparison == 0)
            count += 1
    return correct, count


@pytest.mark.parametrize("margin", [0.0, 0.5, 1.25])
def test_margin_pair_counts_matches_brute_force_with_ties(margin):
    scores = [1.0, 3.0, 3.0, -2.0, 0.5]
    targets = [5.0, 8.0, 7.0, 4.0, 6.0]
    expected = _brute_margin_pair_counts(scores, targets, margin)

    assert margin_pair_counts(scores, targets, affinity_margin=margin) == expected


def test_margin_pair_counts_handles_large_assay_without_pair_matrix():
    targets = np.arange(50_000, dtype=np.float64)
    scores = targets.copy()

    correct, count = margin_pair_counts(scores, targets, affinity_margin=0.5)

    assert count == 50_000 * 49_999 // 2
    assert correct == pytest.approx(float(count))


def test_complete_assay_selection_is_deterministic_and_excludes_short_groups(tmp_path):
    dataset = _analysis_dataset()
    first, first_groups = select_complete_assays(dataset, max_assays=2, seed=19)
    second, second_groups = select_complete_assays(dataset, max_assays=2, seed=19)

    assert first_groups == second_groups
    assert len(first_groups) == 2
    assert set(first["group_id"]) == set(first_groups)
    assert len(first) == 6
    assert "T4__A4" not in first_groups

    manifest_path = tmp_path / "assays.json"
    write_assay_manifest(
        str(manifest_path),
        split="val",
        assay_ids=first_groups,
        dataset=first,
        seed=19,
    )
    assert load_assay_manifest(str(manifest_path), split="val") == first_groups
    with pytest.raises(ValueError, match="does not match"):
        load_assay_manifest(str(manifest_path), split="train")


def test_prediction_analysis_exposes_aligned_and_reversed_assays(tmp_path):
    scored = pd.DataFrame(
        {
            "example_id": range(6),
            "group_id": ["T1__A1"] * 3 + ["T2__A2"] * 3,
            "target_chembl_id": ["T1"] * 3 + ["T2"] * 3,
            "assay_id": ["A1"] * 3 + ["A2"] * 3,
            "compound_id": [f"M{i}" for i in range(6)],
            "pchembl_value": [4.0, 6.0, 8.0, 4.0, 6.0, 8.0],
            "binary_label": [0, 1, 1, 0, 1, 1],
            "ranking_score": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            "activity_logit": [-1.0, 1.0, 2.0, -1.0, 1.0, 2.0],
            "activity_probability": [0.2, 0.7, 0.9, 0.2, 0.7, 0.9],
            "joint_embedding_norm": [1.0] * 6,
            "protein_source_index": range(6),
            "molecule_source_index": range(6),
        }
    )

    predictions, assays, summary = analyze_predictions(
        scored,
        split="val",
        affinity_margin=0.5,
        temperature=1.0,
    )

    by_group = assays.set_index("group_id")
    assert by_group.loc["T1__A1", "spearman"] == pytest.approx(1.0)
    assert by_group.loc["T2__A2", "spearman"] == pytest.approx(-1.0)
    assert by_group.loc["T1__A1", "margin_pair_accuracy"] == pytest.approx(1.0)
    assert by_group.loc["T2__A2", "margin_pair_accuracy"] == pytest.approx(0.0)
    assert summary["weighted_spearman"] == pytest.approx(0.0)
    assert predictions.groupby("group_id")[
        "pl_strength_probability"
    ].sum().tolist() == pytest.approx([1.0, 1.0])

    paths = write_prediction_analysis(
        str(tmp_path),
        predictions=predictions,
        assay_summary=assays,
        split_summary=summary,
    )
    assert all(os.path.exists(path) for path in paths.values())
    assert "Ranking-head inspection" in (tmp_path / "val_report.html").read_text(
        encoding="utf-8"
    )
    assert len(pd.read_parquet(paths["predictions"])) == 6


def test_scoring_and_activation_collection_run_on_cpu_and_restore_train_mode():
    dataset, _ = select_complete_assays(_analysis_dataset())
    model = _dummy_model()
    model.train()

    scored = score_dataset_pairs(
        model,
        dataset,
        batch_size=4,
        device=torch.device("cpu"),
    )
    predictions, activations = capture_head_activations(
        model,
        dataset,
        batch_size=4,
        device=torch.device("cpu"),
    )

    assert model.training is True
    assert len(scored) == len(dataset)
    assert np.isfinite(scored["ranking_score"]).all()
    assert len(predictions) == len(dataset)
    assert activations["module"].str.startswith("ranking_head.").any()
    assert activations["module"].str.startswith("classification_head.").any()
    assert int(activations["nonfinite_count"].sum()) == 0
    assert (activations["count"] > 0).all()


def test_input_sensitivity_detects_ligand_only_and_interacting_scores(tmp_path):
    dataset, _ = select_complete_assays(_analysis_dataset())
    ligand_only = _FormulaModel(interaction=False)
    rows, by_repeat, summary = run_input_sensitivity(
        ligand_only,
        dataset,
        split="val",
        batch_size=4,
        device=torch.device("cpu"),
        num_shuffles=3,
        seed=5,
    )

    protein_metrics = summary["perturbations"]["protein_shuffled"]
    ligand_metrics = summary["perturbations"]["ligand_shuffled"]
    assert protein_metrics["centered_score_mae"]["mean"] == pytest.approx(0.0)
    assert protein_metrics["within_assay_rank_stability"]["mean"] == pytest.approx(1.0)
    assert protein_metrics["target_spearman_delta"]["mean"] == pytest.approx(0.0)
    assert ligand_metrics["centered_score_mae"]["mean"] > 0.0
    assert ligand_metrics["target_spearman_delta"]["mean"] < 0.0
    assert summary["protein_to_ligand_centered_sensitivity_ratio"] == pytest.approx(0.0)
    assert len(rows) == len(dataset) * 2 * 3
    assert len(by_repeat) == 6
    protein_rows = rows[rows["perturbation"] == "protein_shuffled"]
    assert (
        protein_rows["protein_source_group_id"] != protein_rows["group_id"]
    ).all()
    assert (
        protein_rows.groupby(["repeat", "group_id"])["protein_source_group_id"]
        .nunique()
        .eq(1)
        .all()
    )

    paths = write_sensitivity_analysis(
        str(tmp_path),
        split="val",
        sensitivity_rows=rows,
        by_repeat=by_repeat,
        summary=summary,
    )
    assert all(os.path.exists(path) for path in paths.values())
    assert json.loads((tmp_path / "val_input_sensitivity_summary.json").read_text())[
        "num_shuffles"
    ] == 3

    interacting = _FormulaModel(interaction=True)
    _, _, interacting_summary = run_input_sensitivity(
        interacting,
        dataset,
        split="val",
        batch_size=4,
        device=torch.device("cpu"),
        num_shuffles=3,
        seed=5,
    )
    assert (
        interacting_summary["perturbations"]["protein_shuffled"]
        ["centered_score_mae"]["mean"]
        > 0.0
    )


def test_fusion_cosine_analysis_detects_post_fusion_collapse(tmp_path):
    dataset, _ = select_complete_assays(_analysis_dataset())
    model = _FusionCosineFormulaModel(collapse_after_fusion=True)
    model.train()

    scored = score_fusion_cosines(
        model,
        dataset,
        batch_size=4,
        device=torch.device("cpu"),
    )
    assert model.training is True
    assert scored["pre_fusion_cosine"].std() > 0.0
    assert scored["post_fusion_cosine"].std() == pytest.approx(0.0)

    rows, by_repeat, distributions, summary = run_fusion_cosine_sensitivity(
        model,
        dataset,
        split="val",
        batch_size=4,
        device=torch.device("cpu"),
        num_shuffles=2,
        seed=5,
    )

    assert model.training is True
    assert len(rows) == len(dataset) * 2 * 2
    assert len(by_repeat) == 2 * 2 * 2
    assert len(distributions) == 2 + 2 * 2 * 2
    assert summary["pooling_type"] == "mean"
    assert summary["fusion_residual"] is True
    assert summary["stages"]["pre_fusion"]["correct"]["distribution"][
        "std"
    ] > 0.0
    assert summary["stages"]["post_fusion"]["correct"]["distribution"][
        "std"
    ] == pytest.approx(0.0)
    assert summary["stages"]["post_fusion"]["perturbations"][
        "protein_shuffled"
    ]["cosine_mae"]["mean"] == pytest.approx(0.0)
    assert summary["fusion_effect"][
        "correct_cosine_std_ratio_post_to_pre"
    ] == pytest.approx(0.0)

    paths = write_fusion_cosine_analysis(
        str(tmp_path),
        split="val",
        sensitivity_rows=rows,
        by_repeat=by_repeat,
        distributions=distributions,
        summary=summary,
    )
    assert all(os.path.exists(path) for path in paths.values())
    saved = json.loads((tmp_path / "val_fusion_cosine_summary.json").read_text())
    assert saved["num_shuffles"] == 2


def test_fusion_cosine_scoring_runs_through_real_residual_model():
    dataset, _ = select_complete_assays(_analysis_dataset())
    model = _dummy_model(fusion_residual=True)
    model.train()

    scored = score_fusion_cosines(
        model,
        dataset,
        batch_size=4,
        device=torch.device("cpu"),
    )

    assert model.training is True
    assert model.config.fusion_residual is True
    assert len(scored) == len(dataset)
    assert np.isfinite(scored["pre_fusion_cosine"]).all()
    assert np.isfinite(scored["post_fusion_cosine"]).all()
    assert not np.allclose(
        scored["pre_fusion_cosine"], scored["post_fusion_cosine"]
    )


def test_checkpoint_comparison_reads_prediction_artifacts(tmp_path):
    run_specs = []
    for label, spearman in (("step1000", 0.2), ("final", 0.1)):
        run_dir = tmp_path / label
        run_dir.mkdir()
        summary = {
            "val": {
                "metrics": {
                    "split": "val",
                    "weighted_spearman": spearman,
                    "margin_pair_accuracy": 0.6,
                },
                "artifacts": {},
            }
        }
        (run_dir / "prediction_analysis_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        pd.DataFrame(
            {
                "split": ["val"],
                "group_id": ["T1__A1"],
                "spearman": [spearman],
            }
        ).to_csv(run_dir / "val_assay_summary.csv", index=False)
        run_specs.append((label, str(run_dir)))

    metrics, assays = compare_runs(run_specs)

    assert metrics["checkpoint"].tolist() == ["step1000", "final"]
    assert metrics["weighted_spearman"].tolist() == pytest.approx([0.2, 0.1])
    assert assays["checkpoint"].tolist() == ["step1000", "final"]


@pytest.mark.parametrize(
    ("script_name", "expected_option"),
    [
        ("inspect_ranking_predictions.py", "--checkpoint"),
        ("inspect_ranking_input_sensitivity.py", "--checkpoint"),
        ("inspect_fusion_cosine_sensitivity.py", "--checkpoint"),
        ("inspect_ranking_head_activations.py", "--checkpoint"),
        ("compare_ranking_checkpoints.py", "--runs"),
    ],
)
def test_analysis_cli_help_runs(script_name, expected_option):
    result = subprocess.run(
        [sys.executable, script_name, "--help"],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert expected_option in result.stdout
