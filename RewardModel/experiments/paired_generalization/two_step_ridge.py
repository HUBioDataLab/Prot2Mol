"""Two-stage frozen-feature ridge using virtual labels, without zero-filled pairs.

Stage one is observed-only independent molecule regression for each protein.
Stage two regresses those functions on frozen protein features. Linearity lets
us fit their coefficient vectors instead of materializing virtual labels for
every protein/molecule combination. This is a sparse-data two-stage adaptation;
it does not claim the complete-matrix formula applies to unmeasured affinities.
Reference: https://arxiv.org/html/1803.01575v1 , sections 2.3 and 3.1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from .baseline_common import RankingSelection, evaluate_predictions, load_partitions
    from .independent_ridge import predict_tasks
    from .linear_baselines import REGULARIZATION
else:
    from run import ROOT, WORK, sha256, write_json
    from baseline_common import RankingSelection, evaluate_predictions, load_partitions
    from independent_ridge import predict_tasks
    from linear_baselines import REGULARIZATION


def second_stage_path(protein_features, head_coefficients, strengths):
    x, heads = np.asarray(protein_features, float), np.asarray(head_coefficients, float)
    if len(x) != len(heads) or not np.isfinite(x).all() or not np.isfinite(heads).all():
        raise ValueError("Finite aligned protein features and first-stage functions required")
    mean_x, mean_h = x.mean(axis=0), heads.mean(axis=0)
    centered = x - mean_x
    gram = centered.T @ centered / len(x)
    cross = centered.T @ (heads - mean_h) / len(x)
    eigenvalues, vectors = np.linalg.eigh(gram)
    if eigenvalues.min() < -1e-8 * max(1., eigenvalues.max()):
        raise RuntimeError("Protein kernel covariance is not positive semidefinite")
    projected = vectors.T @ cross
    results = []
    for strength in strengths:
        if strength <= 0:
            raise ValueError("Positive second-stage regularization required")
        coefficient = vectors @ (projected / (np.maximum(eigenvalues, 0.) + strength)[:, None])
        results.append((coefficient, mean_h - mean_x @ coefficient))
    return results


def main(regime):
    import fcntl
    if ROOT != Path("/home/atabey/Prot2Mol"):
        raise RuntimeError("Runtime restricted to the authorized Atum project")
    destination = WORK / "tier2/two_step_ridge" / regime
    destination.mkdir(parents=True, exist_ok=True)
    lock = (destination / "fit.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frames, hashes = load_partitions(regime)
    feature_dir = WORK / "tier2/frozen_features"
    manifest = json.loads((feature_dir / "complete.json").read_text())
    provenance = {"regime": regime, "data_sha256": hashes, "feature_provenance": manifest,
                  "molecule_regularization": REGULARIZATION, "protein_regularization": REGULARIZATION}
    if (destination / "complete.json").exists():
        previous = json.loads((destination / "complete.json").read_text())
        if any(previous[k] != v for k, v in provenance.items()):
            raise RuntimeError("Completed two-step model provenance changed")
        return
    first_dir = WORK / "tier2/independent_task_ridge" / regime
    first = json.loads((first_dir / "complete.json").read_text())
    if first["data_sha256"] != hashes or first["feature_provenance"] != manifest["molecule"]:
        raise RuntimeError("Independent-task source mismatch")
    targets = pd.Index(pd.read_parquet(first_dir / "targets.parquet").target_chembl_id)
    first_coefficients = np.load(first_dir / "coefficients.npy", mmap_mode="r")
    first_intercepts = np.load(first_dir / "intercepts.npy", mmap_mode="r")
    features, entities = {}, {}
    for side in ("protein", "molecule"):
        if sha256(feature_dir / f"{side}.npy") != manifest[side]["features_sha256"]:
            raise RuntimeError("Frozen feature checksum mismatch")
        features[side] = np.load(feature_dir / f"{side}.npy", mmap_mode="r")
        entities[side] = pd.Index(pd.read_parquet(feature_dir / f"{side}_entities.parquet", columns=["entity_id"]).entity_id)
    source_positions = entities["protein"].get_indexer(targets)
    indices = {partition: (entities["molecule"].get_indexer(frame.compound_id),
                           entities["protein"].get_indexer(frame.target_chembl_id)) for partition, frame in frames.items()}
    if (source_positions < 0).any() or any((index < 0).any() for pair in indices.values() for index in pair):
        raise RuntimeError("Entity feature mapping is incomplete")
    scaler_dir = WORK / "tier2/linear" / regime / "protein_only_bce"
    scaler_result = json.loads((scaler_dir / "complete.json").read_text())
    if scaler_result["data_sha256"] != hashes or scaler_result["feature_provenance"] != manifest["protein"]:
        raise RuntimeError("Protein standardizer provenance mismatch")
    scaler = np.load(scaler_dir / "model.npz")
    mean, scale = scaler["training_mean"], scaler["training_scale"]
    training_features = (features["protein"][source_positions].astype(float) - mean) / scale
    selector, trials, best = RankingSelection(frames["val"]), [], None
    for j, molecule_strength in enumerate(REGULARIZATION):
        heads = np.column_stack([first_coefficients[:, j], first_intercepts[:, j]])
        path = second_stage_path(training_features, heads, REGULARIZATION)
        for protein_strength, (coefficient, intercept) in zip(REGULARIZATION, path):
            coefficient = coefficient / scale[:, None]
            intercept = intercept - mean @ coefficient
            predicted_heads = features["protein"] @ coefficient + intercept
            scores = predict_tasks(features["molecule"], *indices["val"], predicted_heads[:, :-1], predicted_heads[:, -1], 0.)
            metrics = selector(scores)
            if metrics["macro_spearman"] is None:
                raise RuntimeError("Undefined two-step validation metric")
            trial = {"lambda_molecule": molecule_strength, "lambda_protein": protein_strength, **metrics}
            trials.append(trial)
            if best is None or metrics["macro_spearman"] > best["macro_spearman"]:
                best = trial.copy()
                np.savez(destination / "model.npz", coefficient=coefficient, intercept=intercept)
            write_json(destination / "trials.json", {"validation_only": True, "trials": trials, "best": best})
            print(f"Two-step {regime} molecule={molecule_strength:g} protein={protein_strength:g}: {metrics['macro_spearman']:.6f}", flush=True)
    model = np.load(destination / "model.npz")
    heads = features["protein"] @ model["coefficient"] + model["intercept"]
    metrics = {}
    for partition in ("val", "test"):
        scores = predict_tasks(features["molecule"], *indices[partition], heads[:, :-1], heads[:, -1], 0.)
        predictions = frames[partition].assign(ranking_score=scores, classification_score=scores)
        metrics[partition] = evaluate_predictions(predictions, regime, destination / partition)
    write_json(destination / "complete.json", {**provenance, "trials": trials, "selected": best, "metrics": metrics,
        "model_sha256": sha256(destination / "model.npz"), "first_stage_selected_lambda_used": False,
        "missing_data": "first-stage fits use only measured pairs; stage-two labels are virtual first-stage predictions, never zero-filled unobserved affinities",
        "weighting": "first stage mean loss per protein with observed-row weights; second stage equal weight per training protein, unpenalized intercept",
        "selection": "25 paired regularization settings per split, validation macro Spearman only; test evaluated after selection",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=["A", "B"], required=True)
    main(parser.parse_args().regime)
