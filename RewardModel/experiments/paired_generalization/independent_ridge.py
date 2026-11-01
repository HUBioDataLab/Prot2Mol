"""Frozen-feature independent-task ridge: one molecule regressor per protein.

Observed measurements only, with exact duplicate-entity sufficient statistics.
Linear-kernel instantiation of the independent-task baseline discussed in
https://arxiv.org/html/1803.01575v1 . No missing pair is assigned a negative label.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from .baseline_common import RankingSelection, evaluate_predictions, load_partitions
    from .linear_baselines import REGULARIZATION
else:
    from run import ROOT, WORK, sha256, write_json
    from baseline_common import RankingSelection, evaluate_predictions, load_partitions
    from linear_baselines import REGULARIZATION


def ridge_path(features, label_means, counts, regularization, *, device="cpu"):
    """All ridge strengths from one primal/dual eigendecomposition, with intercept."""
    import torch
    x = torch.as_tensor(features, dtype=torch.float64, device=device)
    y = torch.as_tensor(label_means, dtype=torch.float64, device=device)
    weights = torch.as_tensor(counts, dtype=torch.float64, device=device)
    if len(x) != len(y) or weights.shape != y.shape or bool((weights <= 0).any()) or not bool(torch.isfinite(weights).all()) or not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
        raise ValueError("Finite features, labels and positive aligned observation counts required")
    if bool((y == y[0]).all()):
        return np.zeros((len(regularization), x.shape[1])), np.full(len(regularization), float(y[0]))
    weights = weights / weights.sum()
    mean_x, mean_y = weights @ x, weights @ y
    design = (x - mean_x) * torch.sqrt(weights[:, None])
    target = (y - mean_y) * torch.sqrt(weights)
    dual = len(x) < x.shape[1]
    gram = design @ design.T if dual else design.T @ design
    eigenvalues, vectors = torch.linalg.eigh(gram)
    if float(eigenvalues.min()) < -1e-8 * max(1., float(eigenvalues.max())):
        raise RuntimeError("Ridge covariance is not positive semidefinite")
    eigenvalues = eigenvalues.clamp_min(0.)
    strengths = torch.as_tensor(regularization, dtype=torch.float64, device=device)
    if bool((strengths <= 0).any()):
        raise ValueError("Strictly positive ridge strengths required")
    rhs = vectors.T @ (target if dual else design.T @ target)
    coefficients = vectors @ (rhs[:, None] / (eigenvalues[:, None] + strengths[None, :]))
    if dual:
        coefficients = design.T @ coefficients
    intercept = mean_y - mean_x @ coefficients
    return coefficients.T.cpu().numpy(), intercept.cpu().numpy()


def predict_tasks(features, molecule_indices, target_indices, coefficients, intercepts, fallback, *, batch_size=8192):
    output = np.full(len(molecule_indices), fallback, dtype=float)
    for start in range(0, len(output), batch_size):
        targets = target_indices[start:start + batch_size]
        molecules = molecule_indices[start:start + batch_size]
        known = targets >= 0
        if known.any():
            values = np.asarray(features[molecules[known]], float)
            output[start + np.flatnonzero(known)] = np.einsum("ij,ij->i", values, coefficients[targets[known]]) + intercepts[targets[known]]
    if not np.isfinite(output).all():
        raise RuntimeError("Nonfinite independent-task predictions")
    return output


def main(regime):
    import fcntl
    import torch
    if ROOT != Path("/home/atabey/Prot2Mol"):
        raise RuntimeError("Runtime restricted to the authorized Atum project")
    allowed = subprocess.check_output(["nvidia-smi", "-i", "0,2,5,6", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in allowed or torch.cuda.device_count() != 1:
        raise RuntimeError("Select exactly one authorized GPU UUID")
    torch.set_num_threads(4)
    destination = WORK / "tier2/independent_task_ridge" / regime
    destination.mkdir(parents=True, exist_ok=True)
    lock = (destination / "fit.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frames, hashes = load_partitions(regime)
    feature_dir = WORK / "tier2/frozen_features"
    manifest = json.loads((feature_dir / "complete.json").read_text())["molecule"]
    provenance = {"regime": regime, "data_sha256": hashes, "feature_provenance": manifest, "regularization": REGULARIZATION}
    if (destination / "complete.json").exists():
        previous = json.loads((destination / "complete.json").read_text())
        if any(previous[k] != v for k, v in provenance.items()):
            raise RuntimeError("Completed independent-task provenance changed")
        return
    feature_path = feature_dir / "molecule.npy"
    entities_path = feature_dir / "molecule_entities.parquet"
    if sha256(feature_path) != manifest["features_sha256"] or sha256(entities_path) != manifest["entities_sha256"]:
        raise RuntimeError("Frozen molecule features changed")
    features = np.load(feature_path, mmap_mode="r")
    entities = pd.Index(pd.read_parquet(entities_path, columns=["entity_id"]).entity_id)
    molecule_indices = {name: entities.get_indexer(frame.compound_id) for name, frame in frames.items()}
    if any((index < 0).any() for index in molecule_indices.values()):
        raise RuntimeError("Missing molecule feature row")
    linear_dir = WORK / "tier2/linear" / regime / "molecule_only_ridge"
    linear = json.loads((linear_dir / "complete.json").read_text())
    if linear["data_sha256"] != hashes or linear["feature_provenance"] != manifest:
        raise RuntimeError("Training-only standardizer provenance mismatch")
    scaler = np.load(linear_dir / "model.npz")
    mean, scale = scaler["training_mean"], scaler["training_scale"]
    training = frames["train"].assign(feature_index=molecule_indices["train"])
    pairs = training.groupby(["target_chembl_id", "feature_index"], sort=True).pchembl_value.agg(["mean", "size"]).reset_index()
    groups = pairs.groupby("target_chembl_id", sort=True).indices
    targets = pd.Index(sorted(groups))
    target_indices = {name: targets.get_indexer(frame.target_chembl_id) for name, frame in frames.items()}
    progress_path = destination / "fit_progress.json"
    shape = (len(targets), len(REGULARIZATION), features.shape[1])
    if progress_path.exists():
        previous = json.loads(progress_path.read_text())
        if any(previous[k] != v for k, v in provenance.items()):
            raise RuntimeError("Interrupted independent-task provenance changed")
        completed = previous["completed_targets"]
        coefficients = np.load(destination / "coefficients.npy", mmap_mode="r+")
        intercepts = np.load(destination / "intercepts.npy", mmap_mode="r+")
        if coefficients.shape != shape or intercepts.shape != shape[:2]:
            raise RuntimeError("Interrupted coefficient shape changed")
    else:
        completed = 0
        coefficients = np.lib.format.open_memmap(destination / "coefficients.npy", mode="w+", dtype="float64", shape=shape)
        intercepts = np.lib.format.open_memmap(destination / "intercepts.npy", mode="w+", dtype="float64", shape=shape[:2])
        pd.DataFrame({"target_chembl_id": targets}).to_parquet(destination / "targets.parquet", index=False)
    for i in range(completed, len(targets)):
        part = pairs.iloc[groups[targets[i]]]
        raw = np.asarray(features[part.feature_index.to_numpy()], float)
        standardized = (raw - mean) / scale
        # Tiny decompositions are faster on CPU; large ones use this job's GPU.
        device = "cuda" if min(standardized.shape) >= 128 else "cpu"
        beta, intercept = ridge_path(standardized, part["mean"].to_numpy(), part["size"].to_numpy(), REGULARIZATION, device=device)
        beta /= scale[None, :]
        intercept -= beta @ mean
        coefficients[i] = beta
        intercepts[i] = intercept
        if i % 100 == 0 or i + 1 == len(targets):
            coefficients.flush(); intercepts.flush()
            write_json(progress_path, {**provenance, "completed_targets": i + 1, "targets": len(targets)})
            print(f"Independent ridge {regime}: {i + 1}/{len(targets)} targets", flush=True)
    selector = RankingSelection(frames["val"])
    fallback = float(training.pchembl_value.mean())
    trials = []
    for i, strength in enumerate(REGULARIZATION):
        scores = predict_tasks(features, molecule_indices["val"], target_indices["val"], coefficients[:, i], intercepts[:, i], fallback)
        result = selector(scores)
        if result["macro_spearman"] is None:
            raise RuntimeError("No defined validation ranking groups")
        trials.append({"lambda": strength, **result})
        print(f"Independent ridge {regime} lambda={strength:g}: {result['macro_spearman']:.6f}", flush=True)
    winner = max(range(len(trials)), key=lambda i: trials[i]["macro_spearman"])
    np.savez(destination / "model.npz", coefficient=coefficients[:, winner], intercept=intercepts[:, winner],
             training_mean=mean, training_scale=scale)
    metrics = {}
    for partition in ("val", "test"):
        scores = predict_tasks(features, molecule_indices[partition], target_indices[partition], coefficients[:, winner], intercepts[:, winner], fallback)
        predictions = frames[partition].assign(ranking_score=scores, classification_score=scores)
        metrics[partition] = evaluate_predictions(predictions, regime, destination / partition)
        metrics[partition]["unseen_target_rows"] = int((target_indices[partition] < 0).sum())
    write_json(destination / "complete.json", {**provenance, "selected_lambda": trials[winner]["lambda"], "trials": trials,
        "metrics": metrics, "model_sha256": sha256(destination / "model.npz"), "training_targets": len(targets),
        "standardization": "global training-observation mean/std reused exactly from the molecule-only ridge arm",
        "fit": "independent per-target mean-squared observed-measurement error with unpenalized intercept; duplicate molecule observations collapsed with exact frequency weights; linear frozen-feature kernel",
        "selection": "one shared lambda chosen by original-eligible validation macro Spearman; test used only for final selected model",
        "unseen_target_fallback": "global training pChEMBL mean; frequency reported separately",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=["A", "B"], required=True)
    main(parser.parse_args().regime)
