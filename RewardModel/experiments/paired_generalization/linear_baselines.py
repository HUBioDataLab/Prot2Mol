"""Training-only frozen-feature molecule ridge and protein-only BCE baselines.

Repeated entity observations are compressed into exact sufficient statistics,
not imputed into an unobserved protein-molecule matrix. All five regularization
settings are evaluated only on validation; test is evaluated for the winner.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from .baseline_common import RankingSelection, evaluate_predictions, load_partitions
else:
    from run import ROOT, WORK, sha256, write_json
    from baseline_common import RankingSelection, evaluate_predictions, load_partitions

REGULARIZATION = [1e-6, 1e-4, .01, 1., 100.]


def sufficient_statistics(features, row_indices, labels, *, device="cpu", batch_size=8192):
    import torch
    indices, y = np.asarray(row_indices, int), np.asarray(labels, float)
    if indices.shape != y.shape or not np.isfinite(y).all() or (indices < 0).any() or (indices >= len(features)).any():
        raise ValueError("Invalid aligned training observations")
    counts = np.bincount(indices, minlength=len(features)).astype(float)
    totals = np.bincount(indices, weights=y, minlength=len(features))
    selected = np.flatnonzero(counts)
    dimension = features.shape[1]
    gram = torch.zeros((dimension, dimension), dtype=torch.float64, device=device)
    sum_x = torch.zeros(dimension, dtype=torch.float64, device=device)
    sum_xy = torch.zeros_like(sum_x)
    for start in range(0, len(selected), batch_size):
        current = selected[start:start + batch_size]
        x = torch.as_tensor(np.array(features[current]), dtype=torch.float64, device=device)
        count = torch.as_tensor(counts[current], dtype=torch.float64, device=device)
        total = torch.as_tensor(totals[current], dtype=torch.float64, device=device)
        gram += x.T @ (x * count[:, None])
        sum_x += x.T @ count
        sum_xy += x.T @ total
    mean = sum_x.cpu().numpy() / len(y)
    covariance = gram.cpu().numpy() / len(y) - np.outer(mean, mean)
    scale = np.sqrt(np.maximum(np.diag(covariance), 1e-16))
    cross = sum_xy.cpu().numpy() / len(y) - mean * y.mean()
    return {"mean": mean, "scale": scale, "covariance_standardized": covariance / np.outer(scale, scale),
            "cross_standardized": cross / scale, "label_mean": float(y.mean()),
            "selected": selected, "counts": counts[selected], "label_totals": totals[selected], "rows": len(y)}


def ridge_parameters(statistics, regularization):
    gram = statistics["covariance_standardized"]
    beta = np.linalg.solve(gram + regularization * np.eye(len(gram)), statistics["cross_standardized"])
    coefficient = beta / statistics["scale"]
    return coefficient, float(statistics["label_mean"] - statistics["mean"] @ coefficient)


def logistic_parameters(features, statistics, regularization):
    x = (np.asarray(features[statistics["selected"]], float) - statistics["mean"]) / statistics["scale"]
    counts, positives = statistics["counts"], statistics["label_totals"]
    if (positives < 0).any() or (positives > counts).any():
        raise ValueError("Logistic sufficient statistics require binary labels")
    n = counts.sum()

    def objective(theta):
        beta, intercept = theta[:-1], theta[-1]
        logits = x @ beta + intercept
        residual = (counts * expit(logits) - positives) / n
        value = float((counts @ np.logaddexp(0., logits) - positives @ logits) / n + regularization * (beta @ beta) / 2)
        gradient = np.r_[x.T @ residual + regularization * beta, residual.sum()]
        return value, gradient

    initial = np.zeros(x.shape[1] + 1)
    positive_rate = np.clip(positives.sum() / n, 1e-6, 1 - 1e-6)
    initial[-1] = np.log(positive_rate / (1 - positive_rate))
    result = minimize(objective, initial, jac=True, method="L-BFGS-B", options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-7})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Protein logistic fit did not converge: {result.message}")
    coefficient = result.x[:-1] / statistics["scale"]
    intercept = float(result.x[-1] - statistics["mean"] @ coefficient)
    return coefficient, intercept, {"objective": float(result.fun), "iterations": int(result.nit),
                                    "maximum_gradient": float(np.max(np.abs(result.jac)))}


def predict(features, indices, coefficient, intercept, batch_size=32768):
    unique, inverse = np.unique(indices, return_inverse=True)
    output = np.empty(len(unique))
    for start in range(0, len(unique), batch_size):
        current = unique[start:start + batch_size]
        output[start:start + len(current)] = np.asarray(features[current], float) @ coefficient + intercept
    if not np.isfinite(output).all():
        raise RuntimeError("Nonfinite linear predictions")
    return output[inverse]


def fit_regime(regime):
    import fcntl
    import torch

    if ROOT != Path("/home/atabey/Prot2Mol"):
        raise RuntimeError("Runtime restricted to the authorized Atum project")
    allowed = subprocess.check_output(["nvidia-smi", "-i", "0,2,5,6", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in allowed or torch.cuda.device_count() != 1:
        raise RuntimeError("Select exactly one authorized GPU UUID for sufficient statistics")
    torch.set_num_threads(4)
    destination = WORK / "tier2/linear" / regime
    destination.mkdir(parents=True, exist_ok=True)
    lock = (destination / "fit.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frames, data_hashes = load_partitions(regime)
    feature_dir = WORK / "tier2/frozen_features"
    feature_manifest = json.loads((feature_dir / "complete.json").read_text())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    selection = RankingSelection(frames["val"])
    summaries = {}
    for side, arm in (("molecule", "molecule_only_ridge"), ("protein", "protein_only_bce")):
        output = destination / arm
        output.mkdir(parents=True, exist_ok=True)
        provenance = {"regime": regime, "arm": arm, "data_sha256": data_hashes,
            "feature_provenance": feature_manifest[side], "regularization_grid": REGULARIZATION,
            "fit_objective": "mean squared pChEMBL error / 2 + lambda * ||standardized coefficients||^2 / 2" if side == "molecule" else "mean binary log-loss + lambda * ||standardized coefficients||^2 / 2",
            "standardization": "training-observation weighted mean and standard deviation only; intercept unpenalized",
            "label_policy": "partition-local pair-median binary labels; original continuous measurements retained",
            "seed_dependence": "deterministic convex fits, no random optimization or feature fitting",
            "git_commit": commit}
        if (output / "complete.json").exists():
            previous = json.loads((output / "complete.json").read_text())
            for key in ("data_sha256", "feature_provenance", "regularization_grid"):
                if previous[key] != provenance[key]:
                    raise RuntimeError("Completed linear baseline provenance changed")
            summaries[arm] = previous
            continue
        features_path = feature_dir / f"{side}.npy"
        entities_path = feature_dir / f"{side}_entities.parquet"
        if sha256(features_path) != feature_manifest[side]["features_sha256"] or sha256(entities_path) != feature_manifest[side]["entities_sha256"]:
            raise RuntimeError("Frozen features or entity mapping changed")
        features = np.load(features_path, mmap_mode="r")
        entity_index = pd.Index(pd.read_parquet(entities_path, columns=["entity_id"]).entity_id)
        identity = "compound_id" if side == "molecule" else "target_chembl_id"
        indices = {name: entity_index.get_indexer(frame[identity]) for name, frame in frames.items()}
        if any((values < 0).any() for values in indices.values()):
            raise RuntimeError("Baseline entity missing its feature row")
        labels = frames["train"].pchembl_value if side == "molecule" else frames["train"].binary_label
        stats = sufficient_statistics(features, indices["train"], labels, device="cuda")
        trials, parameters = [], []
        for regularization in REGULARIZATION:
            if side == "molecule":
                coefficient, intercept = ridge_parameters(stats, regularization)
                fit_info = {}
            else:
                coefficient, intercept, fit_info = logistic_parameters(features, stats, regularization)
            scores = predict(features, indices["val"], coefficient, intercept)
            value = selection(scores)["macro_spearman"] if side == "molecule" else float(roc_auc_score(frames["val"].binary_label, scores))
            if value is None or not np.isfinite(value):
                raise RuntimeError("Undefined validation selection criterion")
            trials.append({"lambda": regularization, "validation_score": value, "fit": fit_info})
            parameters.append((coefficient, intercept))
            write_json(output / "trials.json", {"selection_metric": "original-eligible macro Spearman" if side == "molecule" else "BCE ROC AUC",
                                                 "partition": "validation only", "trials": trials})
            print(f"{regime}/{arm} lambda={regularization:g} validation={value:.6f}", flush=True)
        winner = max(range(len(trials)), key=lambda i: trials[i]["validation_score"])
        coefficient, intercept = parameters[winner]
        np.savez(output / "model.npz", coefficient=coefficient, intercept=intercept,
                 training_mean=stats["mean"], training_scale=stats["scale"])
        metrics = {}
        for partition in ("val", "test"):
            scores = predict(features, indices[partition], coefficient, intercept)
            predictions = frames[partition].copy()
            predictions["ranking_score"] = scores
            predictions["classification_score"] = scores
            metrics[partition] = evaluate_predictions(predictions, regime, output / partition,
                                                       classification_is_logit=side == "protein")
        result = {**provenance, "selected_lambda": trials[winner]["lambda"], "trials": trials,
                  "training_observations": stats["rows"], "training_unique_entities": len(stats["selected"]),
                  "model_sha256": sha256(output / "model.npz"), "metrics": metrics}
        write_json(output / "complete.json", result)
        summaries[arm] = result
    write_json(destination / "complete.json", summaries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=["A", "B"], required=True)
    fit_regime(parser.parse_args().regime)
