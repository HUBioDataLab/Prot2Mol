"""Observed-only smoother-kernel version of the linear matrix filter.

Kernel: (1 + identity(protein)) * (1 + identity(molecule)). On complete
matrices this is the linear-filter family in Stock et al., Theorem 3.4:
https://arxiv.org/html/1803.01575v1 . Here ridge loss uses measured pairs only.
This reference is feature-free; its cold-molecule output is necessarily a
protein-specific constant. No artificial zero labels complete the matrix.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy.sparse.linalg import LinearOperator, cg

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from .baseline_common import RankingSelection, evaluate_predictions, load_partitions
else:
    from run import ROOT, WORK, sha256, write_json
    from baseline_common import RankingSelection, evaluate_predictions, load_partitions

RIDGE_SUM_LOSS = [.01, .1, 1., 10., 100.]


def observed_pairs(targets, molecules, labels):
    target_values, target_codes = np.unique(np.asarray(targets), return_inverse=True)
    molecule_values, molecule_codes = np.unique(np.asarray(molecules), return_inverse=True)
    keys = target_codes.astype(np.int64) * len(molecule_values) + molecule_codes
    unique, inverse = np.unique(keys, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    means = np.bincount(inverse, weights=np.asarray(labels, float)) / counts
    if not np.isfinite(means).all():
        raise ValueError("Finite observed labels required")
    return {"target_values": target_values, "molecule_values": molecule_values, "pair_keys": unique,
            "target": unique // len(molecule_values), "molecule": unique % len(molecule_values),
            "counts": counts, "means": means}


def fit_smoother(pairs, alpha):
    if alpha <= 0:
        raise ValueError("Positive ridge regularization required")
    target, molecule = pairs["target"], pairs["molecule"]
    nt, nm, n = len(pairs["target_values"]), len(pairs["molecule_values"]), len(target)
    diagonal = 1. + alpha / pairs["counts"]  # Pair-identity kernel plus repeat-weighted ridge.

    def matvec(vector):
        return diagonal * vector + vector.sum() + np.bincount(target, weights=vector, minlength=nt)[target] + np.bincount(molecule, weights=vector, minlength=nm)[molecule]

    inverse_diagonal = 1. / diagonal
    target_mass = 1. + np.bincount(target, weights=inverse_diagonal, minlength=nt)

    def inverse_target_block(vector):
        weighted = inverse_diagonal * vector
        return weighted - inverse_diagonal * (np.bincount(target, weights=weighted, minlength=nt) / target_mass)[target]

    inverse_ones = inverse_target_block(np.ones(n))
    global_denominator = 1. + inverse_ones.sum()

    def precondition(vector):
        partial = inverse_target_block(vector)
        return partial - inverse_ones * (partial.sum() / global_denominator)

    operator = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    preconditioner = LinearOperator((n, n), matvec=precondition, dtype=np.float64)
    iterations = 0

    def callback(_):
        nonlocal iterations
        iterations += 1

    dual, info = cg(operator, pairs["means"], M=preconditioner, rtol=1e-8, atol=0., maxiter=2000, callback=callback)
    residual = float(np.linalg.norm(matvec(dual) - pairs["means"]) / max(np.linalg.norm(pairs["means"]), 1e-30))
    if info != 0 or residual > 2e-8 or not np.isfinite(dual).all():
        raise RuntimeError(f"Smoother kernel did not converge: info={info}, residual={residual}")
    return {"global": float(dual.sum()), "target_effect": np.bincount(target, weights=dual, minlength=nt),
            "molecule_effect": np.bincount(molecule, weights=dual, minlength=nm), "pair_effect": dual,
            "iterations": iterations, "relative_residual": residual}


def predict_smoother(pairs, model, targets, molecules):
    target = pd.Index(pairs["target_values"]).get_indexer(targets)
    molecule = pd.Index(pairs["molecule_values"]).get_indexer(molecules)
    prediction = np.full(len(target), model["global"])
    known_t, known_m = target >= 0, molecule >= 0
    prediction[known_t] += model["target_effect"][target[known_t]]
    prediction[known_m] += model["molecule_effect"][molecule[known_m]]
    known_both = known_t & known_m
    keys = target[known_both].astype(np.int64) * len(pairs["molecule_values"]) + molecule[known_both]
    lookup = pd.Index(pairs["pair_keys"]).get_indexer(keys)
    positions = np.flatnonzero(known_both)[lookup >= 0]
    prediction[positions] += model["pair_effect"][lookup[lookup >= 0]]
    return prediction


def main():
    import fcntl
    if ROOT != Path("/home/atabey/Prot2Mol"):
        raise RuntimeError("Runtime restricted to the authorized Atum project")
    base = WORK / "tier2/linear_matrix_filter"
    base.mkdir(parents=True, exist_ok=True)
    lock = (base / "fit.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    all_results = {}
    for regime in ("A", "B"):
        output = base / regime
        output.mkdir(parents=True, exist_ok=True)
        frames, hashes = load_partitions(regime)
        if (output / "complete.json").exists():
            result = json.loads((output / "complete.json").read_text())
            if result["data_sha256"] != hashes or result["regularization_grid"] != RIDGE_SUM_LOSS:
                raise RuntimeError("Completed matrix-filter provenance changed")
            all_results[regime] = result
            continue
        training = frames["train"]
        pairs = observed_pairs(training.target_chembl_id, training.compound_id, training.pchembl_value)
        selector, trials, best = RankingSelection(frames["val"]), [], None
        for alpha in RIDGE_SUM_LOSS:
            model = fit_smoother(pairs, alpha)
            scores = predict_smoother(pairs, model, frames["val"].target_chembl_id, frames["val"].compound_id)
            metric = selector(scores)
            # In cold-molecule B this kernel cannot order molecules. All settings
            # tie; choose the first declared setting, without replacing undefined rho by zero.
            criterion = metric["macro_spearman"] if metric["macro_spearman"] is not None else -np.inf
            trial = {"alpha_sum_squared_error": alpha, **metric,
                     "solver_iterations": model["iterations"], "solver_relative_residual": model["relative_residual"]}
            trials.append(trial)
            if best is None or criterion > best["criterion"]:
                best = {"criterion": criterion, "trial": trial}
                np.savez(output / "model.npz", **{k: v for k, v in model.items() if k not in ("iterations", "relative_residual")})
            write_json(output / "trials.json", trials)
            print(f"Matrix filter {regime} alpha={alpha:g}: {metric['macro_spearman']}, CG={model['iterations']}", flush=True)
        parameters = dict(np.load(output / "model.npz"))
        metrics = {}
        for partition in ("val", "test"):
            scores = predict_smoother(pairs, parameters, frames[partition].target_chembl_id, frames[partition].compound_id)
            predictions = frames[partition].assign(ranking_score=scores, classification_score=scores)
            metrics[partition] = evaluate_predictions(predictions, regime, output / partition)
        pd.DataFrame({"target_chembl_id": pairs["target_values"]}).to_parquet(output / "targets.parquet", index=False)
        pd.DataFrame({"compound_id": pairs["molecule_values"]}).to_parquet(output / "molecules.parquet", index=False)
        np.save(output / "pair_keys.npy", pairs["pair_keys"])
        result = {"regime": regime, "data_sha256": hashes, "regularization_grid": RIDGE_SUM_LOSS,
            "selected": best["trial"], "trials": trials, "metrics": metrics, "model_sha256": sha256(output / "model.npz"),
            "kernel": "(1+delta_target)*(1+delta_molecule); feature-free smoother kernel with both theta values fixed at 1",
            "objective": "sum squared error on observed measurements plus alpha times RKHS norm; duplicate pairs collapsed with exact observation-count weights",
            "applicability": "Complete-data smoother KRR is a linear matrix filter; this is its observed-only kernel fit. Unseen molecules have no molecule/pair effect and cannot be ranked within a protein.",
            "selection": "validation eligible macro Spearman; if undefined for every setting, select first declared alpha and report inability to rank",
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
        write_json(output / "complete.json", result)
        all_results[regime] = result
    write_json(base / "complete.json", all_results)


if __name__ == "__main__":
    main()
