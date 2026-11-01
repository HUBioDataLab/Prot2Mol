"""Regularized low-rank bilinear inductive completion on measured pairs only.

f(x,y)=x.T A B.T y + b, using the same frozen feature map as Kronecker ridge.
This tests the inductive-completion model family, with deterministic spectral
initialization and full-batch L-BFGS. It does not claim the alternating-minimum
recovery theorem applies to noisy, observationally sampled assay measurements.
Model reference: https://arxiv.org/html/1306.0626v1 .
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from .baseline_common import RankingSelection, evaluate_predictions, load_partitions
    from .kronecker_ridge import ObservedKronecker, load_feature_maps
else:
    from run import ROOT, WORK, sha256, write_json
    from baseline_common import RankingSelection, evaluate_predictions, load_partitions
    from kronecker_ridge import ObservedKronecker, load_feature_maps

RANKS, STRENGTHS = [8, 32, 128], [1e-6, 1e-4, .01]
MAX_ITERATIONS = 3000


def factor_objective(problem, left, right, strength):
    coefficient = left @ right.T
    applied = problem.operator(coefficient, 0.)
    gradient = applied - problem.cross
    value = .5 * (coefficient * applied).sum() - (coefficient * problem.cross).sum()
    value += .5 * strength * ((left * left).sum() + (right * right).sum())
    return value, gradient @ right + strength * left, gradient.T @ left + strength * right


def fit_factors(problem, rank, strength, *, max_iterations=MAX_ITERATIONS, tolerance=1e-7, progress=None):
    if strength <= 0 or not 0 < rank <= min(problem.cross.shape):
        raise ValueError("Positive rank and regularization required")
    u, singular, vh = torch.linalg.svd(problem.cross, full_matrices=False)
    root = singular[:rank].clamp_min(1e-16).sqrt()
    left = torch.nn.Parameter(u[:, :rank] * root)
    right = torch.nn.Parameter(vh[:rank].T * root)
    optimizer = torch.optim.LBFGS([left, right], lr=1., max_iter=max_iterations,
        max_eval=2 * max_iterations, tolerance_grad=tolerance, tolerance_change=1e-12,
        history_size=50, line_search_fn="strong_wolfe")
    calls, started = 0, time.monotonic()

    @torch.no_grad()
    def closure():
        nonlocal calls
        value, grad_left, grad_right = factor_objective(problem, left, right, strength)
        if not torch.isfinite(value) or not torch.isfinite(grad_left).all() or not torch.isfinite(grad_right).all():
            raise RuntimeError("Nonfinite low-rank objective or gradient")
        left.grad, right.grad = grad_left, grad_right
        calls += 1
        if progress and calls % 25 == 0:
            progress({"function_evaluations": calls, "centered_objective_without_label_constant": float(value),
                      "maximum_gradient": max(float(grad_left.abs().max()), float(grad_right.abs().max())),
                      "elapsed_seconds": time.monotonic() - started})
        return value

    initial = float(closure())
    optimizer.step(closure)
    final = float(closure())
    with torch.no_grad():
        coefficient = left @ right.T
        intercept = problem.label_mean - float((problem.mean * coefficient).sum())
    state = optimizer.state[left]
    maximum_gradient = max(float(left.grad.abs().max()), float(right.grad.abs().max()))
    if final > initial + 1e-8:
        raise RuntimeError("Low-rank fit increased the training objective")
    return left.detach(), right.detach(), intercept, {
        "iterations": int(state.get("n_iter", 0)), "function_evaluations": calls,
        "initial_centered_objective": initial, "final_centered_objective": final,
        "maximum_gradient": maximum_gradient, "gradient_tolerance_met": maximum_gradient <= tolerance,
        "max_iterations": max_iterations, "elapsed_seconds": time.monotonic() - started,
        "convergence_limit": "A nonconvex finite-budget fit; stationarity does not establish a global minimum"}


def main(regime):
    import fcntl
    if ROOT != Path("/home/atabey/Prot2Mol"):
        raise RuntimeError("Runtime restricted to the authorized Atum project")
    allowed = subprocess.check_output(["nvidia-smi", "-i", "0,2,5,6", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in allowed or torch.cuda.device_count() != 1:
        raise RuntimeError("Select exactly one authorized GPU UUID")
    torch.set_num_threads(4)
    # Preserve the initial 300-iteration diagnostic fits. Their gradients and
    # still-improving training objectives motivated this equal-budget extension.
    output = WORK / f"tier2/bilinear_completion_{MAX_ITERATIONS}iter" / regime
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "fit.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frames, hashes = load_partitions(regime)
    maps, indices, manifest, scalers = load_feature_maps(regime, frames, hashes)
    provenance = {"regime": regime, "data_sha256": hashes, "feature_provenance": manifest,
                  "ranks": RANKS, "regularization_grid": STRENGTHS, "max_iterations": MAX_ITERATIONS}
    if (output / "complete.json").exists():
        previous = json.loads((output / "complete.json").read_text())
        if any(previous[key] != value for key, value in provenance.items()):
            raise RuntimeError("Completed bilinear fit provenance changed")
        return
    print(f"Bilinear {regime}: initializing observed-pair moments", flush=True)
    problem = ObservedKronecker(maps["protein"], maps["molecule"], indices["protein"]["train"],
                                indices["molecule"]["train"], frames["train"].pchembl_value, device="cuda")
    del maps
    selector, trials, best = RankingSelection(frames["val"]), [], None
    for rank in RANKS:
        for strength in STRENGTHS:
            trial_dir = output / f"rank{rank}_lambda{strength:g}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            if (trial_dir / "complete.json").exists():
                trial = json.loads((trial_dir / "complete.json").read_text())
                if trial["rank"] != rank or trial["lambda"] != strength or sha256(trial_dir / "model.npz") != trial["model_sha256"]:
                    raise RuntimeError("Completed bilinear trial changed")
            else:
                def progress(state):
                    write_json(output / "fit_progress.json", {"rank": rank, "lambda": strength, **state})
                    print(f"Bilinear {regime} rank={rank} lambda={strength:g}: evaluations={state['function_evaluations']} objective={state['centered_objective_without_label_constant']:.6f}", flush=True)
                left, right, intercept, fit = fit_factors(problem, rank, strength, progress=progress)
                coefficient = left @ right.T
                scores = problem.predict(coefficient, intercept, indices["protein"]["val"], indices["molecule"]["val"])
                metric = selector(scores)
                np.savez(trial_dir / "model.npz", left=left.cpu().numpy(), right=right.cpu().numpy(),
                         coefficient=coefficient.cpu().numpy(), intercept=intercept,
                         **{f"{side}_{key}": value for side, stats in scalers.items() for key, value in stats.items()})
                trial = {"rank": rank, "lambda": strength, **metric, "fit": fit, "model_sha256": sha256(trial_dir / "model.npz")}
                write_json(trial_dir / "complete.json", trial)
            trials.append(trial)
            score = trial["macro_spearman"]
            if score is not None and (best is None or score > best["macro_spearman"]):
                best = trial.copy()
            write_json(output / "trials.json", {"trials": trials, "best": best})
            print(f"Bilinear {regime} rank={rank} lambda={strength:g}: val rho={score}", flush=True)
    if best is None:
        raise RuntimeError("No low-rank fit produced defined validation rankings")
    import shutil
    shutil.copyfile(output / f"rank{best['rank']}_lambda{best['lambda']:g}/model.npz", output / "model.npz")
    model = np.load(output / "model.npz")
    # Preserve the exact matrix used for selection. CPU versus GPU factor
    # multiplication can perturb near-tied rankings despite identical factors.
    coefficient = torch.as_tensor(model["coefficient"], device="cuda")
    metrics = {}
    for part in ("val", "test"):
        scores = problem.predict(coefficient, float(model["intercept"]), indices["protein"][part], indices["molecule"][part])
        if part == "val" and abs(selector(scores)["macro_spearman"] - best["macro_spearman"]) > 1e-12:
            raise RuntimeError("Saved bilinear model did not reproduce validation selection")
        metrics[part] = evaluate_predictions(frames[part].assign(ranking_score=scores, classification_score=scores), regime, output / part)
    write_json(output / "complete.json", {**provenance, "trials": trials, "selected": best, "metrics": metrics,
        "model_sha256": sha256(output / "model.npz"),
        "model": "x.T A B.T y + unpenalized global intercept, same scaled standardized augmented linear features as Kronecker ridge",
        "objective": "mean squared error on measured observations / 2 + lambda * (||A||_F^2 + ||B||_F^2) / 2; no unobserved labels",
        "optimizer": "full-batch L-BFGS with strong-Wolfe line search, 3000 iterations / 6000 evaluations maximum per setting, history 50; deterministic SVD of training feature-label cross-moment initialization",
        "budget_extension_reason": "Initial 300-iteration fits retained under tier2/bilinear_completion; low-regularization training objectives and gradients had not converged. All nine settings on both splits receive the same extended budget, chosen from training diagnostics.",
        "selection": "nine rank/regularization settings per split; validation eligible macro Spearman only; test after selection",
        "interpretation": "Regularized inductive-completion model-family baseline, not a reproduction of exact-recovery guarantees or the source paper's alternating-minimization algorithm",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=["A", "B"], required=True)
    main(parser.parse_args().regime)
