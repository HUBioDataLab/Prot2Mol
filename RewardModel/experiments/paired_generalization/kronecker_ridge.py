"""Observed-pair Kronecker ridge, using exact linear feature maps.

The product kernel is (1 + x.x'/dx)(1 + y.y'/dy), with training-only
standardized frozen features. Matrix-free primal conjugate gradients avoid
both an observation-sized kernel and a zero-filled interaction matrix.
The intercept is unpenalized. No random-feature approximation is necessary
for this finite linear kernel. See https://arxiv.org/html/1803.01575v1 .
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import torch

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from .baseline_common import RankingSelection, evaluate_predictions, load_partitions
    from .linear_baselines import REGULARIZATION
else:
    from run import ROOT, WORK, sha256, write_json
    from baseline_common import RankingSelection, evaluate_predictions, load_partitions
    from linear_baselines import REGULARIZATION


def shuffle_proteins_by_group(frame, original_indices, seed):
    """One reassigned protein per whole assay, preserving group-level frequencies."""
    codes, groups = pd.factorize(frame.group_id, sort=True)
    source = np.full(len(groups), -1, dtype=np.int64)
    for indices in frame.groupby("group_id", sort=True).indices.values():
        values = np.unique(original_indices[indices])
        if len(values) != 1:
            raise ValueError("Each assay must contain exactly one protein")
        source[codes[indices[0]]] = values[0]
    shuffled = np.random.default_rng(seed).permutation(source)
    return shuffled[codes], float(np.mean(shuffled == source))


class ObservedKronecker:
    def __init__(self, protein, molecule, protein_indices, molecule_indices, labels,
                 *, device="cpu", batch_size=32768):
        self.device, self.batch_size = device, batch_size
        self.x = torch.as_tensor(np.array(protein), dtype=torch.float64, device=device)
        self.y = torch.as_tensor(np.array(molecule), dtype=torch.float64, device=device)
        p, m, labels = np.asarray(protein_indices), np.asarray(molecule_indices), np.asarray(labels, float)
        if p.shape != m.shape or p.shape != labels.shape or not np.isfinite(labels).all():
            raise ValueError("Finite aligned observed labels required")
        if (p < 0).any() or (p >= len(protein)).any() or (m < 0).any() or (m >= len(molecule)).any():
            raise ValueError("Invalid observed entity index")
        # Exact compression of repeated measured pairs; no other pairs enter.
        keys, inverse = np.unique(p.astype(np.int64) * len(molecule) + m, return_inverse=True)
        count = np.bincount(inverse).astype(float)
        total = np.bincount(inverse, weights=labels)
        self.p = torch.as_tensor(keys // len(molecule), device=device)
        self.m = torch.as_tensor(keys % len(molecule), device=device)
        self.count = torch.as_tensor(count, dtype=torch.float64, device=device)
        self.total = torch.as_tensor(total, dtype=torch.float64, device=device)
        self.n, self.label_mean = len(labels), float(labels.mean())
        self.mean = self.adjoint(self.count) / self.n
        self.cross = self.adjoint(self.total) / self.n - self.mean * self.label_mean
        # Product of marginal second moments is an SPD preconditioner only.
        # The actual operator always uses the joint observed-pair distribution.
        cx = np.bincount(p, minlength=len(protein)).astype(float)
        cy = np.bincount(m, minlength=len(molecule)).astype(float)
        self.ex, self.ux = self.marginal_eigen(self.x, cx)
        self.ey, self.uy = self.marginal_eigen(self.y, cy)

    def marginal_eigen(self, features, counts):
        gram = torch.zeros((features.shape[1], features.shape[1]), dtype=torch.float64, device=self.device)
        weights = torch.as_tensor(counts, dtype=torch.float64, device=self.device)
        for start in range(0, len(features), self.batch_size):
            z = features[start:start + self.batch_size]
            gram += z.T @ (z * weights[start:start + len(z), None])
        values, vectors = torch.linalg.eigh(gram / self.n)
        if float(values.min()) < -1e-8:
            raise RuntimeError("Marginal feature covariance is not positive semidefinite")
        return values.clamp_min(0), vectors

    def adjoint(self, values):
        grouped = torch.zeros((len(self.x), self.y.shape[1]), dtype=torch.float64, device=self.device)
        for start in range(0, len(self.p), self.batch_size):
            sl = slice(start, start + self.batch_size)
            grouped.index_add_(0, self.p[sl], self.y[self.m[sl]] * values[sl, None])
        return self.x.T @ grouped

    def operator(self, coefficient, strength):
        projected = self.x @ coefficient
        values = torch.empty_like(self.count)
        for start in range(0, len(self.p), self.batch_size):
            sl = slice(start, start + self.batch_size)
            values[sl] = (projected[self.p[sl]] * self.y[self.m[sl]]).sum(1) * self.count[sl]
        return self.adjoint(values) / self.n - self.mean * (self.mean * coefficient).sum() + strength * coefficient

    def solve(self, strength, *, initial=None, tolerance=1e-7, max_iterations=2000, progress=None):
        if strength <= 0:
            raise ValueError("Positive ridge regularization required")
        denominator = self.ex[:, None] * self.ey[None, :] + strength

        def precondition(value):
            return self.ux @ ((self.ux.T @ value @ self.uy) / denominator) @ self.uy.T

        coefficient = torch.zeros_like(self.cross) if initial is None else initial.clone()
        residual = self.cross - self.operator(coefficient, strength)
        rhs_norm = max(float(torch.linalg.vector_norm(self.cross)), 1e-30)
        relative = float(torch.linalg.vector_norm(residual)) / rhs_norm
        z = precondition(residual)
        direction, rz = z.clone(), (residual * z).sum()
        iterations = 0
        started = time.monotonic()
        while relative > tolerance and iterations < max_iterations:
            applied = self.operator(direction, strength)
            curvature = (direction * applied).sum()
            if not torch.isfinite(curvature) or float(curvature) <= 0:
                raise RuntimeError("Kronecker conjugate-gradient curvature lost positivity")
            step = rz / curvature
            coefficient += step * direction
            residual -= step * applied
            iterations += 1
            relative = float(torch.linalg.vector_norm(residual)) / rhs_norm
            if iterations % 20 == 0 or relative <= tolerance:
                if progress:
                    progress({"iterations": iterations, "relative_residual": relative,
                              "elapsed_seconds": time.monotonic() - started})
            if relative <= tolerance:
                break
            z = precondition(residual)
            next_rz = (residual * z).sum()
            direction = z + (next_rz / rz) * direction
            rz = next_rz
        actual = float(torch.linalg.vector_norm(self.cross - self.operator(coefficient, strength))) / rhs_norm
        if actual > max(2 * tolerance, 1e-12) or not torch.isfinite(coefficient).all():
            raise RuntimeError(f"Kronecker solve did not converge: iterations={iterations}, residual={actual}")
        intercept = self.label_mean - float((self.mean * coefficient).sum())
        return coefficient, intercept, {"iterations": iterations, "relative_residual": actual,
                                       "elapsed_seconds": time.monotonic() - started}

    def predict(self, coefficient, intercept, protein_indices, molecule_indices):
        projected = self.x @ coefficient
        p = torch.as_tensor(protein_indices, device=self.device)
        m = torch.as_tensor(molecule_indices, device=self.device)
        output = np.empty(len(p))
        for start in range(0, len(p), self.batch_size):
            sl = slice(start, start + self.batch_size)
            output[sl] = ((projected[p[sl]] * self.y[m[sl]]).sum(1) + intercept).cpu().numpy()
        return output


def load_feature_maps(regime, frames, hashes):
    base = WORK / "tier2/frozen_features"
    manifest = json.loads((base / "complete.json").read_text())
    maps, indices, standardizers = {}, {}, {}
    for side, column, arm in (("protein", "target_chembl_id", "protein_only_bce"),
                              ("molecule", "compound_id", "molecule_only_ridge")):
        path, entities = base / f"{side}.npy", base / f"{side}_entities.parquet"
        if sha256(path) != manifest[side]["features_sha256"] or sha256(entities) != manifest[side]["entities_sha256"]:
            raise RuntimeError("Frozen feature or entity mapping checksum changed")
        reference = WORK / "tier2/linear" / regime / arm
        record = json.loads((reference / "complete.json").read_text())
        if record["data_sha256"] != hashes or record["feature_provenance"] != manifest[side]:
            raise RuntimeError("Training-only feature standardizer provenance mismatch")
        model = np.load(reference / "model.npz")
        mean, scale = model["training_mean"], model["training_scale"]
        raw = np.load(path, mmap_mode="r")
        transformed = np.empty((len(raw), raw.shape[1] + 1), dtype=np.float64)
        for start in range(0, len(raw), 32768):
            transformed[start:start + 32768, :-1] = (raw[start:start + 32768] - mean) / scale / np.sqrt(raw.shape[1])
        transformed[:, -1] = 1.
        maps[side] = transformed
        index = pd.Index(pd.read_parquet(entities, columns=["entity_id"]).entity_id)
        indices[side] = {part: index.get_indexer(frame[column]) for part, frame in frames.items()}
        if any((x < 0).any() for x in indices[side].values()):
            raise RuntimeError("Entity feature mapping incomplete")
        standardizers[side] = {"mean": mean, "scale": scale}
    return maps, indices, manifest, standardizers


def main(regime, shuffled=False):
    import fcntl
    if ROOT != Path("/home/atabey/Prot2Mol"):
        raise RuntimeError("Runtime restricted to the authorized Atum project")
    allowed = subprocess.check_output(["nvidia-smi", "-i", "0,2,5,6", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in allowed or torch.cuda.device_count() != 1:
        raise RuntimeError("Select exactly one authorized GPU UUID")
    torch.set_num_threads(4)
    arm = "protein_shuffled_kronecker" if shuffled else "kronecker_ridge"
    output = WORK / "tier2" / arm / regime
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "fit.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frames, hashes = load_partitions(regime)
    if (output / "complete.json").exists():
        previous = json.loads((output / "complete.json").read_text())
        if previous["data_sha256"] != hashes or previous["regularization_grid"] != REGULARIZATION:
            raise RuntimeError("Completed Kronecker provenance changed")
        return
    print(f"{arm} {regime}: verifying and standardizing frozen features", flush=True)
    maps, indices, manifest, standardizers = load_feature_maps(regime, frames, hashes)
    shuffle = {}
    if shuffled:
        for j, (part, frame) in enumerate(frames.items()):
            indices["protein"][part], unchanged = shuffle_proteins_by_group(frame, indices["protein"][part], 42 + j)
            shuffle[part] = {"seed": 42 + j, "unchanged_group_fraction": unchanged}
            pd.DataFrame({"group_id": frame.group_id, "assigned_protein_feature_row": indices["protein"][part]}).drop_duplicates().to_parquet(output / f"{part}_protein_assignment.parquet", index=False)
    print(f"{arm} {regime}: computing observed-pair moments and marginal preconditioner", flush=True)
    problem = ObservedKronecker(maps["protein"], maps["molecule"], indices["protein"]["train"],
                                indices["molecule"]["train"], frames["train"].pchembl_value, device="cuda")
    del maps
    selector, trials, best, coefficient = RankingSelection(frames["val"]), [], None, None
    for strength in reversed(REGULARIZATION):
        def progress(state):
            write_json(output / "fit_progress.json", {"lambda": strength, **state})
            print(f"{arm} {regime} lambda={strength:g} CG={state['iterations']} residual={state['relative_residual']:.3g}", flush=True)
        coefficient, intercept, fit = problem.solve(strength, initial=coefficient, progress=progress)
        scores = problem.predict(coefficient, intercept, indices["protein"]["val"], indices["molecule"]["val"])
        metric = selector(scores)
        if metric["macro_spearman"] is None:
            raise RuntimeError("Undefined Kronecker validation metric")
        trial = {"lambda": strength, **metric, "solver": fit}
        trials.append(trial)
        if best is None or metric["macro_spearman"] > best["macro_spearman"]:
            best = trial.copy()
            np.savez(output / "model.npz", coefficient=coefficient.cpu().numpy(), intercept=intercept,
                     **{f"{side}_{key}": value for side, stats in standardizers.items() for key, value in stats.items()})
        write_json(output / "trials.json", {"validation_only": True, "trials": trials, "best": best})
        print(f"{arm} {regime} lambda={strength:g}: val rho={metric['macro_spearman']:.6f}", flush=True)
    selected = np.load(output / "model.npz")
    chosen = torch.as_tensor(selected["coefficient"], device="cuda")
    metrics = {}
    for part in ("val", "test"):
        scores = problem.predict(chosen, float(selected["intercept"]), indices["protein"][part], indices["molecule"][part])
        metrics[part] = evaluate_predictions(frames[part].assign(ranking_score=scores, classification_score=scores), regime, output / part)
    write_json(output / "complete.json", {"regime": regime, "arm": arm, "data_sha256": hashes,
        "feature_provenance": manifest, "regularization_grid": REGULARIZATION, "selected": best, "trials": trials,
        "metrics": metrics, "model_sha256": sha256(output / "model.npz"), "shuffle": shuffle,
        "kernel": "(1 + standardized_x_dot_x / 480) * (1 + standardized_y_dot_y / 768)",
        "fit": "observed-measurement mean squared error / 2 + lambda * squared coefficient norm / 2; unpenalized global intercept; exact duplicate-pair frequencies; matrix-free primal conjugate gradients",
        "approximation": "No feature or label approximation; finite linear product kernel, solver true relative residual <=2e-7",
        "shuffle_protocol": "none" if not shuffled else "protein assignments permuted over whole assays independently within train, validation and test; group-level protein frequencies preserved, molecule inputs and labels unchanged; same feature map and five-lambda budget as unshuffled model",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=["A", "B"], required=True)
    parser.add_argument("--shuffled", action="store_true")
    args = parser.parse_args()
    main(args.regime, args.shuffled)
