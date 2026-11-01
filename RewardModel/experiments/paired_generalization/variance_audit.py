"""Scalable, nonnegative variance-component fitting on recovered measurements.

Method: centered restricted Haseman-Elston covariance regression, following
https://pmc.ncbi.nlm.nih.gov/articles/PMC8604792/ (REHE, Yue et al., 2021).
The trace identities for categorical incidence matrices are evaluated exactly,
without materializing an observation-by-observation covariance matrix.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import nnls

if __package__:
    from .run import DATA, ROOT, WORK, write_json
else:
    from run import DATA, ROOT, WORK, write_json


def covariance_moments(values, labels, observation_weights=None):
    """Covariance regression after optional W transform and intercept projection.

    W=diag(sqrt(observation_weights)); P removes W*1. This preserves the
    random-effect variance interpretation while changing covariance-fit weights.
    """
    y = np.asarray(values, dtype=float)
    if not np.isfinite(y).all() or len(y) < 2:
        raise ValueError("Finite replicated observations are required")
    n = len(y)
    weights = np.ones(n) if observation_weights is None else np.asarray(observation_weights, dtype=float)
    if weights.shape != (n,) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Positive finite observation weights must align with values")
    mass = weights.sum()
    y = y - np.average(y, weights=weights)
    names = list(labels) + ["measurement_noise"]
    codes, counts, row_counts, sums, rhs = [], [], [], [], []
    for name in labels:
        code, _ = pd.factorize(labels[name], sort=False)
        if len(code) != n or (code < 0).any():
            raise ValueError("Random-effect labels must align and contain no missing values")
        count = np.bincount(code, weights=weights)
        total = np.bincount(code, weights=weights * y)
        codes.append(code)
        counts.append(count)
        row_counts.append(count[code])
        sums.append(float(count @ count))
        rhs.append(float(total @ total))
    d = len(codes)
    gram = np.zeros((d + 1, d + 1))
    for i in range(d):
        for j in range(i, d):
            key = codes[i].astype(np.int64) * len(counts[j]) + codes[j]
            _, inverse = np.unique(key, return_inverse=True)
            intersections = np.bincount(inverse, weights=weights)
            trace = float(intersections @ intersections)
            centered_trace = trace - 2 * float((weights * row_counts[i]) @ row_counts[j]) / mass + sums[i] * sums[j] / mass**2
            gram[i, j] = gram[j, i] = centered_trace
        squared_weights = weights**2
        gram[i, d] = gram[d, i] = float(squared_weights.sum() - 2 * squared_weights @ row_counts[i] / mass + sums[i] * squared_weights.sum() / mass**2)
    gram[d, d] = float((weights**2).sum() - 2 * (weights**3).sum() / mass + (weights**2).sum()**2 / mass**2)
    rhs.append(float((weights * y) @ (weights * y)))
    return names, gram, np.asarray(rhs)


def fit_components(values, labels, observation_weights=None):
    names, gram, rhs = covariance_moments(values, labels, observation_weights)
    scale = np.sqrt(np.maximum(np.diag(gram), 0.))
    present = scale > 1e-10
    standardized = gram[np.ix_(present, present)] / np.outer(scale[present], scale[present])
    response = rhs[present] / scale[present]
    eigenvalues, eigenvectors = np.linalg.eigh(standardized)
    keep = eigenvalues > max(1., eigenvalues.max()) * 1e-10
    if eigenvalues.min() < -1e-8:
        raise RuntimeError("Covariance design Gram matrix is not positive semidefinite")
    factor = np.sqrt(eigenvalues[keep])[:, None] * eigenvectors[:, keep].T
    target = (eigenvectors[:, keep].T @ response) / np.sqrt(eigenvalues[keep])
    fitted, _ = nnls(factor, target, maxiter=10000)
    unconstrained = np.linalg.pinv(standardized, rcond=1e-10) @ response
    candidate = np.zeros(len(names))
    candidate[present] = fitted / scale[present]
    unconstrained_full = np.zeros(len(names))
    unconstrained_full[present] = unconstrained / scale[present]
    rank = int(keep.sum())
    identifiable = rank == len(names)
    return {
        "observations": len(values), "components": names, "design_rank": rank,
        "identifiable": identifiable,
        "variance_components": dict(zip(names, candidate.tolist())) if identifiable else None,
        "nonnegative_candidate": dict(zip(names, candidate.tolist())),
        "unconstrained_candidate": dict(zip(names, unconstrained_full.tolist())),
        "normalized_design_eigenvalues": eigenvalues.tolist(),
        "normalized_design_condition": float(eigenvalues[keep].max() / eigenvalues[keep].min()),
        "covariance_design_gram": gram.tolist(), "covariance_response": rhs.tolist(),
        "interpretation": "descriptive random-effects covariance fit on raw measurements; non-identifiable components are not reported as unique variance estimates; no causal or Spearman-ceiling claim",
    }


def fit_diagnostics(subset, result):
    grouped = subset.groupby("observation_id").pchembl_value.agg(["size", "var"])
    degrees = int((grouped["size"] - 1).sum())
    direct_noise = float((grouped["var"].fillna(0.) * (grouped["size"] - 1)).sum() / degrees) if degrees else None
    components = result["variance_components"]
    predicted = None
    if components is not None:
        # The last covariance-design column is tr(P Ki), including residual I.
        traces = np.asarray(result["covariance_design_gram"])[:, -1]
        predicted = float(np.array([components[name] for name in result["components"]]) @ traces / len(subset))
    return {
        "observed_centered_variance": float(np.var(subset.pchembl_value)),
        "fitted_expected_centered_variance": predicted,
        "direct_within_observation_noise_variance": direct_noise,
        "replicate_contrast_degrees_of_freedom": degrees,
        "boundary_components": [name for name, value in (components or {}).items() if value <= 1e-12],
        "interpretation": "Boundary estimates do not establish absent biology or noiseless measurements. Covariance least squares can fit large shared groups at the expense of diagonal/replicate contrasts. Compare the fitted marginal variance and noise against these directly observed quantities before attributing variance or deriving a ceiling.",
    }


def main():
    base = WORK / "tier1"
    destination = base / "variance"
    destination.mkdir(parents=True, exist_ok=True)
    observations = pd.read_parquet(base / "replicates/observations.parquet")
    if not observations.verified.all():
        raise RuntimeError("Raw measurement reconstruction must be verified first")
    raw = pd.read_parquet(base / "replicates/raw_measurements.parquet")
    molecules = pd.read_parquet(base / "data/molecule_scaffolds.parquet").set_index("compound_id").scaffold
    raw["scaffold"] = raw.compound_id.map(molecules)
    if raw.scaffold.isna().any():
        raise RuntimeError("Raw measurement lacks a scaffold label")
    for regime, directory in (("A", ROOT / "dataset/processed/chembl_37/target_aware_random_assay_activity_balanced"),
                               ("B", DATA / "scaffold_seed42")):
        for partition in ("train", "val", "test"):
            path = destination / f"{regime}_{partition}.json"
            assays = pd.read_parquet(directory / f"{partition}.parquet", columns=["assay_id"]).assay_id.unique()
            subset = raw[raw.assay_id.isin(assays)].copy()
            print(f"Variance components: {regime}/{partition}, {len(subset)} raw measurements", flush=True)
            labels = {"target": subset.target_chembl_id,
                      "scaffold_main": subset.scaffold,
                      "target_by_scaffold": subset.target_chembl_id + "__" + subset.scaffold,
                      "assay": subset.assay_id,
                      "observation_residual": subset.observation_id}
            result = json.loads(path.read_text()) if path.exists() else fit_components(subset.pchembl_value, labels)
            result["fit_diagnostics"] = fit_diagnostics(subset, result)
            if "assay_balanced_fit" not in result:
                counts = subset.groupby("assay_id").assay_id.transform("size").to_numpy()
                weights = 1. / counts
                weights *= len(weights) / weights.sum()
                balanced = fit_components(subset.pchembl_value, labels, weights)
                balanced["weighting"] = "W_ii=sqrt(1/raw_measurements_in_assay), up to a common constant; reduces giant-assay dominance in the covariance objective"
                # Evaluate the alternative fit against raw, unweighted marginal variance.
                balanced["fit_diagnostics"] = fit_diagnostics(subset, {
                    **balanced, "covariance_design_gram": result["covariance_design_gram"]})
                result["assay_balanced_fit"] = balanced
            unique = subset.drop_duplicates("observation_id")
            assay_size = unique.groupby("assay_id").size()
            cluster_count = unique.groupby(["assay_id", "scaffold"]).size()
            cluster_square = (cluster_count**2).groupby(level=0).sum()
            heterogeneity = 1 - cluster_square / assay_size**2
            nonconstant_factor = 1 - 1 / assay_size
            variance = result["variance_components"]
            result["within_assay_latent_contributions"] = (
                {"scaffold_main": variance["scaffold_main"] * float(heterogeneity.mean()),
                 "target_by_scaffold": variance["target_by_scaffold"] * float(heterogeneity.mean()),
                 "observation_residual": variance["observation_residual"] * float(nonconstant_factor.mean()),
                 "target_and_assay_effects": 0.,
                 "note": "equal-assay average for unique observations; target and assay intercepts cancel. Raw measurement noise is not converted into median noise. Observation residual includes unmodeled molecule-specific signal and cannot be called irreducible noise."}
                if variance is not None else None)
            balanced_variance = result["assay_balanced_fit"]["variance_components"]
            if balanced_variance is not None:
                result["assay_balanced_fit"]["within_assay_latent_contributions"] = {
                    "scaffold_main": balanced_variance["scaffold_main"] * float(heterogeneity.mean()),
                    "target_by_scaffold": balanced_variance["target_by_scaffold"] * float(heterogeneity.mean()),
                    "observation_residual": balanced_variance["observation_residual"] * float(nonconstant_factor.mean()),
                    "target_and_assay_effects": 0.,
                    "note": "same unique-observation, equal-assay estimand as the primary fit; measurement noise is not transformed into median uncertainty"}
            result["assays_with_one_scaffold_fraction"] = float((unique.groupby("assay_id").scaffold.nunique() == 1).mean())
            result["effect_levels"] = {key: int(pd.Series(value).nunique()) for key, value in labels.items()}
            result["method_source"] = "https://pmc.ncbi.nlm.nih.gov/articles/PMC8604792/"
            write_json(path, result)
    write_json(destination / "complete.json", {"fits": 6, "scale": "raw pChEMBL squared", "assumptions": "additive independent random effects with constant component variances; observational curation and assay heterogeneity can violate these assumptions"})


if __name__ == "__main__":
    main()
