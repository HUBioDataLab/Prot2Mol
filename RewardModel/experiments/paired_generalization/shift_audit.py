"""Whole-unit held-out C2ST and validation-to-test black-box label-shift estimation.

C2ST: https://arxiv.org/abs/1610.06545
BBSE: https://proceedings.mlr.press/v80/lipton18a.html
Neither diagnostic identifies the causal contribution of shift to ranking error.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__:
    from .run import WORK, write_json
else:
    from run import WORK, write_json

LOG_FEATURES = ["size", "variance", "span", "replicate_mean", "replicate_max", "target_training_groups"]
FEATURES = [f"log1p_{key}" for key in LOG_FEATURES] + ["positive_rate", "label_midrange"]


def descriptor_features(frame):
    values = np.column_stack([np.log1p(frame[name].to_numpy(float)) for name in LOG_FEATURES] +
                             [frame.positive_rate.to_numpy(float), ((frame.low + frame.high) / 2).to_numpy(float)])
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite group descriptor")
    return values


def unit_domain_weights(frame):
    """Each domain gets half the mass, each unit equal mass within its domain."""
    if frame.groupby("unit").domain.nunique().max() != 1:
        raise ValueError("A unit cannot cross domains in this two-sample design")
    unit_rows = frame.groupby("unit").size()
    domain_units = frame.drop_duplicates("unit").groupby("domain").size()
    weights = 1. / frame.unit.map(unit_rows) / frame.domain.map(domain_units)
    return weights.to_numpy() * len(frame) / 2


def unit_test(rates, domains, *, repetitions=1999, seed=42):
    """Permutation test conditional on a classifier fit on disjoint training units.

    Domain labels are permuted over whole held-out units. A unit's statistic is
    the fraction of its assays classified as domain 1, giving equal unit weight.
    """
    rates, domains = np.asarray(rates, float), np.asarray(domains, int)
    left, right = rates[domains == 0], rates[domains == 1]
    if not len(left) or not len(right):
        raise ValueError("Both held-out domains need units")
    observed = .5 + .5 * (right.mean() - left.mean())
    rng = np.random.default_rng(seed)
    extreme, boots = 0, []
    for _ in range(repetitions):
        shuffled = rng.permutation(rates)
        null = .5 + .5 * (shuffled[len(left):].mean() - shuffled[:len(left)].mean())
        extreme += null >= observed - 1e-15
        boots.append(.5 + .5 * (rng.choice(right, len(right)).mean() - rng.choice(left, len(left)).mean()))
    return {"unit_balanced_accuracy": float(observed), "whole_unit_permutation_p_one_sided": (1 + extreme) / (repetitions + 1),
            "unit_bootstrap_interval": np.quantile(boots, [.025, .975]).tolist(), "permutations": repetitions,
            "seed": seed, "heldout_units_per_domain": [len(left), len(right)],
            "null": "exchangeable whole-unit descriptor distributions, conditional on the fixed fitted classifier",
            "caveat": "assay-weighted and unit-weighted estimands differ; selection constraints and target/scaffold dependence beyond the chosen unit limit population inference"}


def c2st(left, right, *, regime, seed=42, repetitions=1999):
    parts = []
    unit_column = "assay_id" if regime == "A" else "component"
    for domain, source in enumerate((left, right)):
        part = source.copy()
        part["unit"] = part[unit_column].astype(str)
        part["domain"] = domain
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    if frame.groupby("unit").domain.nunique().max() != 1:
        raise ValueError("Partitions share indivisible units")
    units = frame[["unit", "domain"]].drop_duplicates()
    train_units, test_units = train_test_split(units, test_size=.30, random_state=seed, stratify=units.domain)
    training = frame.unit.isin(train_units.unit).to_numpy()
    x, y = descriptor_features(frame), frame.domain.to_numpy()
    weights = unit_domain_weights(frame.loc[training])
    classifier = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=1000, random_state=seed))
    classifier.fit(x[training], y[training], standardscaler__sample_weight=weights,
                   logisticregression__sample_weight=weights)
    heldout = frame.loc[~training, ["group_id", "unit", "domain"]].copy()
    heldout["probability_domain_1"] = classifier.predict_proba(x[~training])[:, 1]
    heldout["predicted_domain_1"] = (heldout.probability_domain_1 >= .5).astype(int)
    by_unit = heldout.groupby("unit").agg(rate=("predicted_domain_1", "mean"), domain=("domain", "first"), groups=("group_id", "size"))
    result = unit_test(by_unit.rate, by_unit.domain, repetitions=repetitions, seed=seed)
    result.update({"features": FEATURES, "regime": regime, "unit_definition": unit_column,
        "descriptor_label_policy": "raw source labels before pair-median binary-label harmonization",
        "classifier": "fixed logistic regression C=1, training-only weighted standardization, no hyperparameter selection",
        "training_assays": int(training.sum()), "heldout_assays": int((~training).sum()),
        "training_units_per_domain": train_units.groupby("domain").size().to_dict(),
        "heldout_assay_balanced_accuracy": float(balanced_accuracy_score(heldout.domain, heldout.predicted_domain_1)),
        "heldout_assay_roc_auc": float(roc_auc_score(heldout.domain, heldout.probability_domain_1)),
        "largest_heldout_unit_assay_share": float(by_unit.groups.max() / by_unit.groups.sum()),
        "largest_training_unit_assay_share": float(frame.loc[training].groupby("unit").size().max() / training.sum()),
        "standardized_coefficients": dict(zip(FEATURES, classifier[-1].coef_[0].tolist())),
        "intercept": float(classifier[-1].intercept_[0]),
        "training_units": train_units.unit.tolist(), "heldout_units": test_units.unit.tolist()})
    if set(result["training_units"]) & set(result["heldout_units"]):
        raise RuntimeError("C2ST unit leakage")
    return result, heldout


def estimate_label_prior(confusion_counts, predicted_counts):
    """BBSE: P(predicted|true) pi_target = P_target(predicted)."""
    counts = np.asarray(confusion_counts, float)
    predicted = np.asarray(predicted_counts, float)
    if counts.shape != (2, 2) or predicted.shape != (2,) or (counts < 0).any() or (predicted < 0).any():
        raise ValueError("Nonnegative binary confusion and predicted counts required")
    if (counts.sum(axis=0) == 0).any() or predicted.sum() == 0:
        return {"identifiable": False, "reason": "missing class or empty target"}
    confusion = counts / counts.sum(axis=0, keepdims=True)
    condition = float(np.linalg.cond(confusion))
    result = {"confusion_predicted_by_true": confusion.tolist(), "condition": condition,
              "source_positive_prior": float(counts[:, 1].sum() / counts.sum()),
              "target_predicted_positive_rate": float(predicted[1] / predicted.sum())}
    if not np.isfinite(condition) or condition > 1e4:
        return {**result, "identifiable": False, "reason": "singular or ill-conditioned confusion matrix"}
    marginal = predicted / predicted.sum()
    raw = np.linalg.solve(confusion, marginal)
    bounded_positive = float(np.clip(raw[1], 0., 1.))
    fitted = confusion @ np.array([1 - bounded_positive, bounded_positive])
    return {**result, "identifiable": True, "unconstrained_positive_prior": float(raw[1]),
            "estimated_positive_prior": bounded_positive, "boundary_constraint_active": not 0 <= raw[1] <= 1,
            "marginal_residual_l2": float(np.linalg.norm(fitted - marginal))}


def bbse(validation, test, unit_map, *, repetitions=1999, seed=42):
    frames = []
    for frame in (validation, test):
        frame = frame.copy()
        frame["unit"] = frame.group_id.str.split("__", n=1).str[1].map(unit_map)
        frame["prediction"] = (frame.activity_logits >= 0).astype(int)
        if frame.unit.isna().any():
            raise ValueError("Unmapped prediction unit")
        frames.append(frame)
    validation, test = frames
    if set(validation.unit) & set(test.unit):
        raise ValueError("BBSE calibration and test share units")
    source_rows = pd.crosstab(validation.unit, validation.prediction * 2 + validation.binary_label.astype(int)).reindex(columns=range(4), fill_value=0).to_numpy()
    target_rows = pd.crosstab(test.unit, test.prediction).reindex(columns=range(2), fill_value=0).to_numpy()
    result = estimate_label_prior(source_rows.sum(axis=0).reshape(2, 2), target_rows.sum(axis=0))
    rng = np.random.default_rng(seed)
    estimates, boundaries = [], 0
    for _ in range(repetitions):
        source = source_rows[rng.integers(len(source_rows), size=len(source_rows))].sum(axis=0).reshape(2, 2)
        target = target_rows[rng.integers(len(target_rows), size=len(target_rows))].sum(axis=0)
        estimate = estimate_label_prior(source, target)
        if estimate["identifiable"]:
            estimates.append(estimate["estimated_positive_prior"])
            boundaries += estimate["boundary_constraint_active"]
    result.update({"unit_bootstrap_interval": np.quantile(estimates, [.025, .975]).tolist() if estimates else None,
        "bootstrap_repetitions": repetitions, "identified_repetitions": len(estimates), "boundary_repetitions": boundaries,
        "calibration_units": len(source_rows), "test_units": len(target_rows), "seed": seed,
        "actual_test_positive_rate_diagnostic_only": float(test.binary_label.mean()),
        "calibration_partition": "held-out validation, not the in-sample training probe",
        "assumption": "P(model prediction | binary label) is invariant between validation and test; an invertible confusion matrix is required",
        "limitations": "validation also selected the checkpoint; this is a post-selection diagnostic, not an independent calibration experiment. Class-conditional shift can invalidate the prior estimate. Test labels do not enter estimation."})
    return result


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, float)
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    adjusted[order] = np.minimum(1., np.maximum.accumulate(pvalues[order] * np.arange(len(pvalues), 0, -1)))
    return adjusted.tolist()


def main():
    base = WORK / "tier1"
    destination = base / "shift"
    destination.mkdir(parents=True, exist_ok=True)
    summaries = []
    for regime in ("A", "B"):
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            name = f"c2st_{regime}_{left}_{right}"
            path = destination / f"{name}.json"
            if path.exists():
                result = json.loads(path.read_text())
            else:
                frames = [pd.read_parquet(base / f"data/groups_{regime}_{part}_raw.parquet") for part in (left, right)]
                result, predictions = c2st(*frames, regime=regime)
                result.update(left=left, right=right, name=name)
                predictions.to_parquet(destination / f"{name}_heldout.parquet", index=False)
                write_json(path, result)
            summaries.append({k: v for k, v in result.items() if k not in ("training_units", "heldout_units")})
            print(f"C2ST {name}: unit-balanced accuracy {result['unit_balanced_accuracy']:.4f}", flush=True)
    adjusted = holm_adjust([r["whole_unit_permutation_p_one_sided"] for r in summaries])
    for row, pvalue in zip(summaries, adjusted):
        row["holm_adjusted_p_six_partition_comparisons"] = pvalue
    write_json(destination / "c2st_summary.json", summaries)
    status = json.loads((WORK / "baseline_status.json").read_text())
    from pathlib import Path
    state = json.loads((Path(status["best_model_checkpoint"]) / "trainer_state.json").read_text())
    epoch = int(state["epoch"])
    val = pd.read_parquet(base / f"predictions/epoch{epoch:02d}_val/predictions.parquet")
    test = pd.read_parquet(base / "predictions/best_test/predictions.parquet")
    units = pd.read_parquet(base / "data/assay_units.parquet").set_index("assay_id").component
    write_json(destination / "bbse_B.json", {"checkpoint": status["best_model_checkpoint"], **bbse(val, test, units)})
    write_json(destination / "complete.json", {"c2st_comparisons": 6, "bbse_regimes": ["B"],
        "pending": ["Split A BBSE once matched model predictions exist", "representation shift diagnostics in mechanistic tier"]})


if __name__ == "__main__":
    main()
