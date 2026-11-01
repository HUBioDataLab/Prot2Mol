"""Tier 1 within-group re-scoring, unit bootstrap, and validation-only calibration."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import ndcg_score, roc_auc_score

if __package__:
    from .metrics import aggregate_groups, group_metrics
    from .run import WORK, write_json
else:
    from metrics import aggregate_groups, group_metrics
    from run import WORK, write_json


def binary_logloss(labels, logits, temperature=1.):
    labels, logits = np.asarray(labels, float), np.asarray(logits, float) / temperature
    return float(np.mean(np.logaddexp(0., logits) - labels * logits))


def fit_temperature(labels, logits):
    result = minimize_scalar(lambda log_t: binary_logloss(labels, logits, np.exp(log_t)),
                             bounds=(-5., 5.), method="bounded")
    if not result.success:
        raise RuntimeError("Temperature fitting failed")
    return float(np.exp(result.x))


def unit_bootstrap(frame, *, repetitions=2000, seed=42):
    """Resample complete units, including units with zero eligible groups."""
    grouped = frame.groupby("component").spearman.agg(["sum", "count"])
    if len(grouped) < 2 or not grouped["count"].sum():
        return {"units": len(grouped), "interval": None, "reason": "fewer than two units or no defined correlations"}
    rng = np.random.default_rng(seed)
    totals, counts = grouped["sum"].to_numpy(), grouped["count"].to_numpy()
    values = []
    for start in range(0, repetitions, 100):
        sampled = rng.integers(len(grouped), size=(min(100, repetitions - start), len(grouped)))
        numerator, denominator = totals[sampled].sum(axis=1), counts[sampled].sum(axis=1)
        values.extend((numerator[denominator > 0] / denominator[denominator > 0]).tolist())
    return {"units": len(grouped), "repetitions": repetitions, "defined_repetitions": len(values),
            "seed": seed, "interval": np.quantile(values, [.025, .975]).tolist() if values else None,
            "interpretation": "percentile interval conditional on these held-out units; not split-realization or training-seed uncertainty"}


def rescore(frame, component_map):
    pathways = {name: [] for name in ("ranking_score", "activity_logits", "cosine_similarity")}
    for group_id, group in frame.groupby("group_id", sort=True):
        labels = group.pchembl_value.to_numpy()
        for name, records in pathways.items():
            scores = group[name].to_numpy()
            record = group_metrics(labels, scores)
            record["group_id"] = group_id
            assay = group_id.split("__", 1)[1]
            record["component"] = int(component_map[assay])
            record["positive_rate"] = float(group.binary_label.mean())
            for k in (1, 3, 5):
                record[f"ndcg_at_{k}_linear"] = (
                    float(ndcg_score((labels - labels.min())[None, :], scores[None, :], k=min(k, len(labels))))
                    if len(labels) > 1 and np.ptp(labels) > 0 else None
                )
            records.append(record)
    summaries = {}
    for name, records in pathways.items():
        table = pd.json_normalize(records)
        summary = aggregate_groups(records)
        summary["by_group_size"] = {size: aggregate_groups([r for r in records if r["size_bin"] == size])
                                     for size in ("1", "2", "3", "4-10", "11-25", ">25")}
        for k in (1, 3, 5):
            summary[f"macro_ndcg_at_{k}_linear"] = float(table[f"ndcg_at_{k}_linear"].mean())
        original = table.copy()
        original.loc[~original.original_ranking_eligible, "spearman"] = np.nan
        summary["unit_bootstrap_all_defined"] = unit_bootstrap(table)
        summary["unit_bootstrap_original_eligible"] = unit_bootstrap(original)
        summaries[name] = summary
        pathways[name] = table
    summaries["classification"] = {
        "logloss": binary_logloss(frame.binary_label, frame.activity_logits),
        "roc_auc": float(roc_auc_score(frame.binary_label, frame.activity_logits)) if frame.binary_label.nunique() == 2 else None,
    }
    return summaries, pathways


def analyze_all():
    base = WORK / "tier1"
    component_map = pd.read_parquet(base / "data/assay_units.parquet").set_index("assay_id").component.to_dict()
    out = base / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    exports = sorted((base / "predictions").glob("*/complete.json"))
    trajectory = []
    for manifest in exports:
        provenance = json.loads(manifest.read_text())
        name = provenance["name"]
        summary_path = out / f"{name}.json"
        if summary_path.exists():
            result = json.loads(summary_path.read_text())
            if result["checkpoint_sha256"] != provenance["checkpoint_sha256"]:
                raise RuntimeError("Metric cache refers to different weights")
        else:
            frame = pd.read_parquet(manifest.parent / "predictions.parquet")
            if len(frame) != provenance["rows"] or not np.array_equal(frame.row_index, np.arange(len(frame))):
                raise RuntimeError("Exported observations are incomplete or misaligned")
            summaries, records = rescore(frame, component_map)
            result = {**provenance, "metrics": summaries,
                      "epsilon_metrics": "pending raw-replicate noise estimation; no guessed noise threshold used"}
            if provenance["split"] == "val":
                temperature = fit_temperature(frame.binary_label, frame.activity_logits)
                result["calibration"] = {"temperature": temperature,
                    "fitting_partition": "validation only",
                    "validation_fit_logloss": binary_logloss(frame.binary_label, frame.activity_logits, temperature),
                    "note": "validation fit is in-sample; test transfer is evaluated only for the preselected checkpoint"}
                state = json.loads((Path(provenance["checkpoint"]) / "trainer_state.json").read_text())
                reference = [x for x in state["log_history"] if "eval_macro_spearman" in x][-1]
                result["original_validation"] = reference
                difference = summaries["ranking_score"]["original_eligible_macro_spearman"] - reference["eval_macro_spearman"]
                result["export_minus_original_macro_spearman"] = difference
                if abs(difference) > .001:
                    raise RuntimeError(f"Checkpoint export does not reproduce the original metric: {name}: {difference}")
            for pathway, table in records.items():
                table.to_parquet(out / f"groups_{name}_{pathway}.parquet", index=False)
            write_json(summary_path, result)
        trajectory.append(result)
        print(f"Re-scored {name}", flush=True)
    status = json.loads((WORK / "baseline_status.json").read_text())
    best = Path(status["best_model_checkpoint"])
    state = json.loads((best / "trainer_state.json").read_text())
    epoch = int(state["epoch"])
    calibration = json.loads((out / f"epoch{epoch:02d}_val.json").read_text())["calibration"]
    test = pd.read_parquet(base / "predictions/best_test/predictions.parquet")
    write_json(out / "best_test_calibration.json", {
        **calibration, "selected_checkpoint": str(best),
        "test_logloss_before": binary_logloss(test.binary_label, test.activity_logits),
        "test_logloss_after": binary_logloss(test.binary_label, test.activity_logits, calibration["temperature"]),
        "test_roc_auc": float(roc_auc_score(test.binary_label, test.activity_logits)),
        "ranking_effect": "positive temperature scaling cannot change the logit's ordering",
    })
    write_json(out / "trajectory.json", trajectory)
    write_json(out / "complete.json", {"exports_analyzed": len(exports),
        "scope": "Split B checkpoint rescores, descriptive training probe, unit bootstrap, calibration",
        "not_complete": ["Split A model predictions", "replicate-noise concordance", "contrastive direction/fixed-pool audit", "formal shift tests", "later experiment tiers"]})


if __name__ == "__main__":
    analyze_all()
