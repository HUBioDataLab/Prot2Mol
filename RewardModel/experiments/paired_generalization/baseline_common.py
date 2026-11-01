"""Shared, fixed evaluation/data contracts for the Tier 2 baseline ladder."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import ndcg_score, roc_auc_score

if __package__:
    from .run import DATA, WORK, sha256, write_json
    from .metrics import aggregate_groups, group_metrics
    from .analyze_predictions import binary_logloss, unit_bootstrap
    from .noise_band_audit import summarize_pairs
else:
    from run import DATA, WORK, sha256, write_json
    from metrics import aggregate_groups, group_metrics
    from analyze_predictions import binary_logloss, unit_bootstrap
    from noise_band_audit import summarize_pairs

SOURCE_COLUMNS = ["assay_id", "target_chembl_id", "compound_id", "pchembl_value", "binary_label"]


def load_partitions(regime):
    directory = DATA / ("random_seed42/partition_median" if regime == "A" else "scaffold_seed42_median")
    frames, hashes = {}, {}
    for partition in ("train", "val", "test"):
        path = directory / f"{partition}.parquet"
        hashes[partition] = sha256(path)
        frame = pd.read_parquet(path, columns=SOURCE_COLUMNS).reset_index(drop=True)
        frame["group_id"] = frame.target_chembl_id + "__" + frame.assay_id
        # The deep-training dataset exposes these labels as float32 tensors.
        frame["pchembl_value"] = frame.pchembl_value.astype("float32").astype("float64")
        frames[partition] = frame
    sets = [set(frame.assay_id) for frame in frames.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Assay leakage between baseline partitions")
    return frames, hashes


class RankingSelection:
    """Cache label ranks only; score candidates with the original eligible macro rho."""
    def __init__(self, frame):
        self.rows = len(frame)
        self.groups = []
        for indices in frame.groupby("group_id", sort=True).indices.values():
            y = frame.pchembl_value.to_numpy()[indices]
            if len(y) < 3 or np.ptp(y) < .5:
                continue
            ranks = rankdata(y)
            ranks -= ranks.mean()
            self.groups.append((indices, ranks / np.linalg.norm(ranks)))

    def __call__(self, scores):
        scores = np.asarray(scores, float)
        if scores.shape != (self.rows,) or not np.isfinite(scores).all():
            raise ValueError("Selection scores must align with validation rows")
        correlations = []
        for indices, labels in self.groups:
            values = scores[indices]
            if np.ptp(values) == 0:
                continue
            ranks = rankdata(values)
            ranks -= ranks.mean()
            correlations.append(float(labels @ ranks / np.linalg.norm(ranks)))
        return {"macro_spearman": float(np.mean(correlations)) if correlations else None,
                "eligible_groups": len(self.groups), "defined_groups": len(correlations),
                "defined_fraction": len(correlations) / len(self.groups) if self.groups else None}


def evaluate_predictions(frame, regime, output, *, classification_is_logit=False):
    if not np.isfinite(frame.ranking_score).all() or not np.isfinite(frame.classification_score).all():
        raise ValueError("Nonfinite baseline prediction")
    units = pd.read_parquet(WORK / "tier1/data/assay_units.parquet")
    unit_map = units.set_index("assay_id").component if regime == "B" else pd.Series(np.arange(len(units)), index=units.assay_id)
    noise = json.loads((WORK / "tier1/replicates/noise_summary.json").read_text())
    epsilon = noise[regime]["train"]["noise_scale"]["rms_replicate_difference"]
    records = []
    for name, group in frame.groupby("group_id", sort=True):
        record = group_metrics(group.pchembl_value, group.ranking_score, epsilon=epsilon)
        record.update(group_id=name, component=int(unit_map[group.assay_id.iloc[0]]))
        y = group.pchembl_value.to_numpy()
        s = group.ranking_score.to_numpy()
        for k in (1, 3, 5):
            record[f"ndcg_at_{k}_linear"] = float(ndcg_score((y - y.min())[None, :], s[None, :], k=min(k, len(y)))) if len(y) > 1 and np.ptp(y) > 0 else None
        records.append(record)
    table = pd.json_normalize(records)
    eligible = table.original_ranking_eligible
    bootstrap = table.copy()
    bootstrap.loc[~eligible, "spearman"] = np.nan
    summary = {**aggregate_groups(records), "original_eligible_groups": int(eligible.sum()),
        "original_eligible_defined_fraction": float(table.loc[eligible].spearman.notna().mean()),
        "original_eligible_unit_bootstrap": unit_bootstrap(bootstrap),
        "unit_definition": "assay" if regime == "A" else "scaffold-assay component",
        "constant_prediction_policy": "undefined Spearman excluded, with defined fraction explicit; pairwise metrics give predicted ties half credit",
        "by_group_size": {size: aggregate_groups([row for row in records if row["size_bin"] == size]) for size in ("1", "2", "3", "4-10", "11-25", ">25")},
        "noise_band": {"epsilon": epsilon, "source": "training raw-replicate RMS", **summarize_pairs(pd.DataFrame([row["epsilon_pairs"] for row in records]))},
        "classification_roc_auc": float(roc_auc_score(frame.binary_label, frame.classification_score)) if frame.binary_label.nunique() == 2 else None}
    for k in (1, 3, 5):
        summary[f"macro_ndcg_at_{k}_linear"] = float(table[f"ndcg_at_{k}_linear"].mean())
    if classification_is_logit:
        summary["classification_logloss"] = binary_logloss(frame.binary_label, frame.classification_score)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output / "predictions.parquet", index=False)
    table.to_parquet(output / "groups.parquet", index=False)
    write_json(output / "metrics.json", summary)
    return summary
