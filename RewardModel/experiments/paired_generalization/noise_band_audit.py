"""Training-fixed replicate-noise bands for composition and checkpoint rescoring."""
from __future__ import annotations

from dataclasses import asdict
import json

import numpy as np
import pandas as pd

if __package__:
    from .label_precision import POLICY, read_predictions, trainer_label_frame
    from .metrics import pair_counts, size_bin
    from .run import DATA, ROOT, WORK, sha256, write_json
else:
    from label_precision import POLICY, read_predictions, trainer_label_frame
    from metrics import pair_counts, size_bin
    from run import DATA, ROOT, WORK, sha256, write_json


def summarize_pairs(table):
    totals = {name: int(table[name].sum()) for name in
              ("all_pairs", "comparable_pairs", "concordant", "discordant", "prediction_ties", "label_band_pairs")}
    all_pairs, comparable = totals["all_pairs"], totals["comparable_pairs"]
    valid = table.comparable_pairs > 0
    return {
        **totals, "groups": len(table), "groups_with_comparable_pairs": int(valid.sum()),
        "near_tie_pair_fraction": totals["label_band_pairs"] / all_pairs if all_pairs else None,
        "pooled_within_group_concordance": (totals["concordant"] - totals["discordant"]) / all_pairs if all_pairs else None,
        "pooled_ordering_accuracy": (totals["concordant"] + .5 * totals["prediction_ties"]) / comparable if comparable else None,
        "macro_ordering_accuracy": float(((table.loc[valid, "concordant"] + .5 * table.loc[valid, "prediction_ties"]) /
                                            table.loc[valid, "comparable_pairs"]).mean()) if valid.any() else None,
    }


def score_bands(frame, thresholds, pathways):
    records = {f"{band}/{pathway}": [] for band in thresholds for pathway in pathways}
    for group_id, group in frame.groupby("group_id", sort=True):
        labels = group.pchembl_value.to_numpy()
        for band, epsilon in thresholds.items():
            for pathway in pathways:
                counts = pair_counts(labels, group[pathway], epsilon)
                records[f"{band}/{pathway}"].append({"group_id": group_id, "size": len(group),
                                                    "size_bin": size_bin(len(group)), **asdict(counts)})
    tables = {name: pd.DataFrame(rows) for name, rows in records.items()}
    summaries = {}
    for name, table in tables.items():
        summaries[name] = {**summarize_pairs(table), "by_group_size": {
            size: summarize_pairs(table[table.size_bin == size]) for size in ("1", "2", "3", "4-10", "11-25", ">25")}}
    return summaries, tables


def main(trainer_labels=False):
    base = WORK / "tier1"
    noise_path = base / "replicates/noise_summary.json"
    noise = json.loads(noise_path.read_text())
    noise_hash = sha256(noise_path)
    destination = base / ("noise_bands_float32" if trainer_labels else "noise_bands")
    destination.mkdir(parents=True, exist_ok=True)
    fixed = {regime: {"training_replicate_rms": noise[regime]["train"]["noise_scale"]["rms_replicate_difference"],
                      "original_ranking_margin": float(np.log10(3.))} for regime in ("A", "B")}
    interpretation = {
        "label_precision": POLICY if trainer_labels else 'original exported label precision',
        "threshold_source": "training measurements only, fixed across validation/test and epochs",
        "noise_definition": "RMS difference between raw replicate pairs; sensitivity band, not an estimated uncertainty interval for aggregated medians",
        "concordance_denominator": "all within-group pairs including label-band ties",
        "accuracy_denominator": "only label-comparable pairs; predicted ties receive half credit",
    }
    for regime, directory in (("A", ROOT / "dataset/processed/chembl_37/target_aware_random_assay_activity_balanced"),
                               ("B", DATA / "scaffold_seed42")):
        for partition in ("train", "val", "test"):
            path = destination / f"composition_{regime}_{partition}.json"
            source = directory / f"{partition}.parquet"
            source_hash = sha256(source)
            if path.exists():
                cached = json.loads(path.read_text())
                if cached["noise_sha256"] != noise_hash or (trainer_labels and cached.get('source_sha256') != source_hash):
                    raise RuntimeError("Noise provenance changed")
                continue
            frame = pd.read_parquet(directory / f"{partition}.parquet", columns=["assay_id", "target_chembl_id", "pchembl_value"])
            if trainer_labels:
                frame = trainer_label_frame(frame)
            frame["group_id"] = frame.target_chembl_id + "__" + frame.assay_id
            summary, tables = score_bands(frame, fixed[regime], ["pchembl_value"])
            # Label-as-score is used only to count near-tie pairs, never as model performance.
            for name, table in tables.items():
                table.to_parquet(destination / f"composition_{regime}_{partition}_{name.split('/')[0]}.parquet", index=False)
            write_json(path, {"thresholds": fixed[regime], "noise_sha256": noise_hash, 'source_sha256': source_hash, **interpretation,
                              "purpose": "near-tie composition only; labels are not predictions", "bands": summary})
            print(f"Near-tie composition {regime}/{partition}", flush=True)
    trajectory = []
    for manifest in sorted((base / "predictions").glob("*/complete.json")):
        provenance = json.loads(manifest.read_text())
        frame, precision = read_predictions(manifest.parent / "predictions.parquet", trainer_labels)
        path = destination / f"{provenance['name']}.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result["noise_sha256"] != noise_hash or result["checkpoint_sha256"] != provenance["checkpoint_sha256"]:
                raise RuntimeError("Prediction or noise provenance changed")
            if trainer_labels and any(result.get(key) != value for key, value in precision.items()):
                raise RuntimeError('Noise metrics have different prediction or precision provenance')
        else:
            if len(frame) != provenance["rows"] or not np.array_equal(frame.row_index, np.arange(len(frame))):
                raise RuntimeError("Prediction rows do not match manifest")
            summaries, tables = score_bands(frame, fixed["B"], ["ranking_score", "activity_logits", "cosine_similarity"])
            result = {**provenance, **precision, "thresholds": fixed["B"], "noise_sha256": noise_hash,
                      **interpretation, "metrics": summaries}
            for name, table in tables.items():
                table.to_parquet(destination / f"groups_{provenance['name']}_{name.replace('/', '_')}.parquet", index=False)
            write_json(path, result)
        trajectory.append(result)
        print(f"Noise-band metrics {provenance['name']}", flush=True)
    write_json(destination / "trajectory.json", trajectory)
    write_json(destination / "complete.json", {"composition_partitions": 6, "prediction_exports": len(trajectory),
                                               "prediction_scope": "Split B seed42 baseline; Split A comparisons still pending"})


if __name__ == "__main__":
    main()
