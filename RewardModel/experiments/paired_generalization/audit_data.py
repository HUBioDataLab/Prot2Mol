"""Six-partition composition and exact scaffold-relation audits from Tier 1."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .run import DATA, ROOT, WORK, sha256, write_json
else:
    from run import DATA, ROOT, WORK, sha256, write_json

COLS = ["assay_id", "target_chembl_id", "compound_id", "smiles", "pchembl_value", "binary_label", "measurement_count"]
AUDIT = WORK / "tier1" / "data"


def describe(values):
    array = np.asarray(values, dtype=float)
    return {"count": len(array), "mean": float(array.mean()),
            "quantiles": dict(zip(["min", "p25", "median", "p75", "p90", "p99", "max"],
                                  np.quantile(array, [0, .25, .5, .75, .9, .99, 1]).tolist()))}


def group_descriptors(frame):
    grouped = frame.groupby(["target_chembl_id", "assay_id"], sort=True)
    groups = grouped.agg(size=("pchembl_value", "size"), low=("pchembl_value", "min"),
                         high=("pchembl_value", "max"), variance=("pchembl_value", "var"),
                         unique_labels=("pchembl_value", "nunique"), positive_rate=("binary_label", "mean"),
                         replicate_mean=("measurement_count", "mean"), replicate_max=("measurement_count", "max"))
    groups["variance"] = groups["variance"].fillna(0) * (groups["size"] - 1) / groups["size"]
    groups["span"] = groups.high - groups.low
    groups["training_span_float32"] = groups.high.astype("float32") - groups.low.astype("float32")
    groups["ranking_eligible"] = (groups["size"] >= 3) & (groups["training_span_float32"] >= .5)
    groups["all_pairs"] = groups["size"] * (groups["size"] - 1) // 2
    tied = frame.groupby(["target_chembl_id", "assay_id", "pchembl_value"]).size()
    tied = (tied * (tied - 1) // 2).groupby(level=[0, 1]).sum()
    groups["tied_label_pairs"] = tied
    groups = groups.reset_index()
    groups["group_id"] = groups.target_chembl_id + "__" + groups.assay_id
    return groups


def partition_summary(frame, groups, training_groups):
    return {
        "rows": len(frame), "groups": len(groups), "molecules": int(frame.compound_id.nunique()),
        "targets": int(frame.target_chembl_id.nunique()), "positive_rate": float(frame.binary_label.mean()),
        "ranking_eligible_groups": int(groups.ranking_eligible.sum()),
        "ranking_eligible_fraction": float(groups.ranking_eligible.mean()),
        "constant_label_groups": int((groups.unique_labels == 1).sum()),
        "all_positive_groups": int((groups.positive_rate == 1).sum()),
        "all_negative_groups": int((groups.positive_rate == 0).sum()),
        "group_size": describe(groups["size"]), "within_group_variance": describe(groups.variance),
        "scores": describe(frame.pchembl_value), "measurement_count": describe(frame.measurement_count),
        "replicated_row_fraction": float((frame.measurement_count > 1).mean()),
        "label_tied_pair_fraction": float(groups.tied_label_pairs.sum() / groups.all_pairs.sum()),
        "target_training_groups": describe(groups.target_chembl_id.map(training_groups).fillna(0)),
    }


def run_audit():
    from reward_model.training.scaffold_component_split import compute_scaffold_nodes, connected_components
    from reward_model.training.label_harmonization import harmonize_labels_by_pair_median

    status = json.loads((WORK / "baseline_status.json").read_text())
    if status["epoch"] != 15 or status["phase"] != "baseline_complete_experiments_pending":
        raise RuntimeError("Baseline must finish first")
    AUDIT.mkdir(parents=True, exist_ok=True)
    source_path = ROOT / "dataset/processed/chembl_37/protein_cluster_50_activity_balanced/all.parquet"
    source = pd.read_parquet(source_path, columns=COLS)
    source_hash = sha256(source_path)
    cache = AUDIT / "scaffold_cache.json"
    if cache.exists():
        if json.loads(cache.read_text())["source_sha256"] != source_hash:
            raise RuntimeError("Scaffold cache source changed")
        molecules = pd.read_parquet(AUDIT / "molecule_scaffolds.parquet")
        units = pd.read_parquet(AUDIT / "assay_units.parquet")
    else:
        mapping, counts = compute_scaffold_nodes(source.smiles, num_proc=8)
        scaffolds = source.smiles.map(mapping)
        components = connected_components(source.assay_id, scaffolds)
        molecules = source[["compound_id", "smiles"]].drop_duplicates().copy()
        molecules["scaffold"] = molecules.smiles.map(mapping)
        if molecules.groupby("compound_id").scaffold.nunique().max() != 1:
            raise RuntimeError("Molecule identity has ambiguous scaffold")
        molecules = molecules.drop_duplicates("compound_id")
        units = components.rename("component").rename_axis("assay_id").reset_index()
        molecules.to_parquet(AUDIT / "molecule_scaffolds.parquet", index=False)
        units.to_parquet(AUDIT / "assay_units.parquet", index=False)
        write_json(cache, {"source_sha256": source_hash, **counts})
    scaffold_map = molecules.set_index("compound_id").scaffold
    component_map = units.set_index("assay_id").component
    summary = {"source_sha256": source_hash, "relation": "exact Bemis-Murcko scaffold identity; binary, not graded chemical similarity", "splits": {}}
    for split, directory in (("A", ROOT / "dataset/processed/chembl_37/target_aware_random_assay_activity_balanced"),
                              ("B", DATA / "scaffold_seed42")):
        raw = {name: pd.read_parquet(directory / f"{name}.parquet", columns=COLS) for name in ("train", "val", "test")}
        transformed, harmonization = harmonize_labels_by_pair_median(raw)
        # Quantify the original transformation's use of held-out labels; do not silently repair it.
        train_only, _ = harmonize_labels_by_pair_median({"train": raw["train"]})
        global_training_labels = transformed["train"].binary_label.to_numpy()
        contamination = int((global_training_labels != train_only["train"].binary_label.to_numpy()).sum())
        training_groups = raw["train"][["target_chembl_id", "assay_id"]].drop_duplicates().groupby("target_chembl_id").size()
        training_molecules = set(raw["train"].compound_id)
        training_scaffolds = set(raw["train"].compound_id.map(scaffold_map))
        training_pairs = pd.MultiIndex.from_frame(raw["train"][["target_chembl_id", "compound_id"]])
        split_summary = {"harmonization": harmonization, "training_labels_changed_by_heldout_measurements": contamination, "partitions": {}}
        for partition, frame in raw.items():
            print(f"audit {split}/{partition}", flush=True)
            current = {}
            for variant, f in (("raw", frame), ("original_harmonization", transformed[partition])):
                groups = group_descriptors(f)
                groups["component"] = groups.assay_id.map(component_map)
                groups["target_training_groups"] = groups.target_chembl_id.map(training_groups).fillna(0)
                if groups.component.isna().any():
                    raise RuntimeError("Missing scaffold component")
                groups.to_parquet(AUDIT / f"groups_{split}_{partition}_{variant}.parquet", index=False)
                current[variant] = partition_summary(f, groups, training_groups)
            rows = frame[["assay_id", "target_chembl_id", "compound_id"]].copy()
            rows["component"] = rows.assay_id.map(component_map)
            rows["scaffold"] = rows.compound_id.map(scaffold_map)
            rows["molecule_seen"] = rows.compound_id.isin(training_molecules)
            rows["scaffold_seen"] = rows.scaffold.isin(training_scaffolds)
            rows["pair_seen"] = pd.MultiIndex.from_frame(rows[["target_chembl_id", "compound_id"]]).isin(training_pairs)
            rows.to_parquet(AUDIT / f"overlap_{split}_{partition}.parquet", index=False)
            current["row_overlap"] = {key: float(rows[key].mean()) for key in ("molecule_seen", "scaffold_seen", "pair_seen")}
            current["component_sizes"] = describe(rows.groupby("component").size())
            rows.groupby("component").agg(rows=("assay_id", "size"), assays=("assay_id", "nunique"), molecules=("compound_id", "nunique"), targets=("target_chembl_id", "nunique")).reset_index().to_parquet(AUDIT / f"components_{split}_{partition}.parquet", index=False)
            split_summary["partitions"][partition] = current
        summary["splits"][split] = split_summary
    write_json(AUDIT / "composition.json", summary)
    print("Six-partition composition audit complete", flush=True)


if __name__ == "__main__":
    run_audit()
