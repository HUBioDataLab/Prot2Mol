"""Prepare the existing random assay split with partition-local pair labels."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

if __package__:
    from .run import DATA, ROOT, WORK, sha256, write_json
else:
    from run import DATA, ROOT, WORK, sha256, write_json


def partition_local_labels(frames):
    from reward_model.training.label_harmonization import harmonize_labels_by_pair_median

    result, summaries = {}, {}
    for name, frame in frames.items():
        labels, summary = harmonize_labels_by_pair_median({name: frame})
        result[name], summaries[name] = labels[name], summary
    return result, summaries


def archive_directory(path):
    archive = Path("/media/16TB/archive/atabey/Prot2Mol/paired-generalization-storage")
    target = archive / path.relative_to(ROOT)
    if path.is_symlink():
        if path.resolve() != target:
            raise RuntimeError(f"Unexpected storage link: {path}")
    elif path.exists() or target.exists():
        raise RuntimeError(f"Refusing to replace existing directory: {path}")
    else:
        target.mkdir(parents=True)
        path.symlink_to(target, target_is_directory=True)
    return target


def main():
    from reward_model.training.entry import prepare_training_examples_from_config

    if ROOT != Path("/home/atabey/Prot2Mol"):
        raise RuntimeError("Preparation is restricted to the authorized Atum project")
    source = ROOT / "dataset/processed/chembl_37/target_aware_random_assay_activity_balanced"
    config = ROOT / "RewardModel/configs/reward_train_paired_generalization_random.yaml"
    base = DATA / "random_seed42"
    archive_directory(base)
    archive_directory(WORK / "baseline_random_seed42_15ep")
    hashes = {name: sha256(source / f"{name}.parquet") for name in ("train", "val", "test")}
    provenance = base / "preparation.json"
    if provenance.exists():
        previous = json.loads(provenance.read_text())
        if previous["source_hashes"] != hashes or previous["config_sha256"] != sha256(config):
            raise RuntimeError("Prepared data or configuration changed")
    else:
        frames = {name: pd.read_parquet(source / f"{name}.parquet") for name in hashes}
        assay_sets = {name: set(f.assay_id) for name, f in frames.items()}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            if assay_sets[a] & assay_sets[b]:
                raise RuntimeError("Random split has overlapping assays")
        transformed, summaries = partition_local_labels(frames)
        destination = base / "partition_median"
        destination.mkdir(parents=True, exist_ok=True)
        for name, frame in transformed.items():
            frame.to_parquet(destination / f"{name}.parquet", index=False)
        write_json(provenance, {"source_hashes": hashes, "config_sha256": sha256(config),
            "label_policy": "pair median computed independently within each partition",
            "reason": "original pooled-label procedure lets held-out measurements alter 5289 training labels",
            "split_assignment": "unchanged existing random assay split, seed 42", "harmonization": summaries})
    ready = WORK / "random_tokenization_ready.json"
    if not ready.exists():
        write_json(ready, prepare_training_examples_from_config(str(config)))
    print(ready.read_text(), flush=True)


if __name__ == "__main__":
    main()
