"""Preparation and reproducible training for the user-specified experiment plan."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "RewardModel"))
WORK = ROOT / "RewardModel/outputs/paired_generalization"
DATA = ROOT / "dataset/processed/chembl_37/paired_generalization"
CONFIG = ROOT / "RewardModel/configs/reward_train_paired_generalization_baseline.yaml"


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare():
    import pandas as pd
    from reward_model.training.scaffold_component_split import write_scaffold_component_split
    from reward_model.training.label_harmonization import harmonize_labels_by_pair_median
    from reward_model.training.entry import prepare_training_examples_from_config

    source = ROOT / "dataset/processed/chembl_37/protein_cluster_50_activity_balanced/all.parquet"
    split_dir = DATA / "scaffold_seed42"
    median_dir = DATA / "scaffold_seed42_median"
    provenance = WORK / "source_provenance.json"
    source_hash = sha256(source)
    if provenance.exists():
        previous = json.loads(provenance.read_text())
        if previous["source_sha256"] != source_hash:
            raise RuntimeError("Source changed after preparation; refusing to reuse cached splits")
    else:
        write_json(provenance, {
            "source": str(source), "source_sha256": source_hash,
            "reference_run": "hubiodatascilab/Prot2Mol_RewardModel/m5guoqkx",
            "original_run_data_hash_available": False,
            "data_status": "rebuilt from the available project-local source; original split identity unverified",
        })
    manifest = split_dir / "scaffold_component_split.json"
    if not manifest.exists():
        write_scaffold_component_split(source, split_dir, seed=42, num_proc=8)
    if not (median_dir / "label_harmonization.json").exists():
        frames = {name: pd.read_parquet(split_dir / f"{name}.parquet") for name in ("train", "val", "test")}
        harmonized, summary = harmonize_labels_by_pair_median(frames)
        # The scaffold baseline must have no shared pair across partitions.
        pair_sets = {name: set(zip(f.target_chembl_id, f.compound_id)) for name, f in frames.items()}
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            if pair_sets[left] & pair_sets[right]:
                raise RuntimeError(f"Scaffold split unexpectedly shares target/compound pairs: {left}/{right}")
        median_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in harmonized.items():
            frame.to_parquet(median_dir / f"{name}.parquet", index=False)
        write_json(median_dir / "label_harmonization.json", summary)
        del frames, harmonized, pair_sets
    ready = WORK / "tokenization_ready.json"
    if not ready.exists():
        write_json(ready, prepare_training_examples_from_config(str(CONFIG)))
    print(ready.read_text(), flush=True)


def make_smoke():
    import pandas as pd
    import yaml
    from reward_model.training.entry import prepare_training_examples_from_config

    config = yaml.safe_load(CONFIG.read_text())
    smoke_dir = DATA / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    source_dir = DATA / "scaffold_seed42_median"
    if not all((source_dir / f"{name}.parquet").is_file() for name in ("train", "val", "test")):
        # This is only a runtime qualification fixture, not a scientific result.
        # It can run while the much larger scaffold preparation is in progress.
        source_dir = ROOT / "dataset/processed/chembl_37/target_aware_random_assay_activity_balanced"
    for name, limit in (("train", 256), ("val", 32), ("test", 32)):
        frame = pd.read_parquet(source_dir / f"{name}.parquet")
        grouped = frame.groupby(["target_chembl_id", "assay_id"], sort=True).pchembl_value.agg(["size", "min", "max"])
        groups = grouped[(grouped["size"] >= 2) & (grouped["size"] <= 16) & ((grouped["max"] - grouped["min"]) >= .5)].head(limit).index
        mask = pd.MultiIndex.from_frame(frame[["target_chembl_id", "assay_id"]]).isin(groups)
        frame.loc[mask].to_parquet(smoke_dir / f"{name}.parquet", index=False)
        config["data"][f"{name}_parquet_path"] = str(smoke_dir / f"{name}.parquet")
    config["data"]["tokenized_dataset_dir"] = str(DATA / "smoke_tokenized")
    config["data"]["tokenization_num_proc"] = 2
    config["training"].update({
        "output_dir": str(WORK / "smoke"), "max_steps": 2, "eval_steps": 2,
        "logging_steps": 1, "report_to": [], "save_total_limit": 2,
    })
    path = WORK / "smoke_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(WORK / "smoke_preparation.json", {
        **prepare_training_examples_from_config(str(path)),
        "source_dir": str(source_dir), "purpose": "runtime qualification only",
    })


def train(config_path):
    import torch
    import yaml
    from transformers import set_seed
    from reward_model.training.entry import train_reward_model_from_config

    path = Path(config_path).resolve()
    raw_config = yaml.safe_load(path.read_text())
    launch_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config_hash = sha256(path)
    # Seed before constructing the randomly initialized fusion and output heads.
    set_seed(int(raw_config["training"]["seed"]))
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        from reward_model.training.config import load_reward_training_config
        output_dir = Path(load_reward_training_config(str(path)).training.output_dir)
        write_json(output_dir / "launch_provenance.json", {
            "git_commit": launch_commit, "config_sha256": config_hash,
            "spec_sha256": sha256(ROOT / "Paired input ranking generalization.md"),
            "config": raw_config, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "initialization_seed_set_before_model_construction": True,
        })
    summary = train_reward_model_from_config(str(path))
    output = Path(summary["output_dir"])
    write_json(output / f"rank_{rank}_complete.json", {
        "config_sha256": config_hash,
        "rank": rank, "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "eval_macro_spearman": summary["eval_metrics"].get("eval_macro_spearman"),
    })
    if rank == 0:
        write_json(output / "experiment_summary.json", {
            **summary, "config_sha256": config_hash,
            "git_commit": launch_commit,
            "spec_sha256": sha256(ROOT / "Paired input ranking generalization.md"),
            "initialization_seed_set_before_model_construction": True,
            "torch_version": torch.__version__,
        })
        print(json.dumps(summary, indent=2, default=str), flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "make-smoke", "train"])
    parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args()
    {"prepare": prepare, "make-smoke": make_smoke, "train": lambda: train(args.config)}[args.action]()


if __name__ == "__main__":
    main()
