"""Export checkpoint predictions for Tier 1, without changing any model weights."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

if __package__:
    from .run import DATA, ROOT, WORK, sha256, write_json
else:
    from run import DATA, ROOT, WORK, sha256, write_json


def checkpoint_jobs():
    status = json.loads((WORK / "baseline_status.json").read_text())
    if status["phase"] != "baseline_complete_experiments_pending" or status["epoch"] != 15:
        raise RuntimeError("The 15-epoch baseline must finish before the audit")
    baseline = WORK / "baseline_scaffold_seed42_15ep"
    checkpoints = sorted(baseline.glob("checkpoint-*/trainer_state.json"),
                         key=lambda p: int(p.parent.name.split("-")[-1]))
    if len(checkpoints) != 15:
        raise RuntimeError("All 15 epoch checkpoints are required")
    jobs = []
    best = Path(status["best_model_checkpoint"])
    jobs.append({"checkpoint": str(best), "split": "test", "name": "best_test"})
    for path in checkpoints:
        state = json.loads(path.read_text())
        epoch = int(state["epoch"])
        jobs.append({"checkpoint": str(path.parent), "split": "val", "name": f"epoch{epoch:02d}_val"})
        jobs.append({"checkpoint": str(path.parent), "split": "train", "name": f"epoch{epoch:02d}_train_probe"})
    return jobs


def train_probe_indices(dataset, number=2048, seed=42):
    """Choose whole assays once, without using labels or predictions."""
    groups = np.asarray(dataset["group_id"])
    unique = np.unique(groups)
    chosen = np.random.default_rng(seed).choice(unique, size=min(number, len(unique)), replace=False)
    return np.flatnonzero(np.isin(groups, chosen))


def score_dataset(model, dataset, output, *, batch_size=32, device="cuda"):
    import pandas as pd
    import torch
    from reward_model.training.data import RewardPairCollator

    collator = RewardPairCollator(
        dynamic_padding=True,
        protein_pad_token_id=model.protein_tokenizer.pad_token_id,
        molecule_pad_token_id=model.molecule_tokenizer.pad_token_id,
    )
    output.mkdir(parents=True, exist_ok=True)
    records = []
    protein = molecule = None
    scales = []
    model.eval()
    for start in range(0, len(dataset), batch_size):
        batch = dataset[start:start + batch_size]
        rows = [dict(zip(batch, values)) for values in zip(*batch.values())]
        inputs = {k: v.to(device) for k, v in collator.collate_example_tokens(rows).items()}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            result = model(**inputs, return_dict=True)
        tensors = {name: getattr(result, name).detach().float().cpu().numpy() for name in (
            "ranking_score", "activity_logits", "cosine_similarity",
            "normalized_protein_embedding", "normalized_molecule_embedding")}
        if not all(np.isfinite(v).all() for v in tensors.values()):
            raise RuntimeError("Nonfinite prediction or representation")
        if protein is None:
            shape = (len(dataset), tensors["normalized_protein_embedding"].shape[1])
            protein = np.lib.format.open_memmap(output / "protein.npy", mode="w+", dtype="float32", shape=shape)
            molecule = np.lib.format.open_memmap(output / "molecule.npy", mode="w+", dtype="float32", shape=shape)
        stop = start + len(rows)
        protein[start:stop] = tensors["normalized_protein_embedding"]
        molecule[start:stop] = tensors["normalized_molecule_embedding"]
        scales.append(float(result.score_scale.detach().float().cpu()))
        for offset, row in enumerate(rows):
            records.append({
                "row_index": start + offset,
                **{name: row[name] for name in ("group_id", "target_chembl_id", "compound_id", "pchembl_value", "binary_label")},
                **{name: float(tensors[name].reshape(-1)[offset]) for name in ("ranking_score", "activity_logits", "cosine_similarity")},
            })
        if start % (batch_size * 256) == 0:
            print(json.dumps({"output": output.name, "scored": stop, "total": len(dataset)}), flush=True)
    if not records or not np.allclose(scales, scales[0], rtol=1e-6, atol=1e-6):
        raise RuntimeError("Missing observations or changing inference scale")
    protein.flush()
    molecule.flush()
    pd.DataFrame(records).to_parquet(output / "predictions.parquet", index=False)
    return {"rows": len(records), "score_scale": scales[0], "dtype": "bf16 autocast; exported float32"}


def worker(index, workers):
    import torch
    from datasets import load_from_disk
    from reward_model.model import load_reward_model

    if ROOT != Path("/home/atabey/Prot2Mol"):
        raise RuntimeError("Runtime is restricted to the authorized Atum project")
    allowed = subprocess.check_output(["nvidia-smi", "-i", "0,2,5,6", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in allowed:
        raise RuntimeError("Each worker must receive one authorized GPU UUID")
    torch.set_num_threads(4)
    datasets = {}
    model = None
    for job in checkpoint_jobs()[index::workers]:
        output = WORK / "tier1" / "predictions" / job["name"]
        checkpoint = Path(job["checkpoint"])
        digest = sha256(checkpoint / "pytorch_model.bin")
        if (output / "complete.json").exists():
            previous = json.loads((output / "complete.json").read_text())
            if previous["checkpoint_sha256"] != digest:
                raise RuntimeError("Completed export belongs to different weights")
            continue
        split = job["split"]
        if split not in datasets:
            ds = load_from_disk(str(DATA / "scaffold_seed42_tokenized" / f"{split}_examples"))
            if split == "train":
                indices = train_probe_indices(ds)
                ds = ds.select(indices.tolist())
                write_json(WORK / "tier1" / f"train_probe_worker{index}.json", {
                    "seed": 42, "assays_requested": 2048, "row_indices": indices.tolist(),
                    "selection": "uniform whole assays without labels; descriptive training probe, not full training metric",
                })
            datasets[split] = ds
        if model is None:
            model = load_reward_model(str(checkpoint), device=torch.device("cpu"))
            model.to("cuda")
        else:
            state = torch.load(checkpoint / "pytorch_model.bin", map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            del state
        summary = score_dataset(model, datasets[split], output)
        write_json(output / "complete.json", {
            **job, **summary, "checkpoint_sha256": digest,
            "dataset_fingerprint": datasets[split]._fingerprint,
            "visible_device": os.environ["CUDA_VISIBLE_DEVICES"],
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "finished_unix": time.time(),
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.worker < args.workers:
        raise ValueError("Invalid worker index")
    worker(args.worker, args.workers)


if __name__ == "__main__":
    main()
