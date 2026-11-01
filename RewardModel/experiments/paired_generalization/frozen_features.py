"""Pretrained pooled entity features for the specified Tier 2 baseline ladder.

No parameters, projection, normalization statistics, or labels are fitted here.
Downstream fitted transforms and model selection must use training/validation
partitions respectively. Test entities receive only a fixed encoder forward pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

if __package__:
    from .run import ROOT, WORK, CONFIG, sha256, write_json
else:
    from run import ROOT, WORK, CONFIG, sha256, write_json

DESTINATION = WORK / "tier2/frozen_features"
SIDES = {"protein": ("target_chembl_id", "protein_sequence"), "molecule": ("compound_id", "compound_selfies")}


def unique_entities(frame, identity, text):
    if frame[[identity, text]].isna().any().any() or (frame[text].str.len() == 0).any():
        raise ValueError("Entity identities and input strings must be present")
    pairs = frame[[identity, text]].drop_duplicates()
    if pairs[identity].duplicated().any():
        raise ValueError("One entity identity maps to multiple encoder input strings")
    pairs = pairs.sort_values(identity).reset_index(drop=True).rename(columns={identity: "entity_id", text: "text"})
    pairs.insert(0, "row_index", np.arange(len(pairs)))
    return pairs


def prepare():
    source = ROOT / "dataset/processed/chembl_37/protein_cluster_50_activity_balanced/all.parquet"
    DESTINATION.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source)
    ready = DESTINATION / "entities.json"
    if ready.exists():
        manifest = json.loads(ready.read_text())
        if manifest["source_sha256"] != source_hash:
            raise RuntimeError("Frozen-feature source changed")
        for side in SIDES:
            if sha256(DESTINATION / f"{side}_entities.parquet") != manifest[side]["sha256"]:
                raise RuntimeError("Entity mapping changed")
        return
    manifest = {"source_sha256": source_hash, "config_sha256": sha256(CONFIG),
                "scope": "pretrained entity encoders before learned projections and cross-attention; no supervised fitting"}
    for side, (identity, column) in SIDES.items():
        entities = unique_entities(pd.read_parquet(source, columns=[identity, column]), identity, column)
        path = DESTINATION / f"{side}_entities.parquet"
        entities.to_parquet(path, index=False)
        manifest[side] = {"entities": len(entities), "unique_input_strings": int(entities.text.nunique()), "sha256": sha256(path)}
        print(f"Prepared {side}: {len(entities)} entities", flush=True)
    write_json(ready, manifest)


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def worker(index, workers):
    import torch
    from reward_model.training.config import load_reward_training_config
    from reward_model.model.encoders import batch_encode_texts, encode_tokens, load_encoder_bundle
    from reward_model.model.fusion import masked_pool

    allowed = subprocess.check_output(["nvidia-smi", "-i", "0,2,5,6", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in allowed or torch.cuda.device_count() != 1:
        raise RuntimeError("Feature workers require exactly one of the authorized GPU UUIDs")
    if not 0 <= index < workers:
        raise ValueError("Invalid feature shard")
    torch.set_num_threads(4)
    manifest = json.loads((DESTINATION / "entities.json").read_text())
    config = load_reward_training_config(str(CONFIG)).model
    if manifest["config_sha256"] != sha256(CONFIG):
        raise RuntimeError("Feature configuration changed")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for side in SIDES:
        destination = DESTINATION / f"{side}_shard{index}"
        destination.mkdir(parents=True, exist_ok=True)
        complete = destination / "complete.json"
        if complete.exists():
            previous = json.loads(complete.read_text())
            if previous["entities_sha256"] != manifest[side]["sha256"] or previous["workers"] != workers:
                raise RuntimeError("Completed feature shard has different provenance")
            continue
        entities = pd.read_parquet(DESTINATION / f"{side}_entities.parquet")
        if sha256(DESTINATION / f"{side}_entities.parquet") != manifest[side]["sha256"]:
            raise RuntimeError("Entity table changed")
        indices = np.arange(index, len(entities), workers)
        # Sorting by input length reduces padding; the explicit indices preserve identity.
        lengths = entities.text.str.len().to_numpy()
        indices = indices[np.argsort(lengths[indices], kind="stable")]
        kwargs = {"local_files_only": True}
        for name in ("hidden_dropout_prob", "attention_probs_dropout_prob"):
            value = getattr(config, f"{side}_{name}")
            if value is not None:
                kwargs[name] = value
        tokenizer_kwargs = {"local_files_only": True}
        if side == "molecule":
            kwargs["trust_remote_code"] = config.molecule_trust_remote_code
            tokenizer_kwargs["trust_remote_code"] = config.molecule_trust_remote_code
            if config.molecule_deterministic_eval:
                kwargs["deterministic_eval"] = True
        name = getattr(config, f"{side}_model_name_or_path")
        bundle = load_encoder_bundle(name, tokenizer_name_or_path=getattr(config, f"{side}_tokenizer_name_or_path"),
                                     tokenizer_kwargs=tokenizer_kwargs, model_kwargs=kwargs)
        weights_hash = state_hash(bundle.model)
        pretrained_revision = getattr(bundle.model.config, "_commit_hash", None)
        bundle.model.eval().requires_grad_(False).to("cuda")
        maximum = getattr(config, f"{side}_max_length")
        pooling = getattr(config, f"{side}_pooling_type") or config.pooling_type
        batch_size = 16 if side == "protein" else 64
        provenance = {"side": side, "worker": index, "workers": workers, "rows": len(indices),
            "entities_sha256": manifest[side]["sha256"], "config_sha256": manifest["config_sha256"],
            "encoder_name": name, "encoder_state_sha256": weights_hash, "pretrained_revision": pretrained_revision,
            "maximum_tokens": maximum, "pooling": pooling, "hidden_size": bundle.hidden_size,
            "batch_size": batch_size, "visible_device": os.environ["CUDA_VISIBLE_DEVICES"],
            "dtype": "fp32 parameters, bf16 autocast, pooled features stored float32", "git_commit": commit,
            "tokens": "same unmodified input strings and special-token behavior as training; dynamic right padding"}
        progress_path = destination / "progress.json"
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            for key in ("entities_sha256", "config_sha256", "encoder_state_sha256", "workers", "worker", "batch_size"):
                if progress[key] != provenance[key]:
                    raise RuntimeError(f"Cannot resume feature shard: changed {key}")
            if not np.array_equal(np.load(destination / "indices.npy"), indices):
                raise RuntimeError("Resumed feature indices changed")
            completed = int(progress["completed_rows"])
            array = np.load(destination / "features.npy", mmap_mode="r+")
            if array.shape != (len(indices), bundle.hidden_size):
                raise RuntimeError("Feature shard shape changed")
        else:
            completed = 0
            np.save(destination / "indices.npy", indices)
            array = np.lib.format.open_memmap(destination / "features.npy", mode="w+", dtype="float32",
                                              shape=(len(indices), bundle.hidden_size))
        for start in range(completed, len(indices), batch_size):
            stop = min(start + batch_size, len(indices))
            tokens = batch_encode_texts(bundle.tokenizer, entities.text.iloc[indices[start:stop]].tolist(),
                                       maximum, device=torch.device("cuda"), padding="longest")
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                encoded = encode_tokens(bundle.model, tokens["input_ids"], tokens.get("attention_mask"))
                pooled = masked_pool(encoded, tokens.get("attention_mask"), pooling).float().cpu().numpy()
            if pooled.shape != (stop - start, bundle.hidden_size) or not np.isfinite(pooled).all():
                raise RuntimeError("Invalid encoder feature output")
            array[start:stop] = pooled
            if start % (batch_size * 64) == 0 or stop == len(indices):
                array.flush()
                write_json(progress_path, {**provenance, "completed_rows": stop, "updated_unix": time.time()})
                print(json.dumps({"side": side, "worker": index, "completed_rows": stop, "rows": len(indices)}), flush=True)
        array.flush()
        write_json(complete, {**provenance, "features_sha256": sha256(destination / "features.npy"),
                              "indices_sha256": sha256(destination / "indices.npy"), "finished_unix": time.time()})
        del bundle, array, tokens, encoded, pooled
        torch.cuda.empty_cache()


def combine(workers):
    manifest = json.loads((DESTINATION / "entities.json").read_text())
    results = {}
    for side in SIDES:
        ready = DESTINATION / f"{side}_complete.json"
        if ready.exists():
            results[side] = json.loads(ready.read_text())
            continue
        shards = [json.loads((DESTINATION / f"{side}_shard{i}/complete.json").read_text()) for i in range(workers)]
        if len({s["encoder_state_sha256"] for s in shards}) != 1 or any(s["entities_sha256"] != manifest[side]["sha256"] or s["workers"] != workers for s in shards):
            raise RuntimeError("Feature shard provenance mismatch")
        rows = manifest[side]["entities"]
        output = DESTINATION / f"{side}.npy"
        array = np.lib.format.open_memmap(output, mode="w+", dtype="float32", shape=(rows, shards[0]["hidden_size"]))
        assigned = np.zeros(rows, dtype=bool)
        for i, shard in enumerate(shards):
            directory = DESTINATION / f"{side}_shard{i}"
            if sha256(directory / "features.npy") != shard["features_sha256"] or sha256(directory / "indices.npy") != shard["indices_sha256"]:
                raise RuntimeError("Feature shard checksum mismatch")
            indices = np.load(directory / "indices.npy")
            features = np.load(directory / "features.npy", mmap_mode="r")
            if len(np.unique(indices)) != len(indices) or (indices < 0).any() or (indices >= rows).any() or assigned[indices].any():
                raise RuntimeError("Duplicated or invalid feature row indices")
            for start in range(0, len(indices), 10000):
                values = features[start:start + 10000]
                if not np.isfinite(values).all():
                    raise RuntimeError("Nonfinite completed feature shard")
                array[indices[start:start + len(values)]] = values
            assigned[indices] = True
        if not assigned.all():
            raise RuntimeError("Missing entity features")
        array.flush()
        results[side] = {"rows": rows, "hidden_size": array.shape[1], "features_sha256": sha256(output),
            "entities_sha256": manifest[side]["sha256"], "encoder_state_sha256": shards[0]["encoder_state_sha256"],
            "pretrained_revision": shards[0]["pretrained_revision"], "pooling": shards[0]["pooling"]}
        write_json(ready, results[side])
    write_json(DESTINATION / "complete.json", results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "worker", "combine"])
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if ROOT != Path("/home/atabey/Prot2Mol"):
        raise RuntimeError("Runtime is restricted to the authorized Atum project")
    {"prepare": prepare, "worker": lambda: worker(args.worker, args.workers), "combine": lambda: combine(args.workers)}[args.action]()


if __name__ == "__main__":
    main()
