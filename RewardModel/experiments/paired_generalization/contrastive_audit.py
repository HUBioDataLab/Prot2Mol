"""Re-evaluate both contrastive directions against unchanged candidate batches."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

if __package__:
    from .run import WORK, write_json
else:
    from run import WORK, write_json

from reward_model.model.losses import ligunity_bidirectional_contrastive_loss
from reward_model.training.evaluation import build_complete_coverage_contrastive_partitions


@dataclass
class CandidateBatch:
    members: torch.Tensor
    representatives: torch.Tensor
    owners: torch.Tensor
    target_ids: torch.Tensor
    molecule_ids: torch.Tensor
    labels: torch.Tensor


def candidate_pool(frame, *, batch_size=12, seed=42, partitions=3, max_ligands=16):
    groups = torch.tensor(pd.factorize(frame.group_id, sort=True)[0])
    targets = torch.tensor(pd.factorize(frame.target_chembl_id, sort=True)[0])
    molecules = torch.tensor(pd.factorize(frame.compound_id, sort=True)[0])
    labels = torch.tensor(frame.pchembl_value.to_numpy(), dtype=torch.float32)
    partition_ids = build_complete_coverage_contrastive_partitions(
        contrastive_group_ids=groups, max_list_size=max_ligands, num_partitions=partitions, seed=seed)
    pools = []
    digest = hashlib.sha256()
    for array in (groups, targets, molecules, labels):
        digest.update(array.numpy().tobytes())
    for partition_index, ids in enumerate(partition_ids):
        count = int(ids.max()) + 1
        counts = torch.bincount(ids, minlength=count)
        order = torch.argsort(ids, stable=True)
        offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
        generator = torch.Generator().manual_seed((seed + 2_000_003 * (partition_index + 1)) % ((1 << 63) - 1))
        shuffled = torch.randperm(count, generator=generator)
        batches = []
        for start in range(0, count, batch_size):
            chosen = shuffled[start:start + batch_size]
            members = [order[offsets[i]:offsets[i + 1]] for i in chosen.tolist()]
            selected = torch.cat(members)
            representatives = torch.tensor([int(m[0]) for m in members])
            owners = torch.repeat_interleave(torch.arange(len(members)), torch.tensor([len(m) for m in members]))
            batch = CandidateBatch(selected, representatives, owners, targets[representatives], molecules[selected], labels[selected])
            if not torch.equal(batch.target_ids[owners], targets[selected]):
                raise RuntimeError("Candidate list mixes target identities")
            batches.append(batch)
            for array in (selected, representatives, owners):
                digest.update(array.numpy().tobytes())
        pools.append(batches)
    return pools, digest.hexdigest()


def directional_losses(protein, molecule, pools, scales):
    values = {name: [] for name in scales}
    for batches in pools:
        total_lists = sum(len(b.representatives) for b in batches)
        accumulated = {name: np.zeros(3) for name in scales}
        for batch in batches:
            cosines = protein[batch.representatives] @ molecule[batch.members].T
            for name, scale in scales.items():
                losses = ligunity_bidirectional_contrastive_loss(
                    cosines * scale, batch.labels, batch.owners, batch.target_ids, batch.molecule_ids,
                    active_threshold=6., strict_active_only=True)
                accumulated[name] += np.asarray([float(v) for v in losses]) * len(batch.representatives)
        for name in scales:
            values[name].append(accumulated[name] / total_lists)
    return {name: dict(zip(("total", "protein_to_molecule", "molecule_to_protein"),
                           np.mean(partitions, axis=0).tolist())) for name, partitions in values.items()}


def main():
    torch.set_num_threads(4)
    base = WORK / "tier1"
    destination = base / "contrastive"
    destination.mkdir(parents=True, exist_ok=True)
    pools = {}
    table = []
    for marker in sorted((base / "predictions").glob("*/complete.json")):
        info = json.loads(marker.read_text())
        name, split = info["name"], info["split"]
        target = destination / f"{name}.json"
        if target.exists():
            previous = json.loads(target.read_text())
            if previous["checkpoint_sha256"] != info["checkpoint_sha256"]:
                raise RuntimeError("Contrastive audit cache contains different weights")
            table.append(previous)
            continue
        frame = pd.read_parquet(marker.parent / "predictions.parquet")
        if split not in pools:
            pools[split] = (candidate_pool(frame), frame[["group_id", "target_chembl_id", "compound_id", "pchembl_value"]])
        (pool, pool_hash), identities = pools[split]
        if not identities.equals(frame[identities.columns]):
            raise RuntimeError("Fixed candidate pool identities or labels changed between checkpoints")
        protein = torch.from_numpy(np.load(marker.parent / "protein.npy"))
        molecule = torch.from_numpy(np.load(marker.parent / "molecule.npy"))
        results = directional_losses(protein, molecule, pool, {"learned_scale": info["score_scale"], "fixed_initial_scale13": 13.})
        active = frame.pchembl_value > 6.
        result = {**info, "candidate_pool_sha256": pool_hash,
            "candidate_policy": "three deterministic complete-coverage partitions, at most 16 ligands per assay-list and 12 assay-lists per candidate batch; identical to baseline validation policy",
            "losses": results, "positive_alignment": {
                "observations": int(active.sum()), "mean_cosine": float(frame.loc[active, "cosine_similarity"].mean()),
                "median_cosine": float(frame.loc[active, "cosine_similarity"].median())},
            "interpretation": "baseline validation already uses a fixed candidate pool; changing scale during rescoring isolates its numerical contribution, not the outcome of retraining with fixed temperature"}
        metric_path = base / "metrics" / f"{name}.json"
        metric = json.loads(metric_path.read_text())
        if "original_validation" in metric:
            reference = metric["original_validation"]["eval_contrastive_loss"]
            difference = results["learned_scale"]["total"] - reference
            result["export_minus_original_contrastive_loss"] = difference
            if abs(difference) > .01:
                raise RuntimeError(f"Contrastive audit differs materially from baseline evaluation: {name}: {difference}")
        write_json(target, result)
        table.append(result)
        print(f"Contrastive directions complete: {name}", flush=True)
    write_json(destination / "trajectory.json", table)


if __name__ == "__main__":
    main()
