"""Nearest-neighbour lookup under the actual binary scaffold relation.

Candidates have measurements for the query protein. Every matching scaffold
has similarity 1; otherwise all candidates have similarity 0. Average all tied
nearest candidates uniformly after one median per training protein/molecule.
No graded chemical similarity or arbitrary molecule-ID tie break is introduced.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from .baseline_common import evaluate_predictions, load_partitions
else:
    from run import ROOT, WORK, sha256, write_json
    from baseline_common import evaluate_predictions, load_partitions


def fit_lookup(training, scaffold_map):
    frame = training.copy()
    frame["scaffold"] = frame.compound_id.map(scaffold_map)
    if frame.scaffold.isna().any():
        raise ValueError("Missing training scaffold")
    pairs = frame.groupby(["target_chembl_id", "compound_id", "scaffold"], sort=True).agg(
        ranking_score=("pchembl_value", "median"), probability=("binary_label", "mean")).reset_index()
    table = pairs.groupby(["target_chembl_id", "scaffold"]).agg(
        ranking_score=("ranking_score", "mean"), probability=("probability", "mean"), neighbors=("compound_id", "size"))
    fallback = pairs.groupby("target_chembl_id").agg(
        ranking_score=("ranking_score", "mean"), probability=("probability", "mean"), neighbors=("compound_id", "size"))
    global_fallback = {"ranking_score": float(pairs.ranking_score.mean()), "probability": float(pairs.probability.mean()), "neighbors": len(pairs)}
    return table, fallback, global_fallback


def predict_lookup(frame, scaffold_map, model):
    table, fallback, global_fallback = model
    scaffolds = frame.compound_id.map(scaffold_map)
    if scaffolds.isna().any():
        raise ValueError("Missing query scaffold")
    keys = pd.MultiIndex.from_arrays([frame.target_chembl_id, scaffolds], names=["target_chembl_id", "scaffold"])
    direct = table.reindex(keys).reset_index(drop=True)
    target_prior = fallback.reindex(frame.target_chembl_id).reset_index(drop=True)
    nearest = direct.combine_first(target_prior).fillna(global_fallback)
    output = frame.copy().reset_index(drop=True)
    output["ranking_score"] = nearest.ranking_score.to_numpy()
    probability = np.clip(nearest.probability.to_numpy(), 1e-6, 1 - 1e-6)
    output["classification_score"] = np.log(probability / (1 - probability))
    output["nearest_similarity"] = direct.neighbors.notna().astype(int).to_numpy()
    output["number_of_tied_nearest_molecules"] = nearest.neighbors.to_numpy().astype(int)
    output["query_target_seen"] = target_prior.neighbors.notna().to_numpy()
    return output


def main():
    import fcntl
    if ROOT != Path("/home/atabey/Prot2Mol"):
        raise RuntimeError("Runtime restricted to the authorized Atum project")
    base = WORK / "tier2/nearest_scaffold_lookup"
    base.mkdir(parents=True, exist_ok=True)
    lock = (base / "fit.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    scaffold_path = WORK / "tier1/data/molecule_scaffolds.parquet"
    scaffold_map = pd.read_parquet(scaffold_path).set_index("compound_id").scaffold
    scaffold_hash = sha256(scaffold_path)
    all_results = {}
    for regime in ("A", "B"):
        output = base / regime
        output.mkdir(parents=True, exist_ok=True)
        frames, hashes = load_partitions(regime)
        if (output / "complete.json").exists():
            result = json.loads((output / "complete.json").read_text())
            if result["data_sha256"] != hashes or result["scaffold_map_sha256"] != scaffold_hash:
                raise RuntimeError("Completed nearest-lookup provenance changed")
            all_results[regime] = result
            continue
        model = fit_lookup(frames["train"], scaffold_map)
        model[0].reset_index().to_parquet(output / "lookup.parquet", index=False)
        model[1].reset_index().to_parquet(output / "target_fallback.parquet", index=False)
        metrics = {}
        for partition in ("val", "test"):
            predictions = predict_lookup(frames[partition], scaffold_map, model)
            metrics[partition] = evaluate_predictions(predictions, regime, output / partition, classification_is_logit=True)
            metrics[partition].update(nearest_scaffold_match_fraction=float(predictions.nearest_similarity.mean()),
                                      query_target_seen_fraction=float(predictions.query_target_seen.mean()))
            print(f"Nearest lookup {regime}/{partition}: {metrics[partition]['original_eligible_macro_spearman']}", flush=True)
        result = {"regime": regime, "data_sha256": hashes, "scaffold_map_sha256": scaffold_hash, "metrics": metrics,
            "relation": "binary exact scaffold node identity, preserving the split builder's pooled acyclic node and private unparseable nodes",
            "candidate_rule": "training molecules measured for the query protein; median continuous label per training protein/molecule; average all similarity-maximizing molecule ties",
            "zero_similarity_rule": "all training molecules for the query protein tie; predict their mean, constant within each assay",
            "unseen_target_rule": "global training protein/molecule-pair mean; report frequency separately",
            "classification": "mean binary probability over the same candidates, clipped to [1e-6,1-1e-6] only for finite logits",
            "tuning": "none; fixed relation and deterministic tie averaging; validation and test labels never enter the lookup",
            "global_fallback": model[2], "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
        write_json(output / "complete.json", result)
        all_results[regime] = result
    write_json(base / "complete.json", all_results)


if __name__ == "__main__":
    main()
