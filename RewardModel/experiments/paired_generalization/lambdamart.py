"""Frozen-feature LambdaMART with intact queries and noise-calibrated linear gains.

XGBoost is used because LightGBM's DCG calculator caps query size at 10,000;
the actual training data include assays up to 49,020 molecules. XGBoost topk
is set to the full largest query, with lambda normalization and linear gain.
https://xgboost.readthedocs.io/en/stable/parameter.html
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from .baseline_common import RankingSelection, evaluate_predictions, load_partitions
else:
    from run import ROOT, WORK, sha256, write_json
    from baseline_common import RankingSelection, evaluate_predictions, load_partitions

VERSION = "3.4.1"
SETTINGS = [
    {"max_depth": 3, "reg_lambda": 1., "min_child_weight": 1.},
    {"max_depth": 5, "reg_lambda": 1., "min_child_weight": 1.},
    {"max_depth": 7, "reg_lambda": 1., "min_child_weight": 1.},
    {"max_depth": 5, "reg_lambda": 10., "min_child_weight": 1.},
    {"max_depth": 5, "reg_lambda": 1., "min_child_weight": 10.},
]
MAX_ROUNDS, EVALUATE_EVERY, PATIENCE_ROUNDS = 500, 10, 100


def query_layout(frame, noise_width):
    if not np.isfinite(noise_width) or noise_width <= 0:
        raise ValueError("A positive training-derived noise width is required")
    ordered = frame.sort_values("group_id", kind="stable").reset_index(drop=True)
    values = ordered.pchembl_value.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Finite continuous labels required")
    minimum = ordered.groupby("group_id", sort=False).pchembl_value.transform("min").to_numpy()
    levels = np.floor((values - minimum) / noise_width).astype(np.int32)
    sizes = ordered.groupby("group_id", sort=False).size().to_numpy(dtype=np.uint32)
    if int(sizes.sum()) != len(frame) or len(ordered) != len(frame):
        raise RuntimeError("Queries must be retained intact")
    return ordered, sizes, levels


def rank_parameters(largest_group, *, device="cuda"):
    if largest_group < 1:
        raise ValueError("At least one training query is required")
    return {"objective": "rank:ndcg", "tree_method": "hist", "device": device,
            "lambdarank_pair_method": "topk", "lambdarank_num_pair_per_sample": int(largest_group),
            "lambdarank_normalization": True, "lambdarank_score_normalization": True,
            "ndcg_exp_gain": False, "lambdarank_unbiased": False,
            "eta": .05, "max_bin": 256, "subsample": 1., "colsample_bytree": 1.,
            "seed": 42, "nthread": 8, "disable_default_eval_metric": True, "validate_parameters": True}


def pair_features(frame, maps, entities):
    blocks = []
    for side, column in (("protein", "target_chembl_id"), ("molecule", "compound_id")):
        positions = entities[side].get_indexer(frame[column])
        if (positions < 0).any():
            raise RuntimeError("Frozen feature lookup incomplete")
        blocks.append(np.asarray(maps[side][positions], dtype=np.float32))
    return np.concatenate(blocks, axis=1)


def fit_trial(xgb, train, validation, selector, params, output):
    output.mkdir(parents=True, exist_ok=True)
    history, selected, started = [], None, time.monotonic()

    class SelectSpearman(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            nonlocal selected
            rounds = epoch + 1
            if rounds % EVALUATE_EVERY and rounds != MAX_ROUNDS:
                return False
            metric = selector(model.predict(validation))
            record = {"rounds": rounds, **metric, "elapsed_seconds": time.monotonic() - started}
            history.append(record)
            if metric["macro_spearman"] is not None and (selected is None or metric["macro_spearman"] > selected["macro_spearman"]):
                selected = record.copy()
                model.save_model(output / "model.ubj")
            write_json(output / "validation_history.json", {"selected": selected, "history": history})
            print(f"LambdaMART {output.name} rounds={rounds}: val rho={metric['macro_spearman']}, coverage={metric['defined_fraction']:.4f}", flush=True)
            return selected is not None and rounds - selected["rounds"] >= PATIENCE_ROUNDS

    model = xgb.train(params, train, num_boost_round=MAX_ROUNDS, callbacks=[SelectSpearman()], verbose_eval=False)
    if selected is None:
        raise RuntimeError("No LambdaMART checkpoint produced defined validation rankings")
    result = {"parameters": params, "selected": selected, "rounds_trained": model.num_boosted_rounds(),
              "model_sha256": sha256(output / "model.ubj"), "elapsed_seconds": time.monotonic() - started}
    write_json(output / "complete.json", result)
    return result


def main(regime):
    import fcntl
    import xgboost as xgb
    if ROOT != Path("/home/atabey/Prot2Mol") or xgb.__version__ != VERSION:
        raise RuntimeError("Use the authorized Atum project and pinned XGBoost version")
    allowed = subprocess.check_output(["nvidia-smi", "-i", "0,2,5,6", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in allowed:
        raise RuntimeError("Select exactly one authorized GPU UUID")
    output = WORK / "tier2/lambdamart" / regime
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "fit.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frames, hashes = load_partitions(regime)
    noise = json.loads((WORK / "tier1/replicates/noise_summary.json").read_text())[regime]["train"]["noise_scale"]["rms_replicate_difference"]
    feature_base = WORK / "tier2/frozen_features"
    manifest = json.loads((feature_base / "complete.json").read_text())
    provenance = {"regime": regime, "data_sha256": hashes, "feature_provenance": manifest,
                  "settings": SETTINGS, "noise_width": noise, "xgboost_version": VERSION,
                  "max_rounds": MAX_ROUNDS, "evaluate_every": EVALUATE_EVERY, "patience_rounds": PATIENCE_ROUNDS}
    if (output / "complete.json").exists():
        old = json.loads((output / "complete.json").read_text())
        if any(old[key] != value for key, value in provenance.items()):
            raise RuntimeError("Completed LambdaMART provenance changed")
        return
    maps, entities = {}, {}
    for side in ("protein", "molecule"):
        path, table = feature_base / f"{side}.npy", feature_base / f"{side}_entities.parquet"
        if sha256(path) != manifest[side]["features_sha256"] or sha256(table) != manifest[side]["entities_sha256"]:
            raise RuntimeError("Frozen features or entity index changed")
        maps[side] = np.load(path, mmap_mode="r")
        entities[side] = pd.Index(pd.read_parquet(table, columns=["entity_id"]).entity_id)
    training, sizes, levels = query_layout(frames["train"], noise)
    parameters = rank_parameters(int(sizes.max()))
    layout = {"rows": len(training), "groups": len(sizes), "largest_query": int(sizes.max()),
              "queries_above_10000": int((sizes > 10000).sum()), "max_relevance_level": int(levels.max()),
              "zero_range_queries": int(pd.DataFrame({"group_id": training.group_id, "level": levels}).groupby("group_id")["level"].nunique().eq(1).sum())}
    write_json(output / "fit_contract.json", {**provenance, "base_parameters": parameters, "training_query_layout": layout})
    print(f"LambdaMART {regime}: building quantile matrices; {layout}", flush=True)
    data = pair_features(training, maps, entities)
    train = xgb.QuantileDMatrix(data, label=levels, group=sizes, max_bin=256, nthread=8)
    del data, training
    validation_frame, validation_sizes, validation_levels = query_layout(frames["val"], noise)
    data = pair_features(validation_frame, maps, entities)
    validation = xgb.QuantileDMatrix(data, label=validation_levels, group=validation_sizes, max_bin=256, ref=train, nthread=8)
    del data
    gc.collect()
    selector, trials = RankingSelection(validation_frame), []
    for number, settings in enumerate(SETTINGS):
        trial_dir = output / f"trial{number:02d}"
        params = {**parameters, **settings}
        if (trial_dir / "complete.json").exists():
            trial = json.loads((trial_dir / "complete.json").read_text())
            if trial["parameters"] != params or sha256(trial_dir / "model.ubj") != trial["model_sha256"]:
                raise RuntimeError("Completed LambdaMART trial changed")
        else:
            trial = fit_trial(xgb, train, validation, selector, params, trial_dir)
        trials.append({"trial": number, **trial})
        write_json(output / "trials.json", trials)
    best = max(trials, key=lambda trial: trial["selected"]["macro_spearman"])
    shutil.copyfile(output / f"trial{best['trial']:02d}/model.ubj", output / "model.ubj")
    model = xgb.Booster(model_file=output / "model.ubj")
    metrics = {}
    # Common final metrics use original continuous labels and original row order.
    for part in ("val", "test"):
        scores = np.empty(len(frames[part]))
        for start in range(0, len(scores), 32768):
            current = frames[part].iloc[start:start + 32768]
            batch = xgb.DMatrix(pair_features(current, maps, entities), nthread=8)
            scores[start:start + len(current)] = model.predict(batch)
        metrics[part] = evaluate_predictions(frames[part].assign(ranking_score=scores, classification_score=scores), regime, output / part)
    write_json(output / "complete.json", {**provenance, "trials": trials, "selected": best, "metrics": metrics,
        "model_sha256": sha256(output / "model.ubj"), "training_query_layout": layout,
        "label_levels": "floor((pChEMBL - within-assay minimum) / training replicate RMS); gains equal integer levels",
        "group_integrity": "All original queries retained, including 49,020-item assay; no query splitting, dropping or subsampling",
        "features": "concatenated unmodified frozen protein and molecule features; quantile cutpoints fitted on training only",
        "selection": "eligible macro validation Spearman, every 10 trees, patience 100 trees, max 500; five settings per split; test evaluated after selection",
        "engine_reason": "LightGBM hard query-size limit is 10,000; XGBoost permits full queries with topk >=largest group, linear gain, and per-query lambda normalization",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=["A", "B"], required=True)
    main(parser.parse_args().regime)
