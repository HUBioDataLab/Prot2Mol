"""Recover original measurements and quantify conditional split-half reliability.

The correction requested in the spec is a heuristic for grouped Spearman, not
a validated upper bound. Coverage and its parallel-measurement assumption are
reported explicitly; singleton measurements are never invented.
"""
from __future__ import annotations

import json
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

if __package__:
    from .run import DATA, ROOT, WORK, sha256, write_json
else:
    from run import DATA, ROOT, WORK, sha256, write_json

KEY = ["assay_id", "target_chembl_id", "compound_id"]
OUT = WORK / "tier1" / "replicates"


def reconstruct(raw, source):
    if source.duplicated(KEY).any() or raw.activity_id.duplicated().any():
        raise ValueError("Ambiguous source observations or duplicated activity IDs")
    stats = raw.groupby(KEY, sort=False).pchembl_value.agg(
        replicate_count="size", reconstructed_median="median", replicate_variance="var")
    stats["replicate_variance"] = stats.replicate_variance.fillna(0.)
    observations = source.reset_index(drop=True).reset_index(names="observation_id").merge(
        stats.reset_index(), on=KEY, how="left", validate="one_to_one")
    observations["count_matches"] = observations.measurement_count == observations.replicate_count
    observations["median_matches"] = np.isclose(observations.pchembl_value, observations.reconstructed_median,
                                                 rtol=0., atol=1e-8, equal_nan=False)
    observations["verified"] = observations.count_matches & observations.median_matches
    matched = raw.merge(observations[[*KEY, "observation_id", "verified"]], on=KEY,
                        how="inner", validate="many_to_one")
    return observations, matched


def noise_scale(observations):
    rows = observations[observations.verified & (observations.replicate_count >= 2)]
    n = rows.replicate_count.to_numpy()
    variance = rows.replicate_variance.to_numpy()
    pairs = n * (n - 1) / 2
    # Sum_{i<j}(x_i-x_j)^2 = n * (n-1) * sample_variance.
    epsilon = float(np.sqrt(np.sum(n * (n - 1) * variance) / pairs.sum())) if pairs.sum() else None
    return {"replicated_observations": len(rows), "within_observation_pairs": int(pairs.sum()),
            "rms_replicate_difference": epsilon,
            "definition": "RMS difference of two raw measurements of the same assay-target-parent molecule, pooled over replicate pairs; not a confidence bound or a median standard error"}


def split_half(observations, raw, *, repetitions=100, seed=42):
    full = observations.copy()
    full["group_id"] = full.target_chembl_id + "__" + full.assay_id
    grouped = full.groupby("group_id").pchembl_value.agg(["size", "min", "max"])
    eligible = grouped[(grouped["size"] >= 3) & ((grouped["max"].astype("float32") - grouped["min"].astype("float32")) >= .5)].index
    replicated = full[full.verified & (full.replicate_count >= 2) & full.group_id.isin(eligible)]
    counts = replicated.groupby("group_id").size()
    usable = counts[counts >= 3].index
    replicated = replicated[replicated.group_id.isin(usable)].reset_index(drop=True)
    result = {"original_eligible_groups": len(eligible), "groups_with_at_least_three_replicated_members": len(usable),
              "eligible_group_coverage": len(usable) / len(eligible) if len(eligible) else None,
              "replicated_members": len(replicated), "repetitions": repetitions, "seed": seed,
              "scope": "only replicated members of qualifying assays; does not identify a ceiling for all observations",
              "assumptions": "exchangeable parallel replicate measurements; Spearman-Brown applied to macro split-half correlation is an unvalidated grouped-rank heuristic"}
    if not len(replicated):
        return {**result, "mean_split_half_macro_spearman": None, "mean_spearman_brown_heuristic": None}
    wanted = set(replicated.observation_id)
    values = raw[raw.observation_id.isin(wanted)].groupby("observation_id").pchembl_value.apply(np.asarray).to_dict()
    replicate_arrays = [values[i] for i in replicated.observation_id]
    groups = list(replicated.groupby("group_id", sort=True).indices.values())
    rng = np.random.default_rng(seed)
    correlations, corrections, defined_counts = [], [], []
    for _ in range(repetitions):
        left, right = np.empty(len(replicated)), np.empty(len(replicated))
        for i, array in enumerate(replicate_arrays):
            shuffled = rng.permutation(array)
            cut = len(shuffled) // 2
            left[i], right[i] = np.median(shuffled[:cut]), np.median(shuffled[cut:])
        rho = [float(spearmanr(left[idx], right[idx]).statistic) for idx in groups
               if np.ptp(left[idx]) > 0 and np.ptp(right[idx]) > 0]
        defined_counts.append(len(rho))
        if rho:
            r = float(np.mean(rho))
            correlations.append(r)
            if r > 0:
                corrections.append(2 * r / (1 + r))
    return {**result,
            "mean_split_half_macro_spearman": float(np.mean(correlations)) if correlations else None,
            "mean_spearman_brown_heuristic": float(np.mean(corrections)) if corrections else None,
            "positive_correlation_repetitions": len(corrections),
            "defined_groups_range": [min(defined_counts), max(defined_counts)],
            "random_half_assignment_range": np.quantile(correlations, [.025, .975]).tolist() if correlations else None,
            "range_interpretation": "random-half-assignment variability, not a population confidence interval"}


def recover():
    sys.path.insert(0, str(ROOT))
    from data_processing.build_chembl_binding_dataset import build_activity_query, _table_has_column

    OUT.mkdir(parents=True, exist_ok=True)
    source_dir = ROOT / "dataset/processed/chembl_37/protein_cluster_50_activity_balanced"
    source = pd.read_parquet(source_dir / "all.parquet", columns=[*KEY, "pchembl_value", "measurement_count"])
    source_hash = sha256(source_dir / "all.parquet")
    provenance_path = OUT / "reconstruction.json"
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text())
        if previous["source_sha256"] != source_hash:
            raise RuntimeError("Measurement reconstruction source changed")
        return pd.read_parquet(OUT / "observations.parquet"), pd.read_parquet(OUT / "raw_measurements.parquet")
    original = json.loads((source_dir / "summary.json").read_text())
    filters = original["filters"]
    downloaded = json.loads((ROOT / "dataset/raw/chembl_37/download_verified.json").read_text())
    if downloaded["sha256"] != original["source"]["sqlite_archive_sha256"]:
        raise RuntimeError("Downloaded release differs from source provenance")
    database = ROOT / "dataset/raw/chembl_37/chembl_37.db"
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    try:
        query, parameters = build_activity_query(
            exact_only=filters["exact_standard_relation_only"],
            include_variants=not filters["exclude_variants"],
            include_questionable=not filters["exclude_questionable_data_validity"],
            min_confidence_score=filters["min_confidence_score"],
            has_variant_id=_table_has_column(connection, "assays", "variant_id"))
        # Retain the original FROM/JOIN/WHERE logic, selecting only the audit columns.
        query = """SELECT act.activity_id, a.chembl_id AS assay_id,
            td.chembl_id AS target_chembl_id, md_parent.chembl_id AS compound_id,
            act.pchembl_value, act.standard_type AS activity_type
            FROM activities act""" + query.split("FROM activities act", 1)[1]
        (OUT / "extraction.sql").write_text(query)
        chunks = []
        for chunk in pd.read_sql_query(query, connection, params=parameters, chunksize=250_000):
            chunks.append(chunk)
            print(f"Recovered {sum(len(c) for c in chunks)} measurements", flush=True)
        raw = pd.concat(chunks, ignore_index=True)
        raw["pchembl_value"] = pd.to_numeric(raw.pchembl_value, errors="raise")
        if not np.isfinite(raw.pchembl_value.to_numpy()).all():
            raise RuntimeError("Nonfinite raw pChEMBL measurement")
    finally:
        connection.close()
    observations, matched = reconstruct(raw, source)
    observations.to_parquet(OUT / "observations.parquet", index=False)
    matched.to_parquet(OUT / "raw_measurements.parquet", index=False)
    observations[~observations.verified].to_parquet(OUT / "reconstruction_mismatches.parquet", index=False)
    write_json(provenance_path, {"source_sha256": source_hash, "archive_sha256": downloaded["sha256"],
        "filters": filters, "source_rows": len(source), "query_measurements": len(raw), "matched_measurements": len(matched),
        "verified_source_rows": int(observations.verified.sum()), "unverified_source_rows": int((~observations.verified).sum()),
        "count_mismatches": int((~observations.count_matches).sum()), "median_mismatches": int((~observations.median_matches).sum())})
    return observations, matched


def main():
    observations, raw = recover()
    if not observations.verified.all():
        raise RuntimeError("Reconstructed measurements disagree with source observations; inspect mismatches before noise inference")
    result = {}
    for regime, directory in (("A", ROOT / "dataset/processed/chembl_37/target_aware_random_assay_activity_balanced"),
                               ("B", DATA / "scaffold_seed42")):
        result[regime] = {}
        for partition in ("train", "val", "test"):
            marker = OUT / f"noise_{regime}_{partition}.json"
            if marker.exists():
                result[regime][partition] = json.loads(marker.read_text())
                continue
            keys = pd.read_parquet(directory / f"{partition}.parquet", columns=KEY)
            subset = observations.merge(keys, on=KEY, how="inner", validate="one_to_one")
            if len(subset) != len(keys):
                raise RuntimeError("Split observations missing from measurement reconstruction")
            print(f"Split-half reliability: {regime}/{partition}", flush=True)
            current = {"noise_scale": noise_scale(subset), "split_half": split_half(subset, raw)}
            write_json(marker, current)
            result[regime][partition] = current
    write_json(OUT / "noise_summary.json", result)
    print("Raw-measurement and replicate reliability audit complete", flush=True)


if __name__ == "__main__":
    main()
