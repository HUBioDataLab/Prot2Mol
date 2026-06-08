#!/usr/bin/env python3
"""Prepare a Boltz-2-style protein-ligand affinity dataset.

The official Boltz-2 affinity training set was not released. This script
implements the public paper's curation rules for our own ChEMBL/BindingDB-like
exports and writes:

1. A full Boltz-2-style affinity CSV with assay metadata and log10(uM) labels.
2. A Prot2Mol-compatible CSV with Target_FASTA, Compound_SELFIES and pChEMBL.
3. A summary JSON documenting filters and retained counts.
"""

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from time import perf_counter
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import selfies as sf
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

RDLogger.DisableLog("rdApp.*")

ACTIVITY_TYPES = {"KI", "KD", "IC50", "XC50", "EC50", "AC50"}
ALLOWED_ASSAY_TYPES = {"B", "F", "BINDING", "FUNCTIONAL", "BIOCHEMICAL"}
EXACT_QUALIFIERS = {"=", ""}
SUPPORTED_UNITS_TO_UM = {
    "M": 1_000_000.0,
    "MM": 1_000.0,
    "UM": 1.0,
    "µM": 1.0,
    "ΜM": 1.0,
    "NM": 0.001,
    "PM": 0.000001,
}

BOLTZ2_OUTPUT_COLUMNS = [
    "source",
    "supervision",
    "task_type",
    "assay_id",
    "assay_group_id",
    "target_id",
    "target_chembl_id",
    "protein_sequence",
    "protein_cluster_90",
    "mutation_info",
    "compound_id",
    "standardized_smiles",
    "compound_selfies",
    "activity_type",
    "activity_value_uM",
    "activity_qualifier",
    "y_log10_uM",
    "pchembl_value",
    "binary_label",
    "is_censored",
    "heavy_atom_count",
    "is_pains",
    "assay_size",
    "assay_exact_size",
    "assay_activity_std",
    "assay_activity_iqr",
    "split",
]

PROT2MOL_OUTPUT_COLUMNS = [
    "Target_FASTA",
    "Target_CHEMBL_ID",
    "Target_ID",
    "AID",
    "Compound_SELFIES",
    "Compound_SMILES",
    "pchembl_value_Median",
    "Compound_CID",
]

STAGED_COLUMNS = [
    "source",
    "assay_id",
    "target_id",
    "target_chembl_id",
    "protein_sequence",
    "mutation_info",
    "compound_id",
    "standardized_smiles",
    "compound_selfies",
    "activity_type",
    "activity_value_uM",
    "activity_qualifier",
    "heavy_atom_count",
    "is_pains",
]


@dataclass(frozen=True)
class Boltz2AffinityConfig:
    min_assay_size: int = 10
    min_unique_values: int = 10
    min_activity_std: float = 0.25
    min_unique_fraction: float = 0.20
    heavy_atom_limit: int = 50
    include_censored: bool = False
    keep_bindingdb_chembl_overlap: bool = False
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    split_seed: int = 42
    rdkit_workers: int = 1
    skip_rdkit_standardization: bool = False


_WORKER_PAINS_CATALOG = None


def _log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _first_present(columns: Iterable[str], aliases: Iterable[str]) -> Optional[str]:
    lower_to_original = {c.lower(): c for c in columns}
    for alias in aliases:
        found = lower_to_original.get(alias.lower())
        if found is not None:
            return found
    return None


def _require_column(df: pd.DataFrame, aliases: Iterable[str], logical_name: str) -> str:
    column = _first_present(df.columns, aliases)
    if column is None:
        raise ValueError(
            f"Missing required column for {logical_name}. "
            f"Accepted aliases: {', '.join(aliases)}"
        )
    return column


def _optional_column(df: pd.DataFrame, aliases: Iterable[str]) -> Optional[str]:
    return _first_present(df.columns, aliases)


def _coalesce_columns(df: pd.DataFrame, aliases: Iterable[str], default="") -> pd.Series:
    result = pd.Series("", index=df.index, dtype=object)
    for alias in aliases:
        column = _optional_column(df, (alias,))
        if column is None:
            continue
        values = df[column]
        result_empty = result.isna() | (result.map(_normalize_string) == "")
        values_present = values.notna() & (values.map(_normalize_string) != "")
        result.loc[result_empty & values_present] = values.loc[result_empty & values_present]
    result_empty = result.isna() | (result.map(_normalize_string) == "")
    result.loc[result_empty] = default
    return result


def _normalize_string(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _normalize_sequence(value) -> str:
    return "".join(_normalize_string(value).split()).upper()


def _normalize_qualifier(value) -> str:
    qualifier = _normalize_string(value).replace("'", "").replace('"', "")
    if qualifier in {"", "=", ">", "<", ">=", "<="}:
        return qualifier
    if qualifier.startswith(">"):
        return ">"
    if qualifier.startswith("<"):
        return "<"
    return qualifier


def _normalize_units(value) -> str:
    units = _normalize_string(value)
    units = units.replace("μ", "µ")
    return units.upper().replace(" ", "")


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = _normalize_string(value).lower()
    return normalized in {"1", "true", "t", "yes", "y"}


def _source_mask(df: pd.DataFrame, source_name: str) -> pd.Series:
    return df["source"].astype(str).str.lower().str.contains(source_name.lower(), na=False)


def _convert_value_to_um(row: pd.Series, value_col: str, unit_col: Optional[str]) -> float:
    value = pd.to_numeric(row[value_col], errors="coerce")
    if pd.isna(value):
        return np.nan
    if unit_col is None:
        return float(value)

    units = _normalize_units(row[unit_col])
    factor = SUPPORTED_UNITS_TO_UM.get(units)
    if factor is None:
        return np.nan
    return float(value) * factor


def _coalesce_activity_value_um(raw_df: pd.DataFrame) -> pd.Series:
    value_um_col = _optional_column(raw_df, ("activity_value_uM",))
    if value_um_col is not None:
        result = pd.to_numeric(raw_df[value_um_col], errors="coerce")
    else:
        result = pd.Series(np.nan, index=raw_df.index, dtype=float)

    raw_value = pd.to_numeric(
        _coalesce_columns(raw_df, ("standard_value", "activity_value", "value"), default=np.nan),
        errors="coerce",
    )
    raw_units = _coalesce_columns(raw_df, ("activity_units", "standard_units", "units"), default="uM")
    needs_conversion = result.isna() & raw_value.notna()

    converted = []
    for value, units in zip(raw_value.loc[needs_conversion], raw_units.loc[needs_conversion]):
        factor = SUPPORTED_UNITS_TO_UM.get(_normalize_units(units))
        converted.append(np.nan if factor is None else float(value) * factor)
    result.loc[needs_conversion] = converted
    return result


def _build_pains_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog(params)


def _init_molecule_worker() -> None:
    global _WORKER_PAINS_CATALOG
    _WORKER_PAINS_CATALOG = _build_pains_catalog()


def _standardize_molecule(smiles: str, pains_catalog: FilterCatalog):
    mol = Chem.MolFromSmiles(_normalize_string(smiles))
    if mol is None:
        return None, np.nan, True
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None, np.nan, True

    canonical = Chem.MolToSmiles(mol, canonical=True)
    heavy_atoms = int(Descriptors.HeavyAtomCount(mol))
    is_pains = bool(pains_catalog.HasMatch(mol))
    return canonical, heavy_atoms, is_pains


def _standardize_molecule_worker(smiles: str):
    catalog = _WORKER_PAINS_CATALOG
    if catalog is None:
        catalog = _build_pains_catalog()
    return smiles, _standardize_molecule(smiles, catalog)


def _to_selfies(smiles: str) -> Optional[str]:
    try:
        return sf.encoder(smiles, strict=False)
    except Exception:
        return None


def _stable_hash(prefix: str, value: str, n: int = 16) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:n]
    return f"{prefix}_{digest}"


def _read_cluster_map(path: Optional[str]) -> dict[str, str]:
    if not path:
        return {}
    sep = "\t" if path.endswith((".tsv", ".txt")) else ","
    cluster_df = pd.read_csv(path, sep=sep)
    sequence_col = _require_column(
        cluster_df,
        ("protein_sequence", "Target_FASTA", "sequence", "Sequence"),
        "cluster-map protein sequence",
    )
    cluster_col = _require_column(
        cluster_df,
        ("protein_cluster_90", "cluster", "cluster_id", "representative"),
        "cluster-map cluster id",
    )
    return {
        _normalize_sequence(row[sequence_col]): _normalize_string(row[cluster_col])
        for _, row in cluster_df.iterrows()
        if _normalize_sequence(row[sequence_col])
    }


def _assign_clusters(df: pd.DataFrame, cluster_map: dict[str, str]) -> pd.Series:
    clusters = []
    for sequence in df["protein_sequence"]:
        if sequence in cluster_map:
            clusters.append(cluster_map[sequence])
        else:
            clusters.append(_stable_hash("seq90", sequence))
    return pd.Series(clusters, index=df.index)


def _assign_splits(df: pd.DataFrame, config: Boltz2AffinityConfig) -> pd.Series:
    clusters = sorted(df["protein_cluster_90"].dropna().unique())
    rng = np.random.RandomState(config.split_seed)
    rng.shuffle(clusters)

    n_clusters = len(clusters)
    n_test = int(round(n_clusters * config.test_ratio))
    n_val = int(round(n_clusters * config.val_ratio))
    if config.test_ratio > 0 and n_clusters > 1:
        n_test = max(1, n_test)
    if config.val_ratio > 0 and n_clusters - n_test > 1:
        n_val = max(1, n_val)
    n_test = min(n_test, n_clusters)
    n_val = min(n_val, max(0, n_clusters - n_test))

    test_clusters = set(clusters[:n_test])
    val_clusters = set(clusters[n_test : n_test + n_val])

    def split_for(cluster: str) -> str:
        if cluster in test_clusters:
            return "test"
        if cluster in val_clusters:
            return "val"
        return "train"

    return df["protein_cluster_90"].map(split_for)


def _normalize_input(raw_df: pd.DataFrame) -> pd.DataFrame:
    _require_column(raw_df, ("target_id", "Target_ID", "target_chembl_id"), "target id")
    _require_column(raw_df, ("protein_sequence", "Target_FASTA", "Sequence", "target_sequence"), "protein sequence")
    _require_column(raw_df, ("standardized_smiles", "smiles", "SMILES", "Compound_SMILES"), "SMILES")
    _require_column(raw_df, ("activity_type", "standard_type", "type"), "activity type")
    _require_column(raw_df, ("activity_value_uM", "standard_value", "activity_value", "value"), "activity value")

    df = pd.DataFrame(
        {
            "source": _coalesce_columns(raw_df, ("source", "Source"), default="unknown").map(_normalize_string),
            "assay_id": _coalesce_columns(raw_df, ("assay_id", "AID", "assay_chembl_id", "doi", "DOI")).map(_normalize_string),
            "target_id": _coalesce_columns(raw_df, ("target_id", "Target_ID", "target_chembl_id")).map(_normalize_string),
            "target_chembl_id": _coalesce_columns(raw_df, ("target_chembl_id", "Target_CHEMBL_ID")).map(_normalize_string),
            "protein_sequence": _coalesce_columns(raw_df, ("protein_sequence", "Target_FASTA", "Sequence", "target_sequence")).map(_normalize_sequence),
            "raw_smiles": _coalesce_columns(raw_df, ("standardized_smiles", "smiles", "SMILES", "Compound_SMILES")).map(_normalize_string),
            "activity_type": _coalesce_columns(raw_df, ("activity_type", "standard_type", "type")).map(lambda v: _normalize_string(v).upper()),
            "activity_qualifier": _coalesce_columns(raw_df, ("activity_qualifier", "standard_relation", "relation", "qualifier"), default="=").map(_normalize_qualifier),
            "mutation_info": _coalesce_columns(raw_df, ("mutation_info", "mutation", "variant")).map(_normalize_string),
            "compound_id": _coalesce_columns(raw_df, ("compound_id", "Compound_CID", "molecule_chembl_id", "cid")).map(_normalize_string),
        }
    )
    df["activity_value_uM"] = _coalesce_activity_value_um(raw_df)
    df["assay_id"] = df["assay_id"].where(df["assay_id"].ne(""), df["source"] + "_" + df["target_id"])

    for optional_name, aliases in {
        "confidence_score": ("confidence_score", "target_confidence_score"),
        "target_type": ("target_type",),
        "assay_category": ("assay_category", "assay_type", "assay_class"),
        "source_unreliable": ("source_unreliable", "unreliable", "src_unreliable"),
        "num_protein_chains": ("num_protein_chains", "protein_chain_count", "chains"),
    }.items():
        column = _optional_column(raw_df, aliases)
        if column is not None:
            df[optional_name] = raw_df[column]

    return df


def _apply_source_filters(df: pd.DataFrame) -> pd.DataFrame:
    keep = pd.Series(True, index=df.index)

    chembl = _source_mask(df, "chembl")
    if "confidence_score" in df:
        confidence = pd.to_numeric(df["confidence_score"], errors="coerce")
        keep &= (~chembl) | (confidence == 9)
    if "target_type" in df:
        keep &= (~chembl) | (df["target_type"].astype(str).str.upper().str.strip() == "SINGLE PROTEIN")
    if "assay_category" in df:
        assay_type = df["assay_category"].astype(str).str.upper().str.strip()
        keep &= (~chembl) | assay_type.isin(ALLOWED_ASSAY_TYPES)

    bindingdb = _source_mask(df, "bindingdb")
    if "num_protein_chains" in df:
        chains = pd.to_numeric(df["num_protein_chains"], errors="coerce")
        keep &= (~bindingdb) | chains.isna() | (chains <= 1)

    if "source_unreliable" in df:
        keep &= ~df["source_unreliable"].map(_to_bool)

    return df.loc[keep].copy()


def _apply_activity_filters(df: pd.DataFrame, config: Boltz2AffinityConfig) -> pd.DataFrame:
    keep = df["activity_type"].isin(ACTIVITY_TYPES)
    keep &= np.isfinite(df["activity_value_uM"]) & (df["activity_value_uM"] > 0)
    keep &= df["protein_sequence"].ne("")
    keep &= df["raw_smiles"].ne("")
    keep &= df["activity_qualifier"].isin({"=", ">"})
    if not config.include_censored:
        keep &= df["activity_qualifier"].isin(EXACT_QUALIFIERS)
    return df.loc[keep].copy()


def _apply_molecule_filters_cached(
    df: pd.DataFrame,
    config: Boltz2AffinityConfig,
    pains_catalog: FilterCatalog,
    molecule_cache: dict,
    selfies_cache: dict,
    chunk_label: str = "",
) -> pd.DataFrame:
    df = df.copy()
    if config.skip_rdkit_standardization:
        _log(
            f"{chunk_label}Skipping RDKit standardization/PAINS/heavy-atom filters; "
            "using source SMILES as standardized_smiles."
        )
        df["standardized_smiles"] = df["raw_smiles"]
        df["heavy_atom_count"] = np.nan
        df["is_pains"] = False
        unique_smiles = pd.unique(df["standardized_smiles"])
        uncached_selfies = [smiles for smiles in unique_smiles if smiles not in selfies_cache]
        if uncached_selfies:
            _log(
                f"{chunk_label}SELFIES: encoding {len(uncached_selfies):,} source SMILES "
                f"({len(unique_smiles):,} unique in chunk, cache={len(selfies_cache):,})."
            )
            for smiles in uncached_selfies:
                selfies_cache[smiles] = _to_selfies(smiles)
        df["compound_selfies"] = df["standardized_smiles"].map(selfies_cache)
        df = df.loc[df["compound_selfies"].notna()].copy()
        df["compound_id"] = df.apply(
            lambda row: row["compound_id"] or _stable_hash("cmp", row["standardized_smiles"]),
            axis=1,
        )
        return df

    unique_raw_smiles = pd.unique(df["raw_smiles"])
    uncached_raw = [smiles for smiles in unique_raw_smiles if smiles not in molecule_cache]
    if uncached_raw:
        _log(
            f"{chunk_label}RDKit: standardizing {len(uncached_raw):,} new raw SMILES "
            f"({len(unique_raw_smiles):,} unique in chunk, cache={len(molecule_cache):,})."
        )
        for smiles in uncached_raw:
            molecule_cache[smiles] = _standardize_molecule(smiles, pains_catalog)
    else:
        _log(f"{chunk_label}RDKit: all {len(unique_raw_smiles):,} unique raw SMILES were cached.")

    standardized = df["raw_smiles"].map(molecule_cache)
    df["standardized_smiles"] = standardized.map(lambda entry: entry[0])
    df["heavy_atom_count"] = standardized.map(lambda entry: entry[1])
    df["is_pains"] = standardized.map(lambda entry: entry[2])
    keep = df["standardized_smiles"].notna()
    keep &= ~df["is_pains"]
    keep &= df["heavy_atom_count"] <= config.heavy_atom_limit
    df = df.loc[keep].copy()
    unique_standardized_smiles = pd.unique(df["standardized_smiles"])
    uncached_selfies = [smiles for smiles in unique_standardized_smiles if smiles not in selfies_cache]
    if uncached_selfies:
        _log(
            f"{chunk_label}SELFIES: encoding {len(uncached_selfies):,} new canonical SMILES "
            f"({len(unique_standardized_smiles):,} unique in chunk, cache={len(selfies_cache):,})."
        )
        for smiles in uncached_selfies:
            selfies_cache[smiles] = _to_selfies(smiles)
    else:
        _log(f"{chunk_label}SELFIES: all {len(unique_standardized_smiles):,} canonical SMILES were cached.")

    df["compound_selfies"] = df["standardized_smiles"].map(selfies_cache)
    df = df.loc[df["compound_selfies"].notna()].copy()
    df["compound_id"] = df.apply(
        lambda row: row["compound_id"] or _stable_hash("cmp", row["standardized_smiles"]),
        axis=1,
    )
    return df


def _apply_molecule_filters(df: pd.DataFrame, config: Boltz2AffinityConfig) -> pd.DataFrame:
    return _apply_molecule_filters_cached(
        df=df,
        config=config,
        pains_catalog=_build_pains_catalog(),
        molecule_cache={},
        selfies_cache={},
    )


def _deduplicate_records(df: pd.DataFrame, config: Boltz2AffinityConfig) -> tuple[pd.DataFrame, dict]:
    """Remove exact duplicate measurements and ChEMBL-covered BindingDB rows."""
    before = len(df)
    exact_subset = [
        "source",
        "assay_id",
        "target_id",
        "protein_sequence",
        "standardized_smiles",
        "activity_type",
        "activity_qualifier",
        "activity_value_uM",
    ]
    df = df.drop_duplicates(subset=exact_subset, keep="first").copy()
    exact_duplicates_removed = before - len(df)

    bindingdb_overlap_removed = 0
    if not config.keep_bindingdb_chembl_overlap:
        df["_value_key"] = np.round(np.log10(df["activity_value_uM"].astype(float)), 4)
        overlap_subset = [
            "protein_sequence",
            "standardized_smiles",
            "activity_type",
            "activity_qualifier",
            "_value_key",
        ]
        chembl_keys = df.loc[_source_mask(df, "chembl"), overlap_subset].drop_duplicates()
        chembl_keys["_chembl_overlap"] = True
        bindingdb_mask = _source_mask(df, "bindingdb")
        bindingdb_overlap = df.loc[bindingdb_mask, overlap_subset].merge(
            chembl_keys,
            on=overlap_subset,
            how="left",
        )["_chembl_overlap"].fillna(False)
        overlap_mask = pd.Series(False, index=df.index)
        overlap_mask.loc[df.index[bindingdb_mask]] = bindingdb_overlap.to_numpy(dtype=bool)
        bindingdb_overlap_removed = int(overlap_mask.sum())
        df = df.loc[~overlap_mask].drop(columns=["_value_key"]).copy()

    return df, {
        "exact_duplicates_removed": int(exact_duplicates_removed),
        "bindingdb_chembl_overlap_removed": int(bindingdb_overlap_removed),
    }


def _apply_assay_filters(df: pd.DataFrame, config: Boltz2AffinityConfig) -> pd.DataFrame:
    df = df.copy()
    df["assay_group_id"] = df["source"] + ":" + df["assay_id"] + ":" + df["target_id"]
    df["y_log10_uM"] = np.log10(df["activity_value_uM"].astype(float))
    df["pchembl_value"] = 6.0 - df["y_log10_uM"]
    df["is_censored"] = df["activity_qualifier"] != "="

    exact = df.loc[~df["is_censored"]].copy()
    grouped = exact.groupby("assay_group_id")["y_log10_uM"]
    stats = grouped.agg(
        assay_exact_size="size",
        assay_activity_std=lambda values: float(np.std(values, ddof=0)),
        assay_unique_values=lambda values: int(pd.Series(values).nunique()),
        assay_activity_iqr=lambda values: float(np.percentile(values, 75) - np.percentile(values, 25)),
    )
    stats["assay_unique_fraction"] = stats["assay_unique_values"] / stats["assay_exact_size"].clip(lower=1)
    stats["assay_size"] = df.groupby("assay_group_id").size()

    passing = stats.index[
        (stats["assay_exact_size"] >= config.min_assay_size)
        & (stats["assay_unique_values"] >= config.min_unique_values)
        & (stats["assay_activity_std"] >= config.min_activity_std)
        & (stats["assay_unique_fraction"] >= config.min_unique_fraction)
    ]
    df = df.loc[df["assay_group_id"].isin(passing)].copy()
    if df.empty:
        return df

    df = df.merge(
        stats[
            [
                "assay_size",
                "assay_exact_size",
                "assay_activity_std",
                "assay_activity_iqr",
            ]
        ],
        left_on="assay_group_id",
        right_index=True,
        how="left",
    )
    return df


def _write_outputs(
    df: pd.DataFrame,
    output: Path,
    config: Boltz2AffinityConfig,
    cluster_map_path: Optional[str],
    counts: dict,
    extra_summary: Optional[dict] = None,
) -> dict:
    if df.empty:
        raise ValueError("No rows remain after Boltz-2 affinity curation filters.")

    cluster_map = _read_cluster_map(cluster_map_path)
    df["protein_cluster_90"] = _assign_clusters(df, cluster_map)
    df["split"] = _assign_splits(df, config)
    df["supervision"] = "values"
    df["task_type"] = "optimization"
    df["binary_label"] = ""

    full_df = df[BOLTZ2_OUTPUT_COLUMNS].sort_values(
        ["split", "source", "assay_group_id", "target_id", "compound_id"]
    )
    full_path = output / "boltz2_affinity_full.csv"
    _log(f"Writing full Boltz-2-style CSV: {full_path}")
    full_df.to_csv(full_path, index=False)

    exact_df = full_df.loc[~full_df["is_censored"]].copy()
    prot2mol_df = pd.DataFrame(
        {
            "Target_FASTA": exact_df["protein_sequence"],
            "Target_CHEMBL_ID": exact_df["target_chembl_id"].where(
                exact_df["target_chembl_id"].astype(str).ne(""),
                exact_df["target_id"],
            ),
            "Target_ID": exact_df["target_id"],
            "AID": exact_df["assay_group_id"],
            "Compound_SELFIES": exact_df["compound_selfies"],
            "Compound_SMILES": exact_df["standardized_smiles"],
            "pchembl_value_Median": exact_df["pchembl_value"],
            "Compound_CID": exact_df["compound_id"],
        }
    )
    prot2mol_path = output / "prot2mol_training.csv"
    _log(f"Writing Prot2Mol training CSV: {prot2mol_path}")
    prot2mol_df[PROT2MOL_OUTPUT_COLUMNS].to_csv(prot2mol_path, index=False)

    summary = {
        "config": asdict(config),
        "counts": counts,
        "outputs": {
            "boltz2_affinity_full": str(full_path),
            "prot2mol_training": str(prot2mol_path),
        },
        "cluster_assignment": "provided_cluster_map" if cluster_map else "exact_sequence_hash_fallback",
        "notes": [
            "pchembl_value = 6 - log10(activity_value_uM).",
            "For exact Boltz-2 leakage control, provide a 90% sequence cluster map generated with mmseqs.",
            "Prot2Mol export contains exact-value rows only; censored '>' rows stay only in the full export.",
        ],
        "split_counts": full_df["split"].value_counts().sort_index().to_dict(),
        "assays": int(full_df["assay_group_id"].nunique()),
        "targets": int(full_df["target_id"].nunique()),
        "protein_clusters_90": int(full_df["protein_cluster_90"].nunique()),
        "compounds": int(full_df["compound_id"].nunique()),
    }
    if extra_summary:
        summary.update(extra_summary)
    summary_path = output / "summary.json"
    _log(f"Writing summary JSON: {summary_path}")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    summary["outputs"]["summary"] = str(summary_path)
    return summary


def prepare_dataset(
    input_paths: list[str],
    output_dir: str,
    config: Boltz2AffinityConfig,
    cluster_map_path: Optional[str] = None,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    raw_frames = []
    for path in input_paths:
        sep = "\t" if path.endswith((".tsv", ".txt")) else ","
        raw_frames.append(pd.read_csv(path, sep=sep))
    raw = pd.concat(raw_frames, ignore_index=True)

    counts = {"raw_rows": int(len(raw))}
    print(f"Loaded {counts['raw_rows']:,} raw rows from {len(input_paths)} source file(s).")
    df = _normalize_input(raw)
    df = _apply_source_filters(df)
    counts["after_source_filters"] = int(len(df))
    print(f"After source filters: {counts['after_source_filters']:,} rows.")
    df = _apply_activity_filters(df, config)
    counts["after_activity_filters"] = int(len(df))
    print(f"After activity filters: {counts['after_activity_filters']:,} rows.")
    df = _apply_molecule_filters(df, config)
    counts["after_molecule_filters"] = int(len(df))
    print(f"After molecule filters: {counts['after_molecule_filters']:,} rows.")
    df, dedupe_counts = _deduplicate_records(df, config)
    counts.update(dedupe_counts)
    counts["after_deduplication"] = int(len(df))
    print(
        "After deduplication: "
        f"{counts['after_deduplication']:,} rows "
        f"({dedupe_counts['exact_duplicates_removed']:,} exact duplicates, "
        f"{dedupe_counts['bindingdb_chembl_overlap_removed']:,} BindingDB-ChEMBL overlaps removed)."
    )
    df = _apply_assay_filters(df, config)
    counts["after_assay_filters"] = int(len(df))
    _log(f"After assay filters: {counts['after_assay_filters']:,} rows.")
    return _write_outputs(df, output, config, cluster_map_path, counts)


def prepare_dataset_chunked(
    input_paths: list[str],
    output_dir: str,
    config: Boltz2AffinityConfig,
    cluster_map_path: Optional[str] = None,
    chunk_size: int = 100_000,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    staging_path = output / "_staging_molecule_filtered.csv"
    if staging_path.exists():
        _log(f"Removing stale staging file: {staging_path}")
        staging_path.unlink()

    counts = {
        "raw_rows": 0,
        "after_source_filters": 0,
        "after_activity_filters": 0,
        "after_molecule_filters": 0,
    }
    timings = {
        "chunk_read_normalize_seconds": 0.0,
        "source_filter_seconds": 0.0,
        "activity_filter_seconds": 0.0,
        "molecule_filter_seconds": 0.0,
        "staging_write_seconds": 0.0,
    }
    started = perf_counter()
    wrote_header = False
    pains_catalog = _build_pains_catalog()
    molecule_cache = {}
    selfies_cache = {}

    _log(
        f"Starting chunked curation: {len(input_paths)} input file(s), "
        f"chunk_size={chunk_size:,}, output={output}"
    )

    for path in input_paths:
        sep = "\t" if path.endswith((".tsv", ".txt")) else ","
        _log(f"Opening source file: {path}")
        reader = pd.read_csv(path, sep=sep, chunksize=chunk_size, dtype=str, low_memory=False)
        for chunk_index, raw_chunk in enumerate(reader, start=1):
            chunk_label = f"{Path(path).name} chunk {chunk_index}: "
            chunk_started = perf_counter()
            raw_count = len(raw_chunk)
            counts["raw_rows"] += raw_count

            stage_started = perf_counter()
            df = _normalize_input(raw_chunk)
            timings["chunk_read_normalize_seconds"] += perf_counter() - stage_started

            stage_started = perf_counter()
            df = _apply_source_filters(df)
            source_count = len(df)
            counts["after_source_filters"] += source_count
            timings["source_filter_seconds"] += perf_counter() - stage_started

            stage_started = perf_counter()
            df = _apply_activity_filters(df, config)
            activity_count = len(df)
            counts["after_activity_filters"] += activity_count
            timings["activity_filter_seconds"] += perf_counter() - stage_started

            if df.empty:
                _log(
                    f"{chunk_label}raw={raw_count:,}, source={source_count:,}, "
                    f"activity=0, molecule=0, elapsed={perf_counter() - chunk_started:.1f}s"
                )
                continue

            stage_started = perf_counter()
            df = _apply_molecule_filters_cached(
                df=df,
                config=config,
                pains_catalog=pains_catalog,
                molecule_cache=molecule_cache,
                selfies_cache=selfies_cache,
                chunk_label=chunk_label,
            )
            molecule_count = len(df)
            counts["after_molecule_filters"] += molecule_count
            timings["molecule_filter_seconds"] += perf_counter() - stage_started

            if not df.empty:
                stage_started = perf_counter()
                df[STAGED_COLUMNS].to_csv(
                    staging_path,
                    index=False,
                    mode="a" if wrote_header else "w",
                    header=not wrote_header,
                )
                wrote_header = True
                timings["staging_write_seconds"] += perf_counter() - stage_started

            _log(
                f"{chunk_label}raw={raw_count:,}, source={source_count:,}, "
                f"activity={activity_count:,}, molecule={molecule_count:,}, "
                f"molecule_cache={len(molecule_cache):,}, selfies_cache={len(selfies_cache):,}, "
                f"elapsed={perf_counter() - chunk_started:.1f}s"
            )

    if not wrote_header:
        raise ValueError("No rows remain after chunked source/activity/molecule filters.")

    _log(f"Chunked staging complete: {staging_path}")
    _log(
        "Chunked counts before global filters: "
        f"raw={counts['raw_rows']:,}, source={counts['after_source_filters']:,}, "
        f"activity={counts['after_activity_filters']:,}, molecule={counts['after_molecule_filters']:,}"
    )
    _log(
        "Chunked timings before global filters: "
        + ", ".join(f"{key}={value:.1f}s" for key, value in timings.items())
    )

    stage_started = perf_counter()
    _log("Loading staged molecule-filtered rows for global dedupe and assay filters.")
    df = pd.read_csv(staging_path, low_memory=False)
    timings["staging_read_seconds"] = perf_counter() - stage_started
    _log(f"Loaded {len(df):,} staged rows in {timings['staging_read_seconds']:.1f}s.")

    stage_started = perf_counter()
    df, dedupe_counts = _deduplicate_records(df, config)
    timings["deduplication_seconds"] = perf_counter() - stage_started
    counts.update(dedupe_counts)
    counts["after_deduplication"] = int(len(df))
    _log(
        f"After global dedupe: {counts['after_deduplication']:,} rows "
        f"({dedupe_counts['exact_duplicates_removed']:,} exact duplicates, "
        f"{dedupe_counts['bindingdb_chembl_overlap_removed']:,} BindingDB-ChEMBL overlaps removed) "
        f"in {timings['deduplication_seconds']:.1f}s."
    )

    stage_started = perf_counter()
    df = _apply_assay_filters(df, config)
    timings["assay_filter_seconds"] = perf_counter() - stage_started
    counts["after_assay_filters"] = int(len(df))
    _log(f"After assay filters: {counts['after_assay_filters']:,} rows in {timings['assay_filter_seconds']:.1f}s.")

    timings["total_seconds"] = perf_counter() - started
    return _write_outputs(
        df,
        output,
        config,
        cluster_map_path,
        counts,
        extra_summary={
            "chunked": True,
            "chunk_size": int(chunk_size),
            "staging_path": str(staging_path),
            "timings_seconds": timings,
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Boltz-2-style affinity data from normalized ChEMBL/BindingDB-like CSV/TSV exports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", nargs="+", required=True, help="Input CSV/TSV files.")
    parser.add_argument("--output-dir", required=True, help="Directory for curated outputs.")
    parser.add_argument("--cluster-map", default=None, help="Optional CSV/TSV mapping protein sequences to 90%% clusters.")
    parser.add_argument("--min-assay-size", type=int, default=10)
    parser.add_argument("--min-unique-values", type=int, default=10)
    parser.add_argument("--min-activity-std", type=float, default=0.25)
    parser.add_argument("--min-unique-fraction", type=float, default=0.20)
    parser.add_argument("--heavy-atom-limit", type=int, default=50)
    parser.add_argument("--include-censored", action="store_true", help="Keep '>' lower-bound labels in the full export.")
    parser.add_argument(
        "--keep-bindingdb-chembl-overlap",
        action="store_true",
        help="Do not remove BindingDB rows already covered by ChEMBL after canonicalization.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="When >0, process source rows in chunks and write a molecule-filtered staging CSV before global filters.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = Boltz2AffinityConfig(
        min_assay_size=args.min_assay_size,
        min_unique_values=args.min_unique_values,
        min_activity_std=args.min_activity_std,
        min_unique_fraction=args.min_unique_fraction,
        heavy_atom_limit=args.heavy_atom_limit,
        include_censored=args.include_censored,
        keep_bindingdb_chembl_overlap=args.keep_bindingdb_chembl_overlap,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
    )
    if args.chunk_size and args.chunk_size > 0:
        summary = prepare_dataset_chunked(
            input_paths=args.input,
            output_dir=args.output_dir,
            config=config,
            cluster_map_path=args.cluster_map,
            chunk_size=args.chunk_size,
        )
    else:
        summary = prepare_dataset(
            input_paths=args.input,
            output_dir=args.output_dir,
            config=config,
            cluster_map_path=args.cluster_map,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
