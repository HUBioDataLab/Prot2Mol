"""Assay-preserving, scaffold-disjoint train/validation/test split.

Splitting unit
--------------
Neither the molecule nor the assay, but the connected component of the bipartite
graph

    assay node  <->  Bemis-Murcko scaffold node

with an edge whenever an assay contains a molecule of that scaffold. Assigning a
whole component to one split makes two constraints hold *by construction*:

    * every assay lies entirely in one split   (assay integrity)
    * every scaffold lies entirely in one split (scaffold disjointness)

Assay integrity matters because within-assay ranking and contrastive objectives
need an assay's full ligand set in the same split.

Measured structure of ChEMBL-37 (1,957,398 rows, 164,312 assays, 339,348
scaffolds): the graph has one giant component holding ~85% of rows and ~93% of
targets, and ~8.6k small components holding the rest. The split therefore is not
free -- the giant is train and the remainder becomes validation/test.

Assignment rules
----------------
1. the largest component (by rows) becomes train;
2. any component holding a target that never appears in train also joins train,
   so every validation/test target has training evidence;
3. the remaining components are dealt to validation and test by a seeded greedy
   that balances row counts *and* positive-label rate simultaneously.

Molecules with no Bemis-Murcko scaffold (acyclic) share one pooled scaffold node,
so they all land in the same split. Molecules RDKit cannot parse keep a private
node each and are counted separately in the manifest; they are never dropped.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

ACYCLIC_NODE = "@@ACYCLIC@@"
UNPARSEABLE_PREFIX = "@@unparseable@@"
REQUIRED_COLUMNS = ("assay_id", "target_chembl_id", "compound_id", "smiles", "binary_label")


# --------------------------------------------------------------------------- #
# scaffolds
# --------------------------------------------------------------------------- #
def _scaffold_worker(smiles: str):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return None


def compute_scaffold_nodes(smiles: Sequence[str], num_proc: int = 8) -> Tuple[pd.Series, Dict[str, int]]:
    """Map unique SMILES -> graph node id, pooling acyclic molecules."""
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    unique = pd.unique(pd.Series(smiles))
    if num_proc > 1:
        from multiprocessing import Pool

        with Pool(num_proc) as pool:
            scaffolds = pool.map(_scaffold_worker, list(unique), chunksize=2000)
    else:
        scaffolds = [_scaffold_worker(s) for s in unique]

    nodes: List[str] = []
    unparseable = acyclic = 0
    for smi, scaffold in zip(unique, scaffolds):
        if scaffold is None:
            nodes.append(f"{UNPARSEABLE_PREFIX}{smi}")
            unparseable += 1
        elif scaffold == "":
            nodes.append(ACYCLIC_NODE)
            acyclic += 1
        else:
            nodes.append(scaffold)
    mapping = pd.Series(nodes, index=unique)
    counts = {
        "unique_molecules": int(len(unique)),
        "unparseable_molecules": int(unparseable),
        "acyclic_molecules": int(acyclic),
        "distinct_scaffolds": int(pd.Series(nodes).nunique()),
    }
    return mapping, counts


# --------------------------------------------------------------------------- #
# graph
# --------------------------------------------------------------------------- #
def connected_components(assay_ids: pd.Series, scaffold_nodes: pd.Series) -> pd.Series:
    """Return a Series mapping assay_id -> component id (union-find, iterative)."""
    edges = pd.DataFrame({"a": assay_ids.to_numpy(), "s": scaffold_nodes.to_numpy()}).drop_duplicates()
    a_code, a_uniq = pd.factorize(edges["a"], sort=False)
    s_code, s_uniq = pd.factorize(edges["s"], sort=False)
    n_assays = len(a_uniq)
    parent = np.arange(n_assays + len(s_uniq), dtype=np.int64)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for u, v in zip(a_code, s_code + n_assays):
        ru, rv = find(int(u)), find(int(v))
        if ru != rv:
            parent[ru] = rv

    roots = np.array([find(i) for i in range(n_assays)])
    _, component = np.unique(roots, return_inverse=True)
    return pd.Series(component, index=a_uniq, name="component")


# --------------------------------------------------------------------------- #
# assignment
# --------------------------------------------------------------------------- #
def _deal_val_test(stats: pd.DataFrame, seed: int, restarts: int) -> Dict[int, str]:
    """Greedy that balances rows and positive rate between validation and test."""
    rng = np.random.default_rng(seed)
    total_rows = float(stats["rows"].sum())
    best: Tuple[float, Dict[int, str]] | None = None

    orders = [stats.sort_values("rows", ascending=False).index.to_numpy()]
    for _ in range(max(0, restarts - 1)):
        order = stats.sort_values("rows", ascending=False).index.to_numpy().copy()
        rng.shuffle(order)
        orders.append(order)

    for order in orders:
        rows = {"val": 0.0, "test": 0.0}
        positives = {"val": 0.0, "test": 0.0}
        assignment: Dict[int, str] = {}
        for component in order:
            entry = stats.loc[component]
            best_split, best_cost = None, None
            for split in ("val", "test"):
                r = dict(rows)
                p = dict(positives)
                r[split] += float(entry["rows"])
                p[split] += float(entry["positives"])
                size_gap = abs(r["val"] - r["test"]) / total_rows
                rate_gap = abs(
                    (p["val"] / r["val"] if r["val"] else 0.0)
                    - (p["test"] / r["test"] if r["test"] else 0.0)
                )
                cost = size_gap + rate_gap
                if best_cost is None or cost < best_cost:
                    best_split, best_cost = split, cost
            assignment[int(component)] = best_split
            rows[best_split] += float(entry["rows"])
            positives[best_split] += float(entry["positives"])

        size_gap = abs(rows["val"] - rows["test"]) / total_rows
        rate_gap = abs(
            (positives["val"] / rows["val"] if rows["val"] else 0.0)
            - (positives["test"] / rows["test"] if rows["test"] else 0.0)
        )
        cost = size_gap + rate_gap
        if best is None or cost < best[0]:
            best = (cost, assignment)

    assert best is not None
    return best[1]


def split_scaffold_components(
    frame: pd.DataFrame,
    *,
    seed: int = 42,
    num_proc: int = 8,
    balance_restarts: int = 64,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"source frame is missing columns: {missing}")
    if frame.empty:
        raise ValueError("source frame must not be empty")
    if frame[["assay_id", "target_chembl_id", "smiles"]].isna().any().any():
        raise ValueError("assay_id, target_chembl_id and smiles must not be null")

    work = frame[list(REQUIRED_COLUMNS)].copy()
    scaffold_map, scaffold_counts = compute_scaffold_nodes(work["smiles"], num_proc=num_proc)
    work["scaffold_node"] = work["smiles"].map(scaffold_map)

    assay_component = connected_components(work["assay_id"], work["scaffold_node"])
    work["component"] = work["assay_id"].map(assay_component).astype(int)

    stats = work.groupby("component").agg(
        rows=("assay_id", "size"),
        positives=("binary_label", "sum"),
    )
    giant = int(stats["rows"].idxmax())

    assignment: Dict[int, str] = {giant: "train"}
    train_targets = set(work.loc[work["component"] == giant, "target_chembl_id"])
    # rule 2: keep every validation/test target represented in train
    others = work[work["component"] != giant]
    rescued = sorted(
        int(c)
        for c, sub in others.groupby("component")
        if not set(sub["target_chembl_id"]).issubset(train_targets)
    )
    for component in rescued:
        assignment[component] = "train"

    pool = stats.drop(index=[giant, *rescued], errors="ignore")
    assignment.update(_deal_val_test(pool, seed=seed, restarts=balance_restarts))

    work["split"] = work["component"].map(assignment)
    frame = frame.copy()
    frame["split"] = work["split"].to_numpy()
    parts = {name: frame[frame["split"] == name].drop(columns=["split"]).reset_index(drop=True)
             for name in ("train", "val", "test")}

    summary = _summarise(work, parts, scaffold_counts, giant, rescued, seed, balance_restarts)
    return parts["train"], parts["val"], parts["test"], summary


def _summarise(work, parts, scaffold_counts, giant, rescued, seed, restarts) -> Dict[str, Any]:
    by_split = {name: work[work["split"] == name] for name in ("train", "val", "test")}
    total = len(work)

    assay_splits = work.groupby("assay_id")["split"].nunique()
    scaffold_splits = work.groupby("scaffold_node")["split"].nunique()
    train_targets = set(by_split["train"]["target_chembl_id"])

    leakage: Dict[str, Any] = {
        "assays_spanning_multiple_splits": int((assay_splits > 1).sum()),
        "scaffolds_spanning_multiple_splits": int((scaffold_splits > 1).sum()),
        "assay_integrity_holds": bool((assay_splits == 1).all()),
        "scaffold_disjointness_holds": bool((scaffold_splits == 1).all()),
    }
    for name in ("val", "test"):
        unseen = set(by_split[name]["target_chembl_id"]) - train_targets
        leakage[f"{name}_targets_absent_from_train"] = int(len(unseen))
        leakage[f"{name}_rows_with_target_absent_from_train"] = int(
            by_split[name]["target_chembl_id"].isin(unseen).sum()
        )

    split_stats = {}
    for name, part in by_split.items():
        split_stats[name] = {
            "rows": int(len(part)),
            "row_fraction": float(len(part) / total) if total else 0.0,
            "assays": int(part["assay_id"].nunique()),
            "molecules": int(part["compound_id"].nunique()),
            "scaffolds": int(part["scaffold_node"].nunique()),
            "targets": int(part["target_chembl_id"].nunique()),
            "components": int(part["component"].nunique()),
            "positive_rate": float(part["binary_label"].mean()) if len(part) else 0.0,
        }

    return {
        "seed": int(seed),
        "balance_restarts": int(restarts),
        "scaffold_definition": "bemis_murcko_no_chirality",
        "acyclic_policy": "pooled_single_node",
        "source_rows": int(total),
        "source_assays": int(work["assay_id"].nunique()),
        "source_targets": int(work["target_chembl_id"].nunique()),
        "source_molecules": int(work["compound_id"].nunique()),
        "scaffold_counts": scaffold_counts,
        "components_total": int(work["component"].nunique()),
        "giant_component_id": int(giant),
        "giant_component_rows": int((work["component"] == giant).sum()),
        "components_forced_to_train_for_target_coverage": int(len(rescued)),
        "split_stats": split_stats,
        "leakage_checks": leakage,
    }


def write_scaffold_component_split(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    num_proc: int = 8,
    balance_restarts: int = 64,
    overwrite: bool = False,
) -> Dict[str, Any]:
    source = Path(source_path).resolve()
    destination = Path(output_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source parquet not found: {source}")

    outputs = {name: destination / f"{name}.parquet" for name in ("train", "val", "test")}
    outputs["manifest"] = destination / "scaffold_component_split.json"
    existing = [p for p in outputs.values() if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "scaffold-component split outputs already exist; pass --overwrite to replace: "
            + ", ".join(str(p) for p in existing)
        )

    frame = pd.read_parquet(source)
    train, val, test, summary = split_scaffold_components(
        frame, seed=seed, num_proc=num_proc, balance_restarts=balance_restarts
    )
    if not summary["leakage_checks"]["assay_integrity_holds"]:
        raise AssertionError("assay integrity violated; refusing to write")
    if not summary["leakage_checks"]["scaffold_disjointness_holds"]:
        raise AssertionError("scaffold disjointness violated; refusing to write")

    destination.mkdir(parents=True, exist_ok=True)
    summary.update({
        "source_parquet": str(source),
        "train_parquet": str(outputs["train"]),
        "val_parquet": str(outputs["val"]),
        "test_parquet": str(outputs["test"]),
    })
    for name, part in (("train", train), ("val", val), ("test", test)):
        temporary = outputs[name].with_name(outputs[name].name + ".tmp")
        part.to_parquet(temporary, index=False)
        os.replace(temporary, outputs[name])
    outputs["manifest"].write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
