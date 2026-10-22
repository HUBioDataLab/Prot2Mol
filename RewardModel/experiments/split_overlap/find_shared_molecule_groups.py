"""Find protein groups that measure the SAME molecules with DIFFERENT rankings.

The test this enables: give the model one molecule set and two different proteins
that both really measured it. If the ground-truth orderings genuinely disagree,
a protein-blind model must fail at least one of them, because it can only emit
one ordering for a fixed molecule set.

A candidate is two (or more) assays that
  * belong to DIFFERENT targets,
  * sit in the SAME split part (both train, or both val),
  * share at least `--min-common` molecules,
  * each span at least `--min-span` pChEMBL units over the shared molecules
    (otherwise that protein has no real ordering to reproduce),
  * and disagree: Spearman between the two ground-truth orderings is low.

`gt_spearman` is the whole point. +1 means the two proteins rank the shared
molecules identically, so the test cannot discriminate and the candidate is
useless however many molecules it shares. Values near 0 or negative are what
make a decisive experiment.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

ROOT = "/home/furkan/Prot2Mol_Project/Prot2Mol"
SPLITS = {
    "scaffold": f"{ROOT}/dataset/processed/chembl_37/scaffold_component_split_median",
    "random":   f"{ROOT}/dataset/processed/chembl_37/target_aware_random_assay_activity_balanced_median",
}


def candidates(df, min_common, min_span, chunk=1000):
    nt = df.groupby("molecule_chembl_id").target_chembl_id.nunique()
    multi = set(nt[nt >= 2].index)
    d = df[df.molecule_chembl_id.isin(multi)]
    cnt = d.groupby("assay_group_id").molecule_chembl_id.nunique()
    keep = set(cnt[cnt >= min_common].index)
    d = d[d.assay_group_id.isin(keep)]
    if d.empty:
        return pd.DataFrame(), {}, {}, {}

    assays = np.array(sorted(d.assay_group_id.unique()))
    mols = np.array(sorted(d.molecule_chembl_id.unique()))
    ai = {a: i for i, a in enumerate(assays)}
    mi = {m: i for i, m in enumerate(mols)}
    rows = d.assay_group_id.map(ai).to_numpy()
    cols = d.molecule_chembl_id.map(mi).to_numpy()
    M = sparse.csr_matrix((np.ones(len(d), np.int32), (rows, cols)),
                          shape=(len(assays), len(mols)))
    target_of = d.groupby("assay_group_id").target_chembl_id.first()
    tgt = np.array([target_of[a] for a in assays])

    # per-assay molecule -> pchembl, for the ground-truth orderings
    pch = {a: dict(zip(g.molecule_chembl_id, g.pchembl_value))
           for a, g in d.groupby("assay_group_id")}
    molset = {a: set(v) for a, v in pch.items()}

    out = []
    Mt = M.T.tocsr()
    for s in range(0, len(assays), chunk):
        e = min(s + chunk, len(assays))
        inter = (M[s:e] @ Mt).toarray()
        gidx = np.arange(s, e)[:, None]                 # global row index
        jidx = np.arange(len(assays))[None, :]
        inter[jidx <= gidx] = 0                         # upper triangle only
        ii, jj = np.nonzero(inter >= min_common)
        for i, j in zip(ii, jj):
            gi = s + i
            if tgt[gi] == tgt[j]:
                continue
            a, b = assays[gi], assays[j]
            common = sorted(molset[a] & molset[b])
            if len(common) < min_common:
                continue
            ya = np.array([pch[a][m] for m in common])
            yb = np.array([pch[b][m] for m in common])
            if ya.max() - ya.min() < min_span or yb.max() - yb.min() < min_span:
                continue
            if len(set(ya)) < 2 or len(set(yb)) < 2:
                continue
            rho = spearmanr(ya, yb).statistic
            if not np.isfinite(rho):
                continue
            out.append(dict(assay_a=a, assay_b=b, target_a=tgt[gi], target_b=tgt[j],
                            n_common=len(common), gt_spearman=float(rho),
                            span_a=float(ya.max()-ya.min()), span_b=float(yb.max()-yb.min()),
                            common=common))
    return pd.DataFrame(out), pch, molset, dict(target_of)


def extend_groups(pairs, pch, molset, min_common, target_of):
    """How many distinct targets have an assay containing a given shared set?"""
    groups = []
    seen = set()
    for r in pairs.sort_values(["n_common", "gt_spearman"], ascending=[False, True]).itertuples():
        key = tuple(r.common)
        if key in seen:
            continue
        seen.add(key)
        members = []
        for a, ms in molset.items():
            if set(key) <= ms:
                ya = np.array([pch[a][m] for m in key])
                if ya.max() - ya.min() >= 0.5 and len(set(ya)) >= 2:
                    members.append(a)
        by_target = {}
        for a in members:
            by_target.setdefault(target_of[a], a)       # one assay per target
        if len(by_target) >= 2:
            groups.append(dict(molecules=list(key), n_molecules=len(key),
                               n_targets=len(by_target),
                               targets=sorted(by_target),
                               assays=[by_target[t] for t in sorted(by_target)]))
        if len(groups) >= 40:
            break
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-common", type=int, default=6)
    ap.add_argument("--min-span", type=float, default=0.5)
    ap.add_argument("--max-gt-spearman", type=float, default=0.5,
                    help="keep pairs whose ground truths disagree at least this much")
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    allres = {}
    for split, path in SPLITS.items():
        for part in ("train", "val"):
            df = pd.read_parquet(f"{path}/{part}.parquet",
                                 columns=["assay_group_id","target_chembl_id",
                                          "molecule_chembl_id","pchembl_value"])
            pairs, pch, molset, target_of_map = candidates(df, args.min_common, args.min_span)
            tag = f"{split}_{part}"
            if pairs.empty:
                allres[tag] = dict(n_pairs=0); print(f"{tag}: none"); continue
            disc = pairs[pairs.gt_spearman <= args.max_gt_spearman]
            allres[tag] = dict(
                n_pairs=int(len(pairs)),
                n_discriminating=int(len(disc)),
                n_targets=int(len(set(pairs.target_a) | set(pairs.target_b))),
                max_common=int(pairs.n_common.max()),
                groups=extend_groups(disc if len(disc) else pairs, pch, molset, args.min_common, target_of_map),
                top=disc.sort_values(["gt_spearman","n_common"], ascending=[True, False])
                        .head(25).drop(columns=["common"]).to_dict("records"),
                top_big=pairs.sort_values(["n_common","gt_spearman"], ascending=[False, True])
                        .head(25).drop(columns=["common"]).to_dict("records"),
            )
            pairs.drop(columns=["common"]).to_csv(
                os.path.join(args.out_dir, f"shared_pairs_{tag}.csv"), index=False)
            print(f"{tag}: {len(pairs)} pairs (>= {args.min_common} common mols), "
                  f"{len(disc)} with gt_spearman <= {args.max_gt_spearman}, "
                  f"max common {pairs.n_common.max()}", flush=True)
    with open(os.path.join(args.out_dir, "shared_groups.json"), "w") as f:
        json.dump(allres, f, indent=2, default=str)
    print("wrote shared_groups.json")


if __name__ == "__main__":
    main()
