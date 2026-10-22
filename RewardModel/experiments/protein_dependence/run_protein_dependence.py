"""Protein-dependence test for within-assay ligand ranking.

QUESTION
    Does the reward model's ranking of ligands inside one assay depend on which
    protein it is given?

DESIGN
    Inside a ranking group the protein is constant, so a score of the form
    f(protein) + g(molecule) orders ligands identically for every protein. This
    script measures whether that is what the model does, by re-scoring the same
    ligands against proteins the model should consider wrong.

    Arms (every arm re-scores the SAME ligands of the SAME assay):
      true              the assay's real protein
      decoy_random_k    a uniformly sampled different target (k independent draws)
      decoy_dissimilar  the test-set target with the lowest 3-mer profile
                        similarity to the true one (deterministic, worst case)
      decoy_shuffled    the true sequence with its residues permuted: identical
                        length and amino-acid composition, destroyed order
      control_molecule  the true protein, but the ligands replaced by random
                        molecules drawn from other assays

    control_molecule is the point of the design. It corrupts the OTHER input by
    the same kind of operation. If corrupting the molecule destroys the ranking
    while corrupting the protein does not, the asymmetry is the result, and a
    null protein effect cannot be blamed on an insensitive measurement.

METRICS
    rho_<arm>   Spearman of that arm's scores against the measured pChEMBL
    tau_<arm>   Kendall tau between the TRUE-protein score order and the arm's
                score order -- "does the wrong protein reorder the ligands?"
    rho_mw      molecular weight alone, a protein-free reference point

STATISTICS
    Per-assay paired design. Reports a one-sided Wilcoxon signed-rank test
    (true > decoy), Cliff's delta, and cluster-bootstrap 95% CIs over assays.
    Assays are sampled uniformly at random from every eligible group; no size
    cap, no filtering on model performance.

USAGE
    python run_protein_dependence.py \
        --ckpt ../../outputs/fusion512_ranking_scaffold_median/checkpoint-5916 \
        --tag  scaffold_ep3 \
        --split ../../../dataset/processed/chembl_37/scaffold_component_split_median/test.parquet
"""
from __future__ import annotations

import argparse, json, os, sys, time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, kendalltau, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from reward_model.model.io import load_reward_model  # noqa: E402

MIN_LIGANDS = 3
MIN_SPAN = 0.5


# ---------------------------------------------------------------- utilities
def eligible_groups(df: pd.DataFrame) -> np.ndarray:
    """Groups the training metric itself scores: >=3 ligands, >=0.5 pChEMBL span."""
    g = df.groupby("assay_group_id")
    size = g.size()
    span = g.pchembl_value.max() - g.pchembl_value.min()
    return np.asarray(sorted(size[(size >= MIN_LIGANDS) & (span >= MIN_SPAN)].index))


def build_dissimilar_map(targets: pd.DataFrame, k: int = 3):
    """For every target, the least 3-mer-similar other target.

    One sparse matmul over the whole target set, computed once. Doing this
    per assay instead costs ~690k Counter intersections and dominates runtime.
    """
    from scipy import sparse

    tids = targets.target_chembl_id.to_numpy()
    vocab: dict[str, int] = {}
    rows, cols, vals = [], [], []
    for i, seq in enumerate(targets.protein_sequence):
        c = Counter(seq[j:j + k] for j in range(max(0, len(seq) - k + 1)))
        for mer, n in c.items():
            rows.append(i); cols.append(vocab.setdefault(mer, len(vocab))); vals.append(n)
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(tids), len(vocab)), dtype=np.float32)
    norms = np.sqrt(X.multiply(X).sum(1)).A.ravel()
    norms[norms == 0] = 1.0
    X = sparse.diags(1.0 / norms) @ X
    sim = (X @ X.T).toarray()
    np.fill_diagonal(sim, np.inf)          # never pick the target itself
    far_idx = sim.argmin(1)
    return ({t: tids[j] for t, j in zip(tids, far_idx)},
            {t: float(sim[i, far_idx[i]]) for i, t in enumerate(tids)})


def cluster_bootstrap_ci(values, weights, n=10000, seed=0):
    """Resample ASSAYS (not rows) so the CI respects the clustered design."""
    values = np.asarray(values, float); weights = np.asarray(weights, float)
    ok = np.isfinite(values)
    values, weights = values[ok], weights[ok]
    if len(values) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    draws = (values[idx] * weights[idx]).sum(1) / weights[idx].sum(1)
    point = float((values * weights).sum() / weights.sum())
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def cliffs_delta(a, b) -> float:
    """Non-parametric paired effect size; |d|<0.147 is conventionally negligible."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) == 0:
        return float("nan")
    d = a - b
    return float((np.sum(d > 0) - np.sum(d < 0)) / len(d))


# ---------------------------------------------------------------- scoring
class Scorer:
    def __init__(self, ckpt: str, device: torch.device, chunk: int = 24):
        # contrastive_loss_weight>0 makes forward demand group ids; zero it for
        # inference only. Structure is untouched, so strict=True still passes.
        self.model = load_reward_model(
            ckpt, device=device, strict=True,
            config_overrides={"contrastive_loss_weight": 0.0},
        ).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device, self.chunk = device, chunk

    @torch.no_grad()
    def score(self, sequence: str, selfies: list[str]) -> np.ndarray:
        pt = self.model.tokenize_proteins([sequence], device=self.device)
        mt = self.model.tokenize_molecules(list(selfies), device=self.device)
        out = []
        for s in range(0, len(selfies), self.chunk):
            e = min(s + self.chunk, len(selfies)); n = e - s
            with torch.autocast("cuda", dtype=torch.bfloat16):
                r = self.model(
                    protein_input_ids=pt["input_ids"].expand(n, -1).contiguous(),
                    protein_attention_mask=pt["attention_mask"].expand(n, -1).contiguous(),
                    molecule_input_ids=mt["input_ids"][s:e],
                    molecule_attention_mask=mt["attention_mask"][s:e],
                )
            out.append(r.ranking_score.float().reshape(-1).cpu().numpy())
        return np.concatenate(out)


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", required=True, help="parquet holding the held-out split")
    ap.add_argument("--n-assays", type=int, default=500)
    ap.add_argument("--n-random-decoys", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_parquet(args.split)
    groups = eligible_groups(df)
    pick = rng.choice(groups, size=min(args.n_assays, len(groups)), replace=False)
    sub = df[df.assay_group_id.isin(set(pick))].copy()
    print(f"[{args.tag}] {len(pick)} of {len(groups)} eligible assays, {len(sub)} rows", flush=True)

    # deterministic decoy pools
    targets = (df.drop_duplicates("target_chembl_id")[["target_chembl_id", "protein_sequence"]]
                 .dropna().reset_index(drop=True))
    print("  precomputing dissimilar-target map ...", flush=True)
    far_of, far_sim = build_dissimilar_map(targets)
    seq_of = dict(zip(targets.target_chembl_id, targets.protein_sequence))
    all_tids = targets.target_chembl_id.to_numpy()
    mol_pool = df.drop_duplicates("compound_selfies")[["compound_selfies", "molecule_chembl_id"]]

    scorer = Scorer(args.ckpt, torch.device("cuda:0"), chunk=args.chunk)

    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Descriptors
        RDLogger.DisableLog("rdApp.*")
        have_rdkit = True
    except Exception:
        have_rdkit = False

    rows = []
    t0 = time.time()
    for i, (gid, d) in enumerate(sub.groupby("assay_group_id")):
        d = d.reset_index(drop=True)
        y = d.pchembl_value.to_numpy()
        selfies = d.compound_selfies.tolist()
        true_seq = d.protein_sequence.iloc[0]
        true_tid = d.target_chembl_id.iloc[0]

        s_true = scorer.score(true_seq, selfies)
        rec = {"assay_group_id": gid, "target_chembl_id": true_tid, "n_ligands": len(d),
               "pchembl_span": float(y.max() - y.min()),
               "rho_true": spearmanr(s_true, y).statistic}

        # --- protein arms -------------------------------------------------
        arms = {}
        others = all_tids[all_tids != true_tid]
        for k in range(args.n_random_decoys):
            t = rng.choice(others)
            arms[f"decoy_random_{k}"] = (seq_of[t], t)

        far = far_of[true_tid]
        arms["decoy_dissimilar"] = (seq_of[far], far)
        rec["dissimilar_kmer_cosine"] = far_sim[true_tid]

        shuffled = "".join(rng.permutation(list(true_seq)))
        arms["decoy_shuffled"] = (shuffled, "SHUFFLED")

        for name, (seq, tid) in arms.items():
            s = scorer.score(seq, selfies)
            rec[f"rho_{name}"] = spearmanr(s, y).statistic
            rec[f"tau_{name}"] = kendalltau(s_true, s).statistic
            rec[f"decoy_id_{name}"] = tid

        # --- molecule control: corrupt the OTHER input --------------------
        fake = mol_pool[~mol_pool.compound_selfies.isin(set(selfies))] \
            .sample(len(d), random_state=int(rng.integers(1 << 31)))
        s_mol = scorer.score(true_seq, fake.compound_selfies.tolist())
        rec["rho_control_molecule"] = spearmanr(s_mol, y).statistic
        rec["tau_control_molecule"] = kendalltau(s_true, s_mol).statistic

        if have_rdkit:
            mw = np.array([Descriptors.MolWt(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s)
                           else np.nan for s in d.smiles], float)
            rec["rho_mw"] = spearmanr(mw, y, nan_policy="omit").statistic if np.isfinite(mw).sum() >= 3 else np.nan

        rows.append(rec)
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            eta = el / (i + 1) * (len(pick) - i - 1)
            print(f"  {i+1}/{len(pick)} assays  {el/60:.1f} min elapsed, ETA {eta/60:.1f} min",
                  flush=True)

    per = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    per.to_csv(os.path.join(args.out_dir, f"per_assay_{args.tag}.csv"), index=False)

    # --------------------------------------------------------- aggregate
    w = per.n_ligands.to_numpy()
    rand_cols = [f"rho_decoy_random_{k}" for k in range(args.n_random_decoys)]
    per["rho_decoy_random_mean"] = per[rand_cols].mean(axis=1)
    tau_rand = [f"tau_decoy_random_{k}" for k in range(args.n_random_decoys)]
    per["tau_decoy_random_mean"] = per[tau_rand].mean(axis=1)

    summary = {"tag": args.tag, "ckpt": args.ckpt, "split": args.split,
               "n_assays": int(len(per)), "n_eligible_total": int(len(groups)),
               "n_ligands_total": int(per.n_ligands.sum()), "seed": args.seed,
               "median_dissimilar_kmer_cosine": float(per.dissimilar_kmer_cosine.median()),
               "weighted_spearman": {}, "median_tau": {}, "tests": {}}

    arm_names = ["true", "decoy_random_mean", "decoy_dissimilar", "decoy_shuffled",
                 "control_molecule"] + (["mw"] if have_rdkit else [])
    for a in arm_names:
        col = f"rho_{a}"
        if col not in per: continue
        pt, lo, hi = cluster_bootstrap_ci(per[col], w, seed=args.seed)
        summary["weighted_spearman"][a] = {"point": pt, "ci95": [lo, hi]}
    for a in ["decoy_random_mean", "decoy_dissimilar", "decoy_shuffled", "control_molecule"]:
        col = f"tau_{a}"
        if col not in per: continue
        v = per[col].dropna()
        summary["median_tau"][a] = {"median": float(v.median()),
                                    "q25": float(v.quantile(.25)), "q75": float(v.quantile(.75))}

    for a in ["decoy_random_mean", "decoy_dissimilar", "decoy_shuffled"]:
        both = per[["rho_true", f"rho_{a}"]].dropna()
        if len(both) < 10: continue
        stat, p = wilcoxon(both.rho_true, both[f"rho_{a}"], alternative="greater")
        diff, lo, hi = cluster_bootstrap_ci(both.rho_true - both[f"rho_{a}"],
                                            np.ones(len(both)), seed=args.seed)
        summary["tests"][a] = {
            "n_paired": int(len(both)), "wilcoxon_stat": float(stat), "p_one_sided": float(p),
            "mean_paired_delta_rho": diff, "delta_ci95": [lo, hi],
            "cliffs_delta": cliffs_delta(both.rho_true, both[f"rho_{a}"]),
        }

    with open(os.path.join(args.out_dir, f"summary_{args.tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    ws = summary["weighted_spearman"]
    print(f"\n[{args.tag}] weighted Spearman")
    for a in arm_names:
        if a in ws:
            c = ws[a]
            print(f"   {a:<22} {c['point']:+.4f}  95% CI [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]")
    print(f"[{args.tag}] median Kendall tau vs the true-protein ordering")
    for a, c in summary["median_tau"].items():
        print(f"   {a:<22} {c['median']:+.3f}  IQR [{c['q25']:+.3f}, {c['q75']:+.3f}]")
    for a, t in summary["tests"].items():
        print(f"[{args.tag}] {a}: delta_rho={t['mean_paired_delta_rho']:+.4f} "
              f"CI[{t['delta_ci95'][0]:+.4f},{t['delta_ci95'][1]:+.4f}] "
              f"p={t['p_one_sided']:.3g} cliffs_d={t['cliffs_delta']:+.3f}")


if __name__ == "__main__":
    main()
