"""Seen/unseen molecule x seen/unseen protein grid, across training epochs.

For each model, ranking-eligible assays are drawn from BOTH the split the model
trained on and the split it was validated on, and each assay's ligands are
re-scored under four protein conditions. Spearman is always measured against the
assay's own measured pChEMBL ordering, so every number answers the same question:
how well does this model rank THESE ligands, given THIS protein?

  molecule axis   train assays -> molecules the model has seen
                  val   assays -> molecules it has not (scaffold split only;
                                  the random split leaks 61.5% of val molecules)

  protein axis    true        the assay's real protein
                  other_train a different protein drawn from the training set
                  other_val   a different protein drawn from the validation set
                  novel       a protein from BIOSNAP that appears in NEITHER
                              ChEMBL training split - genuinely never seen

NOTE ON THE SPLITS: both ChEMBL splits hold out molecules and assays, never
targets. Every validation target also occurs in training (0 val-only targets in
both splits), so `other_val` is a SEEN protein despite coming from the val file.
It is reported anyway: if it tracks `other_train`, that confirms the split gives
no protein-level holdout, which is why the external `novel` arm is needed.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from reward_model.model.io import load_reward_model  # noqa: E402

ROOT = "/home/furkan/Prot2Mol_Project/Prot2Mol"
SPLITS = {
    "scaffold": f"{ROOT}/dataset/processed/chembl_37/scaffold_component_split_median",
    "random":   f"{ROOT}/dataset/processed/chembl_37/target_aware_random_assay_activity_balanced_median",
}
BIOSNAP = f"{ROOT}/dataset/processed/benchmarks/biosnap_random"
MIN_LIGANDS, MIN_SPAN = 3, 0.5


def eligible(df):
    g = df.groupby("assay_group_id")
    sz, sp = g.size(), g.pchembl_value.max() - g.pchembl_value.min()
    return np.asarray(sorted(sz[(sz >= MIN_LIGANDS) & (sp >= MIN_SPAN) & (sz <= 64)].index))


class Scorer:
    def __init__(self, ckpt, device, chunk=48):
        self.m = load_reward_model(ckpt, device=device, strict=True,
                                   config_overrides={"contrastive_loss_weight": 0.0}).eval()
        for p in self.m.parameters(): p.requires_grad_(False)
        self.dev, self.chunk = device, chunk

    @torch.no_grad()
    def __call__(self, seq, selfies):
        pt = self.m.tokenize_proteins([seq], device=self.dev)
        mt = self.m.tokenize_molecules(list(selfies), device=self.dev)
        out = []
        for s in range(0, len(selfies), self.chunk):
            e = min(s + self.chunk, len(selfies)); n = e - s
            with torch.autocast("cuda", dtype=torch.bfloat16):
                r = self.m(protein_input_ids=pt["input_ids"].expand(n, -1).contiguous(),
                           protein_attention_mask=pt["attention_mask"].expand(n, -1).contiguous(),
                           molecule_input_ids=mt["input_ids"][s:e],
                           molecule_attention_mask=mt["attention_mask"][s:e])
            out.append(r.ranking_score.float().reshape(-1).cpu().numpy())
        return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="JSON list of {tag,ckpt,split,epoch}")
    ap.add_argument("--n-assays", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results_grid"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    models = json.load(open(args.models))

    data = {}
    for name, d in SPLITS.items():
        data[name] = {s: pd.read_parquet(f"{d}/{s}.parquet") for s in ("train", "val")}
    seen_seqs = set()
    for name in SPLITS:
        seen_seqs |= set(data[name]["train"].protein_sequence.dropna())
    bs = pd.concat([pd.read_parquet(f"{BIOSNAP}/{s}.parquet", columns=["protein_sequence"])
                    for s in ("train", "val", "test")])
    novel_pool = np.array(sorted({s for s in bs.protein_sequence.dropna()
                                  if s not in seen_seqs and 30 <= len(s) <= 2000}))
    print(f"novel protein pool (BIOSNAP, unseen in both ChEMBL train sets): {len(novel_pool)}", flush=True)

    dev = torch.device("cuda:0")
    rows = []
    for mi, m in enumerate(models):
        sp = data[m["split"]]
        pools = {k: np.array(sorted(set(sp[k].protein_sequence.dropna()))) for k in ("train", "val")}
        print(f"\n[{mi+1}/{len(models)}] {m['tag']}  (epoch {m['epoch']}, {m['split']} split)", flush=True)
        scorer = Scorer(m["ckpt"], dev)
        t0 = time.time()
        for origin in ("train", "val"):
            df = sp[origin]
            rng = np.random.default_rng(args.seed)          # same assays for every model
            elig = eligible(df)
            pick = rng.choice(elig, size=min(args.n_assays, len(elig)), replace=False)
            sub = df[df.assay_group_id.isin(set(pick))]
            for gid, d in sub.groupby("assay_group_id"):
                y = d.pchembl_value.to_numpy()
                sel = d.compound_selfies.tolist()
                true_seq = d.protein_sequence.iloc[0]
                conds = {"true": true_seq}
                for pname, pool in (("other_train", pools["train"]), ("other_val", pools["val"])):
                    cand = pool[pool != true_seq]
                    conds[pname] = str(rng.choice(cand))
                conds["novel"] = str(rng.choice(novel_pool))
                rec = {"model": m["tag"], "epoch": m["epoch"], "split": m["split"],
                       "assay_origin": origin, "assay_group_id": gid, "n_ligands": len(d)}
                for cname, seq in conds.items():
                    rec[f"rho_{cname}"] = spearmanr(scorer(seq, sel), y).statistic
                rows.append(rec)
            print(f"   {origin:<5} assays done ({len(pick)})  {time.time()-t0:.0f}s", flush=True)
        del scorer; torch.cuda.empty_cache()

    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(args.out_dir, "per_assay_grid.csv"), index=False)
    print(f"\nwrote {os.path.join(args.out_dir,'per_assay_grid.csv')}  ({len(per)} assay-rows)")


if __name__ == "__main__":
    main()
