"""Cross-protein ranking test.

Two proteins A and B were each really measured against the SAME molecule set M,
and their measured orderings disagree. Score M twice - once given protein A, once
given protein B - and compare every prediction against BOTH ground truths.

  rho(pred_A, GT_A)   matched     the model was given A and is judged against A
  rho(pred_A, GT_B)   crossed     the model was given A and is judged against B
  rho(pred_B, GT_B)   matched
  rho(pred_B, GT_A)   crossed
  rho(pred_A, pred_B) agreement   LABEL-FREE: did changing the protein change anything?

  delta = mean(matched) - mean(crossed)

Why delta is the right statistic: if the model is protein-blind then
pred_A == pred_B, so the matched pair and the crossed pair are built from exactly
the same two numbers and **delta is identically 0** - whatever the model's
accuracy, whatever the ground truths look like. A target-conditioned model scores
delta > 0. The ceiling is 1 - gt_rho, which is why only disagreeing pairs are used.

Each model is evaluated ONLY on the split it was trained on, in both parts:
`train` assays (molecules it has seen) and `val` assays (molecules it has not).
Mixing splits would confound the seen/unseen axis.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from reward_model.model.io import load_reward_model  # noqa: E402

ROOT = "/home/furkan/Prot2Mol_Project/Prot2Mol"
SPLITS = {"scaffold": f"{ROOT}/dataset/processed/chembl_37/scaffold_component_split_median",
          "random":   f"{ROOT}/dataset/processed/chembl_37/target_aware_random_assay_activity_balanced_median"}
PAIRDIR = os.path.join(HERE, "..", "split_overlap")
COLS = ["assay_group_id","target_chembl_id","protein_accession","protein_sequence",
        "molecule_chembl_id","compound_selfies","pchembl_value"]


def selected_pairs(tag, min_common, max_gt_rho, min_span):
    d = pd.read_csv(os.path.join(PAIRDIR, f"shared_pairs_{tag}.csv"))
    d = d[(d.span_a >= min_span) & (d.span_b >= min_span) &
          (d.n_common >= min_common) & (d.gt_spearman <= max_gt_rho)]
    d["key"] = [tuple(sorted(x)) for x in zip(d.target_a, d.target_b)]
    # one assay pair per target pair: the most strongly disagreeing one
    return d.sort_values("gt_spearman").drop_duplicates("key").reset_index(drop=True)


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


def rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return float("nan")
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--min-common", type=int, default=30)
    ap.add_argument("--max-gt-rho", type=float, default=0.0)
    ap.add_argument("--min-span", type=float, default=1.5)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    models = json.load(open(args.models))

    data, pairs = {}, {}
    for split in SPLITS:
        for part in ("train", "val"):
            tag = f"{split}_{part}"
            data[tag] = pd.read_parquet(f"{SPLITS[split]}/{part}.parquet", columns=COLS)
            pairs[tag] = selected_pairs(tag, args.min_common, args.max_gt_rho, args.min_span)
            print(f"{tag}: {len(pairs[tag])} target pairs selected", flush=True)

    # per-assay molecule -> pchembl, and molecule -> selfies, built once
    assay_map, selfies_of, seq_of, acc_of = {}, {}, {}, {}
    for tag, df in data.items():
        need = set(pairs[tag].assay_a) | set(pairs[tag].assay_b)
        sub = df[df.assay_group_id.isin(need)]
        for a, g in sub.groupby("assay_group_id"):
            assay_map[(tag, a)] = g.groupby("molecule_chembl_id").pchembl_value.median().to_dict()
        selfies_of[tag] = dict(zip(sub.molecule_chembl_id, sub.compound_selfies))
        t = df.drop_duplicates("target_chembl_id").set_index("target_chembl_id")
        seq_of[tag] = t.protein_sequence.to_dict(); acc_of[tag] = t.protein_accession.to_dict()

    dev = torch.device("cuda:0")
    rows = []
    for mi, m in enumerate(models):
        split = m["split"]
        print(f"\n[{mi+1}/{len(models)}] {m['tag']} (epoch {m['epoch']}, {split})", flush=True)
        sc = Scorer(m["ckpt"], dev)
        t0 = time.time()
        for part in ("train", "val"):
            tag = f"{split}_{part}"
            P = pairs[tag]
            for r in P.itertuples():
                ma, mb = assay_map[(tag, r.assay_a)], assay_map[(tag, r.assay_b)]
                common = sorted(set(ma) & set(mb))
                common = [c for c in common if isinstance(selfies_of[tag].get(c), str)
                          and selfies_of[tag][c]]
                if len(common) < args.min_common:
                    continue
                sel = [selfies_of[tag][c] for c in common]
                gtA = np.array([ma[c] for c in common])
                gtB = np.array([mb[c] for c in common])
                pA = sc(seq_of[tag][r.target_a], sel)
                pB = sc(seq_of[tag][r.target_b], sel)
                rows.append(dict(
                    model=m["tag"], epoch=m["epoch"], split=split, part=part,
                    target_a=r.target_a, target_b=r.target_b,
                    acc_a=acc_of[tag].get(r.target_a), acc_b=acc_of[tag].get(r.target_b),
                    n_common=len(common), gt_rho=rho(gtA, gtB),
                    rho_pA_gtA=rho(pA, gtA), rho_pA_gtB=rho(pA, gtB),
                    rho_pB_gtB=rho(pB, gtB), rho_pB_gtA=rho(pB, gtA),
                    rho_pA_pB=rho(pA, pB)))
            print(f"   {part:<5} done ({len(P)} pairs)  {time.time()-t0:.0f}s", flush=True)
        del sc; torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    out["matched"] = out[["rho_pA_gtA","rho_pB_gtB"]].mean(axis=1)
    out["crossed"] = out[["rho_pA_gtB","rho_pB_gtA"]].mean(axis=1)
    out["delta"] = out.matched - out.crossed
    out.to_csv(os.path.join(args.out_dir, "per_pair.csv"), index=False)
    print(f"\nwrote {os.path.join(args.out_dir,'per_pair.csv')}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
