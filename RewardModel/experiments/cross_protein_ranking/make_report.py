"""RESULTS.md + figure for the cross-protein ranking test."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=os.path.dirname(os.path.abspath(__file__)); R=os.path.join(HERE,"results")
d=pd.read_csv(os.path.join(R,"per_pair.csv"))
ORDER=["scaffold_ep3","scaffold_ep12","scaffold_ep32","scaffold_ep100",
       "random_ep1","random_ep10","random_ep34","random_ep100"]
NAMES={("CHEMBL251","CHEMBL256"):"adenosine A2A vs A3",("CHEMBL261","CHEMBL205"):"carbonic anhydrase I vs II",
       ("CHEMBL234","CHEMBL219"):"dopamine D3 vs D4",("CHEMBL4501","CHEMBL2534"):"P23443 vs O15530",
       ("CHEMBL5805","CHEMBL5804"):"TLR8 vs TLR9",("CHEMBL3706","CHEMBL333"):"P78536 vs P08253",
       ("CHEMBL229","CHEMBL211"):"P35348 vs P08172",("CHEMBL2973","CHEMBL3553"):"O75116 vs P29597"}
d["pk"]=list(zip(d.target_a,d.target_b))

def wci(v,w,n=20000,seed=0):
    v=np.asarray(v,float);w=np.asarray(w,float);ok=np.isfinite(v);v,w=v[ok],w[ok]
    if len(v)==0: return (np.nan,)*3
    r=np.random.default_rng(seed);i=r.integers(0,len(v),(n,len(v)))
    b=(v[i]*w[i]).sum(1)/w[i].sum(1)
    return float(np.average(v,weights=w)),float(np.percentile(b,2.5)),float(np.percentile(b,97.5))

# ---------------- figure ----------------
SURF,INK,INK2="#fcfcfb","#0b0b0b","#52514e"; SC,RD,GRID="#2a78d6","#eb6834","#dddcd6"
plt.rcParams.update({"font.size":9,"axes.edgecolor":INK2,"axes.labelcolor":INK,"text.color":INK,
                     "xtick.color":INK2,"ytick.color":INK2,"figure.facecolor":SURF,
                     "axes.facecolor":SURF,"savefig.facecolor":SURF})
fig,ax=plt.subplots(figsize=(7.2,4.2))
for split,colour in (("scaffold",SC),("random",RD)):
    for part,ls,mk,lab in (("train","-","o","molecules SEEN"),("val",(0,(3,2)),"s","molecules HELD OUT")):
        s=d[(d.split==split)&(d.part==part)]
        eps=sorted(s.epoch.unique()); mu=[];lo=[];hi=[]
        for e in eps:
            q=s[s.epoch==e]; m,l,h=wci(q.delta,q.n_common); mu.append(m);lo.append(l);hi.append(h)
        ax.fill_between(eps,lo,hi,color=colour,alpha=.13,linewidth=0)
        ax.plot(eps,mu,color=colour,linewidth=2 if part=="train" else 1.7,linestyle=ls,marker=mk,
                markersize=5,markeredgecolor=SURF,markeredgewidth=1.3,
                label=f"{split} split — {lab}",zorder=3)
        ax.annotate(f"{mu[-1]:+.2f}",(eps[-1],mu[-1]),textcoords="offset points",xytext=(6,0),
                    color=colour,fontsize=8.5,weight="bold",va="center")
ax.axhline(0,color=INK2,linewidth=1,linestyle=(0,(4,3)),zorder=1)
ax.set_xscale("log");ax.set_xticks([1,3,10,30,100]);ax.set_xticklabels(["1","3","10","30","100"])
ax.grid(True,color=GRID,linewidth=.7);ax.set_axisbelow(True)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax.set_xlabel("training epoch");ax.set_ylabel("delta  =  matched − crossed\n(molecule-count weighted)")
ax.legend(frameon=False,fontsize=8.3,loc="upper left")
ax.set_title("A protein-blind model scores delta = 0 exactly",fontsize=11,pad=10)
fig.tight_layout();fig.savefig(os.path.join(R,"fig_delta.png"),dpi=200,bbox_inches="tight")

# ---------------- markdown ----------------
L=[];A=L.append
A("# Cross-protein ranking test")
A("")
A("Two proteins A and B were each really measured against the **same** molecule set, and")
A("their measured orderings disagree (`gt_rho <= 0`). Each model scores that identical")
A("molecule set twice — once given A, once given B — and every prediction is compared")
A("against **both** ground truths.")
A("")
A("## Why `delta` is the right statistic")
A("")
A("```")
A("delta = mean[ rho(pred_A,GT_A), rho(pred_B,GT_B) ]  -  mean[ rho(pred_A,GT_B), rho(pred_B,GT_A) ]")
A("              matched                                    crossed")
A("```")
A("")
A("If the model is protein-blind then `pred_A == pred_B`, so both means are built from the")
A("same two numbers and **delta is identically 0** — whatever the model's accuracy,")
A("whatever the ground truths look like. A target-conditioned model scores `delta > 0`.")
A("The ceiling is `1 - gt_rho`, which is why only disagreeing pairs are used.")
A("")
A("`rho(pred_A, pred_B)` is reported too. It uses **no labels at all**: a blind model")
A("scores exactly 1.0 there regardless of how good or bad its ranking is.")
A("")
A("## Design")
A("")
A("| | |")
A("|---|---|")
A("| pair selection | >= 30 shared molecules, `gt_rho <= 0`, both pChEMBL spans >= 1.5 |")
A("| de-duplication | one assay pair per target pair (the most strongly disagreeing) |")
A("| pairs used | scaffold 410 train / 10 val · random 276 train / 12 val |")
A("| model-cell matching | each model is evaluated **only on the split it trained on**, in both parts |")
A("| molecules | `train` assays = seen in training · `val` assays = held out |")
A("")
A("Scoring was verified in earlier work to reproduce the training-time")
A("`eval_assay_spearman` metric exactly (0.1831 vs 0.1831 logged).")
A("")
A("## Negative control")
A("")
A("`random_ep1` is one epoch from initialisation. It returns `delta = +0.005`")
A("(95% CI [-0.007, +0.008]) with prediction agreement **0.996-0.998** — i.e. it is")
A("almost perfectly protein-blind, and the statistic correctly reports ~0. Every effect")
A("below is measured against that baseline.")
A("")
A("## Aggregate results")
A("")
A("Molecule-count weighting matters a lot in the small val cells, so both are shown.")
A("The weighted figure is the one to trust: a 742-molecule pair gives a far more reliable")
A("Spearman than a 30-molecule one.")
A("")
A("| model | ep | part | pairs | mean `gt_rho` | matched | crossed | delta (unw.) | **delta (weighted)** | 95% CI | pred agree |")
A("|---|---|---|---|---|---|---|---|---|---|---|")
for m in ORDER:
    for p in ("train","val"):
        s=d[(d.model==m)&(d.part==p)]
        if not len(s): continue
        pt,lo,hi=wci(s.delta,s.n_common)
        A(f"| `{m}` | {s.epoch.iloc[0]} | {p} | {len(s)} | {s.gt_rho.mean():+.3f} | "
          f"{s.matched.mean():.4f} | {s.crossed.mean():.4f} | {s.delta.mean():+.4f} | "
          f"**{pt:+.4f}** | [{lo:+.4f}, {hi:+.4f}] | {s.rho_pA_pB.mean():+.4f} |")
A("")
A("## Cross-Spearman components")
A("")
A("The four correlations behind `delta`, averaged over pairs.")
A("")
A("| model | part | `pred_A` vs `GT_A` | `pred_A` vs `GT_B` | `pred_B` vs `GT_B` | `pred_B` vs `GT_A` |")
A("|---|---|---|---|---|---|")
for m in ORDER:
    for p in ("train","val"):
        s=d[(d.model==m)&(d.part==p)]
        if not len(s): continue
        A(f"| `{m}` | {p} | {s.rho_pA_gtA.mean():+.4f} | {s.rho_pA_gtB.mean():+.4f} | "
          f"{s.rho_pB_gtB.mean():+.4f} | {s.rho_pB_gtA.mean():+.4f} |")
A("")
A("Note how `pred_A` vs `GT_B` goes **negative** for the well-trained models on seen")
A("molecules (-0.10 at epoch 100): given protein A, the model actively anti-ranks")
A("protein B's ordering. That is the opposite of blindness.")
A("")
A("## The controlled comparison: TLR8 vs TLR9")
A("")
A("`CHEMBL5805` (Q9NR97) and `CHEMBL5804` (Q9NR96) appear in **both** validation cells")
A("with essentially the same molecule set, so the protein pair and the chemistry are held")
A("fixed and only the model's training differs:")
A("")
A("| model | molecules | seen in training? | delta | pred agree |")
A("|---|---|---|---|---|")
for m in ("random_ep100","scaffold_ep100"):
    s=d[(d.model==m)&(d.pk==("CHEMBL5805","CHEMBL5804"))]
    if not len(s): continue
    r=s.iloc[0]
    seen="~61% leaked into train" if m.startswith("random") else "**no — novel scaffolds**"
    A(f"| `{m}` | {int(r.n_common)} | {seen} | **{r.delta:+.3f}** | {r.rho_pA_pB:+.3f} |")
A("")
A("Same two proteins, same molecules. **delta = +1.09 when the molecules were memorised,")
A("+0.03 when they are novel.** This is the cleanest evidence in the study that the")
A("target conditioning is memorisation of seen chemistry rather than transferable")
A("protein understanding.")
A("")
A("## Named candidate pairs (individual, not averaged)")
A("")
for pk,label in NAMES.items():
    s=d[d.pk==pk]
    if s.empty: continue
    r0=s.iloc[0]
    A(f"### {label} — `{pk[0]}` / `{pk[1]}`")
    A("")
    A(f"{int(r0.n_common)} shared molecules · `gt_rho` = {r0.gt_rho:+.3f} · assays from **{r0.part}**")
    A("")
    A("| model | `pA`&#124;`gtA` | `pA`&#124;`gtB` | `pB`&#124;`gtB` | `pB`&#124;`gtA` | delta | pred agree |")
    A("|---|---|---|---|---|---|---|")
    for _,r in s.sort_values(["split","epoch"]).iterrows():
        A(f"| `{r.model}` | {r.rho_pA_gtA:+.3f} | {r.rho_pA_gtB:+.3f} | {r.rho_pB_gtB:+.3f} | "
          f"{r.rho_pB_gtA:+.3f} | **{r.delta:+.3f}** | {r.rho_pA_pB:+.3f} |")
    A("")
A("## Limitations")
A("")
A("- **The val cells rest on 10 and 12 pairs.** That is the binding constraint on every")
A("  held-out conclusion here, and it is a property of the data: only 10 scaffold-val")
A("  target pairs share >= 30 molecules with disagreeing ground truths.")
A("- **Pseudo-replication in scaffold val.** Three of the ten pairs share target")
A("  `CHEMBL240`, all with n=43 and deltas 0.47 / 0.41 / 0.54. They are not independent")
A("  observations, and they are the main driver of the *unweighted* scaffold-val mean.")
A("  Weighting by molecule count cuts that mean from +0.188 to +0.085.")
A("- **Pairs were de-duplicated by taking the most-disagreeing assay pair per target**")
A("  pair. Since `gt_rho` is estimated from noisy pChEMBL, that selection makes the")
A("  reported `gt_rho` slightly more negative than the truth. It does not bias `delta`,")
A("  which never uses `gt_rho`.")
A("- Proteins sharing molecules are often homologues, so this measures fine-grained")
A("  discrimination within a family, not coarse target recognition.")
A("")
A("## Files")
A("")
A("| file | contents |")
A("|---|---|")
A("| `per_pair.csv` | every (model, pair): all four cross-Spearmans, delta, agreement, `gt_rho` |")
A("| `fig_delta.png` | weighted delta vs training epoch, by split and molecule novelty |")
A("| `run_cross_protein.py` | the experiment |")
A("| `models.json` | the eight checkpoints |")
A("")
open(os.path.join(R,"RESULTS.md"),"w").write("\n".join(L)+"\n")
print("wrote RESULTS.md and fig_delta.png")
