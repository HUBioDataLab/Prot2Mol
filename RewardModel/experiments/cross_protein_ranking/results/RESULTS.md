# Cross-protein ranking test

Two proteins A and B were each really measured against the **same** molecule set, and
their measured orderings disagree (`gt_rho <= 0`). Each model scores that identical
molecule set twice — once given A, once given B — and every prediction is compared
against **both** ground truths.

## Why `delta` is the right statistic

```
delta = mean[ rho(pred_A,GT_A), rho(pred_B,GT_B) ]  -  mean[ rho(pred_A,GT_B), rho(pred_B,GT_A) ]
              matched                                    crossed
```

If the model is protein-blind then `pred_A == pred_B`, so both means are built from the
same two numbers and **delta is identically 0** — whatever the model's accuracy,
whatever the ground truths look like. A target-conditioned model scores `delta > 0`.
The ceiling is `1 - gt_rho`, which is why only disagreeing pairs are used.

`rho(pred_A, pred_B)` is reported too. It uses **no labels at all**: a blind model
scores exactly 1.0 there regardless of how good or bad its ranking is.

## Design

| | |
|---|---|
| pair selection | >= 30 shared molecules, `gt_rho <= 0`, both pChEMBL spans >= 1.5 |
| de-duplication | one assay pair per target pair (the most strongly disagreeing) |
| pairs used | scaffold 410 train / 10 val · random 276 train / 12 val |
| model-cell matching | each model is evaluated **only on the split it trained on**, in both parts |
| molecules | `train` assays = seen in training · `val` assays = held out |

Scoring was verified in earlier work to reproduce the training-time
`eval_assay_spearman` metric exactly (0.1831 vs 0.1831 logged).

## Negative control

`random_ep1` is one epoch from initialisation. It returns `delta = +0.005`
(95% CI [-0.007, +0.008]) with prediction agreement **0.996-0.998** — i.e. it is
almost perfectly protein-blind, and the statistic correctly reports ~0. Every effect
below is measured against that baseline.

## Aggregate results

Molecule-count weighting matters a lot in the small val cells, so both are shown.
The weighted figure is the one to trust: a 742-molecule pair gives a far more reliable
Spearman than a 30-molecule one.

| model | ep | part | pairs | mean `gt_rho` | matched | crossed | delta (unw.) | **delta (weighted)** | 95% CI | pred agree |
|---|---|---|---|---|---|---|---|---|---|---|
| `scaffold_ep3` | 3 | train | 410 | -0.147 | 0.2153 | 0.1691 | +0.0462 | **+0.0434** | [+0.0336, +0.0539] | +0.8926 |
| `scaffold_ep3` | 3 | val | 10 | -0.164 | 0.1255 | 0.0704 | +0.0551 | **-0.0055** | [-0.0292, +0.0862] | +0.8711 |
| `scaffold_ep12` | 12 | train | 410 | -0.147 | 0.5173 | 0.0868 | +0.4305 | **+0.4190** | [+0.3885, +0.4507] | +0.4281 |
| `scaffold_ep12` | 12 | val | 10 | -0.164 | 0.1220 | 0.0295 | +0.0925 | **+0.0359** | [+0.0044, +0.1461] | +0.4908 |
| `scaffold_ep32` | 32 | train | 410 | -0.147 | 0.7273 | -0.0360 | +0.7633 | **+0.7505** | [+0.7191, +0.7807] | +0.1075 |
| `scaffold_ep32` | 32 | val | 10 | -0.164 | 0.1716 | -0.0165 | +0.1881 | **+0.0626** | [-0.0046, +0.3036] | +0.3522 |
| `scaffold_ep100` | 100 | train | 410 | -0.147 | 0.8634 | -0.1030 | +0.9664 | **+0.9505** | [+0.9214, +0.9765] | -0.0568 |
| `scaffold_ep100` | 100 | val | 10 | -0.164 | 0.1688 | -0.0189 | +0.1877 | **+0.0851** | [+0.0326, +0.2775] | +0.2294 |
| `random_ep1` | 1 | train | 276 | -0.152 | 0.1087 | 0.1063 | +0.0024 | **+0.0029** | [+0.0015, +0.0043] | +0.9960 |
| `random_ep1` | 1 | val | 12 | -0.094 | 0.1159 | 0.1167 | -0.0008 | **+0.0054** | [-0.0066, +0.0080] | +0.9975 |
| `random_ep10` | 10 | train | 276 | -0.152 | 0.4355 | 0.1342 | +0.3012 | **+0.3381** | [+0.2833, +0.4028] | +0.5604 |
| `random_ep10` | 10 | val | 12 | -0.094 | 0.4575 | 0.1254 | +0.3321 | **+0.6334** | [+0.2031, +0.7509] | +0.5747 |
| `random_ep34` | 34 | train | 276 | -0.152 | 0.7100 | -0.0314 | +0.7413 | **+0.7622** | [+0.7133, +0.8113] | +0.1165 |
| `random_ep34` | 34 | val | 12 | -0.094 | 0.6227 | -0.0195 | +0.6422 | **+0.8934** | [+0.4952, +0.9907] | +0.1474 |
| `random_ep100` | 100 | train | 276 | -0.152 | 0.8578 | -0.1018 | +0.9596 | **+0.9556** | [+0.9183, +0.9904] | -0.0449 |
| `random_ep100` | 100 | val | 12 | -0.094 | 0.6268 | -0.0103 | +0.6371 | **+0.9276** | [+0.4439, +1.0423] | +0.0886 |

## Cross-Spearman components

The four correlations behind `delta`, averaged over pairs.

| model | part | `pred_A` vs `GT_A` | `pred_A` vs `GT_B` | `pred_B` vs `GT_B` | `pred_B` vs `GT_A` |
|---|---|---|---|---|---|
| `scaffold_ep3` | train | +0.2036 | +0.1784 | +0.2271 | +0.1598 |
| `scaffold_ep3` | val | +0.2412 | -0.0384 | +0.0099 | +0.1793 |
| `scaffold_ep12` | train | +0.5126 | +0.1121 | +0.5220 | +0.0615 |
| `scaffold_ep12` | val | +0.1153 | +0.0099 | +0.1287 | +0.0491 |
| `scaffold_ep32` | train | +0.7198 | -0.0299 | +0.7348 | -0.0422 |
| `scaffold_ep32` | val | +0.2084 | -0.0036 | +0.1349 | -0.0294 |
| `scaffold_ep100` | train | +0.8566 | -0.0996 | +0.8703 | -0.1063 |
| `scaffold_ep100` | val | +0.1523 | +0.0116 | +0.1852 | -0.0494 |
| `random_ep1` | train | +0.1149 | +0.1003 | +0.1024 | +0.1123 |
| `random_ep1` | val | +0.1397 | +0.0929 | +0.0920 | +0.1404 |
| `random_ep10` | train | +0.4310 | +0.1432 | +0.4399 | +0.1252 |
| `random_ep10` | val | +0.3276 | +0.1979 | +0.5873 | +0.0530 |
| `random_ep34` | train | +0.7067 | -0.0223 | +0.7132 | -0.0404 |
| `random_ep34` | val | +0.5840 | +0.0655 | +0.6613 | -0.1045 |
| `random_ep100` | train | +0.8491 | -0.0958 | +0.8665 | -0.1078 |
| `random_ep100` | val | +0.5653 | +0.0636 | +0.6883 | -0.0841 |

Note how `pred_A` vs `GT_B` goes **negative** for the well-trained models on seen
molecules (-0.10 at epoch 100): given protein A, the model actively anti-ranks
protein B's ordering. That is the opposite of blindness.

## The controlled comparison: TLR8 vs TLR9

`CHEMBL5805` (Q9NR97) and `CHEMBL5804` (Q9NR96) appear in **both** validation cells
with essentially the same molecule set, so the protein pair and the chemistry are held
fixed and only the model's training differs:

| model | molecules | seen in training? | delta | pred agree |
|---|---|---|---|---|
| `random_ep100` | 783 | ~61% leaked into train | **+1.090** | -0.172 |
| `scaffold_ep100` | 742 | **no — novel scaffolds** | **+0.031** | +0.622 |

Same two proteins, same molecules. **delta = +1.09 when the molecules were memorised,
+0.03 when they are novel.** This is the cleanest evidence in the study that the
target conditioning is memorisation of seen chemistry rather than transferable
protein understanding.

## Named candidate pairs (individual, not averaged)

### adenosine A2A vs A3 — `CHEMBL251` / `CHEMBL256`

154 shared molecules · `gt_rho` = -0.458 · assays from **train**

| model | `pA`&#124;`gtA` | `pA`&#124;`gtB` | `pB`&#124;`gtB` | `pB`&#124;`gtA` | delta | pred agree |
|---|---|---|---|---|---|---|
| `random_ep1` | -0.128 | +0.056 | +0.058 | -0.119 | **-0.003** | +0.999 |
| `random_ep10` | +0.454 | -0.361 | +0.760 | -0.145 | **+0.861** | -0.168 |
| `random_ep34` | +0.798 | -0.536 | +0.923 | -0.442 | **+1.349** | -0.523 |
| `random_ep100` | +0.896 | -0.475 | +0.948 | -0.420 | **+1.369** | -0.441 |
| `scaffold_ep3` | +0.121 | +0.307 | +0.312 | +0.132 | **-0.003** | +0.989 |
| `scaffold_ep12` | +0.640 | -0.546 | +0.889 | -0.320 | **+1.197** | -0.494 |
| `scaffold_ep32` | +0.821 | -0.428 | +0.915 | -0.376 | **+1.270** | -0.331 |
| `scaffold_ep100` | +0.911 | -0.483 | +0.954 | -0.427 | **+1.387** | -0.455 |

### carbonic anhydrase I vs II — `CHEMBL261` / `CHEMBL205`

46 shared molecules · `gt_rho` = -0.631 · assays from **train**

| model | `pA`&#124;`gtA` | `pA`&#124;`gtB` | `pB`&#124;`gtB` | `pB`&#124;`gtA` | delta | pred agree |
|---|---|---|---|---|---|---|
| `random_ep1` | +0.493 | +0.151 | +0.147 | +0.499 | **-0.005** | +0.999 |
| `random_ep10` | +0.130 | +0.397 | +0.639 | -0.224 | **+0.298** | +0.867 |
| `random_ep34` | +0.880 | -0.588 | +0.929 | -0.579 | **+1.488** | -0.493 |
| `random_ep100` | +0.950 | -0.611 | +0.924 | -0.571 | **+1.528** | -0.535 |
| `scaffold_ep3` | +0.131 | +0.410 | +0.400 | +0.149 | **-0.014** | +0.981 |
| `scaffold_ep12` | -0.126 | +0.614 | +0.878 | -0.498 | **+0.318** | +0.633 |
| `scaffold_ep32` | +0.761 | -0.306 | +0.890 | -0.541 | **+1.249** | -0.225 |
| `scaffold_ep100` | +0.944 | -0.609 | +0.929 | -0.573 | **+1.527** | -0.538 |

### dopamine D3 vs D4 — `CHEMBL234` / `CHEMBL219`

65 shared molecules · `gt_rho` = -0.496 · assays from **train**

| model | `pA`&#124;`gtA` | `pA`&#124;`gtB` | `pB`&#124;`gtB` | `pB`&#124;`gtA` | delta | pred agree |
|---|---|---|---|---|---|---|
| `scaffold_ep3` | +0.683 | -0.264 | -0.243 | +0.670 | **+0.017** | +0.996 |
| `scaffold_ep12` | +0.894 | -0.457 | +0.796 | -0.332 | **+1.240** | -0.323 |
| `scaffold_ep32` | +0.938 | -0.431 | +0.901 | -0.499 | **+1.385** | -0.406 |
| `scaffold_ep100` | +0.966 | -0.458 | +0.952 | -0.499 | **+1.437** | -0.469 |

### P23443 vs O15530 — `CHEMBL4501` / `CHEMBL2534`

36 shared molecules · `gt_rho` = -0.747 · assays from **train**

| model | `pA`&#124;`gtA` | `pA`&#124;`gtB` | `pB`&#124;`gtB` | `pB`&#124;`gtA` | delta | pred agree |
|---|---|---|---|---|---|---|
| `random_ep1` | -0.067 | +0.159 | +0.155 | -0.062 | **-0.005** | +1.000 |
| `random_ep10` | +0.162 | -0.051 | +0.109 | -0.082 | **+0.202** | +0.829 |
| `random_ep34` | +0.666 | -0.440 | +0.896 | -0.723 | **+1.363** | -0.434 |
| `random_ep100` | +0.873 | -0.682 | +0.920 | -0.782 | **+1.628** | -0.695 |
| `scaffold_ep3` | +0.008 | -0.000 | -0.083 | +0.081 | **-0.078** | +0.989 |
| `scaffold_ep12` | +0.226 | -0.159 | +0.033 | -0.006 | **+0.212** | +0.818 |
| `scaffold_ep32` | +0.620 | -0.376 | +0.882 | -0.666 | **+1.272** | -0.271 |
| `scaffold_ep100` | +0.856 | -0.703 | +0.930 | -0.745 | **+1.617** | -0.678 |

### TLR8 vs TLR9 — `CHEMBL5805` / `CHEMBL5804`

742 shared molecules · `gt_rho` = -0.152 · assays from **val**

| model | `pA`&#124;`gtA` | `pA`&#124;`gtB` | `pB`&#124;`gtB` | `pB`&#124;`gtA` | delta | pred agree |
|---|---|---|---|---|---|---|
| `random_ep1` | -0.023 | +0.141 | +0.154 | -0.027 | **+0.009** | +0.998 |
| `random_ep10` | +0.586 | -0.062 | +0.791 | -0.172 | **+0.805** | +0.020 |
| `random_ep34` | +0.828 | -0.107 | +0.906 | -0.226 | **+1.033** | -0.152 |
| `random_ep100` | +0.916 | -0.120 | +0.944 | -0.199 | **+1.090** | -0.172 |
| `scaffold_ep3` | +0.164 | -0.047 | -0.108 | +0.179 | **-0.038** | +0.944 |
| `scaffold_ep12` | +0.149 | +0.002 | -0.039 | +0.121 | **-0.007** | +0.819 |
| `scaffold_ep32` | +0.168 | -0.103 | -0.099 | +0.169 | **+0.002** | +0.535 |
| `scaffold_ep100` | +0.191 | -0.105 | -0.022 | +0.213 | **+0.031** | +0.622 |

### P78536 vs P08253 — `CHEMBL3706` / `CHEMBL333`

36 shared molecules · `gt_rho` = -0.413 · assays from **val**

| model | `pA`&#124;`gtA` | `pA`&#124;`gtB` | `pB`&#124;`gtB` | `pB`&#124;`gtA` | delta | pred agree |
|---|---|---|---|---|---|---|
| `scaffold_ep3` | -0.091 | -0.093 | -0.012 | -0.128 | **+0.059** | +0.989 |
| `scaffold_ep12` | -0.157 | -0.001 | -0.090 | -0.196 | **-0.025** | +0.133 |
| `scaffold_ep32` | +0.372 | -0.216 | +0.028 | -0.192 | **+0.404** | +0.216 |
| `scaffold_ep100` | -0.084 | +0.133 | +0.004 | +0.011 | **-0.112** | +0.160 |

### P35348 vs P08172 — `CHEMBL229` / `CHEMBL211`

40 shared molecules · `gt_rho` = -0.128 · assays from **train**

| model | `pA`&#124;`gtA` | `pA`&#124;`gtB` | `pB`&#124;`gtB` | `pB`&#124;`gtA` | delta | pred agree |
|---|---|---|---|---|---|---|
| `random_ep1` | +0.048 | +0.009 | +0.020 | +0.047 | **+0.006** | +0.999 |
| `random_ep10` | +0.520 | +0.325 | +0.670 | +0.350 | **+0.257** | +0.764 |
| `random_ep34` | +0.713 | +0.100 | +0.752 | +0.066 | **+0.649** | +0.249 |
| `random_ep100` | +0.862 | -0.050 | +0.836 | -0.039 | **+0.893** | +0.030 |
| `scaffold_ep3` | +0.183 | +0.353 | +0.429 | +0.145 | **+0.057** | +0.978 |
| `scaffold_ep12` | +0.694 | +0.217 | +0.809 | +0.140 | **+0.573** | +0.392 |
| `scaffold_ep32` | +0.887 | -0.043 | +0.896 | +0.029 | **+0.898** | +0.103 |
| `scaffold_ep100` | +0.934 | -0.164 | +0.939 | -0.039 | **+1.038** | -0.099 |

### O75116 vs P29597 — `CHEMBL2973` / `CHEMBL3553`

50 shared molecules · `gt_rho` = -0.093 · assays from **val**

| model | `pA`&#124;`gtA` | `pA`&#124;`gtB` | `pB`&#124;`gtB` | `pB`&#124;`gtA` | delta | pred agree |
|---|---|---|---|---|---|---|
| `random_ep1` | -0.332 | +0.071 | +0.062 | -0.336 | **-0.003** | +0.995 |
| `random_ep10` | +0.103 | +0.479 | +0.736 | -0.230 | **+0.296** | +0.697 |
| `random_ep34` | +0.784 | +0.156 | +0.914 | -0.156 | **+0.849** | +0.108 |
| `random_ep100` | +0.750 | +0.219 | +0.831 | -0.267 | **+0.814** | +0.165 |

## Limitations

- **The val cells rest on 10 and 12 pairs.** That is the binding constraint on every
  held-out conclusion here, and it is a property of the data: only 10 scaffold-val
  target pairs share >= 30 molecules with disagreeing ground truths.
- **Pseudo-replication in scaffold val.** Three of the ten pairs share target
  `CHEMBL240`, all with n=43 and deltas 0.47 / 0.41 / 0.54. They are not independent
  observations, and they are the main driver of the *unweighted* scaffold-val mean.
  Weighting by molecule count cuts that mean from +0.188 to +0.085.
- **Pairs were de-duplicated by taking the most-disagreeing assay pair per target**
  pair. Since `gt_rho` is estimated from noisy pChEMBL, that selection makes the
  reported `gt_rho` slightly more negative than the truth. It does not bias `delta`,
  which never uses `gt_rho`.
- Proteins sharing molecules are often homologues, so this measures fine-grained
  discrimination within a family, not coarse target recognition.

## Files

| file | contents |
|---|---|
| `per_pair.csv` | every (model, pair): all four cross-Spearmans, delta, agreement, `gt_rho` |
| `fig_delta.png` | weighted delta vs training epoch, by split and molecule novelty |
| `run_cross_protein.py` | the experiment |
| `models.json` | the eight checkpoints |

