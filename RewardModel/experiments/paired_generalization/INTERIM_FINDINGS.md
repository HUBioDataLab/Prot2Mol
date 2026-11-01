# Interim evidence — study incomplete

This records completed audits, not the final answer to all 34 questions. Runtime
evidence is under `RewardModel/outputs/paired_generalization` on Atum. The
remaining baselines, controlled multi-seed runs and mechanistic probes remain
in scope. Test results do not select models or hyperparameters.

The fresh scaffold baseline finished 15 epochs. Validation macro Spearman
selected epoch 6 (0.18460); its original test macro Spearman was 0.17688.
Epoch 15 minimized aggregate validation loss but had lower validation Spearman
(0.17561). These are one seed and one scaffold realization, not a completed
method comparison. Source: `baseline_status.json`, checkpoint trainer states.

## Measurement and composition

All 1,957,400 curated observations were reconstructed from 2,032,748 retained raw
measurements with zero count or median mismatches. On the replicated subsets:

| Test partition | Split-half macro Spearman | Spearman–Brown heuristic | Eligible assay coverage |
|---|---:|---:|---:|
| A: random assay | 0.5445 | 0.7050 | 420/8278 (5.07%) |
| B: scaffold component | 0.3780 | 0.5485 | 348/3972 (8.76%) |

The correction assumes parallel measurements and is not validated as a grouped
rank ceiling. These selected subsets cannot establish a ceiling for the full
test partitions. Source: `tier1/replicates/noise_summary.json`.

Training-only RMS replicate differences were 0.78055 (A) and 0.76605 (B) pChEMBL.
Test pairs inside those bands were 60.802% and 61.887%, respectively. At the
shared original ranking margin, log10(3), the fractions were 42.297% and 43.640%.
For the selected B checkpoint, ordering accuracy was 0.5892 without a noise band,
0.6305 outside the original margin and 0.6581 outside the replicate-RMS band.
Those accuracies have different denominators. The all-pairs C−D concordance
decreased from 0.1683 to 0.1471 and 0.1205 as more pairs became ties. A larger
filtered accuracy therefore does not mean the model itself improved.
Sources: `tier1/metrics/best_test.json`, `tier1/noise_bands/`.

Six categorical variance-component fits were identifiable algebraically, but
several fitted noise/residual/interaction components reached the nonnegative
boundary. In particular, zero fitted training noise conflicts with observed
replicate disagreement. These descriptive covariance fits do not establish
noiseless data, absent biological interactions or an architectural ceiling.
Direct marginal and replicate-contrast checks accompany each fit.
Source: `tier1/variance/`; method: [REHE](https://pmc.ncbi.nlm.nih.gov/articles/PMC8604792/).

## Distribution shift

A fixed logistic classifier used group descriptors, with whole units held out
of fitting. Each domain and each unit within a domain had equal fitting weight.

| Comparison | Unit-balanced accuracy | Assay-balanced accuracy | Holm-adjusted permutation p |
|---|---:|---:|---:|
| A train / validation | 0.5170 | 0.5170 | 0.003 |
| A train / test | 0.5181 | 0.5181 | 0.003 |
| A validation / test | 0.4958 | 0.4958 | 1.000 |
| B train / validation | 0.8522 | 0.7765 | 0.003 |
| B train / test | 0.8594 | 0.7787 | 0.003 |
| B validation / test | 0.4966 | 0.4916 | 1.000 |

This detects composition differences, not their causal contribution to ranking
error. Target training coverage had the largest standardized coefficient in
the B train/held-out comparisons. The giant training component makes unit and
assay weighting different estimands; this classifier's holdout excluded that
giant component. Failure to distinguish validation and test is not proof that
their distributions are equal. Source: `tier1/shift/c2st_summary.json`;
method: [C2ST](https://arxiv.org/abs/1610.06545).

B validation-to-test BBSE estimated test positive prevalence at 0.7850, with a
whole-unit bootstrap interval of [0.6696, 0.8789]; observed test prevalence was
0.7902. The confusion matrix condition number was 3.81. This diagnostic assumes
stable class-conditional prediction distributions, and validation also selected
the checkpoint. It is not an independent calibration experiment.
Source: `tier1/shift/bbse_B.json`;
method: [BBSE](https://proceedings.mlr.press/v80/lipton18a.html).

Deployment is unseen generated molecules. That does not establish whether their
scaffolds are unseen. The report must distinguish unseen identity from unseen
scaffold, rather than presume either split alone matches the generator.
