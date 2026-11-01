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

The selection metric `eval_macro_spearman` is an unweighted mean of per-assay
correlations. `eval_spearman` is instead an alias for the observation-count
weighted mean; neither is a correlation pooled over observations. Groups need
at least three observations and pChEMBL span≥0.5. Thus two-item assays do not
enter this headline metric. Constant label groups are excluded; tied ranks are
averaged; constant predictions give undefined correlations and reduce coverage.
Source: `reward_model/training/evaluation.py`.

The original prediction exports retained raw float64 labels, while the Trainer
evaluates float32 tensor values. On the identical saved best-test predictions,
raw labels give macro Spearman 0.1770244006 across 3,971 eligible assays; rounding
labels as the Trainer does gives 0.1768796098 across 3,972 assays, exactly matching
the native test result. Label precision fully explains this discrepancy; it is
not evidence of an inference-batch effect. The original post-hoc metric,
noise-band and scorer-agreement reports retain their raw-label semantics.
Separate float32-label analyses are complete, without changing weights,
predictions or checkpoint selection. All 93 export/scorer comparisons match the
production metric's eligible groups and per-assay correlations to 1e-14;
the selected test macro also matches the historical Trainer result exactly.
The corrected reports are in `metrics_float32/`, `noise_bands_float32/` and
`scorer_agreement_float32/`. Sources:
`tier1/label_precision_complete.json`,
`tier1/supervision_composition_v2/complete.json:initial_reference_label_precision`.

| Test composition | A | B |
|---|---:|---:|
| Assays | 16,431 | 9,578 |
| Median / mean assay size | 3 / 11.73 | 2 / 14.30 |
| Eligible assays | 8,278 (50.38%) | 3,972 (41.47%) |
| Mean within-assay pChEMBL variance | 0.3046 | 0.1809 |
| Exactly tied label pairs | 6.20% | 5.65% |
| Rows with multiple raw measurements | 3.22% | 3.91% |

B has a smaller median assay but a larger mean due to its tail. The differences
in eligibility, variance and near-ties preclude interpreting the raw split gap
as a pure similarity effect. Matched neural reweighting remains pending; the
completed fixed depth-3 analysis below supplies supplementary evidence.
Source: `tier1/data/composition.json`.

The actual similarity relation is binary exact scaffold identity across all
training molecules. In A test, 37,174/192,687 rows (19.29%) and 34,625/162,361
unique molecules (21.33%) have no training scaffold match. A further 26,417
rows contain unseen molecule identities with a shared scaffold. All 136,956 B
test rows lack both molecule-identity and scaffold matches. This binary relation
preserves the splitter's pooled acyclic and private unparseable nodes; it is not
a graded chemical-distance measure.

Saved depth-3 predictions give these test macro Spearman values:

| Evaluation cohort | Eligible assays | Macro Spearman |
|---|---:|---:|
| A, complete assays with all scaffolds shared | 5,548 | 0.1390 |
| A, complete assays with mixed scaffold support | 2,497 | 0.1071 |
| A, complete assays with all scaffolds unseen | 233 | 0.1067 |
| A, shared-scaffold rows within each assay | 7,556 | 0.1342 |
| A, unseen-scaffold rows within each assay | 1,738 | 0.0669 |
| B, complete assays, all scaffolds unseen | 3,972 | 0.0607 |

The within-assay views reapply eligibility after restricting rows. Assays can
appear in both views, so their counts are not additive. The unseen-row view is
close to B, but it changes the evaluated groups. Complete all-unseen A assays
instead give 0.1067 with a wide whole-assay interval [0.0564, 0.1559]. These
cohorts do not identify a causal leakage share or establish neural performance.

Composition weights were fixed without prediction scores, using group-size,
positive-fraction and mean-pChEMBL bins. A supports 3,964/3,972 eligible B test
assays (99.80%). On this common support, A falls from 0.1281 to **0.1154** after
weighting; B is **0.0613**, leaving a descriptive gap of **0.0541**. The original
full-partition values are A 0.1285 and B 0.0607. The paired A-unit bootstrap
change is -0.01266, interval [-0.01551, -0.00969]; effective weighted A count is
7,067 groups and defined coverage remains 100%. Validation shows the same
direction: weighted A 0.1221 versus common-support B 0.04884.

Cells match exactly in mass, but residual differences remain inside coarse bins:
weighted mean eligible-assay size is 26.11 on A versus 31.93 on B; mean label span is
2.173 versus 1.954. Thus this adjustment does not remove every composition
difference. Intervals condition on one fit, fixed assignments and empirical
weights; they omit training-seed and weight-estimation uncertainty. No additional
LambdaMART fit or tuning was performed. Neural strata/reweighting and additional
realization comparisons remain pending. Sources:
`tier1/similarity_composition/design/complete.json`,
`tier1/similarity_composition/lambdamart_depth3/complete.json`, and
`tier1/similarity_composition/verification.json` on Atum. All source/artifact
hashes were verified; an independent weighted sum reproduced both partitions,
and maximum cell-mass error was below 7e-18.

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
These noise-band values are unchanged after the label-precision correction.
Sources: `tier1/metrics_float32/best_test.json`, `tier1/noise_bands_float32/`.

Six categorical variance-component fits were identifiable algebraically, but
several fitted noise/residual/interaction components reached the nonnegative
boundary. In particular, zero fitted training noise conflicts with observed
replicate disagreement. These descriptive covariance fits do not establish
noiseless data, absent biological interactions or an architectural ceiling.
Direct marginal and replicate-contrast checks accompany each fit.
Source: `tier1/variance/`; method: [REHE](https://pmc.ncbi.nlm.nih.gov/articles/PMC8604792/).

An assay-balanced sensitivity fit changes the estimated within-assay interaction
contribution from 0 to 0.1159 pChEMBL² on A test and from 0.1716 to 0.0568 on B
test. For B training, balancing improves fitted marginal variance from 3.9844
to 2.0749 (observed 1.9578), but the fitted measurement-noise variance remains
far below direct replicate disagreement. Thus Q19 has no stable numerical
partition under this additive constant-variance model. These fits cannot justify
an interaction ceiling or choosing an architecture from a small interaction share.
Raw-measurement noise also cannot be read as the uncertainty of a median label.

## Ranking eligibility and binary supervision

Using the current partition-local binary labels:

| Partition | Assays | Headline eligible | All-positive assays | All-negative assays | Same-side labels among headline-eligible assays |
|---|---:|---:|---:|---:|---:|
| A train | 131,450 | 48.59% | 63,190 | 28,715 | 44.50% |
| A validation | 16,431 | 48.67% | 8,069 | 3,537 | 45.28% |
| A test | 16,431 | 50.38% | 7,791 | 3,524 | 43.63% |
| B train | 145,454 | 49.64% | 69,799 | 30,457 | 43.87% |
| B validation | 9,280 | 42.77% | 4,683 | 2,443 | 50.09% |
| B test | 9,578 | 41.47% | 4,919 | 2,664 | 54.20% |

Under headline precision, most training eligibility exclusions come from assays with fewer than three items:
62,516 on A and 67,994 on B. A further 5,067/5,259 assays have enough items
but insufficient float32 label span. Eligible training assays have median size
11 versus 1 for ineligible assays on both splits. Mean within-assay variance is
0.5360 versus 0.0863 on A and 0.5492 versus 0.0918 on B. Target composition also
differs: the eligible/ineligible target-frequency distributions have total
variation 0.2904/0.2908, and 3,420/3,393 of 7,873 training targets have no eligible
ranking assay. These are descriptive selection differences. The headline metric
uses the same nominal eligibility rule, so it does not directly score most
excluded assays. The sampler reads float64 cached labels; headline evaluation
casts labels to float32. There are two training-assay inclusion disagreements
on A and one on B at the exact span boundary. The sampler therefore has 72,202
eligible B training assays, versus 72,201 under headline precision. A's two
disagreements cancel in the total count. These tiny boundary effects are kept
separate from the broad size/variance selection described above.

Binary labels give no ordering preference between members with identical labels.
This affects 54.20% of ranking-eligible B test assays, but does not prove BCE
cannot help the shared representation; the loss ablations must test that.
For the threshold at pChEMBL 6, 35.58%/32.21% of A/B test rows have their
partition-local pair median within the training replicate-RMS band
(0.78055/0.76605). This is threshold proximity, not a calibrated probability of
label error or uncertainty interval for median labels.
Source: `tier1/supervision_composition_v2/complete.json`, with six-partition group,
target and unit tables. This supplements the earlier original-label audit.

## Baseline ladder

All results below use frozen pretrained features or entity identities, training
measurements only, and validation-only selection. A uses partition-local binary
labels; continuous labels are unchanged. The Spearman eligibility rule is the
original n≥3 and label span≥0.5. Undefined correlations are excluded, with their
coverage reported explicitly.

| Baseline | A test macro Spearman | Defined eligible assays | B test macro Spearman | Defined eligible assays |
|---|---:|---:|---:|---:|
| Molecule-only linear ridge | 0.0775 | 100% | 0.0624 | 100% |
| Same-protein scaffold-nearest lookup | 0.2572 | 79.72% | undefined | 0% |
| Independent per-protein ridge | 0.2419 | 99.83% | 0.0726 | 99.52% |
| Two-step ridge | 0.1034 | 100% | 0.0488 | 100% |
| Identity-based linear matrix filter | 0.4207 | 93.32% | undefined | 0% |
| Paired Kronecker ridge | 0.1690 | 100% | 0.0631 | 100% |
| Protein-shuffled Kronecker ridge | 0.0662 | 100% | 0.0596 | 100% |
| Low-rank bilinear inductive completion | 0.2126 | 100% | 0.0779 | 100% |
| LambdaMART, user-fixed depth 3 | 0.1285 | 100% | 0.0607 | 100% |

The user ended the larger LambdaMART search and chose the completed depth-3
configuration for the A/B comparison. All four remaining search processes were
verified terminated; the other completed/partial artifacts are preserved. This
is a fixed-configuration result, not the winner of the planned five-setting
search. Tree counts remained validation-selected: 500 on A and 270 on B.
Test whole-unit bootstrap intervals are [0.11960, 0.13733] on A and
[0.04507, 0.07568] on B, conditional on these fits and held-out units. The same
A>B direction appears in this frozen-feature tree baseline; that observation
does not isolate the cause of the split gap. CPU validation rescoring reproduced
A exactly and differed from B's saved fit-time score by -1.4163e-8, with unchanged
coverage. That measured discrepancy is recorded without changing the selected
checkpoint. Sources: `tier2/lambdamart/user_decision.json` and both split
`complete.json`, `val/` and `test/` artifacts; final checksums were verified.

The molecule-only and independent-task models each test five regularization
values per split. Two-step ridge tests the 25 combinations of the two sides'
five values. It fits measured pairs in stage one, then regresses those fitted
functions on protein features. This is an explicit sparse-data adaptation, not
a complete-matrix formula applied to artificial zero labels. Sources:
`tier2/linear/`, `tier2/independent_task_ridge/`, `tier2/two_step_ridge/`.

The lookup uses the actual split relation: exact scaffold identity, with all
acyclic molecules sharing one scaffold node. Similarity ties are averaged over
unique training molecules measured for the query protein. On B every nearest
similarity is zero, so it reduces to a protein prior. Its within-assay ordering
accuracy is 0.5. Source: `tier2/nearest_scaffold_lookup/`.

The matrix filter uses the smoother kernel
`(1 + same_protein) * (1 + same_molecule)`, fitted only to observed measurements
with their original frequencies. Five ridge values are selected on validation.
Its A test bootstrap interval is [0.4111, 0.4304], conditional on the held-out
assays and the 93.32% defined coverage. On B, unseen molecules receive only a
protein-specific score: pairwise ordering accuracy is 0.5. A strong A result
is therefore possible without learning transferable molecular features. This
does not estimate the causal leakage contribution to a neural model's gap.
Source: `tier2/linear_matrix_filter/`; method family:
[Stock et al.](https://arxiv.org/html/1803.01575v1).

The protein-only linear BCE model has validation/test AUROC 0.6687/0.6696 on A
and 0.6452/0.6493 on B. The lookup's B protein-prior scores have validation/test
AUROC 0.6915/0.7060 while carrying no within-assay order. Classification can
therefore reward target base rates. Comparing these AUCs does not identify a
fraction of the full model's signal attributable to those priors; Q32 still
needs the matched deep-model comparison on A.

Kronecker ridge fits the exact augmented linear product kernel on observed
pairs, using a matrix-free solve with verified relative residual below 2e-7.
It uses no completed interaction matrix or random-feature approximation. The
shuffled model is retrained after independently permuting protein assignments
across whole assays within each partition. Group-level protein frequencies,
molecule inputs and labels are preserved. Both versions use the same five-value
regularization grid and select on their own validation macro Spearman. The
comparison covers one permutation per partition, not permutation uncertainty.
Sources: `tier2/kronecker_ridge/`, `tier2/protein_shuffled_kronecker/`.
The paired-minus-shuffled test difference is 0.1029 on A (paired whole-assay
bootstrap interval [0.0920, 0.1138]) and 0.0035 on B (whole-component interval
[−0.0157, 0.0221]). This kernel baseline shows a clear A advantage for correct
protein assignments; the B contrast is inconclusive. These intervals condition
on the fitted models and permutations. Source: `tier2/protein_shuffle_comparison.json`.

Bilinear completion fits `x.T A B.T y + b` only on measured pairs, using the
same frozen, standardized feature map as Kronecker ridge. Nine rank/penalty
settings per split received 3,000 L-BFGS iterations each. Validation selected
rank 128 on A and rank 8 on B, both at penalty 1e-6. Selected validation scores
are 0.2190/0.0782; test scores are 0.2126/0.0779, with whole-unit bootstrap
intervals [0.2035, 0.2212] and [0.0628, 0.0933], conditional on the fitted
models and split. The saved coefficient matrices reproduce validation
selection exactly, avoiding near-tie changes from a new
CPU factor multiplication. Both selected fits reached the iteration limit
with maximum gradients 1.07e-4 and 7.80e-6, above the 1e-7 target. These are
finite-budget model-family baselines, not global optima or signal ceilings.
The initial 300-iteration fits are retained; extending every setting was
motivated by training-objective/gradient diagnostics. Source:
`tier2/bilinear_completion_3000iter/{A,B}/complete.json`.

The bilinear manifest's `git_commit` was captured at completion. The baseline
entrypoint and its project-local dependencies were verified unchanged between
the launch checkout c920520 and completion checkout 355d147. Future runs
capture the launch commit and source hashes before fitting.

## Agreement between scoring pathways

All 31 saved B prediction exports were scored for within-assay agreement using
Trainer-compatible label precision and average ranks, including tied predictions.
The correction preserves the validation agreement values below.
On the same 3,969 eligible validation
assays, every pathway pair was defined at all three epochs below:

| Validation scorer pair | Epoch 1 | Best checkpoint, epoch 6 | Epoch 15 |
|---|---:|---:|---:|
| Ranking head / BCE logit | 0.6798 | 0.6363 | 0.6017 |
| Ranking head / contrastive cosine | 0.0743 | 0.2359 | 0.2341 |
| BCE logit / contrastive cosine | 0.0958 | 0.2759 | 0.2769 |

Ranking–BCE agreement declines by 0.0781 from epoch 1 to 15 (whole-component
bootstrap interval for the change [−0.0985, −0.0600]). Ranking–cosine agreement
instead rises by 0.1598 ([0.1321, 0.1891]). Thus the evidence does not show
general divergence among all three scorers. Despite increasing agreement with
the other heads, cosine's correlation with the affinity labels changes only
from 0.0653 to 0.0749; its change interval [−0.0085, 0.0270] includes zero.
On the fixed training probe, ranking–BCE agreement increases from 0.6452 to
0.6776 over the same mutually defined eligible assays. These are correlations
of predictions, not gradient measurements or proof of objective interference.
The intervals condition on one fitted seed and scaffold realization. Q31 remains
partial until the matched A trajectory is available. Source:
`tier1/scorer_agreement_float32/complete.json` and per-assay trajectory tables.

## Loss and ranking trajectories

In the fresh B run, validation BCE loss reaches its minimum at epoch 2 (0.44531),
ranking loss at epoch 5 (2.79392), and contrastive/total loss at epoch 15
(5.83497/6.68790). At epoch 15, BCE is 0.48300 and ranking loss is 2.92871.
Classification AUROC nevertheless improves from 0.73865 at epoch 2 to 0.77202
at epoch 15. Decreasing contrastive loss therefore masks worsening ranking loss
in the aggregate; the different terms do not share one turning point.
The first adjacent-epoch increases occur at 2→3 for BCE, 3→4 for ranking and
6→7 for contrastive/total loss. These local fluctuations are not themselves
statistical change points or evidence of sustained deterioration.

The fixed 2,048-assay training probe's eligible macro Spearman rises from
0.18336 at epoch 1 to 0.53189 at epoch 15. Validation peaks at 0.18460 in epoch 6
and ends at 0.17561. Later improvements on the training probe transfer poorly
in this run. These observations do not isolate encoder drift, loss conflict or
another cause, and matched A/multi-seed comparisons remain pending.

On the same 3,969 eligible validation assays as Spearman, linear-gain NDCG@1
and NDCG@3 peak at epoch 11, while NDCG@5 and Spearman peak at epoch 6.
NDCG@1/@3/@5 at epoch 6 are 0.59687/0.66438/0.71182; at epoch 15 they are
0.59098/0.66322/0.70751. Thus top-ranking quality has a later local maximum,
but no sustained improvement through epoch 15. Averaging NDCG over all
nonconstant assays instead changes the top-1/top-3 peak to epoch 8, illustrating
why cohort definitions must be aligned before comparing objectives. These
descriptive extrema do not establish significant differences or objective conflict.
Source: `tier1/baseline_trajectory_summary_v2.json`.

The learned logit scale rises from 13.5999 to 18.4685 to 22.5461 at epochs
1/6/15. Validation contrastive loss falls from 8.54974 to 6.00606 to 5.83497;
rescoring the same representations at fixed scale 13 gives 8.52970, 5.82207 and
5.55379. Candidate identities are fixed throughout, including in the original
validation procedure. Sharpening raises loss relative to fixed-scale rescoring,
but there is no sustained total contrastive-loss rise in this fresh run.
This does not predict the outcome of retraining with fixed temperature.
Sources: `baseline_scaffold_seed42_15ep/checkpoint-13155/trainer_state.json`,
`tier1/contrastive/trajectory.json`, `tier1/scorer_agreement_float32/complete.json`.

The matched A reference (`full__A__seed42__lr1p0`, W&B `xffenit7`) now has
native validation and fixed training-probe audits for epochs 1–3. These
cover 195,618 validation rows / 7,997 eligible assays and a fixed 39,740-row /
2,048-eligible-assay training probe. They reproduce both native headline metrics
and preserve the Trainer's float32 label semantics.

| A epoch | Validation macro rho | Training-probe macro rho | Validation BCE loss | Ranking loss | Contrastive loss |
|---|---:|---:|---:|---:|---:|
| 1 | 0.16488 | 0.16214 | 0.52074 | 2.66708 | 4.86771 |
| 2 | 0.21244 | 0.22805 | 0.48617 | 2.61290 | 3.32545 |
| 3 | 0.24317 | 0.27084 | 0.46223 | 2.57423 | 2.72154 |

Validation AUROC rises 0.72635→0.77144→0.79702 and the contrastive scale rises
13.5412→14.5521→15.5213. On the same eligible validation assays, NDCG@1/@3/@5
rises from 0.59835/0.66331/0.71098 at epoch 1 to 0.64363/0.69856/0.74012 at
epoch 3. Epoch-3 BCE-logit and cosine-only macro Spearman are 0.19377 and
0.15481. Positive temperature calibration fitted on that validation set gives
T=1.04318 and reduces its log-loss from 0.46223 to 0.46198; this is an in-sample
fit, with no test transfer.

At epoch 3, complete shared-scaffold, mixed and unseen-scaffold validation
assays score 0.24534/0.24195/0.19703 (5,405/2,402/190 eligible assays).
Restricting to unseen-scaffold rows within assays instead gives 0.20020 on
1,608 eligible assays; these cohorts differ. Fixed composition weighting
changes A common-support validation macro from 0.24348 to 0.23969. This
adjustment uses the B validation label mix, not B predictions or test scores.

Across epochs 1→3, validation ranking–BCE agreement changes 0.63066→0.62831
on the same 7,995 mutually defined eligible assays; the paired whole-assay
interval for the change is [-0.01208, 0.00755]. Ranking–cosine agreement rises
0.12360→0.26288 on 7,997 assays, with change interval [0.12696, 0.15126].
Thus the observed prefix does not show uniform scorer divergence.

These three epochs establish early improvement, not a final optimum or an
overfitting turning point. Training-probe intervals are descriptive. All
unit intervals condition on this single fitted trajectory; full-horizon,
post-selection test, seed and matched B comparisons remain pending, as do A
directional/fixed-temperature contrastive and representation analyses.
Source: `tier1/native_trajectories/full__A__seed42__lr1p0/6349989d65c2f7ef/`
on Atum, including the immutable `trajectory_through_epoch03.json`, per-assay
metric tables and `verification_through_epoch03.json`. Independent average-rank
calculations from the native prediction arrays reproduced all 166,311
per-assay/scorer correlations, including undefined cases, to 1e-14. Brute-force
pairs reproduced 162 sampled noise-band/group/scorer combinations; direct
weighted means reproduced all three
composition-standardized validation scores. All source/artifact hashes passed.

## Protein coverage in the B test set

All 1,723 test proteins have training observations. The number of distinct
training assays per test protein ranges from 1 to 1,303, with median 24.
Weighting proteins by their number of test assays gives median 127 instead.
The complete target list and counts are saved in `tier1/target_coverage/test_targets.parquet`.

| Training assays for the protein | Eligible test assays | Test macro Spearman | Whole-component interval |
|---|---:|---:|---:|
| 1 | 71 | 0.0539 | −0.0701 to 0.1748 |
| 2–5 | 149 | 0.0849 | 0.0018 to 0.1716 |
| 6–20 | 311 | 0.1829 | 0.1324 to 0.2345 |
| 21–100 | 1,184 | 0.1626 | 0.1350 to 0.1891 |
| >100 | 2,257 | 0.1935 | 0.1710 to 0.2148 |

Every eligible assay has defined predictions. The strata reconstruct the
selected checkpoint's overall test macro exactly. Low-support point estimates
are lower and uncertain; the pattern is not monotone across all bins. Assay
composition also differs across bins, so this is not a causal coverage effect.
Intervals condition on one fitted model and split. Source:
`tier1/target_coverage/complete.json` and per-assay support table.

## Sampling and contrastive negatives

The configured first epoch contains 12,820 local microbatches on A and 14,020
on B, with 12 dataset items per device. These are bounded assay lists plus
classification-only rows. Ranking-eligible lists average 55.83%/56.25% of each
batch, with standard deviations 14.68/14.74 percentage points. Both splits'
5th-95th percentile range is 33.33%-83.33%; each has one batch with no ranked
list. This audits the actual sampler implementation and first-epoch ordering,
not a saved trace of historical device execution.

The reference already masks cross-assay same-protein and duplicate-query-
molecule negatives, as well as other candidates in an anchor's own assay.
However, a molecule sampled under another protein can be a known positive for
the query protein in a different training assay. In 512 evenly spaced batches
per split, such entries make up 0.530%/0.714% of protein-to-molecule negative
comparisons on A/B, or 0.437/0.576 per active anchor. Reverse-direction fractions
are 0.441%/0.657%. A numerical gradient check confirms that the current loss
pushes these entries down. Known positives are defined only by curated training
observations above pChEMBL 6; unmeasured positives are unknown. These counts
justify testing a global training-positive mask but do not demonstrate a
performance improvement from it.

Candidate protein frequency is skewed: the top 1% of 7,873 proteins supply
30.07%/30.64% of A/B assay-list opportunities. The median is two per protein;
maxima are 1,587/1,589. No sampling-probability correction is applied. Frequency
skew alone does not establish a learned popularity shortcut. Sources:
`tier1/sampling/{A,B}/complete.json`, with per-batch and per-protein tables.

## Distribution shift

The actual indivisible units have markedly different size distributions:

| Partition | Unit definition | Units | Median rows | 99th percentile rows | Maximum rows |
|---|---|---:|---:|---:|---:|
| A train | assay | 131,450 | 3 | 122.0 | 49,020 |
| A validation | assay | 16,431 | 3 | 143.4 | 1,409 |
| A test | assay | 16,431 | 3 | 137.7 | 1,147 |
| B train | scaffold–assay component | 570 | 6 | 620.0 | 1,661,628 |
| B validation | scaffold–assay component | 4,032 | 5 | 431.4 | 7,382 |
| B test | scaffold–assay component | 4,034 | 5 | 447.4 | 5,683 |

A randomly reserves one training anchor assay per target/sequence, then randomly
assigns the remaining whole assays; it does not preferentially select large
assays. B forces its giant component and 569 target-coverage rescue components
into training, then balances the remaining whole units between validation/test
by row count and positive rate. The giant alone supplies 98.70% of B training
rows. B assignment therefore is not random sampling of comparable training and
held-out units. These counts establish construction asymmetry, not a numerical
causal share of the performance gap. Sources: `tier1/supervision_composition_v2/`,
the original split manifests, `random_assay.py`, and `scaffold_component_split.py`.

The E41 whole-unit learning-curve data are prepared, with no fits performed:

| Requested unit fraction | A rows retained | B rows retained | A binary labels changed from full-training labels | B binary labels changed |
|---|---:|---:|---:|---:|
| 10% | 9.51% | 98.86% | 2,141 | 0 |
| 25% | 25.93% | 99.01% | 4,247 | 0 |
| 50% | 49.85% | 99.52% | 5,732 | 0 |
| 100% | 100% | 100% | 0 | 0 |

Each subset contains whole units from a fixed nested permutation. A binary-label
change means medians were correctly recomputed using only included training
measurements, not evidence that a model improved. At 10% units, A retains targets
covering 93.46% of test assays; B retains 100% target coverage at every fraction.
B's selected prefixes all contain the giant component, so these curves cover
a narrow 98.86–100% row regime. A flat B curve would not establish broad data
saturation. Source: `learning_curves/plan.json` and the eight pinned view manifests.
Production-data CPU checks passed for the 10% A/B views, including row identities,
subset-local binary labels, shared token buffers and the actual sampler. This
qualifies data loading; it does not establish training or performance results.
Source: `learning_curves/qualified_data_views.json`.

E40's fixed-test leak comparison has data-construction evidence, not model
results. Version 2 matches whole-assay protein identity, size, BCE-positive
count, strict contrastive-active count, replicate histogram and ranking
eligibility. The first version matched BCE positives but changed contrastive
actives by -106 on A and +15 in the B proposal; those artifacts remain recorded
and are superseded for future fitting. Both versions have byte-identical test
tables, so the additional constraint did not change test selection.

| E40 construction | Training rows per arm | Matched assay exchanges | Test assays exposed to training scaffold relatives |
|---|---:|---:|---:|
| A, whole assays | 351,451 | 9,261 | 11,786 of 16,431 |
| B, whole training components | 1,683,487 | 0 | 0 of 7,724 |
| B, whole-assay exception proposal, not selected | 1,683,487 | 806 | 2,279 of 7,724 |

A exchanges 14,157 rows and exposes 50,512 test rows to scaffold relatives;
17,186 test rows have exact molecule overlap. Both arms retain 29,471 training
assays, 3,682 proteins, 270,755 BCE positives and 267,768 contrastive actives.
The production cache views passed identity, subset-label and sampler checks:
35,136 sampler items and 183 optimizer steps per epoch in each arm. Protein
support is identical between arms, including 8,951 test rows with unseen
proteins. Performance comparisons must report that common limitation.

B reserves one label-blind whole donor assay from each multi-assay original
test component and fixes the remaining 7,724 test assays before matching.
All 570 training components contain at least one protein absent from this donor
pool, preventing whole-component removal with unchanged protein support for
this construction. The optional whole-assay proposal exchanges 1,529 rows,
partially retains nine training components and exposes 5,494 test rows, including
2,897 exact molecule overlaps. It is awaiting the user's choice; no B exception
training is authorized by these manifests. Both B arms match 1,143,159 BCE
positives and 1,123,169 strict contrastive actives. Sources:
`leak_design_v2/complete.json`, `leak_design_v2/views/A/qualified.json`, and
`leak_design_assay_proposal_v2/complete.json` on Atum. All recorded artifact
checksums were verified after the CPU jobs completed.

The original scaffold splitter's seeds 42, 43 and 44 produce identical component
assignments: the deterministic size-sorted candidate wins all three searches.
They cannot be counted as three realizations. Randomizing input order among
equal-sized components before the same size-priority allocation and 64-restart
balance search produces two distinct additional validation/test allocations.
Against the original, these move 86,592 and 94,118 of the 273,913 held-out rows
(3,987 and 3,962 whole components). All three retain exactly the same 570
training components and 1,683,487 training rows. Validation/test row counts and
raw positive rates are unchanged.

These are dependent reallocations of a fixed held-out pool, with no change to
training support or scaffold geometry. They support a bounded allocation
sensitivity analysis, not independent split draws. Each needs its own validation
checkpoint and learning-rate selection before test scoring. The assignment
audit is complete; performance spread and multi-seed comparisons remain pending.
Source: `tier1/split_realizations/complete.json`.

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
