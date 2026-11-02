# Interim evidence — study incomplete

This records completed audits, not the final answer to all 34 questions. Runtime
evidence is under `RewardModel/outputs/paired_generalization` on Atum. The
remaining controlled multi-seed runs and mechanistic probes remain in scope.
Test results do not select models or hyperparameters.

The ledger has 23 complete, 12 partial and 35 pending experiment entries,
including 12 conditional entries. Tier 2 is complete (11/11 entries); the other
three tiers remain incomplete. Of the 34 questions, 11 are answered, fifteen have
answers with limitations, three are partial and five are pending. The latest
closures concern the two completed reference trajectories: BCE/AUROC, validation
temperature fitting, logit scale, checkpoint locations, top-weighted metrics,
contrastive directions and fixed-pool rescoring.
They do not close the controlled multi-seed or selected A test comparisons.

The fresh scaffold baseline finished 15 epochs. Validation macro Spearman
selected epoch 6 (0.18460); its original test macro Spearman was 0.17688.
Epoch 15 minimized aggregate validation loss but had lower validation Spearman
(0.17561). These are one seed and one scaffold realization, not a completed
method comparison. Source: `baseline_status.json`, checkpoint trainer states.

## Measured execution costs — scope unchanged

`resource_audit.py` (commit `27b06d2`, seven focused tests passed) sampled the
completed full A run and B through epoch11. Median checkpoint save intervals are
1,447.16s on A and1,476.56s on B: approximately6.03/6.15h per15-epoch fit at that
cadence. This includes the work between saves, not isolated training kernels;
startup and final post-training work are excluded from the cadence projection.
Applying that full-model pace to the primary plan's140 A+140 B fits gives71.06
four-GPU wall days (6,822 GPU-hours). This is a reference-paced scenario, not an
ETA or bound: frozen/partial/LoRA/loss arms may differ, and exports, selected
evaluation, reporting, dataset ablations, adaptation and conditional work add
other costs. The280 fits comprise168 tuning+112 confirmation fits. The MD requires
matched tuning budgets and at least three seeds; three LR candidates are the
current protocol's choice, not a numeric requirement of the MD.

Archive quota headroom at this sample was784.57GB; home headroom was32.28GB.
The current B retention policy keeps15 epochs plus a selected root model and an
average. Using observed full-model weight sizes and A's four retained payloads
across the entire primary grid yields a weight-only scenario of1.46TB. It is
not an additional allocation or exact disk forecast, but establishes that all
those weights cannot accumulate within current headroom. The scoped run scan
de-duplicates hard links: A's protected trajectory adds6.46GB beyond its2.09GB run
directory; live B used16.44GB including10.92GB of optimizer states. These are not
whole-project storage totals. No files were deleted or policies changed.

Extra B-allocation confirmations range from zero to four per setting, depending
on the union of validation-selected LRs;112 additional fits is only an all-setting
maximum, not a launch request. Their actual cost and the user's time/compute
constraint are needed for the MD's affordability condition. A budget clarification
is pending; the already authorized run and queues continue unchanged. Evidence:
`resource_audits/full_reference_cost/4a2971a2a83f2d3a.json` and
`resource_audit_runtime/27b06d2/{deployment,audit_receipt}.json` on Atum.

The checkpoint-retention dependency review at source commit `8298c8f` confirms
that scalar exports alone do not permit releasing all B epoch weights. Native
allocation-loss evaluation needs all 15 epochs for allocation-selected jobs;
weight averaging needs every checkpoint in its selected window. Later primary
and allocation report verification reopens selected epoch weights and saved
averages, but does not rehash every unselected epoch payload. However, rerunning
the original `retain()` checks every originally preserved weight, and the training
completion receipt hashes the original retention manifest. A future release
procedure must therefore preserve selected weights and all immutable evidence,
explicitly account for released files, and verify the remaining consumers; simply
deleting files or rewriting the original manifest would break those guarantees.
The generic averaging function also reopens its source weights even when its
saved average already exists, so that rerun path must be considered separately.

No cleanup procedure was implemented. At review time the primary full-model LR
selection, allocation selection, selected evaluations and allocation averages
were still absent. Tier 4 bases and any conditional consumers must be included
in a future release check; the initial fresh reference and protected A trajectory
remain excluded from intermediate cleanup. All ten live process identities and
133 pinned queue inputs were verified. This is a source dependency review, not
a successful post-release test or a completed experiment. Evidence:
`resource_audits/checkpoint_retention_review/source_8298c8f.json` on Atum,
SHA256 `6a8f75cd21dcb9dcbebf1beaaef20e31c81064062c7bf8227041ffbab9f3c9d2`.

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
as a pure similarity effect. Fixed depth-3 test reweighting and matched neural
validation reweighting are complete below; selected neural test and repeated
comparisons remain pending.
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
LambdaMART fit or tuning was performed. Selected neural test and additional
seed/realization comparisons remain pending. Sources:
`tier1/similarity_composition/design/complete.json`,
`tier1/similarity_composition/lambdamart_depth3/complete.json`, and
`tier1/similarity_composition/verification.json` on Atum. All source/artifact
hashes were verified; an independent weighted sum reproduced both partitions,
and maximum cell-mass error was below 7e-18.

The two completed full-model seed 42 base-LR references now have matched
composition comparisons at all 15 validation epochs. The fixed cells support
3,962/3,969 eligible B assays (99.82%). Standardizing A lowers its point
estimate at every epoch; the same-epoch terminal comparison and each run's
own native validation-macro peak are kept separate:

| Validation comparison | A common support | A standardized | B common support | Standardized A minus B |
|---|---:|---:|---:|---:|
| Both epoch 15 | 0.34301 | 0.33462 | 0.17925 | 0.15537 |
| A epoch 14 / B epoch 6, own macro peaks | 0.34337 | 0.33486 | 0.18420 | 0.15066 |

At epoch 15, the paired A-unit weighting change is -0.00839, with conditional
interval [-0.01223, -0.00516]. Residual imbalance remains: weighted mean assay
size is 27.18 on A versus 31.79 on B, and label span is 2.226 versus 2.020.
A's unseen-scaffold rows within assays score 0.19670 (1,608 eligible assays),
compared with B's 0.17975 (3,969); intact all-unseen A assays instead score 0.22976
(190). These are distinct cohorts, not evidence that similarity explains a
specific fraction of the gap.

The analysis uses validation labels for descriptive standardization. Its
intervals condition on one fitted seed and fixed empirical weights; they omit
seed, split, weight-estimation and checkpoint-selection uncertainty. No joint
cross-split gap interval is asserted, because A/B partition roles can overlap.
The peak comparison is in-sample and does not select an LR or test checkpoint.
No model was fitted and no test predictions were read. All 803 recorded input
hashes, the result digest and execution receipt were checked; independent
weighted point estimates and intact-assay stratum means were reproduced.
Within-assay restricted-row statistics and bootstrap intervals reuse the native
analysis and were not independently reconstructed by this report. Evidence:
`tier1/reference_composition/139902be79aacd3f/complete.json` on Atum,
SHA256 `1f0d06faf0aa2623365d5f0b28bb6207fd0d0425e9896bbadb32307388109acd`,
and `reference_composition_runtime/67e2d88/execution.json`.
E08/E09 and Q13 remain partial pending selected neural test and repeated results.

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

The MD's final Tier-2 rung explicitly says: "Score Split B groups with the
current model's contrastive similarity alone." At the jointly trained B model's
validation-selected epoch-6 checkpoint, cosine alone gives test macro Spearman
**0.06623**, with whole-component interval **[0.05048, 0.08216]**, compared with
the ranking head's **0.17688 [0.15991, 0.19362]**. Both are defined on all 3,972
eligible test assays. These are two readouts of the same fitted model, not a
contrastive-only training experiment or a causal test of removing cross-attention.

E32 is therefore complete under the explicit Tier-2 Split-B requirement;
the separate both-split pathway analysis remains partial under E53. A current
check of the retained weights and all 15 validation evaluations confirms the
unchanged epoch-6 selection. Independent average-rank calculations reproduced
both readouts for all 9,578 test assays (maximum difference 2.23e-16), and
independent unit-count aggregation reproduced the 2,000-draw intervals while
retaining all 4,034 components, including those with no eligible assays.
Intervals condition on this fitted seed, selected checkpoint and held-out
components; they do not include seed or selection uncertainty. No fitting,
inference or new checkpoint selection occurred. Sources:
`tier1/metrics_float32/best_test.json`, the saved `predictions/best_test/` export,
and `RewardModel/experiments/paired_generalization/evidence/factorized_B_scope_verification.json`.

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
The intervals condition on one fitted seed and scaffold realization. Together
with the complete A trajectory below, they answer Q31 descriptively; controlled
repeated and selected-test pathway comparisons remain unfinished. Source:
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
another cause; matched controlled and multi-seed comparisons remain pending.

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

The matched A reference (`full__A__seed42__lr1p0`, W&B `xffenit7`) completed
all 15 epochs and 12,030 optimizer steps on GPUs 0,2,5,6 with batch12×4×4.
Native validation and fixed training-probe audits cover all 15 epochs. These
cover 195,618 validation rows / 7,997 eligible assays and a fixed 39,740-row /
2,048-eligible-assay training probe. They reproduce both native headline metrics
and preserve the Trainer's float32 label semantics.

| A epoch | Validation macro rho | Training-probe macro rho | Validation BCE loss | Ranking loss | Contrastive loss |
|---|---:|---:|---:|---:|---:|
| 1 | 0.16488 | 0.16214 | 0.52074 | 2.66708 | 4.86771 |
| 2 | 0.21244 | 0.22805 | 0.48617 | 2.61290 | 3.32545 |
| 3 | 0.24317 | 0.27084 | 0.46223 | 2.57423 | 2.72154 |
| 4 | 0.26145 | 0.30188 | 0.46034 | 2.54027 | 2.42469 |
| 5 | 0.27953 | 0.33753 | 0.44381 | 2.50691 | 2.20079 |
| 6 | 0.28843 | 0.36608 | 0.44842 | 2.49957 | 2.07677 |
| 7 | 0.30225 | 0.38964 | 0.44339 | 2.47470 | 1.96870 |
| 8 | 0.31324 | 0.40509 | 0.44007 | 2.46072 | 1.87887 |
| 9 | 0.31908 | 0.42791 | 0.43174 | 2.45581 | 1.80663 |
| 10 | 0.32604 | 0.45533 | 0.43004 | 2.44086 | 1.76385 |
| 11 | 0.33268 | 0.46520 | 0.42968 | 2.42924 | 1.72345 |
| 12 | 0.33637 | 0.47843 | 0.42907 | 2.42858 | 1.69640 |
| 13 | 0.33915 | 0.48734 | 0.42758 | 2.41911 | 1.67541 |
| 14 | 0.34368 | 0.48929 | 0.42874 | 2.41859 | 1.65916 |
| 15 | 0.34332 | 0.49208 | 0.42772 | 2.41737 | 1.65268 |

Validation AUROC rises 0.72635→0.84629 across epochs 1→15 and the contrastive
scale rises 13.5412→21.8558. BCE first increases at 5→6, then reaches its
minimum at epoch 13 (0.42758). It rises at 14 and partly recovers to 0.42772
at 15. Ranking and total contrastive loss reach their minima at 15. These
adjacent fluctuations do not establish a causal overfitting mechanism.

At the macro-selected checkpoints, neural validation AUROC is 0.84520 for A
and 0.76583 for original B. The corresponding protein-only frozen-feature
baselines give 0.66874 and 0.64521. Relative to chance AUROC 0.5, those baselines
recover about 49% and 55% of the reference models' excess AUROC. This is a
descriptive comparison of separately fitted models, not a causal fraction of
the BCE head's signal. Protein-only predictions are constant within an assay.

On the same 7,997 eligible validation assays, NDCG@1/@3/@5 rises from
0.59835/0.66331/0.71098 at epoch 1 to 0.70900/0.74819/0.78223 at epoch 15.
All three increase from 14→15 while macro Spearman decreases by only 0.00036;
this point contrast is not evidence of a reproducible objective conflict.
Epoch-15 BCE-logit and cosine-only macro Spearman are 0.27035 and 0.17598.
Constant predictions remain undefined rather than being jittered or assigned
artificial correlations. Validation-only temperature fitting gives T=1.29538
and reduces log-loss from 0.42772 to 0.41749. This is an in-sample calibration
fit; no test transfer has been evaluated for A.

At epoch 15, complete shared-scaffold, mixed and unseen-scaffold validation
assays score 0.37470/0.28168/0.22976 (5,405/2,402/190 eligible assays).
All three point estimates are below epoch 14; the all-unseen score is also
below its epoch-11 maximum of 0.23762. Restricting to unseen-scaffold rows
within assays instead gives 0.19670 on 1,608 eligible assays; the cohorts differ.
Fixed composition weighting changes A common-support validation macro from
0.34301 to 0.33462. The weights use the B validation label mix. The completed
matched validation comparison above also reports B predictions; selected
neural test and repeated comparisons remain pending.

Across epochs 1→15, ranking–BCE agreement changes 0.63066→0.63610 on the same
7,995 mutually defined eligible assays. Its whole-assay change interval
[−0.00474, 0.01599] includes zero. Ranking–cosine agreement rises
0.12360→0.29585 on 7,997 assays, with interval [0.15822, 0.18617].
The trajectories fluctuate and do not show uniform scorer divergence.

The completed run selects **epoch 14 by validation macro Spearman (0.34367521)**.
Minimum aggregate validation loss instead occurs at epoch 15 (2.36395550),
whose macro Spearman is 0.34331738. All 15 epoch weights are protected and the
retention hold has been released. Training finished at Unix1793574255.346448.
The native A test remains unscored pending the registered selection protocol.
Training-probe statistics are descriptive; intervals condition on one fit and
allocation. Multi-seed and selected-test method comparisons remain pending.

Evidence: `controlled/runs/full__A__seed42__lr1p0/controlled_training_complete.json`,
`controlled/protected_trajectories/full__A__seed42__lr1p0/complete.json`, and
`tier1/native_trajectories/full__A__seed42__lr1p0/6349989d65c2f7ef/`
with immutable `trajectory_through_epoch15.json` and
`verification_through_epoch15.json`. Independent calculations reproduced all
831,555 per-assay/scorer correlations across 30 snapshots, including undefined
cases, to 1e-14; 894 sampled pair-count checks and all 15 composition-weighted
point estimates passed. Earlier checks are inherited only after exact source,
prefix and table verification. This verification does not independently
reconstruct bootstrap intervals, calibration fits or all NDCG values.

The controlled B run first received a separately verified **ten-epoch prefix**.
Its probe uses the same fixed 2,048-eligible-assay rule as A, unlike the original
B baseline's uniformly sampled assay probe. The sampled assays still differ.

| Epoch | A validation macro rho | B validation macro rho | A probe macro rho | B probe macro rho |
|---|---:|---:|---:|---:|
| 1 | 0.16488 | 0.10992 | 0.16214 | 0.17815 |
| 2 | 0.21244 | 0.14458 | 0.22805 | 0.24104 |
| 3 | 0.24317 | 0.16912 | 0.27084 | 0.29117 |
| 4 | 0.26145 | 0.16577 | 0.30188 | 0.33091 |
| 5 | 0.27953 | 0.17172 | 0.33753 | 0.37329 |
| 6 | 0.28843 | 0.18458 | 0.36608 | 0.39862 |
| 7 | 0.30188 | 0.17889 | 0.38964 | 0.42904 |
| 8 | 0.31324 | 0.18049 | 0.40509 | 0.44715 |
| 9 | 0.31908 | 0.17596 | 0.42791 | 0.46300 |
| 10 | 0.32604 | 0.17776 | 0.45533 | 0.48243 |

At epoch 10, probe-minus-validation rho is 0.12929 on A and 0.30467 on B,
up from 0.05800/0.20157 at epoch5. B's best validation score within the observed
prefix is at epoch6 (0.18458). From6 to10, its probe rises0.39862→0.48243 while
validation falls0.18458→0.17776. These are descriptive probe gaps,
not full-training generalization gaps or evidence of a final plateau. A/B have
different assay populations and 802/877 optimizer steps per epoch. B's probe
contains 43,115 rows from only 25 scaffold-assay components, so its 2,048 assays
are not independent sampling units. B validation has 136,957 rows and 3,969
eligible assays. All 339,840 assay/scorer correlations across twenty snapshots and
600 sampled pair-count checks passed independent verification. This does not
independently verify the bootstrap intervals, calibration fits or all NDCGs.
This prefix was subsequently extended to15 below; repeated controlled fits
remain pending. Prefix evidence:
`tier1/native_trajectories/full__B__seed42__lr1p0/6349989d65c2f7ef/`
with `trajectory_through_epoch10.json` and `verification_through_epoch10.json`.

The controlled B training phase has now completed all 15 epochs and 13,155
optimizer steps. All four rank-completion receipts match its frozen config.
The saved native validation history selects **epoch6, macro Spearman0.18457785**;
epoch15 gives **0.17975193** and the minimum aggregate validation loss,
**6.68894768**. All3,969 eligible validation assays have defined correlations.
All30 allocation prediction exports and managed retention subsequently finished
at Unix1793600398.06, with verified training completion and no selected test
score. Completion-receipt SHA256:
`1372d4e4dc731f697b3deeba5aec0b89fad488cb6a435d6ef899a72ec66c0b8b`.

The initial reference also selects epoch6 (0.18460015), with its minimum loss
at epoch15 (6.68789959; macro0.17561471). The two seed42 runs therefore agree
on the selection-rule distinction. Their different source revisions and
diagnostic configurations preclude treating this as an exact replay or an
independent seed replicate; these observations do not estimate seed variance.
Evidence: both runs' `launch_provenance.json` and
`checkpoint-13155/trainer_state.json`; the controlled checkpoint-state SHA256 is
`f0c91b09d7dfb5877090e7131dd1370bf456e1c54c1cb66e6f163f3d94b071fa`.
Repeated controlled comparisons and all completion counts remain unchanged.

The one-shot CPU handoff (`native_horizon_queue.py`, commit `fec3286`) completed
at Unix1793600683.91 after verifying the original driver's completion. All30
validation/probe snapshots are independently checked: **509,760** per-assay/scorer
correlations and **900** sampled pair-count checks. The identical ten-epoch
prefix was reused only after source/output hashes were verified. This verifier
does not independently reconstruct bootstrap intervals, calibration fits or
NDCG values, and it read no test predictions or model/token payloads.
The frozen handoff plan SHA256 is
`3df846ee379582df8b0c996ed38b3eefaa3c09d75b22ec711d4c5de5b08931e1`.
The final trajectory and verification SHA256 values are respectively
`b5806c564a4b79adeb0f7c73e82a95db928d6b4b45781b917d1ded8cad59e02d` and
`2b178b8d2fc28c84e17b7798d0323dc15b04b7291d955d7b1560480e16f17139`.
Evidence: the native B directory above with `trajectory_through_epoch15.json`
and `verification_through_epoch15.json`, plus
`tier1/native_trajectory_handoff/full__B__seed42__lr1p0/{complete,exit}.json`.

From epoch6 to15, B's fixed-probe macro rises **0.39862→0.51969**, while validation
changes **0.18458→0.17975**. The final probe-minus-validation difference is
**0.33994**. This probe contains2,048 eligible assays but only25 components;
it is not a full-training estimate or independent seed evidence. Native BCE,
ranking and contrastive validation losses attain minima at epochs2,3,15,
respectively. Their first adjacent increases are2→3,3→4,6→7; an adjacent increase
alone does not establish sustained overfitting. Learned scale rises13.5988→22.5452.
At epoch15, validation-only temperature1.4732 changes BCE logloss0.49210→0.46908
without changing score order; this is an in-sample calibration result.

On the same3,969 eligible validation assays, ranking NDCG@1/@3/@5 peaks at
epochs6/13/8. All three finish below their epoch6 values, so this run does not
show sustained late top-ranking improvement. From1→15, within-assay
ranking/BCE agreement falls0.67647→0.59235 on3,968 mutually defined assays,
whereas ranking/cosine agreement rises0.07402→0.23018 on3,969. Final BCE-logit
and cosine label macros are0.13944/0.07754, below the ranking head's0.17975.
These prediction correlations do not establish gradient conflict. Separate
directional/fixed-scale results remain labelled with their original source runs.

After verified B completion, the existing tuning queue launched
`full__A__seed42__lr0p3` (W&B `lekrhvg9`) on physicalGPUs0,2,5,6. At
Unix1793600720.55 it was at epoch0.2122; no LR has been selected and no
confirmation seed has started. Scientific completion counts remain unchanged.

The automatic handoff stopped when its contrastive audit rejected encoder
widths filled in during model loading (480/768 versus unspecified launch
values). No checkpoint was changed and no next training was started. The
repair validates those widths against the actual encoders while retaining
strict checks on every other configuration field. It passed 22 CPU tests and
an Atum preflight checking all 15 saved configurations plus one actual loaded
checkpoint, with unchanged weights and launch configuration. The failed audit
and handoff records are preserved. The corrected audit was launched as
PID1216919 with four workers, each restricted to one permitted GPU. GPU
inference was verified on the four permitted GPUs. All 30 validation/training-probe
exports completed, with all worker handles absent by Unix1793576737.68. Maximum
native/export contrastive-loss difference is 3.24e-6 and macro-rho difference
1.11e-16. These exports use no test data and alter no checkpoint selection.
The next full B15 run (`full__B__seed42__lr1p0`, W&B `ormrd95z`) then started
on the same four GPUs, with batch12×4×4 and validation-macro checkpoint selection.
All 43 qualified training source files remain unchanged. Evidence:
`tier1/reference_contrastive/recovery_{preflight,launch}_2cc800d.json` and
`tier1/reference_contrastive/full__A__seed42__lr1p0/c81891c91f5ed91f/`.

The remaining four registered full-arm tuning fits now have a bounded queue:
A/B at learning-rate multiplier 0.3, followed by A/B at multiplier 3. The CPU
coordinator was verified live as PID1260221 at Unix1793578258.44, waiting on the
exact original B driver, launcher and four ranks. It requires B's verified
completion, including allocation exports and retention, before continuing.
The existing training driver and all 43 qualified source files are unchanged.
Both filesystem capacity and account quotas are checked before each successor;
the preparation check found approximately 746 GiB archive quota remaining.
The queue stops before confirmation or test selection, and never retries a
partial fit. All 23 queue/driver CPU tests passed. An immediate outer-launch
process read encountered a transient empty command line; the already-running
coordinator and its receipts were verified and recorded without restarting it.
This is execution preparation, not another completed scientific experiment.
Evidence: `controlled/full_tuning_queue/{plan,preflight,launch_process,launch_verification}.json`.

A bounded follow-up coordinator (`confirmation_queue.py`, commit `b8169b7`)
waits for that exact tuning coordinator and all six verified full-arm fits.
It then applies the existing B-validation-only learning-rate selection and runs
exactly A/B seeds 43 and 44 at that shared rate. It stops after those four fits,
before test evaluation or another model arm. All 16 CPU orchestration tests
passed, using the real selection/configuration code and simulated training.
Atum verified PID1338025 waiting on PID1260221 at Unix1793582959.97, with GPUs
hidden from the coordinator and the original four training GPU holders unchanged.
The plan pins 85 inputs, including the 43 qualified training source files;
global-lock, physical-space and account-quota checks precede each fit. Failed
or partial fits stop for inspection. No LR has yet been selected and no
confirmation fit has started. This is scheduling preparation, not a new result.
Evidence: `controlled/full_confirmation_queue/{plan,preflight,launch_process,launch_verification}.json`.

The full-arm evaluation/reporting handoff is now waiting behind that exact
confirmation coordinator. `full_evaluation_queue.py` (commit `d8beca3`) requires
its verified terminal completion and frozen B-validation LR decision before
evaluating exactly the six tuning plus four confirmation fits. It then runs
per-fit CPU diagnostics and the complete primary summary. All 21 orchestration
tests passed using real selection/configuration generation and simulated
evaluation, including live-parent waiting, resource gates, partial-output
refusal and changed-evidence rejection. Native inference and reporting have
their separate qualifications above/below; these orchestration tests are not
scientific GPU results. Atum verified 105 pinned inputs and launched coordinator
PID1438015/start153986235 with no visible GPUs. Its observed phase is
`waiting_for_confirmation_coordinator`; all eight prior handles remained live.
It adds no training settings, selects no test result, releases no weights and
does not execute additional allocations. No scientific primary evaluation or
summary has started, and experiment/question counts are unchanged. Evidence:
`controlled/full_evaluation_queue/{plan,preflight,launch_process,launch_verification}.json`.

The next E33 stage is queued in `regularization_queue.py` (commit `5face02`).
It waits for that exact full-evaluation process and its verified ten-fit report,
then executes the already registered weight-decay0.001 and0.1 settings. Each has
six tuning and four confirmation fits, followed by selected native evaluation
and a primary report; the weight-decay0.01 reference is reused. The 20 fits are
already in the frozen primary plan, with no additional setting or tuning budget.
All 24 CPU orchestration tests passed with real configuration/validation-choice
reconstruction and simulated training/evaluation. Atum's actual preflight passed
for both settings and storage quotas. Coordinator PID1486155/start154274700 was
verified live in `waiting_for_full_reference_report`, exposing no GPUs and
pinning133 inputs. The current training holders remain on GPUs0,2,5,6. Existing
or failed outputs stop for inspection, and the queue stops after these two arms.
No regularization fit has started and no experiment/question is newly complete.
Evidence: `controlled/regularization_queue/{plan,preflight,launch_process,launch_verification}.json`
and `regularization_queue_runtime/5face02/deployment.json`.

E33's regularization comparison is implemented in `regularization_analysis.py`
(commit `c8d05e3`). It freezes one decay from native seed42 B validation across
the three existing settings, after each setting's existing three-LR selection.
The same selected decay/LR is used across checkpoint rules; all settings remain
reported. This isolates checkpoint choices on common training procedures rather
than independently tuning every stopping algorithm. The report requires all
30 verified primary analyses and preserves within-fit checkpoint changes, paired
alternative-decay minus reference changes, score coverage and separate seed42/
confirmation-seed summaries. No test score chooses the decay or LR, and no fit
is added. All26 focused tests passed, including actual validation-receipt
reconstruction, test-score independence, source/scope rejection and cached-report
integrity. Atum's read-only preflight mapped18 planned tuning jobs and found no
completed primary summaries; no decay selection or scientific comparison was
created. E33 and all question/experiment completion counts remain unchanged.
Evidence: `regularization_analysis_runtime/c8d05e3/{deployment,preflight}.json`.

Q34/E39's validation-only weight comparison is implemented in
`weight_selection.py` (commit `c13c5e9`). It reuses the five registered axis-sweep
settings and their three LRs: 15 candidates per split, 30 existing tuning fits,
including six shared full-reference fits. A-validation and B-validation choices
are reported separately, with fixed tie rules, defined coverage and explicit
limitations when an A-selected LR lacks seed43/44 confirmation coverage. No new
fit or confirmation request is created and no test score is read. All 20 focused
tests passed, including actual registration/validation-receipt reconstruction,
scope/qualification checks and rejection of changed or incomplete fits. Atum's
read-only preflight verified one completed training receipt and 29 unfinished
fits; it created no selection. This prepares the comparison rather than answering
Q34 or completing E39. Evidence:
`weight_selection_runtime/c13c5e9/{deployment,preflight}.json`.

The static-weight selected-results report is now implemented in
`objective_controls_analysis.py` (commit `71778a0`). It requires all five
canonical primary summaries (30 tuning and 20 confirmation fits, including the
ten shared full-reference fits) and the independently verified, validation-only
A/B weight-selection receipt. Four alternative-minus-reference comparisons use
each setting's frozen B-selected LR, all five checkpoint rules, paired whole-unit
intervals and separate seed42/confirmation-seed summaries. The A-selected choice
is shown diagnostically; if its LR differs, the existing B-selected confirmation
results do not evaluate that A-selected procedure. No cross-regime independence
or historical reference-weight tuning is inferred. All settings remain visible,
and test scores cannot choose the family winner.

All 49 objective-report tests and the existing 20 weight-selector tests passed.
Atum preflight verified the five settings and four contrasts, with one completed
weight-grid fit, 29 unfinished fits and zero primary summaries. Across weights,
contrastive and ranking controls, this reporter covers 90 distinct tuning fits
and 150 total existing primary fits after removing shared-reference duplication;
it adds no fits. Deployment preserved eleven live process identities and 138
pinned inputs. No scientific weight comparison was produced; Q34/E39 and all
completion counts remain unchanged. Evidence:
`objective_controls_analysis_runtime/71778a0/{deployment,preflight}.json` on Atum;
preflight SHA256 `87a6ee1cbf100e4da256d52bd61ed77c0e6e335131c3df09f4f86e774013c6a7`.

Q30/E38's seven-loss report is implemented in `loss_subset_analysis.py`
(commit `074f5aa`). It requires all seven verified primary-arm summaries, covering
42 tuning and 28 confirmation fits including the ten shared full-reference fits.
Declared ranking-head/BCE-logit/cosine outputs stay explicit. R/RB/RC/RBC also
receive four auxiliary-effect contrasts and the conditional interaction
RBC−RB−RC+R on their common eligible, mutually defined assay cohort. The 2,000
paired whole-unit bootstrap draws retain zero-eligible units; seed spread,
confirmation-only spread and different A/B populations stay distinct. These
comparisons include each arm's validation-selected LR/checkpoint and do not
establish gradient conflict. All 27 focused tests passed, including independent
unit-resampling arithmetic, cohort/scope rejection and cached-report integrity.
Atum's CPU-only preflight found no primary-arm summaries and created no fit,
selection or scientific comparison. Deployment preserved all nine live process
identities and 105 pinned inputs. Q30 and E38 remain pending; completion counts
are unchanged. Evidence:
`loss_subset_analysis_runtime/074f5aa/{deployment,preflight}.json`.

E34–E37/Q09 reporting is implemented in `encoder_analysis.py` (commit `56e8cb8`).
It requires the six existing full/frozen/last-two-block/LoRA-4/16/64 primary-arm
summaries: 36 tuning and 24 confirmation fits, including the ten shared full
fits. Each arm keeps its B-validation-selected LR; all five checkpoint rules,
three diagnostic scorers and same-LR seed42 results remain visible. Encoder-minus-
full contrasts use paired whole-unit intervals on mutually defined eligible
assays, with tuning-exposed and confirmation-only seed spreads kept separate.
They compare selected training procedures, not a fixed-LR causal encoder effect.

The reporter also preserves all 60 native 15-epoch validation/fixed-probe
trajectories, component-loss minima, first adjacent increases and ranking peaks.
It requires matching dataset fingerprints and row counts within each regime and
seed, and validates native history against hash-bound prediction exports. These
are reported native metrics, not a fresh independent rescore; a first increase
does not establish sustained overfitting or encoder drift. All 27 focused tests
passed. Atum's completed A trajectory matched all 30 existing independently
rescored snapshots, reproducing macro peak14, final validation0.34331738 and
probe0.49207621. Its aggregate/ranking/contrastive losses had no adjacent increase;
BCE's first increase was5→6 and minimum was13. This is reader qualification on an
existing fit, not a frozen-encoder result. Preflight found zero family summaries;
no new fit, family selection or scientific encoder comparison was created.
Q09/E34–E37 remain pending. Deployment preserved all ten live process identities
and 133 pinned inputs. Evidence:
`encoder_analysis_runtime/56e8cb8/{deployment,preflight}.json` on Atum;
preflight SHA256 `8b5f5da5eca4683c783aba48ab9f0f6201a17b798f677e04227c9903ec82c0f9`.

E42–E47 reporting is implemented in `objective_controls_analysis.py` (commit
`a0ed17b`). The five contrastive settings require 50 existing primary fits; the
seven ranking settings require 70. Their ten full-reference fits are shared,
so the two families cover 110 unique planned fits, not 120 or any new fits.
Four contrastive and six ranking contrasts preserve their intended references:
uniform/top weights both use pairwise objectives; noise-soft/hard targets both
use zero margin; margin removal and assay balancing receive separate contrasts.
The exact model-only changes, train-derived noise temperature, common training
budget and canonical primary evaluation provenance are checked before reporting.

All settings, three scorer outputs, five checkpoint rules and same-LR seed42
results remain visible. Paired contrasts use mutually defined eligible assays
and whole-unit intervals; seed42 exposure, confirmation-only seed spread and
descriptive A-minus-B changes remain separate. Each setting retains its own
B-validation-selected LR, so these are comparisons of selected procedures.
They do not by themselves establish false-negative causality, calibration or
gradient mechanisms. All 30 focused tests passed, including correct contrast
signs/references, changed or incomplete evidence and child-artifact mutation
during reporting. Atum's CPU-only preflight verified 66 distinct registered
tuning configurations and found zero primary summaries for either family.
No training, family selection or scientific comparison was created; E42–E47
remain pending. Deployment preserved eleven live handles and 138 pinned inputs.
Evidence: `objective_controls_analysis_runtime/a0ed17b/{deployment,preflight}.json`
on Atum; preflight SHA256
`14af559a927351225ce28019906b35df1929ec42b17677b15df0aec12804926d`.

The allocation-specific LR selector and supplemental-fit runner are implemented
in `realization_selection.py` (commit `47bcd17`). Each of the three dependent B
allocations uses its own validation trajectory across the same three LRs. The
seed-43/44 requests cover the union of winning LRs and reuse exactly matching
primary B fits; supplemental fits do not create matched A results. The runner
waits for all four primary confirmations and executes at most one additional B
fit under the shared lock and existing GPU/quota checks. All 19 focused CPU
tests passed with synthetic scalar exports and simulated training; the export
registration accepts supplemental fits while the primary evaluator rejects them.
Atum's read-only preflight at commit `47bcd17` verified the three frozen
allocations and identical training support, then reported five incomplete tuning
fits and no primary LR selection. It created no selection, fit or test score.
All live training/queue identities and their pinned sources survived deployment.
Full-loss evaluation, averaging and selected test comparisons for these
allocations remain pending. Experiment/question completion counts are unchanged.
Evidence: `realization_selection_runtime/47bcd17/{deployment,preflight}.json`.

The allocation-native validation-loss and averaging stage is implemented in
`realization_losses.py` (commit `2a7bfac`). It evaluates all 15 checkpoints only
at each allocation's frozen selected LR, retains the existing uniform-inference
Spearman choices, and derives loss/patience rules and post-minimum averages from
that allocation's native validation losses. Identical averaging windows share
one verified artifact; test roles and checkpoint release remain separate.
All 24 focused CPU tests passed, including real tiny-model native inference for
15 epochs under both mask policies, unchanged weights, verified averages,
registration/coverage checks and rejection of CPU artifacts as GPU results.
Atum then verified the complete production token caches and row/label/unit
alignment of all three validation views: each has 136,957 rows; original/seed43/
seed44 contain 9,280/9,448/9,475 assays and 4,032/4,031/4,032 indivisible units.
This qualification loaded no model and computed no loss or test score. The
scientific stage waits for completed fits and frozen allocation LR selections;
no experiment or question is newly closed. Evidence:
`realization_losses_runtime/2a7bfac/{deployment,view_qualification}.json`.

The selected-allocation evaluator is implemented in
`realization_postselection.py` (commit `2b4ce42`). It requires completed native
validation trajectories and verified averages, reuses epoch validation outputs,
and evaluates only the frozen selected weights on each allocation's test role;
averages also receive validation evaluation. All 19 focused CPU tests passed,
including actual tiny-model inference under both mask policies, unchanged
upstream artifacts, exact role/label/fingerprint alignment, no test-driven
selection and rejection of incomplete/corrupted evidence. Atum verified all six
production role views over the same 273,913 held-out rows, with model inference
and test metrics disabled. Scientific selected test evaluation still awaits
training, allocation LR selection and the native-loss/averaging stages. No
completion count changes. Evidence:
`realization_postselection_runtime/2b4ce42/{deployment,input_qualification}.json`.

The allocation-specific CPU diagnostics and seed summaries are implemented in
`realization_analysis.py` (commit `27901fe`). All 19 focused tests passed,
including actual tiny-model predictions, independently calculated Spearman,
whole-component coverage, cached-output integrity and rejection of incomplete
or mismatched seed/allocation grids. Reporting keeps seed42 tuning exposure,
seed43/44 confirmation spread, dependent allocation sensitivity and conditional
whole-unit intervals separate. All five checkpoint rules are required; scores
and coverage remain tied to verified child analyses and current source receipts.
Atum qualified all six descriptor views over 273,913 unique held-out rows.
Original/seed43/seed44 validation eligibility is 3,969/3,963/4,059 assays; test
eligibility is 3,972/3,978/3,882. Recomputed coarse composition cells support
99.80%/99.82%/99.92% of eligible test mass in A; this is descriptor support,
not model performance or evidence that all composition differences disappear.
No model was loaded and no prediction scores were read. Scientific per-fit
diagnostics and the complete seed/allocation summary still await the selected
evaluations. Deployment preserved all eight live training/queue identities and
their frozen inputs. Completion counts are unchanged. Evidence:
`realization_analysis_runtime/27901fe/{deployment,reference_qualification}.json`.

The downstream primary-grid evaluator had the same unresolved-versus-resolved
encoder-width mismatch. It was reproduced with a saved-model fixture and
repaired in commit `ed4393f`: only unspecified launch widths may resolve to
positive integers; retained configurations must match the run, and loaded
widths must match the actual encoders. All other configuration fields remain
checked. All 26 focused CPU tests passed, including native inference without
training for both contrastive-mask policies. Atum verified 19 existing A/B
configuration files and a tiny CPU-only save/load with unchanged parameters.
No scientific test predictions were computed. Deployment preserved all 43
qualified training sources, the six original B process identities and the live
queue. This fixes an evaluation blocker, not a scientific result. Evidence:
`postselection_runtime/ed4393f/{deployment,qualification}.json`.

The CPU analysis of completed primary-grid evaluations is implemented in
`selected_analysis.py` (commit `38f1196`). It requires the frozen validation-only
selection and every checkpoint/partition receipt before reading predictions.
It computes the MD's rank, pair, noise-band, size, scaffold, composition,
calibration, BBSE and pathway-agreement diagnostics, with whole-unit intervals.
Matched A/B comparisons use the same arm, seed, learning rate and checkpoint
rule; test scores cannot select a configuration. All 12 focused tests passed,
including actual tiny native exports and independent rank/pair calculations.
Atum verified all 662,218 rows across A/B validation/test against the design and
cache metadata; both validation fingerprints matched native training exports.
No selected-model scientific predictions or scores were produced. Deployment
preserved active training and its queue. Evidence:
`selected_analysis_runtime/38f1196/{deployment,input_qualification}.json`.

The primary-arm summary is implemented in `controlled_summary.py` (commit
`b7568b4`). It requires six tuning and four selected confirmation analyses,
re-verifies their validation-only decisions and native evaluation evidence, and
reports all three scorers across all five selected-LR checkpoint rules. Seed
spread, seed42 LR sensitivity, same-seed A/B gaps and within-run checkpoint
contrasts remain separate, with explicit score coverage. All 26 focused CPU
tests passed, including artifact integrity and rejection of actual CPU-native
qualification predictions as scientific GPU evidence. Atum cross-checked the
paired calculation on the existing hash-verified A epochs14/15: 7,997 mutually
defined eligible assays, 16,431 resampled whole-assay units, mean change
-0.00035783 and interval [-0.00233925, 0.00164323]. This reproduces the existing
reference result; it is not a new fit, seed or method comparison. Read-only
preflight found no frozen full-arm LR decision and five incomplete tuning fits;
no scientific primary summary was created. All eight active training/queue
handles and their frozen inputs survived deployment. Completion counts remain
unchanged. Evidence:
`controlled_summary_runtime/b7568b4/{deployment,preflight,reference_qualification}.json`.

Q11's scalar localization audit covers all 15 initial B validation epochs and
all 15 verified A validation epochs. B's BCE rises by 0.03768 from its overall minimum at
epoch 2 to epoch 15, with a positive point change in every fixed size bin.
Ranking changes below instead compare the macro-Spearman peak at epoch 6 with
epoch 15, on the same eligible assays at both endpoints.

| Assay size | B validation rows | BCE change, 2→15 | Eligible assays, 6→15 | Macro rho change, 6→15 |
|---|---:|---:|---:|---:|
| 1 | 3,769 | +0.04321 | 0 | undefined |
| 2 | 1,988 | +0.00754 | 0 | undefined |
| 3 | 1,623 | +0.02937 | 329 | −0.08247 |
| 4–10 | 9,812 | +0.02029 | 1,330 | −0.00158 |
| 11–25 | 21,365 | +0.04284 | 1,222 | +0.00120 |
| >25 | 98,400 | +0.03883 | 1,088 | −0.00725 |

Three-item assays account arithmetically for 76.1% of the observed net macro
decline, despite representing 8.3% of eligible assays. The overall decline is
−0.00899 with paired component interval [−0.02078, 0.00302]; the three-item
change interval is [−0.15882, −0.00549]. These descriptive intervals condition
on one fit and the observed endpoint choices, exclude selection/seed uncertainty,
and are not adjusted for multiple strata. They do not establish a reproducible
localized failure. The BCE increase is broader: >25-item assays contribute
74.0% of its net increase and contain 71.8% of rows.

B has only unseen-scaffold groups under the predefined binary relation, so
that split cannot localize the change across different similarity strata.
A's complete trajectory separates the overall BCE improvement from its cold-assay
point trend:

| A scaffold class | Validation rows | Eligible assays | BCE, 1→15 | Macro rho, 1→15 |
|---|---:|---:|---:|---:|
| All shared | 104,730 | 5,405 | 0.52598→0.39065 | 0.16017→0.37470 |
| Mixed | 84,876 | 2,402 | 0.51247→0.45957 | 0.17779→0.28168 |
| All unseen | 6,012 | 190 | 0.54630→0.62373 | 0.13584→0.22976 |

The all-unseen BCE change is +0.07743, but its paired whole-assay interval
[−0.03240, 0.19907] includes zero. The ranking change is +0.09392, interval
[0.03030, 0.15353]. BCE uses all 598 cold assays; headline ranking uses the
190 eligible assays. The cold BCE minimum is epoch 2 and macro-rho maximum
is epoch 11. All six size bins improve in BCE from 1→15, and all four eligible
size bins improve in macro rho. This localizes a cold-assay BCE point trend
alongside improved ranking; it does not establish calibration as the cause.

Across all A validation assays, the 14→15 macro change is −0.00036, with
conditional whole-assay interval [−0.00234, 0.00164]. Thus the small terminal
point decline does not establish a reproducible drop. The cold-only 14→15
macro and BCE change intervals also include zero. From 13→15 the cold BCE
point increase is 0.01004 with interval [0.00179, 0.01978]; these endpoints
were chosen after observing the overall BCE minimum, and the interval omits
endpoint-selection, seed and multiple-stratum uncertainty.

Q11 now has a descriptive answer from both complete reference trajectories;
controlled repeated and causal comparisons remain pending. Source:
`tier1/stratified_trajectories/bb7096aaf6bda626/{complete.json,verification.json}`.
All 30 snapshots reproduce native aggregate BCE within 1e-5 and headline rho
within 1e-14. Independent rank calculations checked 385,665 group correlations,
including undefined cases. Independent Bernoulli log probabilities, fixed
memberships, counts, means and rank-sum AUROC passed for every stratum.
The verifier checks all 30 snapshots directly; it does not inherit earlier
numerical results. All source and table hashes passed. Bootstrap intervals
were not independently reconstructed; no test predictions were read.

The native listwise-loss localization covers all 15 initial B epochs and
all 15 matched A epochs. It preserves the original prediction order, three deterministic
partitions, complete-assay eligibility, top-position/margin weights and
zero-loss lists. The loss averages bounded lists, so a large assay contributes
more than one list; strata below still use the complete assay's size.

| B assay size | Lists per partition | Ranking-loss change, 5→15 | Paired component interval |
|---|---:|---:|---:|
| 3 | 329 | +0.03489 | [−0.00721, 0.07964] |
| 4–10 | 1,330 | +0.07428 | [0.04723, 0.10139] |
| 11–25 | 1,806 | +0.06589 | [0.04007, 0.09489] |
| >25 | 6,476 | +0.17152 | [0.11978, 0.22901] |

Overall ranking loss rises 2.79392→2.92871 from its minimum at epoch 5, a
change of 0.13480 with interval [0.09921, 0.17574]. All eligible size bins have
positive point changes. Using the same epochs 6→15 as the macro-Spearman
comparison, >25-item assays account arithmetically for 80.2% of the net
ranking-loss increase, and three-item assays for 0.9%. The former carry 65.1%
of native list weight but 27.4% of eligible assays; the latter carry 3.3% of
list weight but 8.3% of assays. Thus loss and macro-rank deterioration localize
differently. This descriptive accounting does not establish that changing the
normalization improves generalization; that needs the controlled training arm.
All intervals retain the single-fit, observed-endpoint and multiplicity limits above.

A's ranking loss falls 2.66708→2.41737 over epochs 1→15, reaching its minimum
at 15. In fully unseen-scaffold assays it changes 2.57456→2.54119; the paired
change interval [−0.12493, 0.06786] includes zero. Thus the native loss and
assay-macro ranking metric describe different changes even within cold A.
Source: `tier1/ranking_localization/d05274e4f8707d62/{complete.json,verification.json}`.
All 30 reconstructed global losses differ from native values by at most
3.85e-7. Direct production-loss calls on 390 sampled lists agree within
4.77e-7; all 1,146,645 list descriptors, fixed stratum reductions and source/output
hashes were verified. Individual list losses are sampled, not exhaustively
recomputed, and intervals were not independently reconstructed.
No test predictions, model weights or GPUs were used in this verification.

The contrastive localization audit attributes loss to each query's original
assay while keeping the complete candidate pool and all masks unchanged.
It covers B epochs 1–15 at both the learned scale and fixed scale 13, including
all 9,280 assays, with 15,407 bounded lists per partition. Zero-active lists
retain denominator mass; each active query retains its native inverse-square-root
list-size weight. Strata use full assay size, not bounded-list size. These are
contributions to the original task, not separate retrieval tasks restricted to
each stratum.

| B contrastive direction | Epoch 1 | Epoch 5 | Epoch 6 | Epoch 15 | Minimum epoch |
|---|---:|---:|---:|---:|---:|
| Protein→molecule, learned scale | 5.73059 | 3.99055 | 3.96577 | 3.79393 | 15 |
| Molecule→protein, learned scale | 2.81915 | 2.03734 | 2.04029 | 2.04105 | 5 |
| Protein→molecule, fixed scale 13 | 5.74136 | 4.05543 | 3.99684 | 3.84456 | 15 |
| Molecule→protein, fixed scale 13 | 2.78834 | 1.85659 | 1.82522 | 1.70923 | 15 |

The learned-scale molecule→protein direction first increases at epochs 5→6
and ends only 0.00370 above its epoch-5 minimum. Protein→molecule first
increases at 6→7 but reaches its minimum at 15. Both fixed-scale directions
first increase at 6→7 and reach their minima at 15. Local increases and
minimum epochs describe this trajectory; they are not statistical change points.

On the common macro-peak comparison, epochs 6→15, learned-scale
protein→molecule loss falls by 0.17184 and molecule→protein changes by
+0.00076. The latter rises in the 1-, 3-, 4–10- and 11–25-item bins, while
falling in the 2- and >25-item bins. Only three-item assays show an increase
in the sum of both directions (+0.01709). At fixed scale 13 both directions
fall in every size bin; the three-item total changes by −0.09395. This
isolates a numerical scale effect on the saved representations, not a benefit
of fixed-temperature training. The contrastive three-item cohort contains
541 assays, compared with 329 for eligible macro Spearman; their changes
must not be treated as measurements of the same cohort.

These are descriptive single-fit comparisons with fixed candidate contexts.
No confidence intervals or redrawn-pool uncertainty are claimed: resampling
query units alone would leave the cross-unit candidate context conditional.
Source for the B attribution:
`tier1/contrastive_localization/B/e129980b1592cd85/{complete.json,verification.json}`.

The completed protected A audit now supplies both directions for every validation
and original training-probe checkpoint. Every epoch uses the same candidate-pool
hash within its partition. Both directions decrease at every adjacent epoch on
validation and on the fixed training probe, at both the learned scale and scale13.

| A validation contrastive quantity | Epoch 1 | Epoch 14 | Epoch 15 |
|---|---:|---:|---:|
| Learned protein→molecule | 3.37895 | 1.11556 | 1.11034 |
| Learned molecule→protein | 1.48876 | 0.54360 | 0.54235 |
| Learned total | 4.86771 | 1.65916 | 1.65269 |
| Fixed-scale13 total | 4.89917 | 2.01028 | 2.00395 |

At A epoch15, learned-scale loss is 0.35127 below fixed-scale13 rescoring;
at original B epoch15 it is 0.28118 above it. Thus the numerical scale effect
has opposite signs on these fixed representations. This is not evidence that
changing the training temperature improves generalization. A mean positive-row
cosine rises 0.41887→0.48340 on validation and 0.41171→0.51366 on its training
probe. These are descriptive means on different fixed cohorts, not a calibrated
memorization test. Source:
`tier1/reference_contrastive/full__A__seed42__lr1p0/c81891c91f5ed91f/complete.json`,
SHA256 `94c681a6638c900d327c3973d93ed7b55c6a3ae5306866e56121210422c3a039`.

A's completed stratum attribution shows that the scale effect differs within
that split as well. At epoch15, learned versus fixed-scale13 total losses are:

| A scaffold class | Learned scale | Fixed scale13 |
|---|---:|---:|
| All shared | 1.18614 | 1.50644 |
| Mixed | 2.52134 | 3.02026 |
| All unseen | 3.97861 | 3.77506 |

Within the cold group, learned molecule→protein loss has its minimum at epoch9
(1.45856) and ends at 1.49886; fixed-scale13 molecule→protein loss reaches its
minimum at epoch15 (1.24496). Cold total loss has its minimum at epoch13 at both
scales, with small terminal increases. From the overall macro peak14→15,
both directions decrease in every size and scaffold stratum at both scales.
The three-item learned total changes by −0.01131, contrasting with original
B's +0.01709 over its own macro-peak6→15 interval; these intervals differ.
This describes the observed references and does not establish a repeatable
localized failure, a causal calibration mechanism or a training benefit.

The attribution keeps all candidates and masks fixed and covers 16,431 assays,
24,089 bounded lists per partition and three partitions at each checkpoint.
All30 table hashes and eleven analysis-source hashes passed. Its 1,980 sampled
production-operator checks agree within1.43e-6, and global directional values
reproduce the completed reference audit. These checks do not independently
reconstruct every list contribution; no cross-unit-context confidence interval
is claimed. The first attempt's filename failure is preserved; the successful
retry canonicalizes validated integral epochs without changing model code.
Sources: `tier1/contrastive_localization/A/0a4f60eb277f6f39/`
`{complete.json,completion_check.json}` and
`tier1/contrastive_localization_launch_a2e19a0_A.{json,log}`.

All 1,260 sampled production-batch checks agree within 3.82e-6. Independent
explicit candidate-set calculations check 210 batches / 2,520 query lists;
all 1,386,630 list-instance descriptors and source/output hashes were verified.
An initial absolute tolerance on unnormalized float32 list sums missed one
value by 1.03e-5; float64 reconstruction agrees with float64 production loss
within 3.56e-15, and the saved float32 discrepancy is at most 8.57e-7 in
native batch-loss units. The precision diagnosis is retained in verification.
No training, model inference, GPU use or test predictions were involved.

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

Q22's recorded observation process is now audited. The source curation retains
both sides of the activity threshold: exact, numeric pChEMBL measurements from
single-protein binding assays with confidence score ≥8 and no recorded variant,
with parent-molecule normalization, quality/structure checks and within-assay
median aggregation.
The later activity-balanced resplit copies all source observations and changes
their partition assignments. The 1,957,400 original curated observations are
**69.04% positive** at pChEMBL≥6, before the training pair-median label policy.

Within their 164,312 assays, the archive contains 4,161,420 activity records.
There are 2,034,645 numeric pChEMBL records (69.2176% positive); 2,032,748 of
these feed the curated observations (69.2218% positive before aggregation).
Another 2,126,775 records lack numeric pChEMBL, and 1,897 defined records fall
outside the retained IDs. The broader records include 1,345,088 `kon`/`k_off`
measurements without pChEMBL and 500,807 ordinary inequality-qualified records.
These are activity records with potentially multiple endpoints per pair, not
counts of distinct missing pairs or negative examples.

| Assay provenance | Curated observations | Positive fraction |
|---|---:|---:|
| Scientific Literature | 1,138,209 | 67.79% |
| BindingDB Patent Bioactivity Data | 637,588 | 82.86% |
| PubChem BioAssays | 116,824 | 4.61% |

Thus source composition is heterogeneous, and the near-70% fraction is already
present among the archived numeric measurements. The available tables do not
identify each experiment's sampling frame, per-pair inclusion probability or
tested-but-unpublished results. Preferential selection of promising pairs is
not established by these counts; unobserved pairs remain unlabeled. This is a
qualified answer to Q22, not an estimate of activity prevalence for generated
molecules or a causal explanation of the A/B performance gap.
Source: `tier1/observation_process/{complete.json,verification.json}` on Atum.
All retained activity IDs reconstruct the original medians/counts and match the
raw database. Independent SQL totals and output/source hashes were verified;
the 30.48 GB database's content hash and original release hash are recorded.

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

The E41 native evaluation path is now implemented in
`learning_curve_evaluation.py` (source `2bc0f21`). It requires all ten completed
primary full-reference fits and all six tuning fits for the relevant fraction,
reconstructs the B-validation LR choice, and checks complete 15-epoch histories,
four-rank receipts, training qualifications and retained model bytes. It
verifies that each completed fit used its exact registered subset and labels;
fixed validation/test partitions are reused. The selected LR gets all five
existing checkpoint rules, while other seed-42 LRs get best-macro only. The
100% endpoint is reused, and the shared training lock/device allowlist applies
to evaluation as well as fitting.

All 21 focused tests passed, including real saved tiny-model inference through
the native Trainer with no optimizer, training or weight changes. Tests also
reject a rehashed but incomplete nonselected candidate, changed whole units,
labels, model bytes, selection or device provenance. Atum recomputed the nested
selections and subset-local labels for all eight views,
verified the exact full endpoints and rechecked 28 unique view inputs. An
attempt to plan evaluation of an unfinished curve fit correctly refused it.
These CPU checks read prepared metadata and label/index arrays, not token
payloads or trained weights; they are not scientific inference or performance.

No curve fit, scientific evaluation or new GPU queue has started. All nine
existing live process identities and 138 pinned inputs were preserved during
deployment. E41/Q29 remain partial: fitting, native scoring and statistical
learning-curve reporting are still required. Evidence on Atum:
`learning_curve_evaluation_runtime/2bc0f21/{deployment,preflight,view_qualification}.json`.
Preflight SHA256 `bfeb49c95c52ad64391db077e3704faa55e977dae74ecae4dc9a054e4b7df324`;
view qualification SHA256 `71f364cfdd0edcd3f4e0deb6be1f85ac2f6ab11d3ed45538cb0822c4ced827c0`.

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

## Staged adaptation data design — fitting pending

E49–E52 now have verified assay memberships, not adaptation performance.
An identity-only hash divides original validation units into adaptation and
validation sets; original test groups remain unchanged. A has 8,215/8,216/16,431
assays in these three roles. B has 4,722/4,558/9,578 assays across
2,016/2,016/4,034 components. Across-unit adaptation uses nested whole-assay
prefixes at k=1/2/4/8/16/64, shared between the heads-only and heads-plus-interaction
settings. Token/label views and the fitting/epoch-selection protocol are now
prepared; selected-base binding and the production GPU runtime still need
qualification.
The selected A prefixes at k=1/2/4 and B prefixes at k=1/2 contain no
ranking-eligible assay. Those budgets provide no direct ranking supervision;
an absence of ranking improvement there cannot establish a head-capacity limit.

Within-unit probes reserve a fixed support half and query half of each unit's
assays. Query assays do not change with k. A's indivisible unit is one assay,
so disjoint within-unit support/query assays are impossible under its existing
definition. B's available cohort shrinks sharply with k:

| Support assays k | Rank-evaluable validation units | Rank-evaluable test units |
|---|---:|---:|
| 1 | 515 | 1,023 |
| 2 | 220 | 421 |
| 4 | 61 | 126 |
| 8 | 11 | 26 |
| 16 | 1 | 5 |
| 64 | 0 | 0 |

Here rank-evaluable means at least one query assay passes the fixed headline
eligibility rule. At k=16, validation supplies only one eligible query assay;
at k=64, no validation unit has enough support, and the one sufficiently large
test unit has no eligible query assay. That budget cannot select/evaluate a
ranking adaptation model. Future results must distinguish changing unit cohorts
from adaptation gains and compare k=0 on each identical query cohort.

No model was fit, no prediction was read, and no scientific experiment is marked
complete by this preparation. Base-model selection already used the original
validation units, so these are conditional probes on the existing benchmark.
Subset-local BCE labels are reconstructed for each support/query view. All source hashes,
51,720 held-out assay memberships, 30,697 within-unit records and every budget
count were independently verified. Source:
`tier4/adaptation_design/{complete.json,verification.json}` on Atum.

Compact token/label-view preparation was integrated from
`codex/paired-adaptation-views`, commit `98bb8b5`, and deployed by Git pull.
Its 28 local CPU checks cover
the existing design/views plus new disk-cache alignment, raw-precision pair
medians, support/query label isolation, shared token buffers, native dataset
consumers, tampering and both-regime preparation. Each within-unit task remains
a separate fit with fixed queries, a k=0 baseline and explicit common-cohort
comparison. Production preparation completed for both splits: eight A views and
6,837 B views share the existing token caches and use 6,074,198 bytes of packed
index/label files. The loader rechecked every vector against raw observations
and the frozen memberships. Subset-local labels differ from source labels in
three A views and 450 B views; these overlapping views are not unique-row counts.
All recorded manifest/vector hashes passed. No fitting or inference occurred,
and the 48 source files pinned by the concurrent GPU audit stayed unchanged.
This adds no adaptation performance result; E49–E52 remain pending until the
Tier 3 and training-protocol/runtime gates are satisfied. Evidence:
`tier4/adaptation_views/complete.json`, its A/B `views.json` manifests, and
`tier4/adaptation_views_launch_49904db.{json,log}` on Atum.

The support-only adaptation fitting worker was added in commit `09135aa`.
Its heads-only scope trains active ranking/BCE heads, contrastive projections
and the learnable scale; the second scope additionally trains the fusion
cross-attention blocks. Both hold encoders, shared input projections and
normalizations fixed. Frozen feature modules remain in evaluation mode.
Thirteen local CPU tests cover optimizer membership, unchanged frozen state,
low-k supports lacking ranking supervision, disabled objectives, query-evaluation
exclusion, exact compact-weight restoration and corrupt payload rejection.
Atum also completed two-epoch synthetic CPU fits in both scopes, with two
optimizer steps each; compact restoration reproduced each fitted model exactly.
All saved payload hashes passed verification at Unix1793579029.77.
No scientific checkpoint or adaptation query set entered these fits.
This qualifies the support-fitting code on tiny CPU models, not production
GPU execution or scientific adaptation performance.
E49–E52 remain pending. Evidence:
`tier4/adaptation_runtime/cpu_09135aa/qualification.json`, its per-scope
`{fit_plan,complete}.json` records, and `cpu_09135aa_verification.json`.

The adaptation protocol is now frozen from commit `17f1328`, before scientific
adaptation fits or prediction scores. It inherits the selected full-arm bases
and optimizer settings, with no additional learning-rate search. Each
across/within, trainable-scope and k cell selects an epoch using only B seed42
validation queries, then applies it to the matched A/B comparisons and
seeds43/44. The native defined-correlation macro convention and explicit
coverage are retained. Test queries and confirmation seeds cannot select an
epoch, and k=0 remains an unchanged comparison on each identical query cohort.

Production accounting gives 22 selection cells, comprising 72 across-unit fits
and 24,192 independent fits on small within-unit support sets across both scopes
and three seeds. Infeasible A within-unit and B k=64 ranking-selection cases
remain explicit. Direct reconstruction from frozen group tables reproduced
every selection inventory and fit count at Unix1793580028.39. Across-unit
selection has 1,956 eligible B validation assays; within-unit k=1/2/4/8/16 has
853/500/203/42/1 eligible validation query assays, spread over the previously
reported 515/220/61/11/1 rank-evaluable units. The small k=16 selection cohort is
a material limitation. Twenty-five local protocol/worker tests passed.
No scientific fitting or inference occurred. Tier-3 completion, verified base
checkpoint binding and production GPU qualification remain required; E49–E52
remain pending. Evidence: `tier4/adaptation_protocol/{plan,complete,verification}.json`
and `tier4/adaptation_protocol_17f1328_launch.json` on Atum.

## Representation geometry preparation — inference pending

E54/E55 have fixed, label-free sampling cohorts for each A/B train, validation
and test partition. Every partition has 25,600 observation-weighted rows and
25,600 unique-molecule representatives, plus all its distinct proteins:

| Partition | A distinct proteins | B distinct proteins |
|---|---:|---:|
| Train | 7,873 | 7,873 |
| Validation | 2,380 | 1,721 |
| Test | 2,421 | 1,723 |

Identity hashes determine selection and ordering, with seed42; a canonical
observation supplies each entity representative. The same rows and distance
pairs will be used across checkpoints. Observation-weighted and unique-entity
views measure different distributions and will remain separate. Independent
checks reconstructed every selection from raw identities: 331,191 memberships
across 18 cohorts, with all source/output hashes verified. Source:
`tier4/geometry_design/{complete.json,verification.json}` on Atum.

Nineteen CPU tests qualify singular-value effective rank, normalized alignment
and uniformity, zero-vector coverage, capture of the actual model's pooled
contrastive projection boundaries without changing weights, and identity-based
export alignment despite reordered caches. Wrong labels/identities, changed
artifacts and the wrong cosine pathway are rejected. The rank definition
is uncentered singular-value entropy, as in
[RankMe](https://arxiv.org/abs/2210.02885); the existing older diagnostic's centered
covariance entropy is a different statistic. The
[alignment/uniformity definitions](https://arxiv.org/abs/2005.10242) use normalized
vectors. Cross-modal alignment applies only in the trained common projection
space; unrelated encoder coordinates are not made comparable by padding.

The production mapping passed separately on all six token caches: 311,343
distinct observations supply the 331,191 view memberships. All selected native
float32 continuous labels and raw/cache observation identities match. The
current cache positions happen to equal raw positions; the worker does not
depend on that coincidence. This CPU check verifies source data, cohort files
and cache registration metadata, but does not rehash the full token payloads
or load a model. Source: `tier4/geometry_cache_alignment/complete.json` on Atum.

These are membership, mapping and numerical checks only. No scientific geometry value,
representation-collapse conclusion or improvement is established. Checkpoint,
token and row provenance, permitted-GPU inference qualification and the actual
measurements remain pending behind Tier3. E54/E55 remain **pending**.

## Gradient and cold-validation probe preparation — scientific results pending

E56/E57 now have a numerical worker and independently verified cold-validation
memberships. Each model uses its own split's validation units with no exact
training-scaffold match. A retains 6,012 rows across 598 whole assays, including
190 ranking-eligible assays (3.07% of A validation rows); B retains all 136,957
validation rows across 9,280 assays / 4,032 components, including 3,969 eligible
assays. Membership selection uses identities and scaffolds; labels only describe
feasibility. A's selected cold cohort does not represent the full A validation
distribution or generator deployment.

An exact identity check finds 107,929 B-validation observations (78.81%) in A
training. These are different split realizations; this is not leakage within
either model's own train/validation split. It does mean that global B validation
cannot serve as a held-out probe for the A model. Independent raw scaffold-set,
whole-unit, identity and float32-label checks reproduced both prepared cohorts
and the cross-split overlap. No test data or model predictions were used.

The worker measures R/B/C gradients on actual parameter blocks. The existing
Trainer diagnostic instead measures gradients at normalized contrastive outputs;
the ranking branch does not depend on those outputs in `mlp_contrastive`, so
those boundary logs cannot answer Q33 about shared encoder gradients. Frozen,
disconnected and zero-gradient cases remain distinct, with undefined cosines
reported as undefined. This structural observation is not evidence of harmful
gradient conflict in a trained model.

Lookahead applies a temporary weighted-SGD step to encoders, interaction layers,
or their union, and measures the loss change on a separate cold validation
batch while keeping private heads fixed. It does not replay AdamW or retrain the
checkpoint. Eleven local numerical/design tests passed, including a case where
training loss improves while validation loss worsens. Atum CPU qualification
also passed on three tiny synthetic configurations (full, frozen encoders and
detached contrastive encoders), with checkpoint state and RNG unchanged. These
are implementation checks, not scientific gradient or transfer measurements.

E56/E57 and Q33 remain **pending**. Scientific execution still needs the Tier 3
results, fixed batch/candidate identities and step sizes, checkpoint/token
provenance, and numerical qualification on the permitted GPUs. Sources:
`tier4/objective_probe_design/{complete.json,verification.json}` and
`tier4/objective_probes/qualification_cpu_add5162.json` on Atum; copied evidence
and current local source hashes match.

## A-versus-B variant reporting preparation — scientific results pending

`variant_relation.py` (source `a1aeafb`) prepares the primary-grid component of
E58/Q17. It requires all 28 canonical primary-arm summaries, covering the 280
already registered fits; it creates no fit or model selection. Each setting
retains its declared readout and B-validation-selected LR. The report keeps all
five checkpoint rules and nine ranking/top-weighted metrics, with seed 42,
seeds 43/44 and the two-confirmation mean separate. The three common-LR seed-42
views are separate sensitivity analyses, not pooled additional observations.

Original score coverage remains visible. A second macro-Spearman view restricts
each regime/seed/rule/partition to eligible assays with defined predictions for
every primary variant; it verifies group identities, sizes and whole units and
reports the coverage lost. This intersection can differ across seeds or rules.
The plots use common axis limits and include a numbered variant/readout/LR key.
Pearson and rank correlations, OLS slopes and in-sample R-squared describe these
registered variants. Shared data, references, tuning and overlapping A/B
partition roles preclude interpreting them as independent samples or proof
that selection on A transfers to B. No p-values or association intervals are
claimed, and test scores do not choose variants, LRs, readouts or rules.

All 22 focused tests passed, including analytic tied-rank/linear/inverse cases,
undefined coverage, missing confirmations, changed native evidence and changes
during reporting. Synthetic raster/vector plots were visually checked. Atum
preflight found zero canonical primary summaries and has Matplotlib 3.9.2. Its
actual group-table reader reproduced verified A14/A15 and B6/B15 means on
7,997 and 3,969 common eligible assays respectively; these are reader checks on
existing reference epochs, not variant-association results. All nine live
process identities and 138 pinned inputs were preserved during deployment.

The MD requires every Tier-3 variant. E40 leak-in/out and E41 learning-curve
comparisons remain additional unfinished inputs to the eventual combined plot;
the primary component explicitly cannot mark E58 complete. Q17/E58 remain
**pending**, with zero scientific variant plots or associations. Evidence:
`variant_relation_runtime/a1aeafb/{deployment,preflight,native_reader_qualification}.json`
on Atum. Preflight SHA256:
`e6a02f07cc54fa4db50bbf9036890a71bf9e08a3e1ad4d9ff3bc692753d00e29`;
reader qualification SHA256:
`fec23f4ad4f2c1a32a0b3a280fd0c8d09d1e699a083249c8b87ce448d47adafb`.

Deployment is unseen generated molecules. That does not establish whether their
scaffolds are unseen. The report must distinguish unseen identity from unseen
scaffold, rather than presume either split alone matches the generator.
