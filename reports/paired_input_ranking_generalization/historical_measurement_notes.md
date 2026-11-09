# Detailed historical observations

Historical snapshot text follows. Dates, prefix lengths, provisional status and earlier scope must be read as historical. Current answers and stop status are in answers.md. The native record and full-precision tables take precedence over rounded prose. These paragraphs preserve earlier measurements and qualifications; they do not authorize further work.

# Detailed historical evidence — study stopped, incomplete

**November 9, 2026:** The user stopped experiments and the persistent goal is
paused. Atum has no project-owned training or controller processes running.
The current consolidated results, 14/24 requested fit status, and all 34
question assessments are in CONCISE_REPORT.md. Sections
below retain their dated historical evidence; references to future queues,
additional seeds or wider experiment scope are not current execution plans.

This records completed audits, not the final answer to all 34 questions. Runtime
evidence is under `source reference` on Atum. The
remaining controlled multi-seed runs and mechanistic probes remain in scope.
Test results do not select models or hyperparameters.

The ledger has 23 complete, 12 partial and 35 pending experiment entries,
including 12 conditional entries. Tier 2 is complete (11/11 entries); the other
three tiers remain incomplete. Of the 34 questions, 12 are answered, fifteen have
answers with limitations, three are partial and four are pending. Q34 is answered
by the user's clarification that the original loss weights were arbitrary. Prior
closures concern the two completed reference trajectories: BCE/AUROC, validation
temperature fitting, logit scale, checkpoint locations, top-weighted metrics,
contrastive directions and fixed-pool rescoring.
They do not close the controlled multi-seed or selected A test comparisons.

The fresh similarity_class baseline finished 15 epochs. Validation macro Spearman
selected epoch 6 (0.18460); its original test macro Spearman was 0.17688.
Epoch 15 minimized aggregate validation loss but had lower validation Spearman
(0.17561). These are one seed and one similarity_class realization, not a completed
method comparison. Source: `baseline_status.json`, checkpoint trainer states.

## Historical loss-weight selection — Q34 answered by user clarification

On 2026-11-02, the user answered the question about the original
R/B/C = 0.25/0.25/1 weights with **"arbitrary selection"**. The reference weights
therefore had no validation-based selection rule on either Split A or Split B.
This answer comes from the user; the W&B audit below alone did not establish it.

A read-only W&B audit on 2026-11-02 found six historical `fusion512_tri_*`
runs, covering three recorded static weight settings on both intended split
regimes. Here R/B/C means ranking/BCE/contrastive, while Split B means similarity_class.

| R/B/C weights | Intended Split A run | Intended Split B run |
| --- | --- | --- |
| 0.5 / 0.5 / 0.5 | `f65gc8ot` | `nasmr8n1` |
| 0.5 / 0.25 / 0.25 | `gy28264l` | `0rgisysy` |
| 0.25 / 0.25 / 1 | `ciqykvch` | `m5guoqkx` (requested reference) |

All six record seed42, `metric_for_best_model=eval_loss`, and interrupted
(`killed`) status. The first four were created on October19 and the
contrastive-dominant pair on October22. None has a linked sweep, run group,
selection notes or tags. This shows that several settings were tried; it does
**not establish whether weights were selected by validation performance, or
which split selected them**. Checkpoint selection by loss is a separate decision
from selecting loss weights. Absence of sweep metadata does not rule out manual
tuning. No historical performance winner is inferred from final run summaries.

Split identities here come from run names and saved launch config arguments;
original dataset hashes were not recovered. All six launch config paths are
absent from their recorded Git commits (two commits across the family), so those
commits cannot reconstruct the executed config bytes. This limits provenance;
it does not prove the historical configurations were invalid. The audit covered
this named family, not every project run or off-W&B decision.

Q34's historical selection question is answered. The prospective A/B weight-selection
comparison still requires the registered controlled fits; E39 remains pending.
No new fit or test-based selection was made. Six metadata files were rehashed
and all six config paths checked against the recorded Git trees. Evidence on
Atum: `source reference`;
audit SHA256 `4d88f1d3bef4c99cf466c6bb081c617c762180c8085429a660ed6dfc2bedf0b0`,
verification SHA256 `62e307b41baee786e341daea1f94d925dcfc94f7e09f9b534fd413848e2c40c6`.

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
`source reference` and
`source reference` on Atum.

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
`source reference` on Atum,
SHA256 `6a8f75cd21dcb9dcbebf1beaaef20e31c81064062c7bf8227041ffbab9f3c9d2`.

## Measurement and composition

The selection metric `eval_macro_spearman` is an unweighted mean of per-group
correlations. `eval_spearman` is instead an alias for the observation-count
weighted mean; neither is a correlation pooled over observations. Groups need
at least three observations and s span≥0.5. Thus two-item groups do not
enter this headline metric. Constant label groups are excluded; tied ranks are
averaged; constant predictions give undefined correlations and reduce coverage.
Source: `source reference`.

The original prediction exports retained raw float64 labels, while the Trainer
evaluates float32 tensor values. On the identical saved best-test predictions,
raw labels give macro Spearman 0.1770244006 across 3,971 eligible groups; rounding
labels as the Trainer does gives 0.1768796098 across 3,972 groups, exactly matching
the native test result. Label precision fully explains this discrepancy; it is
not evidence of an inference-batch effect. The original post-hoc metric,
noise-band and scorer-agreement reports retain their raw-label semantics.
Separate float32-label analyses are complete, without changing weights,
predictions or checkpoint selection. All 93 export/scorer comparisons match the
production metric's eligible groups and per-group correlations to 1e-14;
the selected test macro also matches the historical Trainer result exactly.
The corrected reports are in `source reference`, `source reference` and
`source reference`. Sources:
`source reference`,
`source reference`.

| Test composition | A | B |
|---|---:|---:|
| groups | 16,431 | 9,578 |
| Median / mean group size | 3 / 11.73 | 2 / 14.30 |
| Eligible groups | 8,278 (50.38%) | 3,972 (41.47%) |
| Mean within-group s variance | 0.3046 | 0.1809 |
| Exactly tied label pairs | 6.20% | 5.65% |
| Rows with multiple raw measurements | 3.22% | 3.91% |

B has a smaller median group but a larger mean due to its tail. The differences
in eligibility, variance and near-ties preclude interpreting the raw split gap
as a pure similarity effect. Fixed depth-3 test reweighting and matched neural
validation reweighting are complete below; selected neural test and repeated
comparisons remain pending.
Source: `source reference`.

The actual similarity relation is binary exact similarity_class identity across all
training candidates. In A test, 37,174/192,687 rows (19.29%) and 34,625/162,361
unique candidates (21.33%) have no training similarity_class match. A further 26,417
rows contain unseen candidate identities with a shared similarity_class. All 136,956 B
test rows lack both candidate-identity and similarity_class matches. This binary relation
preserves the splitter's pooled acyclic and private unparseable nodes; it is not
a graded candidate-distance measure.

Saved depth-3 predictions give these test macro Spearman values:

| Evaluation cohort | Eligible groups | Macro Spearman |
|---|---:|---:|
| A, complete groups with all similarity_classs shared | 5,548 | 0.1390 |
| A, complete groups with mixed similarity_class support | 2,497 | 0.1071 |
| A, complete groups with all similarity_classs unseen | 233 | 0.1067 |
| A, shared-similarity_class rows within each group | 7,556 | 0.1342 |
| A, unseen-similarity_class rows within each group | 1,738 | 0.0669 |
| B, complete groups, all similarity_classs unseen | 3,972 | 0.0607 |

The within-group views reapply eligibility after restricting rows. groups can
appear in both views, so their counts are not additive. The unseen-row view is
close to B, but it changes the evaluated groups. Complete all-unseen A groups
instead give 0.1067 with a wide whole-group interval [0.0564, 0.1559]. These
cohorts do not identify a causal leakage share or establish neural performance.

Composition weights were fixed without prediction scores, using group-size,
positive-fraction and mean-s bins. A supports 3,964/3,972 eligible B test
groups (99.80%). On this common support, A falls from 0.1281 to **0.1154** after
weighting; B is **0.0613**, leaving a descriptive gap of **0.0541**. The original
full-partition values are A 0.1285 and B 0.0607. The paired A-unit bootstrap
change is -0.01266, interval [-0.01551, -0.00969]; effective weighted A count is
7,067 groups and defined coverage remains 100%. Validation shows the same
direction: weighted A 0.1221 versus common-support B 0.04884.

Cells match exactly in mass, but residual differences remain inside coarse bins:
weighted mean eligible-group size is 26.11 on A versus 31.93 on B; mean label span is
2.173 versus 1.954. Thus this adjustment does not remove every composition
difference. Intervals condition on one fit, fixed assignments and empirical
weights; they omit training-seed and weight-estimation uncertainty. No additional
LambdaMART fit or tuning was performed. Selected neural test and additional
seed/realization comparisons remain pending. Sources:
`source reference`,
`source reference`, and
`source reference` on Atum. All source/artifact
hashes were verified; an independent weighted sum reproduced both partitions,
and maximum cell-mass error was below 7e-18.

The two completed full-model seed 42 base-LR references now have matched
composition comparisons at all 15 validation epochs. The fixed cells support
3,962/3,969 eligible B groups (99.82%). Standardizing A lowers its point
estimate at every epoch; the same-epoch terminal comparison and each run's
own native validation-macro peak are kept separate:

| Validation comparison | A common support | A standardized | B common support | Standardized A minus B |
|---|---:|---:|---:|---:|
| Both epoch 15 | 0.34301 | 0.33462 | 0.17925 | 0.15537 |
| A epoch 14 / B epoch 6, own macro peaks | 0.34337 | 0.33486 | 0.18420 | 0.15066 |

At epoch 15, the paired A-unit weighting change is -0.00839, with conditional
interval [-0.01223, -0.00516]. Residual imbalance remains: weighted mean group
size is 27.18 on A versus 31.79 on B, and label span is 2.226 versus 2.020.
A's unseen-similarity_class rows within groups score 0.19670 (1,608 eligible groups),
compared with B's 0.17975 (3,969); intact all-unseen A groups instead score 0.22976
(190). These are distinct cohorts, not evidence that similarity explains a
specific fraction of the gap.

The analysis uses validation labels for descriptive standardization. Its
intervals condition on one fitted seed and fixed empirical weights; they omit
seed, split, weight-estimation and checkpoint-selection uncertainty. No joint
cross-split gap interval is asserted, because A/B partition roles can overlap.
The peak comparison is in-sample and does not select an LR or test checkpoint.
No model was fitted and no test predictions were read. All 803 recorded input
hashes, the result digest and execution receipt were checked; independent
weighted point estimates and intact-group stratum means were reproduced.
Within-group restricted-row statistics and bootstrap intervals reuse the native
analysis and were not independently reconstructed by this report. Evidence:
`source reference` on Atum,
SHA256 `1f0d06faf0aa2623365d5f0b28bb6207fd0d0425e9896bbadb32307388109acd`,
and `source reference`.
E08/E09 and Q13 remain partial pending selected neural test and repeated results.

All 1,957,400 curated observations were reconstructed from 2,032,748 retained raw
measurements with zero count or median mismatches. On the replicated subsets:

| Test partition | Split-half macro Spearman | Spearman–Brown heuristic | Eligible group coverage |
|---|---:|---:|---:|
| A: random group | 0.5445 | 0.7050 | 420/8278 (5.07%) |
| B: similarity_class component | 0.3780 | 0.5485 | 348/3972 (8.76%) |

The correction assumes parallel measurements and is not validated as a grouped
rank ceiling. These selected subsets cannot establish a ceiling for the full
test partitions. Source: `source reference`.

Training-only RMS replicate differences were 0.78055 (A) and 0.76605 (B) s.
Test pairs inside those bands were 60.802% and 61.887%, respectively. At the
shared original ranking margin, log10(3), the fractions were 42.297% and 43.640%.
For the selected B checkpoint, ordering accuracy was 0.5892 without a noise band,
0.6305 outside the original margin and 0.6581 outside the replicate-RMS band.
Those accuracies have different denominators. The all-pairs C−D concordance
decreased from 0.1683 to 0.1471 and 0.1205 as more pairs became ties. A larger
filtered accuracy therefore does not mean the model itself improved.
These noise-band values are unchanged after the label-precision correction.
Sources: `source reference`, `source reference`.

Six categorical variance-component fits were identifiable algebraically, but
several fitted noise/residual/interaction components reached the nonnegative
boundary. In particular, zero fitted training noise conflicts with observed
replicate disagreement. These descriptive covariance fits do not establish
noiseless data, absent underlying interactions or an architectural ceiling.
Direct marginal and replicate-contrast checks accompany each fit.
Source: `source reference`; method: REHE.

An group-balanced sensitivity fit changes the estimated within-group interaction
contribution from 0 to 0.1159 s² on A test and from 0.1716 to 0.0568 on B
test. For B training, balancing improves fitted marginal variance from 3.9844
to 2.0749 (observed 1.9578), but the fitted measurement-noise variance remains
far below direct replicate disagreement. Thus Q19 has no stable numerical
partition under this additive constant-variance model. These fits cannot justify
an interaction ceiling or choosing an architecture from a small interaction share.
Raw-measurement noise also cannot be read as the uncertainty of a median label.

## Ranking eligibility and binary supervision

Using the current partition-local binary labels:

| Partition | groups | Headline eligible | All-positive groups | All-negative groups | Same-side labels among headline-eligible groups |
|---|---:|---:|---:|---:|---:|
| A train | 131,450 | 48.59% | 63,190 | 28,715 | 44.50% |
| A validation | 16,431 | 48.67% | 8,069 | 3,537 | 45.28% |
| A test | 16,431 | 50.38% | 7,791 | 3,524 | 43.63% |
| B train | 145,454 | 49.64% | 69,799 | 30,457 | 43.87% |
| B validation | 9,280 | 42.77% | 4,683 | 2,443 | 50.09% |
| B test | 9,578 | 41.47% | 4,919 | 2,664 | 54.20% |

Under headline precision, most training eligibility exclusions come from groups with fewer than three items:
62,516 on A and 67,994 on B. A further 5,067/5,259 groups have enough items
but insufficient float32 label span. Eligible training groups have median size
11 versus 1 for ineligible groups on both splits. Mean within-group variance is
0.5360 versus 0.0863 on A and 0.5492 versus 0.0918 on B. query composition also
differs: the eligible/ineligible target-frequency distributions have total
variation 0.2904/0.2908, and 3,420/3,393 of 7,873 training queries have no eligible
ranking group. These are descriptive selection differences. The headline metric
uses the same nominal eligibility rule, so it does not directly score most
excluded groups. The sampler reads float64 cached labels; headline evaluation
casts labels to float32. There are two training-group inclusion disagreements
on A and one on B at the exact span boundary. The sampler therefore has 72,202
eligible B training groups, versus 72,201 under headline precision. A's two
disagreements cancel in the total count. These tiny boundary effects are kept
separate from the broad size/variance selection described above.

Binary labels give no ordering preference between members with identical labels.
This affects 54.20% of ranking-eligible B test groups, but does not prove BCE
cannot help the shared representation; the loss ablations must test that.
For the threshold at s 6, 35.58%/32.21% of A/B test rows have their
partition-local pair median within the training replicate-RMS band
(0.78055/0.76605). This is threshold proximity, not a calibrated probability of
label error or uncertainty interval for median labels.
Source: `source reference`, with six-partition group,
target and unit tables. This supplements the earlier original-label audit.

## Baseline ladder

All results below use frozen pretrained features or entity identities, training
measurements only, and validation-only selection. A uses partition-local binary
labels; continuous labels are unchanged. The Spearman eligibility rule is the
original n≥3 and label span≥0.5. Undefined correlations are excluded, with their
coverage reported explicitly.

| Baseline | A test macro Spearman | Defined eligible groups | B test macro Spearman | Defined eligible groups |
|---|---:|---:|---:|---:|
| candidate-only linear ridge | 0.0775 | 100% | 0.0624 | 100% |
| Same-query similarity_class-nearest lookup | 0.2572 | 79.72% | undefined | 0% |
| Independent per-query ridge | 0.2419 | 99.83% | 0.0726 | 99.52% |
| Two-step ridge | 0.1034 | 100% | 0.0488 | 100% |
| Identity-based linear matrix filter | 0.4207 | 93.32% | undefined | 0% |
| Paired Kronecker ridge | 0.1690 | 100% | 0.0631 | 100% |
| query-shuffled Kronecker ridge | 0.0662 | 100% | 0.0596 | 100% |
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
checkpoint. Sources: `source reference` and both split
`complete.json`, `source reference` and `source reference` artifacts; final checksums were verified.

The candidate-only and independent-task models each test five regularization
values per split. Two-step ridge tests the 25 combinations of the two sides'
five values. It fits measured pairs in stage one, then regresses those fitted
functions on query features. This is an explicit sparse-data adaptation, not
a complete-matrix formula applied to artificial zero labels. Sources:
`source reference`, `source reference`, `source reference`.

The lookup uses the actual split relation: exact similarity_class identity, with all
acyclic candidates sharing one similarity_class node. Similarity ties are averaged over
unique training candidates measured for the query query. On B every nearest
similarity is zero, so it reduces to a query prior. Its within-group ordering
accuracy is 0.5. Source: `source reference`.

The matrix filter uses the smoother kernel
`(1 + same_query) * (1 + same_candidate)`, fitted only to observed measurements
with their original frequencies. Five ridge values are selected on validation.
Its A test bootstrap interval is [0.4111, 0.4304], conditional on the held-out
groups and the 93.32% defined coverage. On B, unseen candidates receive only a
query-specific score: pairwise ordering accuracy is 0.5. A strong A result
is therefore possible without learning transferable candidate features. This
does not estimate the causal leakage contribution to a neural model's gap.
Source: `source reference`; method family:
Stock et al..

The query-only linear BCE model has validation/test AUROC 0.6687/0.6696 on A
and 0.6452/0.6493 on B. The lookup's B query-prior scores have validation/test
AUROC 0.6915/0.7060 while carrying no within-group order. Classification can
therefore reward target base rates. Comparing these AUCs does not identify a
fraction of the full model's signal attributable to those priors; Q32 still
needs the matched deep-model comparison on A.

Kronecker ridge fits the exact augmented linear product kernel on observed
pairs, using a matrix-free solve with verified relative residual below 2e-7.
It uses no completed interaction matrix or random-feature approximation. The
shuffled model is retrained after independently permuting query assignments
across whole groups within each partition. Group-level query frequencies,
candidate inputs and labels are preserved. Both versions use the same five-value
regularization grid and select on their own validation macro Spearman. The
comparison covers one permodification per partition, not permodification uncertainty.
Sources: `source reference`, `source reference`.
The paired-minus-shuffled test difference is 0.1029 on A (paired whole-group
bootstrap interval [0.0920, 0.1138]) and 0.0035 on B (whole-component interval
[−0.0157, 0.0221]). This kernel baseline shows a clear A advantage for correct
query assignments; the B contrast is inconclusive. These intervals condition
on the fitted models and permodifications. Source: `source reference`.

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
`source reference`.

The bilinear manifest's `git_commit` was captured at completion. The baseline
entrypoint and its project-local dependencies were verified unchanged between
the launch checkout c920520 and completion checkout 355d147. Future runs
capture the launch commit and source hashes before fitting.

The MD's final Tier-2 rung explicitly says: "Score Split B groups with the
current model's contrastive similarity alone." At the jointly trained B model's
validation-selected epoch-6 checkpoint, cosine alone gives test macro Spearman
**0.06623**, with whole-component interval **[0.05048, 0.08216]**, compared with
the ranking head's **0.17688 [0.15991, 0.19362]**. Both are defined on all 3,972
eligible test groups. These are two readouts of the same fitted model, not a
contrastive-only training experiment or a causal test of removing cross-attention.

E32 is therefore complete under the explicit Tier-2 Split-B requirement;
the separate both-split pathway analysis remains partial under E53. A current
check of the retained weights and all 15 validation evaluations confirms the
unchanged epoch-6 selection. Independent average-rank calculations reproduced
both readouts for all 9,578 test groups (maximum difference 2.23e-16), and
independent unit-count aggregation reproduced the 2,000-draw intervals while
retaining all 4,034 components, including those with no eligible groups.
Intervals condition on this fitted seed, selected checkpoint and held-out
components; they do not include seed or selection uncertainty. No fitting,
inference or new checkpoint selection occurred. Sources:
`source reference`, the saved `source reference` export,
and `source reference`.

## Agreement between scoring pathways

All 31 saved B prediction exports were scored for within-group agreement using
Trainer-compatible label precision and average ranks, including tied predictions.
The correction preserves the validation agreement values below.
On the same 3,969 eligible validation
groups, every pathway pair was defined at all three epochs below:

| Validation scorer pair | Epoch 1 | Best checkpoint, epoch 6 | Epoch 15 |
|---|---:|---:|---:|
| Ranking head / BCE logit | 0.6798 | 0.6363 | 0.6017 |
| Ranking head / contrastive cosine | 0.0743 | 0.2359 | 0.2341 |
| BCE logit / contrastive cosine | 0.0958 | 0.2759 | 0.2769 |

Ranking–BCE agreement declines by 0.0781 from epoch 1 to 15 (whole-component
bootstrap interval for the change [−0.0985, −0.0600]). Ranking–cosine agreement
instead rises by 0.1598 ([0.1321, 0.1891]). Thus the evidence does not show
general divergence among all three scorers. Despite increasing agreement with
the other heads, cosine's correlation with the observed-score labels changes only
from 0.0653 to 0.0749; its change interval [−0.0085, 0.0270] includes zero.
On the fixed training probe, ranking–BCE agreement increases from 0.6452 to
0.6776 over the same mutually defined eligible groups. These are correlations
of predictions, not gradient measurements or proof of objective interference.
The intervals condition on one fitted seed and similarity_class realization. Together
with the complete A trajectory below, they answer Q31 descriptively; controlled
repeated and selected-test pathway comparisons remain unfinished. Source:
`source reference` and per-group trajectory tables.

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

The fixed 2,048-group training probe's eligible macro Spearman rises from
0.18336 at epoch 1 to 0.53189 at epoch 15. Validation peaks at 0.18460 in epoch 6
and ends at 0.17561. Later improvements on the training probe transfer poorly
in this run. These observations do not isolate encoder drift, loss conflict or
another cause; matched controlled and multi-seed comparisons remain pending.

On the same 3,969 eligible validation groups as Spearman, linear-gain NDCG@1
and NDCG@3 peak at epoch 11, while NDCG@5 and Spearman peak at epoch 6.
NDCG@1/@3/@5 at epoch 6 are 0.59687/0.66438/0.71182; at epoch 15 they are
0.59098/0.66322/0.70751. Thus top-ranking quality has a later local maximum,
but no sustained improvement through epoch 15. Averaging NDCG over all
nonconstant groups instead changes the top-1/top-3 peak to epoch 8, illustrating
why cohort definitions must be aligned before comparing objectives. These
descriptive extrema do not establish significant differences or objective conflict.
Source: `source reference`.

The learned logit scale rises from 13.5999 to 18.4685 to 22.5461 at epochs
1/6/15. Validation contrastive loss falls from 8.54974 to 6.00606 to 5.83497;
rescoring the same representations at fixed scale 13 gives 8.52970, 5.82207 and
5.55379. Candidate identities are fixed throughout, including in the original
validation procedure. Sharpening raises loss relative to fixed-scale rescoring,
but there is no sustained total contrastive-loss rise in this fresh run.
This does not predict the outcome of retraining with fixed temperature.
Sources: `source reference`,
`source reference`, `source reference`.

The matched A reference (`full__A__seed42__lr1p0`, W&B `xffenit7`) completed
all 15 epochs and 12,030 optimizer steps on GPUs 0,2,5,6 with batch12×4×4.
Native validation and fixed training-probe audits cover all 15 epochs. These
cover 195,618 validation rows / 7,997 eligible groups and a fixed 39,740-row /
2,048-eligible-group training probe. They reproduce both native headline metrics
and preserve the Trainer's float32 label semantics.

| A epoch | Validation macro rho | Training-probe macro rho | Validation BCE loss | Ranking loss | Contrastive loss |
|---|---:|---:|---:|---:|---:|
| 1 | 0.16488 | 0.16214 | 0.52074 | 2.66708 | 4.86771 |
| 2 | 0.21244 | 0.22805 | 0.48617 | 2.61290 | 3.32545 |
| 3 | 0.24317 | 0.27084 | 0.46223 | 2.57423 | 2.72154 |
| 4 | 0.26145 | 0.30225 | 0.46034 | 2.54027 | 2.42469 |
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
and 0.76583 for original B. The corresponding query-only frozen-feature
baselines give 0.66874 and 0.64521. Relative to chance AUROC 0.5, those baselines
recover about 49% and 55% of the reference models' excess AUROC. This is a
descriptive comparison of separately fitted models, not a causal fraction of
the BCE head's signal. query-only predictions are constant within an group.

On the same 7,997 eligible validation groups, NDCG@1/@3/@5 rises from
0.59835/0.66331/0.71098 at epoch 1 to 0.70900/0.74819/0.78223 at epoch 15.
All three increase from 14→15 while macro Spearman decreases by only 0.00036;
this point contrast is not evidence of a reproducible objective conflict.
Epoch-15 BCE-logit and cosine-only macro Spearman are 0.27035 and 0.17598.
Constant predictions remain undefined rather than being jittered or assigned
artificial correlations. Validation-only temperature fitting gives T=1.29538
and reduces log-loss from 0.42772 to 0.41749. This is an in-sample calibration
fit; no test transfer has been evaluated for A.

At epoch 15, complete shared-similarity_class, mixed and unseen-similarity_class validation
groups score 0.37470/0.28168/0.22976 (5,405/2,402/190 eligible groups).
All three point estimates are below epoch 14; the all-unseen score is also
below its epoch-11 maximum of 0.23762. Restricting to unseen-similarity_class rows
within groups instead gives 0.19670 on 1,608 eligible groups; the cohorts differ.
Fixed composition weighting changes A common-support validation macro from
0.34301 to 0.33462. The weights use the B validation label mix. The completed
matched validation comparison above also reports B predictions; selected
neural test and repeated comparisons remain pending.

Across epochs 1→15, ranking–BCE agreement changes 0.63066→0.63610 on the same
7,995 mutually defined eligible groups. Its whole-group change interval
[−0.00474, 0.01599] includes zero. Ranking–cosine agreement rises
0.12360→0.29585 on 7,997 groups, with interval [0.15822, 0.18617].
The trajectories fluctuate and do not show uniform scorer divergence.

The completed run selects **epoch 14 by validation macro Spearman (0.34367521)**.
Minimum aggregate validation loss instead occurs at epoch 15 (2.36395550),
whose macro Spearman is 0.34331738. All 15 epoch weights are protected and the
retention hold has been released. Training finished at Unix1793574255.346448.
The native A test remains unscored pending the registered selection protocol.
Training-probe statistics are descriptive; intervals condition on one fit and
allocation. Multi-seed and selected-test method comparisons remain pending.

Evidence: `source reference`,
`source reference`, and
`source reference`
with immutable `trajectory_through_epoch15.json` and
`verification_through_epoch15.json`. Independent calculations reproduced all
831,555 per-group/scorer correlations across 30 snapshots, including undefined
cases, to 1e-14; 894 sampled pair-count checks and all 15 composition-weighted
point estimates passed. Earlier checks are inherited only after exact source,
prefix and table verification. This verification does not independently
reconstruct bootstrap intervals, calibration fits or all NDCG values.

The controlled B run first received a separately verified **ten-epoch prefix**.
Its probe uses the same fixed 2,048-eligible-group rule as A, unlike the original
B baseline's uniformly sampled group probe. The sampled groups still differ.

| Epoch | A validation macro rho | B validation macro rho | A probe macro rho | B probe macro rho |
|---|---:|---:|---:|---:|
| 1 | 0.16488 | 0.10992 | 0.16214 | 0.17815 |
| 2 | 0.21244 | 0.14458 | 0.22805 | 0.24104 |
| 3 | 0.24317 | 0.16912 | 0.27084 | 0.29117 |
| 4 | 0.26145 | 0.16577 | 0.30225 | 0.33091 |
| 5 | 0.27953 | 0.17172 | 0.33753 | 0.37329 |
| 6 | 0.28843 | 0.18458 | 0.36608 | 0.39862 |
| 7 | 0.30225 | 0.17889 | 0.38964 | 0.42904 |
| 8 | 0.31324 | 0.18049 | 0.40509 | 0.44715 |
| 9 | 0.31908 | 0.17596 | 0.42791 | 0.46300 |
| 10 | 0.32604 | 0.17776 | 0.45533 | 0.48243 |

At epoch 10, probe-minus-validation rho is 0.12929 on A and 0.30467 on B,
up from 0.05800/0.20157 at epoch5. B's best validation score within the observed
prefix is at epoch6 (0.18458). From6 to10, its probe rises0.39862→0.48243 while
validation falls0.18458→0.17776. These are descriptive probe gaps,
not full-training generalization gaps or evidence of a final plateau. A/B have
different group populations and 802/877 optimizer steps per epoch. B's probe
contains 43,115 rows from only 25 similarity_class-group components, so its 2,048 groups
are not independent sampling units. B validation has 136,957 rows and 3,969
eligible groups. All 339,840 group/scorer correlations across twenty snapshots and
600 sampled pair-count checks passed independent verification. This does not
independently verify the bootstrap intervals, calibration fits or all NDCGs.
This prefix was subsequently extended to15 below; repeated controlled fits
remain pending. Prefix evidence:
`source reference`
with `trajectory_through_epoch10.json` and `verification_through_epoch10.json`.

The controlled B training phase has now completed all 15 epochs and 13,155
optimizer steps. All four rank-completion receipts match its frozen config.
The saved native validation history selects **epoch6, macro Spearman0.18457785**;
epoch15 gives **0.17975193** and the minimum aggregate validation loss,
**6.68894768**. All3,969 eligible validation groups have defined correlations.
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
`source reference`; the controlled checkpoint-state SHA256 is
`f0c91b09d7dfb5877090e7131dd1370bf456e1c54c1cb66e6f163f3d94b071fa`.
Repeated controlled comparisons and all completion counts remain unchanged.

The one-shot CPU handoff (`native_horizon_queue.py`, commit `fec3286`) completed
at Unix1793600683.91 after verifying the original driver's completion. All30
validation/probe snapshots are independently checked: **509,760** per-group/scorer
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
`source reference`.

From epoch6 to15, B's fixed-probe macro rises **0.39862→0.51969**, while validation
changes **0.18458→0.17975**. The final probe-minus-validation difference is
**0.33994**. This probe contains2,048 eligible groups but only25 components;
it is not a full-training estimate or independent seed evidence. Native BCE,
ranking and contrastive validation losses attain minima at epochs2,3,15,
respectively. Their first adjacent increases are2→3,3→4,6→7; an adjacent increase
alone does not establish sustained overfitting. Learned scale rises13.5988→22.5452.
At epoch15, validation-only temperature1.4732 changes BCE logloss0.49210→0.46908
without changing score order; this is an in-sample calibration result.

On the same3,969 eligible validation groups, ranking NDCG@1/@3/@5 peaks at
epochs6/13/8. All three finish below their epoch6 values, so this run does not
show sustained late top-ranking improvement. From1→15, within-group
ranking/BCE agreement falls0.67647→0.59235 on3,968 mutually defined groups,
whereas ranking/cosine agreement rises0.07402→0.23018 on3,969. Final BCE-logit
and cosine label macros are0.13944/0.07754, below the ranking head's0.17975.
These prediction correlations do not establish gradient conflict. Separate
directional/fixed-scale results remain labelled with their original source runs.

After verified B completion, the existing tuning queue launched
`full__A__seed42__lr0p3` (W&B `lekrhvg9`) on physicalGPUs0,2,5,6. At
Unix1793600720.55 it was at epoch0.2122; no LR has been selected and no
confirmation seed has started. Scientific completion counts remain unchanged.

The first five completed epochs of that already running A LR0.3 candidate now
have an independently verified native validation/probe audit. Compared with
`full__A__seed42__lr1p0` at the same epochs and optimizer steps:

| Epoch | Reference validation rho | LR0.3 validation rho | Reference probe rho | LR0.3 probe rho |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.16488 | 0.13113 | 0.16214 | 0.12658 |
| 2 | 0.21244 | 0.17298 | 0.22805 | 0.17453 |
| 3 | 0.24317 | 0.19579 | 0.27084 | 0.20061 |
| 4 | 0.26145 | 0.21412 | 0.30225 | 0.23170 |
| 5 | 0.27953 | 0.23105 | 0.33753 | 0.25786 |

Both use exactly the same 195,618 validation rows (7,997 eligible groups) and
39,740 fixed-probe rows (2,048 eligible groups), with full defined coverage.
Recorded configurations differ only in output directory and the four learning
rates, all multiplied by 0.3: main/projection 1e-4 to 3e-5 and both encoders
1e-5 to 3e-6. Training qualifications match. At epoch5/step4,010, LR0.3 has
validation rho lower by0.04848 and probe rho lower by0.07968. Its validation
aggregate loss is4.11930 versus2.93847, and eligible NDCG@1/3/5 is also lower
(0.63377/0.69218/0.73471 versus0.67335/0.71781/0.75630).

This is early learning-rate sensitivity on one seed and one A split. At this
five-epoch audit the LR0.3 fit was still running; the prefix does not establish a
preferred LR, final overfitting behavior, test generalization or seed variance.
The training probe is descriptive. Neither the registered training schedule nor
the B-validation selection rule was changed in response to these observations.

All ten snapshots passed independent checks of277,185 group/scorer correlations
and300 sampled noise-band pair counts. The matched comparison rehashed558 inputs
and checked exact validation/probe identity and label equality. Bootstrap
intervals, calibration and NDCG were not independently reconstructed by that
verifier. No model inference, new fitting or test evaluation occurred.
Evidence: `source reference`
`{plan,complete,exit,matched_reference_evidence}.json`, and
`source reference`
`{trajectory_through_epoch05,verification_through_epoch05}.json` on Atum.
Trajectory SHA256 `95abce78a3a1ac386fa2b4e9571725f11d949d436c130f0325643b9cc73f013a`;
verification SHA256 `c713c2e7c190ef9c97ffebc29f3a21f900aeddd2054458001cf932efaf1e128e`;
matched evidence SHA256 `8c5d47fee2df1321891a00327aa5fe3894c7bf0e2c302b600f6f32f7d191660d`.
Q04 and the incomplete metric/controlled-comparison experiments remain partial.

The LR0.3 fit has since completed all15 epochs/12,030 optimizer steps. Its native
validation macro peaks at epoch14/step11,228: **0.28035523**, with all7,997
eligible groups defined. Epoch15 gives0.27927916 and the minimum aggregate loss,
3.29775214 (BCE0.45061868, ranking2.50890136, contrastive2.55787206;
AUROC0.81272319). The native best-macro checkpoint is retained. The existing
completion/retention verifier passed; training-completion SHA256 is
`01430261a06ff4159a3465436cf3cc8d18040a01261ddcba9895b11c03164a78`
and retention SHA256 is
`0338d7df7caef0bc7f6a20d2ea742873cd4de64144995777bd1499da9c09faf8`.
The independent audit has now completed all30 validation/probe snapshots,
checking831,555 group/scorer correlations and900 sampled noise-band pair counts.
It confirms the validation macros above and a final fixed-probe macro of0.33896153.
All15 epochs retain full defined coverage on7,997 validation and2,048 probe
groups. The earlier five-epoch evidence is unchanged. The worker exited0 at
Unix1793623080.57; completion receipts and all snapshot table hashes were
rechecked. Bootstrap intervals, calibration and NDCG were not independently
reconstructed. Evidence: `source reference`
`{complete,exit}.json`, completion SHA256
`dc3fcc9748e8c35eefe141ccfd32db4e188b43e8fb178ca7baf15cd60717cd8c`;
the corresponding native `trajectory_through_epoch15.json`, SHA256
`8ee661e8f322a2d56750c32fa3783a4441765a370be10ac1dab72b7520c56595`,
and `verification_through_epoch15.json`, SHA256
`a1f0cafdb350d73ff5173233a8ddd9c4ec4d7b0bd860845743662bd97eb7fb80`.
The existing tuning queue launched `full__B__seed42__lr0p3` (W&B `bcisozj7`)
on physical GPUs0,2,5,6; its launcher and all four exact ranks were verified
live. No common LR choice, confirmation seed, selected test comparison or new
experiment closure follows.

The completed full-horizon comparison now confirms that A seed42 LR0.3 has
lower validation macro rho at every matched epoch than the base LR. Both
native validation peaks occur at14. The same195,618 validation rows/7,997
eligible groups and39,740 fixed-probe rows/2,048 eligible groups have full
defined coverage throughout. Configurations differ only in output directory
and the four learning rates multiplied by0.3; qualification and labels match.

| Metric | Base LR | LR0.3 | LR0.3 minus base |
| --- | ---: | ---: | ---: |
| Best validation macro rho, epoch14 | 0.343675 | 0.280355 | -0.063320 |
| Final validation macro rho, epoch15 | 0.343317 | 0.279279 | -0.064038 |
| Final fixed training-probe macro rho | 0.492076 | 0.338962 | -0.153115 |
| Final aggregate validation loss | 2.363955 | 3.297752 | +0.933797 |
| Final eligible validation NDCG@1 | 0.708997 | 0.673520 | -0.035477 |

This is learning-rate sensitivity for one A seed and the fixed15-epoch budget.
It does not establish performance after longer training, a preferred LR for
unseen generated candidates, a causal mechanism or seed/split variance. The
registered B-validation selection rule is unchanged. The full report preserves
all15 same-step comparisons, all native loss components and eligible NDCG@1/3/5;
native losses/NDCG reuse the earlier audits rather than being independently
reconstructed by their correlation verifier. All21 focused comparison tests
passed. Atum execution exited0;816 input hashes and both source hashes were
rechecked, and the first five reported epochs exactly match the earlier report.
No fitting, inference or test scoring occurred. Q04 remains partial, and
completion counts are unchanged. Evidence: `source reference`,
SHA256 `e931f093e3f156c4c1ba08642c83632a2c04f2deccaaae15b6676cbea81427e9`;
`source reference`,
execution SHA256 `bd9328e23c887dd43124cc192948f1d9452d901843656a76341af9cfd86e780f`,
verification SHA256 `7db2bdb7deefcfecad5a116854ce7de745e2dc77f65e2e5748ea771c205e8192`.

Automatic CPU audits were queued for the three remaining, already registered
full-arm tuning fits: B LR0.3, then A/B LR3.0. The separate worker
`native_tuning_audits.py` (source `2f586fd`) initially waited for the
B fit, with no visible GPUs and two CPU threads per numerical library. It
requires completed training, four rank receipts and retention before analyzing
all15 validation/probe epochs. It makes no fit, inference, test-scoring or LR
selection call. Existing or failed analysis output requires inspection.

All65 focused worker/handoff tests passed. Atum preparation froze85 inputs,
21 analysis/worker sources and the existing43 training sources; its completion
reader also accepted the already independently verified A LR0.3 fit without
rerunning that audit. The live training launcher and four permitted GPU ranks
were unchanged. At activation this was automation preparation, with zero
additional fits or completed scientific audits and no change to experiment/question counts.
Evidence: `source reference`
and `source reference`
`{deployment,preflight,activation,completion_reader_qualification}.json` on Atum.
Plan SHA256 `493afe4a28e02efa54b22e4ce309fbd48571b7a750d197c11e80e117c40d03f6`;
activation SHA256 `8d17f23ed730d0fd0df73bd4d8b233f6f281ca3f85614348efd4962ef333c4c1`;
reader qualification SHA256
`563ff600030c36678b09c18364f22c380b782c593d3f1108707b199983337dba`.

The B LR0.3 fit has now completed all 15 epochs/13,155 optimizer steps, its
scheduled allocation exports and checkpoint retention. Best native validation
macro is **0.18322109 at epoch 12/step 10,524**; the final epoch scores
**0.18051769** and minimizes aggregate validation loss at **6.92203045**
(BCE 0.44447416, ranking 2.79370379, contrastive 6.11248589; AUROC 0.76762593).
The saved root model and epoch-12 weights have the same SHA256,
`9eb3f1fea142bd1cdc08e313f29c176bc057441b99a153ccda292437b7099570`.
All 15 epoch weights remain protected for the pending allocation analyses.
Of 3,969 eligible validation groups, 3,968 have defined correlations at epochs
1, 3 and 5; all 3,969 are defined at every other epoch. Undefined groups remain
excluded from the native macro with their coverage reported.

The existing CPU worker independently checked all 30 validation/probe snapshots:
509,760 group/scorer correlations and 900 sampled noise-band pair counts.
From epoch 12 to 15, fixed training-probe macro rises 0.36193716→0.37511298
while validation changes 0.18322109→0.18051769. Final BCE/cosine label macros
are 0.12382694/0.07383095. These compare separate fixed cohorts and pathways;
they do not identify a causal mechanism. Bootstrap intervals, calibration and
NDCG are not independently reconstructed by this verifier.

Evidence on Atum: `source reference`
`source reference`, SHA256
`2be77d9e4c726411f58da31078379aecc86c0fb80b91ef01e3080c62c1d339ba`
and `01d3a6f559bce3c6dd82b9ebb446938ee9bc3c3844d2b644c3532b971f48df19`;
`source reference`,
SHA256 `b201186e5b3b04767093e4d8b577da0a3533045090617b5239511e7dfb0bc8e6`;
`source reference`
`{trajectory_through_epoch15,verification_through_epoch15}.json`, SHA256
`ce30524c1a301e80fd62b336ecfc39d3c787b2a9906c59a8d2aaa746817d499d`
and `5e087a1705fe5e3a09a12a147faf30758caa7907a99c943c3869d3c3fb0f3730`.
The audit completed at Unix 1793646616.31; its receipts were rehashed on read.
At that snapshot the queue started `full__A__seed42__lr3p0` (W&B `6w5p03mw`),
with B LR3.0 next. Four controlled fits were complete; common LR, confirmation
seeds and selected controlled test results were pending. Experiment/question
completion counts were unchanged. Per the user's monitoring preference, idle
periods use interruptible sleeps followed by isolated milestone checks.

The A LR3.0 fit (`full__A__seed42__lr3p0`, W&B `6w5p03mw`) has also completed
all 15 epochs/12,030 optimizer steps and its independent native audit. Both
best and final validation macro are **0.37531688 at epoch 15/step 12,030**.
The minimum aggregate validation loss is also at 15: **2.25452399**
(BCE 0.43168974, ranking 2.39050078, contrastive 1.54897630; AUROC 0.85745553).
The root model and epoch-15 checkpoint share SHA256
`c14cd77b1bc3703a113855281fbbc6087d83c2a6cbaa5a4091ea8095a3cc7763`.
Of 7,997 eligible validation groups, 7,996 have defined correlations at epoch 5;
all 7,997 are defined at the other epochs. The fixed training-probe final macro
is 0.57819093 on 2,048 groups/39,740 rows. Final validation BCE/cosine label
macros are 0.31171948/0.17898956.

The independent audit checked all 30 validation/probe snapshots, 831,555
group/scorer correlations and 900 sampled noise-band pair counts. It finished
at Unix 1793668988.65; the trajectory, verification and completion receipts
were rehashed on read. Bootstrap intervals, calibration and NDCG are not
independently reconstructed by this verifier. Compared descriptively with the
completed A base-LR run, the native peak is higher by 0.03164167 and the final
macro by 0.03199949. This is one seed under a fixed 15-epoch budget; the peak at
the final epoch does not establish what longer training would do or select the
common LR for deployment.

Evidence on Atum: `source reference`
`source reference`, SHA256
`59eac1be7ea547102fe7144c6731a44edadd6d876ccd342b66558b2e1ff37e27`
and `17ac01d6290a9f843be44467109bf5ab4512ed784d2a1897778f4699e54c93da`;
`source reference`,
SHA256 `e436874fd329d6d5faf7351891aa686b61327fbfd69350bdaf66d2ca4862a8e8`;
`source reference`
`{trajectory_through_epoch15,verification_through_epoch15}.json`, SHA256
`29cbce6666310e20bc61234c0433de64d1e3d1c27bf658c5fcd47e7c1cfe2da4`
and `1a43d9ef58cd837396029fd786982e4896e8caac40562f4f3c0ba6a347b6721b`.
At that snapshot the queue started `full__B__seed42__lr3p0` (W&B `op384ho2`)
on the four authorized GPUs, with five controlled fits complete. Common-LR
selection, confirmation seeds and selected controlled tests were pending.

The B LR3.0 fit has since completed all 15 epochs/13,155 optimizer steps,
allocation exports, retention and its independent native audit. Best validation
macro is **0.17310025 at epoch 6/step 5,262**; final macro is **0.16400919**.
Aggregate validation loss also reaches its minimum at epoch 6: **7.21106720**,
then finishes at **7.63197660** (BCE 0.57388800, ranking 3.10172462,
contrastive 6.71307325; AUROC 0.75424943). Root and epoch-6 weights share
SHA256 `e89262ae7f44c301420d6f8240ee5a86c7d9b1b62f9654c5319e29a2f1ffbf00`.
All 15 epoch weights remain protected. Of 3,969 eligible validation groups,
3,968 have defined correlations at epoch 11; all 3,969 are defined otherwise.

All 30 validation/probe snapshots passed the independent audit: 509,760
group/scorer correlations and 900 sampled noise-band pair counts. Final fixed
training-probe macro is 0.59629982 on 2,048 groups/43,115 rows; final validation
BCE/cosine label macros are 0.13624922/0.07550496. Bootstrap intervals,
calibration and NDCG were not independently reconstructed. Compared with B base
LR, both native peak and final macro are lower, despite the higher final probe
macro. A LR3.0 improved A validation, whereas B LR3.0 degraded B validation;
these single-seed observations do not establish a mechanism or variant-wide
association.

Evidence on Atum: `source reference`
`source reference`, SHA256
`9a4554eb53f128e06be0374b3ea57db4d912134094f326709b4ebae5a3d40075`
and `bafab69f9aa24558e299273056832111ab2b84e8c7a1d0363dff184ea1a66aec`;
`source reference`,
SHA256 `1d3a09a2bf7a84942352bd53c507b5bce59fa53882718cdd5b47c9f38688b72e`;
`source reference`
`{trajectory_through_epoch15,verification_through_epoch15}.json`, SHA256
`fbf3a9eb8314f08cfdd351b91b519c84c6b1efe28a6b7ef6b4a5e57f2136726e`
and `f5a95395c4accbe2438d84ecf3ac85ae0eeebc52052dffb37e1ebf820a69ccdb`.
The audit worker finished all three registered audits and exited 0 at
Unix 1793692166.52; its process was verified absent. Its overall completion
SHA256 is `e8fa06c420f79bd6cbab7a1fc4cf86ace1cf3be7fb91e9d1037d46a9e4e1bd05`.

All six full-model seed42 tuning fits are now complete and audited. The existing
B-validation selector chose **base LR (×1.0)**: candidate peaks are 0.18457785
(base), 0.18322109 (×0.3) and 0.17310025 (×3), each with full defined-group
coverage at its selected epoch. This choice was reconstructed from the native
validation snapshots and retention records, without test scores. It applies to
both A and B confirmations. The original tuning coordinator finished with its
verified completion receipt and is absent. At selection time the existing
confirmation coordinator started `full__A__seed43__lr1p0` (W&B `hnnyo965`)
on GPUs 0, 2, 5, 6, followed by B seed43, A seed44 and B seed44. Confirmation
and selected controlled-test results were then pending.

Selection evidence: `source reference`, SHA256
`7ccd63ecdd1ece26e2df45ec654b60e38443ea8b51d663adbf02140bfc986d70`;
`source reference`, SHA256
`faa97952273faaebfc381ed3b30ddc0a45798e22b64353bcf6eec4cd3867c7c7`;
`source reference`, SHA256
`b83719020ccc651244c87bca6b9569f5e2db59bb40b110f797b4edbb533ec9e5`.
Six completed fits do not finish the replicated experiments; question and
experiment completion counts remain unchanged.

The first confirmation, `full__A__seed43__lr1p0` (W&B `hnnyo965`), has now
completed all 15 epochs/12,030 optimizer steps and checkpoint retention.
Best and final native validation macro are **0.34493525 at epoch 15**;
aggregate validation loss also reaches its minimum at 15: **2.36235642**
(BCE 0.43771893, ranking 2.41682124, contrastive 1.64872134;
AUROC 0.83988365). Among 7,997 eligible validation groups, 7,996 are defined
at epochs 4 and 7, 7,995 at epoch 13, and all 7,997 at the other epochs.
All 15 saved native validation metrics match finalized Trainer history.
Root and epoch-15 weights share SHA256
`6ea396207d72a389340d347a9e302cb9527e2719b122a58017fb44ee5fb29072`.

The four-rank training completion and retained best checkpoint were verified,
with the original launcher absent. This is a completed confirmation fit,
separate from the six tuning fits and the initial requested baseline. Its
native peak is close to A seed42's base-LR peak 0.34367521, but one confirmation
does not establish seed variability. This read checked saved metric consistency
and checkpoint provenance; it did not independently reconstruct correlations
from predictions. The existing selected-evaluation pipeline follows all four
confirmation fits. Independent trajectory analysis, selected controlled tests
and paired seed43 A/B comparisons remain pending.

Evidence on Atum: `source reference`
`source reference`,
SHA256 `34743d890316f85addafc191d344690a36d7adaf6c592d1e4e6dbb2e0b260fb6`,
`08311fd81b38d747540f1bc35867c85d7f8fa4237f04d445a1d9e60fa1743924`
and `460e7165cf6f78bc65060b5d4a61624a829da54fbdf69a25df5d9ffe5cb06047`.
At that snapshot the confirmation queue started `full__B__seed43__lr1p0`
(W&B `8k6easjx`) on GPUs 0, 2, 5, 6, with A/B seed44 next. Seven controlled
fits were complete, including one confirmation.

The second confirmation, B seed43, has since completed all 15 epochs/13,155
optimizer steps, scheduled allocation exports and checkpoint retention. Best
native validation macro is **0.17738602 at epoch 5/step 4,385**; final macro is
**0.16829567**. Aggregate validation loss reaches its minimum at epoch 15:
**6.64202976** (BCE 0.48945102, ranking 2.94575047, contrastive 5.78322935;
AUROC 0.76699191). Of 3,969 eligible validation groups, 3,968 are defined at
epoch 11 and all 3,969 at the other epochs. All 15 native validation metrics
match finalized Trainer history. Root and epoch-5 weights share SHA256
`ce6fb3ca5664e03ebf7dfbf0a7af79e1b78ccc5f9da489c16038e3d626678d7d`;
all 15 epoch weights remain protected for the allocation analyses.

B seed43 has lower peak and final native validation macro than the tuning-exposed
B seed42 base-LR fit (0.18457785 and 0.17975193). The seed43 A/B own-checkpoint
validation peaks are 0.34493525/0.17738602, on different validation populations
and at epochs 15/5. These are descriptive comparisons; seed44, independent
trajectory analysis and selected controlled-test comparisons remain pending.
The completion check verified all four ranks, the finalized Trainer/native
metric consistency and retained checkpoint provenance; it did not independently
reconstruct correlations from predictions. Training and export launchers were
verified absent before accepting managed completion.

Evidence on Atum: `source reference`
`source reference`,
SHA256 `8e2410eb9afd84fd49b628cc28ea35b30fb1459af0db8d6df7ef39794881f2b0`,
`0b22359f250279c63efe6bed6aa561f1a2e42b3a5931619e4cd895ac54d936a2`
and `187fb9122488bafe7c59a296df0c301a82cc74c5740e6cb70cbc006452b0cc6f`;
`source reference`, SHA256
`ecde2cbef0220bf6f6571c6d150133aea41f3d757fbd8689ad17ad24cc5caca9`.
The existing queue now runs `full__A__seed44__lr1p0` (W&B `j5hc20fr`) on GPUs
0, 2, 5, 6; B seed44 follows. Eight controlled fits are complete, including
two of four confirmations. Question and experiment completion counts are unchanged.

### User amendment: two training seeds total

The user subsequently instructed: **“2 seeds are enough, skip seed stuff.”**
The active training-seed scope is now **42 and 43** for all remaining experiments,
overriding the MD's minimum of three. This changes training repetitions; it does
not remove experiments, LR candidates, held-out allocations or adaptation units.
Seed42 remains tuning-exposed; seed43 is the only confirmation. Reports cannot
estimate variability across confirmation seeds. Partial seed44 results are
excluded from selection and scientific comparisons.

The exact A44 launcher/ranks and the three obsolete dependent coordinators were
terminated after identity, ownership, command and evidence checks. All eight
completed fits and partial A44 output were preserved; B44 never started. All
target handles were verified absent and the four approved GPUs were idle.
Cancellation evidence: `source reference`
`{cancellation_intent,cancellation_complete}.json`; completion SHA256
`4fdad8030a05fcfda3c1f9af46442d9b57c92081e6d3f792ebc64a2f9969efcc`.

Historical plans, source qualification and completed artifacts remain immutable.
The new `two_seed_policy.py` binds the user amendment to that cancellation and
the original primary plan; future confirmation registration admits seed43 only.
The primary grid consequently has 168 tuning plus 56 confirmation fits, **224
total**, instead of 280. This accounting is not a launch authorization or a
claim that the remaining work is affordable. The open compute-budget decision
still governs expansion beyond the already queued scope.

`two_seed_full_evaluation_queue.py` uses the unchanged native evaluator and
per-fit analysis for exactly the eight completed full-model fits, with no new
training or selection. `two_seed_summary.py` retains all checkpoint rules,
pathways, paired whole-unit contrasts, composition comparisons and LR
sensitivity; seed42/43 summaries and the single confirmation remain distinct.
The new summary namespace is `controlled_summaries_two_seed`. All 19 focused
tests passed, including exact fit accounting, exclusion of seed44, immutable
selection, cancellation/identity gates, missing evidence, source changes,
resource waits, partial-output refusal and report artifact association. These
software checks do not complete a scientific experiment. Existing three-seed
downstream reporting and training coordinators must use amended replacements;
the retired coordinators must not be restarted.

The earlier reference-A contrastive handoff stopped when its audit rejected encoder
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
`source reference` and
`source reference`.

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
Evidence: `source reference`.

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
Evidence: `source reference`.

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
`source reference`.

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
Evidence: `source reference`
and `source reference`.

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
Evidence: `source reference`.

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
`source reference`.

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
`source reference` on Atum;
preflight SHA256 `87a6ee1cbf100e4da256d52bd61ed77c0e6e335131c3df09f4f86e774013c6a7`.

Q30/E38's seven-loss report is implemented in `loss_subset_analysis.py`
(commit `074f5aa`). It requires all seven verified primary-arm summaries, covering
42 tuning and 28 confirmation fits including the ten shared full-reference fits.
Declared ranking-head/BCE-logit/cosine outputs stay explicit. R/RB/RC/RBC also
receive four auxiliary-effect contrasts and the conditional interaction
RBC−RB−RC+R on their common eligible, mutually defined group cohort. The 2,000
paired whole-unit bootstrap draws retain zero-eligible units; seed spread,
confirmation-only spread and different A/B populations stay distinct. These
comparisons include each arm's validation-selected LR/checkpoint and do not
establish gradient conflict. All 27 focused tests passed, including independent
unit-resampling arithmetic, cohort/scope rejection and cached-report integrity.
Atum's CPU-only preflight found no primary-arm summaries and created no fit,
selection or scientific comparison. Deployment preserved all nine live process
identities and 105 pinned inputs. Q30 and E38 remain pending; completion counts
are unchanged. Evidence:
`source reference`.

E34–E37/Q09 reporting is implemented in `encoder_analysis.py` (commit `56e8cb8`).
It requires the six existing full/frozen/last-two-block/LoRA-4/16/64 primary-arm
summaries: 36 tuning and 24 confirmation fits, including the ten shared full
fits. Each arm keeps its B-validation-selected LR; all five checkpoint rules,
three diagnostic scorers and same-LR seed42 results remain visible. Encoder-minus-
full contrasts use paired whole-unit intervals on mutually defined eligible
groups, with tuning-exposed and confirmation-only seed spreads kept separate.
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
`source reference` on Atum;
preflight SHA256 `8b5f5da5eca4683c783aba48ab9f0f6201a17b798f677e04227c9903ec82c0f9`.

E42–E47 reporting is implemented in `objective_controls_analysis.py` (commit
`a0ed17b`). The five contrastive settings require 50 existing primary fits; the
seven ranking settings require 70. Their ten full-reference fits are shared,
so the two families cover 110 unique planned fits, not 120 or any new fits.
Four contrastive and six ranking contrasts preserve their intended references:
uniform/top weights both use pairwise objectives; noise-soft/hard queries both
use zero margin; margin removal and group balancing receive separate contrasts.
The exact model-only changes, train-derived noise temperature, common training
budget and canonical primary evaluation provenance are checked before reporting.

All settings, three scorer outputs, five checkpoint rules and same-LR seed42
results remain visible. Paired contrasts use mutually defined eligible groups
and whole-unit intervals; seed42 exposure, confirmation-only seed spread and
descriptive A-minus-B changes remain separate. Each setting retains its own
B-validation-selected LR, so these are comparisons of selected procedures.
They do not by themselves establish false-negative causality, calibration or
gradient mechanisms. All 30 focused tests passed, including correct contrast
signs/references, changed or incomplete evidence and child-artifact modification
during reporting. Atum's CPU-only preflight verified 66 distinct registered
tuning configurations and found zero primary summaries for either family.
No training, family selection or scientific comparison was created; E42–E47
remain pending. Deployment preserved eleven live handles and 138 pinned inputs.
Evidence: `source reference`
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
Evidence: `source reference`.

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
seed44 contain 9,280/9,448/9,475 groups and 4,032/4,031/4,032 indivisible units.
This qualification loaded no model and computed no loss or test score. The
scientific stage waits for completed fits and frozen allocation LR selections;
no experiment or question is newly closed. Evidence:
`source reference`.

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
`source reference`.

The allocation-specific CPU diagnostics and seed summaries are implemented in
`realization_analysis.py` (commit `27901fe`). All 19 focused tests passed,
including actual tiny-model predictions, independently calculated Spearman,
whole-component coverage, cached-output integrity and rejection of incomplete
or mismatched seed/allocation grids. Reporting keeps seed42 tuning exposure,
seed43/44 confirmation spread, dependent allocation sensitivity and conditional
whole-unit intervals separate. All five checkpoint rules are required; scores
and coverage remain tied to verified child analyses and current source receipts.
Atum qualified all six descriptor views over 273,913 unique held-out rows.
Original/seed43/seed44 validation eligibility is 3,969/3,963/4,059 groups; test
eligibility is 3,972/3,978/3,882. Recomputed coarse composition cells support
99.80%/99.82%/99.92% of eligible test mass in A; this is descriptor support,
not model performance or evidence that all composition differences disappear.
No model was loaded and no prediction scores were read. Scientific per-fit
diagnostics and the complete seed/allocation summary still await the selected
evaluations. Deployment preserved all eight live training/queue identities and
their frozen inputs. Completion counts are unchanged. Evidence:
`source reference`.

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
`source reference`.

The CPU analysis of completed primary-grid evaluations is implemented in
`selected_analysis.py` (commit `38f1196`). It requires the frozen validation-only
selection and every checkpoint/partition receipt before reading predictions.
It computes the MD's rank, pair, noise-band, size, similarity_class, composition,
calibration, BBSE and pathway-agreement diagnostics, with whole-unit intervals.
Matched A/B comparisons use the same arm, seed, learning rate and checkpoint
rule; test scores cannot select a configuration. All 12 focused tests passed,
including actual tiny native exports and independent rank/pair calculations.
Atum verified all 662,218 rows across A/B validation/test against the design and
cache metadata; both validation fingerprints matched native training exports.
No selected-model scientific predictions or scores were produced. Deployment
preserved active training and its queue. Evidence:
`source reference`.

The primary-arm summary is implemented in `controlled_summary.py` (commit
`b7568b4`). It requires six tuning and four selected confirmation analyses,
re-verifies their validation-only decisions and native evaluation evidence, and
reports all three scorers across all five selected-LR checkpoint rules. Seed
spread, seed42 LR sensitivity, same-seed A/B gaps and within-run checkpoint
contrasts remain separate, with explicit score coverage. All 26 focused CPU
tests passed, including artifact integrity and rejection of actual CPU-native
qualification predictions as scientific GPU evidence. Atum cross-checked the
paired calculation on the existing hash-verified A epochs14/15: 7,997 mutually
defined eligible groups, 16,431 resampled whole-group units, mean change
-0.00035783 and interval [-0.00233925, 0.00164323]. This reproduces the existing
reference result; it is not a new fit, seed or method comparison. Read-only
preflight found no frozen full-arm LR decision and five incomplete tuning fits;
no scientific primary summary was created. All eight active training/queue
handles and their frozen inputs survived deployment. Completion counts remain
unchanged. Evidence:
`source reference`.

Q11's scalar localization audit covers all 15 initial B validation epochs and
all 15 verified A validation epochs. B's BCE rises by 0.03768 from its overall minimum at
epoch 2 to epoch 15, with a positive point change in every fixed size bin.
Ranking changes below instead compare the macro-Spearman peak at epoch 6 with
epoch 15, on the same eligible groups at both endpoints.

| group size | B validation rows | BCE change, 2→15 | Eligible groups, 6→15 | Macro rho change, 6→15 |
|---|---:|---:|---:|---:|
| 1 | 3,769 | +0.04321 | 0 | undefined |
| 2 | 1,988 | +0.00754 | 0 | undefined |
| 3 | 1,623 | +0.02937 | 329 | −0.08247 |
| 4–10 | 9,812 | +0.02029 | 1,330 | −0.00158 |
| 11–25 | 21,365 | +0.04284 | 1,222 | +0.00120 |
| >25 | 98,400 | +0.03883 | 1,088 | −0.00725 |

Three-item groups account arithmetically for 76.1% of the observed net macro
decline, despite representing 8.3% of eligible groups. The overall decline is
−0.00899 with paired component interval [−0.02078, 0.00302]; the three-item
change interval is [−0.15882, −0.00549]. These descriptive intervals condition
on one fit and the observed endpoint choices, exclude selection/seed uncertainty,
and are not adjusted for multiple strata. They do not establish a reproducible
localized failure. The BCE increase is broader: >25-item groups contribute
74.0% of its net increase and contain 71.8% of rows.

B has only unseen-similarity_class groups under the predefined binary relation, so
that split cannot localize the change across different similarity strata.
A's complete trajectory separates the overall BCE improvement from its cold-group
point trend:

| A similarity_class class | Validation rows | Eligible groups | BCE, 1→15 | Macro rho, 1→15 |
|---|---:|---:|---:|---:|
| All shared | 104,730 | 5,405 | 0.52598→0.39065 | 0.16017→0.37470 |
| Mixed | 84,876 | 2,402 | 0.51247→0.45957 | 0.17779→0.28168 |
| All unseen | 6,012 | 190 | 0.54630→0.62373 | 0.13584→0.22976 |

The all-unseen BCE change is +0.07743, but its paired whole-group interval
[−0.03240, 0.19907] includes zero. The ranking change is +0.09392, interval
[0.03030, 0.15353]. BCE uses all 598 cold groups; headline ranking uses the
190 eligible groups. The cold BCE minimum is epoch 2 and macro-rho maximum
is epoch 11. All six size bins improve in BCE from 1→15, and all four eligible
size bins improve in macro rho. This localizes a cold-group BCE point trend
alongside improved ranking; it does not establish calibration as the cause.

Across all A validation groups, the 14→15 macro change is −0.00036, with
conditional whole-group interval [−0.00234, 0.00164]. Thus the small terminal
point decline does not establish a reproducible drop. The cold-only 14→15
macro and BCE change intervals also include zero. From 13→15 the cold BCE
point increase is 0.01004 with interval [0.00179, 0.01978]; these endpoints
were chosen after observing the overall BCE minimum, and the interval omits
endpoint-selection, seed and multiple-stratum uncertainty.

Q11 now has a descriptive answer from both complete reference trajectories;
controlled repeated and causal comparisons remain pending. Source:
`source reference`.
All 30 snapshots reproduce native aggregate BCE within 1e-5 and headline rho
within 1e-14. Independent rank calculations checked 385,665 group correlations,
including undefined cases. Independent Bernoulli log probabilities, fixed
memberships, counts, means and rank-sum AUROC passed for every stratum.
The verifier checks all 30 snapshots directly; it does not inherit earlier
numerical results. All source and table hashes passed. Bootstrap intervals
were not independently reconstructed; no test predictions were read.

The native listwise-loss localization covers all 15 initial B epochs and
all 15 matched A epochs. It preserves the original prediction order, three deterministic
partitions, complete-group eligibility, top-position/margin weights and
zero-loss lists. The loss averages bounded lists, so a large group contributes
more than one list; strata below still use the complete group's size.

| B group size | Lists per partition | Ranking-loss change, 5→15 | Paired component interval |
|---|---:|---:|---:|
| 3 | 329 | +0.03489 | [−0.00721, 0.07964] |
| 4–10 | 1,330 | +0.07428 | [0.04723, 0.10139] |
| 11–25 | 1,806 | +0.06589 | [0.04007, 0.09489] |
| >25 | 6,476 | +0.17152 | [0.11978, 0.22901] |

Overall ranking loss rises 2.79392→2.92871 from its minimum at epoch 5, a
change of 0.13480 with interval [0.09921, 0.17574]. All eligible size bins have
positive point changes. Using the same epochs 6→15 as the macro-Spearman
comparison, >25-item groups account arithmetically for 80.2% of the net
ranking-loss increase, and three-item groups for 0.9%. The former carry 65.1%
of native list weight but 27.4% of eligible groups; the latter carry 3.3% of
list weight but 8.3% of groups. Thus loss and macro-rank deterioration localize
differently. This descriptive accounting does not establish that changing the
normalization improves generalization; that needs the controlled training arm.
All intervals retain the single-fit, observed-endpoint and multiplicity limits above.

A's ranking loss falls 2.66708→2.41737 over epochs 1→15, reaching its minimum
at 15. In fully unseen-similarity_class groups it changes 2.57456→2.54119; the paired
change interval [−0.12493, 0.06786] includes zero. Thus the native loss and
group-macro ranking metric describe different changes even within cold A.
Source: `source reference`.
All 30 reconstructed global losses differ from native values by at most
3.85e-7. Direct production-loss calls on 390 sampled lists agree within
4.77e-7; all 1,146,645 list descriptors, fixed stratum reductions and source/output
hashes were verified. Individual list losses are sampled, not exhaustively
recomputed, and intervals were not independently reconstructed.
No test predictions, model weights or GPUs were used in this verification.

The contrastive localization audit attributes loss to each query's original
group while keeping the complete candidate pool and all masks unchanged.
It covers B epochs 1–15 at both the learned scale and fixed scale 13, including
all 9,280 groups, with 15,407 bounded lists per partition. Zero-active lists
retain denominator mass; each active query retains its native inverse-square-root
list-size weight. Strata use full group size, not bounded-list size. These are
contributions to the original task, not separate retrieval tasks restricted to
each stratum.

| B contrastive direction | Epoch 1 | Epoch 5 | Epoch 6 | Epoch 15 | Minimum epoch |
|---|---:|---:|---:|---:|---:|
| x→y, learned scale | 5.73059 | 3.99055 | 3.96577 | 3.79393 | 15 |
| y→x, learned scale | 2.81915 | 2.03734 | 2.04029 | 2.04105 | 5 |
| x→y, fixed scale 13 | 5.74136 | 4.05543 | 3.99684 | 3.84456 | 15 |
| y→x, fixed scale 13 | 2.78834 | 1.85659 | 1.82522 | 1.70923 | 15 |

The learned-scale y→x direction first increases at epochs 5→6
and ends only 0.00370 above its epoch-5 minimum. x→y first
increases at 6→7 but reaches its minimum at 15. Both fixed-scale directions
first increase at 6→7 and reach their minima at 15. Local increases and
minimum epochs describe this trajectory; they are not statistical change points.

On the common macro-peak comparison, epochs 6→15, learned-scale
x→y loss falls by 0.17184 and y→x changes by
+0.00076. The latter rises in the 1-, 3-, 4–10- and 11–25-item bins, while
falling in the 2- and >25-item bins. Only three-item groups show an increase
in the sum of both directions (+0.01709). At fixed scale 13 both directions
fall in every size bin; the three-item total changes by −0.09395. This
isolates a numerical scale effect on the saved representations, not a benefit
of fixed-temperature training. The contrastive three-item cohort contains
541 groups, compared with 329 for eligible macro Spearman; their changes
must not be treated as measurements of the same cohort.

These are descriptive single-fit comparisons with fixed candidate contexts.
No confidence intervals or redrawn-pool uncertainty are claimed: resampling
query units alone would leave the cross-unit candidate context conditional.
Source for the B attribution:
`source reference`.

The completed protected A audit now supplies both directions for every validation
and original training-probe checkpoint. Every epoch uses the same candidate-pool
hash within its partition. Both directions decrease at every adjacent epoch on
validation and on the fixed training probe, at both the learned scale and scale13.

| A validation contrastive quantity | Epoch 1 | Epoch 14 | Epoch 15 |
|---|---:|---:|---:|
| Learned x→y | 3.37895 | 1.11556 | 1.11034 |
| Learned y→x | 1.48876 | 0.54360 | 0.54235 |
| Learned total | 4.86771 | 1.65916 | 1.65269 |
| Fixed-scale13 total | 4.89917 | 2.01028 | 2.00395 |

At A epoch15, learned-scale loss is 0.35127 below fixed-scale13 rescoring;
at original B epoch15 it is 0.28118 above it. Thus the numerical scale effect
has opposite signs on these fixed representations. This is not evidence that
changing the training temperature improves generalization. A mean positive-row
cosine rises 0.41887→0.48340 on validation and 0.41171→0.51366 on its training
probe. These are descriptive means on different fixed cohorts, not a calibrated
memorization test. Source:
`source reference`,
SHA256 `94c681a6638c900d327c3973d93ed7b55c6a3ae5306866e56121210422c3a039`.

A's completed stratum attribution shows that the scale effect differs within
that split as well. At epoch15, learned versus fixed-scale13 total losses are:

| A similarity_class class | Learned scale | Fixed scale13 |
|---|---:|---:|
| All shared | 1.18614 | 1.50644 |
| Mixed | 2.52134 | 3.02026 |
| All unseen | 3.97861 | 3.77506 |

Within the cold group, learned y→x loss has its minimum at epoch9
(1.45856) and ends at 1.49886; fixed-scale13 y→x loss reaches its
minimum at epoch15 (1.24496). Cold total loss has its minimum at epoch13 at both
scales, with small terminal increases. From the overall macro peak14→15,
both directions decrease in every size and similarity_class stratum at both scales.
The three-item learned total changes by −0.01131, contrasting with original
B's +0.01709 over its own macro-peak6→15 interval; these intervals differ.
This describes the observed references and does not establish a repeatable
localized failure, a causal calibration mechanism or a training benefit.

The attribution keeps all candidates and masks fixed and covers 16,431 groups,
24,089 bounded lists per partition and three partitions at each checkpoint.
All30 table hashes and eleven analysis-source hashes passed. Its 1,980 sampled
production-operator checks agree within1.43e-6, and global directional values
reproduce the completed reference audit. These checks do not independently
reconstruct every list contribution; no cross-unit-context confidence interval
is claimed. The first attempt's filename failure is preserved; the successful
retry canonicalizes validated integral epochs without changing model code.
Sources: `source reference`
`{complete.json,completion_check.json}` and
`source reference`.

All 1,260 sampled production-batch checks agree within 3.82e-6. Independent
explicit candidate-set calculations check 210 batches / 2,520 query lists;
all 1,386,630 list-instance descriptors and source/output hashes were verified.
An initial absolute tolerance on unnormalized float32 list sums missed one
value by 1.03e-5; float64 reconstruction agrees with float64 production loss
within 3.56e-15, and the saved float32 discrepancy is at most 8.57e-7 in
native batch-loss units. The precision diagnosis is retained in verification.
No training, model inference, GPU use or test predictions were involved.

## query coverage in the B test set

All 1,723 test queries have training observations. The number of distinct
training groups per test query ranges from 1 to 1,303, with median 24.
Weighting queries by their number of test groups gives median 127 instead.
The complete target list and counts are saved in `source reference`.

| Training groups for the query | Eligible test groups | Test macro Spearman | Whole-component interval |
|---|---:|---:|---:|
| 1 | 71 | 0.0539 | −0.0701 to 0.1748 |
| 2–5 | 149 | 0.0849 | 0.0018 to 0.1716 |
| 6–20 | 311 | 0.1829 | 0.1324 to 0.2345 |
| 21–100 | 1,184 | 0.1626 | 0.1350 to 0.1891 |
| >100 | 2,257 | 0.1935 | 0.1710 to 0.2148 |

Every eligible group has defined predictions. The strata reconstruct the
selected checkpoint's overall test macro exactly. Low-support point estimates
are lower and uncertain; the pattern is not monotone across all bins. group
composition also differs across bins, so this is not a causal coverage effect.
Intervals condition on one fitted model and split. Source:
`source reference` and per-group support table.

## Sampling and contrastive negatives

Q22's recorded observation process is now audited. The source curation retains
both sides of the binary label threshold: exact, numeric s measurements from
single-query measurement groups with confidence score ≥8 and no recorded variant,
with canonical-candidate normalization, quality/structure checks and within-group
median aggregation.
The later binary label-balanced resplit copies all source observations and changes
their partition assignments. The 1,957,400 original curated observations are
**69.04% positive** at s≥6, before the training pair-median label policy.

Within their 164,312 groups, the archive contains 4,161,420 binary label records.
There are 2,034,645 numeric s records (69.2176% positive); 2,032,748 of
these feed the curated observations (69.2218% positive before aggregation).
Another 2,126,775 records lack numeric s, and 1,897 defined records fall
outside the retained IDs. The broader records include 1,345,088 `endpoint_K1`/`endpoint_K2`
measurements without s and 500,807 ordinary inequality-qualified records.
These are binary label records with potentially multiple endpoints per pair, not
counts of distinct missing pairs or negative examples.

| group provenance | Curated observations | Positive fraction |
|---|---:|---:|
| Source collection L | 1,138,209 | 67.79% |
| Source collection P | 637,588 | 82.86% |
| Source collection Q | 116,824 | 4.61% |

Thus source composition is heterogeneous, and the near-70% fraction is already
present among the archived numeric measurements. The available tables do not
identify each experiment's sampling frame, per-pair inclusion probability or
tested-but-unpublished results. Preferential selection of promising pairs is
not established by these counts; unobserved pairs remain unlabeled. This is a
qualified answer to Q22, not an estimate of binary label prevalence for generated
candidates or a causal explanation of the A/B performance gap.
Source: `source reference` on Atum.
All retained binary label IDs reconstruct the original medians/counts and match the
raw database. Independent SQL totals and output/source hashes were verified;
the 30.48 GB database's content hash and original release hash are recorded.

The configured first epoch contains 12,820 local microbatches on A and 14,020
on B, with 12 dataset items per device. These are bounded group lists plus
classification-only rows. Ranking-eligible lists average 55.83%/56.25% of each
batch, with standard deviations 14.68/14.74 percentage points. Both splits'
5th-95th percentile range is 33.33%-83.33%; each has one batch with no ranked
list. This audits the actual sampler implementation and first-epoch ordering,
not a saved trace of historical device execution.

The reference already masks cross-group same-query and duplicate-query-
candidate negatives, as well as other candidates in an anchor's own group.
However, a candidate sampled under another query can be a known positive for
the query query in a different training group. In 512 evenly spaced batches
per split, such entries make up 0.530%/0.714% of x-to-y negative
comparisons on A/B, or 0.437/0.576 per active anchor. Reverse-direction fractions
are 0.441%/0.657%. A numerical gradient check confirms that the current loss
pushes these entries down. Known positives are defined only by curated training
observations above s 6; unmeasured positives are unknown. These counts
justify testing a global training-positive mask but do not demonstrate a
performance improvement from it.

Candidate query frequency is skewed: the top 1% of 7,873 queries supply
30.07%/30.64% of A/B group-list opportunities. The median is two per query;
maxima are 1,587/1,589. No sampling-probability correction is applied. Frequency
skew alone does not establish a learned popularity shortcut. Sources:
`source reference`, with per-batch and per-query tables.

## Distribution shift

The actual indivisible units have markedly different size distributions:

| Partition | Unit definition | Units | Median rows | 99th percentile rows | Maximum rows |
|---|---|---:|---:|---:|---:|
| A train | group | 131,450 | 3 | 122.0 | 49,020 |
| A validation | group | 16,431 | 3 | 143.4 | 1,409 |
| A test | group | 16,431 | 3 | 137.7 | 1,147 |
| B train | similarity_class–group component | 570 | 6 | 620.0 | 1,661,628 |
| B validation | similarity_class–group component | 4,032 | 5 | 431.4 | 7,382 |
| B test | similarity_class–group component | 4,034 | 5 | 447.4 | 5,683 |

A randomly reserves one training anchor group per query/input, then randomly
assigns the remaining whole groups; it does not preferentially select large
groups. B forces its giant component and 569 target-coverage rescue components
into training, then balances the remaining whole units between validation/test
by row count and positive rate. The giant alone supplies 98.70% of B training
rows. B assignment therefore is not random sampling of comparable training and
held-out units. These counts establish construction asymmetry, not a numerical
causal share of the performance gap. Sources: `source reference`,
the original split manifests, `random_group.py`, and `similarity_class_component_split.py`.

The E41 whole-unit learning-curve data are prepared, with no fits performed:

| Requested unit fraction | A rows retained | B rows retained | A binary labels changed from full-training labels | B binary labels changed |
|---|---:|---:|---:|---:|
| 10% | 9.51% | 98.86% | 2,141 | 0 |
| 25% | 25.93% | 99.01% | 4,247 | 0 |
| 50% | 49.85% | 99.52% | 5,732 | 0 |
| 100% | 100% | 100% | 0 | 0 |

Each subset contains whole units from a fixed nested permodification. A binary-label
change means medians were correctly recomputed using only included training
measurements, not evidence that a model improved. At 10% units, A retains queries
covering 93.46% of test groups; B retains 100% target coverage at every fraction.
B's selected prefixes all contain the giant component, so these curves cover
a narrow 98.86–100% row regime. A flat B curve would not establish broad data
saturation. Source: `source reference` and the eight pinned view manifests.
Production-data CPU checks passed for the 10% A/B views, including row identities,
subset-local binary labels, shared token buffers and the actual sampler. This
qualifies data loading; it does not establish training or performance results.
Source: `source reference`.

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
deployment. E41/Q29 remain partial: fitting, native scoring and scientific
learning-curve results are still required. Evidence on Atum:
`source reference`.
Preflight SHA256 `bfeb49c95c52ad64391db077e3704faa55e977dae74ecae4dc9a054e4b7df324`;
view qualification SHA256 `71f364cfdd0edcd3f4e0deb6be1f85ac2f6ab11d3ed45538cb0822c4ced827c0`.

The E41 CPU analysis/reporting path is implemented in
`learning_curve_analysis.py` (source `bcd5a85`). It requires all 30 smaller-fraction
fit analyses and the ten-fit canonical full-reference summary, preserving the
shared endpoint, validation-only LR choices, all five checkpoint rules, three
scorers and nine core metrics. It reports individual seeds, confirmation means
and spread, common-LR seed42 sensitivities, and adjacent slopes per actual row
and unit doubling. Common-cohort ranking comparisons use groups eligible and
defined at all four fractions, with paired whole-unit intervals that retain
zero-eligible units. Full-training similarity_class strata are not reused as subset
training support. No result selects a fraction, LR or checkpoint using test data.

All 29 local analysis checks passed, alongside the unchanged 21 native-evaluation
checks. These include real tiny-model native predictions checked against
independent rank metrics, report assembly over synthetic 40-fit records, changed
cohort/artifact rejection, empty-cohort handling, non-monotone curves, equal-size
undefined slopes, and independent row/unit counting. Synthetic PNG/SVG plots
were visually checked. These fixtures are qualification, not model results.

Atum's CPU qualification independently counted all eight selected views and
rehashed 28 inputs. The A views contain 13,145/32,862/65,725/131,450 units and
149,177/406,906/782,230/1,569,095 rows. The B views contain 57/142/285/570 units
and 1,664,366/1,666,810/1,675,358/1,683,487 rows: even the nominal 10% unit prefix
contains 98.864% of B training rows. Thus a B curve here measures changes near
full row coverage, not a low-data regime. One nested subset permodification and fixed
15-epoch training do not isolate sample size from composition, subset-local
labels or optimizer-step exposure.

Preflight found zero curve analyses and zero canonical full-endpoint summaries.
An attempt to analyze a fit without native evaluation was refused before
predictions were read. No trained weights, token payloads or predictions were
read in this CPU qualification; no fitting, inference or scientific curve was
created. E41/Q29 and all completion counts remain unchanged. Evidence on Atum:
`source reference`.
Preflight SHA256 `0435d6847520332094d95aacebf9e202d006fee415dc6493311cb190615f0265`;
axes qualification SHA256 `02be6bc93fc1e350cca9636692b5c1c8ff2aa9e8f0671e707cbef071c3d7ea1f`.

E40's fixed-test leak comparison has data-construction evidence, not model
results. Version 2 matches whole-group query identity, size, BCE-positive
count, strict contrastive-active count, replicate histogram and ranking
eligibility. The first version matched BCE positives but changed contrastive
actives by -106 on A and +15 in the B proposal; those artifacts remain recorded
and are superseded for future fitting. Both versions have byte-identical test
tables, so the additional constraint did not change test selection.

| E40 construction | Training rows per arm | Matched group exchanges | Test groups exposed to training similarity_class relatives |
|---|---:|---:|---:|
| A, whole groups | 351,451 | 9,261 | 11,786 of 16,431 |
| B, whole training components | 1,683,487 | 0 | 0 of 7,724 |
| B, whole-group exception proposal, not selected | 1,683,487 | 806 | 2,279 of 7,724 |

A exchanges 14,157 rows and exposes 50,512 test rows to similarity_class relatives;
17,186 test rows have exact candidate overlap. Both arms retain 29,471 training
groups, 3,682 queries, 270,755 BCE positives and 267,768 contrastive actives.
The production cache views passed identity, subset-label and sampler checks:
35,136 sampler items and 183 optimizer steps per epoch in each arm. query
support is identical between arms, including 8,951 test rows with unseen
queries. Performance comparisons must report that common limitation.

B reserves one label-blind whole donor group from each multi-group original
test component and fixes the remaining 7,724 test groups before matching.
All 570 training components contain at least one query absent from this donor
pool, preventing whole-component removal with unchanged query support for
this construction. The optional whole-group proposal exchanges 1,529 rows,
partially retains nine training components and exposes 5,494 test rows, including
2,897 exact candidate overlaps. It is awaiting the user's choice; no B exception
training is authorized by these manifests. Both B arms match 1,143,159 BCE
positives and 1,123,169 strict contrastive actives. Sources:
`source reference`, `source reference`, and
`source reference` on Atum. All recorded artifact
checksums were verified after the CPU jobs completed.

The original similarity_class splitter's seeds 42, 43 and 44 produce identical component
assignments: the deterministic size-sorted candidate wins all three searches.
They cannot be counted as three realizations. Randomizing input order among
equal-sized components before the same size-priority allocation and 64-restart
balance search produces two distinct additional validation/test allocations.
Against the original, these move 86,592 and 94,118 of the 273,913 held-out rows
(3,987 and 3,962 whole components). All three retain exactly the same 570
training components and 1,683,487 training rows. Validation/test row counts and
raw positive rates are unchanged.

These are dependent reallocations of a fixed held-out pool, with no change to
training support or similarity_class geometry. They support a bounded allocation
sensitivity analysis, not independent split draws. Each needs its own validation
checkpoint and learning-rate selection before test scoring. The assignment
audit is complete; performance spread and multi-seed comparisons remain pending.
Source: `source reference`.

A fixed logistic classifier used group descriptors, with whole units held out
of fitting. Each domain and each unit within a domain had equal fitting weight.

| Comparison | Unit-balanced accuracy | group-balanced accuracy | Holm-adjusted permodification p |
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
group weighting different estimands; this classifier's holdout excluded that
giant component. Failure to distinguish validation and test is not proof that
their distributions are equal. Source: `source reference`;
method: C2ST.

B validation-to-test BBSE estimated test positive prevalence at 0.7850, with a
whole-unit bootstrap interval of [0.6696, 0.8789]; observed test prevalence was
0.7902. The confusion matrix condition number was 3.81. This diagnostic assumes
stable class-conditional prediction distributions, and validation also selected
the checkpoint. It is not an independent calibration experiment.
Source: `source reference`;
method: BBSE.

## Staged adaptation data design — fitting pending

E49–E52 now have verified group memberships, not adaptation performance.
An identity-only hash divides original validation units into adaptation and
validation sets; original test groups remain unchanged. A has 8,215/8,216/16,431
groups in these three roles. B has 4,722/4,558/9,578 groups across
2,016/2,016/4,034 components. Across-unit adaptation uses nested whole-group
prefixes at k=1/2/4/8/16/64, shared between the heads-only and heads-plus-interaction
settings. Token/label views and the fitting/epoch-selection protocol are now
prepared; selected-base association and the production GPU runtime still need
qualification.
The selected A prefixes at k=1/2/4 and B prefixes at k=1/2 contain no
ranking-eligible group. Those budgets provide no direct ranking supervision;
an absence of ranking improvement there cannot establish a head-capacity limit.

Within-unit probes reserve a fixed support half and query half of each unit's
groups. Query groups do not change with k. A's indivisible unit is one group,
so disjoint within-unit support/query groups are impossible under its existing
definition. B's available cohort shrinks sharply with k:

| Support groups k | Rank-evaluable validation units | Rank-evaluable test units |
|---|---:|---:|
| 1 | 515 | 1,023 |
| 2 | 220 | 421 |
| 4 | 61 | 126 |
| 8 | 11 | 26 |
| 16 | 1 | 5 |
| 64 | 0 | 0 |

Here rank-evaluable means at least one query group passes the fixed headline
eligibility rule. At k=16, validation supplies only one eligible query group;
at k=64, no validation unit has enough support, and the one sufficiently large
test unit has no eligible query group. That budget cannot select/evaluate a
ranking adaptation model. Future results must distinguish changing unit cohorts
from adaptation gains and compare k=0 on each identical query cohort.

No model was fit, no prediction was read, and no scientific experiment is marked
complete by this preparation. Base-model selection already used the original
validation units, so these are conditional probes on the existing benchmark.
Subset-local BCE labels are reconstructed for each support/query view. All source hashes,
51,720 held-out group memberships, 30,697 within-unit records and every budget
count were independently verified. Source:
`source reference` on Atum.

Compact token/label-view preparation was integrated from
`source reference`, commit `98bb8b5`, and deployed by Git pull.
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
`source reference`, its A/B `views.json` manifests, and
`source reference` on Atum.

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
`source reference`, its per-scope
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
selection has 1,956 eligible B validation groups; within-unit k=1/2/4/8/16 has
853/500/203/42/1 eligible validation query groups, spread over the previously
reported 515/220/61/11/1 rank-evaluable units. The small k=16 selection cohort is
a material limitation. Twenty-five local protocol/worker tests passed.
No scientific fitting or inference occurred. Tier-3 completion, verified base
checkpoint association and production GPU qualification remain required; E49–E52
remain pending. Evidence: `source reference`
and `source reference` on Atum.

Query scoring is now implemented separately in `adaptation_query.py`. It binds
each saved adapter to its exact base, frozen support task and fit receipt;
validates reserved query identities and float32 labels; and exports native
ranking, BCE and cosine scores. Undefined and ineligible groups retain their
unit identities and coverage. k=0 creates no optimizer or selection candidate.
The base parameters, module modes and random state are restored after every
call, including evaluation failure, to prevent adaptation carryover.

All 58 focused local CPU checks passed (15 new query checks plus the existing
fit, protocol, view and fixed-evaluation checks). These include actual tiny-model
fits in both scopes, native query inference, exact comparison with independently
restored weights, within-unit query isolation, changed-export rejection and
failure restoration. No scientific adaptation fit, query inference, epoch
selection or GPU qualification occurred. E49–E52 remain pending. The query
worker source SHA256 is recorded in `protocol.json`; production orchestration,
base-checkpoint verification and the Tier-3 gate remain outstanding.

The standalone CPU qualification from commit `81ad6b6` subsequently passed on
Atum. Eight tiny two-epoch support fits covered both scopes, A/B across-unit
queries, and B validation/test within-unit queries. There were 24 native query
exports including k=0 comparisons. Independent reconstruction checked 432
group/pathway records, 184 artifact hashes and 26 source hashes. The dedicated
worker exited successfully; no scientific data, selected checkpoint or GPU was
used. Two additional local checks cover the complete qualification and refusal
when GPUs are visible; artifact corruption is rejected. Evidence on Atum:
`source reference` and
`source reference`. Runtime receipt SHA256:
`c1c61c9ce741713d84b5d38b06c3927cf95449bf582d92b011530f5914b014e3`.
This qualifies the tiny CPU runtime only; E49–E52 and production GPU execution
remain pending.

The statistical aggregation layer is prepared in `adaptation_statistics.py`.
It validates complete group/pathway inventories and unchanged query membership
within shared units, then reports available and common-unit cohorts across k.
A separate sensitivity restricts to groups defined for the base and every k,
without removing units from the bootstrap population. Paired changes reuse the
existing whole-unit bootstrap; three-seed spread remains descriptive and keeps
seed42 tuning exposure distinct. All 76 focused local checks passed, including
25 new aggregation cases and native float32 query-schema compatibility. The
first compatibility fixture used an invalid group-ID format; correcting that
fixture resolved the failure. No scientific data or fitting was used.
Artifact/model/epoch provenance, support-exposure accounting and scientific
orchestration remain outside this statistics library. It adds no experimental
completion or adaptation result.

## Representation geometry preparation — inference pending

E54/E55 have fixed, label-free sampling cohorts for each A/B train, validation
and test partition. Every partition has 25,600 observation-weighted rows and
25,600 unique-candidate representatives, plus all its distinct queries:

| Partition | A distinct queries | B distinct queries |
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
`source reference` on Atum.

Nineteen CPU tests qualify singular-value effective rank, normalized alignment
and uniformity, zero-vector coverage, capture of the actual model's pooled
contrastive projection boundaries without changing weights, and identity-based
export alignment despite reordered caches. Wrong labels/identities, changed
artifacts and the wrong cosine pathway are rejected. The rank definition
is uncentered singular-value entropy, as in
RankMe; the existing older diagnostic's centered
covariance entropy is a different statistic. The
alignment/uniformity definitions use normalized
vectors. Cross-modal alignment applies only in the trained common projection
space; unrelated encoder coordinates are not made comparable by padding.

The production mapping passed separately on all six token caches: 311,343
distinct observations supply the 331,191 view memberships. All selected native
float32 continuous labels and raw/cache observation identities match. The
current cache positions happen to equal raw positions; the worker does not
depend on that coincidence. This CPU check verifies source data, cohort files
and cache registration metadata, but does not rehash the full token payloads
or load a model. Source: `source reference` on Atum.

These are membership, mapping and numerical checks only. No scientific geometry value,
representation-collapse conclusion or improvement is established. Checkpoint,
token and row provenance, permitted-GPU inference qualification and the actual
measurements remain pending behind Tier3. E54/E55 remain **pending**.

## Gradient and cold-validation probe preparation — scientific results pending

E56/E57 now have a numerical worker and independently verified cold-validation
memberships. Each model uses its own split's validation units with no exact
training-similarity_class match. A retains 6,012 rows across 598 whole groups, including
190 ranking-eligible groups (3.07% of A validation rows); B retains all 136,957
validation rows across 9,280 groups / 4,032 components, including 3,969 eligible
groups. Membership selection uses identities and similarity_classs; labels only describe
feasibility. A's selected cold cohort does not represent the full A validation
distribution or generator deployment.

An exact identity check finds 107,929 B-validation observations (78.81%) in A
training. These are different split realizations; this is not leakage within
either model's own train/validation split. It does mean that global B validation
cannot serve as a held-out probe for the A model. Independent raw similarity_class-set,
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
results, checkpoint/token
provenance, and numerical qualification on the permitted GPUs. Sources:
`source reference` and
`source reference` on Atum; copied evidence
and current local source hashes match.

Fixed diagnostic batches are now prepared by `objective_probe_batches.py`
(source `cd4ac87`). Each regime has32 training/cold-validation batch pairs,
128 batches in total. The training contexts follow the native epoch-zero,
seed42,12-item sampler. Cold-validation contexts apply the native group-list
builder inside the previously frozen cold cohort, with12-item local batches.
This gives differentiable R/B/C objectives for the probe; it is distinct from
the ordinary32-row validation metric pass. Contexts are selected at evenly
spaced batch indices before observing any gradients or scores and shared across
later checkpoint/seed comparisons. They do not change the training runs.

| Context | Source batches | Selected batches | Selected rows | Selected units |
| --- | ---: | ---: | ---: | ---: |
| A training | 12,820 | 32 | 3,830 | 1,219 groups |
| A cold validation | 57 | 32 | 3,386 | 371 groups |
| B training | 14,020 | 32 | 3,660 | 16 components |
| B cold validation | 978 | 32 | 4,401 | 676 components |

Every selected batch contains at least one flagged ranking list and one strict
active contrastive row; this is feasibility, not evidence of a nonzero gradient.
The diagnostic batches are not independent split units or full-dataset metrics.
The small B training-unit count reflects its concentrated training support.
Source/cache identities and float32 labels, whole-unit cold-cohort membership,
list boundaries and original cache indices were checked during preparation.
All28 focused local tests passed, including exact native feature/collator parity,
reordered caches, and retention of zero-signal contexts. The Atum worker exited0;
28 input files,8 source files and8 output artifacts were rehashed. No token
payloads, test data or models were consumed by preparation. No gradient cosine
or transfer value was produced, and no experiment/question completion is added.

Evidence: `source reference`, SHA256
`acc44c0028ef8fffb2c00917f506b17fb279e3e353055e93c996244d5f981c50`;
runtime receipt `source reference`,
SHA256 `1bed336b7a4272e01c85e5e77c0f76b732ce928f084841fdd8783f3b926e49df`.
Complete token-payload verification and GPU numerical qualification remain
required before scientific probes.

Production reconstruction has also passed at source `70c3252`. On Atum, all128
saved batches reproduced native dataset items exactly, including cached tokens,
labels, identities, list boundaries and original cache row indices. Native
collation also matched exactly across2,304 tensor/metadata fields, including
tensor shapes, dtypes and values. The cached tokenizer padding IDs are1 for
query encoder E and3 for candidate encoder S. All30 focused reconstruction tests passed locally.
The production verifier read the selected token payloads and local tokenizer
metadata, but no model weights or test data; it measured no loss or gradient.
This qualifies reconstruction, not full token-cache integrity or scientific
probe outcomes. The worker exited0 and was confirmed terminal.

Evidence: `source reference`,
SHA256 `3e2d23f5b831e73ec592419ea32f8f2a60e2a681d8d37187929fe97f01516446`;
runtime receipt `source reference`,
SHA256 `6cb1861b87247a758ea809afc0e2070312dde141a6f220c107e9a7a2b033d291`.
Tier3 and selected checkpoints, full token-payload hashes and
GPU numerical qualification still precede the actual E56/E57 measurements.

Checkpoint and step rules are now frozen by `objective_probe_protocol.py`
(source `635f323`). The six references are the full R+B+C A/B models for
seeds42/43/44 at the common B-validation-selected LR, each at its own best
validation-macro checkpoint. This matches the head-adaptation reference policy.
The manifest records unresolved checkpoint slots; it makes no LR or checkpoint
selection and adds no model fits. Each reference uses the same32 saved
training/cold-validation pairs for its regime, giving192 diagnostic pairs.

Each source objective takes one nominal weighted-SGD step for each existing
scope (encoders, interaction, shared), holding private heads fixed. Rates follow
the actual Trainer's side-specific/common/default fallback rules. At multiplier1,
both encoder rates are1e-5 and the interaction rate is1e-4; the existing0.3/3
candidates scale these rates. There is no additional step-size grid. Unequal
projection/fusion rates are rejected because the numerical worker has one
interaction rate. The protocol specifies float32 parameters and forwards,
disabled autocast/TF32/dropout, and reports actual versus planned parameter
steps and explicit undefined or unchanged-loss cases. GPU numerical adequacy
is still unverified. These are conditional local diagnostics, not AdamW replay
or evidence of improved generalization. Batch distributions are reported per
model seed; overlapping batches do not supply independent-unit intervals.

All23 focused protocol/worker tests passed locally. Atum CPU preparation exited0
and rechecked38 input files and6 protocol source files, alongside the existing
batch reconstruction provenance. No model, loss, gradient, test prediction or
new fit was produced. E56/E57/Q33 and the study completion counts are unchanged.
Evidence: `source reference`, plan SHA256
`ecf6496e49ff5f1882f6405b6b54777b6bf6fc17e9ad9bf20412624d7c97c275`;
runtime `source reference`,
completion receipt SHA256
`958fc8f3d84dc19ce90ff3d7344df241a51ea06740aad30a14f68c612cbafe24`.

The batch execution/reporting worker is implemented at `803f83a` in
`objective_probe_execution.py`. It checks each reconstructed batch against the
native tensor/metadata fingerprints, then uses the Trainer's input adapter to
remove export bookkeeping before calling the model. The CPU integration caught
and fixed that adapter requirement before production use. Every pair retains
its block-gradient statistics, transfer values and context hashes; a saved summary is
recomputed from those records and rejects changed model/rate/context evidence.
Undefined cosines, disconnected paths and ineffective steps remain explicit.
The worker preserves parameters, RNG and module modes, restores global precision
settings even on failure, and refuses to overwrite partial output.

All34 focused worker/protocol/numerical checks passed. The11 new worker checks
were rerun after adding source-hash verification and also passed. A standalone
local CPU qualification loaded an actual saved tiny checkpoint and executed two
synthetic native training/validation batch pairs, including all three objectives
and scopes. Its result was reloaded and its descriptive distributions recomputed;
all seven worker source hashes and both saved pair hashes match. These are
synthetic implementation checks, not production measurements or evidence of
gradient conflict in any trained reference model.

Local evidence: `source reference`,
SHA256 `e13468a5eff473585384bd33522c783e59c06cb490aa7169b2de454a7616c564`;
`source reference`, SHA256
`5bfad9b561c8443e8e4619d4ab336367a4af79167bccb8c78e1427453c23e1cf`.
Atum access is restored, and the worker was pulled through Git after checking
all205 existing pinned inputs and the live coordinators. Its standalone CPU
qualification also passed on Atum: an actual saved tiny checkpoint, two native
synthetic batch pairs, all three objectives/scopes, unchanged parameters/RNG,
and reloaded/recomputed distributions. The worker exited0 and was verified
absent; both pair artifacts and all seven worker sources were rehashed.
Evidence: `source reference`
`{cpu_complete,cpu_exit}.json` and `source reference`;
qualification SHA256
`c70461f93de67e9aa44e93091e5861e9ccc123dee6c15863f8d46253c963fc6c`,
runtime receipt SHA256
`196931da7e44ea232ab16dc26f47cdb356df92f1f3bac2a968d64bf56cbbeeea`.
No training job was restarted and no scientific probe was run. E56/E57/Q33
remain pending; the worker provides no scientific launcher or waiver of Tier3,
checkpoint/token provenance, GPU numerical qualification or the shared GPU lock.

## A-versus-B variant reporting preparation — scientific results pending

`variant_relation.py` (source `a1aeafb`) prepares the primary-grid component of
E58/Q17. It requires all 28 canonical primary-arm summaries, covering the 280
already registered fits; it creates no fit or model selection. Each setting
retains its declared readout and B-validation-selected LR. The report keeps all
five checkpoint rules and nine ranking/top-weighted metrics, with seed 42,
seeds 43/44 and the two-confirmation mean separate. The three common-LR seed-42
views are separate sensitivity analyses, not pooled additional observations.

Original score coverage remains visible. A second macro-Spearman view restricts
each regime/seed/rule/partition to eligible groups with defined predictions for
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
7,997 and 3,969 common eligible groups respectively; these are reader checks on
existing reference epochs, not variant-association results. All nine live
process identities and 138 pinned inputs were preserved during deployment.

The MD requires every Tier-3 variant. E40 leak-in/out and E41 learning-curve
comparisons remain additional unfinished inputs to the eventual combined plot;
the primary component explicitly cannot mark E58 complete. Q17/E58 remain
**pending**, with zero scientific variant plots or associations. Evidence:
`source reference`
on Atum. Preflight SHA256:
`e6a02f07cc54fa4db50bbf9036890a71bf9e08a3e1ad4d9ff3bc692753d00e29`;
reader qualification SHA256:
`fec23f4ad4f2c1a32a0b3a280fd0c8d09d1e699a083249c8b87ce448d47adafb`.

Deployment is unseen generated candidates. That does not establish whether their
similarity_classs are unseen. The report must distinguish unseen identity from unseen
similarity_class, rather than presume either split alone matches the generator.


## Two-seed evaluation activation and E33 replacement

The amended full-model evaluator was deployed as `650d76b`. Its runtime
preflight preserved 538 prior source/evidence pins, and the CPU coordinator
was launched once (PID3909811/start169257950). At Unix1793741881.716816 it was
running fixed-checkpoint evaluation for `full__A__seed42__lr1p0`; live native
workers were on authorized physical GPUs0,2,5. Zero per-fit evaluation/analysis
completion receipts existed at that observation. This is execution progress,
not a new scientific result. Evidence: `source reference`
`source reference`; activation SHA256
`724cb1a3ebdea11cd01e215dc4978926a39b6695966ec3ccbe3210922cb00ee8`.

`two_seed_regularization_queue.py` replaces the cancelled E33 coordinator.
It uses only the two already registered alternative decays,0.001 and0.1,
with six seed42 tuning fits plus A/B seed43 confirmation fits for each setting:
16 new fits,24 including the eight shared full-reference fits at decay0.01.
Original configs, LR candidates and the B-validation-only selection rule remain
bound to the historical plan. It waits for the exact full-evaluation process
to finish and verifies all eight completed analyses before training. Existing
outputs, any unexpected seed44 registration, changed evidence, resource/lock
conflicts and failed work cannot silently trigger duplicate fitting.

The companion `two_seed_regularization_analysis.py` retains every registered
decay, all five checkpoint rules, conditional paired whole-unit comparisons,
and the single validation-selected decay/LR decision. Seed43 is the only
confirmation; confirmation-seed variance is unavailable. The family report
runs after both new primary reports. No extra decay or training arm is added.
All 56 local focused checks passed. They exercise orchestration and real
configuration/selection validation with simulated workers, plus paired report
IO; they do not establish
scientific GPU performance. Runtime preparation and activation are recorded
separately under `source reference` and
`source reference` on Atum. E33 and all scientific
completion counts remain unchanged until the prescribed results are verified.


# Verified stop and variant observations

Historical snapshot text follows. Dates, prefix lengths, provisional status and earlier scope must be read as historical. Current answers and stop status are in answers.md. The native record and full-precision tables take precedence over rounded prose. These paragraphs preserve earlier measurements and qualifications; they do not authorize further work.

# Paired-input ranking generalization: stopped-study report

Snapshot: 2026-11-09. **Experiments stopped at the user's request; the study is not complete.** This report follows the
34 questions and experiment specification.
The evidence ledger records every question and all 70 experiment
entries, including 12 conditional interventions. Artifact locations, hashes and
qualification limits are in detailed findings.

**Verified stop:** Atum was reachable on November 9. Two process inventories
found no project-owned training or controller processes; no termination signal
was needed. Training had stopped on the group-balanced A evaluation failure,
and the reporting/probe controllers had stopped without completing any of their
queued scientific jobs. The persistent goal is paused. No experiment was
launched, resumed or rerun for this report. Existing logs, checkpoints and
prediction files were inspected. The server remains at `003cdca`; the evaluation
fix is pushed but was not deployed. The server stop receipt is
`source reference`,
SHA256 `1a61aaaa6b5353956903e178281b985ab6e1c536f34715c154429ba481a213cc`.

**Completed training inventory:** 23 full 15-epoch neural fits: the initial
fresh similarity_class baseline, eight full-model reference/LR/seed fits, and fourteen
requested fixed-seed variant fits. This excludes small runtime checks and the
non-neural baseline fits. Of the requested 24 variants, **14 completed, one
failed, and nine never started**. The earlier seed44 A attempt was cancelled
and excluded. A completed fit does not mean its selected-test comparison is
complete. The initial fresh B baseline has a selected test result; the narrowed
26-model selected-test comparison is unfinished.

**Consolidated fixed-seed results:** Best native validation macro Spearman,
with each run's selected epoch in parentheses. A is the random-group split;
B is the similarity_class-component split. A and B evaluate different populations.

| Setting | A | B |
| --- | ---: | ---: |
| Full reference | 0.34368 (14) | **0.18458 (6)** |
| Frozen encoders | 0.34118 (15) | 0.14643 (6) |
| Last two encoder blocks | 0.34153 (15) | 0.16670 (8) |
| LoRA rank 4 | 0.31065 (15) | 0.17134 (11) |
| LoRA rank 16 | 0.31663 (15) | 0.17703 (11) |
| LoRA rank 64 | 0.32331 (15) | 0.16879 (8) |
| Detach contrastive encoder gradients | **0.36443 (14)** | 0.17571 (7) |
| Normalize listwise ranking per group | 0.33629 (15) | 0.17121 (5) |

No completed requested variant beats full B. Detachment improves A but reduces
B. This is observed single-seed validation evidence, not a completed test-set
comparison or proof that no other intervention could help. All three objectives
remain present, with nominal ranking/BCE/contrastive weights 0.25/0.25/1;
detachment changes gradient flow and normalization changes ranking-loss scale.

**Data and metric work completed:** entity_37 was downloaded and checked, and
all 1,957,400 curated observations were reconstructed from 2,032,748 retained
measurements with zero count/median mismatches. similarity_class-component splitting,
tokenization, partition-local labels, supervision eligibility, query support,
replicate disagreement, contrastive masks, frequency skew, metric precision,
loss/readout trajectories, composition and distribution shift were audited.
The original random-split pair-median procedure used held-out measurements to
change 5,289 training labels; controlled fits use partition-local labels.
Original historical split identity could not be recovered exactly, so this is
not a byte-identical reconstruction of the original `m5guoqkx` dataset.
The 34-question table below states the findings and their limits individually.

**Completed inexpensive baselines:** Test macro Spearman as recorded in the
baseline audit; lookup/identity methods have restricted or undefined coverage.
The query-only BCE baseline is reported separately because it gives constant
within-group scores (test AUROC A/B: 0.6696/0.6493).

| Baseline | A test | B test |
| --- | ---: | ---: |
| candidate-only ridge | 0.0775 | 0.0624 |
| Same-query nearest-similarity_class lookup | 0.2572 | Undefined |
| Independent per-query ridge | 0.2419 | 0.0726 |
| Two-step ridge | 0.1034 | 0.0488 |
| Identity matrix filter | 0.4207 | Undefined |
| Paired Kronecker ridge | 0.1690 | 0.0631 |
| query-shuffled Kronecker ridge | 0.0662 | 0.0596 |
| Bilinear inductive completion | 0.2126 | 0.0779 |
| User-fixed depth-3 LambdaMART | 0.1285 | 0.0607 |

Undefined B scores mean those methods cannot order its within-group candidates;
they are not zero correlations. Lookup and identity-filter A coverage is
79.72% and 93.32%, respectively. The jointly trained initial-B model's cosine
readout scores 0.06623 on test versus 0.17688 for its ranking head, on the same
3,972 eligible groups. This is a readout comparison, not contrastive-only training.
All eleven Tier-2 ledger entries are complete; full details, coverage and
conditional intervals are in the linked baseline ladder in `INTERIM_FINDINGS.md`.

**Work prepared but not completed:** The selected-checkpoint reporting,
adaptation, geometry, gradient-cosine, lookahead-transfer and A/B variant-analysis
code and fixed protocols exist, but the requested scientific results were not
produced. The separate 10,000-pair candidate encoder S/candidate encoder M/query encoder T embedding comparison
has no completed result in this study. Earlier HPO, further seeds, loss-subset
searches and additional LambdaMART runs were excluded by user amendments.
The ten unfinished requested fits and remaining analyses are stopped, not queued
for continuation. Current verification is recorded in
the stop evidence.

**Historical ledger:** 23 experiments complete, 12 partial and 35 pending across
the original 70 entries. These counts include experiments the user subsequently
excluded; they are not the remaining task list. The current assessment in this
report is: **12 answered, 16 answered with limits, 3 partial, 2 pending requested
results (Q17, Q33), and Q30 excluded by the current scope.** Q09 now has a
descriptive answer from the completed frozen A/B trajectories; the historical
ledger and tier counts below retain their earlier snapshot.
“With limits” means an observed or descriptive answer is available, without
claiming independent replication. Only Tier 2 is fully complete.

| Scope | Complete | Partial | Pending |
| --- | ---: | ---: | ---: |
| Tier 1: metric and dataset audits | 12 | 9 | 0 |
| Tier 2: inexpensive baselines | 11 | 0 | 0 |
| Tier 3: controlled training | 0 | 2 | 14 |
| Tier 4: mechanism and representation analysis | 0 | 1 | 9 |
| Conditional interventions | 0 | 0 | 12 |

**Training results:** The requested fresh 15-epoch similarity_class baseline is complete.
Its best validation-macro checkpoint is epoch 6: validation 0.18460 and test 0.17688
(run 4d1vkheo).
All six full-model seed42 tuning fits are complete, including their native trajectory audits:

| Split / learning rate | Best epoch | Best validation macro | Final validation macro |
| --- | ---: | ---: | ---: |
| A / base | 14 | 0.34368 | 0.34332 |
| B / base | 6 | 0.18458 | 0.17975 |
| A / ×0.3 | 14 | 0.28036 | 0.27928 |
| B / ×0.3 | 12 | 0.18322 | 0.18052 |
| A / ×3 | 15 | 0.37532 | 0.37532 |
| B / ×3 | 6 | 0.17310 | 0.16401 |

The A LR0.3 fit scores below base LR at every matched epoch. B LR0.3 peaks later
than base LR, with a slightly lower best macro and slightly higher final macro.
A LR3.0 has the highest observed A peak, 0.03164 above the base-LR peak.
B LR3.0 has a lower peak and final macro than either other B learning rate.
These are descriptive single-seed validation results, not a selected controlled
test comparison. The registered B-validation rule selected **base LR** for both
A and B: B peaks are 0.18458 (base), 0.18322 (×0.3) and 0.17310 (×3).
No test scores entered this selection. **Both required seed43 confirmation fits are complete:**

| Split / seed at base LR | Best epoch | Best validation macro | Final validation macro |
| --- | ---: | ---: | ---: |
| A / 43 | 15 | 0.34494 | 0.34494 |
| B / 43 | 5 | 0.17739 | 0.16830 |

A seed43's peak is close to seed42's 0.34368; B seed43's peak is below
seed42's 0.18458. A/B use different validation populations, so their gap remains
descriptive. Both fits' 15 saved native validation metrics match Trainer history
and their retained best checkpoints are verified. Independent trajectory
analysis and selected-test comparisons remain pending. The user amended the
protocol to **two training seeds total, 42 and 43**, overriding the MD's minimum
of three. A seed44 was cancelled while partial; B seed44 never started. Seed44
is excluded from selection and scientific comparisons. Seed42 is tuning-exposed;
seed43 is the single confirmation, so confirmation-seed variance is unavailable.
That two-seed policy is now superseded. The user stopped the checkpoint and
E33 queues, cancelled parameter optimization, and requested **seed42 only**
for E34–E36, E44–E47, E49–E58, E01–E05, E08–E09 and E19–E20.
Seed42 has the higher existing B base-LR validation peak (0.18458 versus
0.17739). The new fixed-configuration queue contains 24 fits (12 requested
variants on A/B), reusing full A/B42. Evaluation and Tier 4 work remain pending.
**14/24 requested variant fits are verified complete.** Frozen-encoder A42 completed all
15 epochs and selected epoch15, validation macro **0.34118**
(run khvzu281).
It was preserved without retraining. Frozen B42 also completed all 15 epochs:
best validation macro **0.14643 at epoch6**, final **0.13679**
(run h2nhxuqv).
Last-two-blocks A42 completed 15 epochs with best/final macro **0.34153 at
epoch15** (run 7cjkhhsw).
Last-two-blocks B42 completed 15 epochs with best macro **0.16670 at epoch8**
and final **0.15388** (run zba1gr0v).
LoRA rank4 A42 completed all 15 epochs with best/final macro **0.31065 at
epoch15**
(run 8tbg64mm).
LoRA rank4 B42 completed 15 epochs with best macro **0.17134 at epoch11**
and final **0.17124**
(run igxpwkjs).
LoRA rank16 A42 completed all 15 epochs with best/final macro **0.31663 at
epoch15**
(run sqj6wwrw).
LoRA rank16 B42 completed all 15 epochs with best macro **0.17703 at epoch11**
and final **0.16467**
(run nio1y8kl).
LoRA rank64 A42 completed all 15 epochs with best/final macro **0.32331 at
epoch15**
(run 9wbd28xl).
LoRA rank64 B42 completed all 15 epochs with best macro **0.16879 at epoch8**
and final **0.16353**
(run 8r1q7fsj).
Contrastive-detachment A42 completed all 15 epochs with best macro **0.36443
at epoch14** and final **0.36438**
(run ivwxt7ns).
Contrastive-detachment B42 completed all 15 epochs with best macro **0.17571
at epoch7** and final **0.16708**
(run fhl9hwyh).
Per-group listwise normalization A42 completed all 15 epochs with best/final
macro **0.33629 at epoch15**
(run 35lhbwgg).
All fourteen retained best-macro checkpoints are verified. Per-group listwise
normalization B42 completed 15 epochs on GPUs 0, 2, 5 and 6
(run djxcgvr9).
**November 9 verification:** Normalized B's native 15-epoch completion,
best/final checkpoint bytes and 31 prediction manifests are verified on Atum.
Its exact W&B history peaks at **0.17121 at epoch5** and ends at **0.16224 at
epoch15**; the subsequent best-score reevaluation is excluded from that final
epoch value. Both are below full B. The E45 section below records the evidence.
The next fit, group-balanced A
(run zzskjup5),
failed at its first evaluation: prediction batches passed ranking labels into
the model without the sampling weights required by group-balanced training.
Its W&B `output.log` records `ValueError: group-balanced training requires
source-group weights and batch item count` (SHA256
`645bdc088b1b07539d04ef6f4b19c76f1a1618c0519d2ba9253199b5098322b5`).
The failure was reproduced in a local CPU training/evaluation test. A prepared
fix makes evaluation forwards prediction-only; the existing complete-cohort
evaluation still computes all losses and group weights. All 99 targeted CPU
tests pass, including checkpoint saving, both validation and fixed-probe
exports, unequal-group loss reduction and exact prediction/embedding parity.
These tests use small encoders and do not constitute Atum GPU validation.
The tested fix was pushed as `fe5fb5d` on `source reference`.
It has not been deployed to Atum, and recovery has not been launched.
Atum access is restored. The failed attempt has no saved checkpoint directory;
the user stopped the experiments before any recovery or restart.
The selected-checkpoint evaluation and comparison reports are now implemented
for this fixed scope; scientific selected-test results are not yet available.
Preparation and local scheduling tests add no scientific completion credit.
The complete historical ledger is retained; unrequested experiments are not
silently marked complete. See the current scope.
The recovery dispatcher and reporting/probe queues were verified active on
November 4. A controller-exit observation race was repaired, and completed A
was adopted; no training process was signaled or refitted. The queue excludes
legacy extra split-allocation inference and model averaging. This scheduling
correction adds no scientific completion credit.

**Interim E34–E36 comparison:** Within each split, the frozen setting changes
only both encoder-freeze flags and the output directory from the full42
reference; the last-two-blocks setting changes only both unfrozen-layer counts
and the output directory. LoRA rank4 A/B change only encoder LoRA rank/alpha
(4/8) and output directory; rank16 A/B use rank/alpha 16/32 and rank64 A/B use
64/128. Across all 15 epochs,
validation and fixed training-probe
fingerprints, ordered row identities and labels match exactly: A has
195,618/39,740 validation/probe rows; B has 136,957/43,115.
A and B remain different populations.

| Split / setting | Best validation macro (epoch) | Final validation macro | Final probe macro |
| --- | ---: | ---: | ---: |
| A / full encoders | 0.34368 (14) | 0.34332 | 0.49208 |
| A / frozen encoders | 0.34118 (15) | 0.34118 | 0.55463 |
| A / last two blocks | 0.34153 (15) | 0.34153 | 0.50567 |
| A / LoRA rank4 | 0.31065 (15) | 0.31065 | 0.41728 |
| A / LoRA rank16 | 0.31663 (15) | 0.31663 | 0.43578 |
| A / LoRA rank64 | 0.32331 (15) | 0.32331 | 0.44983 |
| B / full encoders | 0.18458 (6) | 0.17975 | 0.51969 |
| B / frozen encoders | 0.14643 (6) | 0.13679 | 0.58219 |
| B / last two blocks | 0.16670 (8) | 0.15388 | 0.52087 |
| B / LoRA rank4 | 0.17134 (11) | 0.17124 | 0.44906 |
| B / LoRA rank16 | 0.17703 (11) | 0.16467 | 0.45937 |
| B / LoRA rank64 | 0.16879 (8) | 0.16353 | 0.46939 |

Frozen validation macro is below its full reference at every matched epoch in
both splits; best-score changes are **A −0.00249, B −0.03815**. Final
probe-minus-validation gaps are larger with freezing: A 0.21345 versus 0.14876;
B 0.44540 versus 0.33994. Freezing did not improve these validation results.

Unfreezing the last two blocks raises the best macro over the frozen setting
by **A +0.00035, B +0.02027**, while remaining below the full reference by
**A −0.00214, B −0.01788**. The B peak occurs at epoch8, later than full/frozen
epoch6, then falls by 0.01282 by epoch15. Its final probe macro is 0.52087 versus
full B's 0.51969, while validation is lower (0.15388 versus 0.17975).
The B ranking-loss minimum remains early, at epoch2 (2.81098), ending at
2.98448. A's ranking minimum is epoch14. These descriptive results do not
establish statistical equivalence on A or a selected-test benefit on B.

LoRA rank4's best validation macro remains below full by **A −0.03303,
B −0.01323**. A improves at every epoch through 15. B peaks at epoch11 and
finishes only 0.00010 below that peak; its best score exceeds frozen B by
0.02492 and last-two-blocks B by 0.00465. Both final validation and training-probe
scores remain below their full references. The smaller final probe-minus-validation
gaps (A 0.10663 versus 0.14876; B 0.27782 versus 0.33994) do not establish
better generalization. B's ranking-loss minimum is epoch4 (2.80742), ending at
2.85655; its trajectory is not monotonic.

LoRA rank16 improves best validation macro over rank4 by **A +0.00598,
B +0.00568**, while remaining below full by **A −0.02704, B −0.00755**.
A's validation macro rises and ranking loss falls at every epoch through 15;
its final probe macro is 0.43578 and probe-minus-validation gap 0.11915.
B peaks at epoch11, then finishes 0.01236 below that peak and 0.00657 below
rank4's final macro. Its final probe macro is 0.45937, with gap 0.29470.
B's ranking-loss minimum is epoch5 (2.80251), ending at 2.85885 with a
nonmonotonic trajectory; its total and contrastive losses reach minima at
epoch13.

LoRA rank64 A improves best validation macro over rank16 by **0.00667** and
over rank4 by **0.01266**, while remaining **0.02037 below full A**. Its
validation macro rises and ranking loss falls at every epoch through 15.
The final probe macro is 0.44983 and probe-minus-validation gap 0.12652.
All three A LoRA ranks peak at epoch15, with higher peaks at higher ranks in
these completed fits.

LoRA rank64 B's best macro is below rank16 by **0.00823**, rank4 by **0.00255**
and full B by **0.01578**. It peaks at epoch8, then finishes 0.00526 below that
peak. Its final probe macro is 0.46939 and probe-minus-validation gap 0.30585.
The ranking-loss minimum is epoch4 (2.80153), ending at 2.87119 with a
nonmonotonic trajectory; total and contrastive losses reach minima at epoch13.
Thus the A improvement with increasing LoRA rank is not reproduced on B:
rank16 has the highest B validation peak among these three prescribed ranks.
All E34–E36 variant fits are complete, but their selected-test comparisons
remain pending. These single-seed validation comparisons do not select a
LoRA rank or establish a test-set benefit.

**Q09:** B's early validation ranking-loss rise survives freezing: its minimum
is epoch2 (2.83134), then it rises at every epoch to 3.22761 at epoch15.
Full B's ranking minimum is epoch3 (2.79424), ending at 2.92986. In A, ranking
minima are frozen12/full15. Both B fits first show a BCE increase at 2→3 and
a contrastive increase at 6→7, but total and contrastive loss still reach their
overall minima at epoch15. Thus freezing does not remove the observed ranking
pattern; these observations do not isolate a causal mechanism.

These are native metrics with verified payload hashes, not independent metric
recalculations or statistical change-point tests. A validation has 7,997
defined eligible groups except frozen and last-two-blocks epoch3, LoRA rank4
epochs4/9/13 and LoRA rank64 epochs6/11 (7,996);
final and peak coverage agree. B validation has 3,969 defined eligible groups
throughout full, frozen, last-two-blocks, rank4 and rank64 fits. Rank16 B has 3,968 at
epoch7 and 3,966 at epoch12, otherwise 3,969, including its peak and final
epochs. B's probe has 2,047 defined eligible groups throughout all fits except
rank64 epochs2/5/9 (2,046); its peak and final coverage agree.
Selected-test comparisons
remain pending, so E34–E36 are not complete. Evidence under
`source reference` on Atum:

- `frozen_A_seed42_native_20261104.json`, SHA256 `0ab551c69e8d64245123eb5ab44346236c916608961d5f6b012c87bd403a5968`.
- `frozen_B_seed42_native_20261104.json`, SHA256 `9c4eb2b7dec6e88b7ac81297415fb3594bc76428c801f2925691b9584ed30003`.
- `last_two_blocks_A_seed42_native_20261104.json`, SHA256 `46dedb629f5a50b7c6f7f6174b6f72491294beb3f06e426092d56090a7b6aea0`.
- `last_two_blocks_B_seed42_native_20261104.json`, SHA256 `cf6ae9651b9400ea42fdb2b8b128b4106c225c04adf2226ef6910bf387bd11d7`.
- `lora4_A_seed42_native_20261104.json`, SHA256 `8906161cd405395fb2041979207a919b5960cbee714093e7519486d2f602b92c`.
- `lora4_B_seed42_native_20261104.json`, SHA256 `c34293c7c75b0d6c69a264856d737f6b1ca5e9d4b7087c01b964b55011c36138`.
- `lora16_A_seed42_native_20261105.json`, SHA256 `4ed3ac9a78b51b47101db8051607dac5cd54dc65ee7198f2b1421f4dc96ec7fe`.
- `lora16_B_seed42_native_20261105.json`, SHA256 `cc514b702fb5331f75a0d6fbaf911b9ab29284b51b481c94207b0ea1b994a5dd`.
- `lora64_A_seed42_native_20261105.json`, SHA256 `9671fc3773bef4583034c3b1d1c2d180deb110b1a2a2680e9b81985379038ade`.
- `lora64_B_seed42_native_20261105.json`, SHA256 `eece92ce0d719968bf3dc4a175bc06a9a8666e5094992df0ee005682c2ee14d0`.

These artifacts bind native trajectories, prediction payloads and configuration
hashes. No additional inference or fitting was performed for this comparison.

**Interim E44 comparison:** Detachment A/B change only
`contrastive_detach_encoders=True` and the output directory from their full42
references.
This blocks contrastive gradients into both encoders while retaining the
ranking, BCE and contrastive losses with their original weights. Within each
split, all 30 native validation/probe exports have matching ordered rows and
labels. A coverage is 7,997/2,048 defined eligible groups throughout. B has
3,969/2,047 except detachment validation at epoch3 (3,968); peak and final
coverage agree with full B.

| Split / setting | Best validation macro (epoch) | Final validation macro | Final probe macro |
| --- | ---: | ---: | ---: |
| A / full reference | 0.34368 (14) | 0.34332 | 0.49208 |
| A / detach contrastive encoder gradients | 0.36443 (14) | 0.36438 | 0.56094 |
| B / full reference | 0.18458 (6) | 0.17975 | 0.51969 |
| B / detach contrastive encoder gradients | 0.17571 (7) | 0.16708 | 0.58785 |

Detachment's validation macro exceeds full A at every matched epoch. The best
and final improvements are **+0.02076 and +0.02106**. Its final probe-minus-validation
gap is larger, 0.19656 versus 0.14876. Final contrastive loss is higher
(3.05338 versus 1.65268), as is ranking loss (2.43732 versus 2.41737), despite
the higher ranking macro. Detachment's ranking/BCE loss minima occur at epoch12;
total and contrastive minima occur at epoch15.

Detachment B's best and final validation macro are **0.00887 and 0.01268 below
full B**. Its peak occurs at epoch7, then falls by 0.00864 by epoch15. Validation
macro exceeds full B only at epochs1, 2 and 4, while the fixed training-probe
macro exceeds full B at all 15 epochs. Its final probe-minus-validation gap is
larger, 0.42077 versus 0.33994. Ranking loss reaches its minimum at epoch2
(2.80283), ending at 3.12983 versus full B's 2.92986. BCE also reaches its
minimum at epoch2; total and contrastive minima occur at epoch13. Final
contrastive loss is 7.26590 versus full B's 5.83346. Thus the validation gain
on A is not reproduced on B in these completed fits.

These are descriptive single-seed native metrics, not independent
recalculations or evidence that gradient conflict has been measured. Both E44
fits are complete, but selected-test comparisons and Q33's gradient analysis
remain pending. Evidence under `source reference` on Atum:

- `detach_contrastive_A_seed42_native_20261106.json`, SHA256 `fe35394e9c315d2437a24c241397bfde2842fe0dade880622cb7593365570801`.
- `detach_contrastive_B_seed42_native_20261106.json`, SHA256 `8642000c2eb7405c543c297e17412b8b3f87ae6ee65c32df557a51c96e2fe9c6`.

**Interim E45 normalization comparison:** Normalized A changes only
`ranking_normalize_per_group=True` and the output directory from full A42.
Each list's ranking loss is divided by the sum of its contributing positional
weights before averaging across lists. All three objectives and their nominal
weights remain unchanged, but this normalization changes the ranking-loss
scale; raw ranking and total loss values are not directly comparable across
these two settings. All 30 validation/probe exports have identical ordered
rows and labels (195,618/39,740 rows).

| A / setting | Best validation macro (epoch) | Final validation macro | Final probe macro |
| --- | ---: | ---: | ---: |
| Full reference | 0.34368 (14) | 0.34332 | 0.49208 |
| Per-group listwise normalization | 0.33629 (15) | 0.33629 | 0.48462 |

Normalized A's validation macro rises at every epoch, but both validation and
fixed training-probe macro remain below full A at all 15 matched epochs. Its
best and final validation changes are **−0.00738 and −0.00702**. The final
probe-minus-validation gap is 0.14833 versus full A's 0.14876. Within the
normalized trajectory, ranking loss has its minimum at epoch13, BCE at
epoch11, and total/contrastive losses at epoch15. Validation/probe coverage is
7,997/2,048 defined eligible groups except normalized validation at epoch14
(7,996); peak and final coverage agree with full A. These are descriptive
single-seed native metrics, not independent recalculations. Both group-balanced
fits and selected-test comparisons are unfinished, so E45 is not complete. Evidence:
`source reference`, SHA256
`69c0f74a799959565f5ae92cc84956379b84a949f131b4eed44df51d051d687b`.

**Normalized B, native verification completed November 9:** Both finished runs'
histories contain epochs1–15 at matching optimizer steps, followed by a
separate post-training reevaluation matching each run's best score. Normalized B
peaks at **0.17121 at epoch5** versus full B's **0.18458 at epoch6** (difference
**−0.01337**). Its actual epoch15 macro is **0.16224** versus **0.17975**
(difference **−0.01751**); it is below full B at all 15 logged epochs. The later
reevaluation at step13155 reports 0.17121 and must not be mistaken for the final
epoch's score. Native receipts confirm no test evaluation. All 30 matched
validation/probe export pairs have identical ordered identities and labels
(136,957/43,115 rows). Defined validation/probe group counts are 3,969/2,047,
except normalized validation epochs1/11 (3,968) and probe epoch2 (2,046).
Best/final coverage matches the reference. Final probe macro is 0.50577 versus
full B's 0.51969. Retained root/best weights have SHA256
`74010b0e999de17864ba235b1deeba2be04931163cbb34524411c7c8db045922`;
final weights have SHA256
`477e62b58f01b1a9a40ece76e8ff123ca359231799c0c12143d648109b827e8f`.
Local native-comparison evidence:
`source reference`,
SHA256 `f454c8c52cc4af411d2d66dd1230aa89af49aa98681ce9f5973a7a8a365b5a25`.
Earlier W&B evidence: `source reference`,
SHA256 `fa0281b5874942e1264c116fbec0fa6e1f4a3d75a8d1de0238947fd533677f04`;
normalized/full W&B history SHA256 respectively
`286bf71da91056ce2d08f3c5b5a531e4a08bf2daf28a79a2bfe6ce81cc8ad381` /
`f72bb6c24f1e435ae3a2ae14675f5d3684fe9f209b2f3357b43111f4ef7f6c22`.

**Reference labels below:** “Initial B” means the fresh similarity_class baseline above.
“Controlled A/B” means the separate base-LR seed42 reference fits. Their results
are kept distinct. Deployment is unseen generated candidates; unseen identity
does not imply an unseen similarity_class.

| Question | Status | Current answer and limit |
| --- | --- | --- |
| **Q01 — Which Spearman?** | Answered | Checkpoint selection uses the unweighted mean of eligible, defined per-group correlations. `eval_spearman` is instead observation-count weighted. Neither is a pooled correlation. |
| **Q02 — Eligibility, constants and ties?** | Answered | Headline eligibility is n≥3 and label span≥0.5, using float32 labels and average ranks. Undefined correlations are excluded with coverage reported. Eligible test groups: A 50.38%, B 41.47%. Float64 export arithmetic changes one B eligibility decision. |
| **Q03 — Different test composition?** | Answered | Yes. Median group sizes are A 3/B 2; mean within-group variance is 0.3046/0.1809. Replicate counts and tie distributions also differ. These differences do not establish their causal contribution to the performance gap. |
| **Q04 — Seed, bootstrap and split spread?** | Partial | Whole-unit intervals exist conditional on fitted models. Original splitter seeds 42/43/44 give identical assignments. Two dependent held-out reallocations preserve training support but have no selected-performance result. Additional seeds and allocation searches are outside the current requested scope. E20 will report conditional whole-unit uncertainty for the selected fixed-seed models; the A LR comparison does not measure seed variance. |
| **Q05 — Noise-tolerant and top-weighted metrics?** | With limits | Spearman has no noise band. Initial-B test ordering accuracy changes 0.5892→0.6581 when pairs inside the training replicate-RMS band are excluded, changing the denominator. Controlled-B NDCG@1/3/5 peaks at 6/13/8; all finish below their epoch 6 values. Different metric optima are descriptive, not proof of reproducible objective conflict. |
| **Q06 — Which loss rises first?** | With limits | Initial-B BCE has minimum 2, ranking minimum 5, and learned y→x contrastive minimum 5; the total contrastive minimum is 15. BCE can worsen while AUROC improves. A ranking and both contrastive directions decrease overall throughout 15 epochs. These single-run extrema are not statistical change points. |
| **Q07 — Loss or Spearman checkpoint?** | With limits | Controlled A/B select validation macro Spearman at 14/6. Minimum aggregate loss is 15 on both. The different checkpoint locations are established. Requested evaluation uses only best-macro checkpoints; comparing alternative selection rules and adding seed variation are outside the narrowed scope. |
| **Q08 — Training versus validation?** | With limits | Controlled B 6→15 probe macro improves 0.39862→0.51969 while validation changes 0.18458→0.17975. Final probe/validation values are A 0.49208/0.34332 and B 0.51969/0.17975. These are distinct fixed eligible-group probes, not full-training or matched causal gap estimates. |
| **Q09 — Frozen-encoder behavior?** | With limits | Yes, the early ranking-loss rise survives on B: frozen minimum at epoch2, increasing every epoch thereafter; full B minimum at epoch3. Frozen B best macro is 0.14643 versus full 0.18458, with a larger final probe/validation gap. A ranking minima are frozen12/full15. Native curves and matched rows/labels are verified; no causal mechanism, independent replication or selected-test benefit is established. |
| **Q10 — Temperature and fixed candidates?** | With limits | Learned scales sharpen on A and initial B; candidate pools stay fixed across epochs. Fixed-scale13 rescoring raises A's final contrastive loss but lowers initial B's. Neither fresh reference shows a sustained overall contrastive-loss rise. Rescoring does not prove fixed-temperature training helps. |
| **Q11 — Which sizes or similarity strata?** | With limits | In initial B 6→15, >25-item groups account for 80.2% of net ranking-loss increase, while three-item groups account for 76.1% of net macro decline. A cold-similarity_class strata differ from shared strata. Localization is descriptive and conditional on these fitted trajectories. |
| **Q12 — Deployment regime?** | Answered | Deployment is unseen generated candidates. Report unseen identity and unseen similarity_class separately; the generator's future similarity_class mixture is unknown. Neither split alone is established as an exact deployment match. |
| **Q13 — Similarity distribution and stratified quality?** | Partial | Binary similarity_class support is measured:19.29% of A test rows have unseen similarity_classs, versus all B rows. Depth3 A test rho is 0.1342 on shared-similarity_class rows and0.0669 on unseen rows, versus B 0.0607; eligibility is reapplied. Restricted rows and intact groups are different cohorts. Selected fixed-seed neural test comparisons remain unfinished; additional seed or allocation comparisons are excluded. |
| **Q14 — Indivisible units and assignment?** | Answered | A splits whole groups with target coverage anchors. B splits similarity_class-group components; its largest component supplies98.70% of training rows, and target-coverage rescue components are forced into training. Assignment is not exchangeable across train and held-out units. |
| **Q15 — Distribution shift?** | With limits | Raw-source descriptor C2ST balanced accuracy is about 0.85–0.86 for B train versus held-out, versus 0.52 for A. B's training positive rate is 0.6790 versus 0.7902 in test. BBSE estimates 0.7850, but its class-conditional invariance assumption is unverified. Split constraints limit population-level interpretation. |
| **Q16 — query support in B test?** | With limits | All1,723 B test queries occur in training, with1–1,303 training groups each. Initial-B test macro ranges0.0539 for one training group to 0.1935 for>100. Low-support intervals are wide and groups differ in composition; this is not a causal support effect. |
| **Q17 — Do A/B variant scores move together?** | Pending | The required variant performance grid is incomplete. No scientific variant-association result is available. |
| **Q18 — Replicate noise ceiling?** | With limits | Split-half test macros are A 0.5445/B 0.3780; the Spearman–Brown heuristics are 0.7050/0.5485. Coverage is only 5.07%/8.76% of eligible groups. These are not established ceilings for the entire test set. |
| **Q19 — Main effect, interaction and noise shares?** | With limits | The fitted variance decomposition is unstable under group balancing: the A interaction contribution changes 0→0.1159 and B 0.1716→0.0568 s². No stable interaction share or architectural ceiling is established. |
| **Q20 — Pairs within replicate noise?** | Answered | Training raw-replicate RMS bands are A 0.7806/B 0.7661 s. They contain60.80%/61.89% of test within-group label differences. Raw-replicate disagreement is not uncertainty of the aggregated median label. |
| **Q21 — Why only about half support ranking?** | Answered | Mainly n<3, followed by insufficient label span. About49% of training groups qualify; eligible/ineligible median sizes are 11/1, with different query composition. Small float64-sampler versus float32-metric eligibility discrepancies are documented. |
| **Q22 — How were observations selected?** | With limits | Curation retains exact numeric association measurements on both sides of the threshold. Retained groups' archived numeric measurements are already69.22% positive, with large source differences. Original testing frames and unpublished outcomes are unavailable; missing observations are not negatives or prevalence estimates for generated candidates. |
| **Q23 — Same-side labels and threshold noise?** | Answered |43.63%/54.20% of eligible A/B test groups have all binary labels on one side of 6. Also35.58%/32.21% of test rows lie within the raw-replicate RMS band around 6. BCE then lacks a within-group label-order preference; this does not prove it cannot help shared features. |
| **Q24 — Batch construction?** | Answered | Batches mix bounded group lists and classification-only rows. Eligible-list fractions average0.5583/0.5625 on A/B and vary substantially; each epoch-zero ordering has one local batch without a ranking-eligible list. The audit reconstructs the sampler, not historical device traces. |
| **Q25 — Contrastive false negatives?** | With limits | Own-group, same-query and duplicate-query exclusions already exist. Other training-known positives can remain negatives:0.530%/0.714% of audited x→y negative comparisons on A/B. This is a known-positive lower bound. The additional masking experiment is outside the narrowed scope, so its performance effect remains unknown. |
| **Q26 — Projections before or after fusion?** | Answered | Contrastive projections precede cross-attention and use separate pooled encoder outputs. Ranking and BCE use fused representations. Cross-pair contrastive scores therefore use embedding products. |
| **Q27 — Ranking-loss normalization?** | Answered | The reference sums weighted Plackett–Luce selections, then averages sampled lists without normalizing by retained weight sum. Larger groups get more bounded-list opportunities, so original groups are not equally weighted. Per-group normalization reduces best validation macro on A/B to 0.33629/0.17121 versus 0.34368/0.18458. The separate group-balanced training comparison is unfinished. |
| **Q28 — query-frequency skew?** | With limits | The top 1% of queries supply 30.07%/30.64% of A/B group-list opportunities. The reference has no log-Q correction. Skew is established; use of a learned popularity shortcut is not. |
| **Q29 — Whole-unit subsets?** | Partial | E41 views use whole groups on A and whole components on B. B's10% of units retains98.86% of rows, limiting saturation conclusions. E40 A views are qualified; strict B matching has no whole-component exchange. E40/E41 performance experiments and the proposed B exception are outside the narrowed active scope; no performance conclusion is claimed. |
| **Q30 — Loss-alone and pairwise combinations?** | Not run: user scope | No completed loss-subset comparison exists. The user excluded that grid from the active task. Full-model observations cannot identify each loss's contribution; the requested ranking variants and contrastive detachment address different comparisons. |
| **Q31 — Do scorer orders agree?** | With limits | Controlled-B ranking/BCE agreement falls0.67647→0.59235, while ranking/cosine agreement rises0.07402→0.23018. Final BCE/cosine label macros0.13944/0.07754 remain below ranking0.17975. Prediction agreement does not measure gradient conflict. |
| **Q32 — query-only BCE performance?** | With limits | Frozen-feature query-only models recover about 48.9%/54.6% of excess validation AUROC above chance for A/initial B. Their scores are constant within groups. This comparison of separate predictors is not a causal fraction of the full model's signal. |
| **Q33 — Auxiliary gradient alignment?** | Pending | No scientific gradient-alignment or transfer result yet. Fixed batches and CPU implementation qualification are prepared; actual trained-checkpoint probes remain gated behind the preceding experiments. |
| **Q34 — How were original weights selected?** | Answered | The user confirmed arbitrary selection of ranking/BCE/contrastive weights 0.25/0.25/1. W&B trials alone did not establish the decision. E39's prospective weight search is excluded from the active scope. |

**Unfinished at the user stop (not scheduled):** Ten of the 24 fixed-seed fits
are incomplete: group-balanced A failed, and group-balanced B plus eight pairwise
fits never started. Selected-model evaluations for the 24 fits and the two full
references, and the requested adaptation, geometry and gradient/transference
probes are unfinished. E01–E05, E08–E09, E19 and E20 have existing baseline/audit
evidence, but their extension to the selected variants and final updates to all
34 answers were not completed. E20 is whole-unit bootstrap
uncertainty, not an allocation search. The old E33, multi-seed and parameter-search
queues are stopped; no compute-budget or within-component exception is awaiting
approval. All experiment queues are stopped. Unrequested experiments remain incomplete in the
historical ledger and will not be run. No additional LambdaMART fits are planned;
the fixed depth 3 A/B test results0.12849/0.06071 remain the requested baseline evidence.

Original random-split pair-median labeling changed 5,289 training labels using
held-out measurements; the corresponding similarity_class count was zero. Controlled
runs use partition-local binary-label construction.
This correction, the constrained B split, unfinished selected-test comparisons
and the fixed-seed uncertainty limit must
remain visible in the final conclusions. Completing software or an audit
watcher does not complete a scientific experiment.
