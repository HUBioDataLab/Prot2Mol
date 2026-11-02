# Controlled training protocol

This document fixes operational choices before Tier 3 results are available.
The governing scope remains `Paired input ranking generalization.md` and every
item in `protocol.json`; this document does not replace or narrow that scope.

## Common comparison and tuning rules

- Deployment is an unseen generated molecule. Split B validation selects
  hyperparameters; Split A is a matched comparison, not the deployment selector.
- Use both partitions A and B, seeds 42, 43 and 44, and the same 15-epoch
  horizon, optimizer, schedule and effective batch as the fresh reference.
  The requested first reference remains unchanged and retains all 15 checkpoints.
- Each configuration arm receives three learning-rate candidates: multiply all
  its nonzero base learning rates by 0.3, 1 or 3. Train each candidate for the
  same full horizon at tuning seed 42 on both A and B. Select the candidate
  using maximum B validation macro Spearman, then run that same candidate on
  A and B at seeds 43 and 44. Keep seed-42 tuning and seed-43/44 confirmation
  results identifiable; seed-42 scores have hyperparameter-selection exposure.
- Ties select the lower multiplier. Nonfinite/undefined metric values are not
  winners. Report the defined fraction alongside the score; a loss-subset arm
  cannot win by silently dropping constant-prediction groups.
- Report A-versus-B comparisons for every matched candidate and for the common
  B-selected candidate, so independently tuned A/B settings do not create the
  apparent relationship. No test result controls training, selection or budget.
- For the weight sweep and multiple LoRA ranks, each declared setting is one
  arm and gets the same three-candidate budget. Report total family budgets as
  well; a family with several settings has more trials than a single arm.
- Use unit-level uncertainty and seed spread. Additional B realizations remain
  required subject to the specification's affordability condition; assess that
  with measured runtime and storage, not by silently omitting them.
- For A, binary labels use medians computed within each partition. The original
  pooled A transformation leaked held-out measurements into training labels.
  Continuous labels and the B reference transformation are unchanged.

The user ended the Tier 2 LambdaMART search and selected the already completed
depth-3 models on both splits. `user_decision.json` pins that choice and disables
further tree fitting; `finalize_lambdamart.py` computes fixed-model validation
and test results without training. The neural-training baseline requirement is
satisfied by those finalized user-selected results, not by completion of the
cancelled five-setting search. This override applies only to LambdaMART.

## Materialized control grid and accounting

`control_plan.py` declares 28 settings: the full reference; weight decay 0.001
and 0.1 (the reference uses 0.01); frozen and last-two-block encoders; LoRA ranks
4/16/64; six additional loss subsets; four static-weight alternatives; four
contrastive controls; and six ranking controls. It generates 168 tuning configs
(28 × 3 learning rates × 2 splits), then 112 confirmation configs only after
B-validation selection (28 × 2 further seeds × 2 splits). Every fit retains the
15-epoch horizon. These 280 fits exclude the dataset ablations, additional B
realizations, Tier 4 and conditional method-catalog experiments.

The MD requires equal tuning budgets and at least three seeds; it does not
mandate three LR candidates. The three-candidate grid is this protocol's declared
implementation choice. `resource_audit.py` records measured reference save-time
cadence, quota headroom and hard-link-aware file footprints without scheduling
work or changing that choice. Its primary-grid projection assumes every setting
runs at the full-reference pace; it is a cost scenario, not a completion ETA.
Frozen/partial/LoRA arms, final exports, selected evaluations, CPU reports and
the additional experiment families have different or unmeasured costs.

The current B policy keeps all15 epoch weights for allocation-native losses and
averaging. Retaining that policy across the entire primary grid exceeds current
archive headroom under the full-model weight-size scenario. Existing quota gates
protect individual launches; a separate verified release policy will be needed
before accumulating the entire grid. The initial reference and required selected
weights remain protected. No resource audit deletes files or treats unmeasured
optional split work as affordable. User time/compute constraints and measured
allocation costs must inform the MD's affordability condition.

The five-point static-weight sweep includes the shared reference R/B/C weights
0.25/0.25/1. Change B alone to 0.0625 or 1, or C alone to 0.25 or 4. This is an
axis sweep, not a full Cartesian search or evidence of a globally optimal weight
mix. Raw loss scales change as well as the relative weights; the matched LR
budget does not make those effects identical. The regularization comparison
similarly uses the shared full reference plus its two declared decay settings.
Report each family’s total tuning count, including its shared reference, rather
than comparing family maxima as if all families had the same number of trials.

`weight_selection.py` reports Q34/E39's A-validation versus B-validation choice
over these same five settings and three LRs. It requires all 30 existing tuning
fits, their unchanged registered configurations, native validation exports and
the same verified training qualification. It makes no new fit or confirmation
request and reads no test scores. Each setting uses the existing LR rule;
cross-setting score ties prefer the shared full reference, then BCE-low,
BCE-high, contrastive-low and contrastive-high in that fixed order. Undefined
scores and zero defined coverage cannot win, while differing nonzero coverage
remains explicit rather than being silently treated as an identical cohort.

The A-validation choice is a diagnostic comparison with the deployment-oriented
B-validation choice. The report lists 15 tuning candidates per split, 30 unique
fits including six shared full-reference fits, and the 20 confirmations already
in the primary plan. If A selects a different LR from that setting's B-selected
LR, only seed42 covers that A-selected configuration; the report must not borrow
seed43/44 evidence at another LR. These validation choices do not establish test
improvement or explain how the reference weights were chosen before this study.

The historical Q34 audit is separate from this prospective selection. W&B records
three R/B/C settings (0.5/0.5/0.5, 0.5/0.25/0.25, 0.25/0.25/1) in six named
three-objective runs, each setting on the intended random and scaffold regimes.
Their saved launch metadata and configurations do not identify a weight-selection
decision or its validation split. No linked sweep or notes does not rule out
manual tuning. All six launch config paths are absent from their recorded Git
commits, and original split hashes were not recovered. These historical runs do
not substitute for the current matched controls or establish which weights were
best. Evidence: `historical_weight_audit/wandb_20261102/{audit,verification}.json`
on Atum. On 2026-11-02 the user clarified that the original weights were an
"arbitrary selection", supplying the answer to Q34's historical question.
They were not selected through validation tuning. The current controlled
weight-sweep experiment E39 remains pending.

Checkpoint selection-rule and averaging contrasts use the same B-selected LR
within each setting to isolate those changes. They do not independently retune
the LR for each checkpoint rule. The original requested B reference remains a
separate protected run; it is not silently reused in this grid, which adds a
fixed training evaluation probe and prediction export instrumentation.

Preparation writes immutable YAML configs, job metadata and a plan fingerprint
without launching training. It hashes the input partitions and uses only the
training replicate RMS for soft targets. `control_driver.py` runs one ready job
per invocation. It requires all cheap baselines to finish on the same data,
checks all four authorized GPUs are free, and qualifies the 28 model settings
plus the real Trainer on four GPUs before starting the grid. A partial existing
run is never automatically restarted. Test evaluation remains a separate step
after validation-only selection; training completion is not experiment completion.

## Encoder freedom and loss readouts

- Full, frozen, last-two-transformer-block unfreezing, and LoRA ranks 4, 16 and
  64 are separate settings. Partial unfreezing also trains ESM's final layer
  norm. Frozen embeddings/lower blocks stay in evaluation mode; adapted blocks
  train normally. LoRA adapts attention query/value matrices on both encoders,
  with alpha twice the rank, zero initial delta, and no additional dropout.
  Original encoder dropout remains active in LoRA/full training.
- Fork the initialization RNG around adapter creation, preserving the common
  initial projection, fusion and head weights across encoder-freedom arms.
- The model has separate ranking, BCE and contrastive scoring pathways. For
  {R}, {RB}, {RC}, {RBC}, use the ranking head. For {B}, {BC}, use the BCE logit.
  For {C}, use cosine similarity. Persist this policy in the checkpoint config
  and use it for validation selection, evaluation, export and deployment.
- Freeze modules belonging exclusively to disabled objectives. Do not change
  the common architecture or initialize a different head for a subset arm.
  Fixed contrastive scale stays at the original 13. Stop-gradient detaches
  encoder outputs only before the separate contrastive projections; R/B still
  update the encoders and C still updates its projection heads.
- Set `data.sample_all_assays: true` for every controlled arm. Otherwise the
  historical sampler switches from all assays to ranking-eligible assays when
  C is disabled, changing list composition, dataset-item count and update count
  along with the objective. Preserve the reference's full row coverage and
  classification-row packing even in arms where B is disabled.
- Score all trained pathways separately. Comparisons of R versus RB/RC/RBC
  isolate auxiliary-objective effects on the same readout. The entire seven-arm
  grid compares complete scoring systems; a factorial interaction computed
  across changing readouts is not evidence of gradient conflict by itself.

`loss_subset_analysis.py` assembles Q30/E38 from all seven completed primary-arm
reports. Each must pass the native evaluation, frozen selection and artifact
checks in `controlled_summary.py`. The compared fits must share the metric,
composition, noise and qualified training implementations. The family uses 42
tuning and 28 confirmation fits, including the ten shared full-reference fits;
the report creates no further fit or selection.

All seven systems retain their declared scoring outputs, individual seed scores,
coverage and selected LR for each checkpoint rule. For the four systems with R
present, paired contrasts on the same eligible, mutually defined assays are
RB−R, RBC−RC, RC−R, RBC−RB, and the conditional interaction RBC−RB−RC+R. Their
2,000 bootstrap draws resample whole split units, including units with no
eligible scores, and share draws across contrasts. Original-cohort means remain
visible alongside common-cohort means. Empty common cohorts give undefined
effects, never zero effects.

This interaction concerns B and C with R present and a common ranking-head
output. It is not a three-factor interaction, an estimate from an untrained
empty-loss cell, or evidence of gradient alignment. Each arm's validation-chosen
LR and checkpoint remain part of the training procedure being compared. Seed42
has tuning exposure; all-three-seed and seed43/44-only spreads stay distinct from
the conditional unit intervals. A−B effect differences compare different
populations and receive no paired cross-split confidence interval. No test score
changes the setting, selection or training budget. A preflight inventory of
report paths does not establish a scientific loss comparison.

## Runtime qualification and retention

`qualify_controls.py` exercises all seven loss subsets plus frozen, partial,
LoRA, fixed-scale and stop-gradient settings through distributed forward,
backward and three optimizer updates with accumulation/no_sync. It compares
fitted parameters across ranks. Tiny ESM/RoBERTa CPU qualification is followed
by actual cached-encoder qualification on the authorized GPU UUIDs before
starting these new controlled arms. These checks establish execution, not
scientific performance or sufficient statistical power.

The 19-case cached-encoder qualification, including the reference/unmasked/
training-known-positive policies, completed on GPUs 5 and 6 at commit cd6cea1
(`controls_gpu_qualification_cd6cea1.json`). A separate real-Trainer smoke run
at 585ed07 completed two epochs and six optimizer updates on the same two GPUs.
It verified both ranks, four trajectory and two selected-weight exports, skipped
test scoring, exact singleton-average tensors, and retained artifacts after
cleanup (`trainer_export_qualification_585ed07/qualification_verification.json`).
These earlier checks were followed by qualification signature
`26d6c2fc024f95dbacf433387f9488bff1f6c615318a85c68108a7429a54209e`:
all 28 settings passed with the real encoders on GPUs 0,2,5,6, followed by a
two-epoch real-Trainer smoke run with all four rank completions and retention
evidence verified. These remain runtime checks on synthetic/smoke inputs.
The first full-data A fit subsequently started as W&B run `xffenit7`;
its training and scientific comparisons remain unfinished.

## Ranking controls

The reference is a top-weighted Plackett-Luce selection loss, not a sum of
pairwise logistic losses. Its three-item-or-larger sampled lists are averaged,
but larger original assays generate more lists. These distinctions are part
of the comparison, not interchangeable uses of the word "group".

- Within-list normalization divides the reference's sum by the retained
  selection-weight sum before averaging lists. A selection is retained only
  when at least one weaker item lies beyond the affinity margin.
- Pairwise top and uniform controls both use one logistic term per unordered
  pair beyond the same affinity margin. Top weights use the higher-label item's
  tie-averaged reference position weight. Divide by the retained pair-weight
  sum within each list. Comparing these two isolates pair weighting; comparing
  either to the reference also changes the loss family.
- A zero-margin hard-pair control precedes the soft-pair comparison. Soft
  targets are sigmoid(label difference / T), include near ties and give exact
  ties probability 0.5. Hard exact ties are omitted.
- Fix T from each training partition's raw replicate-difference RMS d using
  T = sqrt(3) d / pi, which matches the variance pi^2 T^2 / 3 of a logistic
  difference distribution to d^2. This is a noise-calibrated sensitivity model,
  not a claim that median-aggregated labels have the uncertainty of individual
  raw measurements. Validation/test measurements do not determine T.
- Training and full-dataset evaluation reconstruct the same configured ranking
  objective. The metric eligibility rule remains fixed for comparability.

Original-assay balancing is a separate setting in the per-group comparison.
Let N be training dataset items, G eligible original assays and o_g the number
of sampled lists from assay g in an epoch. Its normalized list loss is weighted
by N / (G o_g), summed in each microbatch and divided by that microbatch's item
count. At fixed parameters, the complete epoch's item-weighted mean is exactly
the macro assay mean of its sampled-list losses. This corrects multiplicity
without changing sampled rows, batches or optimizer-update count. It does not
make a sampled subset of pairs equal to every pair in a large assay.
Validation instead averages its complete-coverage list losses within each
original eligible assay, then averages assays and fixed partitions.

## Contrastive masking controls

Compare the reference, no mask, and the reference plus all training-known
positive target/molecule pairs. The reference already hides same-assay
co-candidates and cross-assay same-target/duplicate-query-molecule candidates.
The no-mask setting removes this whole bundle; its comparison is not an
isolated estimate of one masking mechanism. The added training-known mask
isolates the remaining observed false negatives found in the sampling audit.

Training-known means any curated observation in the actual training partition
strictly above pChEMBL 6 after conversion to the float32 supervision dtype.
Do not add validation/test labels or original measurements that were excluded
from training. This conservative rule can mask pairs with conflicting training
assay results; it treats them as potentially positive, not confirmed negatives.
Keep supervised targets. Record the canonical index digest, threshold, source
row count and pair count with each run. The training collator and full-dataset
loss evaluator share this index. Evaluation maps identities into its own
integer namespace without consulting its labels. Saved models score new inputs
without the index, which is used only for contrastive loss construction.

Controlled retention must preserve both selection rules, the trajectory
measurements and the averaging candidates. It must fit the authorized storage
quota without deleting original data or the first reference's checkpoints.
Before controlled-run checkpoint cleanup, the implemented retention/export
checks must first demonstrate that required artifacts survive.
Training with both selection metrics will distinguish
best-within-15 selection from an actual patience-based early-stopping rule.

Controlled runs enable compact per-epoch prediction exports for validation and
a fixed probe of 2,048 rank-eligible training assays, selected by training seed.
These come from the existing evaluation pass and retain all scalar scorers,
labels and row identities; they do not retain full embedding matrices. Probe
results describe those selected eligible assays, not the full training set.
The selected checkpoint's final evaluation has a separate namespace from the
last epoch, even though Trainer's global step is the same after best-weight reload.
Hyperparameter candidates set `evaluate_test: false`, which also avoids loading
test examples. Test scoring follows validation-only hyperparameter selection.

`retain_controlled.py` reconstructs best macro Spearman and minimum aggregate
loss from all 15 epochs, plus both patience-3 prefix selections (strict
improvement, earliest tie). Prefix selections use the original 15-epoch learning
rate schedule; they are counterfactual early stopping, not measured compute savings
or retraining with a shorter schedule. It preserves all four selected checkpoints.
The averaging window runs from the minimum validation-loss epoch through epoch
15, inclusive, with uniform native-parameter weights. If the minimum is at epoch
15, this is a singleton and must not be reported as an averaging improvement.
This is the specification's post-minimum averaging comparison, not a claim to
reproduce all SWA/SWAD or ensemble algorithms in its method catalog.

Before any optional cleanup, retention verifies all ranks completed, configuration
provenance, every validation/training snapshot checksum, checkpoint epoch/step,
and tensor equality of the root model to the selected macro checkpoint. It saves
and verifies the averaged model, preserves selection weights and all checkpoint
metadata, then removes only redundant weights/optimizer payloads of a marked
controlled run. Original data and the first reference are not eligible. Cleanup
removes optimizer-resume capability from these already completed runs.

The matched A reference `full__A__seed42__lr1p0` additionally needs every epoch
for the Tier 1 directional contrastive audit and representation analyses.
Compact scalar snapshots omit the embeddings required for those calculations;
they cannot substitute for the model weights. `protect_trajectory.py` preserves
all 15 weights, model configurations and epoch states under
`controlled/protected_trajectories/full__A__seed42__lr1p0` on the existing archive.
Weights are hard-linked, so source-path cleanup does not free their data; this
is retention protection, not an independent byte-level backup.

The already running driver imported the original retention function. The
protector therefore temporarily sets only its managed-run cleanup permission
to false, keeping an exact copy of the original marker. Training configuration,
optimization and GPU processes are unchanged. It restores the original marker
only after all epoch payloads verify. If the driver reaches cleanup first, that
stage fails safely without pruning; finish retention after protection is verified,
without rerunning training. Interrupted protection leaves the cleanup hold in
place. This mechanism is restricted to the first matched A reference, and does
not declare any scientific audit complete. The full original B reference was
already retained separately.

`checkpoint_evaluation.py` provides fixed-weight native evaluation for the
retained neural models. It preserves the original evaluation partition seed,
candidate-pool assay batch size and ranking readout, exports the native scalar
predictions, and verifies unchanged model parameters and absence of an optimizer.
Training-known masks require the original training-positive index; held-out
labels never build that index. Tiny CPU execution passed for all three readouts
and the known-positive-mask path.

`postselection.py` connects this evaluator to completed primary-grid runs.
Before writing an immutable evaluation plan it verifies all six matched tuning
completions, reconstructs the saved decision from B validation alone, and checks
the retained selections, model weights, configurations, data and software.
Every matched LR candidate contributes its best-macro checkpoint. Only the
common B-selected LR receives all five checkpoint-rule comparisons, including
the weight average. Rules selecting the same epoch share inference; a singleton
average stays explicitly identified. Seeds retain their tuning/confirmation
roles. Validation and test are re-evaluated through the same native evaluator;
the original four-rank validation scores still determine every selection.

Use `postselection.py prepare --job-id ID` to freeze a ready run, then
`postselection.py run --job-id ID --workers 4` to evaluate it. The launcher uses
the existing training lock and only the four allowed GPU UUIDs. Results live
under the authorized archive via `outputs/paired_generalization/postselection`.
An incomplete output requires inspection; complete outputs must pass checksum
and provenance verification before reuse. A CPU qualification path cannot pass
the scientific completion verifier. Saved tiny-model execution exercises both
held-out partitions and the reconstruction of the original training-positive
index, without training or modifying retained training artifacts.

Scientific postselection evaluation has not run yet. Statistical comparisons,
calibration summaries, additional B allocations and multi-seed reporting remain
separate required work; completing native inference alone does not finish them.

`controlled_summary.py` collects the complete ten-fit primary arm: six seed42
tuning fits and four A/B seed43/44 confirmations at the B-selected LR. It
re-verifies the frozen validation decision, native evaluation receipts, current
metric/data provenance and each checkpoint's saved diagnostics. All three LR
candidates contribute matched A/B best-macro results; only the selected LR
contributes all five checkpoint rules. Seed42 tuning exposure stays explicit.
Reports contain individual scores/coverage, three-seed and confirmation-only
descriptive spread, same-seed A-minus-B differences and separate seed42 LR
sensitivity. All three scoring pathways and the existing rank, pair and eligible
NDCG summaries are retained without selecting a new metric or model from test.

Checkpoint changes relative to best macro also use paired whole-unit bootstrap
intervals within each fitted run and partition. These contrasts match assay
identity, unit, size and eligibility, restrict correlations to assays defined
at both checkpoints, and keep units with no eligible correlations in the
resampling population. They are distinct from differences between separately
aggregated scores and from seed spread. The report also preserves matched A/B
composition comparisons, with no cross-split interval. This finishes neither
the scientific arm nor the broader study until actual selected evaluations
exist; cross-arm, leak, learning-curve and dependent-allocation comparisons
remain separate deliverables. `preflight` inventories completion markers without
interpreting marker existence as verified scientific completion.

`full_evaluation_queue.py` provides the bounded handoff after the existing full
confirmation coordinator. It waits for that exact owned process to terminate,
reconstructs its completion chain and frozen B-validation LR decision, and then
evaluates exactly the six tuning plus four confirmation fits. Native inference
uses the existing shared GPU lock and authorized four-device evaluator, with
storage checks before launch. The coordinator exposes no GPUs; evaluation
children inherit `ACCELERATE_USE_CPU=false` and expose only their assigned
authorized UUID. Per-fit diagnostics and the complete primary summary run on CPU.
The queue pins its parent, configurations, reporting/evaluation sources, packages,
descriptor design and noise record. An existing partial output, failed worker or
changed source/evidence stops it for inspection; there is no automatic retry.
Completion means only that this full-arm primary report exists. It does not
train another arm, select from test scores, run additional allocations, release
weights or mark the study complete.

`native_active_handoff.py` is a separate CPU-only handoff for the already running
`full__A__seed42__lr0p3` fit. It records the launcher and four rank identities
separately from the persistent tuning coordinator, allowing that coordinator to
finish retention and proceed to its next registered fit. Analysis starts only
after the five original training handles terminate and all fifteen native
validation/probe epochs have verified completion evidence. The verified five-epoch
prefix, source files, configuration and runtime qualification remain pinned.
Transient process/status-read failures cause another observation of the same
handles; changed provenance or failed analysis requires inspection. The worker
never retries fitting or analysis, runs inference, reads test predictions or
creates another training job. It reuses the existing trajectory analyzer and
independent verifier; preparing or launching it does not count as a scientific
result or completed experiment.

`native_tuning_audits.py` is a separate CPU-only worker for exactly the three
remaining fits in the existing tuning queue: B at LR0.3, then A/B at LR3.0.
It requires the completed A LR0.3 audit, freezes the existing queue and source
evidence, and waits for each driver's verified completion receipt. That receipt
must include all four rank completions, the training summary and retention;
the first fit's observed launcher must also terminate. Future fits may finish
while an earlier audit runs. Each then receives the unchanged native analyzer
and an independent check of all30 validation/probe snapshots. Existing or
partial analysis output is never adopted or retried. Observation errors re-poll
the same coordinator; changed or missing scientific evidence stops the worker.
The worker performs no fitting, inference, LR selection or test scoring, and
does not expand the existing tuning queue. Its preparation gives no experiment
completion credit. All65 focused new-worker and existing-handoff checks passed.

`native_lr_comparison.py` reports Q04's completed A seed42 base-LR versus LR0.3
sensitivity across all15 matched epochs. It requires the exact independently
verified trajectories, identical reference rows/labels and runtime qualification,
and configuration equality except output directory and the four rates scaled
by0.3. It rehashes the saved predictions and audit tables and preserves headline
coverage. Validation and fixed training-probe points, native loss components and
eligible NDCG remain separate. Descriptive validation peaks do not choose the
study LR; there is no test scoring, fitting, seed-variance estimate or interval
treating epochs as independent replicates. Q04 still requires the remaining
controlled comparisons.

`regularization_queue.py` continues E33 only after that exact full-evaluation
coordinator terminates and its ten-fit report passes verification. It runs the
already registered weight-decay 0.001 and 0.1 settings, in that order. Each gets
six A/B seed42 tuning fits, B-validation LR selection, four A/B seed43/44
confirmations, selected-checkpoint evaluation and a verified primary summary.
The existing weight-decay 0.01 reference is reused. This is 20 existing planned
fits, with no new setting or search expansion. The queue then stops before
encoder-freedom, loss-subset, allocation and other study work.

The coordinator has no visible GPUs. Training uses the existing four-GPU driver
and native evaluation uses the existing evaluator, both protected by the shared
lock and physical-GPU UUID allowlist. Archive/home quota and occupancy are checked
before each launch. Parent receipts, configurations and source files are pinned;
new confirmation configurations must exactly match the frozen validation choice.
Existing output, failed training/evaluation or changed evidence requires
inspection, never an automatic retry. All earlier arm receipts are checked again
before final completion. Scientific comparison and E33 closure require the actual
reports; preparing or launching this handoff does not complete an experiment.

`regularization_analysis.py` first freezes a decay choice from the eighteen
existing native seed42 tuning fits. Each setting keeps its existing
B-validation-selected LR; the highest best-macro B-validation score then chooses
decay. Exact family ties prefer the full reference (0.01), then 0.001, then 0.1.
The family therefore has nine decay/LR candidates per split. No test score or
confirmation-seed score participates in this choice, and no new fit is created.
The chosen decay is shared across all checkpoint rules. This isolates checkpoint
rule changes on common training procedures; it does not independently optimize
the hyperparameters of every stopping algorithm.

The report requires all thirty verified primary analyses and their three
canonical arm summaries, with unchanged qualified training and metric sources.
It preserves every setting, each B-selected LR, all five checkpoint rules,
coverage, seed42 tuning exposure and seed43/44 confirmations. It carries the
within-fit checkpoint changes and adds each alternative decay minus the shared
reference on paired assays/units within each split and seed. Each contrast uses
the two fitted models' mutually defined eligible cohort, which can differ between
alternatives. Conditional whole-unit intervals and seed spread remain separate;
A−B changes are descriptive across different held-out populations, without a
paired cross-split confidence interval. Completed paths alone do not pass the
native inference, selection and artifact gates. The reference remains one of
three declared settings, not a claim of globally optimal regularization.

The user confirmed the leak experiment’s two axes: molecule-only versus full
protein–molecule scoring, crossed with leak-in versus leak-out training. Test
groups remain fixed, and leak-in training must match leak-out training size,
group-size mix, positive rate and replicate counts. `leak_design.py` version 2
also exactly matches protein counts, per-assay ranking eligibility and the
number of strict contrastive actives (float32 pChEMBL >6). BCE positives use
partition-local pair medians >=6, so matching BCE counts alone was insufficient.
The original construction is preserved; version 2 keeps its fixed test sets.

A keeps the original test assays and exchanges only whole training assays.
Its matched views passed production cache/label/sampler qualification, with
equal optimizer steps in both arms. B's fixed donor pool cannot exchange any
whole training component while preserving protein support: every training
component contains a protein absent from that pool. The separate B whole-assay
proposal partially removes nine training components and requires the user's
choice before any B experiment configuration or training. This is an explicit
exception to the MD's indivisible-unit rule, not a selected design.

The molecule-only cell replaces every protein sequence/mask with the same
tokenized `G` sequence before deduplication. It preserves initialization,
parameter count, loss weights and training group metadata; inference scores
cannot depend on the supplied protein. Content/mask/length invariance, including
training gradients with matched RNG, and two-rank tiny CPU-DDP updates passed.
This does not qualify the real four-GPU model or establish a performance result.
E40 fitting/selection/evaluation and Tier 4 adaptation remain pending. E41 has
its own prepared manifests and execution path described below.

## Additional B allocations and checkpoint protection

The original splitter changes only validation/test assignment after fixing
training membership. On the actual data, seeds 42/43/44 alone give identical
assignments. The audited supplement randomizes equal-size component priority
for seeds 43/44 within the same size-priority allocation, balancing objective
and 64-restart budget. It produces three distinct held-out allocations with
identical training support. They are dependent allocation sensitivities, not
independent split draws or variation in scaffold geometry.

`realization_protocol.py` freezes the additional assignments separately from
the immutable primary control plan. It maps raw source rows to the actual
cached-token order by assay, protein and molecule identities, checks labels at
the training float32 precision, and rejects missing/duplicate observations,
changed held-out support or fragmented units. Binary labels must match the
prepared pair-median partitions; raw observation labels may differ.

`realization_predictions.py` exports scalar predictions for every epoch over
the fixed held-out union using batch size 32 and BF16 inference. It computes
only each allocation's validation macro Spearman, breaking ties at the earliest
epoch and reporting coverage. No test metric is calculated in this export or
selection step. The primary Trainer selection remains unchanged; uniformly
batched inference selections are reported separately because BF16 batch
numerics can differ. This pipeline requires runtime qualification before
claiming it has produced scientific results.

Controlled B cleanup now requires all 30 epoch/partition exports, their verified
checksums and the allocation-specific validation selections. It currently keeps
all B epoch weights because allocation-specific full-loss and averaging analyses
still need them. Completed optimizer payloads can be removed. A separate verified
release protocol is required before reducing those model weights to the union
of selected checkpoints and averages. Scalar exports live on the authorized
archive and omit large embedding matrices. The protected first reference remains
untouched.

Each allocation must select its own LR from the common three candidates. If
winning LRs differ, confirmation training must cover their union at seeds 43/44;
the primary 280-fit count does not cover those additions automatically. Identical
training support permits sharing an exactly matched fit, but does not justify
sharing validation selection. Allocation-specific aggregate-loss and averaging
comparisons still require full loss evaluation, which scalar exports alone do
not provide. Measured inference runtime/storage and any remaining affordability
limits must be reported; none of these pending comparisons is declared complete.

`realization_selection.py` implements this separate validation-only LR decision.
It requires all six primary tuning fits and the frozen primary LR decision,
verifies every allocation export against completed retention evidence, and freezes
the three allocation decisions plus their union of confirmation requests. Matching
primary B seed-43/44 fits are reused; only missing B fits are supplemental. They
do not create matched A results at those additional LRs. `run-next` waits for all
four primary confirmations, holds the shared driver lock, and runs at most one
supplement using the existing four-GPU driver. It refuses partial-run retries and
checks quota, GPU ownership and frozen configurations. `preflight` is read-only;
`prepare` freezes selection, and `verify` reconstructs it. This does not score
test roles, evaluate allocation full losses, average or release weights, or close
the realization experiment by itself.

`realization_losses.py` evaluates every epoch on each allocation's validation
view only at its frozen selected LR, for tuning seed42 and confirmation seeds
43/44. Views concatenate the original validation/test token caches in their
frozen order, then select the frozen allocation indices without retokenizing,
relabeling or fragmenting units. Original test rows can belong to an alternative
allocation's validation role; that allocation's test role is never evaluated in
this stage. Training-known positives still come exclusively from the unchanged
training cache and its verified index.

The production single-device evaluator preserves configured ranking partitions,
assay pools, precision and eligibility. Its native validation losses define the
loss-based checkpoint and patience-3 rules; the already frozen uniform-batch-32
Spearman trajectory continues to define LR and macro-based rules. Native
Spearman is retained separately and never replaces those choices. This preserves
the existing numerical distinction between inference protocols. All 15 passes
must verify before creating uniform native-parameter averages from each
allocation's minimum-loss epoch through epoch15; identical windows share one
average, and singleton windows stay explicit. Cost receipts record timed pass
seconds and output bytes. The shared GPU lock and quota checks protect running
training, and explicit CPU qualification artifacts cannot count as scientific
GPU results. Selected test evaluation, seed/allocation summaries and any weight
release remain separate requirements.

`realization_postselection.py` requires the verified native validation trajectory
and completed averages before it freezes any test tasks. It reconstructs the
checkpoint rules and averaging windows from that validation evidence. Rules
choosing the same epoch share inference; existing epoch validation outputs are
reused, while each selected average receives validation and test evaluation.
Every test role uses its frozen allocation indices. All predictions are checked
against the actual row identities, float32 labels and dataset fingerprint,
including reused validation outputs. The runner uses the shared GPU lock and
records output costs. Training-known positives remain training-only. Neither
test scores nor this evaluator change any LR, epoch, average or training budget.
Scientific seed/allocation comparisons and weight release still require their
own completed evidence.

`realization_analysis.py` applies the existing CPU diagnostics to completed
selected-model predictions on each allocation's exact validation/test membership.
It verifies row identities, labels, whole-component membership, group counts,
frozen evaluation receipts and training-only noise thresholds. Descriptor cells
are recomputed for each allocation without using model scores. Per-fit outputs
include all three scorers, whole-unit bootstrap intervals, pair/noise-band and
top-k metrics, scorer agreement, validation-fitted calibration and shift checks.
These outputs do not infer missing A model results at supplemental B-only LRs.

The summary requires the complete selected three-seed by three-allocation grid
for all five checkpoint rules. It reports seed42 as tuning-exposed, seeds43/44
as confirmations, and gives their individual scores and defined coverage. Seed
means/SDs/ranges within each allocation and allocation sensitivity within each
seed stay separate. The latter includes changes in test membership and in the
allocation-specific validation LR/checkpoint choices; it does not isolate split
assignment alone. Neither descriptive spread becomes an independent-realization
confidence interval, and the nine cells are never pooled as independent runs.
Missing/changed evidence, different metric code and CPU qualification outputs
cannot enter a scientific summary. `qualify-references` checks production
descriptor membership without loading models or computing prediction scores.

## Whole-unit learning curves

`learning_curve_plan.py` fixes nested 10%, 25%, 50% and 100% unit prefixes from
one label-blind permutation with subset seed 42. A unit is an entire assay for
A and an entire scaffold–assay component for B. Use floor(fraction × unit count),
with a minimum of one unit. The permutation neither forces nor excludes the
giant component. Keep the same subsets across optimizer seeds 42/43/44 so seed
spread is not mixed with variation in the sampled subsets. Report actual row
counts/fractions, assay sizes, target coverage and component sizes alongside
the requested fractions; a unit fraction is not a row fraction.

Binary pair medians are recomputed from the selected training measurements
only, at the original source precision and threshold 6. Continuous labels stay
unchanged. Omitted training assays must not influence the subset's binary labels.
The manifest pins the full source/cache bytes, selected indices, labels and
unit map. The training loader independently checks whole-unit inclusion,
identity alignment and the subset medians. It replaces the small label column
before applying an Arrow row view, avoiding token payload duplication or index
flattening. Tests cover multi-shard caches, shared token buffers, worker
serialization, sampling, fixed training probes and the training-positive index.

Each smaller fraction gets the same three LR candidates on both A/B at seed42,
then seeds43/44 at the B-validation-selected common LR: 18 tuning and 12
confirmation fits in total. The exact full-data endpoint reuses the primary
full-arm grid after checking row/label identity; it is not retrained. These are
15-epoch curves at batch 12 × 4 GPUs × 4 accumulation. Updates and processed
observations vary with subset size, so this is equal-epoch, not equal-compute,
comparison. A flat B curve near the full-row regime cannot establish general
data saturation. Runtime qualification, training, validation-only selection,
test evaluation and any additional B-allocation comparisons remain required.

`learning_curve_driver.py` executes one prepared job per invocation using the
primary driver's shared lock, GPU allowlist, runtime qualification, training
and checkpoint-retention path. It first requires all ten primary full-arm
fits (six tuning and four confirmations), then reuses them as the 100% endpoint.
For each smaller fraction, all six A/B tuning jobs must complete before B
validation selects the common LR and seeds 43/44 are materialized. Confirmation
configs copy the winning tuning config, changing only seed and output directory.
Registered B curve fits also export every epoch for the additional fixed
held-out allocations before retention. Test comparisons remain a later step.
The primary and curve drivers cannot own the GPU training lock simultaneously;
an existing incomplete run requires inspection rather than automatic relaunch.

`learning_curve_evaluation.py` supplies native fixed-weight evaluation for the
three smaller fractions. It reads the ten completed primary full-endpoint fits
without materializing or refitting them, reconstructs each fraction's LR choice
from its six completed A/B tuning fits using B validation, and verifies every
input fit's complete 15-epoch history, four rank receipts and common training
qualification. It recomputes the frozen nested unit selection and subset-local
binary labels, verifies exact full-endpoint row/label identity, and checks that
the training summary records the same view and realized size. The held-out
partitions remain those of the primary plan. The registered reference mask and
ranking readout are retained; no new loss or masking setting is introduced.

The selected LR receives the existing five checkpoint rules; other seed-42 LRs
receive best-macro evaluation only. Identical checkpoints share inference.
`learning_curve_evaluation.py preflight` only inventories preparation. Once the
required fits and selection exist, `prepare --job-id ID` freezes a checked
evaluation plan and `run --job-id ID --workers 4` uses the existing training
lock and only GPUs 0,2,5,6. Native receipts bind source data, checkpoints,
predictions, precision and device provenance; CPU qualification cannot be used
as scientific GPU evidence. Partial evaluation output requires inspection.
`learning_curve_analysis.py` provides CPU reporting after those native evaluations
complete. `job --job-id ID` verifies the completed GPU evaluation before reading
predictions, reproduces the native headline metric and saves exact per-assay
tables for all three scorers. It uses the fixed held-out identities, labels and
indivisible units; full-training scaffold support is not treated as support in
the smaller training subset.

`report --analyses PATH... --endpoint-summary PATH` requires all 30 smaller-fit
analyses and the canonical ten-fit primary full-reference summary. It reuses the
100% endpoint, checks the registered B-validation choices and all five checkpoint
rules, and independently counts rows and units from each saved subset. Results
include all nine core metrics, individual seeds 42/43/44, confirmation-only means
and spread, and separate same-LR seed42 curves. Adjacent slopes use actual row
and unit doublings; equal sizes produce an undefined slope, and dips remain
visible. For the ranking-head headline, a sensitivity analysis intersects the
eligible, defined assay cohort across all four fractions before paired whole-unit
bootstrap contrasts. Zero-eligible units remain in the bootstrap; intervals are
conditional on fitted models and assignments, not seed or subset-redraw intervals.

The PNG/SVG overview shows best-macro test results against actual rows and units,
with common score axes and explicit seed roles. The JSON retains all metrics,
rules, coverage, selected LRs and actual training statistics. Fifteen epochs
fixes passes, not optimizer updates or example exposure. Each fraction may select
a different LR, so common-LR seed42 sensitivity remains separate. The one nested
subset permutation does not quantify subset-selection uncertainty. Split B's
near-full row coverage does not support low-data extrapolation.

`preflight` is read-only preparation inventory. No curve fit, scientific
evaluation, performance curve or GPU queue has started. Actual performance
reporting and any conditional additional-allocation work remain required.
