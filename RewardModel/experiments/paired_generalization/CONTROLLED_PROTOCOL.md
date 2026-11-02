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

The five-point static-weight sweep includes the shared reference R/B/C weights
0.25/0.25/1. Change B alone to 0.0625 or 1, or C alone to 0.25 or 4. This is an
axis sweep, not a full Cartesian search or evidence of a globally optimal weight
mix. Raw loss scales change as well as the relative weights; the matched LR
budget does not make those effects identical. The regularization comparison
similarly uses the shared full reference plus its two declared decay settings.
Report each family’s total tuning count, including its shared reference, rather
than comparing family maxima as if all families had the same number of trials.

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
