# Paired input ranking generalization

The governing specification is `/Paired input ranking generalization.md`. Do not
expand the scientific scope. Answer its 34 questions and execute its suggested
experiments, distinguishing required comparisons, conditional interventions,
inapplicable methods, and missing evidence.

User-approved execution constraints:

- Code and execution stay inside `/home/atabey/Prot2Mol` on Atum. After the
  150 GB home quota blocked preparation, the user explicitly authorized moving
  storage to `/media/16TB/archive/atabey/` and back as needed. Verified archive
  copies retain access through symlinks at their original project paths.
- Physical GPUs 0, 2, 5, 6 only, selected by their queried UUIDs.
- Code is developed locally, committed and pushed on
  `codex/paired-ranking-generalization`, then pulled on Atum.
- Deployment means unseen molecules produced by a molecule generator.
- First train a fresh 15-epoch scaffold baseline using run `m5guoqkx`'s config.
- Approved batching is 12 per device, 4 devices, 4 accumulation steps: effective
  batch 192 assay-list items, versus 48 in the single-GPU reference. Contrastive
  candidates remain local to each forward pass. This is not an exact replication
  of the reference optimization trajectory.
- Keep the original learning rates and a 15-epoch linear schedule with 6% warmup.
- Select the maximum validation macro Spearman. Retain every epoch checkpoint
  for the specified checkpoint analyses. Test results never select checkpoints.
- Seed before model initialization as well as sampling, to make the requested
  seed comparisons reproducible. The original entrypoint initializes the model
  before Trainer sets the seed; this wrapper makes that difference explicit.
- Download raw ChEMBL 37 into the project while training runs, for replicate
  analyses. Do not infer original replicates from median values/counts.

For E08/E09, `similarity_composition.py prepare` freezes an identity/label-only
design before prediction scores are read. Maximum similarity is the actual
binary scaffold relation over all training molecules, irrespective of protein.
The report preserves full assays in all-shared, all-unseen and mixed categories,
and separately recomputes within-stratum metrics with eligibility reapplied.
These two views have different cohorts; neither supplies graded chemical distance.

Composition cells use the MD's eligible group-size bins (3, 4–10, 11–25, >25),
positive-fraction bins (zero, (0,.25], (.25,.5], (.5,.75], (.75,1), one), and mean
pChEMBL bins (<5, [5,6), [6,7), [7,8), >=8). For each partition, each eligible
A group receives its B cell's mass divided by the A cell count. Cells absent in
A are reported as unsupported; B comparisons then use the same common support.
No clipping, score-dependent binning or tuning is performed. The report includes
descriptor balance, effective group count and weighted defined-Spearman coverage.
This is descriptive held-out-label standardization, not a deployable estimator
or causal attribution of the A/B gap; residual differences within bins remain.

`similarity_composition.py depth3` applies the frozen design to the already
selected depth-3 predictions, with no additional fitting or inference. Bootstrap
intervals use whole assays on A and whole components on B, including units with
no eligible stratum. The paired change from unweighted to weighted A uses the
same A draws. These intervals condition on empirical composition weights and
one fitted model; they do not measure weight-estimation, training-seed or split
uncertainty. Matched neural results still require their own saved predictions.

The source is the available 1,957,400-row project-local `all.parquet`. Its hash is
recorded. The original run's data hashes are unavailable, so rebuilt split
identity is not claimed. Scaffold components preserve assays; target coverage,
scaffold disjointness, and pair disjointness are checked before training. The
original median label transformation is preserved for this baseline.

For Q22, `observation_process.py` audits the recorded observation provenance
and curation on CPU. It verifies the original source observations against the
recovered activity IDs, medians and counts, and verifies those IDs against the
read-only ChEMBL37 database. Its comparison cohort is every archived activity
record in assays represented by the curated data. It tabulates missing/censored
records, recorded quality exclusions, and source/document/assay-test annotations.
Raw activity records and median-aggregated observations retain separate
denominators. Missing pChEMBL values are never labelled negative. The archive
and curation code do not identify original laboratory inclusion probabilities,
unpublished measurements or the activity prevalence of generated molecules.
Results are saved in `tier1/observation_process/`; no training data are changed.

Execution order: prepare data; validate four-GPU training/evaluation/checkpoint
loading with a two-step smoke run; complete the 15-epoch baseline; then perform
the specification's Tier 1 audits, Tier 2 baselines, Tier 3 controlled retraining,
Tier 4 mechanistic probes, and supported conditional interventions. Tier 3 uses
both splits, equal tuning budgets and at least three seeds; additional Split B
realizations are subject to the specification's affordability condition.

Runtime artifacts and logs live in `RewardModel/outputs/paired_generalization`.
Scientific experiments are not complete merely because preparation or the
distributed smoke check passes.

`adaptation_design.py` prepares only group memberships and feasibility counts
for Tier 4 E49–E52, on CPU. It splits original validation units by an
identity-only hash into disjoint adaptation/validation sets and preserves all
original test groups. Across-unit budgets are nested whole-assay prefixes at
k=1/2/4/8/16/64. Within each validation/test unit, a fixed half of its assays
is reserved for support and the remaining assays for queries; unused support
assays never enter evaluation when k changes. Feasible and rank-evaluable unit
counts are reported for every k. A's one-assay units cannot support this
within-unit contrast; no assay is split to manufacture one. Each fitted result
must eventually be compared with k=0 on the identical query cohort.

Outputs under `tier4/adaptation_design/` are not training-ready token/label
views or scientific performance. Subset-local BCE labels must be reconstructed
when those views are built. Tier 3 completion and a frozen adaptation fitting
protocol remain prerequisites to model fitting. Original validation labels were
already used for base-model selection, so these are conditional probes on the
existing benchmark, not an independent new test population.

`adaptation_views.py` adds a separate CPU preparation step for those frozen
memberships. It packs only row indices and subset-local binary labels; all
protein/molecule token buffers remain in the existing caches. The loader checks
the design, primary plan, raw source bytes, actual token shards, example order
and float32 continuous labels, then recomputes every view's labels
from raw-precision pair medians in that exact support/query set (positive at
median pChEMBL≥6). Within-unit tasks remain separate fits, with a fixed query
view and unchanged k=0 baseline.
It records feasible and rank-evaluable cohorts separately; paired comparisons
across k must use their common unit identities and identical query views.
Unsupported budgets do not borrow support or query assays from other units.

Run `CUDA_VISIBLE_DEVICES='' ACCELERATE_USE_CPU=true .venv/bin/python
RewardModel/experiments/paired_generalization/adaptation_views.py` on Atum only
after deploying the reviewed code. Outputs are under `tier4/adaptation_views/`.
This command starts no fitting or inference. Local tests cover disk-cache
alignment, subset-label isolation, shared token buffers, native dataset
consumers, source/artifact tampering and both-regime preparation. Production
data-view execution, adaptation protocol selection and runtime qualification
are still required; the Tier-3 gate remains in force.

`adaptation_fit.py` supplies the separate support-only fitting worker. The
`heads` scope trains the active ranking/BCE heads, contrastive projections and
learnable contrastive scale. `heads_cross_attention` additionally trains
`fusion` and `extra_fusion_layers`. Encoders, shared input projections and
normalizations stay frozen in both scopes. Frozen feature modules, including
the shared projection dropout, remain in evaluation mode across Trainer mode
changes. Disabled objectives and a fixed contrastive scale stay disabled.

The worker receives no query dataset, runs the native assay sampler and
optimizer for the caller's fixed whole-epoch horizon, and performs no evaluation,
checkpoint selection or implicit resume. It checks exact optimizer membership
and unchanged frozen parameters/buffers. Epoch payloads contain only the
trainable parameter subset; restoration requires the exact original base model
and verifies payload checksums, names, shapes and dtypes before mutation.
Each task must start from that same base, preventing carryover between units.
The fit record includes its support fingerprint, sampler exposure, optimizer
groups, source and configuration. Low-k supports with no ranking-eligible assay
remain usable by their available objectives and are reported as such.

Its `--qualify-cpu --output PATH` command uses tiny randomly initialized encoders
and synthetic support only. Scientific GPU fitting, the adaptation
budget/validation-selection protocol and the Tier-3 gate remain separate
requirements; this worker and its qualification do not complete E49–E52.

`representation_geometry.py` prepares only fixed identity-based cohorts for
E54/E55. Each A/B train/validation/test partition has separate observation,
unique-protein and unique-molecule views, capped at 25,600 each. The latter two
avoid interpreting repeated entity frequency as representation collapse.
Representatives and ordering use identities only; no label or prediction is
read. All checkpoints must use these same rows and pair draws. Outputs are in
`tier4/geometry_design/`; scientific inference remains behind Tier 3 and must
first verify checkpoint, token and row provenance on the permitted GPUs.

The numerical helpers implement [Wang–Isola](https://arxiv.org/abs/2005.10242)
alignment with squared normalized distance and uniformity with t=2. Uniformity
uses every distinct-row pair when there are at most 262,144, otherwise that
many uniform unordered draws with replacement, seed 42. This is a descriptive
Monte Carlo log-mean, without an uncertainty claim. Zero vectors have no sphere
direction; excluded pairs and coverage are reported instead of jittered.

Effective rank follows the uncentered singular-value entropy in
[RankMe](https://arxiv.org/abs/2210.02885), using the zero-epsilon limit and an
explicit rank-zero convention for an all-zero matrix. It is not the older
diagnostic helper's centered covariance-eigenvalue entropy. Rank is calculated
on raw pooled encoder vectors and raw contrastive projection outputs; sphere
metrics normalize those vectors separately. Sample counts and dimensions bound
rank and must accompany comparisons. The capture hooks target the separate
contrastive projections, not the token projections feeding cross-attention.

Cross-modal alignment is defined only after projection into the trained common
space. Unrelated encoder coordinates cannot be compared by padding or by merely
matching dimensions. Report all sampled observed pairs and, separately, the
native float32 pChEMBL>6 subset; this latter alignment is label-conditioned,
whereas uniformity and rank are label-free. Geometry is not a substitute for
held-out ranking performance. Local numerical/model tests and frozen memberships
do not count as scientific E54/E55 results.

`geometry_export.py` supplies the fixed-weight export worker and analysis. It
joins raw cohort memberships to token-cache identities rather than assuming
identical row order, checks selected continuous labels at native precision, and
exports four raw matrices plus the exact observation/view mapping. It verifies
unchanged model parameters and reproduction of the native cosine pathway, then
calculates each side/stage on the observation and corresponding unique-entity
cohorts. Source/output checksums and CPU-qualification labels accompany results.
The independent scientific launcher must still verify the Tier 3 gate, selected
checkpoint, full token-payload provenance and the shared GPU lock before calling
this worker. Tiny-model CPU tests do not establish scientific GPU qualification.

`objective_probes.py` is the E56/E57 numerical worker, with no scientific job
launcher. It measures weighted R/B/C gradients on disjoint parameter blocks:
individual encoder layers, embeddings/other encoder parameters, token
projections/norms, fusion blocks and objective-specific heads/projections/scale.
Frozen, disconnected and zero-gradient cases remain explicit; undefined cosines
are not replaced with zero. The existing Trainer helper instead measures R/C
at normalized contrastive embeddings. In `mlp_contrastive`, R does not use that
separate output branch, so its boundary diagnostic cannot answer Q33 about
shared encoder or cross-attention parameters.

Following the relative-loss construction in
[Fifty et al.](https://arxiv.org/abs/2109.04617), lookahead records
`1 - validation_loss_after / validation_loss_before`. The source gradient comes
from the training batch; the target is a separate fixed cold validation batch,
as requested by the MD, whereas the original TAG definition uses one training
batch. Each temporary weighted-SGD step updates trainable encoders, interaction,
or their union while holding private parameters fixed. This is not AdamW replay,
task grouping or model retraining. Source weights and explicit per-block step
sizes are recorded; zero target losses and ineffective steps have undefined
relative transfer. Cloned functional parameters/buffers, disabled dropout and
restored module/RNG state keep the loaded checkpoint unchanged.

`objective_probe_design.py` prepares cold validation memberships on CPU. Each
regime uses its own held-out units with no training scaffold match: whole assays
for A, whole components for B. Mixed units are excluded intact. Selection uses
identities/scaffolds only; label-derived eligibility and active counts describe
feasibility. The A cohort is a cold subset, not the full A validation estimand.
The design also records exact overlap between B validation and A training so
those tables cannot silently be treated as disjoint. Fixed training/validation
batch identities, candidate pools, step sizes, token/checkpoint provenance,
GPU numerical resolution and the Tier 3 gate still require qualification before
scientific probes. No E56/E57 result follows from these preparation checks.

`reference_handoff.py` provides one bounded unattended handoff for the live A
reference. Prepare it while the registered control driver, training launcher
and checkpoint protector are all live, then invoke `run` once. The coordinator
uses CPU only and records each process's PID, start ticks, command, owner and
working directory, plus the host boot identity. It waits for those exact
processes to finish; missing or reused PIDs never trigger a training restart.
It pins the specification, primary plan, job/configuration, runtime packages and
Python source files, and stops if they change. Report-only commits are allowed.

After all 15 protected weights and the original marker restoration verify, it
checks training completion. If only managed postprocessing was interrupted by
the retention hold, the existing retention verifier must pass every rank,
horizon, prediction and weight check before completing that postprocessing.
It then runs the protected A contrastive audit on GPUs0,2,5,6 and verifies the
30-export receipt. Only afterward does it start one fresh next job selected by
the existing frozen-grid driver. Its completion means that the next training
launcher was observed live, not that this fit or the study is complete.

Child commands/logs/PIDs and coordinator status live under
`controlled/reference_handoff/`. A nonzero child exit, partial audit, existing
next-run directory, missing completion evidence or changed provenance stops
the handoff for inspection; it never retries a failed or partial fit. The
coordinator cannot be reinvoked blindly after an interrupted invocation.
Its children have independent process sessions and recorded handles, so inspect
those handles after any coordinator interruption before launching other work.
The coordinator's CPU-only environment is explicitly removed from GPU children;
the next scientific training run uses online W&B with no inherited run ID.

`control_queue.py` provides a separate, bounded continuation after the matched
B reference `full__B__seed42__lr1p0`. It queues exactly the four already registered
full-arm tuning fits, in order: A/B at LR multiplier 0.3, then A/B at multiplier 3.
It stops before confirmation runs or postselection; this adds no scientific
settings or LambdaMART fits. Prepare it with the live B `--driver-pid` and
`--training-pid`, then invoke `run` once. Its CPU coordinator requires
`CUDA_VISIBLE_DEVICES=''`, `ACCELERATE_USE_CPU=false` and `WANDB_MODE=online`;
the existing driver exposes only the four authorized UUIDs to its children.

The queue pins all six original process identities, the host boot, registered
jobs/configs, specification, qualified training sources and runtime packages.
It waits for the original driver, launcher and ranks to terminate, then verifies
the predecessor's completion receipt, including allocation exports and retention.
Every successor uses the existing driver under the global training lock, with
the same baseline/data/runtime checks. It requires both physical space and
remaining account quota of at least 30 GiB on the archive and 4 GiB on home before
each fit. Busy GPUs, lock contention and low storage wait; source changes,
missing completion, nonzero child exits and existing outputs stop for inspection.
No partial fit or interrupted coordinator is automatically retried. Receipts and
live status are under `controlled/full_tuning_queue/`; completion of this finite
queue does not establish scientific experiment or study completion.

`native_analysis.py --job-id full__A__seed42__lr1p0` incrementally audits
completed native validation and fixed training-probe snapshots on CPU while
training continues. It verifies row identities, labels, dataset fingerprints,
source and snapshot hashes, and exact native headline metric parity before
reporting alternative metrics, noise bands, scorer agreement, validation-only
temperature fits, scaffold strata and fixed composition standardization.
The exact production training-probe selection is reconstructed from the cache;
its statistics describe the selected eligible assays, not all training data.
Training source/token provenance remains the registered launch: the audit
checks every consumed identity and raw label but does not read or rehash token
payloads. Test predictions are excluded from this command.

Results live in `tier1/native_trajectories/<job>/<software-signature>/`;
`latest.json` identifies the currently analysed contiguous epoch prefix. Re-run
the command after later epochs finish. Immutable snapshot analyses are reused
only after provenance and output checksum verification. Observing all epochs
does not itself establish training completion, final checkpoint selection,
test performance or multi-seed conclusions. Directional contrastive losses,
fixed-temperature inference and geometry still need the protected epoch weights.

`verify_native_analysis.py --directory <analysis-directory> --previous-epoch N`
independently verifies the next observed prefix and writes immutable
`trajectory_through_epochNN.json` and `verification_through_epochNN.json`.
Omit `--previous-epoch` for a first verification. The optional predecessor must
already contain checks for both partitions of every earlier epoch; its snapshot
records must be identical, and every old prediction/table checksum is checked
again before numerical checks are inherited. New snapshots use average-rank
dot products for every assay/scorer, explicit pair enumeration in one bounded
assay per available size bin, and direct composition-weighted means. This
verifier does not independently reconstruct bootstrap intervals, temperature
fits or all NDCG values. It shares the analysis lock, reads no test predictions
or model weights, and requires a CPU-only Atum process.

`stratified_trajectory.py --a-trajectory <immutable-A-trajectory.json>` answers
the scalar part of Q11 on CPU. It uses saved validation scores for all fifteen
initial B epochs and an independently verified A prefix. Fixed whole-assay
size bins and binary scaffold-support classes carry row-weighted BCE, assay
macro BCE, AUROC and eligible macro Spearman; paired endpoint changes resample
whole units, including units outside each stratum. The earliest observed BCE
minimum and ranking maximum are descriptive endpoints, not new checkpoint
selection. Intervals condition on those endpoints and omit selection/seed
uncertainty and multiple-stratum adjustment. A prefix is not a completed
trajectory; B has no shared-scaffold groups under this relation. Listwise and
contrastive loss localization remain separate. Results and exact source hashes
are saved in `tier1/stratified_trajectories/`; no test predictions are read.

`ranking_localization.py --a-trajectory <immutable-A-trajectory.json>` extends
Q11 to the reference's actual listwise ranking objective. It preserves native
prediction row order and the three deterministic complete-coverage partitions.
Lists retain their original full-assay size/scaffold labels and their native
weight, including zero-loss lists. A bounded vectorized implementation is
checked against the production loss, and every scientific epoch must reproduce
its saved native loss within 1e-5. Whole-unit endpoint intervals are descriptive
and conditional on the observed endpoints. This command accepts only the fixed
reference objective/normalization; it does not reinterpret controlled loss arms.
It reads no test scores or model weights. Results live in
`tier1/ranking_localization/`; contrastive-loss localization remains separate.

After both reference trajectories reach 15 epochs, run
`verify_localization.py --root /home/atabey/Prot2Mol --a-trajectory <immutable-A-trajectory.json> --scalar <scalar-output-directory> --ranking <ranking-output-directory>`
with `CUDA_VISIBLE_DEVICES=''`. It checks provenance and every scalar group
correlation, BCE sum, stratum reduction and rank-sum AUROC against the saved
validation rows. It also reconstructs all native list descriptors and checks
production listwise loss on samples from each partition and assay-size bin.
Both output directories receive `verification.json` only after every epoch
passes; existing receipts are never overwritten. Bootstrap intervals and
endpoint-selection uncertainty are outside this independent verification.
It performs no model inference, fitting or test-set evaluation.

`contrastive_localization.py --regime B` attributes both contrastive directions
to the original assay-size and scaffold strata using the existing fifteen
validation embedding exports. It keeps the original candidate batches intact,
including all negative candidates, masks, strict-active cutoff, query weights
and zero-active lists. Stratum means are query-origin attribution, not scores
against a new stratum-only candidate pool. Learned-scale and scale-13 values
must reproduce the previous directional audit, and sampled batches are checked
directly against the production loss. Endpoints include the original native
validation-macro peak, not a hard-coded epoch. No confidence intervals are
claimed: simply resampling query contributions would hold cross-unit candidate
contexts fixed. After the protected A embedding audit completes, the same
command accepts `--regime A --a-reference <reference-contrastive-complete.json>`.
Outputs live in `tier1/contrastive_localization/<regime>/`; no fitting, model
inference or test scoring occurs.

After the matched A reference has finished training and retention,
`reference_contrastive.py run --workers 4` completes its remaining checkpoint
audit before the next controlled fit. It requires all fifteen verified protected
epochs, uses the shared GPU lock, and assigns one of the authorized GPU UUIDs to
each worker. Every epoch is evaluated on validation and the exact original
fixed training probe; it never reads a test partition or fits any parameters.
All token payloads are checked against the frozen training plan before launch.

The export preserves normalized embeddings, quantifies differences from native
four-rank scalar predictions, and recomputes both contrastive directions on the
same three complete-coverage candidate partitions at the learned scale and at
the initial scale13. It uses the original B audit's material-drift limits
(0.01 contrastive loss and 0.001 macro Spearman); differences are reported even
when below those limits. Candidate-pool hashes must match across all epochs.
The command is restricted to the original reference mask, threshold and seed,
so controlled masking variants cannot silently reuse the wrong loss policy.
Outputs live in `tier1/reference_contrastive/`. Local tiny-model checks and a
refusal to run before training completes are runtime qualification only;
the thirty scientific exports and their loss trajectories must still execute.

The reference run recorded W&B 0.22.3, Torch 2.13.0+cu129, Transformers 5.5.0,
Datasets 2.21.0, and Accelerate 1.10.1. Atum matched the training libraries but
had W&B 0.17.6, which failed during server-settings introspection before any
optimizer update. `runtime-requirements.txt` pins the reference logging version.
