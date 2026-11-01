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
