# Paired input ranking generalization

The governing specification is `/Paired input ranking generalization.md`. Do not
expand the scientific scope. Answer its 34 questions and execute its suggested
experiments, distinguishing required comparisons, conditional interventions,
inapplicable methods, and missing evidence.

User-approved execution constraints:

- Remote work only inside `/home/atabey/Prot2Mol` on Atum.
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

The source is the available 1,957,400-row project-local `all.parquet`. Its hash is
recorded. The original run's data hashes are unavailable, so rebuilt split
identity is not claimed. Scaffold components preserve assays; target coverage,
scaffold disjointness, and pair disjointness are checked before training. The
original median label transformation is preserved for this baseline.

Execution order: prepare data; validate four-GPU training/evaluation/checkpoint
loading with a two-step smoke run; complete the 15-epoch baseline; then perform
the specification's Tier 1 audits, Tier 2 baselines, Tier 3 controlled retraining,
Tier 4 mechanistic probes, and supported conditional interventions. Tier 3 uses
both splits, equal tuning budgets and at least three seeds; additional Split B
realizations are subject to the specification's affordability condition.

Runtime artifacts and logs live in `RewardModel/outputs/paired_generalization`.
Scientific experiments are not complete merely because preparation or the
distributed smoke check passes.
