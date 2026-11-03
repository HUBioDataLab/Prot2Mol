# Requested execution scope — 2026-11-03

The user's latest instructions override the MD's seed and tuning budgets:
execute E34–E36 and E44–E47, then E49–E58; also finish E01–E05,
E08–E09 and E19–E20. Use the single best existing seed and no parameter
optimization. Reuse completed full-model references rather than repeating them.

Seed42 is selected from the completed base-LR B validation histories:
seed42 peaks at epoch6, macro Spearman 0.18457785491707168;
seed43 peaks at epoch5, 0.1773860175233056. No test metric enters the choice.
This is prior validation exposure, not an independent seed confirmation.
Bootstrap intervals condition on the chosen training seed and configuration.

| Experiments | Fixed training variants |
| --- | --- |
| E34 | Frozen encoders |
| E35 | Last two encoder blocks unfrozen |
| E36 | LoRA ranks 4, 16, 64 |
| E44 | Contrastive gradient detached from encoders |
| E45 | Per-group listwise normalization; assay-balanced sampling |
| E46 | Top-weighted pairs; uniform pairs |
| E47 | Zero-margin hard pair targets; replicate-noise soft pair targets |

Each of these 12 variants gets one A fit and one B fit: 24 new fits total.
All use seed42, original base learning rates, 15 epochs, batch 12×4×4,
and the original remaining configuration. Select checkpoints by maximum native
validation macro Spearman. The two completed full A/B42 references are reused.
Only GPUs 0, 2, 5 and 6 are authorized. LoRA ranks and pair-target controls
are the requested comparisons, without an additional hyperparameter search.

`requested_single_seed.py prepare` verifies the original configurations,
completed references, original specification, existing four-GPU qualification,
and receipts proving the old queues stopped. `run` executes only those 24
configurations through the existing qualified native driver. It acquires the
shared GPU driver lock, checks resources, and stops for inspection on partial
or failed work. It neither traverses the former tuning grid nor creates new
confirmation seeds. Its training completion does not mean the experiments or
selected-test comparisons are complete.

`requested_evaluation.py` evaluates only the best validation-macro checkpoint
of each of the 26 requested/reference fits. It does not require or create other
learning-rate candidates. Each fixed-model worker uses one authorized GPU;
up to four completed models can be evaluated together under the shared training
lock. Existing complete native validation output from full A42 is reusable
after checking its original plan, weights, data and receipt. Its interrupted
test output remains preserved; the amended evaluation uses a separate path.
The worker stops on partial output in its own path. Local checks cover actual
tiny-model full, frozen, LoRA and training-known-positive inference, without
claiming scientific GPU results.

`requested_results.py` applies the existing native metric, noise-band,
similarity and composition diagnostics to the selected validation/test
predictions and checks all 15 saved native validation/training-probe epochs.
It reports all three scoring pathways. Across the 13 requested/reference
settings it retains every A/B point, matched variant-minus-full effects, and
the separate uniform/top and soft/hard contrasts. E58's plot includes the
common-defined-eligible-assay sensitivity within each split. The seed and
configuration are fixed; no seed variance, test-selected winner or causal
interpretation of A/B association is reported. Adaptation, geometry, gradients,
transference and allocation comparisons remain separate required work.

`requested_probe_protocol.py` narrows the existing E49–E52 and E56–E57
protocols to seed42 and resolves the two full-reference base checkpoints.
It preserves the frozen task/group memberships, k values, query cohorts,
optimizer and step rules. E54–E55 use the same two reference models on all six
own-regime train/validation/test geometry cohorts. The amended Tier 3 gate
requires the 24 requested training completions and all 26 selected analyses.
Preparing the protocol does not qualify or start scientific probe execution.

The existing adaptation design contains 22 available kind/scope/k cells.
With one seed, it requires 8,088 short support fits: 24 across-unit fits and
8,064 within-unit fits. These update the heads, or heads plus cross-attention,
under fixed settings; the encoders remain frozen. Each feasible scaffold unit
has its own support fit. A within-assay adaptation is structurally unavailable,
and unavailable large-k cohorts stay explicitly unavailable. No new k budget,
learning rate, seed, task or unit is introduced by this amendment.

The verified full-reference state dictionaries contain 6,898,572 raw bytes for
each head-only snapshot and 13,198,220 bytes for heads plus cross-attention.
Retaining all 15 B-validation snapshots and at most selected/final snapshots
elsewhere requires at most 51,484 adapter snapshots, approximately 482 GiB of
raw tensors. Metadata, predictions, logs and main-model checkpoints are extra.
This is a storage estimate, not a reservation or a runtime estimate; the GPU
execution stage must check actual quota. Existing checkpoints remain preserved.

E49–E58 retain the MD's adaptation, pathway, geometry, gradient, transference
and A/B variant-comparison requirements, under the same single-seed amendment.
E01–E05, E08–E09 and E19–E20 reuse existing data/metric evidence and complete
the selected prediction-based comparisons. Their execution and reporting must
also be adapted before use; this training queue does not run them.

`requested_reporting_queue.py` binds to the exact live requested-training
controller and waits for all 24 native training completions. Its waiting checks
run every ten minutes without loading models or exposing GPUs. It then calls
the fixed-checkpoint evaluator in batches of at most four models under the
shared GPU lock, analyzes their saved predictions, and writes the 26-fit
comparison and Tier 3 handoff. Process exit alone never counts as completion.
Failures stop the controller for inspection; it does not restart failed jobs
or invoke a legacy search queue. This handoff leaves adaptation, geometry,
gradient/transference probes and remaining audit work explicitly pending.

`requested_geometry.py` runs E54/E55 on the two full-reference best-macro
checkpoints after that Tier 3 handoff. Two independent workers export the six
frozen train/validation/test cohorts under the shared GPU lock. They verify
source data, token payloads, cohort identity and native projection-score parity,
and preserve the model weights. Both pooled and projected representations,
observation and unique-entity views, and active/all-pair alignment are retained.
Preparing or qualifying this runner does not count as scientific geometry.

`requested_objectives.py` runs E56/E57 behind the same Tier 3 prerequisite.
It binds the two full-reference checkpoints to all 64 frozen batch pairs and
the original nominal component rates. Before each model's scientific pass,
two measurements of its first frozen pair check float32 forwards, disabled
TF32/autocast, repeatability, effective steps, memory use and weight preservation.
The fixed numerical check records small transfer sign changes and observed
precision limits; it never chooses new step sizes or excludes pairs. Failed
qualification stops execution with its evidence retained. The subsequent
measurements preserve all per-block norms/cosines, transfer values and undefined
states, with no fitting, test-data access or independent-batch uncertainty claim.

`requested_adaptation_tasks.py` enumerates every support fit and deduplicates
unchanged k=0 baselines only when their query views are identical. Its task
runtime resets the allowed head/fusion parameters to the same base after each
fit and rejects changes in frozen state. Query verification checks the exact
base-plus-adapter parameter digest, source rows and labels, all three pathway
scores, independent SciPy correlations and complete epoch-selection records.
The scheduler still owns the Tier 3 gate, B-validation epoch choice, assigned
GPUs, production qualification and aggregate reporting; this task library does
not choose epochs or launch the adaptation inventory.

The user stopped the old full-checkpoint and E33 queues before this amendment.
Keep all previous receipts, registrations and partial outputs. Do not restart
legacy one-, two- or three-seed queues or launch LR, weight-decay, loss-weight
or seed searches. The historical metadata field `phase=tuning` is preserved
only to identify the original configuration bytes. E33 and other unrequested
experiments remain outside the active scope, not scientifically completed.

Code is developed locally, pushed on `codex/paired-ranking-generalization`,
then pulled on Atum. No implementation source is edited on the server.
