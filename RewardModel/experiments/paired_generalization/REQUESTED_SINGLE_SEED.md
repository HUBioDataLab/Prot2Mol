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

E49–E58 retain the MD's adaptation, pathway, geometry, gradient, transference
and A/B variant-comparison requirements, under the same single-seed amendment.
E01–E05, E08–E09 and E19–E20 reuse existing data/metric evidence and complete
the selected prediction-based comparisons. Their execution and reporting must
also be adapted before use; this training queue does not run them.

The user stopped the old full-checkpoint and E33 queues before this amendment.
Keep all previous receipts, registrations and partial outputs. Do not restart
legacy one-, two- or three-seed queues or launch LR, weight-decay, loss-weight
or seed searches. The historical metadata field `phase=tuning` is preserved
only to identify the original configuration bytes. E33 and other unrequested
experiments remain outside the active scope, not scientifically completed.

Code is developed locally, pushed on `codex/paired-ranking-generalization`,
then pulled on Atum. No implementation source is edited on the server.
