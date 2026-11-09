# Paired-input ranking generalization: stopped-study report

Snapshot: 2026-11-09. **Experiments stopped at the user's request; the study is not complete.** This report follows the
[34 questions and experiment specification](<../../../Paired input ranking generalization.md>).
The [evidence ledger](protocol.json) records every question and all 70 experiment
entries, including 12 conditional interventions. Artifact locations, hashes and
qualification limits are in [detailed findings](INTERIM_FINDINGS.md).

**Verified stop:** Atum was reachable on November 9. Two process inventories
found no project-owned training or controller processes; no termination signal
was needed. Training had stopped on the assay-balanced A evaluation failure,
and the reporting/probe controllers had stopped without completing any of their
queued scientific jobs. The persistent goal is paused. No experiment was
launched, resumed or rerun for this report. Existing logs, checkpoints and
prediction files were inspected. The server remains at `003cdca`; the evaluation
fix is pushed but was not deployed. The server stop receipt is
`RewardModel/outputs/paired_generalization/protocol_amendments/user_stop_20261109/stop_verified.json`,
SHA256 `1a61aaaa6b5353956903e178281b985ab6e1c536f34715c154429ba481a213cc`.

**Completed training inventory:** 23 full 15-epoch neural fits: the initial
fresh scaffold baseline, eight full-model reference/LR/seed fits, and fourteen
requested fixed-seed variant fits. This excludes small runtime checks and the
non-neural baseline fits. Of the requested 24 variants, **14 completed, one
failed, and nine never started**. The earlier seed44 A attempt was cancelled
and excluded. A completed fit does not mean its selected-test comparison is
complete. The initial fresh B baseline has a selected test result; the narrowed
26-model selected-test comparison is unfinished.

**Consolidated fixed-seed results:** Best native validation macro Spearman,
with each run's selected epoch in parentheses. A is the random-assay split;
B is the scaffold-component split. A and B evaluate different populations.

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

**Data and metric work completed:** ChEMBL37 was downloaded and checked, and
all 1,957,400 curated observations were reconstructed from 2,032,748 retained
measurements with zero count/median mismatches. Scaffold-component splitting,
tokenization, partition-local labels, supervision eligibility, protein support,
replicate disagreement, contrastive masks, frequency skew, metric precision,
loss/readout trajectories, composition and distribution shift were audited.
The original random-split pair-median procedure used held-out measurements to
change 5,289 training labels; controlled fits use partition-local labels.
Original historical split identity could not be recovered exactly, so this is
not a byte-identical reconstruction of the original `m5guoqkx` dataset.
The 34-question table below states the findings and their limits individually.

**Completed inexpensive baselines:** Test macro Spearman as recorded in the
baseline audit; lookup/identity methods have restricted or undefined coverage.
The protein-only BCE baseline is reported separately because it gives constant
within-assay scores (test AUROC A/B: 0.6696/0.6493).

| Baseline | A test | B test |
| --- | ---: | ---: |
| Molecule-only ridge | 0.0775 | 0.0624 |
| Same-protein nearest-scaffold lookup | 0.2572 | Undefined |
| Independent per-protein ridge | 0.2419 | 0.0726 |
| Two-step ridge | 0.1034 | 0.0488 |
| Identity matrix filter | 0.4207 | Undefined |
| Paired Kronecker ridge | 0.1690 | 0.0631 |
| Protein-shuffled Kronecker ridge | 0.0662 | 0.0596 |
| Bilinear inductive completion | 0.2126 | 0.0779 |
| User-fixed depth-3 LambdaMART | 0.1285 | 0.0607 |

Undefined B scores mean those methods cannot order its within-assay molecules;
they are not zero correlations. Lookup and identity-filter A coverage is
79.72% and 93.32%, respectively. The jointly trained initial-B model's cosine
readout scores 0.06623 on test versus 0.17688 for its ranking head, on the same
3,972 eligible assays. This is a readout comparison, not contrastive-only training.
All eleven Tier-2 ledger entries are complete; full details, coverage and
conditional intervals are in the linked baseline ladder in `INTERIM_FINDINGS.md`.

**Work prepared but not completed:** The selected-checkpoint reporting,
adaptation, geometry, gradient-cosine, lookahead-transfer and A/B variant-analysis
code and fixed protocols exist, but the requested scientific results were not
produced. The separate 10,000-pair SELFormer/MoLFormer/ProtT5 embedding comparison
has no completed result in this study. Earlier HPO, further seeds, loss-subset
searches and additional LambdaMART runs were excluded by user amendments.
The ten unfinished requested fits and remaining analyses are stopped, not queued
for continuation. Current verification is recorded in
[the stop evidence](evidence/user_stop_20261109.json).

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

**Training results:** The requested fresh 15-epoch scaffold baseline is complete.
Its best validation-macro checkpoint is epoch 6: validation 0.18460 and test 0.17688
([run 4d1vkheo](https://wandb.ai/hubiodatascilab/Prot2Mol_RewardModel/runs/4d1vkheo)).
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
([run khvzu281](https://wandb.ai/hubiodatascilab/Prot2Mol_RewardModel/runs/khvzu281)).
It was preserved without retraining. Frozen B42 also completed all 15 epochs:
best validation macro **0.14643 at epoch6**, final **0.13679**
([run h2nhxuqv](https://wandb.ai/hubiodatascilab/Prot2Mol_RewardModel/runs/h2nhxuqv)).
Last-two-blocks A42 completed 15 epochs with best/final macro **0.34153 at
epoch15** ([run 7cjkhhsw](https://wandb.ai/hubiodatascilab/Prot2Mol_RewardModel/runs/7cjkhhsw)).
Last-two-blocks B42 completed 15 epochs with best macro **0.16670 at epoch8**
and final **0.15388** ([run zba1gr0v](https://wandb.ai/hubiodatascilab/Prot2Mol_RewardModel/runs/zba1gr0v)).
LoRA rank4 A42 completed all 15 epochs with best/final macro **0.31065 at
epoch15**
([run 8tbg64mm](https://wandb.ai/hubiodatascilab/Prot2Mol_RewardModel/runs/8tbg64mm)).
LoRA rank4 B42 completed 15 epochs with best macro **0.17134 at epoch11**
and final **0.17124**
([run igxpwkjs](https://wandb.ai/hubiodatascilab/Prot2Mol_RewardModel/runs/igxpwkjs)).
LoRA rank16 A42 completed all 15 epochs with best/final macro **0.31663 at
epoch15**
([run sqj6wwrw](https://wandb.ai/hubiodatascilab/Prot2Mol_RewardModel/runs/sqj6wwrw)).
LoRA rank16 B42 completed all 15 epochs with best macro **0.17703 at epoch11**
and final **0.16467**
([run nio1y8kl](https://wandb.ai/hubiodatascilab/Prot2Mol_RewardModel/runs/nio1y8kl)).
LoRA rank64 A42 completed all 15 epochs with best/final macro **0.32331 at
epoch15**
([run 9wbd28xl](https://wandb.ai/hubiodatascilab/Prot2Mol_RewardModel/runs/9wbd28xl)).
LoRA rank64 B42 completed all 15 epochs with best macro **0.16879 at epoch8**
and final **0.16353**
([run 8r1q7fsj](https://wandb.ai/hubiodatascilab/Prot2Mol_RewardModel/runs/8r1q7fsj)).
Contrastive-detachment A42 completed all 15 epochs with best macro **0.36443
at epoch14** and final **0.36438**
([run ivwxt7ns](https://wandb.ai/hubiodatascilab/Prot2Mol_RewardModel/runs/ivwxt7ns)).
Contrastive-detachment B42 completed all 15 epochs with best macro **0.17571
at epoch7** and final **0.16708**
([run fhl9hwyh](https://wandb.ai/hubiodatascilab/Prot2Mol_RewardModel/runs/fhl9hwyh)).
Per-group listwise normalization A42 completed all 15 epochs with best/final
macro **0.33629 at epoch15**
([run 35lhbwgg](https://wandb.ai/hubiodatascilab/Prot2Mol_RewardModel/runs/35lhbwgg)).
All fourteen retained best-macro checkpoints are verified. Per-group listwise
normalization B42 completed 15 epochs on GPUs 0, 2, 5 and 6
([run djxcgvr9](https://wandb.ai/hubiodatascilab/Prot2Mol_RewardModel/runs/djxcgvr9)).
**November 9 verification:** Normalized B's native 15-epoch completion,
best/final checkpoint bytes and 31 prediction manifests are verified on Atum.
Its exact W&B history peaks at **0.17121 at epoch5** and ends at **0.16224 at
epoch15**; the subsequent best-score reevaluation is excluded from that final
epoch value. Both are below full B. The E45 section below records the evidence.
The next fit, assay-balanced A
([run zzskjup5](https://wandb.ai/hubiodatascilab/Prot2Mol_RewardModel/runs/zzskjup5)),
failed at its first evaluation: prediction batches passed ranking labels into
the model without the sampling weights required by assay-balanced training.
Its W&B `output.log` records `ValueError: Assay-balanced training requires
source-assay weights and batch item count` (SHA256
`645bdc088b1b07539d04ef6f4b19c76f1a1618c0519d2ba9253199b5098322b5`).
The failure was reproduced in a local CPU training/evaluation test. A prepared
fix makes evaluation forwards prediction-only; the existing complete-cohort
evaluation still computes all losses and assay weights. All 99 targeted CPU
tests pass, including checkpoint saving, both validation and fixed-probe
exports, unequal-assay loss reduction and exact prediction/embedding parity.
These tests use small encoders and do not constitute Atum GPU validation.
The tested fix was pushed as `fe5fb5d` on `codex/paired-ranking-generalization`.
It has not been deployed to Atum, and recovery has not been launched.
Atum access is restored. The failed attempt has no saved checkpoint directory;
the user stopped the experiments before any recovery or restart.
The selected-checkpoint evaluation and comparison reports are now implemented
for this fixed scope; scientific selected-test results are not yet available.
Preparation and local scheduling tests add no scientific completion credit.
The complete historical ledger is retained; unrequested experiments are not
silently marked complete. See [the current scope](REQUESTED_SINGLE_SEED.md).
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
defined eligible assays except frozen and last-two-blocks epoch3, LoRA rank4
epochs4/9/13 and LoRA rank64 epochs6/11 (7,996);
final and peak coverage agree. B validation has 3,969 defined eligible assays
throughout full, frozen, last-two-blocks, rank4 and rank64 fits. Rank16 B has 3,968 at
epoch7 and 3,966 at epoch12, otherwise 3,969, including its peak and final
epochs. B's probe has 2,047 defined eligible assays throughout all fits except
rank64 epochs2/5/9 (2,046); its peak and final coverage agree.
Selected-test comparisons
remain pending, so E34–E36 are not complete. Evidence under
`requested_interim/` on Atum:

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
labels. A coverage is 7,997/2,048 defined eligible assays throughout. B has
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
remain pending. Evidence under `requested_interim/` on Atum:

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
7,997/2,048 defined eligible assays except normalized validation at epoch14
(7,996); peak and final coverage agree with full A. These are descriptive
single-seed native metrics, not independent recalculations. Both assay-balanced
fits and selected-test comparisons are unfinished, so E45 is not complete. Evidence:
`requested_interim/listwise_normalized_A_seed42_native_20261106.json`, SHA256
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
(136,957/43,115 rows). Defined validation/probe assay counts are 3,969/2,047,
except normalized validation epochs1/11 (3,968) and probe epoch2 (2,046).
Best/final coverage matches the reference. Final probe macro is 0.50577 versus
full B's 0.51969. Retained root/best weights have SHA256
`74010b0e999de17864ba235b1deeba2be04931163cbb34524411c7c8db045922`;
final weights have SHA256
`477e62b58f01b1a9a40ece76e8ff123ca359231799c0c12143d648109b827e8f`.
Local native-comparison evidence:
`artifacts/paired_generalization/user_stop_20261109/normalized_B_native_comparison_v2.json`,
SHA256 `f454c8c52cc4af411d2d66dd1230aa89af49aa98681ce9f5973a7a8a365b5a25`.
Earlier W&B evidence: `artifacts/paired_generalization/reconnect_20261107/normalized_B_wandb_comparison.json`,
SHA256 `fa0281b5874942e1264c116fbec0fa6e1f4a3d75a8d1de0238947fd533677f04`;
normalized/full W&B history SHA256 respectively
`286bf71da91056ce2d08f3c5b5a531e4a08bf2daf28a79a2bfe6ce81cc8ad381` /
`f72bb6c24f1e435ae3a2ae14675f5d3684fe9f209b2f3357b43111f4ef7f6c22`.

**Reference labels below:** “Initial B” means the fresh scaffold baseline above.
“Controlled A/B” means the separate base-LR seed42 reference fits. Their results
are kept distinct. Deployment is unseen generated molecules; unseen identity
does not imply an unseen scaffold.

| Question | Status | Current answer and limit |
| --- | --- | --- |
| **Q01 — Which Spearman?** | Answered | Checkpoint selection uses the unweighted mean of eligible, defined per-assay correlations. `eval_spearman` is instead observation-count weighted. Neither is a pooled correlation. |
| **Q02 — Eligibility, constants and ties?** | Answered | Headline eligibility is n≥3 and label span≥0.5, using float32 labels and average ranks. Undefined correlations are excluded with coverage reported. Eligible test assays: A 50.38%, B 41.47%. Float64 export arithmetic changes one B eligibility decision. |
| **Q03 — Different test composition?** | Answered | Yes. Median assay sizes are A 3/B 2; mean within-assay variance is 0.3046/0.1809. Replicate counts and tie distributions also differ. These differences do not establish their causal contribution to the performance gap. |
| **Q04 — Seed, bootstrap and split spread?** | Partial | Whole-unit intervals exist conditional on fitted models. Original splitter seeds 42/43/44 give identical assignments. Two dependent held-out reallocations preserve training support but have no selected-performance result. Additional seeds and allocation searches are outside the current requested scope. E20 will report conditional whole-unit uncertainty for the selected fixed-seed models; the A LR comparison does not measure seed variance. |
| **Q05 — Noise-tolerant and top-weighted metrics?** | With limits | Spearman has no noise band. Initial-B test ordering accuracy changes 0.5892→0.6581 when pairs inside the training replicate-RMS band are excluded, changing the denominator. Controlled-B NDCG@1/3/5 peaks at 6/13/8; all finish below their epoch 6 values. Different metric optima are descriptive, not proof of reproducible objective conflict. |
| **Q06 — Which loss rises first?** | With limits | Initial-B BCE has minimum 2, ranking minimum 5, and learned molecule→protein contrastive minimum 5; the total contrastive minimum is 15. BCE can worsen while AUROC improves. A ranking and both contrastive directions decrease overall throughout 15 epochs. These single-run extrema are not statistical change points. |
| **Q07 — Loss or Spearman checkpoint?** | With limits | Controlled A/B select validation macro Spearman at 14/6. Minimum aggregate loss is 15 on both. The different checkpoint locations are established. Requested evaluation uses only best-macro checkpoints; comparing alternative selection rules and adding seed variation are outside the narrowed scope. |
| **Q08 — Training versus validation?** | With limits | Controlled B 6→15 probe macro improves 0.39862→0.51969 while validation changes 0.18458→0.17975. Final probe/validation values are A 0.49208/0.34332 and B 0.51969/0.17975. These are distinct fixed eligible-assay probes, not full-training or matched causal gap estimates. |
| **Q09 — Frozen-encoder behavior?** | With limits | Yes, the early ranking-loss rise survives on B: frozen minimum at epoch2, increasing every epoch thereafter; full B minimum at epoch3. Frozen B best macro is 0.14643 versus full 0.18458, with a larger final probe/validation gap. A ranking minima are frozen12/full15. Native curves and matched rows/labels are verified; no causal mechanism, independent replication or selected-test benefit is established. |
| **Q10 — Temperature and fixed candidates?** | With limits | Learned scales sharpen on A and initial B; candidate pools stay fixed across epochs. Fixed-scale13 rescoring raises A's final contrastive loss but lowers initial B's. Neither fresh reference shows a sustained overall contrastive-loss rise. Rescoring does not prove fixed-temperature training helps. |
| **Q11 — Which sizes or similarity strata?** | With limits | In initial B 6→15, >25-item assays account for 80.2% of net ranking-loss increase, while three-item assays account for 76.1% of net macro decline. A cold-scaffold strata differ from shared strata. Localization is descriptive and conditional on these fitted trajectories. |
| **Q12 — Deployment regime?** | Answered | Deployment is unseen generated molecules. Report unseen identity and unseen scaffold separately; the generator's future scaffold mixture is unknown. Neither split alone is established as an exact deployment match. |
| **Q13 — Similarity distribution and stratified quality?** | Partial | Binary scaffold support is measured:19.29% of A test rows have unseen scaffolds, versus all B rows. Depth3 A test rho is 0.1342 on shared-scaffold rows and0.0669 on unseen rows, versus B 0.0607; eligibility is reapplied. Restricted rows and intact assays are different cohorts. Selected fixed-seed neural test comparisons remain unfinished; additional seed or allocation comparisons are excluded. |
| **Q14 — Indivisible units and assignment?** | Answered | A splits whole assays with target coverage anchors. B splits scaffold-assay components; its largest component supplies98.70% of training rows, and target-coverage rescue components are forced into training. Assignment is not exchangeable across train and held-out units. |
| **Q15 — Distribution shift?** | With limits | Raw-source descriptor C2ST balanced accuracy is about 0.85–0.86 for B train versus held-out, versus 0.52 for A. B's training positive rate is 0.6790 versus 0.7902 in test. BBSE estimates 0.7850, but its class-conditional invariance assumption is unverified. Split constraints limit population-level interpretation. |
| **Q16 — Protein support in B test?** | With limits | All1,723 B test proteins occur in training, with1–1,303 training assays each. Initial-B test macro ranges0.0539 for one training assay to 0.1935 for>100. Low-support intervals are wide and groups differ in composition; this is not a causal support effect. |
| **Q17 — Do A/B variant scores move together?** | Pending | The required variant performance grid is incomplete. No scientific variant-association result is available. |
| **Q18 — Replicate noise ceiling?** | With limits | Split-half test macros are A 0.5445/B 0.3780; the Spearman–Brown heuristics are 0.7050/0.5485. Coverage is only 5.07%/8.76% of eligible assays. These are not established ceilings for the entire test set. |
| **Q19 — Main effect, interaction and noise shares?** | With limits | The fitted variance decomposition is unstable under assay balancing: the A interaction contribution changes 0→0.1159 and B 0.1716→0.0568 pChEMBL². No stable interaction share or architectural ceiling is established. |
| **Q20 — Pairs within replicate noise?** | Answered | Training raw-replicate RMS bands are A 0.7806/B 0.7661 pChEMBL. They contain60.80%/61.89% of test within-assay label differences. Raw-replicate disagreement is not uncertainty of the aggregated median label. |
| **Q21 — Why only about half support ranking?** | Answered | Mainly n<3, followed by insufficient label span. About49% of training assays qualify; eligible/ineligible median sizes are 11/1, with different target composition. Small float64-sampler versus float32-metric eligibility discrepancies are documented. |
| **Q22 — How were observations selected?** | With limits | Curation retains exact numeric binding measurements on both sides of the threshold. Retained assays' archived numeric measurements are already69.22% positive, with large source differences. Original testing frames and unpublished outcomes are unavailable; missing observations are not negatives or prevalence estimates for generated molecules. |
| **Q23 — Same-side labels and threshold noise?** | Answered |43.63%/54.20% of eligible A/B test assays have all binary labels on one side of 6. Also35.58%/32.21% of test rows lie within the raw-replicate RMS band around 6. BCE then lacks a within-assay label-order preference; this does not prove it cannot help shared features. |
| **Q24 — Batch construction?** | Answered | Batches mix bounded assay lists and classification-only rows. Eligible-list fractions average0.5583/0.5625 on A/B and vary substantially; each epoch-zero ordering has one local batch without a ranking-eligible list. The audit reconstructs the sampler, not historical device traces. |
| **Q25 — Contrastive false negatives?** | With limits | Own-assay, same-protein and duplicate-query exclusions already exist. Other training-known positives can remain negatives:0.530%/0.714% of audited protein→molecule negative comparisons on A/B. This is a known-positive lower bound. The additional masking experiment is outside the narrowed scope, so its performance effect remains unknown. |
| **Q26 — Projections before or after fusion?** | Answered | Contrastive projections precede cross-attention and use separate pooled encoder outputs. Ranking and BCE use fused representations. Cross-pair contrastive scores therefore use embedding products. |
| **Q27 — Ranking-loss normalization?** | Answered | The reference sums weighted Plackett–Luce selections, then averages sampled lists without normalizing by retained weight sum. Larger assays get more bounded-list opportunities, so original assays are not equally weighted. Per-group normalization reduces best validation macro on A/B to 0.33629/0.17121 versus 0.34368/0.18458. The separate assay-balanced training comparison is unfinished. |
| **Q28 — Protein-frequency skew?** | With limits | The top 1% of proteins supply 30.07%/30.64% of A/B assay-list opportunities. The reference has no log-Q correction. Skew is established; use of a learned popularity shortcut is not. |
| **Q29 — Whole-unit subsets?** | Partial | E41 views use whole assays on A and whole components on B. B's10% of units retains98.86% of rows, limiting saturation conclusions. E40 A views are qualified; strict B matching has no whole-component exchange. E40/E41 performance experiments and the proposed B exception are outside the narrowed active scope; no performance conclusion is claimed. |
| **Q30 — Loss-alone and pairwise combinations?** | Not run: user scope | No completed loss-subset comparison exists. The user excluded that grid from the active task. Full-model observations cannot identify each loss's contribution; the requested ranking variants and contrastive detachment address different comparisons. |
| **Q31 — Do scorer orders agree?** | With limits | Controlled-B ranking/BCE agreement falls0.67647→0.59235, while ranking/cosine agreement rises0.07402→0.23018. Final BCE/cosine label macros0.13944/0.07754 remain below ranking0.17975. Prediction agreement does not measure gradient conflict. |
| **Q32 — Protein-only BCE performance?** | With limits | Frozen-feature protein-only models recover about 48.9%/54.6% of excess validation AUROC above chance for A/initial B. Their scores are constant within assays. This comparison of separate predictors is not a causal fraction of the full model's signal. |
| **Q33 — Auxiliary gradient alignment?** | Pending | No scientific gradient-alignment or transfer result yet. Fixed batches and CPU implementation qualification are prepared; actual trained-checkpoint probes remain gated behind the preceding experiments. |
| **Q34 — How were original weights selected?** | Answered | The user confirmed arbitrary selection of ranking/BCE/contrastive weights 0.25/0.25/1. W&B trials alone did not establish the decision. E39's prospective weight search is excluded from the active scope. |

**Unfinished at the user stop (not scheduled):** Ten of the 24 fixed-seed fits
are incomplete: assay-balanced A failed, and assay-balanced B plus eight pairwise
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
held-out measurements; the corresponding scaffold count was zero. Controlled
runs use partition-local binary-label construction.
This correction, the constrained B split, unfinished selected-test comparisons
and the fixed-seed uncertainty limit must
remain visible in the final conclusions. Completing software or an audit
watcher does not complete a scientific experiment.
