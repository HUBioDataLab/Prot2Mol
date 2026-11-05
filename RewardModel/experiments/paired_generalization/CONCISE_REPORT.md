# Paired-input ranking generalization: interim evidence report

Snapshot: 2026-11-05. **The study is not complete.** This report follows the
[34 questions and experiment specification](<../../../Paired input ranking generalization.md>).
The [evidence ledger](protocol.json) records every question and all 70 experiment
entries, including 12 conditional interventions. Artifact locations, hashes and
qualification limits are in [detailed findings](INTERIM_FINDINGS.md).

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
**8/24 requested variant fits are complete.** Frozen-encoder A42 completed all
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
All eight retained best-macro checkpoints are verified. LoRA rank64 A42 was
verified training on GPUs 0, 2, 5 and 6 on November 5
([run 9wbd28xl](https://wandb.ai/hubiodatascilab/Prot2Mol_RewardModel/runs/9wbd28xl)).
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
(4/8) and output directory; rank16 A/B use rank/alpha 16/32. Across all 15 epochs,
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
| B / full encoders | 0.18458 (6) | 0.17975 | 0.51969 |
| B / frozen encoders | 0.14643 (6) | 0.13679 | 0.58219 |
| B / last two blocks | 0.16670 (8) | 0.15388 | 0.52087 |
| B / LoRA rank4 | 0.17134 (11) | 0.17124 | 0.44906 |
| B / LoRA rank16 | 0.17703 (11) | 0.16467 | 0.45937 |

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
epoch13. Rank64 A/B and selected-test results remain pending for E36; this
single-seed validation comparison does not select a LoRA rank or establish
a test-set benefit.

**Q09:** B's early validation ranking-loss rise survives freezing: its minimum
is epoch2 (2.83134), then it rises at every epoch to 3.22761 at epoch15.
Full B's ranking minimum is epoch3 (2.79424), ending at 2.92986. In A, ranking
minima are frozen12/full15. Both B fits first show a BCE increase at 2→3 and
a contrastive increase at 6→7, but total and contrastive loss still reach their
overall minima at epoch15. Thus freezing does not remove the observed ranking
pattern; these observations do not isolate a causal mechanism.

These are native metrics with verified payload hashes, not independent metric
recalculations or statistical change-point tests. A validation has 7,997
defined eligible assays except frozen and last-two-blocks epoch3 and LoRA rank4
epochs4/9/13 (7,996);
final and peak coverage agree. B validation has 3,969 defined eligible assays
throughout full, frozen, last-two-blocks and rank4 fits. Rank16 B has 3,968 at
epoch7 and 3,966 at epoch12, otherwise 3,969, including its peak and final
epochs. B's probe has 2,047 defined eligible assays in all five fits throughout.
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

These artifacts bind native trajectories, prediction payloads and configuration
hashes. No additional inference or fitting was performed for this comparison.

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
| **Q27 — Ranking-loss normalization?** | Answered | The reference sums weighted Plackett–Luce selections, then averages sampled lists without normalizing by retained weight sum. Larger assays get more bounded-list opportunities, so original assays are not equally weighted. Normalization variants' performance remains pending. |
| **Q28 — Protein-frequency skew?** | With limits | The top 1% of proteins supply 30.07%/30.64% of A/B assay-list opportunities. The reference has no log-Q correction. Skew is established; use of a learned popularity shortcut is not. |
| **Q29 — Whole-unit subsets?** | Partial | E41 views use whole assays on A and whole components on B. B's10% of units retains98.86% of rows, limiting saturation conclusions. E40 A views are qualified; strict B matching has no whole-component exchange. E40/E41 performance experiments and the proposed B exception are outside the narrowed active scope; no performance conclusion is claimed. |
| **Q30 — Loss-alone and pairwise combinations?** | Not run: user scope | No completed loss-subset comparison exists. The user excluded that grid from the active task. Full-model observations cannot identify each loss's contribution; the requested ranking variants and contrastive detachment address different comparisons. |
| **Q31 — Do scorer orders agree?** | With limits | Controlled-B ranking/BCE agreement falls0.67647→0.59235, while ranking/cosine agreement rises0.07402→0.23018. Final BCE/cosine label macros0.13944/0.07754 remain below ranking0.17975. Prediction agreement does not measure gradient conflict. |
| **Q32 — Protein-only BCE performance?** | With limits | Frozen-feature protein-only models recover about 48.9%/54.6% of excess validation AUROC above chance for A/initial B. Their scores are constant within assays. This comparison of separate predictors is not a causal fraction of the full model's signal. |
| **Q33 — Auxiliary gradient alignment?** | Pending | No scientific gradient-alignment or transfer result yet. Fixed batches and CPU implementation qualification are prepared; actual trained-checkpoint probes remain gated behind the preceding experiments. |
| **Q34 — How were original weights selected?** | Answered | The user confirmed arbitrary selection of ranking/BCE/contrastive weights 0.25/0.25/1. W&B trials alone did not establish the decision. E39's prospective weight search is excluded from the active scope. |

**Remaining active work:** Complete the 24 fixed-seed requested variant fits,
the selected-model evaluations for those fits and the two existing full references,
and the requested adaptation, geometry and gradient/transference probes. Verify
E01–E05, E08–E09, E19 and E20 against those saved predictions, then update all
34 answers with the results and explicit limits. E20 is whole-unit bootstrap
uncertainty, not an allocation search. The old E33, multi-seed and parameter-search
queues are stopped; no compute-budget or within-component exception is awaiting
approval for this active scope. Unrequested experiments remain incomplete in the
historical ledger and will not be run. No additional LambdaMART fits are planned;
the fixed depth 3 A/B test results0.12849/0.06071 remain the requested baseline evidence.

Original random-split pair-median labeling changed 5,289 training labels using
held-out measurements; the corresponding scaffold count was zero. Controlled
runs use partition-local binary-label construction.
This correction, the constrained B split, unfinished selected-test comparisons
and the fixed-seed uncertainty limit must
remain visible in the final conclusions. Completing software or an audit
watcher does not complete a scientific experiment.
