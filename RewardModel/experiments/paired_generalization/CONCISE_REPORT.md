# Paired-input ranking generalization: interim evidence report

Snapshot: 2026-11-03. **The study is not complete.** This report follows the
[34 questions and experiment specification](<../../../Paired input ranking generalization.md>).
The [evidence ledger](protocol.json) records every question and all 70 experiment
entries, including 12 conditional interventions. Artifact locations, hashes and
qualification limits are in [detailed findings](INTERIM_FINDINGS.md).

**Progress:** 23 experiments complete, 12 partial and35 pending. Questions:
12 answered, 15 answered with limits, 3 partial and4 pending. “With limits” means
an observed or descriptive answer is available; it does not complete the
remaining replicated experiments. Only Tier 2 is fully complete.

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
No test scores entered this selection. The existing queue is now running A
seed43 at base LR on GPUs 0, 2, 5, 6
([run hnnyo965](https://wandb.ai/hubiodatascilab/Prot2Mol_RewardModel/runs/hnnyo965));
B seed43 and A/B seed44 follow. Confirmation results remain pending.

**Reference labels below:** “Initial B” means the fresh scaffold baseline above.
“Controlled A/B” means the separate base-LR seed42 reference fits. Their results
are kept distinct. Deployment is unseen generated molecules; unseen identity
does not imply an unseen scaffold.

| Question | Status | Current answer and limit |
| --- | --- | --- |
| **Q01 — Which Spearman?** | Answered | Checkpoint selection uses the unweighted mean of eligible, defined per-assay correlations. `eval_spearman` is instead observation-count weighted. Neither is a pooled correlation. |
| **Q02 — Eligibility, constants and ties?** | Answered | Headline eligibility is n≥3 and label span≥0.5, using float32 labels and average ranks. Undefined correlations are excluded with coverage reported. Eligible test assays: A 50.38%, B 41.47%. Float64 export arithmetic changes one B eligibility decision. |
| **Q03 — Different test composition?** | Answered | Yes. Median assay sizes are A 3/B 2; mean within-assay variance is 0.3046/0.1809. Replicate counts and tie distributions also differ. These differences do not establish their causal contribution to the performance gap. |
| **Q04 — Seed, bootstrap and split spread?** | Partial | Whole-unit intervals exist conditional on fitted models. Original splitter seeds 42/43/44 give identical assignments. Two dependent held-out reallocations preserve training support; their selected performance and training-seed spread remain pending. The A LR comparison does not measure seed variance. |
| **Q05 — Noise-tolerant and top-weighted metrics?** | With limits | Spearman has no noise band. Initial-B test ordering accuracy changes 0.5892→0.6581 when pairs inside the training replicate-RMS band are excluded, changing the denominator. Controlled-B NDCG@1/3/5 peaks at 6/13/8; all finish below their epoch 6 values. Different metric optima are descriptive, not proof of reproducible objective conflict. |
| **Q06 — Which loss rises first?** | With limits | Initial-B BCE has minimum 2, ranking minimum 5, and learned molecule→protein contrastive minimum 5; the total contrastive minimum is 15. BCE can worsen while AUROC improves. A ranking and both contrastive directions decrease overall throughout 15 epochs. These single-run extrema are not statistical change points. |
| **Q07 — Loss or Spearman checkpoint?** | With limits | Controlled A/B select validation macro Spearman at 14/6. Minimum aggregate loss is 15 on both. The different checkpoint locations are established; selected controlled test effects and seed variation remain pending. |
| **Q08 — Training versus validation?** | With limits | Controlled B 6→15 probe macro improves 0.39862→0.51969 while validation changes 0.18458→0.17975. Final probe/validation values are A 0.49208/0.34332 and B 0.51969/0.17975. These are distinct fixed eligible-assay probes, not full-training or matched causal gap estimates. |
| **Q09 — Frozen-encoder behavior?** | Pending | No completed frozen-encoder performance comparison yet. Implemented or qualified training/reporting code is not experimental evidence. |
| **Q10 — Temperature and fixed candidates?** | With limits | Learned scales sharpen on A and initial B; candidate pools stay fixed across epochs. Fixed-scale13 rescoring raises A's final contrastive loss but lowers initial B's. Neither fresh reference shows a sustained overall contrastive-loss rise. Rescoring does not prove fixed-temperature training helps. |
| **Q11 — Which sizes or similarity strata?** | With limits | In initial B 6→15, >25-item assays account for 80.2% of net ranking-loss increase, while three-item assays account for 76.1% of net macro decline. A cold-scaffold strata differ from shared strata. Localization is descriptive and conditional on these fitted trajectories. |
| **Q12 — Deployment regime?** | Answered | Deployment is unseen generated molecules. Report unseen identity and unseen scaffold separately; the generator's future scaffold mixture is unknown. Neither split alone is established as an exact deployment match. |
| **Q13 — Similarity distribution and stratified quality?** | Partial | Binary scaffold support is measured:19.29% of A test rows have unseen scaffolds, versus all B rows. Depth3 A test rho is 0.1342 on shared-scaffold rows and0.0669 on unseen rows, versus B 0.0607; eligibility is reapplied. Restricted rows and intact assays are different cohorts. Selected neural test and repeated comparisons remain unfinished. |
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
| **Q25 — Contrastive false negatives?** | With limits | Own-assay, same-protein and duplicate-query exclusions already exist. Other training-known positives can remain negatives:0.530%/0.714% of audited protein→molecule negative comparisons on A/B. This is a known-positive lower bound; masking's performance effect is pending. |
| **Q26 — Projections before or after fusion?** | Answered | Contrastive projections precede cross-attention and use separate pooled encoder outputs. Ranking and BCE use fused representations. Cross-pair contrastive scores therefore use embedding products. |
| **Q27 — Ranking-loss normalization?** | Answered | The reference sums weighted Plackett–Luce selections, then averages sampled lists without normalizing by retained weight sum. Larger assays get more bounded-list opportunities, so original assays are not equally weighted. Normalization variants' performance remains pending. |
| **Q28 — Protein-frequency skew?** | With limits | The top 1% of proteins supply 30.07%/30.64% of A/B assay-list opportunities. The reference has no log-Q correction. Skew is established; use of a learned popularity shortcut is not. |
| **Q29 — Whole-unit subsets?** | Partial | E41 views use whole assays on A and whole components on B. B's10% of units retains98.86% of rows, limiting saturation conclusions. E40 A views are qualified; strict B matching has no whole-component exchange. Performance experiments and the proposed B exception remain unresolved. |
| **Q30 — Loss-alone and pairwise combinations?** | Pending | No completed controlled comparison of the required loss subsets yet. Full-model observations cannot identify each loss's contribution. |
| **Q31 — Do scorer orders agree?** | With limits | Controlled-B ranking/BCE agreement falls0.67647→0.59235, while ranking/cosine agreement rises0.07402→0.23018. Final BCE/cosine label macros0.13944/0.07754 remain below ranking0.17975. Prediction agreement does not measure gradient conflict. |
| **Q32 — Protein-only BCE performance?** | With limits | Frozen-feature protein-only models recover about 48.9%/54.6% of excess validation AUROC above chance for A/initial B. Their scores are constant within assays. This comparison of separate predictors is not a causal fraction of the full model's signal. |
| **Q33 — Auxiliary gradient alignment?** | Pending | No scientific gradient-alignment or transfer result yet. Fixed batches and CPU implementation qualification are prepared; actual trained-checkpoint probes remain gated behind the preceding experiments. |
| **Q34 — How were original weights selected?** | Answered | The user confirmed arbitrary selection of ranking/BCE/contrastive weights 0.25/0.25/1. W&B trials alone did not establish the decision. The prospective controlled weight and A/B validation-choice comparison remains pending under E39. |

**Remaining work:** Finish the registered full-model confirmation and selected-test
evaluations, then the unfinished controlled variants, dataset experiments and
Tier 4 measurements. Conditional interventions depend on the prescribed earlier
results. The existing training queue continues; expanding it awaits the open
compute-budget decision. E40's proposed B within-component exception is also
unselected. No additional LambdaMART fits are planned; the fixed depth 3 A/B
test results0.12849/0.06071 are retained as the requested baseline evidence.

Original random-split pair-median labeling changed 5,289 training labels using
held-out measurements; the corresponding scaffold count was zero. Controlled
runs use partition-local binary-label construction.
This correction, the constrained B split and the selected-test/seed gaps must
remain visible in the final conclusions. Completing software or an audit
watcher does not complete a scientific experiment.
