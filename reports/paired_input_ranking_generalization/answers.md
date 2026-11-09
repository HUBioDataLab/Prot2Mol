# Grouped ranking study: empirical answers

This report addresses 34 questions about a paired-input ranking benchmark, in their specified order. Query **x**, candidate **y**, comparison group **g**, scalar label **s** and candidate category are abstract mathematical roles. The classification cutoff is denoted **θ**; its recorded parameter value is retained in the numeric register. Original input types and source-specific metadata are intentionally omitted.

**The study was stopped before completion.** There are 23 completed 15-epoch neural fits: one Initial B fit, eight full-model reference, learning-rate or seed fits, and fourteen requested variants. The latest 24-variant queue finished fourteen fits; one failed and nine never started. No completed requested variant exceeds the full reference on B validation. Selected-test comparisons for the requested model grid, adaptation and mechanism analyses remain unfinished. A completed fit is not a completed experiment.

Of the 34 questions, **12 are answered, 16 have answers with limits, three are partial, two have pending requested results, and one was excluded by the user's scope amendment**. These statuses describe the available answer, not a guarantee of independent replication. Only the inexpensive-baseline tier is fully complete. No experiment was launched or resumed to prepare this report.

## Reading the evidence

This document gives the answer, measurements and interpretation limits for each question. Its accompanying files retain the detailed evidence:

- [Epoch measurements](epoch_measurements.md): all 15 epochs of all 23 completed fits, including validation losses and available fixed-training-probe metrics.
- [Anonymous model catalog](measurements/model_catalog.csv): model aliases and their experimental settings.
- [Full-precision epoch curves](measurements/epoch_curves.csv) and [fit summary](measurements/fit_summary.csv).
- [Data, noise and variance tables](data_measurements.md): six-partition measurements and both variance weighting conventions.
- [Historical measurement notes](historical_measurement_notes.md): the earlier quantitative observations, runtime qualifications and intermediate results, expressed in the same terminology. Historical plans and statuses are superseded by this report.
- [Experiment disposition](experiment_disposition.md): every E01–E70 entry, including subsequent exclusions and unfinished work.
- [Complete numeric register](measurements/all_numeric_records.csv.gz): **1,420,985 numeric, boolean or null fields from 1,786 collected records**, with [anonymous record index](measurements/evidence_index.json). The register includes repeated historical snapshots and nonidentifying runtime measurements; its rows are not independent measurements or experiments. Raw observation-level predictions and model tensors are not expanded into this register.
- [Extraction verification](verification.json) and [file checksums](report_checksums.json): measurement preservation, native-curve checks, coverage counts and shared-file integrity.

Record IDs such as **R0923** refer to the index. All registered numeric values retain source precision. Readable tables round values; missing or undefined measurements remain missing or undefined. The record index uses neutral titles and field names. The original-source mapping is kept outside the shared bundle. Record IDs do not encode original names or locations. Input-specific inclusion checks are represented by anonymous filter labels.

“Initial B” denotes the first fresh 15-epoch fit. “Controlled B” denotes the later full-model reference. They are different fits, and their trajectories must not be spliced. Unless specified otherwise, comparisons of requested variants below use seed 42, the selected base learning rate, native validation macro Spearman and each fit's own best epoch. A and B contain different evaluation populations, so their difference is not a paired treatment effect.

The common objective is **0.25 R + 0.25 B + 1 C**, where R is ranking, B is binary cross-entropy and C is bidirectional contrastive loss. The full-scale training layout uses **12 items per device × 4 accumulation steps × 4 workers**. Items can be bounded group lists or classification-only rows; this does not mean 192 observed pairs per update. The full A/B schedules have 802/877 updates per epoch and 12,030/13,155 updates over 15 epochs.

## Completed fit results

The following is validation evidence, not the unfinished selected-test comparison. All three objectives remain present. Detachment changes a gradient path; normalization changes the numerical scale of R.

| Setting | A best (epoch) | B best (epoch) | A final | B final |
| --- | --- | --- | --- | --- |
| Full reference | 0.34367521 (14) | 0.18457785 (6) | 0.34331738 | 0.17975193 |
| Frozen encoders | 0.34118048 (15) | 0.14642708 (6) | 0.34118048 | 0.13678884 |
| Last two encoder blocks | 0.34153066 (15) | 0.16669711 (8) | 0.34153066 | 0.15387883 |
| Low-rank update, rank 4 | 0.31064943 (15) | 0.17134499 (11) | 0.31064943 | 0.17124082 |
| Low-rank update, rank 16 | 0.31663178 (15) | 0.17702553 (11) | 0.31663178 | 0.16466730 |
| Low-rank update, rank 64 | 0.32330585 (15) | 0.16879311 (8) | 0.32330585 | 0.16353485 |
| Detach contrastive encoder gradients | 0.36443202 (14) | 0.17571160 (7) | 0.36437840 | 0.16707651 |
| Normalize listwise ranking | 0.33629307 (15) | 0.17121073 (5) | 0.33629307 | 0.16224005 |


The full B reference peaks at **0.18457785**. The strongest completed requested B variant is rank-16 low-rank updating at **0.17702553**, a difference of **−0.00755233**. Detachment improves A from **0.34367521 to 0.36443202** but reduces B to **0.17571160**. These observations do not establish statistical superiority or inferiority across seeds or unseen allocations.

The Initial B best checkpoint is epoch 6: validation **0.18460015**, selected test **0.17687961**. Its final validation value is **0.17561471**. Initial B is the neural fit with a completed selected-test result in this report.

## Q01 — Is the reported Spearman computed per group and averaged, weighted by group size or pair count, or pooled across all observations?

**Answered.** Checkpoint selection uses the unweighted mean of defined correlations among eligible groups. Within each group, ranks are assigned with average ranks for ties. The separately logged `eval_spearman` is the observation-count-weighted mean of group correlations. Neither is a correlation computed by pooling scores across groups. Pair-count-weighted quantities are reported separately.

For the Initial B selected test checkpoint, all three readouts are defined on the same **3,972 headline-eligible groups**. The broader all-defined group counts differ by readout:

| Readout | Eligible macro rho | All-defined macro rho | Size-weighted rho | Macro tau-b | Pair accuracy | Pair concordance | All-defined groups |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ranking | 0.17687961 | 0.15896975 | 0.20438360 | 0.12735837 | 0.58919340 | 0.16829074 | 5382 |
| BCE logit | 0.14029485 | 0.11969446 | 0.16144274 | 0.09422852 | 0.56994871 | 0.13197973 | 5389 |
| Cosine | 0.06623441 | 0.05589053 | 0.08071059 | 0.04336494 | 0.54234937 | 0.07990511 | 5397 |


Only the first numeric column uses headline eligibility. All-defined macro, size-weighted rho and macro tau-b use each readout's broader defined-group cohort. Pair metrics use the full within-group pair population. There are **11,491,969 within-group pairs**, of which **10,841,565 have unequal observed labels**. Pair concordance is `(concordant − discordant) / all within-group pairs`. Pair accuracy is `(concordant + 0.5 × prediction ties) / label-comparable pairs`. These denominators explain why the statistics differ. Source: **R0923**.

## Q02 — Which groups enter the metric? How are size-one groups, constant groups and ties created by median aggregation handled? Does the eligible fraction differ between splits?

**Answered.** Headline eligibility is **n ≥ 3 and observed-score span ≥ 0.5**, calculated through the native float32-label path. Size-one groups, insufficient-span groups and undefined correlations do not enter the headline average. Constant predictions also make a correlation undefined; coverage must accompany the score. Label and prediction ties use average ranks, rather than arbitrary tie breaking.

Eligible test coverage is **8,278/16,431 = 50.38%** on A and **3,972/9,578 = 41.47%** on B. The broader all-defined ranking cohorts contain more groups and answer a different question. An untied two-item group has correlation ±1; an untied three-item group has values {−1, −0.5, 0.5, 1}. The headline excludes the former but retains the coarse latter values.

Precision is consequential at the eligibility boundary: float64 export arithmetic gives Initial B test macro **0.1770244006 over 3,971 groups**, whereas the native float32 path gives **0.1768796098 over 3,972**. After aligning the native precision rule, **93 export/readout comparisons agreed within 10⁻¹⁴**. This is an arithmetic and cohort effect, not evidence that inference batch size changed model predictions. Sources: **R0923**, the metric-verification records and **R1598**.

## Q03 — Do the two test partitions differ in group-size distribution, within-group score variance, near-tie rate or repeat count?

**Answered.** Yes. All six partitions were audited:

| Split | Partition | Rows | Groups | Unique x IDs | Unique y IDs | Eligible groups | Median size | Mean within-group variance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | train | 1569095 | 131450 | 7873 | 882510 | 63867 | 3 | 0.304807 |
| A | val | 195618 | 16431 | 2380 | 166367 | 7997 | 3 | 0.296278 |
| A | test | 192687 | 16431 | 2421 | 162361 | 8278 | 3 | 0.304571 |
| B | train | 1683487 | 145454 | 7873 | 839318 | 72201 | 3 | 0.318857 |
| B | val | 136957 | 9280 | 1721 | 80690 | 3969 | 2 | 0.196982 |
| B | test | 136956 | 9578 | 1723 | 81140 | 3972 | 2 | 0.180864 |


In particular, A/B test median sizes are **3/2**, and mean within-group score variances are **0.30457078/0.18086350**. Repeat and near-tie distributions also differ; the complete partition tables are in [data measurements](data_measurements.md). These differences make the raw A/B averages different estimands, but do not by themselves quantify how much of the score gap they cause.

A composition-matched depth-3 baseline comparison covers **3,964/3,972 B eligible groups (99.80%)**. On common support, A macro changes **0.1281 → 0.1154** after weighting to B's composition, versus B **0.0613**. The remaining gap is **0.0541**. The A weighting change is **−0.01266**, conditional interval **[−0.01551, −0.00969]**, with effective group count **7,067**. Residual mean group sizes remain **26.11/31.93**, and score spans **2.173/1.954**; coarse matching is incomplete.

For controlled reference validation, common support covers **3,962/3,969 B groups (99.82%)**. At epoch 15, common-support A **0.34301** becomes **0.33462**, while B is **0.17925**, leaving **0.15537**. At the respective A14/B6 peaks, the values are **0.34337 → 0.33486**, versus **0.18420**, leaving **0.15066**. A's final weighting change is **−0.00839 [−0.01223, −0.00516]**. Residual sizes are **27.18/31.79**, spans **2.226/2.020**. Matching explains some of these observed differences, without eliminating them or identifying a causal effect. Sources: **R0882, R1426, R1580, R1582**.

## Q04 — How large is the spread over seeds, over a bootstrap of indivisible split units, and over alternative Split B realizations?

**Partial.** Whole-unit conditional intervals and two completed reference seeds exist; a distribution of performance across independent B allocations does not.

At base learning rate, A seed 42/43 best validation scores are **0.34367521/0.34493525**, and B scores are **0.18457785/0.17738602**. The absolute two-seed differences are **0.00126004** and **0.00719184**. Two selected peaks cannot establish a reliable seed-variance estimate. The earlier A seed-44 attempt was cancelled; further seed searches were excluded.

The Initial B ranking test macro has a whole-component interval **[0.15991, 0.19362]**. The corresponding BCE-logit interval is **[0.12415439, 0.15624835]**, and cosine interval **[0.05047744, 0.08216130]**. These are 95% percentile intervals using **2,000 whole-component draws** over **4,034 test components**, including components without an eligible group. They condition on the fitted model, selected checkpoint and realized split; they do not include training or model-selection uncertainty.

The original splitter produces identical assignments for seeds **42, 43 and 44**. Two alternative held-out reallocations preserve the training set but have no selected-performance result, and are dependent reallocations rather than independent full split realizations. No allocation-spread explanation of the A/B gap is established. See **R1786**, the bootstrap records and the allocation records in the numeric register.

## Q05 — Does the metric tolerate small score differences the way the loss does? How do top-weighted metrics move?

**With limits.** Ordinary Spearman has no repeat-noise band. On Initial B test, pair accuracy is **0.5892** without a band, **0.6305** under the fixed margin and **0.6581** after excluding label differences inside the training repeat-RMS band. All-pair concordance changes **0.1683 → 0.1471 → 0.1205**. These are changes to the scoring rule and denominator, not improvements to the model.

Initial B eligible-group NDCG@1/@3/@5 changes from **0.59687/0.66438/0.71182** at epoch 6 to **0.59098/0.66322/0.70751** at epoch 15. The first two peak at epoch 11 and the third at epoch 6. With the broader nonconstant cohort, the first two instead peak at epoch 8. Controlled B has NDCG@1/@3/@5 peaks at epochs **6/13/8**, and all three end below their epoch-6 values. Thus metric and cohort choice can change the apparent preferred epoch. This does not establish reproducible objective conflict. Sources: **R0923**, the noise-aware and trajectory records.

## Q06 — Which term of the composite validation loss rises first on each split: BCE, the ranking loss, or each direction of the contrastive loss? Does BCE AUC move with BCE log-loss?

**With limits.** Initial B BCE reaches its minimum at epoch **2: 0.44531**, ranking at **5: 2.79392**, and total contrastive and composite loss at **15: 5.83497/6.68790**. Final BCE/ranking losses are **0.48300/2.92871**. First adjacent rises occur at BCE **2→3**, ranking **3→4**, and contrastive/total **6→7**. These local increases differ from the locations of the global minima. The learned y→x contrastive direction has an early minimum at epoch 5, while the summed contrastive loss ultimately improves.

Initial B BCE AUROC rises from **0.73865 at epoch 2 to 0.77202 at epoch 15**, while its BCE log-loss worsens. Controlled B likewise has loss minima at **2/3/15** for BCE/ranking/contrastive, ending at **0.49210343/2.92985845/5.83345747**, with composite loss **6.68894768**.

Controlled A ends at AUROC **0.84629** and BCE **0.42772**. Its BCE minimum is **0.42758 at epoch 13**, with its first adjacent rise at **5→6**. Ranking and contrastive minima are at epoch 15. Final composite loss is **2.36395550**. Both contrastive directions improve overall across the run; “overall” does not assert monotonic improvement at every epoch.

Validation-fitted scalar temperature **1.29538** reduces final A BCE from **0.42772 to 0.41749**. For controlled B, temperature **1.4732** reduces it from **0.49210 to 0.46908**. A positive scalar temperature preserves score order. These are validation-fitted calibration diagnostics, not independently evaluated calibration benefits. All extrema are descriptive single-fit observations, not statistical change points. See [epoch measurements](epoch_measurements.md) and the calibration records.

## Q07 — Are checkpoints selected on aggregate loss or on validation Spearman? Where do the best-loss and best-Spearman checkpoints fall on each split?

**With limits.** The requested fresh and controlled fits select **validation macro Spearman**, not composite loss. Controlled A/B select epochs **14/6**, while their composite-loss minima are both at **15**. Initial B also selects macro epoch **6**, rather than total-loss epoch **15**. Its selected checkpoint has the completed test macro **0.17687961**.

The location difference is established; a controlled selected-test comparison of the two selection rules was not completed and was excluded from the narrowed scope. Repeatedly inspecting validation is part of this study's selection history. No prospective, untouched test result exists for the complete requested variant grid.

## Q08 — How do training and validation ranking metrics diverge over time on each split?

**With limits.** Controlled A's fixed-probe macro rises **0.16214 → 0.49208**, while validation rises **0.16488 → 0.34332** from epochs 1 to 15. Controlled B's probe rises **0.39862 → 0.51969** from epochs 6 to 15, while validation changes **0.18458 → 0.17975**. Its final probe-minus-validation difference is **0.33994**.

The A probe has **39,740 rows and 2,048 groups**; B has **43,115 rows and 2,048 groups**, drawn from only **25 indivisible components**. A/B validation have **195,618/136,957 rows** and **7,997/3,969 headline-eligible groups**. The probes are fixed eligible-group samples, not full-training evaluations or distribution-matched estimates of a causal generalization gap.

Initial B used a different, uniformly sampled-group probe: macro rises **0.18336 → 0.53189**, while validation peaks at **0.18460** and ends at **0.17561**. It must not be substituted for the controlled B probe.

Independent export checks verified **831,555 A group/readout correlations across 30 snapshots** to **10⁻¹⁴**, plus **894 sampled pair checks**; B checks covered **509,760 correlations and 900 sampled pair checks**. These checks do not independently validate every NDCG, bootstrap or calibration output. Native curves are in the epoch appendix.

## Q09 — Does the early rise survive with frozen encoders?

**With limits.** Yes, on B the ranking-loss rise survives freezing: frozen B reaches its ranking-loss minimum at **epoch 2** and increases at every subsequent epoch. Full B reaches its minimum at **epoch 3**. Frozen B's best macro is **0.14642708 at epoch 6**, versus full B **0.18457785 at epoch 6**; final macros are **0.13678884/0.17975193**. Frozen B's final probe macro is **0.58219296**, versus full B **0.51968852**, giving an even larger probe/validation separation.

On A, frozen/full ranking-loss minima are at **12/15**. Frozen/full best macros are **0.34118048/0.34367521** and final probe macros **0.55463288/0.49207621**. Matched row identities, labels and native curves were verified. Freezing does not remove the B behavior, but these comparisons do not isolate a causal mechanism or provide a selected-test or repeated benefit.

## Q10 — Does the learned contrastive temperature keep sharpening? Does the contrastive validation loss still rise against a fixed candidate pool?

**With limits.** The learned logit scale sharpens. Initial B scales at epochs 1/6/15 are **13.5999/18.4685/22.5461**. Its learned-scale contrastive losses are **8.54974/6.00606/5.83497**; rescoring with scale fixed at **13** gives **8.52970/5.82207/5.55379**. Candidate pools are held fixed across epochs.

Controlled A scale rises **13.5412 → 21.8558**. Learned-scale contrastive loss falls **4.86771 → 1.65269**, and fixed-scale-13 loss falls **4.89917 → 2.00395**. Controlled B scale rises **13.5988 → 22.5452**. Neither fresh reference has a sustained rise in the overall fixed-pool contrastive loss.

Rescoring at a fixed scale changes a measurement; it is not fixed-temperature retraining. The separate fixed-temperature intervention was not completed. Local direction- and group-specific differences remain, as Q11 describes.

## Q11 — Is the rise concentrated in particular group sizes or similarity strata?

**With limits.** In Initial B from epochs 6 to 15, groups with **more than 25 items account for 80.2% of the net ranking-loss increase**, while **three-item groups account for 76.1% of the net macro decline**. These are shares of net changes, with different denominators for loss and macro correlation.

For controlled A groups containing only previously unseen similarity classes, BCE rises **0.54630 → 0.62373**, change interval **[−0.03240, 0.19907]**; macro rises **0.13584 → 0.22976**, change interval **[0.03030, 0.15353]**; ranking loss changes **2.57456 → 2.54119**, interval **[−0.12493, 0.06786]**. That cohort's BCE minimum is epoch 2 and macro maximum is **0.23762 at epoch 11**. Overall A macro changes **−0.00036** from 14 to 15, interval **[−0.00234, 0.00164]**.

Initial B's overall learned-scale contrastive loss decreases **0.17109** from 6 to 15, while the x→y contribution for three-item groups increases **0.01709**. At fixed scale 13, all size-bin losses improve; the three-item change is **−0.09395**. The contrastive three-item cohort contains **541 groups**, whereas ranking eligibility retains **329**.

Final A learned-scale contrastive losses for all-shared/mixed/all-unseen groups are **1.18614/2.52134/3.97861**; fixed-scale values are **1.50644/3.02026/3.77506**. In the all-unseen stratum, learned y→x loss minimizes at epoch 9, fixed-scale loss at 15. Both directions improve from 14 to 15 in all A strata. This localizes the observed trajectories; it does not establish their mechanism.

## Q12 — Which split matches deployment? Will future y mostly be near neighbours of known y, or come from new regions?

**Answered.** Deployment concerns **new candidates produced by a generator**. Unseen identity and unseen similarity class are separate properties: a new y can still belong to a class represented in training. B tests entirely unseen candidate classes under the predefined relation; A contains both shared and unseen classes.

B is therefore the relevant measured stress regime for class novelty. The future generator's class mixture, score distribution and observation process were not measured. Neither A nor B has been established as an exact sample of deployment. Selection used B validation; A alone should not be treated as validated deployment performance.

## Q13 — How is each test y's maximum similarity to training y distributed in both splits, and how does Split A perform in each similarity stratum?

**Partial.** The predefined relation is **binary similarity-class membership**, not a graded distance. A test has **37,174/192,687 rows (19.29%)** in unseen classes, representing **34,625/162,361 unique y IDs (21.33%)**. Another **26,417 rows** have new y IDs within shared classes. All **136,956 B test rows** have both new IDs and unseen classes. A graded maximum-similarity distribution was not established.

For the depth-3 baseline on A test, intact all-shared/mixed/all-unseen groups give macros **0.1390/0.1071/0.1067**, over **5,548/2,497/233 groups**. The all-unseen interval is **[0.0564, 0.1559]**. Restricting rows before reapplying eligibility instead gives **0.1342 on 7,556 shared-row groups** and **0.0669 on 1,738 unseen-row groups**, versus B **0.0607 on 3,972 groups**. Row-restricted and intact-group estimates must not be interchanged.

At controlled A epoch 15 validation, intact shared/mixed/unseen macros are **0.37470/0.28168/0.22976**, with **5,405/2,402/190 eligible groups**. The unseen-row estimate is **0.19670 on 1,608 eligible groups**, versus full B validation **0.17975 on 3,969 groups**. Requested selected-test neural comparisons remain unfinished. Sources: **R1579–R1582**, reference-stratum records.

## Q14 — What is the size distribution of indivisible units in each partition? Were units assigned at random, or did large units default to training?

**Answered.** A assigns whole groups, with query-coverage anchors in training. B assigns connected units defined jointly by groups and similarity classes, with a giant unit and query-coverage rescue units forced into training.

| Split / partition | Units | Median rows | 99th percentile rows | Largest unit rows |
| --- | ---: | ---: | ---: | ---: |
| A train | 131,450 | 3 | 122 | 49,020 |
| A validation | 16,431 | 3 | 143.4 | 1,409 |
| A test | 16,431 | 3 | 137.7 | 1,147 |
| B train | 570 | 6 | 620 | 1,661,628 |
| B validation | 4,032 | 5 | 431.4 | 7,382 |
| B test | 4,034 | 5 | 447.4 | 5,683 |

The giant B unit supplies **98.70% of training rows**; the remaining **569 training units** provide coverage rescue. Held-out allocation balances rows and labels. Training and held-out units are not exchangeably sampled. The realized B allocation is approximately **86/7/7**, rather than the nominal **80/10/10**. This is a consequence of the constraints, not evidence that the same population was randomly divided at those proportions. Source: data and unit records, **R0882**.

## Q15 — Do the positive rate, score distribution or group descriptors differ across partitions, and does a formal test flag the difference?

**With limits.** The group-descriptor classifier two-sample tests detect train/held-out differences:

| Comparison | A balanced accuracy | B balanced accuracy | Adjusted permutation p |
| --- | ---: | ---: | ---: |
| Train vs validation | 0.5170 | 0.8522 | 0.003 for each |
| Train vs test | 0.5181 | 0.8594 | 0.003 for each |
| Validation vs test | 0.4958 | 0.4966 | 1 for each |

These are unit-aware descriptive tests with Holm correction. The small p-values do not make the much smaller A effect as large as B's. B binary-positive fractions are **0.679042 train, 0.792950 validation, 0.790239 test**.

B black-box shift estimation gives positive fraction **0.7849832490**, interval **[0.6696273941, 0.8789467582]**. The calibration confusion matrix, rows predicted class and columns true class, is **[[0.4296646331, 0.1050460405], [0.5703353669, 0.8949539595]]**, with condition number **3.8095792271**. Predicted test-positive fraction is **0.8251555244**; equation residual is **2.78×10⁻¹⁷**. There are **1,999 identified bootstrap estimates and zero boundary estimates**.

The near-exact residual is algebraic fit, not evidence for class-conditional invariance. That assumption remains unverified. The observed test prevalence is a diagnostic comparison, not a model-selection input. A lacks the corresponding completed shift-estimation result. Sources: **R1569–R1576**.

## Q16 — Which x appear in Split B's test groups, and how many training groups does each have?

**With limits.** All **1,723 B test x IDs** occur in training. Their training-group counts range **1–1,303**, median **24**; weighting by test-group frequency gives median **127**.

| Training groups for x | Eligible B test groups | Initial B test macro | Conditional interval |
| --- | ---: | ---: | --- |
| 1 | 71 | 0.0539 | [−0.0701, 0.1748] |
| 2–5 | 149 | 0.0849 | [0.0018, 0.1716] |
| 6–20 | 311 | 0.1829 | [0.1324, 0.2345] |
| 21–100 | 1,184 | 0.1626 | [0.1350, 0.1891] |
| >100 | 2,257 | 0.1935 | [0.1710, 0.2148] |

Weak-support intervals are wide and bins differ in composition. Performance is not monotone across all bins. This establishes uneven support, not a causal effect of adding groups for a query. Source: **R1599**.

## Q17 — Across model variants, do Split A and Split B scores move together?

**Pending requested result.** The requested variant grid and selected-test analysis were not completed. No formal correlation, fitted line or reliable proxy-selection rule was established.

The completed validation pairs already show that improvements need not transfer between splits: detachment raises A but lowers B; increasing the full-model learning rate threefold raises A's peak to **0.37531688** but lowers B's peak to **0.17310025**. This is descriptive evidence against assuming every A improvement helps B. It is not the unfinished E58 association analysis, and does not establish a general inverse relationship.

## Q18 — What is the repeat-based noise ceiling of the chosen metric on each split?

**With limits.** The measured test split-half macros are **A 0.54450467 / B 0.37800495**, with Spearman–Brown heuristics **0.70503949/0.54854535**. Qualifying repeated-member coverage is only **420/8,278 = 5.07%** of eligible A groups and **348/3,972 = 8.76%** of eligible B groups.

Each estimate uses **100 random half assignments**. A's assignment range is **[0.52771567, 0.56065509]**, with **413–416 defined groups** per repetition. All corresponding B values and the four other partitions are retained in [data measurements](data_measurements.md) and **R1500**.

The correction assumes exchangeable parallel measurements. Applying it to a macro grouped-rank statistic is a heuristic, and the repeated subset is selective. These numbers are **not established ceilings for the full test populations**. Dividing whole-test model performance by them would not establish the fraction of attainable performance achieved.

## Q19 — How does within-group variance of s divide among the y-cluster main effect, the x×y interaction, and repeat noise?

**With limits.** Six partition-specific variance fits and six group-balanced alternatives were produced. The full component and within-group contribution tables are in [data measurements](data_measurements.md), with full precision in **R1602–R1608**.

On A test, the primary fit estimates query, group, class-main, interaction, raw-measurement-noise and observation-residual variances of **0.25080549, 0.92082342, 0.47299692, 0, 0.10354355 and 0.03582322**, respectively, in **s²**. After within-group centering and equal-group averaging, the class-main contribution is **0.18376655** and observation residual **0.01888998**; query and group intercepts cancel.

The estimated within-group interaction contribution changes **0 → 0.11592039** on A test and **0.1716375 → 0.0567531** on B test when using group-balanced fitting. B train's fitted marginal variance changes **3.9844 → 2.0749**, versus observed **1.9578**. Boundary noise estimates conflict with directly observed repeat disagreement.

The decomposition is therefore not stable enough to identify a transferable interaction share or an architectural ceiling. Raw-measurement variance is not uncertainty of the aggregated median. Observation residual can contain unmodeled y-specific signal; it cannot all be called irreducible noise.

## Q20 — What fraction of within-group pairs differ by less than the repeat noise?

**Answered.** Training raw-repeat RMS differences are **A 0.78055161 / B 0.76605203** in s units. These bands contain **60.802%/61.887%** of test within-group label differences. A fixed band of **0.47712125471966244** contains **42.297%/43.640%**.

These large fractions make near-tie handling consequential. RMS raw-repeat disagreement measures differences between individual repeated measurements; it is neither a confidence interval nor the standard error of an aggregated label. Test-specific and validation-specific RMS values are also recorded, but the reported test noise-band evaluation uses training-derived bands. Source: **R1500** and the near-tie records.

## Q21 — Why do only half the groups support ranking supervision, and does that half differ in size, variance or x?

**Answered.** Most exclusions are groups with fewer than three candidates. A/B training have **62,516/67,994** such groups, followed by **5,067/5,259** groups with enough items but insufficient native-label span. Native headline eligibility is **63,867/72,201 groups**, about **48.59%/49.64%**.

The float64 sampler has **63,867/72,202 eligible groups**. A has two membership disagreements with no net count difference; B has one, due to boundary precision. This discrepancy was documented rather than silently treating sampler and metric eligibility as identical.

Eligible/ineligible training median group sizes are **11/1**. Mean within-group variances are **0.5360/0.0863 on A** and **0.5492/0.0918 on B**. Query-frequency total-variation distances between eligible and ineligible cohorts are **0.2904/0.2908**. Of **7,873 training queries**, **3,420/3,393** have no headline-eligible group. The ranking-supervised subset is therefore selective in size, variability and query composition. Source: **R1598**.

## Q22 — How were pairs chosen for observation, given the 70% positive rate?

**With limits.** The dataset was reconstructed from input records that passed predefined inclusion checks. Records were grouped by query, checked for consistency and aggregated within groups by the median. It retained measurements on both sides of the binary threshold.

All **1,957,400 retained observations** were reconstructed from **2,032,748 retained raw recorded labels**, with **zero count or median mismatches**. The input groups contain **4,161,420 measurement records across 164,312 groups**. Of these, **2,034,645** have the required numeric score, **2,126,775** do not, and **1,897** numeric records fall outside retained observation identities. Other input-record categories U1/U2 account for **1,345,088 records**; **500,807** records have non-exact-label flags. These categories are not counts of distinct missing pairs or valid negatives.

The original aggregated observations are **69.04% positive**. Numeric input records are **69.2176% positive**, and retained raw recorded labels **69.2218%**. Source collections differ strongly: **C01: 1,138,209 rows, 67.79% positive; C02: 637,588, 82.86%; C03: 116,824, 4.61%**. These three collections do not cover all observations; **64,779 rows** belong to other collections.

Original sampling frames, unrecorded outcomes and inclusion probabilities are unavailable. The data demonstrate source selection and composition, not a quantified mechanism of preferential observation. Missing pairs are not negatives, and observed prevalence is not the prevalence among newly generated candidates.

An additional provenance issue was found: the original pooled pair-median procedure allowed held-out measurements to change **5,289 A training labels**, versus **zero on B**. Controlled fits use partition-local pair medians for binary labels. The exact historical split identity behind the original configuration could not be recovered; the new fits are not byte-identical reconstructions of that historical dataset. Sources: **R1380–R1381**, partition-label verification records.

## Q23 — How many groups have all members on one side of the threshold, and how far is the threshold from the repeat noise?

**Answered.** Binary labels use the **partition-local pair median ≥θ**; strict contrastive-positive anchors use **s >θ**. These boundary rules differ.

| Split / partition | All-positive groups | All-negative groups | Same-side fraction among eligible groups |
| --- | ---: | ---: | ---: |
| A train | 63,190 | 28,715 | 44.50% |
| A validation | 8,069 | 3,537 | 45.28% |
| A test | 7,791 | 3,524 | 43.63% |
| B train | 69,799 | 30,457 | 43.87% |
| B validation | 4,683 | 2,443 | 50.09% |
| B test | 4,919 | 2,664 | 54.20% |

The first two count columns concern all groups; the last column uses only ranking-eligible groups. On A/B test, **35.58%/32.21% of rows' pair-median scores** are within the training raw-repeat RMS band around θ. Using each original observation score instead gives **35.51%/32.03%**. BCE supplies no within-group binary ordering preference when every member has the same binary label. This does not prove that BCE cannot improve shared features. Source: **R1598**.

## Q24 — Are batches built from whole groups or from observations, and is the fraction of ranking-eligible groups per batch fixed?

**Answered.** The sampler mixes bounded group lists with classification-only observations. Large groups can yield multiple lists; a training item is not always one observation or one entire original group.

The reconstructed first-epoch order has **12,820 A / 14,020 B local microbatches**. Eligible-list fractions average **0.5583/0.5625**, standard deviations **0.1468/0.1474**, with fifth/ninety-fifth percentiles **1/3 and 5/6**. Each split has **one local batch with no ranking-eligible list**. The proportion is not fixed. This is sampler reconstruction, not a recording of every historical device batch. Sources: **R1502–R1503**.

## Q25 — Does the contrastive loss treat other positives of the same x in the batch as negatives?

**With limits.** Existing masks already exclude co-candidates in the same group, cross-group same-query comparisons and duplicate query–candidate identities. Nevertheless, other pairs known positive in the training data can remain contrastive negatives under the actual construction.

Across **512 audited batches**, training-known positive fractions among x→y negative comparisons are **0.530%/0.714% on A/B**, or **0.437/0.576 comparisons per anchor**. The y→x fractions are **0.441%/0.657%**. A numerical fixture verifies the sign of the resulting gradient pressure.

This is a lower bound based on observed training positives with **s>θ**. It is not the prevalence of all true positives, nor a measured change in generalization from additional masking. The masking intervention was outside the narrowed scope and was not completed.

## Q26 — Are the contrastive projections taken before or after cross-attention?

**Answered.** Contrastive projections operate on separate pooled encoder outputs **before cross-attention**. Cross-pair contrastive scores can therefore be computed from embedding products. Ranking and BCE instead use fused representations after cross-attention, with distinct scalar heads.

Thus cosine, ranking and classification scores are different readouts, with different access to pair interactions. Equal vector dimension alone does not make two separately pretrained encoders' coordinate systems comparable. No result from the separate requested 10,000-pair embedding exercise was completed, and no embedding-only ranking improvement can be reported from it.

## Q27 — How is the listwise loss normalized across groups of different sizes?

**Answered.** The reference sums weighted Plackett–Luce selections, with tie-averaged weights of the form **1 / (sqrt(n) × log(position+1))**, then averages sampled lists. It does not divide each list's loss by its retained weight sum. Original groups larger than 32 items receive **ceil(n/32)** bounded-list opportunities, so averaging lists does not weight original groups equally.

The completed list-normalization variant gives best validation macros **A 0.33629307 / B 0.17121073**, versus full **0.34367521/0.18457785**. Final values are **0.33629307/0.16224005**, with final probe macros **0.48462296/0.50576951**. This variant changes loss scaling as well as relative weighting; raw ranking-loss magnitudes are not directly comparable to the reference.

Normalized B was verified through native completion metadata, best/final weight retention and **31 export manifests**. Across **30 matched validation/probe exports**, ordered identities and labels match the reference exactly: **136,957 validation rows / 43,115 probe rows**. Defined-group coverage is normally **3,969/2,047**; normalized validation epochs 1 and 11 have **3,968**, and probe epoch 2 has **2,046**. Best and final comparison coverage matches. The final native metric is **0.16224005**; a post-training re-evaluation of best weights at the same final step gives **0.17121073** and is not the epoch-15 result.

The distinct inverse-opportunity group-balanced variant failed at its first evaluation and produced no completed checkpoint. Its evaluation-only repair passed **99 CPU tests** but was not deployed or qualified on the training hardware. No group-balanced performance result follows from those software tests. Source: **R1786**, normalized-B comparison and retention records.

## Q28 — Are x frequencies in batches skewed, and is any sampling-probability correction applied?

**With limits.** Among **7,873 training queries**, the top **1%** supply **30.07%/30.64%** of A/B group-list opportunities. The median opportunity count is **2**, and maxima are **1,587/1,589**. The reference applies no log-Q sampling-probability correction.

The skew is measured. A learned query-popularity shortcut is a hypothesis, not established by counts alone. Query-only predictive performance in Q32 is relevant but is also not a causal decomposition of the full model.

## Q29 — Are subsets for ablations and learning curves drawn by indivisible unit?

**Partial.** Prepared E41 views preserve whole groups for A and whole connected components for B. Requested unit fractions **10/25/50/100%** retain **9.51/25.93/49.85/100% of A rows**, but **98.86/99.01/99.52/100% of B rows**. The giant B training component makes unit fraction a poor proxy for row budget.

Eight subset views were prepared; the 10% A/B loader and sampler paths were qualified. No corresponding learning-curve fits completed. There is no evidence that performance saturates at 10% of the training observations.

For E40, the user specified **candidate-only versus full paired model** as the second axis, alongside fixed-test leak-in/leak-out training. Whole-group A matching was qualified. Strict whole-component exchange on B yields **zero eligible exchanges** under fixed query support. A proposal affecting nine partial components was not selected. E40/E41 performance runs and that exception were excluded from the narrowed scope. Prepared data and runtime checks are not performance evidence.

## Q30 — What ranking quality does each loss deliver alone, and in each pair, on each split?

**Excluded by user scope.** The R, B, C, RB, RC, BC and RBC training grid was not completed. The completed full-model and requested variant fits retain all three losses at nominal weights **0.25/0.25/1**. Contrastive detachment changes which parameters receive C gradients; it is not C removal. Reading cosine from a jointly trained fit is not contrastive-only training.

No empirical ranking contribution can be assigned to each standalone loss or loss pair from the available experiments.

## Q31 — How well do the ranking head, the BCE logit and the contrastive similarity agree within groups, and does their agreement fall during training?

**With limits.** Agreement depends on which pair of readouts is considered:

| Fit / epochs | Ranking–BCE | Ranking–cosine | BCE–cosine |
| --- | ---: | ---: | ---: |
| Initial B, 1 | 0.6798 | 0.0743 | 0.0958 |
| Initial B, 6 | 0.6363 | 0.2359 | 0.2759 |
| Initial B, 15 | 0.6017 | 0.2341 | 0.2769 |
| Controlled A, 1 | 0.63066 | 0.12360 | 0.14741 |
| Controlled A, 15 | 0.63610 | 0.29585 | 0.33631 |
| Controlled B, 1 | 0.67647 | 0.07402 | See full register |
| Controlled B, 15 | 0.59235 | 0.23018 | See full register |

Initial B ranking–BCE agreement changes **−0.0781 [−0.0985, −0.0600]**, while ranking–cosine changes **+0.1598 [0.1321, 0.1891]**. Its cosine–label macro changes **0.0653 → 0.0749**, difference interval **[−0.0085, 0.0270]**. On its training probe, ranking–BCE agreement instead increases **0.6452 → 0.6776**.

Controlled A ranking–BCE agreement uses **7,995 mutually defined groups**, with change interval **[−0.00474, 0.01599]**. Ranking–cosine uses **7,997**, change interval **[0.15822, 0.18617]**. Controlled B uses **3,968/3,969** mutually defined groups for those comparisons.

Final controlled A BCE/cosine label macros are **0.27035/0.17598**, below ranking **0.34332**. Controlled B values are **0.13944/0.07754**, below ranking **0.17975**. Initial B selected-test cosine is **0.06623441**, versus ranking **0.17687961**, on the same **3,972 eligible groups**. Prediction agreement is not gradient alignment; these results do not prove optimization conflict.

## Q32 — How much of the BCE head's validation AUC does an x-only model reproduce?

**With limits.** Query-only BCE validation/test AUROCs are **A 0.6687/0.6696** and **B 0.6452/0.6493**. Relative to chance, the validation comparison recovers about **48.9%** of full A's excess AUROC and **54.6%** of Initial B's: full values **0.8451981/0.7658289**, query-only **0.6687445/0.6452095**.

These are comparisons of separately trained predictors, not causal shares of the full model's information. A query-only score is constant within a comparison group and cannot establish its internal candidate order. A separate B query-prior lookup has validation/test AUROC **0.6915/0.7060**, with pair accuracy **0.5**; it should not be conflated with the frozen-feature query-only baseline.

## Q33 — Do the auxiliary gradients align with the ranking gradient in the encoders and in cross-attention?

**Pending requested result.** No trained-checkpoint gradient-cosine or lookahead-transfer scientific result was produced. The plan fixes **64 batch pairs**, and CPU implementation qualification exists, but this does not measure encoder- or cross-attention-level gradient alignment.

The adaptation design contains **22 kind/scope/k cells** and **8,088 planned short support fits** (**24 across-unit + 8,064 within-unit**); none executed. Six geometry cohorts were prepared; no corresponding measured geometry result completed. These are design sizes, not experimental sample sizes or findings. E49–E58 remain incomplete as detailed in the disposition appendix.

## Q34 — Were the loss weights tuned, and on which split's validation data?

**Answered.** The user confirmed that **R/B/C = 0.25/0.25/1 were chosen arbitrarily**, without validation-based weight selection. Historical tracking showed six runs: three settings on each split, **0.5/0.5/0.5**, **0.5/0.25/0.25**, and **0.25/0.25/1**. All six historical runs were marked killed, used loss-based checkpoint selection, and had no linked sweep establishing a selection decision. Their configuration files were absent from version control. The run inventory alone does not contradict the user's account of how the final weights were chosen.

The later learning-rate comparisons are separate from loss-weight selection:

| Split | Seed | Learning-rate multiplier | Best epoch | Best validation macro | Final validation macro |
| --- | --- | --- | --- | --- | --- |
| A | 42 | 0.3 | 14 | 0.28035523 | 0.27927916 |
| A | 42 | 1.0 | 14 | 0.34367521 | 0.34331738 |
| A | 42 | 3.0 | 15 | 0.37531688 | 0.37531688 |
| A | 43 | 1.0 | 15 | 0.34493525 | 0.34493525 |
| B | 42 | 0.3 | 12 | 0.18322109 | 0.18051769 |
| B | 42 | 1.0 | 6 | 0.18457785 | 0.17975193 |
| B | 42 | 3.0 | 6 | 0.17310025 | 0.16400919 |
| B | 43 | 1.0 | 5 | 0.17738602 | 0.16829567 |


B validation selected the base learning rate for both A and B: B peaks **0.18457785**, **0.18322109**, **0.17310025** for base, ×0.3 and ×3. No test score entered that decision. Further optimization, weight searches and additional seeds were excluded. These historical comparisons are reported because they happened; they are not an ongoing search.

## Inexpensive baseline evidence

All eleven baseline-tier ledger entries completed. Scores below are test macro Spearman under their recorded coverage rules:

| Baseline | A test | B test | Defined coverage A / B |
| --- | ---: | ---: | --- |
| Candidate-only ridge | 0.0775 | 0.0624 | 100% / 100% |
| Same-query nearest-class lookup | 0.2572 | Undefined | 79.72% / 0% |
| Independent per-query ridge | 0.2419 | 0.0726 | 99.83% / 99.52% |
| Two-step ridge | 0.1034 | 0.0488 | 100% / 100% |
| Identity matrix filter | 0.4207 | Undefined | 93.32% / 0% |
| Paired Kronecker ridge | 0.1690 | 0.0631 | 100% / 100% |
| Query-shuffled Kronecker ridge | 0.0662 | 0.0596 | 100% / 100% |
| Bilinear inductive completion | 0.2126 | 0.0779 | 100% / 100% |
| Fixed depth-3 LambdaMART | 0.1285 | 0.0607 | 100% / 100% |

Undefined B scores mean those methods produce no usable within-group ordering for its unseen classes, not zero correlation. Coverage-selective A scores cannot be ranked as though all methods were evaluated on an identical group population. In particular, the identity filter's **0.4207** with **93.32%** coverage is not a full-coverage reference score; its conditional interval is **[0.4111, 0.4304]**.

Single-axis ridge baselines used five penalty values; two-step ridge used **25 combinations**. Kernel solver relative residuals were **<2×10⁻⁷**. The paired-minus-query-shuffled difference is **A 0.1029 [0.0920, 0.1138]**, versus **B 0.0035 [−0.0157, 0.0221]**. The comparison uses one permutation per partition and does not quantify permutation variability. Source: **R1767** and baseline records.

Bilinear completion evaluated **nine rank/penalty settings per split**, with **3,000 L-BFGS iterations**. Selected ranks were **A 128 / B 8**, penalty **10⁻⁶** for both, and validation macros **0.2190/0.0782**. Test intervals are **[0.2035, 0.2212]/[0.0628, 0.0933]**. Both selected fits hit the iteration limit; maximum gradients **1.07×10⁻⁴/7.80×10⁻⁶** exceed the **10⁻⁷** target. Earlier 300-iteration fits remain in the register. These are finite-optimization results, not certified optima.

The user-fixed depth-3 LambdaMART selected **500 A / 270 B trees**, with test intervals **[0.11960, 0.13733]/[0.04507, 0.07568]**. CPU rescoring reproduced A exactly and changed B by only **−1.4163×10⁻⁸**. Further runs were stopped at the user's request. Its common-support validation comparison gives weighted A **0.1221** versus B **0.04884**.

For Initial B, pair accuracy by group size is **n=2: 0.56846081 (1,150 groups); n=3: 0.56365503 (537); n=4–10: 0.55544266 (1,521); n=11–25: 0.57201655 (1,255); n>25: 0.58955236 (1,152)**. The **3,963 singleton groups** contribute no pairs. These bin counts concern the broader pairwise population, not the headline-eligible macro cohort.

## Verification, corrections and stop state

The stop verification found **no study-owned training or controller processes**, and **no termination signal was needed**. Training had stopped at the group-balanced A evaluation failure; downstream reporting and probe queues completed no scientific jobs. The study remains paused. The evaluation repair was not deployed. Consolidated evidence: **R1786**.

All fourteen completed requested fits have verified completion and retention records. Original run identifiers, source hashes, infrastructure details and file locations are excluded from this shared report. The file checksums cover only the anonymized package. An anonymous record index connects each reported value to its retained evidence fields.

An earlier ten-epoch narrative table transcribed controlled A epoch 7 as **0.30188**. The native value is **0.3022522362359804**; the epoch CSV and corrected historical appendix use that value, rounded to **0.30225** where appropriate. The incorrect transcription is not a second measurement.

The evidence supports observed composition and support shifts, training/validation divergence on B, distinct behavior among objectives and readouts, and the absence of an improvement over full B among the completed requested variants. It does **not** establish a population noise ceiling, a causal interaction-variance share, gradient conflict, a successful adaptation intervention, or completed selected-test rankings for the requested grid. Those distinctions are retained throughout the report and measurement register.
