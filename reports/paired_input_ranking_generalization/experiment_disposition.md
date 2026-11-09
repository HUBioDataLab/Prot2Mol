# Experiment disposition at the stop

This is the complete E01–E70 inventory. Historical ledger status is retained in its own column, rather than silently rewritten as current completion. The current column distinguishes a completed fit from a completed selected-test or mechanism experiment. No entry authorizes continuation.

Historical totals: 23 complete, 12 partial, 35 pending. These include subsequently excluded work. Tier totals are 12/9/0 for Tier 1; 11/0/0 for Tier 2; 0/2/14 for Tier 3; 0/1/9 for Tier 4; and 0/0/12 for conditional interventions (complete/partial/pending). Only Tier 2 is fully complete.

| ID | Tier | Specified experiment | Historical ledger | Current evidence / disposition |
| --- | --- | --- | --- | --- |
| E01 | Tier 1 | Re-score saved predictions with per-group averaged Spearman | partial | Initial B and reference audits completed; requested selected-model comparison unfinished. |
| E02 | Tier 1 | Compute concordant-minus-discordant concordance over within-group pairs | partial | Initial B and reference audits completed; requested selected-model comparison unfinished. |
| E03 | Tier 1 | Compute tie-adjusted Kendall tau-b | partial | Initial B and reference audits completed; requested selected-model comparison unfinished. |
| E04 | Tier 1 | Treat differences within repeat noise as ties | partial | Noise-band evidence completed for Initial B; requested comparison unfinished. |
| E05 | Tier 1 | Report pairwise ordering accuracy in the specified group-size bins | partial | Group-size pair-accuracy evidence completed for Initial B; requested comparison unfinished. |
| E06 | Tier 1 | Audit size, eligibility, positive rate, scores, variance, repeats, and query coverage across all six partitions | complete | Completed recorded audit or baseline; qualifications are in the answers and measurement notes. |
| E07 | Tier 1 | Tabulate indivisible component sizes and assignment rules | complete | Completed recorded audit or baseline; qualifications are in the answers and measurement notes. |
| E08 | Tier 1 | Stratify Split A by maximum training-candidate similarity under the predefined relation | partial | Binary class strata measured for baseline and references; requested selected-test extension unfinished. |
| E09 | Tier 1 | Reweight Split A to Split B group-size and label composition | partial | Baseline and reference composition matching measured; requested extension unfinished. |
| E10 | Tier 1 | Estimate repeat split-half ceiling with assumptions and coverage stated | complete | Completed recorded audit or baseline; qualifications are in the answers and measurement notes. |
| E11 | Tier 1 | Fit candidate-cluster main effect, query-by-cluster interaction, and noise components where identifiable | complete | Completed recorded audit or baseline; qualifications are in the answers and measurement notes. |
| E12 | Tier 1 | Decompose saved checkpoint losses, including both contrastive directions | complete | Completed recorded audit or baseline; qualifications are in the answers and measurement notes. |
| E13 | Tier 1 | Compare BCE discrimination with log-loss over checkpoints | complete | Completed recorded audit or baseline; qualifications are in the answers and measurement notes. |
| E14 | Tier 1 | Fit temperature using validation only and evaluate calibration | complete | Completed recorded audit or baseline; qualifications are in the answers and measurement notes. |
| E15 | Tier 1 | Track learned contrastive scale over checkpoints | complete | Completed recorded audit or baseline; qualifications are in the answers and measurement notes. |
| E16 | Tier 1 | Re-evaluate contrastive loss against a fixed candidate pool | complete | Completed recorded audit or baseline; qualifications are in the answers and measurement notes. |
| E17 | Tier 1 | Locate best validation macro-Spearman and best-loss checkpoints | complete | Completed recorded audit or baseline; qualifications are in the answers and measurement notes. |
| E18 | Tier 1 | Classifier two-sample test on group descriptors | complete | Completed recorded audit or baseline; qualifications are in the answers and measurement notes. |
| E19 | Tier 1 | Label-marginal shift estimation/test with invertibility checked | partial | B estimate completed with assumptions unresolved; A and requested extension unfinished. |
| E20 | Tier 1 | Compute uncertainty by resampling indivisible split units | partial | Conditional whole-unit intervals exist; requested variant comparison unfinished. |
| E21 | Tier 1 | Compare top-weighted metrics with the headline metric as requested by the metric questions | complete | Completed recorded audit or baseline; qualifications are in the answers and measurement notes. |
| E22 | Tier 2 | Train/evaluate the candidate-only baseline | complete | Completed recorded audit or baseline; qualifications are in the answers and measurement notes. |
| E23 | Tier 2 | Train and evaluate a query-shuffled model; distinguish this from inference-only shuffling | complete | Completed recorded audit or baseline; qualifications are in the answers and measurement notes. |
| E24 | Tier 2 | Evaluate a query-only BCE baseline | complete | Completed recorded audit or baseline; qualifications are in the answers and measurement notes. |
| E25 | Tier 2 | Predict from the nearest training candidate under the predefined relation | complete | Completed recorded audit or baseline; qualifications are in the answers and measurement notes. |
| E26 | Tier 2 | Frozen-feature independent-task ridge baseline | complete | Completed recorded audit or baseline; qualifications are in the answers and measurement notes. |
| E27 | Tier 2 | Frozen-feature two-step kernel ridge baseline | complete | Completed recorded audit or baseline; qualifications are in the answers and measurement notes. |
| E28 | Tier 2 | Linear matrix-filter baseline using the identity smoother kernel, with observed-only fitting | complete | Completed recorded audit or baseline; qualifications are in the answers and measurement notes. |
| E29 | Tier 2 | Frozen-feature observed-pair Kronecker ridge using a scalable exact finite-feature solver | complete | Completed recorded audit or baseline; qualifications are in the answers and measurement notes. |
| E30 | Tier 2 | Bilinear inductive matrix-completion baseline | complete | Completed recorded audit or baseline; qualifications are in the answers and measurement notes. |
| E31 | Tier 2 | Frozen-feature LambdaMART with linear gains, noise-calibrated levels and group normalization | complete | Completed recorded audit or baseline; qualifications are in the answers and measurement notes. |
| E32 | Tier 2 | Score the current model contrastive similarity alone on Split B | complete | Completed recorded audit or baseline; qualifications are in the answers and measurement notes. |
| E33 | Tier 3 | Compare validation-Spearman and aggregate-loss early-stopping/selection rules | pending | Selection-rule retraining comparison excluded by the user; descriptive optima reported. |
| E34 | Tier 3 | Train interaction and heads with both encoders frozen | pending | A/B frozen fits complete; native validation evidence reported; selected-test comparison unfinished. |
| E35 | Tier 3 | Partially unfreeze encoders | pending | A/B last-two-block fits complete; native validation evidence reported; selected-test comparison unfinished. |
| E36 | Tier 3 | Evaluate several low-rank update ranks | pending | Ranks 4/16/64 on A/B complete (six fits); selected-test comparison unfinished. |
| E37 | Tier 3 | Full fine-tuning reference | pending | Eight full reference/LR/seed fits complete; narrowed selected-test comparison unfinished. |
| E38 | Tier 3 | Evaluate R, B, C, RB, RC, BC and RBC | pending | Loss-subset grid excluded by the user; no completed result. |
| E39 | Tier 3 | Evaluate a static loss-weight sweep with equal tuning budgets | pending | Loss-weight search excluded by the user; no completed prospective result. |
| E40 | Tier 3 | Fixed-test 2x2 leak-in/leak-out design with composition and training-size matching | partial | Design and A runtime qualification only; strict B exchange infeasible; further work excluded. |
| E41 | Tier 3 | Subsample complete indivisible units and report realized sizes | partial | Eight whole-unit views prepared and 10% loader qualified; performance runs excluded. |
| E42 | Tier 3 | Mask same-query co-positives | pending | Known-positive audit complete; retraining intervention excluded from narrowed scope. |
| E43 | Tier 3 | Compare fixed and learned contrastive temperature | pending | Fixed-scale rescoring exists; fixed-temperature retraining excluded from narrowed scope. |
| E44 | Tier 3 | Detach contrastive gradients from encoders | pending | A/B detachment fits complete; selected-test comparison unfinished. |
| E45 | Tier 3 | Normalize ranking objectives per group | pending | A/B list-normalized fits complete; group-balanced A failed, B unstarted; selected-test comparison unfinished. |
| E46 | Tier 3 | Compare uniform pair weights with top-weighting | pending | Requested uniform-pair fits unstarted. |
| E47 | Tier 3 | Calibrate soft pair queries using repeat noise | pending | Requested noise-calibrated pair fits unstarted. |
| E48 | Tier 3 | Evaluate checkpoint averaging, as specified in the gated execution sequence | pending | Checkpoint-averaging intervention excluded from narrowed scope. |
| E49 | Tier 4 | Adapt heads on adaptation units and evaluate disjoint validation/test units | pending | Adaptation design and software preparation only; no scientific job completed. |
| E50 | Tier 4 | Repeat adaptation including cross-attention | pending | Adaptation design and software preparation only; no scientific job completed. |
| E51 | Tier 4 | Adapt and evaluate on disjoint groups of the same unit | pending | Adaptation design and software preparation only; no scientific job completed. |
| E52 | Tier 4 | Vary k across all three adaptation settings | pending | Adaptation design and software preparation only; no scientific job completed. |
| E53 | Tier 4 | Score each pathway separately on both splits | partial | Initial B and reference readout evidence exists; requested selected-model grid unfinished. |
| E54 | Tier 4 | Measure alignment/uniformity per side and split before/after projections | pending | Geometry cohorts prepared; no scientific result. |
| E55 | Tier 4 | Measure representation effective rank at the specified representation stages | pending | Geometry cohorts prepared; no scientific result. |
| E56 | Tier 4 | Measure per-block objective gradient cosines | pending | Fixed-batch design and CPU qualification only; no trained-checkpoint result. |
| E57 | Tier 4 | Measure TAG-style lookahead transfer on cold validation batches | pending | Fixed-batch design and CPU qualification only; no trained-checkpoint result. |
| E58 | Tier 4 | Plot Split A versus Split B across trained variants | pending | Descriptive completed validation pairs available; formal requested analysis unfinished. |
| E59 | Conditional interventions | Staged frozen-head training followed by low-rate fine-tuning | pending | Conditional intervention not run and outside the latest requested scope. |
| E60 | Conditional interventions | Interpolate encoder weights toward pretrained weights | pending | Conditional intervention not run and outside the latest requested scope. |
| E61 | Conditional interventions | Use multi-positive contrastive queries | pending | Conditional intervention not run and outside the latest requested scope. |
| E62 | Conditional interventions | Use a positive/negative/masked sigmoid label matrix | pending | Conditional intervention not run and outside the latest requested scope. |
| E63 | Conditional interventions | Use query-conditional rank-aware contrastive training | pending | Conditional intervention not run and outside the latest requested scope. |
| E64 | Conditional interventions | Use a compatible pointwise/ranking loss pair | pending | Conditional intervention not run and outside the latest requested scope. |
| E65 | Conditional interventions | Share an anchored scalar score between BCE and ranking | pending | Conditional intervention not run and outside the latest requested scope. |
| E66 | Conditional interventions | Evaluate a normalized soft-Spearman objective | pending | Conditional intervention not run and outside the latest requested scope. |
| E67 | Conditional interventions | Evaluate head-level regularization when probes justify it | pending | Conditional intervention not run and outside the latest requested scope. |
| E68 | Conditional interventions | Evaluate factorized scoring plus a gated interaction residual | pending | Conditional intervention not run and outside the latest requested scope. |
| E69 | Conditional interventions | Assess the document-listed domain methods last, after their applicability checks | pending | Conditional intervention not run and outside the latest requested scope. |
| E70 | Conditional interventions | Assess the document-listed specialized optimizers last | pending | Conditional intervention not run and outside the latest requested scope. |


## Latest requested training queue

Fourteen of 24 variant fits completed. One failed at evaluation; nine were never started. The following table records actual execution, not the historical experiment ledger.

| Fit | Observed state |
| --- | --- |
| M004 | Completed 15 epochs |
| M005 | Completed 15 epochs |
| M017 | Completed 15 epochs |
| M018 | Completed 15 epochs |
| M023 | Completed 15 epochs |
| M024 | Completed 15 epochs |
| M021 | Completed 15 epochs |
| M022 | Completed 15 epochs |
| M025 | Completed 15 epochs |
| M026 | Completed 15 epochs |
| M002 | Completed 15 epochs |
| M003 | Completed 15 epochs |
| M019 | Completed 15 epochs |
| M020 | Completed 15 epochs |
| M015 | Failed at first evaluation; no completed checkpoint |
| M016 | Unstarted |
| M031 | Unstarted |
| M032 | Unstarted |
| M033 | Unstarted |
| M034 | Unstarted |
| M027 | Unstarted |
| M028 | Unstarted |
| M029 | Unstarted |
| M030 | Unstarted |


The separate requested 10,000-pair encoding comparison has no completed scientific result. It is not counted as an E01–E70 completion.
