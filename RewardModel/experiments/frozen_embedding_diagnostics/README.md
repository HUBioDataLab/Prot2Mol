# Frozen representation diagnostics

This is a separate user-authorized study. It does not resume any previous
training, adaptation or probe queue. No optimizer is created. Only physical
GPUs 0, 2, 5 and 6 are permitted, addressed through the UUID allowlist in
`study.py`. Code is edited locally, pushed to the study branch and pulled on
the execution host.

## Fixed design

- The existing B training and validation partitions are retained; test is not read.
- Exactly 1,000 comparison groups and 10 distinct candidates per group are
  selected in each partition: 20,000 observations in total.
- All queries with an eligible group in both partitions are included once,
  followed by balanced rounds over queries with remaining eligible groups.
- For each group, five candidates are sampled uniformly and five through
  category-balanced sampling. Duplicates are resolved by label-independent
  random choice. Labels are never used to choose observations.
- Score centering uses the original complete source-group mean, independently
  in each partition. The original scalar observations retain float32 label
  precision for benchmark-consistent eligibility.
- The primary readout uses pretrained final-layer content-mean embeddings.
  Embedding stage 0 and all twelve blocks, two poolings, and one random encoder
  initialization are also reported: 52 cases, with no best-layer selection.
- Both native widths are 768. The query checkpoint uses its published expanded
  symbol vocabulary and fixed four-token example prefix. Query unknown tokens
  are forbidden. Content mean excludes padding, special tokens and that prefix;
  first-token pooling uses the actual first token. Token caps are 1,024 and 202.
  Truncation and repeated-forward determinism are recorded. Parameters are
  hashed before and after extraction. No weights or token arrays are published.

## Measurements and controls

1. Fixed sampled training-pair curves: same group, same query/different group,
   and different query. Group-balanced centred-score products and squared
   differences are summarized in fixed cosine-distance bins. These are
   descriptive associations and do not identify an interaction decomposition.
2. Five-neighbour inverse-distance readouts using same-query references,
   other-query references, and a reference-count-matched other-query control.
   Outcomes are training-only. Repeated query/candidate observations are
   averaged before retrieval; other-query candidates are averaged equally
   across their contributing queries. Numerical distance epsilon is 1e-6.
3. Within-group unit-vector differences for label gaps greater than 0.78,
   averaged first within group and then within query. Own, leave-query-out
   global, nearest-query-borrowed and support-matched random-borrowed directions
   are compared. Borrowing uses five queries and the same weighting in both
   borrowing controls. Query and distance support strata are preserved.
4. Training-only profile agreement over observed intersections with at least
   five common measured entries. At least ten edges are required to summarize
   an axis. Whole-object permutations preserve the profile observation graph
   and are stratified by support bins. The 999-draw tests receive Holm
   adjustment across every measured axis/case. This is not standard CKA.
5. Direct native cosine with no dimensionality mapping, plus a fixed global
   coordinate permutation in every case and 999 coordinate permutations for
   the primary case.

Every case includes consistent candidate-identity swaps within categories and
one fixed within-group score shuffle in both partitions. Frozen primary
predictions additionally receive a 999-draw conditional validation-label null;
the 26 pretrained direct-cosine layer/pooling readouts receive a shared
maximum-statistic correction. This conditional shuffle does not model every
cross-group dependence, and its probabilities must be interpreted accordingly.

Macro Spearman uses the existing eligible-group rule and preserves undefined
constant-prediction groups as missing with explicit coverage. Comparisons use
mutually defined groups. Conditional intervals use 2,000 whole-split-unit
bootstrap draws. The 0.78 tolerance applies to sign statistics and direction
construction, not to Spearman eligibility. There is no population noise-ceiling
claim and no matched comparison to earlier full-cohort models.

## Reproduction and artifacts

`study.py prepare` freezes data membership and provenance. `extract_all.py`
downloads pinned model revisions and runs the four extraction workers.
`analyze_all.py` assigns the 52 cases to four disjoint workers and verifies them
before finalization. `report.py` performs the fixed additional
null checks, writes the anonymous tables and plots, scans shared text/schema,
and produces a ZIP with standardized metadata. Scientific commands enforce
the authorized execution directory and GPU environment.

Output is under `artifacts/frozen_embedding_diagnostics_20261114`. The `private`
directory contains original inputs, model specifications and mappings and is
excluded from the shareable bundle. Exact numerical values remain potentially
matchable to other copies. Existing untracked files and earlier experiment
outputs are untouched.

An initial input serialization mistakenly used the single-character format of
a different query checkpoint. The tokenizer audit identified unknown-token
collapse. Those exports and all derived analyses were rejected and archived
privately; sample membership was retained. Input-contract version 2 uses the
checkpoint's published vocabulary and prefix, records unknown-token counts,
and is required by the report writer. Preliminary query-dependent numbers from
the rejected extraction must not be interpreted as model evidence.

Local qualification:

```sh
PYTHONPATH=RewardModel OPENBLAS_NUM_THREADS=4 .venv/bin/python -m pytest -q \
  RewardModel/tests/test_frozen_embedding_sampling.py \
  RewardModel/tests/test_frozen_embedding_analysis.py
```

Tests cover label-independent membership, identity uniqueness, same-query
exclusion in reference construction, held-out-label independence of predictions,
within-group direction invariance, missing-overlap handling, metric coverage,
and conditional permutation statistics. These CPU tests are separate from the
actual frozen encoder exports and measured scientific results.

## Two requested decision follow-ups

`followup.py` scores the existing selected full B reference (epoch 6) with true
and identity-swapped queries on the original 955 eligible pilot groups. One
fixed derangement has no self-matches and is consistent across repeated query
occurrences. Candidates, outcomes, groups, precision and batch padding remain
fixed. Ranking changes and score offsets are reported separately. Similar
macro scores do not establish input invariance or equivalence.

The second measurement samples 50 queries with at least 40 eligible training
groups and five validation groups. Availability and sampling do not inspect
scores. Forty training groups and five validation groups, ten candidates per
group, are sampled for each query. Directions use the original frozen candidate
encoder and final-layer content mean. If m of a query's 40 groups contribute
pairs above the fixed 0.78 gap, its global control uses exactly m contributing
groups, one from each of m other selected queries. The global control uses only
training eligibility and one fixed draw; no query or layer is selected by
performance. A minimum of 24 contributing groups per selected query is checked
before inference. Failure does not silently replace the selected query.

Both measurements preserve predictions and independently verify group-wise
Spearman. Paired split-unit bootstrap intervals, query-cluster sensitivity and
equal-query differences are reported. The checkpoint and frozen pilot use
different encoder architectures and scoring paths, so a swap drop cannot isolate
fine-tuning as the cause. Existing training queues remain stopped. Artifacts
are isolated under `decision_followup`; the anonymous report includes no model
names or source inputs. Run with `PYTHONPATH=RewardModel` and
`python -m experiments.frozen_embedding_diagnostics.followup run` on the authorized
execution host. The three inference workers use GPUs 0, 2 and 5 only.
