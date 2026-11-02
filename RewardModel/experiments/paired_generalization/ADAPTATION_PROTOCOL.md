# E49–E52: fixed adaptation protocol

This covers the MD's head-only, head-plus-cross-attention, within-unit and
vary-k probes. It adds no intervention or learning-rate search. Data memberships
and k=1/2/4/8/16/64 are already frozen by `adaptation_design.py` and
`adaptation_views.py`. Preparation is not authorization to bypass Tier 3 or
production GPU qualification, and is not an adaptation performance result.

## Bases and optimization

For A and B, use the primary grid's full R+B+C model at its common
B-validation-selected learning rate, for seeds 42/43/44. Each base uses that
run's own best validation-macro checkpoint. No test score or winning alternative
architecture chooses a base. Checkpoints and all source/data receipts must be
resolved and verified after the relevant Tier-3 comparisons finish.

Inherit each selected full-run optimizer, learning rates, weight decay, warmup,
loss definitions/weights, precision and 15-epoch schedule. There is no additional
adaptation learning-rate or weight sweep. Train each adaptation model on one
permitted GPU, with batch 12 and accumulation 16: the original nominal effective
batch of 192 and local contrastive pool size are preserved. The four permitted
GPUs can execute independent models. Small supports produce shorter accumulation
windows; record actual optimizer steps, sampled lists and objective exposure.

The heads scope trains active ranking and BCE heads, contrastive projections
and learnable scale. The second scope additionally trains `fusion` and
`extra_fusion_layers`. Encoders, shared input projections and normalizations
remain fixed; frozen feature modules remain in evaluation mode. Disabled
objectives and a fixed scale remain disabled. Every support task starts from
the exact same base for its regime/seed. Within-unit supports never pool across
units, and adapted parameters never carry over to another task.

## Epoch selection and query isolation

For each kind (across/within), scope and k, fit the B seed 42 selection tasks
first. Across-unit models fit the k groups from the adaptation units and are
evaluated on the disjoint validation units. Within-unit selection fits a
separate model on each validation unit's support, then evaluates its reserved
query assays. Aggregate the latter as the unweighted mean of eligible assay
correlations, not the mean of per-unit means.

Select the maximum native float32-label validation macro Spearman over positive
epochs 1–15; an exact tie selects the earliest epoch. Undefined predictions keep
their explicit assay rows and reduce reported coverage, using the existing
headline metric's defined-correlation convention. Require all frozen eligible
assays to be represented at every epoch. If all epochs are undefined, record
that no epoch can be selected; do not substitute test scores or a zero metric.

Freeze that one epoch for the corresponding A/B comparison and seeds 42/43/44.
A validation and confirmation seeds may be reported but cannot choose it.
Within-unit test models fit only their own support; their query labels never
select an epoch. Keep the complete original 15-epoch learning-rate schedule
when retaining the preselected epoch; this is checkpoint selection, not a claim
that a shorter schedule or actual early stopping saved compute.

The unadapted k=0 base is evaluated on each identical query cohort. It is a
reported comparison, not an epoch candidate. No optimizer is constructed for
k=0. Validation/test evaluation happens outside the support-only fitting worker.
Support labels are recomputed inside each subset as specified by the frozen
views. In within-unit tests, support observations intentionally come from the
original test units; reserved query observations remain outside optimization.

## Coverage, accounting and reporting

Fit every available task at a budget with a rank-evaluable B validation cohort,
including units whose own query ranking is undefined. Report their coverage and
other applicable metrics. A has one assay per unit, so its within-unit arm is
structurally infeasible. B k=64 has no rank-evaluable validation cohort and
cannot select a ranking model; no substitute units or smaller k are used.
At B k=16, only one validation query assay is ranking-eligible. Report this
selection limitation explicitly rather than treating it as broad validation.

For each supported cell, report per-assay macro rho, eligible/defined counts,
all pathway scores, support size and objective exposure, and paired changes
against k=0. Across k, show both the available cohort and the intersection of
unit identities with unchanged query views. Keep differing prediction coverage
visible, including common-defined-assay sensitivity comparisons. Use paired
whole-unit bootstrap intervals within each seed and report seed spread
separately; repeated seeds are not independent units.

The preparation manifest counts every independent support fit, including all
three seeds and both trainable scopes; a within-unit task is not a full-dataset
training run. Compact parameter payloads require the pinned base checkpoint.
Per-unit receipts remain local; W&B logs aggregate progress/results at
kind/scope/k/regime/seed level to avoid a separate logging run per small unit.

Original validation units already contributed to base-model selection. These
are conditional mechanistic probes on this benchmark. A weak adaptation result
under inherited optimization settings is not proof that no head-training
configuration could transfer. Supports without ranking-eligible assays offer
no direct ranking supervision and must be reported separately. Seed spread
conditions on the frozen support memberships; it does not measure variation
from redrawing adaptation groups.
