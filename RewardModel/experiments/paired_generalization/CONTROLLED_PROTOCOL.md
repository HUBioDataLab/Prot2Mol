# Controlled training protocol

This document fixes operational choices before Tier 3 results are available.
The governing scope remains `Paired input ranking generalization.md` and every
item in `protocol.json`; this document does not replace or narrow that scope.

## Common comparison and tuning rules

- Deployment is an unseen generated molecule. Split B validation selects
  hyperparameters; Split A is a matched comparison, not the deployment selector.
- Use both partitions A and B, seeds 42, 43 and 44, and the same 15-epoch
  horizon, optimizer, schedule and effective batch as the fresh reference.
  The requested first reference remains unchanged and retains all 15 checkpoints.
- Each configuration arm receives three learning-rate candidates: multiply all
  its nonzero base learning rates by 0.3, 1 or 3. Train each candidate for the
  same full horizon at tuning seed 42 on both A and B. Select the candidate
  using maximum B validation macro Spearman, then run that same candidate on
  A and B at seeds 43 and 44. Keep seed-42 tuning and seed-43/44 confirmation
  results identifiable; seed-42 scores have hyperparameter-selection exposure.
- Ties select the lower multiplier. Nonfinite/undefined metric values are not
  winners. Report the defined fraction alongside the score; a loss-subset arm
  cannot win by silently dropping constant-prediction groups.
- Report A-versus-B comparisons for every matched candidate and for the common
  B-selected candidate, so independently tuned A/B settings do not create the
  apparent relationship. No test result controls training, selection or budget.
- For the weight sweep and multiple LoRA ranks, each declared setting is one
  arm and gets the same three-candidate budget. Report total family budgets as
  well; a family with several settings has more trials than a single arm.
- Use unit-level uncertainty and seed spread. Additional B realizations remain
  required subject to the specification's affordability condition; assess that
  with measured runtime and storage, not by silently omitting them.
- For A, binary labels use medians computed within each partition. The original
  pooled A transformation leaked held-out measurements into training labels.
  Continuous labels and the B reference transformation are unchanged.

## Encoder freedom and loss readouts

- Full, frozen, last-two-transformer-block unfreezing, and LoRA ranks 4, 16 and
  64 are separate settings. Partial unfreezing also trains ESM's final layer
  norm. Frozen embeddings/lower blocks stay in evaluation mode; adapted blocks
  train normally. LoRA adapts attention query/value matrices on both encoders,
  with alpha twice the rank, zero initial delta, and no additional dropout.
  Original encoder dropout remains active in LoRA/full training.
- Fork the initialization RNG around adapter creation, preserving the common
  initial projection, fusion and head weights across encoder-freedom arms.
- The model has separate ranking, BCE and contrastive scoring pathways. For
  {R}, {RB}, {RC}, {RBC}, use the ranking head. For {B}, {BC}, use the BCE logit.
  For {C}, use cosine similarity. Persist this policy in the checkpoint config
  and use it for validation selection, evaluation, export and deployment.
- Freeze modules belonging exclusively to disabled objectives. Do not change
  the common architecture or initialize a different head for a subset arm.
  Fixed contrastive scale stays at the original 13. Stop-gradient detaches
  encoder outputs only before the separate contrastive projections; R/B still
  update the encoders and C still updates its projection heads.
- Set `data.sample_all_assays: true` for every controlled arm. Otherwise the
  historical sampler switches from all assays to ranking-eligible assays when
  C is disabled, changing list composition, dataset-item count and update count
  along with the objective. Preserve the reference's full row coverage and
  classification-row packing even in arms where B is disabled.
- Score all trained pathways separately. Comparisons of R versus RB/RC/RBC
  isolate auxiliary-objective effects on the same readout. The entire seven-arm
  grid compares complete scoring systems; a factorial interaction computed
  across changing readouts is not evidence of gradient conflict by itself.

## Runtime qualification and retention

`qualify_controls.py` exercises all seven loss subsets plus frozen, partial,
LoRA, fixed-scale and stop-gradient settings through distributed forward,
backward and three optimizer updates with accumulation/no_sync. It compares
fitted parameters across ranks. Tiny ESM/RoBERTa CPU qualification is followed
by actual cached-encoder qualification on the authorized GPU UUIDs before
starting these new controlled arms. These checks establish execution, not
scientific performance or sufficient statistical power.

## Ranking controls

The reference is a top-weighted Plackett-Luce selection loss, not a sum of
pairwise logistic losses. Its three-item-or-larger sampled lists are averaged,
but larger original assays generate more lists. These distinctions are part
of the comparison, not interchangeable uses of the word "group".

- Within-list normalization divides the reference's sum by the retained
  selection-weight sum before averaging lists. A selection is retained only
  when at least one weaker item lies beyond the affinity margin.
- Pairwise top and uniform controls both use one logistic term per unordered
  pair beyond the same affinity margin. Top weights use the higher-label item's
  tie-averaged reference position weight. Divide by the retained pair-weight
  sum within each list. Comparing these two isolates pair weighting; comparing
  either to the reference also changes the loss family.
- A zero-margin hard-pair control precedes the soft-pair comparison. Soft
  targets are sigmoid(label difference / T), include near ties and give exact
  ties probability 0.5. Hard exact ties are omitted.
- Fix T from each training partition's raw replicate-difference RMS d using
  T = sqrt(3) d / pi, which matches the variance pi^2 T^2 / 3 of a logistic
  difference distribution to d^2. This is a noise-calibrated sensitivity model,
  not a claim that median-aggregated labels have the uncertainty of individual
  raw measurements. Validation/test measurements do not determine T.
- Training and full-dataset evaluation reconstruct the same configured ranking
  objective. The metric eligibility rule remains fixed for comparability.

Original-assay balancing is a separate setting in the per-group comparison.
Let N be training dataset items, G eligible original assays and o_g the number
of sampled lists from assay g in an epoch. Its normalized list loss is weighted
by N / (G o_g), summed in each microbatch and divided by that microbatch's item
count. At fixed parameters, the complete epoch's item-weighted mean is exactly
the macro assay mean of its sampled-list losses. This corrects multiplicity
without changing sampled rows, batches or optimizer-update count. It does not
make a sampled subset of pairs equal to every pair in a large assay.
Validation instead averages its complete-coverage list losses within each
original eligible assay, then averages assays and fixed partitions.

## Contrastive masking controls

Compare the reference, no mask, and the reference plus all training-known
positive target/molecule pairs. The reference already hides same-assay
co-candidates and cross-assay same-target/duplicate-query-molecule candidates.
The no-mask setting removes this whole bundle; its comparison is not an
isolated estimate of one masking mechanism. The added training-known mask
isolates the remaining observed false negatives found in the sampling audit.

Training-known means any curated observation in the actual training partition
strictly above pChEMBL 6 after conversion to the float32 supervision dtype.
Do not add validation/test labels or original measurements that were excluded
from training. This conservative rule can mask pairs with conflicting training
assay results; it treats them as potentially positive, not confirmed negatives.
Keep supervised targets. Record the canonical index digest, threshold, source
row count and pair count with each run. The training collator and full-dataset
loss evaluator share this index. Evaluation maps identities into its own
integer namespace without consulting its labels. Saved models score new inputs
without the index, which is used only for contrastive loss construction.

Controlled retention must preserve both selection rules, the trajectory
measurements and the averaging candidates. It must fit the authorized storage
quota without deleting original data or the first reference's checkpoints.
Before controlled-run checkpoint cleanup, the implemented retention/export
checks must first demonstrate that required artifacts survive.
Training with both selection metrics will distinguish
best-within-15 selection from an actual patience-based early-stopping rule.

The static weight grid, leak-in/out design, unit learning-curve fractions,
co-positive masking audit, averaging rule, and Tier 4
adaptation schedule still require their explicit manifests and qualification.
Nothing here labels those experiments complete.
