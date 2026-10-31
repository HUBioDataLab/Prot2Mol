# Diagnose the split before blaming the model

The observations do not yet show that the model generalizes badly. They show that two evaluation regimes, which differ in at least five ways, produce different numbers, and that a composite validation loss turns upward earlier on one of them. A drop from 0.41 to 0.20 is within the range the pair-input literature reports when evaluation moves from pairs whose components were both seen in training to pairs with one unseen side. On one drug–target benchmark, AUC-PR fell from **82.9 to 36.1** under exactly that change ([Pahikkala et al. 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4364066/)). The size of the drop is therefore not evidence of a defect. Three cheap checks decide how to read it. The first is whether "Spearman" is averaged per group: an untied two-item group can only score ±1, and a three-item group only ±1 or ±0.5. The second is whether Split B's test partition differs from Split A's in group size, label rate, within-group variance and replicate noise. The third is how much of Split A's 0.41 a model that sees only y can reproduce. Each group contains a single x, so any x main effect cancels within the group, and within-group order depends only on the y main effect and the x×y interaction (inference). The plan below therefore runs no-training audits and baselines before any controlled retraining or mechanistic probe, and each step is keyed to a table that says which result supports which hypothesis. The methods literature mostly reports null results for specialized machinery. Domain-generalization penalties lose to tuned empirical risk minimization on molecular scaffold splits, and gradient-surgery optimizers match tuned loss weighting but do not beat it. Its positive results come from vision, text retrieval, web search and recommendation: keeping encoders close to their pretrained weights, weight averaging, multi-positive contrastive targets, and ranking losses whose optima agree with the pointwise loss. None was tested on within-group ranking with groups of two or three items under held-out entity regions. No method here can be expected to close the gap. Part of the gap is irreducible if most of the learnable within-group variance is a y main effect that does not transfer across similarity regions.

## A halving is ordinary for warm-to-cold shifts, but five confounds hide its cause

Split A and Split B are best read as two cells of the pair-input evaluation taxonomy, not as one task at two difficulty levels. Park and Marcotte classified test pairs by whether both components, one, or neither appeared in training. **More than 99% of each randomly generated test set fell in the both-seen class**, so random-split estimates tracked the easiest regime almost exactly. For asymmetric pairs they recommended reporting four cells rather than one number ([Park & Marcotte 2012](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC3531800/fullTextXML)). Stock et al. formalized the same distinction as Settings A–D: both objects seen, a new row object, a new column object, or neither seen. They showed that Setting A reduces to a transductive problem, which "explains why Setting A is the most easy setting" ([Stock et al. 2018](https://arxiv.org/abs/1803.01575)).

In that vocabulary, Split A is Setting A, or Setting A by proxy wherever a test y's similarity neighbours sit in training. Split B is Setting C at the level of indivisible split units, because x recurs across partitions (inference).

Published magnitudes put a halving in the ordinary range. Moving from both-seen to drug-unseen pairs on one benchmark, Pahikkala et al. report the concordance index falling from **96.0 to 83.7** ([Pahikkala et al. 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4364066/)). SPECTRA reports a fine-tuned protein language model falling from **Spearman above 0.9 to below 0.4** as cross-split overlap shrank ([Ektefaie et al. 2024](https://www.biorxiv.org/content/10.1101/2024.02.25.581982.full.pdf)). This evidence comes from biology and chemistry. It shows that the gap is unremarkable in size, not what causes it here. Pahikkala et al. also found that which cold direction hurts more depends on the asymmetry in object counts. With roughly 125 distinct y per distinct x, the held-out side here is the large, sparsely observed one (inference).

What the observations leave open matters more than what they show. Five gaps stand out.

1. **The splits differ in more than y similarity.** The realized proportions, 86/7/7 against a nominal 80/10/10, show that the constraint system pushed the split off target. DataSAIL's authors document that two-dimensional splitting silently drops interactions whose components land in different folds ([Joeres et al. 2025](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11978981/fullTextXML)). Petti and Eddy showed that **splits satisfying identical similarity criteria still produce benchmarks of different difficulty**, and that cluster-based splits place test items farther from their nearest training neighbour ([Petti & Eddy 2022](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8929697/fullTextXML)).
2. **"Spearman" is unspecified.** A per-group average and a pooled correlation answer different questions, and a pooled value is invalid when scores are comparable only within groups.
3. **"Validation loss" is a weighted sum of three terms.** Each term can rise while the ranking stays fixed.
4. **Nothing shows that 0.41 measures pair interaction.** A model that learned only "which y regions score high" could earn much of it, because the x main effect cancels inside a group.
5. **Neither number carries an uncertainty estimate.** There is no noise ceiling and no spread across split realizations.

A few derivations from the problem statement shape every later step. They are arithmetic, not citations, and should be checked against the actual data. About 2 million observations over 8,000 x gives **roughly 250 observations and 20 groups per x**. About 1 million y gives **about two observations per y**. On an untied group of two, per-group Spearman can only be ±1. On three items it can only be **{1, 0.5, −0.5, −1}**, with a standard deviation near 0.71 under random ordering. A per-group average over groups with a median size of three is therefore dominated by coarse, high-variance values. Summing a pairwise loss over all pairs gives a 25-item group 300 pairs against 3 for a three-item group, **a 100-fold weight difference**. Per-group averaging weights the two groups equally.

Throughout this report, *inference* marks such derivations. *Hypothesis* marks untested claims about this setup. Literature findings carry the domain they came from.

## Thirty-four questions separate measurement artifacts from model failures

The questions are grouped by failure category, and the order matters. Metric and composition questions come first because their answers decide whether the remaining comparisons mean anything. Each table states why a question matters and what different answers would point to.

### Metric definition decides whether 0.41 and 0.20 are comparable

The literature gives little direct guidance here. This research found no peer-reviewed source on per-group rank correlation at n = 2–3, or on macro- versus micro-averaging of grouped rank metrics, so most of this block rests on derivation. Two established facts still apply. As usually implemented, Spearman is Pearson correlation on ranks with no tie correction. Kendall's tau-b adjusts its denominator for ties, and SciPy recommends permutation tests for small tied samples ([SciPy](https://docs.scipy.org/doc/scipy/tutorial/stats/hypothesis_kendalltau.html)). Pahikkala et al. also found that **binarizing at stringent thresholds raised accuracy in every setting**. That is a warning that the 70%-positive threshold label will flatter the model relative to its continuous ranking ([Pahikkala et al. 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4364066/)).

| Question | Why it matters | What different answers point to |
|---|---|---|
| Is the reported Spearman computed per group and averaged, weighted by group size or pair count, or pooled across all observations? | Pooling mixes within-group order with cross-group score offsets, which a translation-invariant ranking loss never pins down ([Yan et al. 2022](https://research.google/pubs/scale-calibration-of-deep-ranking-models/)). A per-group average gives a two-item group, which carries one bit, the same weight as a 25-item group with 300 pairs (inference). | **Pooled:** part of both numbers measures the cross-group calibration of a head that was never trained for it. Evaluate the BCE logit or an anchored score instead. **Per-group average:** the value is driven by groups of two or three, so compare it with a pair-weighted concordance before reading anything into the plateau. |
| Which groups enter the metric? How are size-one groups, constant groups and ties created by median aggregation handled? Does the eligible fraction differ between splits? | Spearman is undefined for constant groups. The median of an even number of replicates can create exact ties. An eligibility rule that differs from the training rule scores pairs the loss ignored. | If the eligible fraction or tie rate differs between the two test partitions, part of the gap is compositional rather than a failure to generalize. |
| Do the two test partitions differ in group-size distribution, within-group score variance, near-tie rate or replicate count? | Each of these bounds the attainable per-group correlation, independently of the model (inference). | If Split B's groups are smaller, flatter or noisier, reweighting Split A's per-group results to Split B's mix should reproduce part of the drop without any retraining. |
| How large is the spread over seeds, over a bootstrap of indivisible split units, and over alternative Split B realizations? | Split realizations built to identical criteria differed in difficulty while preserving method rankings ([Petti & Eddy 2022](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8929697/fullTextXML)). A sparse learning-rate grid produced a standard deviation 6–7 times the seed-to-seed value ([Xin et al. 2022](https://arxiv.org/abs/2209.11379)). | A realization spread near the 0.21 gap means the gap itself is poorly estimated. Comparisons within one realization, such as model against baseline, stay trustworthy either way. |
| Does the metric tolerate small score differences the way the loss does? How do top-weighted metrics move? | The loss deliberately declines to order near-ties and emphasizes the top of each group. Spearman weights every pair equally. | A large difference between Spearman and a tie-aware or epsilon-band concordance means the metric and the loss disagree about what counts as an error. If top-k metrics keep improving after Spearman plateaus, the loss is optimizing something the metric does not reward. |

### Overfitting: find out which loss term turns first

Guo et al. established that modern networks are poorly calibrated. They also showed that temperature scaling repairs most of the miscalibration, and because temperature scaling is monotone it cannot change any ranking ([Guo et al. 2017](https://arxiv.org/abs/1706.04599)). A rising BCE log-loss can therefore coexist with an unchanged ordering. Guo et al. are often cited as showing that NLL rises while classification error keeps falling during training; this research did not verify that claim against the paper body. DomainBed's authors argue that a method without a stated model-selection rule is incomplete, because the selection rule alone moves reported out-of-distribution performance ([Gulrajani & Lopez-Paz 2021](https://arxiv.org/abs/2007.01434)).

| Question | Why it matters | What different answers point to |
|---|---|---|
| Which term of the composite validation loss rises first on each split: BCE, the ranking loss, or each direction of the contrastive loss? Does BCE AUC move with BCE log-loss? | The aggregate is dominated by whichever term has the largest magnitude times weight. Each term can rise for reasons unrelated to ranking. | If BCE log-loss rises while AUC and Spearman stay flat, that is the calibration pattern and it does not harm ranking. If the ranking term's own loss rises, the model is overfitting the target. If logits grow uniformly, log-loss rises sooner on the split where discrimination is weaker, which alone could produce a turning point at 10% on one split and 30% on the other (inference). |
| Are checkpoints selected on aggregate loss or on validation Spearman? Where do the best-loss and best-Spearman checkpoints fall on each split? | Selection is part of the method. | If the best-Spearman checkpoint comes much later than the best-loss checkpoint on Split B, the early turning point is a selection artifact, not the model's limit. |
| How do training and validation ranking metrics diverge over time on each split? | This separates fitting non-transferable features from a limit on signal or capacity. | Training Spearman rising while Split B stays flat from early on suggests fitting specific to training units. Both flat suggests little learnable signal. |
| Does the early rise survive with frozen encoders? | Full fine-tuning with randomly initialized heads distorts pretrained features ([Kumar et al. 2022](https://arxiv.org/abs/2202.10054)). | If the rise disappears or comes later with frozen encoders, encoder drift drives it. If it persists, the interaction module or the heads drive it. |
| Does the learned contrastive temperature keep sharpening? Does the contrastive validation loss still rise against a fixed candidate pool? | InfoNCE values depend on the number and similarity of in-batch candidates ([van den Oord et al. 2018](https://arxiv.org/abs/1807.03748)) and on temperature ([Wang & Liu 2021](https://arxiv.org/abs/2012.09740)). | A rise that vanishes under a fixed pool or a fixed temperature is an artifact of batch composition or overconfidence. A rise that persists alongside a widening gap in positive-pair alignment between training and validation indicates memorization. |
| Is the rise concentrated in particular group sizes or similarity strata? | This localizes where fitting fails to transfer. | Concentration in low-similarity groups supports fitting that is specific to the training distribution. A uniform spread supports calibration drift. |

### Distribution shift: establish what changed besides similarity

| Question | Why it matters | What different answers point to |
|---|---|---|
| Which split matches deployment? Will future y mostly be near neighbours of known y, or come from new regions? | Park and Marcotte accept random splits only when the warm regime is the sole regime of interest. DataSAIL's authors say leakage-reduced splitting is the wrong choice when application data resemble training data ([DataSAIL](https://github.com/kalininalab/DataSAIL)). | **Warm deployment:** 0.41 is the relevant number and Split B is a stress test. **Cold deployment:** Split B is the target, and Split A should not drive tuning. |
| How is each test y's maximum similarity to training y distributed in both splits, and how does Split A perform in each similarity stratum? | One recent similarity-aware evaluation found only **16 of 873** random-split test samples below one-third similarity, with performance falling sharply among them ([Zhang et al. 2025](https://arxiv.org/abs/2504.09481v1)). | If Split A's lowest stratum scores near 0.20, Split B measures a stratum that Split A barely contains, and there is no anomaly to explain. If that stratum stays near 0.41, something other than similarity drives the drop. |
| What is the size distribution of indivisible units in each partition? Were units assigned at random, or did large units default to training? | Single-linkage grouping lets a few "bridge" members merge otherwise dissimilar groups, and larger families fail to split more often ([Petti & Eddy 2022](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8929697/fullTextXML)). | A giant training unit alongside small, peripheral test units means Split B's test set was selected for remoteness. Non-random assignment also breaks the exchangeability between validation and test units that checkpoint selection relies on. |
| Do the positive rate, score distribution or group descriptors differ across partitions, and does a formal test flag the difference? | Label shift alone can move Spearman without any change in inputs. BBSE tests for label shift using any black-box classifier whose confusion matrix is invertible ([Lipton et al. 2018](https://proceedings.mlr.press/v80/lipton18a.html)). Classifier two-sample tests show where two samples differ ([Lopez-Paz & Oquab 2017](https://arxiv.org/abs/1610.06545)). | A significant shift in labels or group descriptors assigns part of the gap to composition. It also argues for re-splitting with class-ratio constraints, as DataSAIL does. |
| Which x appear in Split B's test groups, and how many training groups does each have? | x is assumed to be warm, but thin coverage of x turns part of Split B into the harder neither-seen setting. | If test x have few training groups, part of the gap comes from coldness on the x side. Report Split B separately by each x's training count. |
| Across model variants, do Split A and Split B scores move together? | Out-of-distribution accuracy often tracks in-distribution accuracy linearly ([Miller et al. 2021](https://arxiv.org/abs/2107.04649)). The relation is "largely dataset-dependent" ([Wenzel et al. 2022](https://arxiv.org/abs/2207.09239)) and sometimes inverse ([Teney et al. 2022](https://arxiv.org/abs/2209.00613)). | A tight line means tuning on Split A is a valid proxy. A flat or inverse relation means tuning on Split A actively misleads. All of this evidence is classification accuracy on images, so applying it to grouped rank metrics is an extrapolation. |

### Supervision limits: measure the learnable, transferable signal

| Question | Why it matters | What different answers point to |
|---|---|---|
| What is the replicate-based noise ceiling of the chosen metric on each split? | A noise ceiling marks the point where "any remaining mismatch is then attributable to the noise in the measurements" ([Nili et al. 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC3990488)). The replicates that are currently aggregated by median make it computable: split each pair's replicates into halves, score one half against the other with the headline metric, and apply a Spearman–Brown correction. | If Split B's ceiling is lower, part of the gap is a difference in measurement quality, and the right report is the fraction of the ceiling reached. Two caveats apply. Spearman–Brown assumes parallel measurements ([SAGE](https://sk.sagepub.com/ency/edvol/embed/the-sage-encyclopedia-of-research-design-2e/chpt/spearman-brown-prophecy-formula)), and this research found no ML precedent for a noise ceiling on a grouped rank metric. |
| How does within-group variance of s divide among the y-cluster main effect, the x×y interaction, and replicate noise? | Generalizability theory estimates exactly this interaction component, and it needs replication to separate the interaction from residual noise ([G-theory notes](https://marklhc.quarto.pub/psyc520-2025spring/slides/07-g-theory.html)). Cross-attention can only add value on the interaction share (inference). | A small interaction share means Split A's 0.41 rested largely on y main effects that do not transfer across similarity regions. In that case 0.20 sits near the cold-y limit for any architecture (hypothesis). Because most y are seen about twice, fit the model at the y-cluster level. |
| What fraction of within-group pairs differ by less than the replicate noise? | Near-ties add noise to both training and evaluation. | A large fraction lowers the ceiling and argues for tie-tolerant losses and metrics. |
| Why do only half the groups support ranking supervision, and does that half differ in size, variance or x? | The ranking head learns from a subset that may not represent the evaluation groups. | A systematic difference means the ranking head is trained on a different distribution from the one it is evaluated on, even on Split A. |
| How were pairs chosen for observation, given the 70% positive rate? | If pairs were observed because they looked promising, unobserved pairs are unlabeled, not negative. | If observation was selected by promise, in-batch "negatives" are partly positive. That changes how contrastive negatives and baselines should be built. |
| How many groups have all members on one side of the threshold, and how far is the threshold from the replicate noise? | BCE carries no within-group ordering information for items on the same side of the threshold. | If many groups are all-positive, BCE contributes little to the ranked target and mostly shapes calibration and x priors. |

### Sampling: check how split units, batches and negatives are drawn

| Question | Why it matters | What different answers point to |
|---|---|---|
| Are batches built from whole groups or from observations, and is the fraction of ranking-eligible groups per batch fixed? | The ranking loss is defined on only about half the groups, so its effective weight fluctuates with batch composition. | Large fluctuation means adaptive loss-weighting methods read noise. Use stratified batches before comparing objectives. |
| Does the contrastive loss treat other positives of the same x in the batch as negatives? | With group batching and diagonal-only targets, a positive anchor in a 25-item group at 70% positives sees about 16–17 labeled positives as negatives (inference). Standard two-tower practice masks only identical candidate IDs ([TFRS](https://www.tensorflow.org/recommenders/api_docs/python/tfrs/tasks/Retrieval)). | If co-positives are treated as negatives, the contrastive loss directly opposes the BCE and ranking losses. Masking them, or using multi-positive targets, is the first fix to test. |
| Are the contrastive projections taken before or after cross-attention? | Pair-independent embeddings allow cheap in-batch negatives and a full softmax over all 8,000 x. Pair-dependent embeddings require recomputing cross-attention for every off-diagonal pair. | This architectural detail determines which contrastive remedies are feasible. |
| How is the listwise loss normalized across groups of different sizes? | Summing over pairs lets the roughly 10% of large groups dominate the gradient. Per-group averaging matches a per-group-averaged metric (inference). | If loss weighting and metric weighting differ, the model is optimizing groups the metric barely counts. |
| Are x frequencies in batches skewed, and is any sampling-probability correction applied? | Uncorrected in-batch softmax is biased under power-law frequencies ([Yi et al. 2019](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/)). The y-to-x direction can learn an offset for x popularity (inference). | A strong x-popularity component in the contrastive branch is a shortcut that needs no information about y. |
| Are subsets for ablations and learning curves drawn by indivisible unit? | Subsampling by observation leaks similar items back into training. | Only subsampling by unit gives interpretable learning curves on Split B. |

### Objective conflicts: test whether the three losses want the same ordering

| Question | Why it matters | What different answers point to |
|---|---|---|
| What ranking quality does each loss deliver alone, and in each pair, on each split? | A loss-subset grid measures each loss's contribution and their interactions directly. Arms without the ranking head are scored by the BCE logit or the contrastive similarity. | An interaction term that is positive on Split A and negative on Split B means the auxiliary losses help in-distribution but hurt transfer (hypothesis). |
| How well do the ranking head, the BCE logit and the contrastive similarity agree within groups, and does their agreement fall during training? | This quantifies the stated mismatch between ranking order and contrastive geometry. | Near-zero within-group correlation between contrastive similarity and s means the contrastive branch encodes cross-group compatibility, not within-group order. |
| How much of the BCE head's validation AUC does an x-only model reproduce? | Within-group Spearman ignores anything constant within a group, but BCE rewards per-x base rates (inference). | If an x-only model recovers most of the AUC, BCE mostly teaches x priors that do not affect within-group order. |
| Do the auxiliary gradients align with the ranking gradient in the encoders and in cross-attention? | Gradient conflict flags interference in optimization, but low conflict does not predict good generalization ([Royer et al. 2023](https://arxiv.org/abs/2310.08910); [Mueller et al. 2025](https://arxiv.org/abs/2408.14677)). | Frequent conflict points to a problem. Its absence rules nothing out. |
| Were the loss weights tuned, and on which split's validation data? | Weights tuned on random-group validation optimize for the warm regime. | If the weights change when tuned on Split-B-style validation data, the value of the auxiliary losses depends on the shift. |

One more observation cuts across these blocks. Validation loss keeps improving on Split A until about 30% of training, but turns at about 10% on Split B. Pure gradient conflict between objectives would be expected to hurt both splits, so a turn this early and specific to one split fits shift-specific fitting or the calibration mechanism better (hypothesis).

## Four tiers of diagnostics, with training only after the audits

Each tier conditions the interpretation of the next, and every comparison is made within one split realization. For confidence intervals, resample whole indivisible units, because those are the blocks across which assignment is not independent (inference). Bootstrap schemes that respect this kind of dependence scale well to large datasets ([Uncertainty in online experiments with dependent data, 2013](https://arxiv.org/pdf/1304.7406)).

### Tier 1: audits on existing predictions, data and checkpoints

The first tier needs no training.

1. **Re-score saved test predictions.** Compute each of these metrics:
   - per-group averaged Spearman;
   - a pooled within-group concordance (concordant minus discordant pairs, over all within-group pairs);
   - Kendall tau-b;
   - an epsilon-band concordance that scores pairs within replicate noise as ties;
   - pairwise ordering accuracy by group-size bin (2, 3, 4–10, 11–25, above 25).
2. **Audit composition across all six partitions.** Compare group sizes, ranking-eligible fraction, positive rate, score distribution, within-group variance, replicate counts and x coverage. Also tabulate the sizes of Split B's indivisible units.
3. **Stratify Split A by similarity.** Compute each test y's maximum similarity to training y under the predefined relation, stratify Split A's results by it, and reweight Split A's per-group results to Split B's group-size and label mix. If the relation is graded, stratification is possible; if it is binary, only a two-class comparison is.
4. **Estimate the noise ceiling and variance components.** Compute the replicate-based noise ceiling for each split, in the same units as the headline metric. Fit a variance-components model at the y-cluster level. Nested designs limit which components can be estimated ([G-theory notes](https://marklhc.quarto.pub/psyc520-2025spring/slides/07-g-theory.html)), and group and cluster effects will be collinear if groups are homogeneous in cluster membership.
5. **Examine saved checkpoints.**
   - Decompose the validation loss by term and by contrastive direction.
   - Compute BCE AUC alongside BCE log-loss.
   - Refit a temperature for the BCE logit on validation data.
   - Log the learned contrastive logit scale.
   - Re-evaluate the contrastive loss against a fixed candidate pool.
   - Locate the best-Spearman checkpoint.
6. **Test for shift formally.** Run a classifier two-sample test on group descriptors and BBSE on the label marginal.

If checkpoints were not saved, step 5 moves into Tier 3.

### Tier 2: baselines that need no deep training

The baseline ladder is the most diagnostic of the cheap experiments.

**A y-only model**, scored by within-group Spearman, measures how much of the ranking needs no interaction at all. Paired-input fields have documented this shortcut before:
- In natural language inference, models that saw only the hypothesis labelled about **67% of SNLI** correctly ([Gururangan et al. 2018](https://arxiv.org/abs/1803.02324)).
- Protein–ligand binding models were shown to exploit how many annotations each entity had rather than the entities' features ([Chatterjee et al. 2023](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10082765/)).

**Shuffling and lookup baselines.**
- Add an x-shuffled model, and an x-only model scored by BCE AUC.
- Add a **nearest-neighbour lookup** that predicts a test y's score from its most similar training y under the predefined relation. On Split A this approximates what memorizing neighbours can achieve. Its drop on Split B measures the leakage share of the gap with no learned model involved (inference).

**A frozen-embedding ladder** follows. Stock et al. prove that independent-task ridge regression, two-step kernel ridge regression and a linear matrix filter are all special cases of Kronecker kernel ridge regression. They also recommend separate regularization for the two sides when one side explains most of the variance ([Stock et al. 2018](https://arxiv.org/abs/1803.01575)). Add a bilinear inductive matrix completion model ([Jain & Dhillon 2013](https://arxiv.org/abs/1306.0626)). At a million y, exact kernel methods need random-feature or Nyström approximations.

**A LambdaMART baseline on frozen features.** On web-search benchmarks, most published neural rankers did worse than a tuned LightGBM LambdaMART baseline ([Qin et al. 2021](https://research.google/pubs/are-neural-rankers-still-outperformed-by-gradient-boosted-decision-trees/)). Configure it to match Spearman rather than NDCG ([LightGBM](https://lightgbm.readthedocs.io/en/latest/Parameters.html)):
- integer within-group levels whose width matches the replicate noise;
- linear rather than exponential label gain;
- per-query normalization on;
- a truncation level at or above the largest group size.

**A factorized baseline.** Score Split B groups with the current model's contrastive similarity alone.

Kapoor and Narayanan surveyed leakage across **at least 294 papers in 17 fields** and report that complex models often stopped beating decades-old simple models once leakage was corrected ([Kapoor & Narayanan 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856)). This ladder is how to run that check here.

The dependency is explicit. Every frozen-embedding rung assumes that the pretrained encoders' geometry supports neighbourhoods and kernels relevant to s. That depends on the withheld modality and encoders. A weak result on these rungs is uninformative rather than reassuring.

### Tier 3: controlled retraining

Every arm gets the same tuning budget, at least three seeds, both splits and, if affordable, at least three Split B realizations. The arms are:

- **(a) Selection rule:** early stopping on validation Spearman versus on aggregate loss.
- **(b) Encoder freedom:**
  - frozen encoders, with cross-attention and heads trained;
  - partial unfreezing;
  - low-rank updates at several ranks;
  - full fine-tuning.
- **(c) Loss subsets:** the grid {R}, {B}, {C}, {R,B}, {R,C}, {B,C}, {R,B,C}, plus a static weight sweep, where R is the ranking loss, B the BCE loss and C the contrastive loss.
- **(d) Leak-in versus leak-out (2×2):**
  - Hold the test groups fixed and vary only whether their similarity relatives appear in training.
  - Subsample the leak-in arm to match the leak-out arm's size, group-size mix, positive rate and replicate counts.
  - No published precedent was found, so this is a novel ablation. It is motivated by the confound that Petti and Eddy and DataSAIL document.
- **(e) Learning curves:** subsample by whole unit and read the curves as slopes. Learning curves can be non-monotone, so a dip is not evidence of a bug ([Viering & Loog 2023](https://arxiv.org/pdf/2103.10948)).
- **(f) Contrastive variants:**
  - masked same-x co-positives;
  - fixed versus learned temperature;
  - stop-gradient from the encoders.
- **(g) Ranking variants:**
  - per-group normalization;
  - uniform pair weights versus top-weighting;
  - soft pair targets whose temperature is set from the replicate noise.

### Tier 4: mechanistic probes

**Staged head retraining across units.** Hold out three disjoint unit sets: one for adaptation, one for validation, one for test.
1. Retrain only the heads on k groups from the adaptation units, and evaluate on the test units.
2. Repeat, also retraining cross-attention.
3. Repeat within single units: adapt on some groups of a unit and evaluate on other groups of the same unit.
4. Vary k across all three.

This adapts last-layer retraining from the spurious-correlation literature. There, features from plain ERM often already sufficed, and retraining only the last layer recovered worst-group accuracy of **97% on Waterbirds** but **only 50% on FMoW** ([Izmailov et al. 2022](https://arxiv.org/abs/2210.11369); [Rosenfeld et al. 2022](https://arxiv.org/abs/2202.06856)). The idea does not transfer universally.

**Further probes.**
- **Pathway scoring:** score each pathway separately on both splits.
- **Representation geometry:** measure alignment and uniformity ([Wang & Isola 2020](https://arxiv.org/abs/2005.10242)) and effective rank ([Garrido et al. 2023](https://arxiv.org/abs/2210.02885)), per side, per split, and before and after the projections. These measures are label-free and were validated for self-supervised representation quality, not for ranking held-out entities.
- **Gradient diagnostics:** compute per-block gradient cosines, and a lookahead transference score in the style of TAG ([Fifty et al. 2021](https://arxiv.org/abs/2109.04617)) measured against Split-B-style validation batches. TAG itself measured affinity on training batches.
- **Accuracy on the line:** plot Split A against Split B across every variant trained in Tier 3.

### Which result supports which hypothesis

| Hypothesis (failure category) | Signature if true | Signature if false | Cheapest discriminating step |
|---|---|---|---|
| Metric or composition artifact (evaluation) | Split A reweighted to Split B's composition falls well below 0.41. Per-group and pair-weighted metrics diverge. Split B's noise ceiling is lower. | The reweighted Split A score stays near 0.41 and the ceilings match. | Tier 1 re-scoring and audit |
| Split A inflated by neighbour leakage and y main effects (shift) | The y-only and nearest-neighbour baselines recover much of 0.41 on Split A and little on Split B. Split A's lowest-similarity stratum scores near 0.20. The leak-out arm drops toward Split B. | The y-only baseline scores near zero on Split A. Stratified Split A is flat across similarity. | Tier 1 stratification, Tier 2 ladder, Tier 3 2×2 |
| Split construction selects remote test items (shift, sampling) | One or a few giant units sit in training while the test set holds small peripheral units. Test-to-train similarity is lower than the threshold alone implies. The Split B score varies widely across realizations. | Unit-size distributions are similar across partitions, and spread across realizations is low. | Tier 1 unit audit, Tier 3 realizations |
| Validation-loss rise driven by calibration (benign overfitting of the loss) | The rise comes from BCE log-loss while BCE AUC and Spearman stay flat. Temperature scaling removes most of it. The best-Spearman checkpoint comes after the best-loss checkpoint. | The ranking term's own loss rises and Spearman declines. | Tier 1 checkpoint decomposition |
| Encoder drift distorts pretrained features (overfitting) | Frozen or low-rank encoders match or beat full fine-tuning on Split B while losing on Split A. The early rise disappears or comes later. | Frozen encoders are worse on both splits by similar margins. | Tier 3 arm (b) |
| Supervision ceiling (supervision) | Split B has a low noise ceiling. The x×y interaction component is small. The Split B learning curve is flat early. Within-unit few-shot works but cross-unit few-shot does not. | The ceiling is well above 0.20, or the Split B curve is still rising. | Tier 1 ceiling and decomposition, Tier 3 curves |
| Too little data from cold regions (supervision, sampling) | The Split B curve is still rising at full data. Retraining heads on modest k across units helps. | Flat curve; few-shot does not help. | Tier 3 curves, Tier 4 probes |
| Objective conflict (conflict) | The interaction term in the loss-subset grid is negative on Split B only. Detaching C or B from the encoders helps Split B. Agreement among the three scorers falls during training. | The ranking-only arm does no better on Split B, or the auxiliary losses help Split B. | Tier 3 arm (c), Tier 4 scorer agreement |
| Contrastive sampling artifacts: false negatives, temperature (sampling) | Masking co-positives or fixing temperature changes Split B. The logit scale climbs. The contrastive loss rise vanishes under a fixed candidate pool. | No change under masking, fixed temperature or a fixed pool. | Tier 1 logit-scale log and fixed-pool evaluation, Tier 3 arm (f) |
| Representation suffices but the heads or interaction fail to transfer (overfitting, shift) | Retraining only the heads on a few held-out units lifts other held-out units. Frozen features with a simple head come close on Split B. | Head retraining does not help. Probes show the y representation lacks the information. | Tier 4 staged probes |
| Ranking objective misaligned with the metric (supervision, conflict) | Uniform pair weights or per-group normalization raises Spearman. Top-k metrics move while Spearman stays flat. | No change. | Tier 3 arm (g) |
| Split A does not predict Split B (selection) | Across variants, Split A and Split B scores are uncorrelated or inversely correlated. | A tight line. | Tier 4 plot over Tier 3 variants |

## Recent methods: strong evidence for evaluation design, null results for specialized machinery

The evidence base comes overwhelmingly from vision, text retrieval, web search, recommendation and molecular biology. **No method reviewed here was evaluated on within-group ranking with groups of two or three items under held-out entity regions.** Each table therefore states the domain the evidence came from, how strong it is, how the method fits this setup, and whether applying it depends on withheld details.

### Evaluation and split design are the most directly transferable family

| Method and primary source | Evidence domain and strength | Fit here | Depends on withheld details? |
|---|---|---|---|
| Pair-input taxonomies ([Park & Marcotte 2012](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC3531800/fullTextXML); [Pahikkala et al. 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4364066/); [Stock et al. 2018](https://arxiv.org/abs/1803.01575)) | Protein interactions and drug–target binding. The pattern is consistent across studies. Stock et al. is theory with no new experiments. | Report results per cell: Split A as Setting A, Split B as Setting C. Use nested selection: Pahikkala found simple cross-validation over-optimistic, especially in the coldest setting. | No |
| DataSAIL ([Joeres et al. 2025](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11978981/fullTextXML)) | Molecular data. Lowest leakage among the compared splits. Documents the loss of interactions under 2D splitting. | Re-split under joint leakage and class-ratio constraints. This research could not confirm whether it can keep comparison groups intact. | Needs a similarity matrix; the predefined relation suffices |
| SPECTRA ([Ektefaie et al. 2024](https://www.biorxiv.org/content/10.1101/2024.02.25.581982.full.pdf); details from the preprint, not the journal version) | Protein and molecular tasks. Building one set of splits took 5 minutes to 9 hours, and a full performance curve up to 5 days. | Replace one hard split with a sweep over overlap. Independent-set selection separates how much overlap there is from which items are eligible. | Needs a property graph; the relation suffices |
| Independent-set splitting ([Petti & Eddy 2022](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8929697/fullTextXML)) | Biological sequences. Measured how often each method found a valid split and how hard the resulting benchmarks were. | Explains the concern about giant units, and offers an alternative way to construct splits. | Relation suffices |
| Similarity-binned evaluation ([Zhang et al. 2025](https://arxiv.org/abs/2504.09481v1)) | Drug–target affinity; ICLR 2025 oral. | Stratify or construct test bins by maximum similarity to training. | Needs a graded relation for bins |
| Shift tests: C2ST ([Lopez-Paz & Oquab 2017](https://arxiv.org/abs/1610.06545)), "Failing Loudly" ([Rabanser et al. 2019](https://arxiv.org/abs/1810.11953)), BBSE ([Lipton et al. 2018](https://proceedings.mlr.press/v80/lipton18a.html)) | Image and document classification; well established. Two-sample tests on pretrained-classifier representations worked best. | Test for shifts in composition, labels and representations between partitions. | Representation-based tests use encoder outputs. Tests on tabular group descriptors do not. |
| Accuracy and agreement on the line ([Miller et al. 2021](https://arxiv.org/abs/2107.04649); [Baek et al. 2022](https://arxiv.org/abs/2206.13089)) | Image classification. Weaker on some shifts. Agreement on the line appears to hold only for neural classifiers. Inverse cases exist ([Teney et al. 2022](https://arxiv.org/abs/2209.00613)). | Check whether Split A predicts Split B across model variants. | No, but applying it to rank metrics is an extrapolation |
| Model-selection protocol, DomainBed ([Gulrajani & Lopez-Paz 2021](https://arxiv.org/abs/2007.01434)) | Seven image datasets. No method beat tuned ERM by more than one point under any selection rule. Oracle selection added about 2 points. | Fix the selection rule in advance and select on held-out units. | No |

Petti and Eddy provide the most actionable result for reporting. Their four splitting algorithms produced different absolute difficulty, yet **the relative ranking of the methods they compared stayed the same** across algorithms ([Petti & Eddy 2022](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8929697/fullTextXML)). Comparisons within a split realization are therefore trustworthy even when a single Split B number is not. The same paper found that a single-linkage cluster method split only **19.1%** of one family set, against **34.4%** for their best independent-set algorithm.

SPECTRA adds a caution about the pretrained encoders. For one protein model, the area under the performance curve correlated at **Spearman 0.9 with similarity to the model's pretraining corpus**, so apparent generality was largely explained by overlap with the pretraining data ([Ektefaie et al. 2024](https://www.biorxiv.org/content/10.1101/2024.02.25.581982.full.pdf)). Whether Split B's held-out regions are also under-represented in the encoders' pretraining data cannot be judged without the withheld details. It is a plausible contributor to the gap.

On selection, Split B's validation and test sets are different cold units. If units were assigned at random, they are exchangeable, and selecting checkpoints on them is legitimate. The 0.20 is then not pessimistic because of selection. If anything it is mildly optimistic relative to a deployment that has no cold validation data (inference).

### Ranking objectives were designed for lists 40–100 times longer than these groups

| Method and primary source | Evidence domain and strength | Fit here | Depends on withheld details? |
|---|---|---|---|
| Listwise and pairwise classics: ListNet ([Cao et al. 2007](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf)), ListMLE ([Xia et al. 2008](https://icml.cc/Conferences/2008/papers/167.pdf)), LambdaRank and LambdaMART ([Burges 2010](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/), not re-verified here) | Web search. Benchmark lists average about 119 items per query on Web30K and 316 on Istella ([Bai et al. 2023](https://arxiv.org/abs/2211.01494)). | At n = 2, all of these reduce to a pairwise logistic loss on the score difference. They differ only in the pair target and the pair weight (inference). | No |
| Differentiable ranking: soft rank ([Blondel et al. 2020](https://proceedings.mlr.press/v119/blondel20a.html)), monotonic sorting networks ([Petersen et al. 2022](https://arxiv.org/abs/2203.09630)), NeuralNDCG ([Pobrotyn & Białobrzeski 2021](https://arxiv.org/abs/2102.07831)), PiRank ([Swezey et al. 2021](https://arxiv.org/abs/2012.06731)) | Soft rank beat the baseline on **15 of 21** label-ranking datasets with linear models. Other evidence is from vision and web search. | Soft Spearman is a direct surrogate for the metric, provided groups are normalized. NeuralNDCG and PiRank target NDCG, not rank correlation. | No |
| Scale calibration ([Yan et al. 2022](https://research.google/pubs/scale-calibration-of-deep-ranking-models/)) | Web search and ads. Translation-invariant ranking losses can diverge indefinitely in continuous training. | Anchor the ranking loss so that BCE and ranking act on one calibrated score. This matters most if Spearman is pooled. | Uses the existing threshold |
| Regression-compatible ranking, RCR ([Bai et al. 2023](https://arxiv.org/abs/2211.01494)) | Web search and YouTube Search, binary labels only. Fully deployed. | A combination of pointwise and listwise losses whose minimizers coincide. The graded variant is a hypothesis. | No |
| Tie-aware listwise loss ([Zhu et al. 2020](https://arxiv.org/abs/2001.01828)) | Web search | Handles ties created by median aggregation. | No |
| Gradient-boosted trees: LambdaMART ([Qin et al. 2021](https://research.google/pubs/are-neural-rankers-still-outperformed-by-gradient-boosted-decision-trees/); [XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html)) | Web search: most neural rankers did worse. Industry results are mixed: neural models were more accurate with embeddings in one short-video case ([Jeunen et al. 2023](https://arxiv.org/abs/2312.01760)), and a DNN beat LambdaMART on clicks at one retailer ([OTTO 2025](https://arxiv.org/abs/2507.20753v1)). | A baseline on frozen features with settings aligned to Spearman. | Interaction features depend on the encoders' geometry |

The tiny-group regime changes what these losses do. The following are derivations from the losses' definitions and should be checked numerically.

1. **At n = 2, ListMLE is a hard-target pairwise logistic loss that ignores the size of the label gap.** A pair separated only by measurement noise gets a full-strength gradient.
2. **ListNet with a label temperature T becomes RankNet with soft target σ(δ/T),** where δ is the label difference. It is therefore tolerant of near-ties by construction.
3. **NDCG has little dynamic range here.** An inverted two-item group still scores at least 0.63, so in groups of two or three most of the "top emphasis" comes from the gain function. Exponential gains amplify noise among low scores.
4. **In the 25-item tail, top-weighting under-trains the mid-to-bottom ordering** that Spearman counts in full.
5. **Soft Spearman needs per-group normalization.** Its loss grows roughly with n³. Optimizing a per-group average requires dividing each group's loss by n(n²−1)/6, which is **1, 4 and 2,600 for n = 2, 3 and 25**.
6. **At n = 2, soft Spearman with quadratic regularization is a bounded ramp.** It resists flipped pairs but stops learning on confidently wrong ones, which argues for warm-starting with a logistic loss.
7. **Gradient sign matters even for three items.** Petersen et al. showed that non-monotonic sorting relaxations can produce gradients of the wrong sign, so monotonic or projection-based operators are preferable.

This research found **no 2023–2026 primary study of learning-to-rank at a median list size near three**. Recommendations for this regime therefore rest on the derivations above.

Yan et al. supply the link between the two supervised heads. They prove that RankNet with an anchor label y₀ is equivalent to RankNet plus a log-loss on the label 1[y > y₀]. A threshold BCE combined with a within-group pairwise loss *on the same scalar score* is therefore already a calibrated ranking loss ([Yan et al. 2022](https://research.google/pubs/scale-calibration-of-deep-ranking-models/)). The setup instead uses separate heads, which is Yan's "multi-task" variant, and that variant leaves the ranking head uncalibrated across groups. On Web30K, plain softmax ranking reached a log-loss of **2.66** against **0.63** for calibrated softmax, at essentially equal NDCG@10 (0.5002 against 0.5014). Bai et al. state the conflict directly: in a standard weighted sum, pointwise and listwise losses "are inherently conflicting." Their regression-compatible loss shares a global minimizer with the pointwise loss. In YouTube Search it improved AUCPR by **+0.22%**, where the standard multi-objective sum lost 0.37% ([Bai et al. 2023](https://arxiv.org/abs/2211.01494)).

Extending this to graded labels is a hypothesis Bai et al. did not test. One version would replace the hard label in both terms with a soft label σ((s − t)/T), where t is the threshold and T is set from the replicate noise.

### The contrastive branch carries documented false-negative and collapse risks

| Method and primary source | Evidence domain and strength | Fit here | Depends on withheld details? |
|---|---|---|---|
| Multi-positive targets: SupCon ([Khosla et al. 2020](https://arxiv.org/abs/2004.11362)), UniCL ([Yang et al. 2022](https://arxiv.org/abs/2204.03610)), MIL-NCE ([Miech et al. 2020](https://arxiv.org/abs/1912.06430)), multi-positive retrieval objectives ([Wang et al. 2026](https://arxiv.org/abs/2602.12727)) | Vision, vision–language and retrieval. SupCon's variant with the sum outside the log performed significantly better. In retrieval, the log-sum-exp pairwise form was most robust, and the other forms were sensitive to the quality of positives. | Treat observed same-x positives as positives. UniCL is the closest template for "few distinct x, many y." | Batch-wide target matrices are cheap only if projections are pair-independent |
| False-negative handling: debiased contrastive learning ([Chuang et al. 2020](https://arxiv.org/abs/2007.00224)), false-negative cancellation ([Huynh et al. 2022](https://arxiv.org/abs/2011.11765)), incremental detection ([Chen et al. 2022](https://arxiv.org/abs/2106.03719)), positive-unlabeled contrastive learning ([Acharya et al.](https://arxiv.org/abs/2206.01206), preprint) | Vision and retrieval. In RocketQA, **about 70%** of top-retrieved unlabeled passages were actually positive or highly relevant. Undenoised hard negatives cut MRR@10 from **33.32 to 26.03**; denoised ones raised it to **36.38** ([Qu et al. 2021](https://arxiv.org/abs/2010.08191)). Negatives from a stronger retriever made the ranker worse ([Cai et al. 2022](https://arxiv.org/abs/2209.05072)). | Mask known co-positives instead of debiasing. At a positive prior near 0.7, the correction factor of about 3.3 inflates variance (inference). Treat unobserved pairs as unlabeled. Use labeled within-group negatives as the hard negatives, and do not mine unobserved pairs by similarity. | The positive rate among unobserved pairs is unknown |
| Sampling corrections: logQ ([Yi et al. 2019](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/)), corrected logQ ([Khrylchenko et al. 2025](https://arxiv.org/abs/2507.09331)), mixed negative sampling ([Yang et al. 2020](https://research.google/pubs/mixed-negative-sampling-for-learning-two-tower-neural-networks-in-recommendations/)), CLRec ([Zhou et al. 2021](https://arxiv.org/abs/2005.12964)) | Recommendation, including live A/B tests. CLRec shows that in-batch softmax itself acts as a popularity debiaser. | Apply logQ on the skewed x side. Use uniformly sampled y negatives to cover more of y space. | Batch construction |
| Sigmoid pairwise losses: SigLIP ([Zhai et al. 2023](https://arxiv.org/abs/2303.15343)); decoupled contrastive learning ([Yeh et al. 2022](https://arxiv.org/abs/2110.06848)) | Vision–language. Sigmoid beats softmax "by a large margin" below a batch size of 16k and is more robust to label noise. Decoupled contrastive learning gained +6.4% at batch 256. | A ±1-or-masked label matrix could encode co-positives, labeled negatives and down-weighted unlabeled pairs. SigLIP did not test this (extrapolation). | No |
| Temperature ([Wang & Liu 2021](https://arxiv.org/abs/2012.09740); [Kukleva et al. 2023](https://arxiv.org/abs/2303.13664)) | Vision. Temperature trades uniformity against tolerance of similar samples. | Compare fixed and learned temperature. A small temperature multiplies the damage from any remaining false negatives. | No |
| Target-aware geometry: Rank-N-Contrast ([Zha et al. 2023](https://arxiv.org/abs/2210.01189)), RankSim ([Gong et al. 2022](https://proceedings.mlr.press/v162/gong22a.html)), ConR ([Keramati et al. 2024](https://arxiv.org/abs/2309.06651)), PH-Reg ([Zhang et al. 2024](https://arxiv.org/abs/2404.13904)); class collapse ([Chen et al. 2022](https://arxiv.org/abs/2204.07596)) | Vision, gaze and EEG. Rank-N-Contrast improved MAE on unseen target ranges from **18.40 to 15.92**. Collapse hurt transfer. A 2026 stress test found deep imbalanced regression methods "transfer unevenly" to time series and are seed-sensitive ([Puetz et al. 2026](https://arxiv.org/abs/2609.25152), preprint). | A conditional Rank-N-Contrast with x as anchor would align contrastive geometry with within-group order. None of these methods was tested under held-out input regions. | No |
| Contrastive auxiliaries for sparse entities ([Yao et al. 2021](https://arxiv.org/abs/2007.12865); CLCRec, [Wei et al. 2021](https://arxiv.org/abs/2107.05315)) | Recommendation. Gains were largest "on slices that lack supervision," and cold items improved. | Supports keeping the contrastive loss coupled to the encoders for sparse y. | CLCRec needs interaction-derived embeddings, which may not exist here |

Four derivations sharpen the fit. All are labelled inference.

1. **The two contrastive directions do different jobs.**
   - The x-to-y direction ranks in-batch y for a given x, and mostly tests how well the y side generalizes.
   - The y-to-x direction is close to an 8,000-way classification of y into x "classes," which can be solved partly by memorizing x identity and popularity.
   - If x embeddings are taken before cross-attention, the y-to-x direction can use a full softmax over all x. That removes in-batch sampling bias, at the cost of treating every other x for which y is positive as an unlabeled negative.
2. **The contrastive branch learns a cross-group notion of compatibility.** Off-diagonal negatives come mostly from *other* groups, so the branch learns something the within-group metric never measures. This is one plausible reason ranking order and embedding distance disagree.
3. **Threshold positives invite class collapse.** Defining positives by a threshold and then pulling all of an x's positives together reproduces the class-collapse setting that Chen et al. showed hurts transfer. It erases the within-positive order that Spearman measures.
4. **Conditional Rank-N-Contrast is ListMLE on similarities.** Take x's projection as the anchor and the group's y projections as candidates ordered by s. Rank-N-Contrast's loss then becomes exactly the Plackett–Luce (ListMLE) likelihood of the observed order on sim/τ. Its "≥" set definition handles ties Breslow-style.
   - Making the contrastive objective this in-group listwise loss, with noise-tolerant handling of near-ties and per-group normalization, would directly target the stated mismatch between ranking-head order and contrastive distances.
   - Only the x-anchored direction carries order, because most y appear in too few groups to form lists.

Validation contrastive loss is also hard to read on its own. Contrastive losses bound mutual information at log N. McAllester and Stratos showed that any distribution-free high-confidence lower bound estimated from N samples is at most O(ln N) ([McAllester & Stratos 2020](https://arxiv.org/abs/1811.04251)). A similarity-constrained validation set probably packs mutually similar y into the same batches, making negatives harder at equal ranking quality (hypothesis). Fixed-pool retrieval metrics and within-group rank correlation are the cleaner signals.

### Tuned loss weighting is the bar, and specialized optimizers rarely beat it

| Method and primary source | Evidence domain and strength | Fit here | Depends on withheld details? |
|---|---|---|---|
| Controlled comparisons ([Kurin et al. 2022](https://arxiv.org/abs/2201.04122); [Xin et al. 2022](https://arxiv.org/abs/2209.11379); [Royer et al. 2023](https://arxiv.org/abs/2310.08910); [Elich et al. 2024](https://arxiv.org/abs/2311.04698v5)) | Vision, multilingual translation and reinforcement learning. **Null results:** no specialized optimizer consistently beat a tuned sum of losses. Optimizer solutions fall on the scalarization Pareto front, and their adaptive weights barely move. In low-resource translation, only a static weighting (≈0.3/0.7) found the global optimum. Adam is partly invariant to loss scale. | The bar to beat is a tuned weighted sum, plus regularization, plus early stopping on Split-B-style Spearman. | No |
| Theoretical niche ([Hu et al. 2023](https://arxiv.org/abs/2308.13985)) | Linear theory. Scalarization cannot reach balanced trade-offs on non-convex Pareto fronts. | Of marginal relevance when ranking is the primary objective. | No |
| Gradient-manipulation optimizers: MGDA, PCGrad, CAGrad, Nash-MTL, Aligned-MTL, FAMO, FairGrad (venues confirmed via [LibMTL](https://github.com/median-research-group/LibMTL); FairGrad, [Ban & Ji 2024](https://arxiv.org/abs/2402.15638); other mechanisms not re-verified here) | Scene understanding, translation and RL. Per-task gradients cut throughput from **11.52 to 4.81 steps/s** in one study. PCGrad ran roughly **35 times** slower on one benchmark. | Last in the queue. Prefer variants that accept preference weights favouring ranking. Representation-level gradients cut the cost. | No |
| Auxiliary-aware weighting: gradient-similarity gating ([Du et al.](https://arxiv.org/abs/1812.02224), preprint), ForkMerge ([Jiang et al. 2023](https://arxiv.org/abs/2301.12618)), Auto-Lambda ([Liu et al. 2022](https://arxiv.org/abs/2202.03091), not re-verified here) | Vision, RL, general multi-task learning. ForkMerge argues that negative transfer is a generalization problem, not a gradient-conflict problem. | Weight the auxiliary losses by their effect on held-out ranking in Split-B-style validation. | Needs a cold validation fold |
| Structural fixes: Recon ([Shi et al. 2023](https://arxiv.org/abs/2302.11289)), compatible losses (RCR), staged training (LP-FT; In-N-Out, [Xie et al. 2021](https://arxiv.org/abs/2012.04550)) | Vision and time series. Gradient surgery did not reduce how often conflicts occur. Auxiliary information used as an input helped in-distribution but could hurt out-of-distribution performance; used as a pretraining target, it improved out-of-distribution performance. | Detach C or B from the encoders. Stage C before R. Un-share the layers with the most conflict. | No |
| Transference diagnostics, TAG ([Fifty et al. 2021](https://arxiv.org/abs/2109.04617)) | Taskonomy. 11.6 times faster than the prior grouping method, with 10% lower test loss. | Measure lookahead transference against held-out ranking. | No |

Three implications follow.

**Test the conflict hypothesis by outcomes, not gradients.** Mueller et al. found that gradient conflict correlates with harm to optimization but does not explain the generalization gap that multi-task training creates ([Mueller et al. 2025](https://arxiv.org/abs/2408.14677)). The loss-subset grid tests outcomes; gradient cosines only flag interference.

**Loss-magnitude weighting misreads irreducible floors.** This applies to uncertainty weighting, dynamic weight averaging, FAMO and GradNorm's training rate (inference).
- A listwise cross-entropy against a soft target cannot go below the entropy of that target.
- InfoNCE's achievable floor depends on the number of positives and negatives.
- Both floors look like unfinished work to a magnitude-based weighting method. Use the KL form of the listwise loss, or balance on gradients instead.

**Specialized optimizers act mainly as regularizers.** Kurin et al. found that without explicit regularization, these optimizers "delay overfitting." **Early stopping alone was sufficient** for the plain sum to match them ([Kurin et al. 2022](https://arxiv.org/abs/2201.04122)). Given that Split B's validation loss rises early, regularization and selection are the levers this evidence points to first.

### Keeping encoders near pretrained weights has the strongest positive evidence, all from vision and language

| Method and primary source | Evidence domain and strength | Fit here | Depends on withheld details? |
|---|---|---|---|
| Linear probe then fine-tune, LP-FT ([Kumar et al. 2022](https://arxiv.org/abs/2202.10054)) | Ten image distribution-shift datasets. Full fine-tuning was **+2% in-distribution but −7% out-of-distribution** against linear probing. LP-FT was +1% and **+10%** against full fine-tuning. | Train cross-attention, heads and projections with frozen encoders first; then unfreeze at a low learning rate, or not at all. | No; the mechanism acts on parameters |
| Weight interpolation, WiSE-FT ([Wortsman et al. 2022](https://arxiv.org/abs/2109.01903)) | CLIP image models. +4 to 6 points under distribution shift. | Interpolate encoder weights toward their pretrained values, or interpolate between a frozen-encoder model and a fine-tuned one. | The original needs a usable zero-shot endpoint, which depends on how the encoders were pretrained |
| Fine-tune like you pretrain, FLYP ([Goyal et al. 2023](https://arxiv.org/abs/2212.00638)) | CLIP. +4.2% out-of-distribution on average over standard fine-tuning across 7 datasets. | Reuse the pretraining contrastive objective. | Only if the encoders were pretrained contrastively in a compatible form |
| L2-SP ([Li et al. 2018](https://arxiv.org/abs/1802.01483)), surgical fine-tuning ([Lee et al. 2023](https://arxiv.org/abs/2210.11466)), MIRO ([Cha et al. 2022](https://arxiv.org/abs/2203.10789)) | Mostly vision; MIRO evaluated on DomainBed. | Penalize drift from pretrained weights, tune only some layers, or regularize mutual information with the pretrained features. | No |
| LoRA ([Hu et al. 2022](https://arxiv.org/abs/2106.09685)); "LoRA learns less and forgets less" ([Biderman et al. 2024](https://arxiv.org/abs/2405.09673)) | Language models. LoRA underperforms full fine-tuning on code and math but forgets less. Full fine-tuning learns perturbations of 10–100 times higher rank. | A rank sweep traces the trade-off between warm and cold performance. | Needs encoder layers that accept adapters |
| Weight averaging and ensembles: SWA ([Izmailov et al. 2018](https://arxiv.org/abs/1803.05407)), SWAD ([Cha et al. 2021](https://arxiv.org/abs/2102.08604)), model soups ([Wortsman et al. 2022](https://arxiv.org/abs/2203.05482)), DiWA ([Ramé et al. 2022](https://arxiv.org/abs/2205.09739)), ratatouille ([Ramé et al. 2023](https://arxiv.org/abs/2212.10445)), ensemble of averages ([Arpit et al. 2022](https://arxiv.org/abs/2110.10832)) | Images. SWAD gained **+1.6 points** out-of-domain. Ensembles of averages gained about 4–6 points over vanilla ERM, at ensemble inference cost. DiWA's theory says averaging helps when variance dominates, which happens under marginal shift. | Average checkpoints from after the validation minimum onward. Averaging across runs needs a shared head initialization. | No |
| Sharpness-aware minimization, SAM ([Foret et al. 2021](https://arxiv.org/abs/2010.01412)), with counter-evidence ([Andriushchenko et al. 2023](https://arxiv.org/abs/2302.07011)) | Vision and transformers. Sharpness "does not correlate well with generalization," and in several cases sharper minima had lower out-of-distribution error. | Not a principled fix for out-of-distribution performance. At most a regularizer to test. | No |

Two cautions apply. The averaging results conflict on magnitude: DomainBed found nothing beating ERM by more than one point, while ensembles of averages report 4–6. The difference reflects different baselines and the extra inference cost of ensembles. Arpit et al. also describe the out-of-distribution performance of individual models as "chaotic" across seeds ([Arpit et al. 2022](https://arxiv.org/abs/2110.10832)), so changes of a few hundredths in Split B Spearman need several seeds before they are attributed to a method.

Kumar et al.'s mechanism maps onto this architecture as a hypothesis. Cross-attention, heads and projections are all randomly initialized and trained jointly with unfrozen encoders, which is the configuration that mechanism predicts is worst for cold generalization. Tier 3 arm (b) tests it directly.

### Unseen-group methods: domain penalties mostly lose to ERM, and cold-start recommenders need IDs

| Method and primary source | Evidence domain and strength | Fit here | Depends on withheld details? |
|---|---|---|---|
| Group- and domain-robust objectives: GroupDRO ([Sagawa et al. 2020](https://arxiv.org/abs/1911.08731)), IRM ([Arjovsky et al. 2019](https://arxiv.org/abs/1907.02893)), V-REx ([Krueger et al. 2021](https://arxiv.org/abs/2003.00688)), Deep CORAL ([Sun & Saenko 2016](https://arxiv.org/abs/1607.01719)), DANN ([Ganin et al. 2016](http://jmlr.org/papers/v17/15-239.html)), Fish ([Shi et al. 2022](https://arxiv.org/abs/2104.09937)) | Images, text, molecules. On the WILDS scaffold-split molecular dataset, **ERM scored 27.2 AP against 17.9 for CORAL, 15.6 for IRM and 22.4 for GroupDRO** ([WILDS leaderboard](https://wilds.stanford.edu/leaderboard/), fetched 2026-10-31). Across the 10 WILDS datasets, CORAL beat ERM on 3, IRM on 2 and GroupDRO on 1. IRM can "fail catastrophically" in the non-linear regime ([Rosenfeld et al. 2021](https://arxiv.org/abs/2010.05761)). Fishr's authors write that "all existing approaches fail to show systematic benefits" under controlled protocols ([Ramé et al. 2022](https://arxiv.org/abs/2109.02934)). | Domains would be thousands of tiny units with noisy per-domain statistics. GroupDRO targets the worst *seen* domain, not the average unseen one, and needs strong regularization. Invariance to cluster removes legitimate signal if clusters predict s. | Domains come from the relation; no modality dependency |
| Reweighting without group labels: JTT ([Liu et al. 2021](https://arxiv.org/abs/2107.09044)), EIIL ([Creager et al. 2021](https://arxiv.org/abs/2010.07249)) | Image and text spurious-correlation benchmarks. JTT closed 75% of the worst-group gap between ERM and GroupDRO. | Upweight groups or units with high loss. Split B is a shift in support, not a minority subgroup, so transfer is uncertain. | No |
| Last-layer retraining: DFR ([Kirichenko et al. 2023](https://arxiv.org/abs/2204.02937)), DARE ([Rosenfeld et al. 2022](https://arxiv.org/abs/2202.06856)) | Spurious-correlation benchmarks. Only 50% worst-group accuracy on FMoW. | More useful as a Tier 4 probe than as a fix. | No |
| C-Mixup ([Yao et al. 2022](https://arxiv.org/abs/2210.05775)) | Regression. Tops WILDS PovertyMap at **0.53** worst-group Pearson r against 0.45 for ERM. | Mix y-side representations of same-x pairs with similar s. | Mixing raw inputs depends on the modality |
| Content-based dyadic models: Kronecker and two-step kernel ridge regression ([Stock et al. 2018](https://arxiv.org/abs/1803.01575)), inductive matrix completion ([Jain & Dhillon 2013](https://arxiv.org/abs/1306.0626)) | Theory, plus prior work on biological networks and recommenders. | Directly inductive to new y. Separate regularization for x and y matches the asymmetric data. | Kernel choice depends on the modality unless taken from the encoders |
| Cold-start recommenders built on ID embeddings: DropoutNet ([Volkovs et al. 2017](https://proceedings.neurips.cc/paper/2017/file/dbd22ba3bd0df8f385bdac3e9f8be207-Paper.pdf)), ALDI ([Huang et al. 2023](https://doi.org/10.1145/3539618.3591732)), MeLU ([Lee et al. 2019](https://arxiv.org/abs/1908.00413)) | Recommendation | Not applicable as published: the model has no ID embeddings, and new y have no labels. Two untested analogues: randomly dropping the fine-tuned encoder delta (DropoutNet), and distilling from a teacher allowed to memorize into a student with frozen encoders (ALDI). | No |
| Interaction versus factorized scoring: BEIR ([Thakur et al. 2021](https://arxiv.org/abs/2104.08663)), cross-encoders for zero-shot retrieval ([Rosa et al. 2022](https://arxiv.org/abs/2212.06121)), learned similarity versus dot product ([Rendle et al. 2020](https://arxiv.org/abs/2005.09683)) | In text retrieval, pretrained cross-encoders generalized best zero-shot, beating a strong bi-encoder by more than 4 points. In recommendation, learned MLP similarities lost to a dot product even in the warm setting. | The cross-attention here is randomly initialized, which is closer to Rendle's case (hypothesis). Compare the pathways directly. | Scaling the encoders depends on which pretrained sizes exist |

This research found **no domain-general controlled evidence** on whether interaction architectures memorize cold entities more or less than factorized ones. The setup already contains both pathways, so Tier 2's contrastive-only scoring and Tier 4's per-pathway comparison are the cheapest way to answer that for this model. Stock et al. note that "providing a different regularization for instances and tasks also makes sense" when one side explains most of the variance ([Stock et al. 2018](https://arxiv.org/abs/1803.01575)). Applied here, that suggests trying stronger regularization of the y encoder than the x encoder (hypothesis), because x has many observations each and y has one or two.

### Uncertainty methods flag cold y even when nothing closes the gap

Uncertainty methods cannot raise the score, but they can mark which predictions to trust.
- **Ensembles are the most robust option under shift.** Ovadia et al. found that post-hoc calibration "falls short" under dataset shift, while ensembles held up best ([Ovadia et al. 2019](https://arxiv.org/abs/1906.02530)).
- **SNGP is a single-model alternative.** It offers distance-aware uncertainty competitive with ensembles ([Liu et al. 2020](https://arxiv.org/abs/2006.10108)).

**Weighted conformal prediction is formally inapplicable at the cluster level.** It requires a likelihood ratio that is known or accurately estimated ([Tibshirani et al. 2019](https://arxiv.org/abs/1904.06019)). Split B gives every test cluster zero training probability by construction, so the ratio is undefined at the cluster level. In embedding space it would be extreme and fragile (inference). Importance weighting also fades over training epochs in deep networks ([Byrd & Lipton 2019](https://arxiv.org/abs/1812.03372)).

**Unit-level calibration is the workable alternative.** If units were assigned at random, calibrating on held-out units gives valid coverage for new units. Hierarchical conformal methods handle observations grouped within units ([Dunn et al.](https://arxiv.org/abs/1809.07441)). An applicability-domain score can be built from maximum similarity to training y or from distance-aware uncertainty. Validate it by checking that it rises monotonically with per-group error on Split B and on the stratified Split A.

## A gated sequence, and what is established versus hypothesized

**Step 1: run the Tier 1 audits.**
- If the reweighted Split A score and Split A's lowest-similarity stratum both land near 0.20, and the noise ceilings differ, treat the gap mainly as a property of the evaluation. In that case, report per cell, with unit-level intervals and as a fraction of the ceiling, before changing the model.
- If the loss rise comes from BCE log-loss alone and the best-Spearman checkpoint is late, switch selection to validation Spearman and stop interpreting the 10% turning point.

**Step 2: run the Tier 2 baselines.**
- If the y-only and nearest-neighbour baselines reproduce most of 0.41 on Split A, and the variance decomposition shows a small interaction share, any architecture is likely bounded on Split B. More cold-region groups and more replicates then matter more than architecture work.

**Step 3: run the Tier 3 arms, in order of cost and strength of evidence.**
1. Selection on Spearman with tuned regularization.
2. Frozen or low-rank encoders, with checkpoint averaging.
3. The loss-subset grid and the weight sweep.
4. The 2×2 leak design.
5. Learning curves.

**Step 4: run the Tier 4 probes, then the targeted intervention the evidence points to.**

| If the evidence shows | Try |
|---|---|
| Encoder drift | The LP-FT schedule, weight interpolation, low-rank updates and averaging. |
| Contrastive artifacts | Co-positive masking, a multi-positive or sigmoid label matrix, a fixed temperature, conditional rank-aware contrast, or detaching the contrastive loss from the encoders. |
| A conflict between the optima of the pointwise and ranking losses | A compatible loss pair, or one anchored score shared by BCE and ranking. |
| Metric misalignment | Per-group normalization, uniform pair weights, soft targets calibrated to replicate noise, or soft Spearman. |
| A head-transfer failure | Use the probe results to choose between head-level regularization and a late-fusion design: a factorized score plus a gated interaction residual. |

Domain-generalization penalties and specialized multi-task optimizers come last, because they carry the weakest and most often negative evidence. None of these steps promises a fix. Each is chosen because a specific signature makes it the most informative next experiment.

| Claim | Status | Basis |
|---|---|---|
| Random pair splits measure almost only the warm regime | Established (protein interactions) | Over 99% of random test pairs were the both-seen class ([Park & Marcotte](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC3531800/fullTextXML)) |
| Halving or worse is common when one side goes cold | Established as a range (biology, chemistry); says nothing about cause here | [Pahikkala et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC4364066/); [SPECTRA](https://www.biorxiv.org/content/10.1101/2024.02.25.581982.full.pdf) |
| Splits meeting the same similarity criteria differ in difficulty; cluster splits yield more remote test items | Established (biological sequences) | [Petti & Eddy](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8929697/fullTextXML) |
| Split B's test set is peripheral because of giant units | Hypothesis | Tier 1 unit audit |
| The model-selection rule changes out-of-distribution conclusions; tuned ERM is hard to beat | Established (images) | [DomainBed](https://arxiv.org/abs/2007.01434) |
| Domain-invariance penalties lose to ERM on scaffold-domain splits | Established (molecules; live leaderboard) | [WILDS](https://wilds.stanford.edu/leaderboard/) |
| Specialized multi-task optimizers do not beat tuned scalarization with regularization | Established (vision, translation, RL); untested for this loss mix | [Kurin](https://arxiv.org/abs/2201.04122); [Xin](https://arxiv.org/abs/2209.11379) |
| Full fine-tuning hurts out-of-distribution performance relative to staged or interpolated tuning | Established (vision, CLIP); hypothesis for this architecture | [Kumar et al.](https://arxiv.org/abs/2202.10054); [WiSE-FT](https://arxiv.org/abs/2109.01903) |
| Undenoised hard negatives hurt when unlabeled positives are common | Established (text retrieval) | [RocketQA](https://arxiv.org/abs/2010.08191) |
| Pointwise and listwise losses conflict at the optimum unless designed to be compatible | Established (web search, binary labels); graded extension untested | [Bai et al.](https://arxiv.org/abs/2211.01494) |
| Within-group order depends only on the y main effect and the x×y interaction | Inference (arithmetic) | Problem statement |
| Spearman on groups of 2–3 takes only ±1 and ±0.5; pair-sum weighting differs 100-fold | Inference (arithmetic) | Problem statement |
| Listwise losses reduce to pairwise logistic at n = 2; conditional Rank-N-Contrast equals ListMLE on similarities | Inference (derivation; check numerically) | Loss definitions |
| Group batching makes about 16–17 labeled positives into negatives per positive anchor in 25-item groups | Inference (arithmetic, diagonal-only targets) | Problem statement |
| Split A's 0.41 rests mostly on y main effects and neighbour memorization | Hypothesis | Tier 2 ladder, Tier 3 2×2 |
| The Split B loss rise is driven by calibration | Hypothesis | Tier 1 decomposition |
| A replicate noise ceiling bounds Split B lower than Split A | Hypothesis; method established in neuroscience, with no ML ranking precedent found | [Nili et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC3990488) |
| The leak-in/leak-out 2×2 and the variance-component bound | Novel designs on established foundations | No precedent found |
| Guo et al. show NLL rising while error falls during training | Widely cited; not verified here | [Guo et al.](https://arxiv.org/abs/1706.04599) |

## Conclusion

The most consequential reframing is that this problem is first a measurement problem and only second a modelling problem. Grouping guarantees that the x main effect cancels, groups are mostly two or three items, and replicates are already collected. Together these make it possible to bound how much of the gap any model could close before training anything new: through the noise ceiling, the interaction variance component, and the y-only baseline. If those bounds come back low, the right deliverable is a carefully reported Split B number with intervals and a ceiling, plus a decision about collecting more data from cold regions. Architecture work would not move it.

The second takeaway concerns where the literature's positive evidence clusters. It does not favour invariance penalties or gradient surgery; on comparable scaffold-split data and in controlled multi-task studies those mostly fail to beat tuned baselines. It favours letting the encoders move less and making the three objectives agree on what a good ordering is. That means:
- masking co-positives instead of contrasting them;
- tying BCE and ranking to one compatible score;
- training the contrastive branch with a within-group, rank-aware loss rather than threshold positives.

These are design corrections suggested by documented mechanisms, not proven remedies for held-out entity regions. They are worth testing only after the audits show that the gap is not mostly an artifact of how Split B was built and scored.
