# Ranking-head diagnostics

Run these tools from `RewardModel/`. Model inference can run on CPU, although a
GPU is substantially faster for complete validation assays.

The primary training config now uses `pair_scoring_mode: scaled_cosine`. After
fusion, residual addition, and configured pooling, protein and ligand vectors
are independently L2-normalized. One shared score is then used by both tasks:

```text
cosine = dot(normalize(fused_protein), normalize(fused_ligand))
ranking_score = exp(logit_scale) * cosine
activity_logit = ranking_score + classification_logit_bias
```

The configured model dropout is applied after the protein and molecule
projections and to fusion attention probabilities. It remains disabled during
evaluation and does not override either pretrained encoder's native dropout
configuration.

`logit_scale` starts at `log(13)`, following LigUnity, and is capped at a
configured positive scale of 100. Unlike LigUnity's released ranking-loss code,
which detaches its scale, this implementation lets both listwise ranking and
classification update the same scale and cosine geometry. The classification
bias changes only the activity threshold; it cannot change ligand ordering.
Legacy checkpoints remain loadable because configs without `pair_scoring_mode`
default to the former `mlp` heads.

Reference comparison:

| Detail | LigUnity released code | This reward model |
| --- | --- | --- |
| Representation | Separate pocket/protein and ligand encoders; token 0 projected to 128 dimensions | Pair-conditioned cross-fusion with residual; configured pooling in 512 dimensions |
| Similarity | L2-normalized dot product over the pocket-by-ligand matrix | L2-normalized dot product for each supplied protein-ligand pair |
| Scale | `exp(logit_scale)`, initialized at 13 and detached in the released ranking loss | Same initialization, positive and capped, jointly learned by ranking and classification |
| Objectives | Retrieval contrastive loss plus listwise ranking | Listwise Plackett-Luce ranking plus binary activity classification |
| Bias | Declared by the model but unused by the released ranking loss | Learned classification threshold only; absent from ranking |

The relevant LigUnity references are its
[ranking model](https://github.com/IDEA-XL/LigUnity/blob/main/unimol/models/pocket_ranking.py)
and [released joint loss](https://github.com/IDEA-XL/LigUnity/blob/main/unimol/losses/contras_rank_loss.py).

## Simple cosine ablation

`configs/reward_train_simple_cosine.yaml` removes the fusion block and all
non-encoder processing except one learned linear projection per encoder. Its
complete scoring path is:

```text
protein = normalize(masked_mean(protein_projection(protein_encoder(tokens))))
ligand = normalize(masked_mean(ligand_projection(ligand_encoder(tokens))))
ranking_score = dot(protein, ligand)
```

There is no projection LayerNorm or dropout, cross-attention fusion, MLP head,
learned cosine scale, classification bias, or classification loss in this
mode. Its ranking metrics profile keeps the dashboard to `loss`, `grad_norm`,
`learning_rate`, `cosine_std`, `pair_accuracy`, `spearman`, and `pearson`
during training, and `eval_loss`,
`eval_spearman`, `eval_pearson`, `eval_cosine_std`, and the margin-aware
`eval_pair_accuracy` during evaluation. Protein-shuffle evaluation is disabled
for this run. Start it directly from the pretrained encoders; a fusion checkpoint is
architecturally incompatible and must not be supplied:

```bash
python train_reward_model.py \
  --config configs/reward_train_simple_cosine.yaml
```

The full-dataset scale-10 experiment keeps the same simple raw-cosine scorer
and applies the successful overfit setting through
`ranking_temperature: 0.1`. It also carries over the successful split learning
rates (`1e-5` for both encoders and `1e-3` for the two projections) and gradient
clip norm of 10, while restoring the standard dataset cache and ranking-list
construction:

```bash
python train_reward_model.py \
  --config configs/reward_train_simple_cosine_scale10.yaml
```

The MoLFormer variant keeps the same scoring path but changes only the
molecule encoder and its input representation. It loads
`ibm/MoLFormer-XL-both-10pct` and its tokenizer with remote code enabled,
uses deterministic evaluation, and tokenizes the split Parquets' canonical
`smiles` column into a separate cache:

```bash
python prepare_reward_training_data.py \
  --config configs/reward_train_simple_cosine_molformer.yaml
python train_reward_model.py \
  --config configs/reward_train_simple_cosine_molformer.yaml
```

## Two-stage training

The primary config warms up the projection, fusion, residual, and cosine
parameters with both encoders frozen. Start phase two from those model weights
with the lower-memory unfrozen config:

```bash
python train_reward_model.py \
  --config configs/reward_train_unfrozen.yaml \
  --init-from-checkpoint outputs/chembl_37_mmseqs50_activity_balanced_scaled_cosine_1000_steps/checkpoint-1000
```

This is deliberately a weight-only warm start, not a full Trainer resume. It
strictly loads the model weights, applies the phase-two freeze settings, and
starts a fresh optimizer, learning-rate schedule, and global step in a separate
output directory. The current phase-two config sets
`classification_loss_weight: 0.0`, so optimization and best-checkpoint
selection are ranking-only; classification metrics remain diagnostic. It also
sets `max_classification_only_per_item: 0`, so rows outside the sampled ranking
lists do not consume encoder work or create zero-gradient batches. The phase-two
batch size is 12 with four-step gradient
accumulation, preserving an effective batch size of 48 while retaining encoder
activations for backpropagation.

The live diagnostic log is intentionally compact for this scoring mode. It
keeps the learned `cosine_scale`, classification bias, cosine mean/spread,
Plackett-Luce entropy, affinity-margin pair accuracy/median gap, losses,
Spearman, and gradient norm. The former hypothetical tanh transforms and
redundant scaled-score distribution statistics are no longer emitted. The
unmargined pair accuracy, per-ranked-example loss, and duplicate training
`total_loss` were also removed; their margin-aware or trainer-native versions
already carry the useful information.

```bash
python analyze_ranking_score_diagnostics.py \
  outputs/RUN/ranking_score_diagnostics.jsonl \
  --tail 20 --follow
```

## 1. Inspect predictions

Run the same deterministic assay selection at each checkpoint:

```bash
python inspect_ranking_predictions.py \
  --checkpoint outputs/RUN/checkpoint-1000 \
  --config configs/reward_train.yaml \
  --output-dir diagnostics/step1000 \
  --splits train val \
  --max-train-assays 100 \
  --max-val-assays 100 \
  --device cpu
```

The output includes per-molecule Parquet predictions, per-assay CSV metrics,
largest rank errors, JSON summaries, an assay manifest, and an HTML report with
representative score-versus-pChEMBL plots. Set an assay limit to `0` for every
eligible assay.

## 2. Test protein and ligand sensitivity

Reuse the prediction manifest so the perturbation test examines the same
complete assays:

```bash
python inspect_ranking_input_sensitivity.py \
  --checkpoint outputs/RUN/checkpoint-1000 \
  --config configs/reward_train.yaml \
  --output-dir diagnostics/step1000_sensitivity \
  --split val \
  --assay-manifest diagnostics/step1000/val_assays.json \
  --num-shuffles 3 \
  --device cpu
```

Protein shuffling replaces each assay's protein with a different assay's
protein while retaining its ligands. Ligand shuffling retains proteins and
replaces ligands. A protein rank-stability value near one, a near-zero protein
target-Spearman change, and a protein-to-ligand centered-sensitivity ratio near
zero indicate ligand-only ranking behavior.

## 3. Inspect fusion cosine geometry

Before changing the scalar heads, compare the cosine geometry immediately
before fusion with the geometry after fusion and its optional residual:

```bash
python inspect_fusion_cosine_sensitivity.py \
  --checkpoint outputs/RUN/checkpoint-1000 \
  --config configs/reward_train.yaml \
  --output-dir diagnostics/step1000_fusion_cosine \
  --split val \
  --assay-manifest diagnostics/step1000_sensitivity/val_assays.json \
  --num-shuffles 3 \
  --device cpu
```

The tool uses the model's configured pooling rule at both stages. It records
correct-pair, protein-shuffled, and ligand-shuffled cosine distributions,
within-assay rank stability, centered sensitivity, target Spearman changes,
and post/pre fusion variance ratios. A narrow post-fusion distribution together
with high protein-shuffle rank stability indicates that post-fusion cosine would
not provide a useful protein-conditioned ranking score.

## 4. Inspect legacy MLP head activations

```bash
python inspect_ranking_head_activations.py \
  --checkpoint outputs/RUN/checkpoint-1000 \
  --config configs/reward_train.yaml \
  --output-dir diagnostics/step1000_activations \
  --split val \
  --assay-manifest diagnostics/step1000/val_assays.json \
  --device cpu
```

This records streaming mean, standard deviation, range, near-zero fraction,
and non-finite counts for every linear and layer-normalization module in both
legacy heads. Scaled-cosine checkpoints have no MLP heads, so use sections 2
and 3 plus the logged `cosine_scale` for them.

## 5. Compare checkpoints

After running prediction inspection for step 1000, the best checkpoint, and
the final model:

```bash
python compare_ranking_checkpoints.py \
  --runs \
    step1000=diagnostics/step1000 \
    best=diagnostics/best \
    final=diagnostics/final \
  --output-dir diagnostics/checkpoint_comparison
```

The comparison retains assay-level trajectories, so a falling aggregate score
can be separated into widespread degradation versus failure on a few large
assays.
