# Ranking-head diagnostics

Run these tools from `RewardModel/`. Model inference can run on CPU, although a
GPU is substantially faster for complete validation assays.

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

## 3. Inspect head activations

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
heads.

## 4. Compare checkpoints

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
