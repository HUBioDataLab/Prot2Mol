# RewardModel overfit experiments

## Phase 1: 50-molecule identical-split overfit test

This phase keeps the ranking-only simple-cosine model and uses:

- protein encoder: `facebook/esm2_t12_35M_UR50D`
- molecule encoder: `HUBioDataLab/SELFormer` with SELFIES
- one target-assay group containing exactly 50 unique molecules
- identical untouched rows in `train_examples`, `val_examples`, and `test_examples`
- one 50-molecule training list and one evaluation partition
- unfrozen protein encoder, molecule encoder, and projections
- the existing LigUnity listwise ranking loss

Build the overfit dataset from the existing full SELFormer tokenized cache:

```bash
cd RewardModel
../.venv/bin/python prepare_reward_overfit_data.py \
  --source-dir reward_model/data/tokenized/chembl_37_mmseqs50_activity_balanced \
  --output-dir reward_model/data/tokenized/chembl_37_mmseqs50_activity_balanced_overfit_50
```

The builder refuses to overwrite an existing output directory. It writes
`overfit_manifest.json`, including the selected `target__assay` group, pChEMBL
range, legal ordered-pair count, and the shared SHA-256 for all three splits.
Use `--group-id TARGET__ASSAY` to choose a specific group instead of the
deterministic automatic selection.

### Full configuration grid

| Config prefix | Encoder LR | Projection LR |
|---|---:|---:|
| `uniformlr` | `1e-5` | `1e-5` |
| `splitlr` | `1e-5` | `1e-3` |

Each LR setting is crossed with `max_grad_norm` in `{1, 10}` and raw-cosine
ranking temperature in `{1, 0.1}`, producing eight YAML files under
`configs/overfit_grid/`. Temperature changes only the logits passed to the
listwise loss; the model still emits raw normalized cosine similarity.

Run one configuration with:

```bash
CUDA_VISIBLE_DEVICES=1 WANDB_NAME="overfit_50_uniformlr_clip1_temp1" \
  ../.venv/bin/python train_reward_model.py \
  --config configs/overfit_grid/reward_overfit_50_uniformlr_clip1_temp1.yaml
```

The comparison settings not named by the grid remain common across all eight
files, including BF16, 8 data-loader workers, the existing optimizer and
scheduler behavior, 10,000 maximum steps, and evaluation every 500 steps.

## Phase 2 note: contradictory assay supervision

Do not implement this phase until the 50-molecule overfit grid is evaluated.
The requested follow-up is:

1. Build a diagnostic set of about 1,000 unique molecules where the same
   protein-molecule pair occurs in different assays and the assays imply
   contradictory ranks.
2. Keep a no-assay baseline.
3. Add a learned assay embedding and test whether adding it to the projected
   protein-side representation before normalization lets the model distinguish
   the conflicting assay contexts.
4. Compare the baseline and assay-conditioned model on the same contradiction
   set.

The exact contradiction rule and treatment of unseen assay IDs should be fixed
before building this second dataset.
