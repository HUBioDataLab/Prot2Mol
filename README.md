# Prot2Mol

Prot2Mol is a protein-conditioned molecular design framework based on an encoder-decoder architecture. The generator maps protein sequences to molecular SELFIES. Protein-ligand scoring and ranking are handled separately by the standalone `RewardModel`.

## Scientific Scope

- Protein encoder: `ESM2` (default) or `ProtT5`, using the same checkpoint for
  model and tokenizer.
- Molecule decoder: the pretrained MolGen-large BART decoder with cross-attention
  over projected, normalized per-residue protein representations.
- Molecule representation: SELFIES.
- Training objective: molecule language modeling conditioned on protein token representations.
- Reward objective: standalone listwise ranking and activity classification in `RewardModel/`.

## Installation

```bash
git clone https://github.com/atabeyunlu/Prot2Mol.git
cd Prot2Mol
pip install -r requirements.txt
```

## Repository Organization

```text
prot2mol/
  main.py                 # Single meta-entrypoint
  configs/                # YAML templates (train/generate)
  core/                   # Core model and protein encoders
  training/               # Training pipeline, trainer, metrics, services
  inference/              # Molecule generation pipeline
  data/                   # Shared data/tokenization pipeline
  io/                     # HF/model/config I/O utilities
  chem/                   # Cheminformatics utilities and fingerprints
data_processing/
  build_chembl_generation_dataset.py # Strict-positive MMseqs50 train/validation split
  preprocess_dataset.py              # One-time tokenization into HF disk cache
```

## Unified Command Interface

All operational modes are executed through a single entrypoint:

```bash
python prot2mol/main.py --help
```

Available commands:

- `train`
- `generate`

Examples:

```bash
python prot2mol/main.py train --help
python prot2mol/main.py generate --help
```

## YAML-Based Configuration

Each command supports `--config` to load arguments from YAML.

- Supported template files:
  - `prot2mol/configs/train.yaml`
  - `prot2mol/configs/generate.yaml`
- Expected top-level sections in YAML:
  - `train` for `train`
  - `generate` for `generate`

Argument precedence is:

1. CLI arguments
2. YAML values (`--config`)
3. parser defaults

This allows concise runs with selective CLI overrides.

## Recommended Workflow

### 1. Build the ChEMBL37 generation split

```bash
python data_processing/build_chembl_generation_dataset.py
```

This reads the canonical ChEMBL37 binding table, retains only rows with
`pchembl_value > 6.0`, and assigns complete MMseqs50 clusters to a 95% train / 5%
validation split. Duplicate protein-canonical-SMILES pairs are collapsed before
splitting, while their source-row and assay counts are preserved as provenance.
Examples that exceed the ESM2 or MolGen context are filtered explicitly; the
pipeline deliberately writes no test split and never silently truncates targets.

### 2. Preprocess once

```bash
python data_processing/preprocess_dataset.py
```

The matched MolGen tokenizer uses right padding and is checked for unknown-token
coverage and encoded length across both splits; the protein tokenizer also uses
right padding. Raw provenance columns remain available for bounded generation
evaluation, while the training collator passes only tensors to the model.
Training verifies this preprocessing manifest against the configured encoder,
decoder, tokenizer contexts, and split sizes before loading any weights.

### 3. Train

```bash
python prot2mol/main.py train \
  --config prot2mol/configs/train.yaml
```

Override any parameter at runtime:

```bash
python prot2mol/main.py train \
  --config prot2mol/configs/train.yaml \
  --epoch 20 --learning_rate 5e-6
```

Warm the learned protein projection and molecular decoder with a frozen protein
encoder:

```bash
python prot2mol/main.py train \
  --config prot2mol/configs/train.yaml \
  --no-train_encoder_model \
  --train_projection_model \
  --train_decoder_model \
  --load_pretrained_model /path/to/encoder_decoder_checkpoint
```

### 3.1 Selectable Training Execution Mode

`train` supports explicit execution mode control through `--training_mode` (or `train.training_mode` in YAML):

- `auto`: infer from launcher environment (`WORLD_SIZE`, `LOCAL_WORLD_SIZE`)
- `single_gpu`: one-process training (no distributed process group)
- `multi_gpu`: single-node distributed training (multi-GPU)
- `multi_node`: multi-node distributed training (HPC)

The default config and CLI default are `single_gpu`. Use `auto` only when you intentionally want launch environment variables to decide the mode.

Single GPU:

```bash
python prot2mol/main.py train \
  --config prot2mol/configs/train.yaml \
  --training_mode single_gpu
```

Multi-GPU (single node):

```bash
torchrun --standalone --nproc_per_node=4 \
  prot2mol/main.py train \
  --config prot2mol/configs/train.yaml \
  --training_mode multi_gpu
```

Multi-node (example with 2 nodes, 4 GPUs/node):

```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  prot2mol/main.py train \
  --config prot2mol/configs/train.yaml \
  --training_mode multi_node
```

### 4. Generate Molecules

```bash
python prot2mol/main.py generate \
  --config prot2mol/configs/generate.yaml
```

Generation accepts an explicit `generate.protein_sequence`, or selects a target from
the raw ChEMBL generation Parquets using `generate.protein_id`. Use `RewardModel/`
to score generated protein-ligand pairs. Final generator evaluation reports
reference similarity and exact recovery within each protein's own validation set,
then macro-averages across proteins; references are never pooled across targets.

## Minimal Data Requirements

### Raw generation Parquets

Required columns:

- `protein_sequence`
- `compound_selfies`
- `smiles`
- `protein_cluster_50`
- `pchembl_value` (strictly greater than 6.0)

The generator uses pChEMBL only to select positive training examples; affinity is
not a model target.

## Citation

If you use Prot2Mol in research, please cite:

```bibtex
Ünlü, A., Çevrim, E., Doğan, T. (2024).
Prot2Mol: Target based molecule generation using protein embeddings and SELFIES molecule representation.
GitHub. https://github.com/HUBioDataLab/Prot2Mol
```
