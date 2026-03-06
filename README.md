# Prot2Mol

Prot2Mol is a protein-conditioned molecular design framework based on an encoder-decoder architecture with an auxiliary affinity head. The model maps protein sequences to molecular SELFIES and supports pChEMBL prediction for generated or external compounds.

## Scientific Scope

- Protein encoder: `ProtT5`, `ESM2`, or `SaProt`.
- Molecule decoder: GPT-2 with cross-attention over protein representations.
- Molecule representation: SELFIES.
- Auxiliary task: pChEMBL regression head built with FusionDTI-style token fusion plus an MLP regressor.
- Multi-task training: language-model objective + pChEMBL objective, with optional blocking of pChEMBL gradients into encoder/decoder.

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
  configs/                # YAML templates (train/generate/predict)
  core/                   # Core model and protein encoders
  training/               # Training pipeline, trainer, metrics, services
  inference/              # Generation and pChEMBL prediction pipelines
  data/                   # Shared data/tokenization pipeline
  io/                     # HF/model/config I/O utilities
  chem/                   # Cheminformatics utilities and fingerprints
data_processing/
  preprocess_dataset.py   # One-time preprocessing into HF disk cache
```

## Unified Command Interface

All operational modes are executed through a single entrypoint:

```bash
python prot2mol/main.py --help
```

Available commands:

- `train`
- `generate`
- `predict`

Examples:

```bash
python prot2mol/main.py train --help
python prot2mol/main.py generate --help
python prot2mol/main.py predict --help
```

## YAML-Based Configuration

Each command supports `--config` to load arguments from YAML.

- Supported template files:
  - `prot2mol/configs/train.yaml`
  - `prot2mol/configs/generate.yaml`
  - `prot2mol/configs/predict.yaml`
- Expected top-level sections in YAML:
  - `train` for `train`
  - `generate` for `generate`
  - `predict` for `predict`

Argument precedence is:

1. CLI arguments
2. YAML values (`--config`)
3. parser defaults

This allows concise runs with selective CLI overrides.

## Recommended Workflow

### 1. Preprocess Dataset (one-time per dataset)

```bash
python data_processing/preprocess_dataset.py \
  --selfies_path /path/to/dataset.csv \
  --prot_emb_model saprot \
  --max_mol_len 256 \
  --prot_max_length 1024
```

This builds tokenized data in the HF disk cache and stores pChEMBL normalization statistics used during training/evaluation.

### 2. Train

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

Frozen encoder-decoder fine-tuning for pChEMBL only:

```bash
python prot2mol/main.py train \
  --config prot2mol/configs/train.yaml \
  --train_encoder_model false \
  --train_decoder_model false \
  --train_pchembl_head true \
  --stop_pchembl_gradients true \
  --load_pretrained_model /path/to/encoder_decoder_checkpoint
```

The pChEMBL head uses protein token embeddings from the frozen encoder and decoder last hidden states from the frozen molecule decoder, then applies FusionDTI-style token fusion followed by an MLP regressor. The trained checkpoint writes a `config.json` with the full head architecture so `predict` and generation-time pChEMBL scoring can reload the same setup automatically.

### 2.1 Selectable Training Execution Mode

`train` supports explicit execution mode control through `--training_mode` (or `train.training_mode` in YAML):

- `auto`: infer from launcher environment (`WORLD_SIZE`, `LOCAL_WORLD_SIZE`)
- `single_gpu`: one-process training (no distributed process group)
- `multi_gpu`: single-node distributed training (multi-GPU)
- `multi_node`: multi-node distributed training (HPC)

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

### 3. Generate Molecules

```bash
python prot2mol/main.py generate \
  --config prot2mol/configs/generate.yaml
```

### 4. Predict pChEMBL

```bash
python prot2mol/main.py predict \
  --config prot2mol/configs/predict.yaml
```

Reproduce validation-split predictions (same split strategies used in training):

```bash
python prot2mol/main.py predict \
  --config prot2mol/configs/predict.yaml \
  --reproduce random
```

Use `--reproduce aid` for AID hold-out validation split.
If local model caches are not under the default path, set `predict.models_base` in YAML
or export `MODELS_BASE_PATH` to the directory that contains `models--zjunlp--MolGen-large`.
For ID-based target resolution, set:
- `predict.chembl_uniprot_mapping_path` (CHEMBL -> UniProt mapping file)
- `predict.protein_targets_path` (Papyrus protein targets TSV with `target_id` and `Sequence`)
Output CSVs automatically exclude internal columns used only for preprocessing/inference
(e.g., `Target_FASTA`, token IDs/masks, labels, and train flags).

## Minimal Data Requirements

### Training dataset CSV

Required columns:

- `Target_FASTA`
- `Compound_SELFIES`
- `pchembl_value_Median`

Optional columns (used for advanced split/ranking metrics):

- `AID`
- `Target_ID`

### Prediction input CSV (`predict`)

Required:

- molecule column (`smiles`/`Compound_SMILES`/`selfies`/`Compound_SELFIES`, etc.)

Target specification:

- either `Target_FASTA`
- or `UniProt_ID` (resolved as `<UniProt>_WT` in protein target TSV)
- or `Target_CHEMBL_ID` (resolved via CHEMBL->UniProt mapping, then `<UniProt>_WT`)

For `--reproduce aid`, include `AID` in the input CSV.

## Citation

If you use Prot2Mol in research, please cite:

```bibtex
Ünlü, A., Çevrim, E., Doğan, T. (2024).
Prot2Mol: Target based molecule generation using protein embeddings and SELFIES molecule representation.
GitHub. https://github.com/HUBioDataLab/Prot2Mol
```
