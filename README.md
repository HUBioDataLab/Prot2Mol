# Prot2Mol

Prot2Mol is a protein-conditioned molecular design framework based on an encoder-decoder architecture with an auxiliary affinity head. The model maps protein sequences to molecular SELFIES and supports pChEMBL prediction for generated or external compounds.

## Scientific Scope

- Protein encoder: `ProtT5`, `ESM2`, or `SaProt`.
- Molecule decoder: GPT-2 with cross-attention over protein representations.
- Molecule representation: SELFIES.
- Auxiliary task: pChEMBL regression head.
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
- or `Target_CHEMBL_ID` with `--data_path` for sequence lookup

## Citation

If you use Prot2Mol in research, please cite:

```bibtex
Ünlü, A., Çevrim, E., Doğan, T. (2024).
Prot2Mol: Target based molecule generation using protein embeddings and SELFIES molecule representation.
GitHub. https://github.com/HUBioDataLab/Prot2Mol
```
