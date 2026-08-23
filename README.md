# Prot2Mol

Prot2Mol is a protein-conditioned molecular design framework based on an encoder-decoder architecture. The generator maps protein sequences to molecular SELFIES. Protein-ligand scoring and ranking are handled separately by the standalone `RewardModel`.

## Scientific Scope

- Protein encoder: `ESM2` (default) or `ProtT5`, using the same checkpoint for
  model and tokenizer.
- Molecule decoder: selectable protein-conditioned GPT-2 (default) or the
  experimental MolGen-large BART decoder.
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
  rewards/                # Frozen post-training reward-model ablations
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
python prot2mol/main.py grpo --help
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

## Decoder Architectures

Set `decoder_type` to `gpt2` or `molgen` in the train and generate configs.

### GPT-2 (default)

The restored Prot2Mol decoder is a causal `GPT2LMHeadModel` initialized with
`add_cross_attention=true`. Its cross-attention keys and values are protein
encoder states, so protein-to-molecule conditioning is part of this decoder's
training contract. `n_layer`, `n_head`, and `n_emb` configure the architecture.
Use `load_pretrained_model` to initialize it from an existing Prot2Mol GPT-2
checkpoint.

### MolGen (experimental alternative)

`zjunlp/MolGen-large` is a BART model pretrained on 100 million ZINC-15
molecules represented as SELFIES. Its encoder receives a corrupted SELFIES for a
molecule and its autoregressive decoder reconstructs the original SELFIES. Thus,
its pretrained decoder cross-attention learned molecule-to-molecule denoising,
not protein-to-molecule conditioning. Prot2Mol can replace those encoder states
with projected protein states, but this cross-modal alignment is new and must be
learned during downstream training; the MolGen checkpoint does not provide that
alignment. Sources: [ICLR 2024 paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/ed7dd1e32cf9b0abf664bf0e891527e5-Paper-Conference.pdf)
and [MolGen-large model card](https://huggingface.co/zjunlp/MolGen-large).

For both decoder types, `decoder_model_id` identifies the matched SELFIES
tokenizer. In `molgen` mode it also identifies the pretrained BART weights.

## Recommended Workflow

### 1. Build the ChEMBL37 generation split

```bash
python data_processing/build_chembl_generation_dataset.py
```

This reads the canonical ChEMBL37 binding table, retains only rows with
`pchembl_value > 6.0`, and assigns complete MMseqs50 clusters to a 95% train / 5%
validation split. Duplicate protein-canonical-SMILES pairs are collapsed before
splitting, while their source-row and assay counts are preserved as provenance.
Examples that exceed the configured protein or molecule context are filtered explicitly; the
pipeline deliberately writes no test split and never silently truncates targets.

### 2. Preprocess once

```bash
python data_processing/preprocess_dataset.py
```

The matched SELFIES tokenizer uses right padding and is checked for unknown-token
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

Load an existing GPT-2 Prot2Mol checkpoint with a frozen protein encoder:

```bash
python prot2mol/main.py train \
  --config prot2mol/configs/train.yaml \
  --no-train_encoder_model \
  --no-train_projection_model \
  --train_decoder_model \
  --load_pretrained_model /path/to/encoder_decoder_checkpoint
```

Select MolGen explicitly for an ablation:

```bash
python prot2mol/main.py train \
  --config prot2mol/configs/train.yaml \
  --decoder_type molgen \
  --train_projection_model
```

### 3.1 GRPO post-training core

`prot2mol.training.GRPOTrainer` performs online protein-conditioned GRPO with
either decoder. For every protein it samples `group_size` molecules (8 by
default; 16 is also supported), validates and decodes SELFIES to SMILES, scores
only valid molecules with the selected frozen probability scorer (`RewardModel`
on SMILES or the FusionDTI ablation on SELFIES), normalizes the 0-1 rewards
within that protein's group, and applies the clipped GRPO objective with a
frozen reference-policy KL penalty. Invalid generations receive reward zero.

For the structure-aware FusionDTI ablation, load the released BindingDB binary
classifier and pass the Foldseek/SaProt structure-aware strings separately from
the generator's protein strings:

```python
from prot2mol.rewards import FusionDTIActivityScorer
from prot2mol.training import GRPOTrainer

reward_scorer = FusionDTIActivityScorer.from_pretrained(
    dataset="BindingDB",  # also: Biosnap or Human
    device="cuda",
    batch_size=8,
)
trainer = GRPOTrainer(
    policy=policy,
    reference_policy=reference_policy,
    optimizer=optimizer,
    tokenizer=molecule_tokenizer,
    reward_function=reward_scorer,
)
output = trainer.step(
    protein_input_ids=protein_input_ids,
    protein_attention_mask=protein_attention_mask,
    protein_sequences=amino_acid_sequences,
    reward_protein_sequences=structure_aware_sequences,
)
```

The scorer is frozen and kept in evaluation mode. It automatically receives
the generated SELFIES, encodes structure-aware proteins with
`westlake-repl/SaProt_650M_AF2`, encodes SELFIES with
`HUBioDataLab/SELFormer`, and returns the released sigmoid binary-activity
probability. The Space revision, both encoder revisions, checkpoint checksum,
and custom 428-token SELFIES vocabulary are pinned. The generator may still use
ESM2 or ProtT5; only `reward_protein_sequences` must contain SaProt-compatible
structure-aware tokens such as `MdEvLp`.

Every `step()` returns a `metrics` dictionary suitable for W&B or another
logger. Read the metrics together:

- Learning signal: `grpo/reward_mean`, `grpo/reward_std`,
  `grpo/group_reward_std_mean`, `grpo/zero_variance_group_fraction`,
  `grpo/valid_reward_low_saturation_fraction`, and
  `grpo/valid_reward_high_saturation_fraction`.
- Update direction: `grpo/advantage_weighted_logprob_change` should normally be
  positive; `grpo/mean_abs_logprob_change` confirms that an update occurred.
- Stability: `grpo/post_update_kl`, `grpo/post_update_old_policy_approx_kl`,
  `grpo/post_update_clip_fraction`, `grpo/post_update_ratio_mean`, and
  `grpo/grad_norm` expose overly large or clipped updates.
- Generation health: `grpo/valid_fraction`, `grpo/valid_unique_fraction`,
  `grpo/group_unique_fraction`, `grpo/eos_fraction`, and sequence-length
  metrics expose invalid-output, diversity-collapse, and termination failures.

A rising training reward is not enough by itself: reward increase accompanied
by falling validity or uniqueness, rising KL/clip fraction, or no held-out
generation improvement is a warning for collapse or reward-model exploitation.
The deterministic unit smoke tests exercise 2 proteins x 8 rollouts through
generation and a separate structure-aware-protein/SELFIES FusionDTI ablation
through 1 protein x 8 rollouts, grouped advantages, backpropagation, and a GPT-2
policy update. Checkpoint compatibility is strict; the released BindingDB head
has also been compared tensor-for-tensor with the upstream CAN implementation.
Set `PROT2MOL_RUN_FUSIONDTI_LIVE=1` to run the heavyweight tests with the real
pinned SaProt, SELFormer, tokenizer, and BindingDB checkpoint, including an
8-molecule FusionDTI reward group that drives an actual GPT-2 GRPO update.

### 3.2 Full random-assay GRPO training

The full runner derives one row per unique `protein_sequence` from the
target-aware random-assay **training** split and samples one eight-molecule group
per optimizer step. It freezes ESM2, the conditioning projection, the initial
reference policy, and FusionDTI; only the GPT-2 molecule decoder is optimized.

The current random-assay Parquet has plain amino-acid sequences but no
Foldseek/SaProt representation. Before launch, provide
`structure_aware_proteins.parquet` with these columns:

| column | meaning |
| --- | --- |
| `protein_accession` | join key present in the random-assay split |
| `structure_aware_sequence` | SaProt residue/3Di paired string, for example `MdEvLp` |

The runner validates complete coverage and rejects plain amino-acid strings in
that column. Launch with:

```bash
python prot2mol/main.py grpo --config prot2mol/configs/grpo.yaml
```

The supplied config matches the downloaded GPT-2 checkpoint: 12 layers, 16
heads, hidden size 1280, protein length 1000, molecule length 200, BF16, group
size 8, and protein batch size 1. GRPO is restricted to a fixed 40-protein
cohort: AKT1 (`P31749` / `CHEMBL4282`), CDK2 (`P24941` / `CHEMBL301`), and 38
seed-selected proteins with at least 200 unique active validation references.
Proteins longer than the generator's 1000-residue context are excluded from the
random selection. The exact cohort is saved as `cohort.json` and
`cohort.parquet` before model loading.

W&B receives every GRPO loss/reward/KL/advantage/update metric plus EOS,
truncation, validity, valid uniqueness, QED, SAS, logP, step time, process RAM,
and CUDA allocated/reserved/peak memory. Periodic evaluation adds macro reward,
validity, uniqueness, QED, SAS, logP, target-conditional FCD, and one
`eval/per_protein` table containing all 40 proteins. FCD is not computed from an
eight-molecule training group; the endpoint evaluation generates
40 x 64 = 2,560 molecules and compares each protein only with its own validation
actives. Only AKT1 and CDK2 get individual scalar series under
`eval/targets/<protein_accession>/...`; the other 38 do not create W&B plots.

Fixed-seed molecule-level start and end snapshots are written to:

```text
evaluation/start/generated_molecules.parquet
evaluation/start/per_protein_metrics.parquet
evaluation/end/generated_molecules.parquet
evaluation/end/per_protein_metrics.parquet
```

Each molecule row retains SELFIES, canonical SMILES, EOS/validity status,
FusionDTI probability, QED, SAS, and logP. Matching seeds make the baseline and
post-GRPO distributions directly comparable; both endpoint folders also include
JSON metrics and sampling metadata.

Intermediate checkpoints contain the trainable decoder, optimizer, scheduler,
RNG, epoch/protein position, W&B run ID, and base-checkpoint identity. Resume
without changing the base generator or training split:

```bash
python prot2mol/main.py grpo \
  --config prot2mol/configs/grpo.yaml \
  --resume_from_checkpoint outputs/prot2mol-grpo/random-assay-fusiondti-gpt2/checkpoint-500
```

### 3.3 Selectable Training Execution Mode

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
