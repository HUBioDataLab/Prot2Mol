# Prot2Mol
Target based molecule generation using protein embeddings and SELFIES molecule representation.

&nbsp;
&nbsp;

# Installation

```bash
git clone https://github.com/atabeyunlu/Prot2Mol.git

pip install -r requirements.yaml
```

&nbsp;
&nbsp;

# How to run?

Prot2Mol model can be run following the below documentation top-down. It is necessary to prepare Protein-Compound data however you can process a single protein embedding to run the model. 

# Data Processing
&nbsp;
&nbsp;

## Prepare Papyrus Protein-Compound Data

This script prepares Papyrus data by downloading and decompressing molecule and protein files from specified URLs.
It then processes the data by filtering based on specified thresholds and parameters.
The processed data is saved as a CSV file in the specified output directory.

Usage:
```bash
python papyrus_data.py [--pchembl_threshold P] [--prot_len L] [--human_only H]
```
```bash
Arguments:
    --pchembl_threshold (int): pchembl threshold for filtering compounds (default: None)
    --prot_len (int): maximum protein length for filtering proteins (default: None)
    --human_only (bool): flag to filter only human proteins (default: False)
```
Example:
```bash
python papyrus_data.py --pchembl_threshold 6 --prot_len 500 --human_only True
```
&nbsp;
&nbsp;
## AlphaFold2 Embedding Generator for Protein Sequences

This script downloads, processes, and organizes AlphaFold2 (AF2) embeddings into a format suitable for further analysis or model training. It handles the downloading of zipped AF2 embedding files, unzipping them, padding protein structures, and saving the embeddings in `.npz` format.


```bash
python af2_embeddings.py [--max_len L]
```
```bash
Arguments:
    --max_len (int): Maximum protein length to pad the embeddings to (default: 500).
```
Example:

```bash
python af2_embeddings.py --max_len 500
```
&nbsp;
&nbsp;
## ESM-2 Embedding Generator for Protein Sequences

This script generates embeddings for protein sequences using the ESM-2 model. It processes a dataset of protein sequences, filters them based on a specified maximum length, and saves the resulting embeddings along with the corresponding protein IDs.


```bash
python esm2_embeddings.py [--dataset PATH] [--prot_len L]
```
```bash
Arguments:
    --dataset (str): Path to the input dataset containing protein sequences (default: ../data/papyrus/prot_comp_set_pchembl_None_protlen_500_human_False.csv).
    --prot_len (int): Maximum length of the protein sequences to be considered for embedding (default: 500).
```
Example:

```bash
python esm2_embeddings.py --dataset ../data/my_dataset.csv --prot_len 500
```

&nbsp;
&nbsp;
## ESM-3 Embedding Generator for Protein Sequences

This script generates embeddings for protein sequences using the ESM-3 model. It processes a dataset of protein sequences, applies padding, and filters sequences based on a specified maximum length. The resulting embeddings are then saved along with the corresponding protein IDs.


```bash
python esm3_embeddings.py [--dataset PATH] [--max_len L] [--huggingface_token TOKEN]
```
```bash
Arguments:
    --dataset (str): Path to the input dataset containing protein sequences (default: ../data/papyrus/prot_comp_set_pchembl_6_protlen_500.csv).
    --max_len (int): Maximum length of the protein sequences to be considered for embedding (default: 500).
    --huggingface_token (str): User's Hugging Face token for authentication (required).
```
Example:

```bash
python esm3_embeddings.py --dataset ../data/my_dataset.csv --max_len 500 --huggingface_token my_hf_token
```

&nbsp;
&nbsp;
## ProtT5 Embedding Generator for Protein Sequences

This script generates protein embeddings using the ProtT5 model from the Rostlab. It processes a dataset containing protein sequences, encodes the sequences using the ProtT5 model, and saves the resulting embeddings in `.npz` format.



```bash
python prot_t5_embeddings.py [--dataset DATASET_PATH] [--prot_len PROTEIN_LENGTH]
```
```bash
Arguments:
    --dataset (str): Path to the input CSV file containing protein sequences (default: ../data/papyrus/prot_comp_set_pchembl_8_protlen_150_human_False.csv).
    --prot_len (int): Maximum length of the protein sequences to consider (default: 500).
```
Example:

```bash
python prot_t5_embeddings.py --dataset ../data/my_protein_data.csv --prot_len 200
```
&nbsp;
&nbsp;
# Model & Training
&nbsp;
&nbsp;
## Prot2Mol Pre-Training Script

This script is designed to train and evaluate a GPT-2 model with cross-attention for generating molecular structures based on protein embeddings. The script utilizes SELFIES strings, and the protein embeddings can be derived from various models like ProtT5, ESM, or AlphaFold2 embeddings.


```bash
python pretrain.py [--selfies_path SELFIES_PATH] [--prot_emb_model PROT_EMB_MODEL] [--prot_ID PROT_ID] [--learning_rate LEARNING_RATE] [--train_batch_size TRAIN_BATCH_SIZE] [--valid_batch_size VALID_BATCH_SIZE] [--epoch EPOCH] [--weight_decay WEIGHT_DECAY] [--n_layer N_LAYER] [--n_head N_HEAD]
```
```bash
Arguments:

Dataset Parameters:
    --selfies_path (str): Path to the CSV file containing SELFIES strings and other related data (default: ../data/papyrus/prot_comp_set_pchembl_8_protlen_150_human_False.csv).
    --prot_emb_model (str): Specifies the protein embedding model to use (choices: prot_t5, esm2, esm3, af2_single, af2_struct, af2_combined; default: prot_t5).
    --prot_ID (str): Protein ID for filtering the dataset (default: CHEMBL4282).

Model Parameters:
    --learning_rate (float): Learning rate for the optimizer (default: 1.0e-5).
    --train_batch_size (int): Batch size for training (default: 64).
    --valid_batch_size (int): Batch size for validation (default: 64).
    --epoch (int): Number of training epochs (default: 50).
    --weight_decay (float): Weight decay for the optimizer (default: 0.0005).
    --n_layer (int): Number of layers in the GPT-2 model (default: 1).
    --n_head (int): Number of attention heads in the GPT-2 model (default: 4).
```
Example:

```bash 
python pretrain.py --selfies_path ../data/my_selfies_data.csv --prot_emb_model esm3 --prot_ID CHEMBL4296327 --learning_rate 2e-5 --train_batch_size 32 --epoch 30 --n_layer 4 --n_head 8
```
&nbsp;
&nbsp;
## Prot2Mol Fine-Tuning Script

This script fine-tunes a pre-trained Prot2Mol model on a specific target protein embedding. The fine-tuning process is tailored to a specific target ID (e.g., a ChEMBL ID) and involves further training the model on a subset of data related to that target.


```bash
python finetune.py [--selfies_path SELFIES_PATH] [--target_id TARGET_ID] [--prot_emb_model PROT_EMB_MODEL] [--pretrained_model_path PRETRAINED_MODEL_PATH] [--learning_rate LEARNING_RATE] [--train_batch_size TRAIN_BATCH_SIZE] [--valid_batch_size VALID_BATCH_SIZE] [--epoch EPOCH] [--weight_decay WEIGHT_DECAY] [--n_layer N_LAYER]
```
```bash
Arguments:

Dataset Parameters:
    --selfies_path (str): Path to the CSV file containing SELFIES strings and other related data (default: ../data/fasta_to_selfies_500.csv).
    --target_id (str): The ChEMBL ID of the target protein for fine-tuning (default: CHEMBL4282).
    --prot_emb_model (str): Specifies the protein embedding model to use (choices: prot_t5, esm2, esm3, af2_single, af2_struct, af2_combined; default: prot_t5).

Model Parameters:
    --pretrained_model_path (str): Path to the pre-trained model checkpoint to be fine-tuned (default: ./saved_models/set_100_saved_model/checkpoint-31628).
    --learning_rate (float): Learning rate for the optimizer during fine-tuning (default: 1.0e-5).
    --train_batch_size (int): Batch size for fine-tuning (default: 64).
    --valid_batch_size (int): Batch size for validation during fine-tuning (default: 64).
    --epoch (int): Number of epochs for fine-tuning (default: 50).
    --weight_decay (float): Weight decay for the optimizer during fine-tuning (default: 0.0005).
    --n_layer (int): Number of layers in the GPT-2 model during fine-tuning (default: 4).
```
Example:

```bash
python prot2mol_finetune_script.py --selfies_path ../data/my_selfies_data.csv --target_id CHEMBL12345 --prot_emb_model esm3 --pretrained_model_path ./saved_models/my_pretrained_model --learning_rate 2e-5 --train_batch_size 32 --epoch 30 --n_layer 6
```
&nbsp;
&nbsp;
## Molecule Generation 

This script is designed to generate molecular structures based on a pretrained model and evaluate them against a reference dataset. It loads the necessary protein embeddings and molecular data, generates new molecules, and calculates evaluation metrics. The generated molecules and metrics are then saved to specified files.

## Usage:
```bash
python produce_molecules.py [--model_file PATH] [--prot_emb_model PATH] [--generated_mol_file PATH] [--selfies_path PATH] [--attn_output BOOL] [--prot_id ID] [--num_samples N] [--bs N]
```
```bash
Arguments:
    --model_file (str): Path of the pretrained model file (default: ./finetuned_models/set_100_finetuned_model/checkpoint-3100).
    --prot_emb_model (str): Path of the pretrained protein embedding model (default: ./data/prot_embed/prot_t5/prot_comp_set_pchembl_None_protlen_None/embeddings).
    --generated_mol_file (str): Path of the output file where generated molecules will be saved (default: ./saved_mols/_kt_finetune_mols.csv).
    --selfies_path (str): Path of the input SELFIES dataset (default: ./data/papyrus/prot_comp_set_pchembl_None_protlen_500_human_False).
    --attn_output (bool): Flag to output attention weights during molecule generation (default: False).
    --prot_id (str): Target Protein ID for molecule generation (default: CHEMBL4282).
    --num_samples (int): Number of samples to generate (default: 10000).
    --bs (int): Batch size for molecule generation (default: 100).
```
Example:
```bash
python produce_molecules.py --model_file ./finetuned_models/set_100_finetuned_model/checkpoint-3100  --prot_emb_model ./data/prot_embed/prot_t5/prot_comp_set_pchembl_None_protlen_None/embeddings --generated_mol_file ./saved_mols/generated_molecules.csv  --selfies_path ./data/papyrus/prot_comp_set_pchembl_None_protlen_500_human_False --attn_output False  --prot_id CHEMBL4282  --num_samples 10000  --bs 100
```

