#!/usr/bin/env python3
"""
One-time dataset preprocessing for Prot2Mol training.

This script preprocesses (tokenizes) large datasets without distributed training overhead.
Run this ONCE before training, then all training runs will use the cached data.

Usage:
    python preprocess_dataset.py --selfies_path /path/to/dataset.csv --prot_emb_model prot_t5 --cache_dir /path/to/cache

After preprocessing, the cached data will be saved to the specified cache directory (default: /gpfs/projects/etur29/atabey/datasets).
"""

import os
import sys
import argparse
import logging
import re
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from transformers import BartTokenizer
from prot2mol.protein_encoders import get_protein_tokenizer
import torch

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class DatasetPreprocessor:
    def __init__(self, config):
        self.config = config
        self.logger = self._setup_logging()
        
        # Prepare pChEMBL normalization
        self._prepare_normalization()
        
        # Initialize tokenizers
        self._init_tokenizers()
        
        # Define cache path
        if config.cache_dir:
            self.cache_dir = config.cache_dir
        else:
            self.cache_dir = os.environ.get('DATASETS_CACHE_DIR', "/gpfs/projects/etur29/atabey/datasets")

        dataset_name = config.selfies_path.split("/")[-1].split(".")[0]
        self.processed_data_path = os.path.join(self.cache_dir, dataset_name)
        
        self.logger.info(f"Cache will be saved to: {self.processed_data_path}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"preprocessing_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file)
            ]
        )
        return logging.getLogger(__name__)
    
    def _prepare_normalization(self):
        """Calculate pChEMBL normalization constants."""
        import pandas as pd
        self.logger.info("Preparing normalization constants from dataset...")
        df = pd.read_csv(self.config.selfies_path)
        pchembl_values = df['pchembl_value_Median'].dropna()
        
        self.pchembl_mean = pchembl_values.mean()
        self.pchembl_std  = pchembl_values.std(ddof=0)
        self.pchembl_threshold = 6.0
        
        self.logger.info(f"pChEMBL normalization range: mean={self.pchembl_mean:.3f}, std={self.pchembl_std:.3f}")
        self.logger.info(f"pChEMBL positive threshold set to: >={self.pchembl_threshold}")
    
    def _init_tokenizers(self):
        """Initialize tokenizers for proteins and molecules."""
        self.logger.info("Initializing tokenizers...")
        
        # Molecule tokenizer
        mol_model_path = self._get_model_path("zjunlp--MolGen-large")
        self.mol_tokenizer = BartTokenizer.from_pretrained(mol_model_path, padding_side="left")
        
        # Protein tokenizer
        self.prot_tokenizer = get_protein_tokenizer(self.config.prot_emb_model)
        
        self.logger.info("Tokenizers initialized successfully")
    
    def _get_model_path(self, model_name):
        """Get the correct path for a locally cached model."""
        models_base = os.environ.get('MODELS_BASE_PATH', './models')
        base_path = os.path.join(models_base, f"models--{model_name}")
        snapshots_path = os.path.join(base_path, "snapshots")
        
        if os.path.exists(snapshots_path):
            snapshots = os.listdir(snapshots_path)
            if snapshots:
                return os.path.join(snapshots_path, snapshots[0])
        
        return base_path
    
    def tokenize_prot_function(self, batch):
        """Tokenize protein sequences."""
        try:
            # Replace non-standard amino acids
            if self.config.prot_emb_model == "prot_t5":
                sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in batch["Target_FASTA"]]
            else:
                sequence_examples = [re.sub(r"[UZOB]", "X", seq) for seq in batch["Target_FASTA"]]
            
            # Tokenize
            ids = self.prot_tokenizer.batch_encode_plus(
                sequence_examples,
                add_special_tokens=True,
                truncation=True,
                max_length=self.config.prot_max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            return {
                'prot_input_ids': ids['input_ids'],
                'prot_attention_mask': ids['attention_mask']
            }
        except Exception as e:
            self.logger.error(f"Error in protein tokenization: {str(e)}")
            raise
    
    def tokenize_mol_function(self, batch):
        """Tokenize molecule SELFIES strings."""
        try:
            # Tokenize SELFIES
            ids = self.mol_tokenizer.batch_encode_plus(
                batch["Compound_SELFIES"],
                add_special_tokens=True,
                truncation=True,
                max_length=self.config.max_mol_len,
                padding="max_length",
                return_tensors="pt"
            )
            
            pchembl_values = batch["pchembl_value_Median"]
            labels = ids['input_ids'].clone()
            
            # Mask padded positions in labels
            pad_mask = ids['input_ids'] == self.mol_tokenizer.pad_token_id
            labels[pad_mask] = -100
            
            # Normalize pchembl values and determine train_lm flags
            normalized_pchembl = []
            train_lm_flags = []
            
            for val in pchembl_values:
                train_lm_flags.append(val >= self.pchembl_threshold)
                normalized_val = (val - self.pchembl_mean) / (self.pchembl_std + 1e-8)
                normalized_pchembl.append(normalized_val)
            
            return {
                'mol_input_ids': ids['input_ids'],
                'mol_attention_mask': ids['attention_mask'],
                'labels': labels,
                'pchembl_values': torch.tensor(normalized_pchembl, dtype=torch.float),
                'train_lm': torch.tensor(train_lm_flags, dtype=torch.bool)
            }
        except Exception as e:
            self.logger.error(f"Error in molecule tokenization: {str(e)}")
            raise
    
    def preprocess(self):
        """Main preprocessing function."""
        # Check if already processed
        if os.path.exists(self.processed_data_path):
            self.logger.warning(f"Processed dataset already exists at: {self.processed_data_path}")
            response = input("Do you want to reprocess? (yes/no): ").lower()
            if response != 'yes':
                self.logger.info("Skipping preprocessing. Using existing cache.")
                return
            else:
                self.logger.info("Removing existing cache and reprocessing...")
                import shutil
                shutil.rmtree(self.processed_data_path)
        
        # Load raw dataset
        self.logger.info(f"Loading dataset from: {self.config.selfies_path}")
        dataset = load_dataset("csv", data_files=self.config.selfies_path, cache_dir=self.cache_dir)
        
        self.logger.info(f"Dataset loaded: {len(dataset['train'])} samples")
        
        # Tokenize protein sequences
        self.logger.info("Tokenizing protein sequences...")
        self.logger.info(f"Using num_proc={self.config.num_proc}, batch_size={self.config.batch_size}")
        
        dataset = dataset.map(
            self.tokenize_prot_function,
            batched=True,
            num_proc=self.config.num_proc,
            batch_size=self.config.batch_size,
            desc="Tokenizing protein sequences"
        )
        
        # Tokenize molecule sequences
        self.logger.info("Tokenizing molecule sequences...")
        dataset = dataset.map(
            self.tokenize_mol_function,
            batched=True,
            num_proc=self.config.num_proc,
            batch_size=self.config.batch_size,
            desc="Tokenizing molecule sequences"
        )
        
        # Save processed dataset
        os.makedirs(self.cache_dir, exist_ok=True)
        self.logger.info(f"Saving processed dataset to: {self.processed_data_path}")
        dataset.save_to_disk(self.processed_data_path)
        
        self.logger.info("✅ Preprocessing completed successfully!")
        self.logger.info(f"Cache saved to: {self.processed_data_path}")
        self.logger.info(f"You can now run training - it will load from this cache instantly!")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess dataset for Prot2Mol training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--selfies_path",
        required=True,
        help="Path to the SELFIES dataset CSV file"
    )
    parser.add_argument(
        "--prot_emb_model",
        default="prot_t5",
        choices=["prot_t5", "esm2", "saprot"],
        help="Protein embedding model to use"
    )
    parser.add_argument(
        "--max_mol_len",
        type=int,
        default=200,
        help="Maximum molecule sequence length"
    )
    parser.add_argument(
        "--prot_max_length",
        type=int,
        default=1000,
        help="Maximum protein sequence length"
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=8,
        help="Number of processes for parallel tokenization"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size for tokenization"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to store processed datasets (overrides default path)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Verify file exists
    if not os.path.exists(args.selfies_path):
        print(f"ERROR: File not found: {args.selfies_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("Prot2Mol Dataset Preprocessing")
    print("=" * 80)
    print(f"Dataset: {args.selfies_path}")
    print(f"Protein model: {args.prot_emb_model}")
    print(f"Max molecule length: {args.max_mol_len}")
    print(f"Max protein length: {args.prot_max_length}")
    print(f"Parallel processes: {args.num_proc}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 80)
    print()
    
    # Create preprocessor and run
    preprocessor = DatasetPreprocessor(args)
    preprocessor.preprocess()
    
    print()
    print("=" * 80)
    print("✅ Done! Your dataset is ready for training.")
    print("=" * 80)


if __name__ == "__main__":
    main()

