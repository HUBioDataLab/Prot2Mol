#!/usr/bin/env python3
"""
One-time dataset preprocessing for Prot2Mol training.

This script preprocesses (tokenizes) large datasets without distributed training overhead.
Run this ONCE before training, then all training runs will use the cached data.

Usage:
    python preprocess_dataset.py --selfies_path /path/to/dataset.csv --prot_emb_model prot_t5 --cache_dir /path/to/cache

After preprocessing, the cached data will be saved to the specified cache directory (default: /gpfs/projects/etur29/atabey/datasets).

Use --eval_split and --eval_split_ratio to compute pChEMBL normalization from the
train split only (recommended to avoid leakage).
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from prot2mol.protein_encoders import get_protein_tokenizer, format_protein_sequences
from prot2mol.hf_utils import load_molgen_tokenizer
import torch

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class DatasetPreprocessor:
    def __init__(self, config):
        self.config = config
        self.logger = self._setup_logging()

        # Define cache path early (used for saving stats)
        if config.cache_dir:
            self.cache_dir = config.cache_dir
        else:
            self.cache_dir = os.environ.get('DATASETS_CACHE_DIR', "/gpfs/projects/etur29/atabey/datasets")

        dataset_name = config.selfies_path.split("/")[-1].split(".")[0]
        self.processed_data_path = os.path.join(self.cache_dir, dataset_name)
        
        self.logger.info(f"Cache will be saved to: {self.processed_data_path}")
        
        # Prepare pChEMBL normalization
        self._prepare_normalization()
        self._prepare_group_ids()
        
        # Initialize tokenizers
        self._init_tokenizers()
    
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
        usecols = ["pchembl_value_Median"]
        if self.config.eval_split == "aid":
            usecols.append("AID")
        df = pd.read_csv(self.config.selfies_path, usecols=usecols)

        if self.config.eval_split == "aid" and "AID" in df.columns:
            aids = df["AID"].astype(str).unique()
            rng = np.random.RandomState(self.config.split_seed)
            rng.shuffle(aids)
            n_holdout = max(1, int(len(aids) * self.config.eval_split_ratio))
            holdout_aids = set(aids[:n_holdout])
            train_mask = ~df["AID"].astype(str).isin(holdout_aids)
            pchembl_values = df.loc[train_mask, "pchembl_value_Median"].dropna()
            self.logger.info(
                f"Using AID hold-out for normalization: {n_holdout} AIDs held out "
                f"({self.config.eval_split_ratio:.3f} of {len(aids)})"
            )
        else:
            pchembl_values = df["pchembl_value_Median"].dropna()
            if 0.0 < self.config.eval_split_ratio < 1.0:
                pchembl_values = pchembl_values.sample(
                    frac=1.0 - self.config.eval_split_ratio,
                    random_state=self.config.split_seed
                )
        
        self.pchembl_mean = pchembl_values.mean()
        self.pchembl_std  = pchembl_values.std(ddof=0)
        self.pchembl_threshold = 6.0
        
        self.logger.info(f"pChEMBL normalization range: mean={self.pchembl_mean:.3f}, std={self.pchembl_std:.3f}")
        self.logger.info(f"pChEMBL positive threshold set to: >={self.pchembl_threshold}")

        # Persist stats for training/eval to avoid leakage
        try:
            os.makedirs(self.processed_data_path, exist_ok=True)
            stats_path = os.path.join(self.processed_data_path, "pchembl_stats.json")
            import json
            with open(stats_path, "w") as f:
                json.dump({
                    "pchembl_mean": float(self.pchembl_mean),
                    "pchembl_std": float(self.pchembl_std),
                    "pchembl_threshold": float(self.pchembl_threshold),
                    "eval_split": self.config.eval_split,
                    "eval_split_ratio": float(self.config.eval_split_ratio),
                    "split_seed": int(self.config.split_seed)
                }, f, indent=2)
            self.logger.info(f"Saved pChEMBL stats to: {stats_path}")
        except Exception as e:
            self.logger.warning(f"Could not save pChEMBL stats: {e}")

    def _prepare_group_ids(self):
        """Prepare AID+Target_ID group mapping for pairwise ranking loss."""
        import pandas as pd
        self.group_id_map = None
        try:
            df = pd.read_csv(self.config.selfies_path, usecols=["AID", "Target_ID"])
        except Exception as e:
            self.logger.warning(f"Could not load AID/Target_ID columns for grouping: {e}")
            return

        if "AID" not in df.columns or "Target_ID" not in df.columns:
            self.logger.warning("AID/Target_ID columns not found in dataset. Pairwise loss will be disabled.")
            return

        group_keys = df["AID"].astype(str) + "__" + df["Target_ID"].astype(str)
        unique_keys = pd.unique(group_keys)
        self.group_id_map = {k: i for i, k in enumerate(unique_keys)}
        self.logger.info(f"Prepared group_id map with {len(self.group_id_map):,} AID+Target_ID groups")
    
    def _init_tokenizers(self):
        """Initialize tokenizers for proteins and molecules."""
        self.logger.info("Initializing tokenizers...")
        
        # Molecule tokenizer
        self.mol_tokenizer = load_molgen_tokenizer(padding_side="left")
        
        # Protein tokenizer
        self.prot_tokenizer = get_protein_tokenizer(self.config.prot_emb_model)
        
        self.logger.info("Tokenizers initialized successfully")
    
    def tokenize_prot_function(self, batch):
        """Tokenize protein sequences."""
        try:
            sequence_examples = format_protein_sequences(batch["Target_FASTA"], self.config.prot_emb_model)
            
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

            group_ids = None
            if self.group_id_map is not None and "AID" in batch and "Target_ID" in batch:
                group_keys = [f"{a}__{t}" for a, t in zip(batch["AID"], batch["Target_ID"])]
                group_ids = [self.group_id_map.get(k, -1) for k in group_keys]
            
            return {
                'mol_input_ids': ids['input_ids'],
                'mol_attention_mask': ids['attention_mask'],
                'labels': labels,
                'pchembl_values': torch.tensor(normalized_pchembl, dtype=torch.float),
                'train_lm': torch.tensor(train_lm_flags, dtype=torch.bool),
                'group_id': torch.tensor(group_ids, dtype=torch.long) if group_ids is not None else torch.tensor([-1] * len(batch["Compound_SELFIES"]), dtype=torch.long)
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
    parser.add_argument(
        "--eval_split",
        type=str,
        default="random",
        choices=["random", "aid"],
        help="Split strategy used for normalization statistics"
    )
    parser.add_argument(
        "--eval_split_ratio",
        type=float,
        default=0.01,
        help="Fraction of data (or AIDs) held out for validation"
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Random seed for split reproducibility"
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
