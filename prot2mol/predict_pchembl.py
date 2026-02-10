#!/usr/bin/env python3
"""
pChEMBL Prediction Script for Prot2Mol

This script predicts pChEMBL values for molecules against protein targets.
It supports:
1. Direct protein sequences via 'Target_FASTA' column.
2. Protein ID lookup via 'Target_CHEMBL_ID' column (using --data_path).
3. Batch processing of mixed targets.
"""

import os
import sys
import json
import logging
import argparse
import warnings
from typing import List, Optional, Tuple

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datasets import load_dataset
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prot2mol.model import Prot2MolModel
from prot2mol.protein_encoders import get_protein_tokenizer, format_protein_sequences
from prot2mol.hf_utils import load_molgen_tokenizer
import selfies as sf
from rdkit import RDLogger

# Suppress warnings and logs
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
# Set plotting style
sns.set_theme(style="whitegrid")

class PChemblPredictor:
    """
    Predictor class for pChEMBL values using Prot2Mol architecture.
    """
    
    def __init__(self, config: argparse.Namespace):
        self.config = config
        self.logger = self._setup_logging()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.mol_tokenizer = None
        self.prot_tokenizer = None
        self.model = None
        self.sequence_cache = {}  # Cache for ID -> Sequence lookup
        
        # Load model and tokenizers
        self._auto_configure_model()
        self._load_components()

    def _auto_configure_model(self):
        """Attempt to load model configuration from config.json to override defaults."""
        config_path = os.path.join(self.config.model_path, "config.json")
        if os.path.exists(config_path):
             self.logger.info(f"Found config.json at {config_path}, loading configuration...")
             try:
                 with open(config_path, 'r') as f:
                     json_config = json.load(f)
                 
                 # Key parameters to sync
                 param_map = {
                     'n_layer': 'n_layer',
                     'n_head': 'n_head', 
                     'n_emb': 'n_emb',
                     'prot_emb_model': 'prot_emb_model',
                     'max_mol_len': 'max_mol_len',
                     'prot_max_length': 'prot_max_length'
                 }
                 
                 for json_key, arg_key in param_map.items():
                     if json_key in json_config:
                         json_val = json_config[json_key]
                         curr_val = getattr(self.config, arg_key)
                         if json_val != curr_val:
                             self.logger.info(f"Overriding default {arg_key}={curr_val} with config value {json_val}")
                             setattr(self.config, arg_key, json_val)
             except Exception as e:
                 self.logger.warning(f"Failed to load config.json: {e}")
        else:
             self.logger.warning(f"No config.json found in {self.config.model_path}. Using CLI arguments/defaults. Ensure they match the trained model!")
        
    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f'pchembl_prediction_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
        return logging.getLogger(__name__)

    def _load_components(self):
        """Load tokenizers and model."""
        self.logger.info("Loading components...")
        
        # Load molecule tokenizer
        self.logger.info("Loading molecule tokenizer...")
        models_base = os.environ.get('MODELS_BASE_PATH', '/gpfs/projects/etur29/atabey/models')
        if not os.path.exists(models_base):
            models_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        self.mol_tokenizer = load_molgen_tokenizer(models_base=models_base, padding_side="left")
        
        # Add SELFIES alphabet to tokenizer
        # Note: Ideally we should use the same alphabet as training. 
        # Here we adding standard selfies tokens or relying on what's in the model or adding from data if provided.
        # Since we might not load the full training dataset here, we rely on the tokenizer being robust or adding from input.
        # Ideally, we should add tokens from the input file if they are new.
        
        # Load protein tokenizer
        self.logger.info("Loading protein tokenizer...")
        self.prot_tokenizer = get_protein_tokenizer(self.config.prot_emb_model)
        
        # Load the prediction model
        self.logger.info(f"Loading prediction model from {self.config.model_path}")
        self.model = self._load_model(self.config.model_path)
        self.model.eval()

    def _load_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with tqdm(desc=f"Loading model weights", unit="MB") as pbar:
            model_state = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=self.device)
            pbar.update(1)
            
        model_config = {
            'prot_emb_model': self.config.prot_emb_model,
            'n_layer': self.config.n_layer,
            'n_head': self.config.n_head,
            'n_emb': self.config.n_emb,
            'max_mol_len': self.config.max_mol_len,
            'prot_max_length': self.config.prot_max_length,
            'train_encoder_model': False,
            'mol_tokenizer': self.mol_tokenizer
        }
        
        model = Prot2MolModel(model_config)
        # strict=True ensures we catch architecture mismatches immediately!
        try:
            model.load_state_dict(model_state, strict=True)
        except RuntimeError as e:
            self.logger.error(f"Failed to load model state dict strictly: {e}")
            self.logger.warning("Attempting strict=False load (NOT RECOMMENDED if architecture differs)...")
            model.load_state_dict(model_state, strict=False)
            
        model.to(self.device)
        return model

    def _get_sequence_for_id(self, target_id: str) -> Optional[str]:
        """
        Look up protein sequence for a given Target CHEMBL ID.
        Checks the data directory for existing files.
        """
        if target_id in self.sequence_cache:
            return self.sequence_cache[target_id]
        
        if not self.config.data_path or not os.path.exists(self.config.data_path):
            self.logger.warning(f"Data path not provided or invalid, cannot look up sequence for {target_id}")
            return None

        # Logic: Look for test_{target_id}.csv or check train.csv
        # 1. Check specific test file
        test_file = os.path.join(self.config.data_path, f"test_{target_id}.csv")
        if os.path.exists(test_file):
            try:
                df = pd.read_csv(test_file)
                if 'Target_FASTA' in df.columns:
                    seq = df['Target_FASTA'].iloc[0]
                    self.sequence_cache[target_id] = seq
                    return seq
            except Exception as e:
                self.logger.warning(f"Error reading {test_file}: {e}")

        # 2. Check general files if not found (e.g. test.csv or train.csv)
        # This might be slow if we do it for every ID. 
        # But we only do it if specific file not found.
        # Let's check 'test.csv' first
        common_files = ['test.csv', 'train.csv']
        for fname in common_files:
            fpath = os.path.join(self.config.data_path, fname)
            if os.path.exists(fpath):
                # We don't want to read the whole big file every time.
                # But without an index, we have to.
                # Optimization: Maybe load these once if we have many misses?
                # For now, let's assume specific test files exist or user provides FASTA.
                pass
        
        return None

    def _prepare_protein_embeddings(self, sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch tokenize protein sequences."""
        formatted_sequences = format_protein_sequences(sequences, self.config.prot_emb_model)
        
        prot_tokens = self.prot_tokenizer.batch_encode_plus(
            formatted_sequences,
            add_special_tokens=True,
            max_length=self.config.prot_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return prot_tokens['input_ids'].to(self.device), prot_tokens['attention_mask'].to(self.device)

    def evaluate(self):
        """
        Evaluate model performance on held-out test set replicating pretrain.py split logic.
        """
        self.logger.info(f"Loading dataset for evaluation from: {self.config.input_file}")

        # IMPORTANT: Load from preprocessed cache to match eval_pchembl.py!
        # Try to load preprocessed dataset first
        cache_dir = os.environ.get('DATASETS_CACHE_DIR', "/gpfs/projects/etur29/atabey/datasets")
        dataset_name = os.path.splitext(os.path.basename(self.config.input_file))[0]
        processed_data_path = os.path.join(cache_dir, dataset_name)

        if os.path.exists(processed_data_path):
            self.logger.info(f"Loading preprocessed dataset from {processed_data_path}")
            from datasets import load_from_disk
            dataset = load_from_disk(processed_data_path)
        else:
            self.logger.warning(f"Preprocessed data not found at {processed_data_path}")
            self.logger.warning("Falling back to loading raw CSV and tokenizing on-the-fly")
            # Load dataset using datasets library to match pretrain logic
            try:
                 dataset = load_dataset("csv", data_files=self.config.input_file)
            except Exception as e:
                 self.logger.error(f"Failed to load dataset: {e}")
                 return

        # Replicate split logic from pretrain.py
        self.logger.info("Splitting dataset (test_size=0.01, seed=42)...")
        dataset = dataset["train"].train_test_split(test_size=0.01, seed=42)
        test_data = dataset["test"]
        
        self.logger.info(f"Evaluation set size: {len(test_data)}")
        
        # Convert to DataFrame for easier processing with our predict primitives
        df_test = test_data.to_pandas()
        
        # Calculate normalization stats from the FULL dataset (train + test) to match training distribution
        # Note: In pretrain.py/eval_pchembl.py, stats are calc'd from the full CSV
        # dataset variable here is the FULL dataset before splitting (if we loaded it via load_dataset("csv"))
        # But wait, we did `dataset = dataset["train"].train_test_split(...)`
        # So `dataset` variable was reassigned to the DictDatasetWrapper.
        # We need access to the full original values ideally.
        
        # Actually, let's load the full DF securely to calc stats
        full_df = pd.read_csv(self.config.input_file)
        if 'pchembl_value_Median' in full_df.columns:
            valid_pchembl = full_df['pchembl_value_Median'].dropna()
            calc_mean = valid_pchembl.mean()
            calc_std = valid_pchembl.std(ddof=0) # Match eval_pchembl.py ddof=0 or default? eval uses ddof=0
            
            self.logger.info(f"Calculated pChEMBL stats from data: Mean={calc_mean:.4f}, Std={calc_std:.4f}")
            self.logger.info(f"Overriding defaults (Mean={self.config.pchembl_mean}, Std={self.config.pchembl_std})")
            
            self.config.pchembl_mean = calc_mean
            self.config.pchembl_std = calc_std
        else:
            self.logger.warning("Could not calculate pChEMBL stats from data (column missing). Using defaults.")

        # Check if we're using preprocessed data
        has_preprocessed_tokens = 'prot_input_ids' in test_data.column_names

        if has_preprocessed_tokens:
            self.logger.info("Using preprocessed tokens (same as eval_pchembl.py)...")
            results_df = self._predict_from_preprocessed(test_data)
        else:
            self.logger.info("Using on-the-fly tokenization (may give different results)...")
            # We need to ensure we can predict for these.
            # They should have Target_FASTA and SMILES/SELFIES.
            results_df = self._predict_dataframe(df_test)
        
        # Filter for valid pChEMBL values in ground truth
        if 'pchembl_value_Median' not in results_df.columns:
             self.logger.error("Dataset missing 'pchembl_value_Median' column. Cannot evaluate.")
             return
             
        # Drop rows where truth or prediction is NaN
        eval_df = results_df.dropna(subset=['pchembl_value_Median', 'Predicted_pChEMBL'])
        
        if len(eval_df) == 0:
             self.logger.error("No valid samples with both Truth and Prediction.")
             return
             
        y_true = eval_df['pchembl_value_Median'].values
        y_pred = eval_df['Predicted_pChEMBL'].values
        
        # metrics
        # metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        try:
            r2 = r2_score(y_true, y_pred)
        except:
            r2 = float('nan')
            
        pearson_corr, pearson_pval = pearsonr(y_true, y_pred)
        spearman_corr, spearman_pval = spearmanr(y_true, y_pred)

        self.logger.info("-" * 40)
        self.logger.info("Evaluation Metrics:")
        self.logger.info(f"MSE:  {mse:.4f}")
        self.logger.info(f"RMSE: {rmse:.4f}")
        self.logger.info(f"MAE:  {mae:.4f}")
        self.logger.info(f"R2:   {r2:.4f}")
        self.logger.info(f"Pearson: {pearson_corr:.4f} (p={pearson_pval:.4e})")
        self.logger.info(f"Spearman: {spearman_corr:.4f} (p={spearman_pval:.4e})")
        self.logger.info("-" * 40)
        
        # Save metrics (convert numpy types to Python native types for JSON serialization)
        metrics = {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R2': float(r2),
            'Pearson_r': float(pearson_corr),
            'Pearson_pval': float(pearson_pval),
            'Spearman_rho': float(spearman_corr),
            'Spearman_pval': float(spearman_pval),
            'Count': int(len(eval_df))
        }
        metrics_file = self.config.output_file.replace('.csv', '_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        # Plotting
        self._create_plots(y_true, y_pred)
        
        # Save csv
        self.logger.info(f"Saving evaluation results to {self.config.output_file}")
        results_df.to_csv(self.config.output_file, index=False)

    def _create_plots(self, y_true, y_pred):
        """Create extended distribution plots similar to eval_pchembl.py"""
        out_base = os.path.splitext(self.config.output_file)[0]
        
        # Calculate metrics for plot titles
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        pearson_corr, pearson_pval = pearsonr(y_true, y_pred)
        spearman_corr, spearman_pval = spearmanr(y_true, y_pred)

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Pearson Correlation
        ax1 = plt.subplot(2, 3, 1)
        ax1.scatter(y_true, y_pred, alpha=0.5, s=20, c='blue')
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax1.set_xlabel('True pChEMBL Values', fontsize=12)
        ax1.set_ylabel('Predicted pChEMBL Values', fontsize=12)
        ax1.set_title(f'Pearson Correlation: r = {pearson_corr:.4f}\n(p-value = {pearson_pval:.4e})', 
                     fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # Add text box with correlation info
        textstr = f'Pearson r: {pearson_corr:.4f}\nSignificance: {"***" if pearson_pval < 0.001 else "**" if pearson_pval < 0.01 else "*" if pearson_pval < 0.05 else "ns"}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
        # 2. KDE plot
        ax2 = plt.subplot(2, 3, 2)
        sns.kdeplot(data=y_true, label='True Values', color='blue', ax=ax2, fill=True, alpha=0.3)
        sns.kdeplot(data=y_pred, label='Predictions', color='red', ax=ax2, fill=True, alpha=0.3)
        ax2.set_xlabel('pChEMBL Value')
        ax2.set_ylabel('Density')
        ax2.set_title('Kernel Density Estimation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Scatter plot: Predicted vs True
        ax3 = plt.subplot(2, 3, 3)
        ax3.scatter(y_true, y_pred, alpha=0.5, s=10)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        ax3.set_xlabel('True pChEMBL Values')
        ax3.set_ylabel('Predicted pChEMBL Values')
        ax3.set_title(f'Scatter Plot (R² = {r2:.4f})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Residual plot
        ax4 = plt.subplot(2, 3, 4)
        residuals = y_pred - y_true
        ax4.scatter(y_true, residuals, alpha=0.5, s=10)
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.set_xlabel('True pChEMBL Values')
        ax4.set_ylabel('Residuals (Predicted - True)')
        ax4.set_title(f'Residual Plot (MAE = {mae:.4f})')
        ax4.grid(True, alpha=0.3)
        
        # 5. Spearman Correlation
        ax5 = plt.subplot(2, 3, 5)
        ax5.scatter(y_true, y_pred, alpha=0.5, s=20, c='purple')
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax5.set_xlabel('True pChEMBL Values', fontsize=12)
        ax5.set_ylabel('Predicted pChEMBL Values', fontsize=12)
        ax5.set_title(f'Spearman Correlation: ρ = {spearman_corr:.4f}\n(p-value = {spearman_pval:.4e})', 
                     fontsize=13, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        # Add text box with correlation info
        textstr = f'Spearman ρ: {spearman_corr:.4f}\nSignificance: {"***" if spearman_pval < 0.001 else "**" if spearman_pval < 0.01 else "*" if spearman_pval < 0.05 else "ns"}'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        ax5.text(0.05, 0.95, textstr, transform=ax5.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
        # 6. Violin plot
        ax6 = plt.subplot(2, 3, 6)
        df_plot = pd.DataFrame({
            'pChEMBL': np.concatenate([y_true, y_pred]),
            'Type': ['True'] * len(y_true) + ['Predicted'] * len(y_pred)
        })
        sns.violinplot(data=df_plot, x='Type', y='pChEMBL', ax=ax6)
        ax6.set_title('Violin Plot Comparison')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'pChEMBL Prediction Analysis (n={len(y_pred)})\n'
                    f'MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}, '
                    f'Pearson r={pearson_corr:.4f}, Spearman ρ={spearman_corr:.4f}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{out_base}_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _predict_from_preprocessed(self, eval_dataset):
        """Run prediction using preprocessed tokens (same as eval_pchembl.py)."""
        batch_size = self.config.batch_size
        all_preds = []
        all_true_values = []

        num_batches = (len(eval_dataset) + batch_size - 1) // batch_size

        self.logger.info(f"Starting prediction on {len(eval_dataset)} preprocessed samples...")

        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Predicting Batches"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(eval_dataset))

                batch = eval_dataset[start_idx:end_idx]

                # Use preprocessed tokens directly (same as eval_pchembl.py!)
                prot_input_ids = torch.tensor(batch['prot_input_ids']).to(self.device)
                prot_attention_mask = torch.tensor(batch['prot_attention_mask']).to(self.device)
                mol_input_ids = torch.tensor(batch['mol_input_ids']).to(self.device)
                pchembl_values = torch.tensor(batch['pchembl_values']).to(self.device)
                labels = torch.tensor(batch['labels']).to(self.device)

                # Forward pass (EXACTLY like eval_pchembl.py - including labels!)
                outputs = self.model(
                    prot_input_ids=prot_input_ids,
                    prot_attention_mask=prot_attention_mask,
                    mol_input_ids=mol_input_ids,
                    labels=labels,                    # ← CRITICAL: Must pass labels like eval_pchembl.py
                    pchembl_values=pchembl_values     # ← Also pass pchembl_values
                )

                # Extract predictions
                pchembl_preds = outputs['pchembl_predictions'].cpu().numpy()
                pchembl_true = pchembl_values.cpu().numpy()

                all_preds.extend(pchembl_preds)
                all_true_values.extend(pchembl_true)

        # Denormalize
        predictions_denorm = np.array(all_preds) * self.config.pchembl_std + self.config.pchembl_mean
        true_values_denorm = np.array(all_true_values) * self.config.pchembl_std + self.config.pchembl_mean

        # Create results dataframe
        results_df = eval_dataset.to_pandas()
        results_df['Predicted_pChEMBL'] = predictions_denorm
        results_df['pchembl_value_Median'] = true_values_denorm  # Add ground truth

        return results_df

    def _predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper to run prediction on a dataframe"""
        # Check required columns
        mol_col = None
        is_selfies = False
        
        # Identify molecule column
        for col in df.columns:
            if 'selfies' in col.lower():
                mol_col = col
                is_selfies = True
                break
        if not mol_col:
            for col in df.columns:
                if 'smiles' in col.lower():
                    mol_col = col
                    break
        
        if not mol_col:
            raise ValueError("Input file must contain a SMILES or SELFIES column")
            
        # Ensure we have Target Sequence
        if 'Target_FASTA' not in df.columns:
            self.logger.info("'Target_FASTA' column missing. Attempting lookup via 'Target_CHEMBL_ID'...")
            if 'Target_CHEMBL_ID' not in df.columns:
                raise ValueError("Input file must contain either 'Target_FASTA' or 'Target_CHEMBL_ID'")
            
            # Lookup sequences
            sequences = []
            
            unique_ids = df['Target_CHEMBL_ID'].unique()
            # ... (lookup logic) ... 
            # Reusing existing logic but applying to this DF
            # If I modify the DF in place it should be fine as it's a copy from load_dataset or read_csv
            
            found_count = 0
            for uid in tqdm(unique_ids, desc="Looking up sequences"):
                seq = self._get_sequence_for_id(uid)
                if seq:
                    self.sequence_cache[uid] = seq
                    found_count += 1
            
            df['Target_FASTA'] = df['Target_CHEMBL_ID'].map(self.sequence_cache)
            df = df.dropna(subset=['Target_FASTA'])
        
        if len(df) == 0:
            return df

        # Prepare Molecules
        molecules_list = df[mol_col].tolist()
        selfies_list = []
        if not is_selfies:
            self.logger.info("Converting SMILES to SELFIES...")
            for smi in tqdm(molecules_list, desc="SMILES->SELFIES"):
                try:
                    s = sf.encoder(smi)
                    selfies_list.append(s if s else "[nop]")
                except:
                    selfies_list.append("[nop]")
        else:
            selfies_list = molecules_list

        batch_size = self.config.batch_size
        all_preds = []
        protein_sequences = df['Target_FASTA'].tolist()
        num_batches = (len(df) + batch_size - 1) // batch_size
        
        self.logger.info(f"Starting prediction on {len(df)} samples...")
        
        for i in tqdm(range(num_batches), desc="Predicting Batches"):
            batch_slice = slice(i*batch_size, (i+1)*batch_size)
            batch_selfies = selfies_list[batch_slice]
            batch_prots = protein_sequences[batch_slice]
            
            # 1. Tokenize Proteins
            prot_ids, prot_mask = self._prepare_protein_embeddings(batch_prots)
            
            # 2. Tokenize Molecules
            mol_tokens = self.mol_tokenizer.batch_encode_plus(
                batch_selfies,
                add_special_tokens=True,
                truncation=True,
                max_length=self.config.max_mol_len,
                padding='max_length',
                return_tensors='pt'
            )
            mol_ids = mol_tokens['input_ids'].to(self.device)
            mol_mask = mol_tokens['attention_mask'].to(self.device)
            
            # 3. Predict
            with torch.no_grad():
                outputs = self.model(
                    mol_input_ids=mol_ids,
                    prot_input_ids=prot_ids,
                    prot_attention_mask=prot_mask,
                    train_lm=False
                )
                preds = outputs['pchembl_predictions'].cpu().numpy()
                denorm_preds = preds * self.config.pchembl_std + self.config.pchembl_mean
                all_preds.extend(denorm_preds)

        df['Predicted_pChEMBL'] = all_preds
        return df

    def predict(self):
        self.logger.info(f"Loading input file: {self.config.input_file}")
        df = pd.read_csv(self.config.input_file)
        
        result_df = self._predict_dataframe(df)
        
        # Save
        self.logger.info(f"Saving results to {self.config.output_file}")
        result_df.to_csv(self.config.output_file, index=False)
        self.logger.info("Done.")

def parse_args():
    parser = argparse.ArgumentParser(description="Predict pChEMBL values")
    
    parser.add_argument("--input_file", required=True, help="Input CSV with molecules and targets")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--output_file", required=True, help="Output CSV path")
    parser.add_argument("--data_path", help="Path to dataset directory for looking up sequences (if Target_FASTA missing)")
    
    # Model Params
    parser.add_argument("--prot_emb_model", default="prot_t5", choices=["prot_t5", "esm2", "saprot"])
    parser.add_argument("--n_layer", type=int, default=1)
    parser.add_argument("--n_head", type=int, default=16)
    parser.add_argument("--n_emb", type=int, default=1024)
    parser.add_argument("--prot_max_length", type=int, default=1024)
    parser.add_argument("--max_mol_len", type=int, default=200)
    
    # Prediction Params
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pchembl_mean", type=float, default=5.924)
    parser.add_argument("--pchembl_std", type=float, default=1.362)
    
    parser.add_argument("--eval", action="store_true", help="Run evaluation on held-out test set (split from input file)")

    return parser.parse_args()

def main():
    args = parse_args()
    predictor = PChemblPredictor(args)
    if args.eval:
        predictor.evaluate()
    else:
        predictor.predict()

if __name__ == "__main__":
    main()
