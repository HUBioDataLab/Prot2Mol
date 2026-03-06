#!/usr/bin/env python3
"""
pChEMBL Prediction Script for Prot2Mol

This script predicts pChEMBL values for molecules against protein targets.
It supports:
1. Direct protein sequences via 'Target_FASTA' column.
2. UniProt accession lookup via 'UniProt_ID' + protein targets TSV.
3. Target CHEMBL lookup via 'Target_CHEMBL_ID' + CHEMBL->UniProt mapping.
4. Batch processing of mixed targets.
"""

import os
import sys
import json
import logging
import argparse
import warnings
import time
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from prot2mol.io.config import parse_args_with_config
from prot2mol.core.protein_encoders import get_protein_tokenizer
from prot2mol.io.hf_utils import (
    find_model_config_path,
    load_molgen_tokenizer,
    load_prot2mol_inference_model,
    load_saved_model_config,
)
from prot2mol.data.pipeline import (
    find_molecule_column,
    load_processed_dataset,
    split_train_eval_dataset,
    to_selfies_list,
    tokenize_protein_sequences_for_inference,
    tokenize_selfies_for_inference,
)
from prot2mol.training.metrics import compute_pchembl_metrics
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
        self._chembl_to_uniprot: Optional[Dict[str, str]] = None
        self._target_id_to_sequence: Optional[Dict[str, str]] = None
        
        # Load model and tokenizers
        self._auto_configure_model()
        self._load_components()

    def _resolve_model_config_path(self) -> Optional[str]:
        """Find config.json in checkpoint dir or its parents."""
        return find_model_config_path(self.config.model_path)

    def _auto_configure_model(self):
        """Attempt to load model configuration from config.json to override defaults."""
        config_path = self._resolve_model_config_path()
        if config_path is not None:
             self.logger.info(f"Found config.json at {config_path}, loading configuration...")
             try:
                 json_config = load_saved_model_config(self.config.model_path, logger=self.logger)

                 for arg_key, json_val in json_config.items():
                     if not hasattr(self.config, arg_key):
                         continue
                     curr_val = getattr(self.config, arg_key)
                     if json_val != curr_val:
                         self.logger.info(f"Overriding default {arg_key}={curr_val} with config value {json_val}")
                         setattr(self.config, arg_key, json_val)
             except Exception as e:
                 self.logger.warning(f"Failed to load config.json: {e}")
        else:
             self.logger.warning(
                 "No config.json found in checkpoint directory or parents for %s. "
                 "Using CLI/YAML values; ensure architecture args match training.",
                 self.config.model_path,
             )
        
    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            force=True,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f'pchembl_prediction_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
        logger = logging.getLogger(__name__)
        logger.propagate = False
        return logger

    def _load_components(self):
        """Load tokenizers and model."""
        self.logger.info("Loading components...")
        
        # Load molecule tokenizer
        self.logger.info("Loading molecule tokenizer...")
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        configured_base = getattr(self.config, "models_base", None)
        models_base = configured_base or os.environ.get("MODELS_BASE_PATH")
        fallback_bases = [
            os.path.join(project_root, "models"),
            os.path.join(os.path.expanduser("~"), "Prot2Mol", "models"),
        ]
        try:
            self.mol_tokenizer = load_molgen_tokenizer(
                models_base=models_base,
                fallback_bases=fallback_bases,
                padding_side="left",
            )
        except OSError:
            self.logger.error(
                "Failed to load MolGen tokenizer. Set --models_base (or MODELS_BASE_PATH) "
                "to the folder containing 'models--zjunlp--MolGen-large'. "
                "Example: /home/<user>/Prot2Mol/models"
            )
            raise
        
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
        with tqdm(desc="Loading model weights", unit="model") as pbar:
            model = load_prot2mol_inference_model(
                model_path=model_path,
                device=self.device,
                mol_tokenizer=self.mol_tokenizer,
                prot_emb_model=self.config.prot_emb_model,
                n_layer=self.config.n_layer,
                n_head=self.config.n_head,
                n_emb=self.config.n_emb,
                max_mol_len=self.config.max_mol_len,
                prot_max_length=self.config.prot_max_length,
                pchembl_tf_hidden_dim=self.config.pchembl_tf_hidden_dim,
                pchembl_tf_num_heads=self.config.pchembl_tf_num_heads,
                pchembl_tf_group_size=self.config.pchembl_tf_group_size,
                pchembl_tf_agg_mode=self.config.pchembl_tf_agg_mode,
                pchembl_tf_dropout=self.config.pchembl_tf_dropout,
                strict=True,
                allow_strict_fallback=True,
                logger=self.logger,
            )
            pbar.update(1)
        return model

    def _resolve_optional_file(
        self,
        configured_path: Optional[str],
        env_var: Optional[str],
        candidates: List[str],
    ) -> Optional[str]:
        """Return first existing file path among explicit/env/candidate locations."""
        ordered = []
        if configured_path:
            ordered.append(configured_path)
        if env_var:
            env_value = os.environ.get(env_var)
            if env_value:
                ordered.append(env_value)
        ordered.extend(candidates)

        seen = set()
        for path in ordered:
            if not path:
                continue
            expanded = os.path.expanduser(path)
            if expanded in seen:
                continue
            seen.add(expanded)
            if os.path.isfile(expanded):
                return expanded
        return None

    def _resolve_chembl_mapping_path(self) -> Optional[str]:
        """Resolve CHEMBL->UniProt mapping file path."""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidates = [
            os.path.join(project_root, "dataset", "chembl_uniprot_mapping.txt"),
            os.path.join(project_root, "data", "chembl_uniprot_mapping.txt"),
        ]
        return self._resolve_optional_file(
            configured_path=getattr(self.config, "chembl_uniprot_mapping_path", None),
            env_var="CHEMBL_UNIPROT_MAPPING_PATH",
            candidates=candidates,
        )

    def _resolve_protein_targets_path(self) -> Optional[str]:
        """Resolve Papyrus protein-target TSV path for sequence lookup."""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_path = getattr(self.config, "data_path", None)
        data_candidates = []
        if data_path:
            expanded = os.path.expanduser(data_path)
            if os.path.isfile(expanded):
                data_candidates.append(expanded)
            elif os.path.isdir(expanded):
                data_candidates.extend(
                    [
                        os.path.join(expanded, "05.5_combined_set_protein_targets.tsv"),
                        os.path.join(expanded, "papyrus", "05.5_combined_set_protein_targets.tsv"),
                    ]
                )
        candidates = data_candidates + [
            os.path.join(project_root, "dataset", "papyrus", "05.5_combined_set_protein_targets.tsv"),
            os.path.join(project_root, "data", "papyrus", "05.5_combined_set_protein_targets.tsv"),
        ]
        return self._resolve_optional_file(
            configured_path=getattr(self.config, "protein_targets_path", None),
            env_var="PROTEIN_TARGETS_PATH",
            candidates=candidates,
        )

    def _load_chembl_to_uniprot_map(self) -> Dict[str, str]:
        """Load CHEMBL target id -> UniProt accession mapping."""
        if self._chembl_to_uniprot is not None:
            return self._chembl_to_uniprot

        mapping_path = self._resolve_chembl_mapping_path()
        mapping: Dict[str, str] = {}
        if not mapping_path:
            self.logger.warning(
                "No CHEMBL mapping file found. Set --chembl_uniprot_mapping_path "
                "or CHEMBL_UNIPROT_MAPPING_PATH to enable Target_CHEMBL_ID resolution."
            )
            self._chembl_to_uniprot = mapping
            return mapping

        try:
            with open(mapping_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    row = line.strip()
                    if not row or row.startswith("#"):
                        continue
                    cols = row.split("\t")
                    if len(cols) < 2:
                        continue
                    uniprot = cols[0].strip()
                    chembl = cols[1].strip()
                    if not uniprot or not chembl:
                        continue
                    mapping[chembl.upper()] = uniprot
            self.logger.info("Loaded %s CHEMBL->UniProt mappings from %s", len(mapping), mapping_path)
        except Exception as exc:
            self.logger.warning("Failed to parse CHEMBL mapping file %s: %s", mapping_path, exc)
            mapping = {}

        self._chembl_to_uniprot = mapping
        return mapping

    def _load_target_sequence_lookup(self) -> Dict[str, str]:
        """Load target_id -> sequence lookup from Papyrus protein target file."""
        if self._target_id_to_sequence is not None:
            return self._target_id_to_sequence

        targets_path = self._resolve_protein_targets_path()
        lookup: Dict[str, str] = {}
        if not targets_path:
            self.logger.warning(
                "No protein target file found. Set --protein_targets_path (or data_path) "
                "to resolve Target_CHEMBL_ID / UniProt_ID to Target_FASTA."
            )
            self._target_id_to_sequence = lookup
            return lookup

        try:
            sep = "\t" if targets_path.endswith(".tsv") else ","
            targets_df = pd.read_csv(targets_path, sep=sep)
            if "target_id" not in targets_df.columns or "Sequence" not in targets_df.columns:
                self.logger.warning(
                    "Protein target file %s missing required columns 'target_id' and 'Sequence'.",
                    targets_path,
                )
                self._target_id_to_sequence = lookup
                return lookup

            ids = targets_df["target_id"].astype(str)
            seqs = targets_df["Sequence"].astype(str)
            for target_id, seq in zip(ids, seqs):
                clean_id = target_id.strip()
                clean_seq = seq.strip()
                if clean_id and clean_seq and clean_seq.lower() != "nan":
                    lookup[clean_id] = clean_seq
            self.logger.info("Loaded %s protein target sequences from %s", len(lookup), targets_path)
        except Exception as exc:
            self.logger.warning("Failed to parse protein targets file %s: %s", targets_path, exc)
            lookup = {}

        self._target_id_to_sequence = lookup
        return lookup

    def _sequence_from_uniprot(self, uniprot_id: str) -> Optional[str]:
        """Resolve sequence from UniProt accession using target lookup file."""
        if not isinstance(uniprot_id, str):
            return None
        token = uniprot_id.strip()
        if not token:
            return None

        lookup = self._load_target_sequence_lookup()
        if not lookup:
            return None

        # Papyrus target IDs are commonly stored as "<UniProt>_WT".
        candidates = [token]
        if token.endswith("_WT"):
            candidates.append(token[:-3])
        else:
            candidates.append(f"{token}_WT")

        for candidate in candidates:
            seq = lookup.get(candidate)
            if seq:
                return seq
        return None

    def _fill_target_fasta_from_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Populate missing Target_FASTA from UniProt_ID / Target_CHEMBL_ID columns."""
        if "Target_FASTA" not in df.columns:
            df["Target_FASTA"] = pd.NA

        missing_mask = df["Target_FASTA"].isna() | (df["Target_FASTA"].astype(str).str.strip() == "")
        if not missing_mask.any():
            return df

        resolved = 0
        total_missing = int(missing_mask.sum())

        # 1) Prefer direct UniProt_ID when present.
        if "UniProt_ID" in df.columns:
            uni_values = df.loc[missing_mask, "UniProt_ID"].astype(str)
            uni_values = uni_values[uni_values.str.strip() != ""]
            uni_map = {uid: self._sequence_from_uniprot(uid) for uid in uni_values.unique()}
            uni_resolved = df.loc[missing_mask, "UniProt_ID"].map(uni_map)
            has_seq = uni_resolved.notna()
            fill_idx = uni_resolved.index[has_seq]
            df.loc[fill_idx, "Target_FASTA"] = uni_resolved[has_seq].values
            resolved += int(has_seq.sum())

        # Refresh missing mask after UniProt resolution.
        missing_mask = df["Target_FASTA"].isna() | (df["Target_FASTA"].astype(str).str.strip() == "")

        # 2) Resolve Target_CHEMBL_ID -> UniProt -> sequence.
        if missing_mask.any() and "Target_CHEMBL_ID" in df.columns:
            chembl_map = self._load_chembl_to_uniprot_map()
            if chembl_map:
                chembl_values = df.loc[missing_mask, "Target_CHEMBL_ID"].astype(str).str.upper()
                chembl_values = chembl_values[chembl_values.str.strip() != ""]
                seq_map = {}
                for chembl_id in chembl_values.unique():
                    uniprot = chembl_map.get(chembl_id)
                    seq_map[chembl_id] = self._sequence_from_uniprot(uniprot) if uniprot else None
                chembl_resolved = df.loc[missing_mask, "Target_CHEMBL_ID"].astype(str).str.upper().map(seq_map)
                has_seq = chembl_resolved.notna()
                fill_idx = chembl_resolved.index[has_seq]
                df.loc[fill_idx, "Target_FASTA"] = chembl_resolved[has_seq].values
                resolved += int(has_seq.sum())

        # 3) Legacy fallback: directory-based lookup by Target_CHEMBL_ID.
        missing_mask = df["Target_FASTA"].isna() | (df["Target_FASTA"].astype(str).str.strip() == "")
        if missing_mask.any() and "Target_CHEMBL_ID" in df.columns:
            legacy_ids = df.loc[missing_mask, "Target_CHEMBL_ID"].astype(str).unique()
            legacy_map = {cid: self._get_sequence_for_id(cid) for cid in legacy_ids}
            legacy_resolved = df.loc[missing_mask, "Target_CHEMBL_ID"].astype(str).map(legacy_map)
            has_seq = legacy_resolved.notna()
            fill_idx = legacy_resolved.index[has_seq]
            df.loc[fill_idx, "Target_FASTA"] = legacy_resolved[has_seq].values
            resolved += int(has_seq.sum())

        final_missing = int((df["Target_FASTA"].isna() | (df["Target_FASTA"].astype(str).str.strip() == "")).sum())
        self.logger.info(
            "Target sequence resolution: resolved=%s missing=%s (initial missing=%s)",
            resolved,
            final_missing,
            total_missing,
        )
        return df

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Return first matching column using exact then case-insensitive matching."""
        lower_to_original = {str(col).lower(): col for col in df.columns}
        for name in candidates:
            if name in df.columns:
                return name
            alt = lower_to_original.get(name.lower())
            if alt is not None:
                return alt
        return None

    def _get_sequence_for_id(self, target_id: str) -> Optional[str]:
        """
        Legacy lookup path: use data_path directory files keyed by Target_CHEMBL_ID.
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
        return tokenize_protein_sequences_for_inference(
            sequences=sequences,
            prot_tokenizer=self.prot_tokenizer,
            prot_emb_model=self.config.prot_emb_model,
            prot_max_length=self.config.prot_max_length,
            device=self.device,
        )

    def _load_eval_split_dataset(self, split_mode: str, split_ratio: float, split_seed: int):
        """Load dataset and return the validation split used for reproduction/evaluation."""
        cache_dir = os.environ.get('DATASETS_CACHE_DIR', "/gpfs/projects/etur29/atabey/datasets")

        try:
            dataset, processed_data_path = load_processed_dataset(self.config.input_file, cache_dir=cache_dir)
            self.logger.info(f"Loading preprocessed dataset from {processed_data_path}")
        except FileNotFoundError:
            self.logger.warning("Preprocessed data not found in cache.")
            self.logger.warning("Falling back to loading raw CSV and tokenizing on-the-fly.")
            try:
                dataset = load_dataset("csv", data_files=self.config.input_file)
            except Exception as e:
                self.logger.error(f"Failed to load dataset: {e}")
                return None

        full_data = dataset["train"]
        _, eval_data = split_train_eval_dataset(
            full_data=full_data,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
            logger=self.logger,
        )
        return eval_data

    def _override_normalization_from_input_file(self):
        """Update normalization using pChEMBL values from the full input file."""
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

    def _sanitize_output_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop internal preprocessing columns from exported CSV outputs."""
        drop_columns = [
            "Target_FASTA",
            "prot_input_ids",
            "prot_attention_mask",
            "mol_input_ids",
            "mol_attention_mask",
            "labels",
            "train_lm",
            "train_m",  # Backward-compatible typo guard
        ]
        existing = [col for col in drop_columns if col in df.columns]
        if not existing:
            return df
        self.logger.info("Dropping internal output columns: %s", ", ".join(existing))
        return df.drop(columns=existing)

    def _summarize_prediction_metrics(self, y_true, y_pred) -> Dict[str, float]:
        metrics = compute_pchembl_metrics(
            pchembl_predictions=np.asarray(y_pred, dtype=np.float32),
            pchembl_targets=np.asarray(y_true, dtype=np.float32),
            pchembl_mean=float(self.config.pchembl_mean),
            pchembl_std=float(self.config.pchembl_std),
            inputs_are_normalized=False,
            logger=self.logger,
        )
        summary = {
            "MSE": float(metrics["pchembl_mse_raw"]),
            "RMSE": float(metrics["pchembl_rmse_raw"]),
            "MAE": float(metrics["pchembl_mae_raw"]),
            "R2": float(metrics["pchembl_r2"]),
            "Count": int(metrics["pchembl_valid_count"]),
        }
        if "pchembl_pearson_raw" in metrics:
            summary["Pearson_r"] = float(metrics["pchembl_pearson_raw"])
        if "pchembl_spearman_raw" in metrics:
            summary["Spearman_rho"] = float(metrics["pchembl_spearman_raw"])
        return summary

    def _log_metric_summary(self, metrics: Dict[str, float], include_significance: bool = False, y_true=None, y_pred=None):
        self.logger.info("-" * 40)
        self.logger.info("Evaluation Metrics:")
        for key in ["MSE", "RMSE", "MAE", "R2", "Pearson_r", "Spearman_rho", "Count"]:
            if key not in metrics:
                continue
            value = metrics[key]
            if key == "Count":
                self.logger.info("%s: %s", key, value)
            else:
                self.logger.info("%s: %.4f", key, value)
        if include_significance and y_true is not None and y_pred is not None:
            pearson_corr, pearson_pval = pearsonr(y_true, y_pred)
            spearman_corr, spearman_pval = spearmanr(y_true, y_pred)
            metrics["Pearson_pval"] = float(pearson_pval)
            metrics["Spearman_pval"] = float(spearman_pval)
            self.logger.info("Pearson p-value: %.4e", pearson_pval)
            self.logger.info("Spearman p-value: %.4e", spearman_pval)
        self.logger.info("-" * 40)

    def _run_split_evaluation(self, split_mode: str, split_ratio: float, split_seed: int):
        """Run prediction/evaluation on a deterministic validation split."""
        eval_start = time.perf_counter()
        if not (0.0 < split_ratio < 1.0):
            raise ValueError(f"reproduce_split_ratio must be in (0, 1), got {split_ratio}")

        self.logger.info(f"Loading dataset for split evaluation from: {self.config.input_file}")
        self.logger.info(
            "Using split strategy: mode=%s ratio=%.3f seed=%s",
            split_mode,
            split_ratio,
            split_seed,
        )
        test_data = self._load_eval_split_dataset(
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )
        if test_data is None:
            return

        self.logger.info(f"Evaluation set size: {len(test_data)}")
        if len(test_data) == 0:
            self.logger.error("Validation split is empty. Cannot evaluate.")
            return

        self.logger.info(
            "Using checkpoint normalization for reproduction: mean=%.4f std=%.4f",
            self.config.pchembl_mean,
            self.config.pchembl_std,
        )

        # Check if we're using preprocessed data
        has_preprocessed_tokens = hasattr(test_data, "column_names") and 'prot_input_ids' in test_data.column_names

        if has_preprocessed_tokens:
            self.logger.info("Using preprocessed tokens (same as eval_pchembl.py)...")
            results_df = self._predict_from_preprocessed(test_data)
        else:
            self.logger.info("Using on-the-fly tokenization (may give different results)...")
            # We need to ensure we can predict for these.
            # They should have Target_FASTA and SMILES/SELFIES.
            results_df = self._predict_dataframe(test_data.to_pandas())
        
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
        metrics = self._summarize_prediction_metrics(y_true, y_pred)
        self._log_metric_summary(metrics, include_significance=True, y_true=y_true, y_pred=y_pred)
        metrics_file = self.config.output_file.replace('.csv', '_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        # Plotting
        self._create_plots(y_true, y_pred, metrics)
        
        # Save csv
        output_df = self._sanitize_output_dataframe(results_df)
        self.logger.info(f"Saving evaluation results to {self.config.output_file}")
        output_df.to_csv(self.config.output_file, index=False)
        self.logger.info("Split evaluation finished in %.2fs", time.perf_counter() - eval_start)

    def reproduce(self, split_mode: str):
        """Run prediction pipeline on validation split for result reproduction."""
        split_ratio = float(getattr(self.config, "reproduce_split_ratio", 0.01))
        split_seed = int(getattr(self.config, "reproduce_split_seed", 42))
        self._run_split_evaluation(
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )

    def _create_plots(self, y_true, y_pred, metrics: Optional[Dict[str, float]] = None):
        """Create extended distribution plots similar to eval_pchembl.py"""
        out_base = os.path.splitext(self.config.output_file)[0]
        
        if metrics is None:
            metrics = self._summarize_prediction_metrics(y_true, y_pred)
        mse = metrics["MSE"]
        mae = metrics["MAE"]
        rmse = metrics["RMSE"]
        r2 = metrics["R2"]
        pearson_corr = metrics.get("Pearson_r", float("nan"))
        spearman_corr = metrics.get("Spearman_rho", float("nan"))
        pearson_pval = metrics.get("Pearson_pval", float("nan"))
        spearman_pval = metrics.get("Spearman_pval", float("nan"))

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
        prediction_start = time.perf_counter()
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

                # Forward pass for pChEMBL prediction only.
                outputs = self.model(
                    prot_input_ids=prot_input_ids,
                    prot_attention_mask=prot_attention_mask,
                    mol_input_ids=mol_input_ids,
                    train_lm=False,
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
        self.logger.info("Preprocessed prediction finished in %.2fs", time.perf_counter() - prediction_start)

        return results_df

    def _predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper to run prediction on a dataframe"""
        prediction_start = time.perf_counter()
        # Normalize expected target column names (supports lowercase variants).
        target_fasta_col = self._find_column(df, ["Target_FASTA"])
        uniprot_col = self._find_column(df, ["UniProt_ID", "uniprot_id"])
        chembl_col = self._find_column(df, ["Target_CHEMBL_ID", "target_chembl_id"])
        rename_map = {}
        if target_fasta_col and target_fasta_col != "Target_FASTA":
            rename_map[target_fasta_col] = "Target_FASTA"
        if uniprot_col and uniprot_col != "UniProt_ID":
            rename_map[uniprot_col] = "UniProt_ID"
        if chembl_col and chembl_col != "Target_CHEMBL_ID":
            rename_map[chembl_col] = "Target_CHEMBL_ID"
        if rename_map:
            df = df.rename(columns=rename_map)

        mol_col, is_selfies = find_molecule_column(df.columns)
        if not mol_col:
            raise ValueError("Input file must contain a SMILES or SELFIES column")

        if (
            "Target_FASTA" not in df.columns
            and "UniProt_ID" not in df.columns
            and "Target_CHEMBL_ID" not in df.columns
        ):
            raise ValueError(
                "Input file must contain target context via one of: "
                "'Target_FASTA', 'UniProt_ID', or 'Target_CHEMBL_ID'."
            )
            
        # Ensure we have target sequences. If missing, resolve from UniProt_ID / Target_CHEMBL_ID.
        has_missing_fasta = (
            "Target_FASTA" not in df.columns
            or (df["Target_FASTA"].isna() | (df["Target_FASTA"].astype(str).str.strip() == "")).any()
        )
        if has_missing_fasta:
            if "Target_FASTA" not in df.columns:
                self.logger.info(
                    "'Target_FASTA' missing; attempting resolution via 'UniProt_ID' and/or 'Target_CHEMBL_ID'."
                )
            df = self._fill_target_fasta_from_ids(df.copy())
            df = df.dropna(subset=["Target_FASTA"])

        if len(df) == 0:
            return df

        # Prepare Molecules
        if not is_selfies:
            self.logger.info("Converting SMILES to SELFIES...")
        selfies_list = to_selfies_list(
            molecules=df[mol_col].tolist(),
            is_selfies=is_selfies,
            invalid_token="[nop]",
        )

        df = df.copy()
        df["_selfies_input"] = selfies_list
        predictions = np.zeros(len(df), dtype=np.float32)
        batch_size = self.config.batch_size

        self.logger.info(f"Starting prediction on {len(df)} samples...")
        grouped = df.groupby("Target_FASTA", sort=False).indices
        for sequence, row_indices in tqdm(grouped.items(), desc="Predicting Targets"):
            prot_ids, prot_mask = tokenize_protein_sequences_for_inference(
                sequences=[sequence],
                prot_tokenizer=self.prot_tokenizer,
                prot_emb_model=self.config.prot_emb_model,
                prot_max_length=self.config.prot_max_length,
                device=self.device,
            )
            with torch.no_grad():
                protein_embeddings = self.model.encode_protein(prot_ids, prot_mask)

            row_indices = list(row_indices)
            for start in range(0, len(row_indices), batch_size):
                batch_indices = row_indices[start : start + batch_size]
                batch_selfies = df.iloc[batch_indices]["_selfies_input"].tolist()
                mol_ids, _ = tokenize_selfies_for_inference(
                    selfies_list=batch_selfies,
                    mol_tokenizer=self.mol_tokenizer,
                    max_mol_len=self.config.max_mol_len,
                    device=self.device,
                )
                with torch.no_grad():
                    preds = self.model.predict_pchembl_from_protein_embeddings(
                        mol_input_ids=mol_ids,
                        protein_embeddings=protein_embeddings.repeat(len(batch_indices), 1, 1),
                        prot_attention_mask=prot_mask.repeat(len(batch_indices), 1),
                    ).cpu().numpy()
                predictions[batch_indices] = preds * self.config.pchembl_std + self.config.pchembl_mean

        df['Predicted_pChEMBL'] = predictions.tolist()
        df = df.drop(columns=["_selfies_input"])
        self.logger.info("Dataframe prediction finished in %.2fs", time.perf_counter() - prediction_start)
        return df

    def predict(self):
        self.logger.info(f"Loading input file: {self.config.input_file}")
        df = pd.read_csv(self.config.input_file)
        
        result_df = self._predict_dataframe(df)
        output_df = self._sanitize_output_dataframe(result_df)
        metrics = None
        if "pchembl_value_Median" in result_df.columns:
            eval_df = result_df.dropna(subset=["pchembl_value_Median", "Predicted_pChEMBL"])
            if len(eval_df) > 0:
                metrics = self._summarize_prediction_metrics(
                    eval_df["pchembl_value_Median"].values,
                    eval_df["Predicted_pChEMBL"].values,
                )
                self._log_metric_summary(
                    metrics,
                    include_significance=True,
                    y_true=eval_df["pchembl_value_Median"].values,
                    y_pred=eval_df["Predicted_pChEMBL"].values,
                )
                metrics_path = self.config.output_file.replace(".csv", "_metrics.json")
                with open(metrics_path, "w") as handle:
                    json.dump(metrics, handle, indent=2)
                self.logger.info("Saved prediction metrics to %s", metrics_path)
                self._create_plots(
                    eval_df["pchembl_value_Median"].values,
                    eval_df["Predicted_pChEMBL"].values,
                    metrics,
                )
        
        # Save
        self.logger.info(f"Saving results to {self.config.output_file}")
        output_df.to_csv(self.config.output_file, index=False)
        self.logger.info("Done.")

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Predict pChEMBL values")
    
    parser.add_argument("--input_file", required=True, help="Input CSV with molecules and targets")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--output_file", required=True, help="Output CSV path")
    parser.add_argument("--models_base", type=str, default=None, help="Base directory containing local HF model caches (models--*).")
    parser.add_argument(
        "--data_path",
        help="Legacy dataset directory or file path used for sequence lookup fallback.",
    )
    parser.add_argument(
        "--chembl_uniprot_mapping_path",
        type=str,
        default=None,
        help="Path to CHEMBL->UniProt mapping file (e.g., chembl_uniprot_mapping.txt).",
    )
    parser.add_argument(
        "--protein_targets_path",
        type=str,
        default=None,
        help="Path to Papyrus protein target file (e.g., 05.5_combined_set_protein_targets.tsv).",
    )
    
    # Model Params
    parser.add_argument("--prot_emb_model", default="prot_t5", choices=["prot_t5", "esm2", "saprot"])
    parser.add_argument("--n_layer", type=int, default=1)
    parser.add_argument("--n_head", type=int, default=16)
    parser.add_argument("--n_emb", type=int, default=1024)
    parser.add_argument("--prot_max_length", type=int, default=1024)
    parser.add_argument("--max_mol_len", type=int, default=200)
    parser.add_argument("--pchembl_tf_hidden_dim", type=int, default=768)
    parser.add_argument("--pchembl_tf_num_heads", type=int, default=8)
    parser.add_argument("--pchembl_tf_group_size", type=int, default=1)
    parser.add_argument("--pchembl_tf_agg_mode", type=str, choices=["cls", "mean", "mean_all_tok"], default="mean")
    parser.add_argument("--pchembl_tf_dropout", type=float, default=0.1)
    
    # Prediction Params
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pchembl_mean", type=float, default=5.924)
    parser.add_argument("--pchembl_std", type=float, default=1.362)
    
    parser.add_argument(
        "--reproduce_split_ratio",
        type=float,
        default=0.01,
        help="Fraction of data (or AIDs) held out for validation reproduction",
    )
    parser.add_argument(
        "--reproduce_split_seed",
        type=int,
        default=42,
        help="Random seed for split reproducibility",
    )
    parser.add_argument(
        "--reproduce",
        choices=["random", "aid"],
        default=None,
        help="Load validation split from input file and run prediction pipeline.",
    )

    return parse_args_with_config(parser, section="predict", argv=argv)

def main():
    args = parse_args()
    predictor = PChemblPredictor(args)
    if args.reproduce:
        predictor.reproduce(args.reproduce)
    else:
        predictor.predict()

if __name__ == "__main__":
    main()
