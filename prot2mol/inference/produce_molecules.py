#!/usr/bin/env python3
"""
Molecule Generation Script for Prot2Mol

This script generates molecules for given protein targets using a trained Prot2Mol model.
The script supports both single protein target generation and batch processing.
"""

import os
import sys
import json
import logging
import argparse
import warnings
import time
from typing import List, Dict, Optional, Tuple

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import GenerationConfig

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from prot2mol.core.protein_encoders import get_protein_tokenizer
from prot2mol.io.config import parse_args_with_config
from prot2mol.io.hf_utils import load_molgen_tokenizer, load_prot2mol_inference_model, load_saved_model_config
from prot2mol.data.pipeline import (
    find_molecule_column,
    has_matching_precomputed_split,
    load_processed_dataset,
    load_processed_stats,
    split_train_eval_dataset,
    to_selfies_list,
    tokenize_protein_sequences_for_inference,
    tokenize_selfies_for_inference,
)
from prot2mol.chem.utils import metrics_calculation, canonicalize_smiles_list, decode_selfies_list
import selfies as sf
from rdkit import RDLogger

# Suppress warnings and logs
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
sns.set_theme(style="whitegrid")

class MoleculeGenerator:
    """
    Unified molecule generator using the new Prot2MolModel architecture.
    
    This class handles model loading, data processing, and molecule generation
    for protein-to-molecule generation tasks.
    """
    
    def __init__(self, config: argparse.Namespace):
        """
        Initialize the molecule generator.
        
        Args:
            config: Configuration object containing all parameters
        """
        self.config = config
        self.logger = self._setup_logging()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Log GPU info if available
        if torch.cuda.is_available():
            self.logger.info(f"🚀 GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            self.logger.warning("⚠️ CUDA not available, using CPU")
        
        # Initialize components
        self.mol_tokenizer = None
        self.prot_tokenizer = None
        self.generation_model = None
        self.prediction_model = None
        self.train_data = None
        self.train_vec = None
        self.saved_model_config: Dict[str, object] = {}
        self.sequence_cache = {}
        self._chembl_to_uniprot: Optional[Dict[str, str]] = None
        self._target_id_to_sequence: Optional[Dict[str, str]] = None
        self._normalize_mode()
        
        # Load model and tokenizers
        self._load_components()

    def _normalize_mode(self):
        """Normalize the generation/prediction mode after YAML/CLI merging."""
        raw_mode = getattr(self.config, "mode", None)
        mode = str(raw_mode).strip().lower() if raw_mode is not None else ""
        if not mode:
            mode = "generation"
        if mode not in {"generation", "prediction"}:
            raise ValueError(f"Unsupported mode: {raw_mode!r}. Expected 'generation' or 'prediction'.")
        self.config.mode = mode
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            force=True,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f'molecule_generation_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
        logger = logging.getLogger(__name__)
        logger.propagate = False
        return logger

    def _apply_saved_config_overrides(self, model_path: str):
        saved_config = load_saved_model_config(model_path, logger=self.logger)
        if saved_config:
            self.saved_model_config.update(saved_config)
        for key, value in saved_config.items():
            if not hasattr(self.config, key):
                continue
            current_value = getattr(self.config, key)
            if current_value != value:
                self.logger.info("Overriding %s=%s with checkpoint value %s", key, current_value, value)
                setattr(self.config, key, value)

    def _resolve_optional_file(
        self,
        configured_path: Optional[str],
        env_var: Optional[str],
        candidates: List[str],
    ) -> Optional[str]:
        """Return the first existing file from explicit/env/default candidates."""
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
                    if uniprot and chembl:
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

    def _fill_target_fasta_from_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Populate missing Target_FASTA from UniProt_ID / Target_CHEMBL_ID columns."""
        if "Target_FASTA" not in df.columns:
            df["Target_FASTA"] = pd.NA

        missing_mask = df["Target_FASTA"].isna() | (df["Target_FASTA"].astype(str).str.strip() == "")
        if not missing_mask.any():
            return df

        if "UniProt_ID" in df.columns:
            uni_values = df.loc[missing_mask, "UniProt_ID"].astype(str)
            uni_values = uni_values[uni_values.str.strip() != ""]
            uni_map = {uid: self._sequence_from_uniprot(uid) for uid in uni_values.unique()}
            uni_resolved = df.loc[missing_mask, "UniProt_ID"].map(uni_map)
            has_seq = uni_resolved.notna()
            fill_idx = uni_resolved.index[has_seq]
            df.loc[fill_idx, "Target_FASTA"] = uni_resolved[has_seq].values

        missing_mask = df["Target_FASTA"].isna() | (df["Target_FASTA"].astype(str).str.strip() == "")
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

        missing_mask = df["Target_FASTA"].isna() | (df["Target_FASTA"].astype(str).str.strip() == "")
        if missing_mask.any() and "Target_CHEMBL_ID" in df.columns:
            legacy_ids = df.loc[missing_mask, "Target_CHEMBL_ID"].astype(str).unique()
            legacy_map = {cid: self._get_sequence_for_id(cid) for cid in legacy_ids}
            legacy_resolved = df.loc[missing_mask, "Target_CHEMBL_ID"].astype(str).map(legacy_map)
            has_seq = legacy_resolved.notna()
            fill_idx = legacy_resolved.index[has_seq]
            df.loc[fill_idx, "Target_FASTA"] = legacy_resolved[has_seq].values

        return df

    def _resolve_reference_dataset_path(self) -> Optional[str]:
        """Return dataset CSV used for training-reference comparisons."""
        configured = getattr(self.config, "reference_dataset", None)
        if configured:
            expanded = os.path.expanduser(configured)
            if os.path.exists(expanded):
                return expanded
            raise FileNotFoundError(f"Reference dataset not found: {expanded}")

        saved_path = self.saved_model_config.get("dataset_source_path")
        if isinstance(saved_path, str) and saved_path.strip():
            expanded = os.path.expanduser(saved_path.strip())
            if os.path.exists(expanded):
                return expanded
            self.logger.warning("Checkpoint dataset_source_path does not exist: %s", expanded)
        return None

    def _resolve_split_settings(self) -> Tuple[str, float, int]:
        """Resolve train/eval split metadata from checkpoint configuration."""
        split_mode = self.saved_model_config.get("eval_split")
        split_ratio = self.saved_model_config.get("eval_split_ratio")
        split_seed = self.saved_model_config.get("split_seed")
        if split_mode not in {"random", "aid"} or split_ratio is None or split_seed is None:
            raise ValueError(
                "Checkpoint does not contain eval split metadata. "
                "Set --reference_dataset only if the checkpoint also saved eval split metadata."
            )
        return str(split_mode), float(split_ratio), int(split_seed)

    def _load_requested_split_dataset(
        self,
        dataset_path: str,
        split_mode: str,
        split_ratio: float,
        split_seed: int,
    ):
        """Load a dataset path and reconstruct the requested train/eval split."""
        cache_dir = os.environ.get("DATASETS_CACHE_DIR", "/gpfs/projects/etur29/atabey/datasets")
        dataset = None
        processed_data_path = None

        try:
            dataset, processed_data_path = load_processed_dataset(dataset_path, cache_dir=cache_dir)
            self.logger.info("Loading preprocessed dataset from %s", processed_data_path)
        except FileNotFoundError:
            self.logger.warning("Preprocessed data not found in cache for %s.", dataset_path)

        stats = load_processed_stats(dataset_path, cache_dir=cache_dir)
        if dataset is not None and has_matching_precomputed_split(
            dataset,
            stats,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
        ):
            self.logger.info("Using precomputed train/eval splits from cache.")
            return dataset["train"], dataset["test"]

        use_cached_full_dataset = dataset is not None and hasattr(dataset, "keys") and "train" in dataset
        if dataset is not None and stats is not None and hasattr(dataset, "keys") and "test" in dataset:
            self.logger.warning(
                "Cached split configuration at %s does not match the requested settings "
                "(cached: mode=%s ratio=%s seed=%s; requested: mode=%s ratio=%s seed=%s). "
                "Falling back to raw CSV split reconstruction.",
                processed_data_path,
                stats.get("eval_split"),
                stats.get("eval_split_ratio"),
                stats.get("split_seed"),
                split_mode,
                split_ratio,
                split_seed,
            )
            use_cached_full_dataset = False

        if use_cached_full_dataset:
            self.logger.info("Using cached full dataset to reconstruct requested split in-memory.")
            full_data = dataset["train"]
        else:
            self.logger.info("Loading raw CSV dataset from %s for split reconstruction", dataset_path)
            from datasets import load_dataset

            try:
                dataset = load_dataset("csv", data_files=dataset_path)
            except Exception as exc:
                self.logger.error("Failed to load dataset from %s: %s", dataset_path, exc)
                return None, None
            full_data = dataset["train"]

        train_data, eval_data = split_train_eval_dataset(
            full_data=full_data,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
            logger=self.logger,
        )
        return train_data, eval_data

    def _load_train_reference_rows(self, protein_id: str) -> pd.DataFrame:
        """Load real training rows for a specific protein from the training dataset split."""
        dataset_path = self._resolve_reference_dataset_path()
        if dataset_path is None:
            raise ValueError(
                "No reference dataset is available for protein comparison. "
                "Set --reference_dataset or use a checkpoint that saved dataset_source_path."
            )

        split_mode, split_ratio, split_seed = self._resolve_split_settings()
        train_data, _ = self._load_requested_split_dataset(
            dataset_path=dataset_path,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )
        if train_data is None:
            raise ValueError(f"Failed to load train split from reference dataset: {dataset_path}")

        train_df = train_data.to_pandas() if hasattr(train_data, "to_pandas") else pd.DataFrame(train_data)
        token = str(protein_id).strip()
        if not token:
            return train_df.iloc[0:0].copy()

        candidate_columns = ["Target_CHEMBL_ID", "UniProt_ID", "Target_ID", "Protein_ID"]
        for column_name in candidate_columns:
            actual = self._find_column(train_df, [column_name])
            if actual is None:
                continue
            values = train_df[actual].astype(str).str.strip()
            mask = values.str.casefold() == token.casefold()
            if mask.any():
                return train_df.loc[mask].copy()

        if "Target_FASTA" in train_df.columns:
            values = train_df["Target_FASTA"].astype(str).str.strip()
            mask = values == token
            if mask.any():
                return train_df.loc[mask].copy()

        return train_df.iloc[0:0].copy()

    def _save_distribution_plot(
        self,
        batch_predictions: List[float],
        output_path: str,
        title: str,
        batch_label: str,
        reference_values: Optional[List[float]] = None,
        reference_label: Optional[str] = None,
    ) -> Optional[str]:
        """Save a predicted pChEMBL distribution plot, optionally with a reference distribution."""
        batch_values = np.asarray(batch_predictions, dtype=np.float32)
        batch_values = batch_values[np.isfinite(batch_values)]
        if batch_values.size == 0:
            self.logger.warning("Skipping distribution plot because there are no finite predictions.")
            return None

        ref_values = None
        if reference_values is not None:
            ref_values = np.asarray(reference_values, dtype=np.float32)
            ref_values = ref_values[np.isfinite(ref_values)]
            if ref_values.size == 0:
                ref_values = None

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        bins = max(10, min(int(getattr(self.config, "distribution_bins", 40)), max(10, int(np.sqrt(batch_values.size)) * 2)))
        common_kwargs = {"stat": "density", "bins": bins, "element": "step", "fill": True, "alpha": 0.35}

        sns.histplot(batch_values, color="#1f77b4", label=batch_label, ax=ax, **common_kwargs)
        if batch_values.size > 1 and np.unique(batch_values).size > 1:
            sns.kdeplot(batch_values, color="#1f77b4", ax=ax, linewidth=2)
        ax.axvline(batch_values.mean(), color="#1f77b4", linestyle="--", linewidth=1.5)

        subtitle = [f"{batch_label}: n={batch_values.size}, mean={batch_values.mean():.3f}, std={batch_values.std(ddof=0):.3f}"]
        if ref_values is not None and reference_label:
            sns.histplot(ref_values, color="#d62728", label=reference_label, ax=ax, **common_kwargs)
            if ref_values.size > 1 and np.unique(ref_values).size > 1:
                sns.kdeplot(ref_values, color="#d62728", ax=ax, linewidth=2)
            ax.axvline(ref_values.mean(), color="#d62728", linestyle="--", linewidth=1.5)
            subtitle.append(
                f"{reference_label}: n={ref_values.size}, mean={ref_values.mean():.3f}, std={ref_values.std(ddof=0):.3f}"
            )

        ax.set_title(title)
        ax.set_xlabel("pChEMBL")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.text(0.5, 0.01, " | ".join(subtitle), ha="center", fontsize=10)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        self.logger.info("Saved distribution plot to %s", output_path)
        return output_path

    def _save_prediction_plots(self, results_df: pd.DataFrame):
        """Save predicted pChEMBL distribution plots for generation or scoring outputs."""
        if not getattr(self.config, "save_distribution_plot", True):
            return
        if "Predicted_pChEMBL" not in results_df.columns:
            return

        out_base = os.path.splitext(self.config.output_file)[0]
        self._save_distribution_plot(
            batch_predictions=results_df["Predicted_pChEMBL"].values.tolist(),
            output_path=f"{out_base}_predicted_distribution.png",
            title="Predicted pChEMBL Distribution",
            batch_label="Batch predicted pChEMBL",
        )

        compare_protein_id = getattr(self.config, "compare_protein_id", None) or getattr(self.config, "prot_id", None)
        if not compare_protein_id:
            return

        try:
            reference_df = self._load_train_reference_rows(compare_protein_id)
            if len(reference_df) == 0 or "pchembl_value_Median" not in reference_df.columns:
                self.logger.warning(
                    "No usable training reference rows found for protein %s. Skipping comparison plot.",
                    compare_protein_id,
                )
                return

            safe_protein = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(compare_protein_id)).strip("._-") or "value"
            self._save_distribution_plot(
                batch_predictions=results_df["Predicted_pChEMBL"].values.tolist(),
                output_path=f"{out_base}_comparison_{safe_protein}.png",
                title=f"Batch Predictions vs Training Distribution for {compare_protein_id}",
                batch_label="Batch predicted pChEMBL",
                reference_values=reference_df["pchembl_value_Median"].values.tolist(),
                reference_label=f"Train real pChEMBL ({compare_protein_id})",
            )
        except Exception as exc:
            self.logger.warning("Failed to build protein comparison plot for %s: %s", compare_protein_id, exc)
    
    def _load_components(self):
        """Load tokenizers and model."""
        self.logger.info("Loading tokenizers and model...")
        self._apply_saved_config_overrides(self.config.model_file)
        if self.config.prediction_model_file:
            self._apply_saved_config_overrides(self.config.prediction_model_file)
        
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
                "to the folder containing 'models--zjunlp--MolGen-large'."
            )
            raise
        
        # Load protein tokenizer
        self.logger.info("Loading protein tokenizer...")
        self.prot_tokenizer = get_protein_tokenizer(self.config.prot_emb_model)
        
        # Load the generation model
        self.logger.info(f"Loading generation model from {self.config.model_file}")
        self.generation_model = self._load_single_model(self.config.model_file)
        self.generation_model.eval()
        
        # Load the prediction model
        if self.config.prediction_model_file:
            self.logger.info(f"Loading prediction model from {self.config.prediction_model_file}")
            self.prediction_model = self._load_single_model(self.config.prediction_model_file)
            self.prediction_model.eval()
        else:
            self.logger.info("No separate prediction model specified, using generation model for prediction")
            self.prediction_model = self.generation_model

        # Verify model is on correct device
        model_device = next(self.generation_model.parameters()).device
        self.logger.info(f"✅ Models loaded successfully on {model_device}")
        

        if torch.cuda.is_available() and model_device.type == 'cuda':
            self.logger.info(f"💾 GPU Memory after model loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        # Set generation config only if in generation mode
        if self.config.mode == "generation":
             self.generation_config = GenerationConfig(
                max_length=200,
                do_sample=True,
                pad_token_id=1,
                bos_token_id=1,
                eos_token_id=self.mol_tokenizer.eos_token_id,
                temperature=1.0,
                top_p=0.9
            )        
    def _load_single_model(self, model_path: str):
        """Helper to load a single Prot2Mol model instance."""
        with tqdm(desc=f"Loading model weights from {os.path.basename(model_path)}", unit="model") as pbar:
            model = load_prot2mol_inference_model(
                model_path=model_path,
                device=self.device,
                mol_tokenizer=self.mol_tokenizer,
                prot_emb_model=self.config.prot_emb_model,
                n_layer=getattr(self.config, 'n_layer', 1),
                n_head=getattr(self.config, 'n_head', 16),
                n_emb=getattr(self.config, 'n_emb', 1024),
                max_mol_len=getattr(self.config, 'max_mol_len', 256),
                prot_max_length=getattr(self.config, 'prot_max_length', 1024),
                pchembl_tf_hidden_dim=getattr(self.config, "pchembl_tf_hidden_dim", 768),
                pchembl_tf_num_heads=getattr(self.config, "pchembl_tf_num_heads", 8),
                pchembl_tf_group_size=getattr(self.config, "pchembl_tf_group_size", 1),
                pchembl_tf_agg_mode=getattr(self.config, "pchembl_tf_agg_mode", "mean"),
                pchembl_tf_dropout=getattr(self.config, "pchembl_tf_dropout", 0.1),
                strict=False,
                allow_strict_fallback=False,
                logger=self.logger,
            )
            pbar.update(1)
        return model
    
    def _load_prediction_data(self) -> pd.DataFrame:
        """
        Load data for prediction mode.
        """
        self.logger.info(f"Loading molecules for prediction from {self.config.input_molecules}")
        
        if not os.path.exists(self.config.input_molecules):
             raise FileNotFoundError(f"Input molecules file not found: {self.config.input_molecules}")
             
        df = pd.read_csv(self.config.input_molecules)
        molecule_column, _ = find_molecule_column(df.columns)
        if molecule_column is None:
            raise ValueError(
                "Input file must contain a column with SMILES or SELFIES "
                "(e.g., 'smiles', 'Compound_SMILES', 'selfies', 'Compound_SELFIES', 'Generated_SELFIES')"
            )
        
        self.logger.info(f"Loaded {len(df)} molecules for prediction")
        
        # Check if we need to load train data for metrics? No, prediction mode usually just predicts.
        # But if we wanted to compute novelty etc we would need it. For now assuming just pchembl prediction.
        
        return df

    def _load_dataset(self) -> Tuple[pd.DataFrame, Optional[np.ndarray], pd.DataFrame]:
        """
        Load and process the dataset for generation mode.
        
        Returns:
            Tuple of (train_data, train_vec, test_data)
        """
        self.logger.info("Loading dataset...")
        
        # Check if selfies_path is a file or directory
        if os.path.isfile(self.config.selfies_path):
            # Single file case - load and filter by protein ID
            self.logger.info(f"Loading data from single file: {self.config.selfies_path}")
            with tqdm(desc="Loading dataset", unit="rows") as pbar:
                all_data = pd.read_csv(self.config.selfies_path)
                pbar.update(len(all_data))
            
            # Validate required columns
            required_columns = ['Target_FASTA', 'Target_CHEMBL_ID', 'Compound_SELFIES']
            missing_columns = [col for col in required_columns if col not in all_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}. Available columns: {list(all_data.columns)}")
            
            # Filter for target protein
            self.logger.info(f"Filtering data for protein: {self.config.prot_id}")
            with tqdm(desc=f"Filtering for {self.config.prot_id}", unit="rows") as pbar:
                test_data = all_data[all_data['Target_CHEMBL_ID'] == self.config.prot_id].reset_index(drop=True)
                pbar.update(len(all_data))
            
            if len(test_data) == 0:
                # Show available protein IDs to help user
                available_proteins = all_data['Target_CHEMBL_ID'].unique()
                self.logger.error(f"No data found for protein {self.config.prot_id}")
                self.logger.info(f"Available protein IDs: {available_proteins[:10]}...")  # Show first 10
                raise ValueError(f"No data found for protein {self.config.prot_id}. Available proteins: {len(available_proteins)} total")
            
            # Use remaining data as training data for metrics calculation
            train_data = all_data[all_data['Target_CHEMBL_ID'] != self.config.prot_id].reset_index(drop=True)
            train_vec = None  # No pre-computed vectors for single file mode
            
            self.logger.info(f"Found {len(test_data)} samples for target protein {self.config.prot_id}")
            self.logger.info(f"Using {len(train_data)} samples from other proteins as training reference")
            
        else:
            # Directory case - use original logic
            self.logger.info(f"Loading data from directory: {self.config.selfies_path}")
            
            # Load training data for metrics calculation
            train_path = os.path.join(self.config.selfies_path, "train.csv")
            if os.path.exists(train_path):
                train_data = pd.read_csv(train_path)
                
                # Load training vectors if available
                train_vec_path = os.path.join(self.config.selfies_path, "train_vecs.npy")
                train_vec = np.load(train_vec_path) if os.path.exists(train_vec_path) else None
                
            else:
                self.logger.warning("Training data not found. Metrics calculation may be limited.")
                train_data = pd.DataFrame()
                train_vec = None
            
            # Load test data for the target protein
            test_path = os.path.join(self.config.selfies_path, f"test_{self.config.prot_id}.csv")
            if not os.path.exists(test_path):
                # Try alternative naming
                test_path = os.path.join(self.config.selfies_path, "test.csv")
                if os.path.exists(test_path):
                    test_data = pd.read_csv(test_path)
                    # Filter for target protein
                    if 'Target_CHEMBL_ID' in test_data.columns:
                        test_data = test_data[test_data['Target_CHEMBL_ID'] == self.config.prot_id].reset_index(drop=True)
                else:
                    raise FileNotFoundError(f"Test data not found for protein {self.config.prot_id}")
            else:
                test_data = pd.read_csv(test_path)
        
        # Keep tokenizer vocabulary fixed at inference time to match model training/load.
        # Runtime tokenizer growth can desynchronize vocab assumptions from checkpoint weights.
        self.logger.info("Using fixed MolGen tokenizer vocabulary (no runtime token additions).")
        
        self.logger.info(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples")
        
        return train_data, train_vec, test_data

    def _get_sequence_for_id(self, target_id: str) -> Optional[str]:
        """Legacy lookup path: inspect data_path/selfies_path files keyed by Target_CHEMBL_ID."""
        if target_id in self.sequence_cache:
            return self.sequence_cache[target_id]

        lookup_roots = []
        if getattr(self.config, "data_path", None):
            lookup_roots.append(self.config.data_path)
        if getattr(self.config, "selfies_path", None):
            lookup_roots.append(self.config.selfies_path)

        existing_roots = [path for path in lookup_roots if path and os.path.exists(path)]
        for root in existing_roots:
            if os.path.isfile(root):
                try:
                    df = pd.read_csv(root, usecols=["Target_CHEMBL_ID", "Target_FASTA"])
                    matched = df[df["Target_CHEMBL_ID"].astype(str) == str(target_id)]
                    if not matched.empty:
                        seq = str(matched["Target_FASTA"].iloc[0]).strip()
                        if seq:
                            self.sequence_cache[target_id] = seq
                            return seq
                except Exception:
                    continue
                continue

            test_file = os.path.join(root, f"test_{target_id}.csv")
            if os.path.exists(test_file):
                try:
                    df = pd.read_csv(test_file)
                    if "Target_FASTA" in df.columns:
                        seq = str(df["Target_FASTA"].iloc[0]).strip()
                        if seq:
                            self.sequence_cache[target_id] = seq
                            return seq
                except Exception as exc:
                    self.logger.warning("Error reading %s: %s", test_file, exc)

        return None
    
    def _get_protein_embeddings(self, protein_sequence: str) -> torch.Tensor:
        """
        Get protein embeddings for a given sequence.
        
        Args:
            protein_sequence: Protein sequence string
            
        Returns:
            Protein embeddings tensor
        """
        prot_input_ids, prot_attention_mask = tokenize_protein_sequences_for_inference(
            sequences=[protein_sequence],
            prot_tokenizer=self.prot_tokenizer,
            prot_emb_model=self.config.prot_emb_model,
            prot_max_length=self.config.prot_max_length,
            device=self.device,
        )
        return prot_input_ids, prot_attention_mask

    def _encode_protein_for_model(self, model, prot_input_ids: torch.Tensor, prot_attention_mask: torch.Tensor):
        with torch.no_grad():
            return model.encode_protein(prot_input_ids, prot_attention_mask)

    def _generate_molecules_batch_with_tokens(
        self,
        protein_embeddings: torch.Tensor,
        prot_attention_mask: torch.Tensor,
        num_samples: int,
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Generate molecules for a batch of protein sequences.
        
        Args:
            prot_input_ids: Tokenized protein sequences
            prot_attention_mask: Attention mask for protein sequences
            num_samples: Number of molecules to generate
            
        Returns:
            List of generated SELFIES strings
        """
        generated_molecules = []
        generated_token_batches = []
        
        # Generate in batches to manage memory
        batch_size = min(self.config.batch_size, num_samples)
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Generating molecules"):
                current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
                
                batch_protein_embeddings = protein_embeddings.repeat(current_batch_size, 1, 1)
                batch_prot_attention_mask = prot_attention_mask.repeat(current_batch_size, 1)
                
                # Generate molecules
                try:
                    generated_tokens = self.generation_model.generate_from_protein_embeddings(
                        protein_embeddings=batch_protein_embeddings,
                        prot_attention_mask=batch_prot_attention_mask,
                        num_return_sequences=1,
                        max_length=self.generation_config.max_length,
                        do_sample=self.generation_config.do_sample,
                        temperature=self.generation_config.temperature,
                        top_p=self.generation_config.top_p,
                        pad_token_id=self.generation_config.pad_token_id,
                        bos_token_id=self.generation_config.bos_token_id,
                        eos_token_id=self.generation_config.eos_token_id,
                        output_attentions=self.config.attn_output
                    )
                    
                    # Decode generated tokens
                    batch_selfies = [
                        self.mol_tokenizer.decode(tokens, skip_special_tokens=True)
                        for tokens in generated_tokens
                    ]
                    
                    generated_molecules.extend(batch_selfies)
                    generated_token_batches.append(generated_tokens.detach().cpu())
                    
                except Exception as e:
                    self.logger.error(f"Error generating batch {batch_idx}: {str(e)}")
                    # Add empty strings for failed generations
                    generated_molecules.extend([''] * current_batch_size)
                    fallback_tokens, _ = tokenize_selfies_for_inference(
                        selfies_list=["[nop]"] * current_batch_size,
                        mol_tokenizer=self.mol_tokenizer,
                        max_mol_len=self.config.max_mol_len,
                        device=torch.device("cpu"),
                    )
                    generated_token_batches.append(fallback_tokens.cpu())

        generated_token_ids = torch.cat(generated_token_batches, dim=0) if generated_token_batches else torch.empty(0)
        return generated_molecules, generated_token_ids

    def _generate_molecules_batch(
        self,
        protein_embeddings: torch.Tensor,
        prot_attention_mask: torch.Tensor,
        num_samples: int,
    ) -> List[str]:
        generated_molecules, _ = self._generate_molecules_batch_with_tokens(
            protein_embeddings=protein_embeddings,
            prot_attention_mask=prot_attention_mask,
            num_samples=num_samples,
        )
        return generated_molecules
    
    def _predict_pchembl_batch(
        self,
        protein_embeddings: torch.Tensor,
        prot_attention_mask: torch.Tensor,
        mol_input_ids: torch.Tensor,
        mol_attention_mask: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Predict pChEMBL values for a batch of molecules.
        """
        with torch.no_grad():
            predictions = self.prediction_model.predict_pchembl_from_protein_embeddings(
                mol_input_ids=mol_input_ids,
                protein_embeddings=protein_embeddings,
                prot_attention_mask=prot_attention_mask,
            ).cpu().numpy()
            
            # Denormalize
            denormalized_preds = predictions * self.config.pchembl_std + self.config.pchembl_mean
            return denormalized_preds

    def predict_pchembl(self, protein_sequence: str, molecules_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Predict pChEMBL values for a dataframe of molecules against one or more targets.

        Backward-compatible forms:
        - predict_pchembl("MKT...", df)
        - predict_pchembl(df)
        """
        if molecules_df is None:
            shared_sequence = None
            molecules_df = protein_sequence
        else:
            shared_sequence = protein_sequence

        self.logger.info(f"Predicting pChEMBL for {len(molecules_df)} molecules...")
        start_time = time.perf_counter()

        df = molecules_df.copy()
        target_fasta_col = self._find_column(df, ["Target_FASTA"])
        uniprot_col = self._find_column(df, ["UniProt_ID", "uniprot_id"])
        chembl_col = self._find_column(df, ["Target_CHEMBL_ID", "target_chembl_id", "Protein_ID", "protein_id"])
        rename_map = {}
        if target_fasta_col and target_fasta_col != "Target_FASTA":
            rename_map[target_fasta_col] = "Target_FASTA"
        if uniprot_col and uniprot_col != "UniProt_ID":
            rename_map[uniprot_col] = "UniProt_ID"
        if chembl_col and chembl_col != "Target_CHEMBL_ID":
            rename_map[chembl_col] = "Target_CHEMBL_ID"
        if rename_map:
            df = df.rename(columns=rename_map)

        forced_target_chembl_id = str(getattr(self.config, "target_chembl_id", "") or "").strip()
        if forced_target_chembl_id:
            forced_target_chembl_id = forced_target_chembl_id.upper()
            self.logger.info(
                "Using target_chembl_id=%s for all rows; ignoring any target sequence context in the input file.",
                forced_target_chembl_id,
            )
            df["Target_CHEMBL_ID"] = forced_target_chembl_id
            if "Target_FASTA" in df.columns:
                df["Target_FASTA"] = pd.NA
            if "UniProt_ID" in df.columns:
                df["UniProt_ID"] = pd.NA

        mol_col, is_selfies = find_molecule_column(df.columns)
        if not mol_col:
            raise ValueError("Could not find SMILES or SELFIES column")

        if shared_sequence:
            if "Target_FASTA" not in df.columns:
                df["Target_FASTA"] = shared_sequence
            else:
                missing_mask = df["Target_FASTA"].isna() | (df["Target_FASTA"].astype(str).str.strip() == "")
                if missing_mask.any():
                    df.loc[missing_mask, "Target_FASTA"] = shared_sequence
        elif (
            "Target_FASTA" not in df.columns
            and "UniProt_ID" not in df.columns
            and "Target_CHEMBL_ID" not in df.columns
            and self.config.prot_id
        ):
            df["Target_CHEMBL_ID"] = self.config.prot_id

        if (
            "Target_FASTA" not in df.columns
            and "UniProt_ID" not in df.columns
            and "Target_CHEMBL_ID" not in df.columns
        ):
            raise ValueError(
                "Prediction mode requires target context via one of: "
                "'Target_FASTA', 'UniProt_ID', 'Target_CHEMBL_ID', or --prot_id."
            )

        has_missing_fasta = (
            "Target_FASTA" not in df.columns
            or (df["Target_FASTA"].isna() | (df["Target_FASTA"].astype(str).str.strip() == "")).any()
        )
        if has_missing_fasta:
            df = self._fill_target_fasta_from_ids(df)
            if self.config.prot_id and not forced_target_chembl_id:
                seq = self._get_sequence_for_id(self.config.prot_id)
                if seq:
                    if "Target_FASTA" not in df.columns:
                        df["Target_FASTA"] = seq
                    else:
                        missing_mask = df["Target_FASTA"].isna() | (df["Target_FASTA"].astype(str).str.strip() == "")
                        if missing_mask.any():
                            df.loc[missing_mask, "Target_FASTA"] = seq
            df = df.dropna(subset=["Target_FASTA"])

        if len(df) == 0:
            return df

        batch_size = self.config.batch_size
        predictions = np.zeros(len(df), dtype=np.float32)
        
        if not is_selfies:
            self.logger.info("Converting SMILES to SELFIES for prediction...")
        selfies_list = to_selfies_list(
            molecules=df[mol_col].tolist(),
            is_selfies=is_selfies,
            invalid_token="[nop]",
        )
        df["_selfies_input"] = selfies_list

        grouped = df.groupby("Target_FASTA", sort=False).indices
        for sequence, row_indices in tqdm(grouped.items(), desc="Predicting Targets"):
            prot_input_ids, prot_attention_mask = self._get_protein_embeddings(sequence)
            protein_embeddings = self._encode_protein_for_model(self.prediction_model, prot_input_ids, prot_attention_mask)

            row_indices = list(row_indices)
            for start in range(0, len(row_indices), batch_size):
                batch_indices = row_indices[start : start + batch_size]
                batch_selfies = df.iloc[batch_indices]["_selfies_input"].tolist()

                batch_mol_ids, _ = tokenize_selfies_for_inference(
                    selfies_list=batch_selfies,
                    mol_tokenizer=self.mol_tokenizer,
                    max_mol_len=self.config.max_mol_len,
                    device=self.device,
                )

                current_batch_len = len(batch_indices)
                batch_protein_embeddings = protein_embeddings.repeat(current_batch_len, 1, 1)
                batch_prot_mask = prot_attention_mask.repeat(current_batch_len, 1)
                preds = self._predict_pchembl_batch(batch_protein_embeddings, batch_prot_mask, batch_mol_ids, None)
                predictions[batch_indices] = preds

        df["Predicted_pChEMBL"] = predictions.tolist()
        df = df.drop(columns=["_selfies_input"])
        self.logger.info("pChEMBL prediction finished in %.2fs", time.perf_counter() - start_time)
        return df

    def generate_molecules(self, protein_sequence: str, num_samples: int) -> pd.DataFrame:
        """
        Generate molecules for a given protein sequence.
        
        Args:
            protein_sequence: Target protein sequence
            num_samples: Number of molecules to generate
            
        Returns:
            DataFrame with generated molecules
        """
        self.logger.info(f"Generating {num_samples} molecules for protein sequence...")
        generation_start = time.perf_counter()
        
        # Get protein embeddings
        prot_input_ids, prot_attention_mask = self._get_protein_embeddings(protein_sequence)
        generation_protein_embeddings = self._encode_protein_for_model(
            self.generation_model,
            prot_input_ids,
            prot_attention_mask,
        )
        
        # Generate molecules
        generated_output = self._generate_molecules_batch_with_tokens(
            generation_protein_embeddings,
            prot_attention_mask,
            num_samples,
        )
        generated_selfies, generated_token_ids = generated_output
        
        # Calcuate pChEMBL for generated molecules
        self.logger.info("Predicting pChEMBL for generated molecules...")
        scoring_start = time.perf_counter()
        
        prediction_protein_embeddings = generation_protein_embeddings
        prediction_protein_mask = prot_attention_mask
        if self.prediction_model is not self.generation_model:
            prediction_protein_embeddings = self._encode_protein_for_model(
                self.prediction_model,
                prot_input_ids,
                prot_attention_mask,
            )

        batch_size = self.config.batch_size
        all_preds = []
        
        num_batches = (generated_token_ids.shape[0] + batch_size - 1) // batch_size if generated_token_ids.numel() else 0
        
        for i in tqdm(range(num_batches), desc="Predicting pChEMBL"):
            batch_mol_ids = generated_token_ids[i * batch_size : (i + 1) * batch_size].to(self.device)
            current_batch_len = batch_mol_ids.shape[0]
            batch_protein_embeddings = prediction_protein_embeddings.repeat(current_batch_len, 1, 1)
            batch_prot_mask = prediction_protein_mask.repeat(current_batch_len, 1)
            preds = self._predict_pchembl_batch(batch_protein_embeddings, batch_prot_mask, batch_mol_ids, None)
            all_preds.extend(preds)
        
        # Extract model name from model file path
        model_name = os.path.basename(self.config.model_file.rstrip('/'))
        if not model_name:  # Handle case where path ends with '/'
            model_name = os.path.basename(os.path.dirname(self.config.model_file))
        
        # Create DataFrame with comprehensive metadata
        generation_timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        
        results_df = pd.DataFrame({
            'Generated_SELFIES': generated_selfies,
            'Predicted_pChEMBL': all_preds,
            'Protein_ID': [self.config.prot_id] * len(generated_selfies),
            'Model_Name': [model_name] * len(generated_selfies),
            'Protein_Encoder': [self.config.prot_emb_model] * len(generated_selfies),
            'Generation_Temperature': [self.generation_config.temperature] * len(generated_selfies),
            'Generation_Top_p': [self.generation_config.top_p] * len(generated_selfies),
            'Max_Length': [self.generation_config.max_length] * len(generated_selfies),
            'Batch_Size': [self.config.batch_size] * len(generated_selfies),
            'Generation_Timestamp': [generation_timestamp] * len(generated_selfies)
        })
        
        self.logger.info(f"Generated {len(generated_selfies)} molecules")
        self.logger.info(
            "Generation time: %.2fs, pChEMBL scoring time: %.2fs",
            scoring_start - generation_start,
            time.perf_counter() - scoring_start,
        )
        
        return results_df
    
    def calculate_metrics(self, generated_df: pd.DataFrame, reference_smiles: List[str]) -> Dict:
        """
        Calculate generation metrics.
        
        Args:
            generated_df: DataFrame with generated molecules
            reference_smiles: Reference SMILES for comparison
            
        Returns:
            Dictionary with calculated metrics
        """
        self.logger.info("Calculating metrics...")
        metrics_start = time.perf_counter()
        
        try:
            metrics, results_df = metrics_calculation(
                predictions=generated_df['Generated_SELFIES'].tolist(),
                references=reference_smiles,
                train_data=self.train_data,
                train_vec=self.train_vec,
                training=False,
                return_details=True,
            )
            
            # Extract SMILES from results DataFrame and add to main DataFrame
            if 'smiles' in results_df.columns:
                generated_df['Generated_SMILES'] = results_df['smiles'].tolist()
            else:
                # Fallback: convert SELFIES to SMILES directly
                self.logger.info("Converting SELFIES to SMILES using sf.decoder...")
                generated_smiles = []
                for selfies in tqdm(generated_df['Generated_SELFIES'], desc="Converting SELFIES→SMILES"):
                    try:
                        smiles = sf.decoder(selfies.replace(" ", ""))
                        generated_smiles.append(smiles if smiles else "")
                    except Exception as e:
                        self.logger.warning(f"Failed to decode SELFIES: {selfies}, error: {e}")
                        generated_smiles.append("")
                generated_df['Generated_SMILES'] = generated_smiles
            
            # Add additional molecular properties if available in results
            if 'sa' in results_df.columns:
                generated_df['sa'] = results_df['sa'].tolist()
            if 'qed' in results_df.columns:
                generated_df['qed'] = results_df['qed'].tolist()
            if 'logp' in results_df.columns:
                generated_df['logp'] = results_df['logp'].tolist()
            if 'similarity_eval' in results_df.columns:
                generated_df['similarity_eval'] = results_df['similarity_eval'].tolist()
            if 'similarity_train' in results_df.columns:
                generated_df['similarity_train'] = results_df['similarity_train'].tolist()
            
            self.logger.info("Metrics calculated successfully")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"{key}: {value:.4f}")
                else:
                    self.logger.info(f"{key}: {value}")
            self.logger.info("Metric calculation finished in %.2fs", time.perf_counter() - metrics_start)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            # Fallback: at least convert SELFIES to SMILES
            try:
                self.logger.info("Attempting fallback SELFIES to SMILES conversion...")
                generated_smiles = []
                for selfies in tqdm(generated_df['Generated_SELFIES'], desc="Fallback SELFIES→SMILES"):
                    try:
                        smiles = sf.decoder(selfies.replace(" ", ""))
                        generated_smiles.append(smiles if smiles else "")
                    except Exception as decode_error:
                        self.logger.warning(f"Failed to decode SELFIES: {selfies}, error: {decode_error}")
                        generated_smiles.append("")
                generated_df['Generated_SMILES'] = generated_smiles
                self.logger.info("Fallback SMILES conversion completed")
            except Exception as fallback_error:
                self.logger.error(f"Fallback SMILES conversion failed: {fallback_error}")
            
            return {}
    
    def save_results(self, generated_df: pd.DataFrame, metrics: Dict, output_path: str):
        """
        Save generation results and metrics.
        
        Args:
            generated_df: DataFrame with generated molecules
            metrics: Calculated metrics
            output_path: Path to save results
        """
        # Create output directory
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Log what columns are being saved
        self.logger.info(f"Saving DataFrame with columns: {list(generated_df.columns)}")
        
        has_generated_selfies = 'Generated_SELFIES' in generated_df.columns
        has_generated_smiles = 'Generated_SMILES' in generated_df.columns
        molecule_column, is_selfies = find_molecule_column(generated_df.columns)

        if has_generated_selfies and has_generated_smiles:
            self.logger.info("✅ Both SELFIES and SMILES representations will be saved")
            valid_selfies = generated_df['Generated_SELFIES'].notna().sum()
            valid_smiles = (generated_df['Generated_SMILES'].notna() & 
                          (generated_df['Generated_SMILES'] != "")).sum()
            self.logger.info(f"📊 Valid SELFIES: {valid_selfies}/{len(generated_df)}")
            self.logger.info(f"📊 Valid SMILES: {valid_smiles}/{len(generated_df)}")
        elif has_generated_selfies:
            self.logger.warning("⚠️ Only SELFIES representation available")
        elif has_generated_smiles:
            self.logger.warning("⚠️ Only SMILES representation available")
        elif molecule_column is not None:
            rep = "SELFIES" if is_selfies else "SMILES"
            self.logger.info("Saving prediction results with input %s column '%s'", rep, molecule_column)
        else:
            self.logger.error("❌ Neither SELFIES nor SMILES representations found!")
        
        # Save molecules
        generated_df.to_csv(output_path, index=False)
        self.logger.info(f"💾 Molecules saved to {output_path}")
        
        # Save metrics
        metrics_path = output_path.replace('.csv', '_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        self.logger.info(f"📈 Metrics saved to {metrics_path}")
    
    def run_generation(self):
        """Run the complete pipeline based on configuration mode."""
        
        if self.config.mode == "prediction":
             self.logger.info("Starting pChEMBL prediction pipeline...")
             
             # Step 1: Load molecules
             molecules_df = self._load_prediction_data()

             # Step 2: Predict
             results_df = self.predict_pchembl(molecules_df)
             
             # Step 3: Save
             self.save_results(results_df, {}, self.config.output_file)
             self._save_prediction_plots(results_df)
             
             return results_df, {}

        else:
            # Generation Mode
            self.logger.info("Starting molecule generation pipeline...")
            
            # Overall progress tracking
            total_steps = 4  # dataset loading, generation, metrics/conversion, saving
            with tqdm(total=total_steps, desc="🧬 Prot2Mol Pipeline", unit="step") as overall_pbar:
                
                # Step 1: Load dataset
                overall_pbar.set_description("📂 Loading dataset")
                self.train_data, self.train_vec, test_data = self._load_dataset()
                overall_pbar.update(1)
                
                # Get protein sequence
                if len(test_data) == 0:
                    raise ValueError(f"No test data found for protein {self.config.prot_id}")
                
                # Use the first protein sequence (they should all be the same for the target protein)
                protein_sequence = test_data.iloc[0]['Target_FASTA']
                reference_smiles = []
                if 'Compound_SMILES' in test_data.columns:
                    reference_smiles = canonicalize_smiles_list(test_data['Compound_SMILES'].tolist(), drop_invalid=True)
                elif 'Compound_SELFIES' in test_data.columns:
                    decoded = decode_selfies_list(test_data['Compound_SELFIES'].tolist())
                    reference_smiles = canonicalize_smiles_list(decoded, drop_invalid=True)
                else:
                    self.logger.warning("No Compound_SMILES or Compound_SELFIES column available for reference metrics")
                
                # Step 2: Generate molecules
                overall_pbar.set_description("🔬 Generating molecules")
                generated_df = self.generate_molecules(protein_sequence, self.config.num_samples)
                overall_pbar.update(1)
                
                # Step 3: Calculate metrics or convert SELFIES
                overall_pbar.set_description("📊 Processing results")
                metrics = {}
                if reference_smiles:
                    metrics = self.calculate_metrics(generated_df, reference_smiles)
                else:
                    # If no reference data, still convert SELFIES to SMILES
                    self.logger.info("No reference data available for metrics, but converting SELFIES to SMILES...")
                    try:
                        generated_smiles = []
                        for selfies in tqdm(generated_df['Generated_SELFIES'], desc="Converting SELFIES→SMILES"):
                            try:
                                smiles = sf.decoder(selfies.replace(" ", ""))
                                generated_smiles.append(smiles if smiles else "")
                            except Exception as e:
                                self.logger.warning(f"Failed to decode SELFIES: {selfies}, error: {e}")
                                generated_smiles.append("")
                        generated_df['Generated_SMILES'] = generated_smiles
                        self.logger.info("SELFIES to SMILES conversion completed")
                    except Exception as e:
                        self.logger.error(f"Error converting SELFIES to SMILES: {e}")
                overall_pbar.update(1)
                
                # Step 4: Save results
                overall_pbar.set_description("💾 Saving results")
                self.save_results(generated_df, metrics, self.config.output_file)
                self._save_prediction_plots(generated_df)
                overall_pbar.update(1)
                
                overall_pbar.set_description("✅ Pipeline completed")
            
            self.logger.info("Molecule generation pipeline completed successfully!")
            
            return generated_df, metrics


def parse_arguments(argv=None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate molecules for protein targets using Prot2Mol",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        "--model_file",
        required=True,
        help="Path to the trained generation model directory"
    )
    parser.add_argument(
        "--prediction_model_file",
        default=None,
        help="Path to the trained prediction model directory. If not specified, uses generation model."
    )
    parser.add_argument(
        "--prot_emb_model",
        default="prot_t5",
        choices=["prot_t5", "esm2", "saprot"],
        help="Protein embedding model used in training"
    )
    
    # Data paths
    parser.add_argument(
        "--selfies_path",
        required=True,
        help="Path to the SELFIES dataset directory"
    )
    parser.add_argument(
        "--models_base",
        type=str,
        default=None,
        help="Base directory containing local HF model caches (models--*).",
    )
    parser.add_argument(
        "--data_path",
        default=None,
        help="Legacy dataset directory or file path used for sequence lookup fallback.",
    )
    parser.add_argument(
        "--chembl_uniprot_mapping_path",
        type=str,
        default=None,
        help="Path to CHEMBL->UniProt mapping file.",
    )
    parser.add_argument(
        "--protein_targets_path",
        type=str,
        default=None,
        help="Path to Papyrus protein target file with target_id and Sequence columns.",
    )
    parser.add_argument(
        "--reference_dataset",
        type=str,
        default=None,
        help="Training dataset CSV used for protein-level comparison. Defaults to checkpoint dataset_source_path.",
    )
    parser.add_argument(
        "--compare_protein_id",
        type=str,
        default=None,
        help="Protein identifier used to pull real training rows for comparison plots. Defaults to --prot_id when available.",
    )
    parser.add_argument(
        "--save_distribution_plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save a predicted pChEMBL distribution plot for generated or scored molecules.",
    )
    parser.add_argument(
        "--distribution_bins",
        type=int,
        default=40,
        help="Maximum histogram bin count for saved pChEMBL distribution plots.",
    )
    parser.add_argument(
        "--prot_id",
        required=False,
        help="Target protein CHEMBL ID (required for generation, optional for prediction if provided in input file)"
    )
    parser.add_argument(
        "--target_chembl_id",
        type=str,
        default=None,
        help="Optional CHEMBL target id to score all input molecules against in prediction mode. When set, overrides target columns in the input file.",
    )
    
    # Operation modes
    parser.add_argument(
        "--mode",
        choices=["generation", "prediction"],
        default="generation",
        help="Operation mode: 'generation' to generate new molecules, 'prediction' to predict pChEMBL for existing molecules"
    )
    parser.add_argument(
        "--input_molecules",
        help="Path to CSV file containing molecules for prediction mode. Must contain 'smiles' or 'selfies' column."
    )
    
    # Normalization parameters
    parser.add_argument(
        "--pchembl_mean",
        type=float,
        default=5.924,
        help="Mean pChEMBL value for denormalization"
    )
    parser.add_argument(
        "--pchembl_std",
        type=float,
        default=1.362,
        help="Standard deviation of pChEMBL value for denormalization"
    )
    
    # Generation parameters
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of molecules to generate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--prot_max_length",
        type=int,
        default=1024,
        help="Maximum protein sequence length"
    )
    parser.add_argument(
        "--max_mol_len",
        type=int,
        default=200,
        help="Maximum molecule sequence length"
    )
    
    # Output options
    parser.add_argument(
        "--output_file",
        help="Output file path for generated molecules"
    )
    parser.add_argument(
        "--attn_output",
        action="store_true",
        help="Output attention weights"
    )
    
    # Model architecture (needed for model loading)
    parser.add_argument("--n_layer", type=int, default=1, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--n_emb", type=int, default=1024, help="Embedding dimension")
    
    return parse_args_with_config(parser, section="generate", argv=argv)


def main():
    """Main entry point."""
    # Parse arguments
    config = parse_arguments()
    
    # Generate output file path if not provided
    if config.output_file is None:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Get directory name safely and truncate if too long
        model_name = os.path.basename(os.path.normpath(config.model_file))
        if len(model_name) > 60:
            model_name = model_name[:30] + "..." + model_name[-27:]
            
        config.output_file = f"./generated_molecules_{model_name}_{config.prot_emb_model}_{config.prot_id}_{timestamp}.csv"
    
    try:
        # Initialize generator and run
        generator = MoleculeGenerator(config)
        generated_df, metrics = generator.run_generation()
        
        print(f"\n✅ Generation completed successfully!")
        print(f"📊 Generated {len(generated_df)} molecules")
        print(f"💾 Results saved to: {config.output_file}")
        
        if metrics:
            print(f"\n📈 Key Metrics:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
