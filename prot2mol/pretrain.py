# Standard library imports
import os
import sys
import math
import argparse
import logging
import datetime
import re
from typing import Dict, List, Tuple, Any, Optional
from torch.distributed import init_process_group, destroy_process_group, barrier
# Third-party library imports
import numpy as np
import torch
import wandb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datasets import load_dataset, Dataset, load_from_disk
from transformers import (
    TrainingArguments,
    BartTokenizer,
    T5Tokenizer,
    AutoTokenizer
)

# Local application imports
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing import train_val_test
from prot2mol.trainer import GPT2_w_crs_attn_Trainer
from prot2mol.utils import metrics_calculation, canonicalize_smiles_list, decode_selfies_list
from prot2mol.protein_encoders import get_protein_tokenizer
from prot2mol.model import create_prot2mol_model

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DIR"] = "/gpfs/projects/etur29/atabey/"

class TrainingScript:
    """Trainer for the Prot2Mol model that handles both pre-training and fine-tuning.
    
    Attributes:
        model_config (dict): Configuration for the model architecture
        training_config (dict): Configuration for training parameters
        selfies_path (str): Path to the SELFIES dataset
        pretrain_save_to (str): Directory to save the trained model
        run_name (str): Unique identifier for this training run
        mol_tokenizer: Tokenizer for molecule sequences
        prot_tokenizer: Tokenizer for protein sequences
        model: The unified Prot2Mol model (protein encoder + molecule decoder)
        training_vec: Training vectors for similarity calculation
    """

    def __init__(self, config, selfies_path, pretrain_save_to, dataset_name, run_name):
        """Initialize the training script with configuration and paths.
        
        Args:
            config: Parsed command line arguments
            selfies_path: Path to the SELFIES dataset
            pretrain_save_to: Directory to save the trained model
            dataset_name: Name of the dataset being used
            run_name: Unique identifier for this training run
        """
        self.logger = logging.getLogger(__name__)
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        
        # Organize configurations into logical groups
        self.model_config = {
            'prot_emb_model': config.prot_emb_model,
            'n_layer': config.n_layer,
            'n_head': config.n_head,
            'n_emb': config.n_emb,
            'max_mol_len': config.max_mol_len,
            'prot_max_length': config.prot_max_length,
            'train_encoder_model': config.train_encoder_model,
            'train_decoder_model': config.train_decoder_model,
            'train_pchembl_head': config.train_pchembl_head
        }
        
        self.training_config = {
            'train_batch_size': config.train_batch_size,
            'valid_batch_size': config.valid_batch_size,
            'epochs': config.epoch,
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'gradient_accumulation_steps': config.gradient_accumulation_steps,
            'dataloader_num_workers': config.dataloader_num_workers,
            'resume_from_checkpoint': config.resume_from_checkpoint,
            'load_pretrained_model': config.load_pretrained_model,
            'ignore_mismatched_optimizer': config.ignore_mismatched_optimizer
        }

        self.selfies_path = selfies_path
        self.pretrain_save_to = pretrain_save_to
        self.run_name = run_name
        self.train_smiles_list = []
        self.eval_reference_smiles = []
        
        # Log checkpoint information if resuming
        if self.training_config['resume_from_checkpoint']:
            self.logger.info(f"Checkpoint resume mode enabled: {self.training_config['resume_from_checkpoint']}")
        
        # Log pretrained model loading if specified
        if self.training_config['load_pretrained_model']:
            self.logger.info(f"Will load pretrained model weights from: {self.training_config['load_pretrained_model']}")
            # Ensure both aren't set at the same time
            if self.training_config['resume_from_checkpoint']:
                raise ValueError("Cannot use both --resume_from_checkpoint and --load_pretrained_model. "
                               "Use --resume_from_checkpoint to continue training with same dataset, "
                               "or --load_pretrained_model to fine-tune on a new dataset.")
        
        self._prepare_normalization_and_thresholds()
        
        # Load training vectors for similarity calculation
        self._load_training_vectors()
        
        # Initialize tokenizers and models
        self._init_tokenizers()
        self._init_models()
        
        self.logger.info(f"Model parameter count: {self.model.num_parameters():,}")

    def _prepare_normalization_and_thresholds(self):
        """Calculate and store normalization constants and thresholds from the dataset."""
        # OPTIMIZATION: Skip if not training pChEMBL head
        if not self.model_config['train_pchembl_head']:
            self.logger.info("pChEMBL head training is disabled. Skipping normalization setup.")
            self.pchembl_mean = 0.0
            self.pchembl_std = 1.0
            self.pchembl_threshold = 6.0
            return

        import pandas as pd
        self.logger.info("Preparing normalization constants from dataset...")
        df = pd.read_csv(self.selfies_path)
        pchembl_values = df['pchembl_value_Median'].dropna()
        
        self.pchembl_mean = pchembl_values.mean()
        self.pchembl_std  = pchembl_values.std(ddof=0)
        self.pchembl_threshold = 6.0
        
        self.logger.info(f"pChEMBL normalization range: mean={self.pchembl_mean:.3f}, std={self.pchembl_std:.3f}")
        self.logger.info(f"pChEMBL positive threshold set to: >={self.pchembl_threshold}")

    def _load_training_vectors(self):
        """Load training vectors for similarity calculation if available."""
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        train_vecs_path = os.path.join(data_dir, "train_vecs.npy")
        
        if not os.path.exists(train_vecs_path):
            self.logger.warning(f"Training vectors file not found at {train_vecs_path}")
            self.logger.info("Attempting to generate training vectors automatically...")
            
            # Try to generate training vectors automatically
            try:
                self._generate_training_vectors(data_dir, train_vecs_path)
                self.training_vec = np.load(train_vecs_path)
                self.logger.info(f"Successfully generated and loaded training vectors from {train_vecs_path}")
            except Exception as e:
                self.logger.warning(f"Failed to auto-generate training vectors: {e}")
                self.logger.warning("You can manually generate train_vecs.npy using: python generate_train_vecs.py")
                self.training_vec = None
        else:
            self.logger.info(f"Loading existing training vectors from {train_vecs_path}")
            self.training_vec = np.load(train_vecs_path)
            self.logger.info(f"Loaded training vectors with shape: {self.training_vec.shape}")

    def _extract_smiles_list(self, data_source, drop_invalid=False):
        """Extract SMILES strings from various dataset formats without expensive canonicalization."""
        smiles_values = []
        needs_canonicalization = False
        try:
            if hasattr(data_source, "column_names"):
                if "Compound_SMILES" in data_source.column_names:
                    smiles_values = list(data_source["Compound_SMILES"])
                elif "Compound_SELFIES" in data_source.column_names:
                    smiles_values = decode_selfies_list(list(data_source["Compound_SELFIES"]))
                    needs_canonicalization = True
            elif hasattr(data_source, "columns"):
                if "Compound_SMILES" in data_source.columns:
                    smiles_values = data_source["Compound_SMILES"].tolist()
                elif "Compound_SELFIES" in data_source.columns:
                    smiles_values = decode_selfies_list(data_source["Compound_SELFIES"].tolist())
                    needs_canonicalization = True
            elif isinstance(data_source, list):
                smiles_values = data_source
            else:
                self.logger.warning(f"Unsupported data source for SMILES extraction: {type(data_source)}")
        except Exception as exc:
            self.logger.warning(f"Failed to extract SMILES: {exc}")
            smiles_values = []
        
        if not smiles_values:
            return []
        
        cleaned_smiles = []
        for entry in smiles_values:
            if isinstance(entry, str):
                stripped = entry.strip()
                if stripped:
                    cleaned_smiles.append(stripped)
            elif isinstance(entry, bytes):
                stripped = entry.decode("utf-8").strip()
                if stripped:
                    cleaned_smiles.append(stripped)
        
        if not cleaned_smiles:
            return []
        
        if needs_canonicalization:
            return canonicalize_smiles_list(cleaned_smiles, drop_invalid=drop_invalid)
        
        if drop_invalid:
            return [s for s in cleaned_smiles if s]
        return cleaned_smiles
    
    def _generate_training_vectors(self, data_dir: str, output_path: str):
        """
        Generate training vectors using the same dataset path that will be used for training.
        
        Args:
            data_dir: Data directory path
            output_path: Path where to save the training vectors
        """
        import pandas as pd
        import selfies as sf
        from prot2mol.utils_fps import generate_morgan_fingerprints_parallel
        
        self.logger.info("Generating training vectors from dataset...")
        
        # Use the same dataset path from the training configuration
        dataset_path = self.selfies_path
        
        # Load dataset and convert SELFIES to SMILES
        self.logger.info(f"Loading dataset from: {dataset_path}")
        
        # Read in chunks to handle large datasets efficiently
        chunk_size = 10000
        smiles_list = []
        total_rows = 0
        
        for chunk_df in pd.read_csv(dataset_path, chunksize=chunk_size):
            total_rows += len(chunk_df)
            
            # Extract SELFIES and convert to SMILES
            selfies_chunk = chunk_df['Compound_SELFIES'].tolist()
            
            for selfies_str in selfies_chunk:
                try:
                    if selfies_str and isinstance(selfies_str, str):
                        smiles = sf.decoder(selfies_str)
                        if smiles and smiles.strip():
                            smiles_list.append(smiles)
                except Exception:
                    continue  # Skip invalid SELFIES
            
            self.logger.info(f"Processed {total_rows} rows, {len(smiles_list)} valid SMILES so far...")
            
            # Limit processing for very large datasets to avoid memory issues
            if len(smiles_list) >= 100000:  # Process max 100k molecules for training vectors
                self.logger.info(f"Limiting to {len(smiles_list)} molecules for training vectors")
                break
        
        if not smiles_list:
            raise ValueError("No valid SMILES found in the dataset")
        
        self.logger.info(f"Generating Morgan fingerprints for {len(smiles_list)} molecules...")
        
        # Generate fingerprints using parallel processing
        train_vecs = generate_morgan_fingerprints_parallel(
            smiles=smiles_list,
            radius=2,
            nBits=1024,
            n_jobs=None  # Auto-detect, max 10 cores
        )
        
        # Save the vectors
        self.logger.info(f"Saving training vectors to: {output_path}")
        os.makedirs(data_dir, exist_ok=True)
        np.save(output_path, train_vecs)
        
        self.logger.info(f"Generated training vectors with shape: {train_vecs.shape}")

    def ddp_setup(self):
        """Initialize DDP with proper error handling and device setup."""
        try:
            local_rank = int(os.environ["LOCAL_RANK"])
            global_rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            
            # Set CUDA device
            torch.cuda.set_device(local_rank)
            
            # Initialize process group
            init_process_group(backend="nccl", rank=global_rank, world_size=world_size)
            
            self.logger.info(f"Initialized DDP with rank {global_rank}/{world_size} on device {local_rank}")
            
            # Verify CUDA is available
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available but DDP is being initialized")
                
            # Log GPU memory info
            device = torch.cuda.current_device()
            memory_total = torch.cuda.get_device_properties(device).total_memory / 1e9
            self.logger.info(f"Rank {global_rank}: Using GPU {device} with {memory_total:.1f}GB memory")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize DDP: {e}")
            raise

    def _get_model_path(self, model_name):
        """Get the correct path for a locally cached model."""
        # Use environment variable or fallback to hardcoded path
        models_base = os.environ.get('MODELS_BASE_PATH', './models')
        base_path = os.path.join(models_base, f"models--{model_name}")
        snapshots_path = os.path.join(base_path, "snapshots")
        
        if os.path.exists(snapshots_path):
            # Get the first (and typically only) snapshot directory
            snapshots = os.listdir(snapshots_path)
            if snapshots:
                return os.path.join(snapshots_path, snapshots[0])
        
        return base_path

    def _init_tokenizers(self):
        """Initialize tokenizers for proteins and molecules."""
        self.logger.info("Initializing tokenizers...")
        mol_model_path = self._get_model_path("zjunlp--MolGen-large")
        self.mol_tokenizer = BartTokenizer.from_pretrained(mol_model_path, padding_side="left")
        self.prot_tokenizer = get_protein_tokenizer(self.model_config['prot_emb_model'])

    def _init_models(self):
        """Initialize the unified Prot2Mol model."""
        self.logger.info("Initializing unified Prot2Mol model...")
        
        # Add mol_tokenizer to model config for model initialization
        model_config_with_tokenizer = self.model_config.copy()
        model_config_with_tokenizer['mol_tokenizer'] = self.mol_tokenizer
        
        # Create the unified model
        self.model = create_prot2mol_model(model_config_with_tokenizer)

    def _load_pretrained_weights(self, pretrained_model_path):
        """Load pretrained model weights for fine-tuning.
        
        This method loads ONLY the model weights from a pretrained checkpoint,
        without loading the training state (optimizer, scheduler, epoch count, etc.).
        This is useful for fine-tuning on a new dataset with a fresh training state.
        
        Args:
            pretrained_model_path: Path to the pretrained model directory or checkpoint
        """
        import glob
        
        self.logger.info(f"Loading pretrained weights from: {pretrained_model_path}")
        
        # Check if this is a checkpoint directory or a model directory
        if os.path.exists(os.path.join(pretrained_model_path, "pytorch_model.bin")):
            # This is a saved model directory
            model_file = os.path.join(pretrained_model_path, "pytorch_model.bin")
        elif os.path.exists(os.path.join(pretrained_model_path, "model.safetensors")):
            # SafeTensors format
            model_file = os.path.join(pretrained_model_path, "model.safetensors")
        else:
            # Maybe it's a checkpoint directory, look for the best checkpoint
            checkpoint_pattern = os.path.join(pretrained_model_path, "checkpoint-*", "pytorch_model.bin")
            checkpoints = glob.glob(checkpoint_pattern)
            
            if not checkpoints:
                # Try safetensors
                checkpoint_pattern = os.path.join(pretrained_model_path, "checkpoint-*", "model.safetensors")
                checkpoints = glob.glob(checkpoint_pattern)
            
            if not checkpoints:
                # Maybe the path itself is a checkpoint
                if os.path.exists(os.path.join(pretrained_model_path, "pytorch_model.bin")):
                    model_file = os.path.join(pretrained_model_path, "pytorch_model.bin")
                else:
                    raise FileNotFoundError(
                        f"Could not find model weights in {pretrained_model_path}. "
                        f"Expected pytorch_model.bin or model.safetensors"
                    )
            else:
                # Use the last checkpoint (highest number)
                model_file = sorted(checkpoints)[-1]
        
        self.logger.info(f"Loading model weights from: {model_file}")
        
        # Load the state dict
        if model_file.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(model_file)
        else:
            state_dict = torch.load(model_file, map_location='cpu')
        
        # Load the state dict into the model
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            self.logger.warning(f"Missing keys when loading pretrained weights: {missing_keys}")
        if unexpected_keys:
            self.logger.warning(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")
        
        self.logger.info("Successfully loaded pretrained weights!")
        self.logger.info("Training will start from epoch 0 with fresh optimizer state.")

    def tokenize_prot_function(self, batch):
        """Tokenize protein sequences in the batch.
        
        Args:
            batch: Batch of data containing protein sequences
            
        Returns:
            dict: Dictionary with tokenized protein data
        """
        try:
            # Replace non-standard amino acids with 'X'
            if self.model_config['prot_emb_model'] == "prot_t5":
                sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in batch["Target_FASTA"]]
            else:
                sequence_examples = [re.sub(r"[UZOB]", "X", seq) for seq in batch["Target_FASTA"]]

            # Tokenize the sequences
            ids = self.prot_tokenizer.batch_encode_plus(
                sequence_examples, 
                add_special_tokens=True, 
                truncation=True,
                max_length=self.model_config['prot_max_length'],
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
        """Tokenize molecule SELFIES strings in the batch.
        
        Args:
            batch: Batch of data containing molecule SELFIES strings
            
        Returns:
            dict: Dictionary with tokenized molecule data
        """
        try:
            # Tokenize SELFIES
            ids = self.mol_tokenizer.batch_encode_plus(
                batch["Compound_SELFIES"], 
                add_special_tokens=True, 
                truncation=True,
                max_length=self.model_config['max_mol_len'],
                padding="max_length",
                return_tensors="pt"
            )
            
            pchembl_values = batch["pchembl_value_Median"]
            labels = ids['input_ids'].clone()
            
            # Mask padded positions in labels (set to -100 to ignore in loss)
            # This is critical for left-padded sequences
            pad_mask = ids['input_ids'] == self.mol_tokenizer.pad_token_id
            labels[pad_mask] = -100
            
            pchembl_values = batch["pchembl_value_Median"]
            normalized_pchembl = []
            train_lm_flags = []
            
            # Check if pChEMBL head is trained
            should_run_pchembl = self.model_config['train_pchembl_head']
            
            if not should_run_pchembl:
                # If head is disabled, treat ALL samples as valid for LM
                # We ignore pChEMBL values entirely
                train_lm_flags = [True] * len(pchembl_values)
                normalized_pchembl = [0.0] * len(pchembl_values) # Dummy values
                
            else:
                # Normalize pchembl values and determine train_lm flag
                # Only train LM on positive samples (pchembl >= threshold)
                negative_sample_count = 0
                
                for i, val in enumerate(pchembl_values):
                    
                    if val >= self.pchembl_threshold:
                        train_lm_flags.append(True)
                    else:
                        train_lm_flags.append(False)
                        negative_sample_count += 1
                    
                    # Apply Min-Max normalization
                    normalized_val = (val - self.pchembl_mean) / (self.pchembl_std + 1e-8)
                    normalized_pchembl.append(normalized_val)
                
                # Debug: log how many samples are negative
                if negative_sample_count > 0:
                    print(f"DEBUG tokenize_mol: {negative_sample_count}/{len(pchembl_values)} negative samples (pchembl < {self.pchembl_threshold}) - train_lm=False for these")

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

    def preprocess_logits_for_metrics(self, logits, labels):
        """
        Extracts predictions for language modeling and pChEMBL prediction from the model output.
        
        Returns only the language model predictions for Transformers compatibility.
        The pChEMBL predictions will be handled separately in compute_metrics.
        """
        try:
            
            # Handle different logit formats from the unified model
            if isinstance(logits, dict) and 'logits' in logits:
                # Extract main LM logits from model output dict
                raw_logits = logits['logits']
            elif isinstance(logits, (tuple, list)) and len(logits) > 1:
                # Handle tuple/list format - the actual logits are usually at index 1
                # Based on debug: logits[0] is loss, logits[1] is actual logits, logits[2] is pchembl_preds
                raw_logits = logits[1] if isinstance(logits[1], torch.Tensor) and len(logits[1].shape) >= 2 else logits[0]
            elif isinstance(logits, (tuple, list)) and len(logits) > 0:
                # Fallback for single element tuple
                raw_logits = logits[0] if isinstance(logits[0], torch.Tensor) else logits
            elif isinstance(logits, torch.Tensor):
                # Direct tensor format
                raw_logits = logits
            else:
                self.logger.error(f"Unexpected logits format: {type(logits)}")
                return torch.argmax(labels, dim=-1) if labels is not None else None


            
            # Get predictions from logits
            pred_ids = torch.argmax(raw_logits, dim=-1)
            # Ensure shapes match
            if labels is not None and labels.shape != pred_ids.shape:
                self.logger.warning(f"Shape mismatch: labels {labels.shape} vs pred_ids {pred_ids.shape}")
                # Try to fix common shape mismatches
                if pred_ids.numel() == labels.numel():
                    pred_ids = pred_ids.reshape(labels.shape)
                else:
                    print(f"DEBUG preprocess_logits - Cannot reshape, returning None")
                    return None

            # Return only the LM predictions as a tensor for concatenation compatibility
            
            return pred_ids

        except Exception as e:
            self.logger.error(f"Error preprocessing logits: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics for both language modeling and pChEMBL prediction.
        
        Important:
        - LM metrics are computed ONLY for positive samples (train_lm=True, pChEMBL >= threshold)
        - pChEMBL metrics are computed for ALL samples (both positive and negative)
        
        Args:
            eval_pred: EvalPrediction object with predictions, labels, and inputs
            
        Returns:
            dict: Dictionary with computed metrics
        """
        try:
            predictions = eval_pred.predictions
            labels = eval_pred.label_ids


            # Initialize metrics dictionary
            metrics = {}
            
            # Check if predictions is None or empty
            if predictions is None:
                self.logger.warning(f"Rank {self.global_rank}: Predictions is None, skipping metrics computation")
                return {}
            
            # Handle language modeling metrics (only for positive samples with valid labels)
            if predictions is not None and labels is not None:
                # Convert predictions to numpy if they're tensors
                if hasattr(predictions, 'cpu'):
                    predictions_np = predictions.cpu().numpy()
                elif hasattr(predictions, 'numpy'):
                    predictions_np = predictions.numpy()
                else:
                    predictions_np = predictions
                    
                lm_metrics = self._compute_lm_metrics(predictions_np, labels)
                if lm_metrics:
                    metrics.update(lm_metrics)
                 
                else:
                    self.logger.info(f"Rank {self.global_rank}: No LM metrics (batch had no positive samples)")
            
            # Handle pChEMBL metrics using stored predictions from the trainer
            # Note: pChEMBL metrics are computed for ALL samples (positive and negative)
            # Note: In DDP, predictions are gathered from all ranks for complete metrics
            # Note: self.trainer is set during model_training() when trainer is created
            if hasattr(self, 'trainer') and self.trainer is not None:
                pchembl_preds, pchembl_targets = self.trainer.get_pchembl_predictions()
                if pchembl_preds is not None and pchembl_targets is not None:
                    # If there's a mismatch, only use the first N pChEMBL predictions that match LM count
                    # This handles cases where Transformers deduplicates predictions in DDP mode
                    if predictions is not None:
                        lm_sample_count = len(predictions_np) if 'predictions_np' in locals() else predictions.shape[0]
                        pchembl_sample_count = len(pchembl_preds)
                        
                        if lm_sample_count != pchembl_sample_count:
                            # Trim pChEMBL predictions to match LM count (removes duplicates from DDP)
                            pchembl_preds = pchembl_preds[:lm_sample_count]
                            pchembl_targets = pchembl_targets[:lm_sample_count]
                    
                    pchembl_metrics = self._compute_pchembl_metrics(pchembl_preds, pchembl_targets)
                    metrics.update(pchembl_metrics)
                    
                    # Clear pChEMBL predictions after computing metrics to prevent accumulation
                    self.trainer.clear_pchembl_predictions()
            
            
        except Exception as e:
            self.logger.error(f"Rank {self.global_rank}: Error computing metrics: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            metrics = {}

        return metrics
    
    def _compute_lm_metrics(self, predictions, labels):
        """Compute language modeling metrics.
        
        Note: This only computes metrics for samples with valid labels (train_lm=True).
        Negative samples (train_lm=False) have all labels masked as -100, so they are
        skipped here. pChEMBL metrics are computed separately for ALL samples.
        """
        try:
            # Check if predictions is None (error in preprocessing)
            if predictions is None:
                self.logger.error("LM predictions is None, likely due to logits preprocessing error")
                return {}
            
            # Check if we have any valid labels (non-negative samples)
            valid_labels = (labels != -100).sum()
            total_labels = labels.size
            
            
            if valid_labels == 0:
                self.logger.info(f"Rank {self.global_rank}: No valid LM labels (all negative samples). Skipping LM metrics.")
                return {}
            
            # Ensure predictions and labels have the same shape
            if predictions.shape != labels.shape:
                self.logger.error(f"Shape mismatch: predictions {predictions.shape} vs labels {labels.shape}")
                # If they have the same number of elements, reshape predictions
                if predictions.size == labels.size:
                    predictions = predictions.reshape(labels.shape)
                    self.logger.info(f"Reshaped predictions to: {predictions.shape}")
                else:
                    self.logger.error(f"Cannot reshape: different number of elements")
                    return {}
            
            # Replace -100 tokens with pad_token_id for decoding
            # Also mask predictions at positions where labels are -100 (padded/ignored positions)
            labels_for_decoding = np.where(labels != -100, labels, self.mol_tokenizer.pad_token_id)
            predictions_for_decoding = np.where(labels != -100, predictions, self.mol_tokenizer.pad_token_id)
            
            # Decode predictions and labels
            decoded_preds = self.mol_tokenizer.batch_decode(
                predictions_for_decoding, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )
            
            # Extract ground-truth SELFIES from labels (decoded)
            decoded_labels = self.mol_tokenizer.batch_decode(
                labels_for_decoding,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Prefer pre-cached SMILES from the dataset; fallback to decoding labels
            reference_smiles = getattr(self, "eval_reference_smiles", None)
            if not reference_smiles:
                fallback_decoded = decode_selfies_list(decoded_labels)
                reference_smiles = canonicalize_smiles_list(fallback_decoded, drop_invalid=True)
            lm_metrics = metrics_calculation(
                predictions=decoded_preds, 
                references=reference_smiles, 
                train_data=self.train_smiles_list, 
                train_vec=self.training_vec,
                training=True
            )
            
            # Add prefix to distinguish LM metrics
            return {f"lm_{k}": v for k, v in lm_metrics.items()}
            
        except Exception as e:
            self.logger.error(f"Error computing LM metrics: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {}
    
    def _compute_pchembl_metrics(self, pchembl_predictions, pchembl_targets):
        """Compute pChEMBL prediction evaluation metrics.
        
        Args:
            pchembl_predictions: Predicted pChEMBL values (numpy array)
            pchembl_targets: True pChEMBL values (numpy array, normalized)
        
        Returns:
            dict: Dictionary with pChEMBL regression metrics
        """
        try:
            # Ensure we have valid predictions and targets
            if pchembl_predictions is None or pchembl_targets is None:
                self.logger.warning("pChEMBL predictions or targets are None")
                return {}
            
            # Convert to numpy if needed
            if isinstance(pchembl_predictions, torch.Tensor):
                pchembl_predictions = pchembl_predictions.detach().cpu().numpy()
            if isinstance(pchembl_targets, torch.Tensor):
                pchembl_targets = pchembl_targets.detach().cpu().numpy()
            
            # Remove any NaN values
            valid_mask = ~(np.isnan(pchembl_predictions) | np.isnan(pchembl_targets))
            if not valid_mask.any():
                self.logger.warning("No valid pChEMBL predictions/targets found (all NaN)")
                return {}
            
            valid_preds = pchembl_predictions[valid_mask]
            valid_true = pchembl_targets[valid_mask]
            pred_raw = valid_preds * (self.pchembl_std + 1e-8) + self.pchembl_mean
            true_raw = valid_true * (self.pchembl_std + 1e-8) + self.pchembl_mean
            mse  = mean_squared_error(valid_true, valid_preds)          # z-score space (keep if you like)
            mae  = mean_absolute_error(valid_true, valid_preds)
            rmse = np.sqrt(mse)
            mse_raw  = mean_squared_error(true_raw, pred_raw)           # original units (recommended)
            mae_raw  = mean_absolute_error(true_raw, pred_raw)
            rmse_raw = np.sqrt(mse_raw)            
            # Compute R² score (handle edge cases)
            try:
                r2 = r2_score(valid_true, valid_preds)
            except ValueError:
                r2 = float('nan')
            
            return {
                'pchembl_mse': mse,
                'pchembl_mae': mae,
                'pchembl_rmse': rmse,
                'pchembl_r2': r2,
                'pchembl_valid_count': len(valid_preds),
                'pchembl_mean_pred': np.mean(valid_preds),
                'pchembl_std_pred': np.std(valid_preds),
                'pchembl_mean_true': np.mean(valid_true),
                'pchembl_std_true': np.std(valid_true),
                'pchembl_mse_raw': mse_raw,
                'pchembl_mae_raw': mae_raw,
                'pchembl_rmse_raw': rmse_raw
            }
            
        except Exception as e:
            self.logger.error(f"Error computing pChEMBL metrics: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {}
        
    def model_training(self):
        """Execute the model training process."""
        self.logger.info(f"Starting training process for run: {self.run_name}")
        
        try:
            # Set trainable components based on args
            self.model.update_trainable_components(
                trainable_encoder=self.model_config['train_encoder_model'],
                trainable_decoder=self.model_config['train_decoder_model'],
                trainable_pchembl_head=self.model_config['train_pchembl_head']
            )
            
            # Load pretrained weights if specified (for fine-tuning)
            if self.training_config['load_pretrained_model']:
                self._load_pretrained_weights(self.training_config['load_pretrained_model'])

            # --- Load Pre-processed Dataset ---
            # NOTE: Run preprocess_dataset.py BEFORE training to create the cache!
            cache_dir = os.environ.get('DATASETS_CACHE_DIR', "/gpfs/projects/etur29/atabey/datasets")
            dataset_name = self.selfies_path.split("/")[-1].split(".")[0]
            processed_data_path = os.path.join(cache_dir, dataset_name)
            
            # Check if preprocessed data exists
            if not os.path.exists(processed_data_path):
                error_msg = (
                    f"\n{'='*80}\n"
                    f"ERROR: Preprocessed dataset not found!\n"
                    f"{'='*80}\n"
                    f"Expected location: {processed_data_path}\n\n"
                    f"Please run preprocessing BEFORE training:\n\n"
                    f"  sbatch preprocess_job.sh\n\n"
                    f"Or manually:\n\n"
                    f"  python preprocess_dataset.py \\\n"
                    f"    --selfies_path {self.selfies_path} \\\n"
                    f"    --prot_emb_model {self.model_config['prot_emb_model']} \\\n"
                    f"    --max_mol_len {self.model_config['max_mol_len']} \\\n"
                    f"    --prot_max_length {self.model_config['prot_max_length']}\n\n"
                    f"After preprocessing completes, rerun training.\n"
                    f"{'='*80}\n"
                )
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            # All ranks load the processed dataset from the shared location
            self.logger.info(f"Rank {self.global_rank}: Loading pre-processed dataset from {processed_data_path}")
            dataset = load_from_disk(processed_data_path)

            # --- End of Caching Logic ---
            
            # Split dataset
            self.logger.info(f"Splitting dataset into train and test sets...")
            dataset = dataset["train"].train_test_split(test_size=0.01, seed=42)
            self.train_data = dataset["train"]
            self.test_data = dataset["test"]
            
            self.logger.info(f"Dataset split: {len(self.train_data)} train, {len(self.test_data)} test samples")
            
            # Cache canonical SMILES for both training and evaluation references
            self.logger.info("Caching canonical SMILES for metrics...")
            self.train_smiles_list = self._extract_smiles_list(self.train_data, drop_invalid=True)
            self.eval_reference_smiles = self._extract_smiles_list(self.test_data, drop_invalid=True)
            if not self.train_smiles_list:
                self.logger.warning("Training data does not contain valid SMILES entries")
            if not self.eval_reference_smiles:
                self.logger.warning("Evaluation data does not contain valid SMILES entries")
            
            self.logger.info(f"Dataset loaded with {len(self.train_data)} training examples")
            
            # Initialize wandb on rank 0 only
            if self.global_rank == 0:
                try:
                    wandb.init(
                        project="prot2mol",
                        name=self.run_name,
                        config={
                            **self.model_config,
                            **self.training_config,
                            'dataset_name': self.run_name.split('_')[0],
                            'global_rank': self.global_rank,
                            'local_rank': self.local_rank
                        }
                    )
                    self.logger.info("Wandb initialized successfully on rank 0")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize wandb on rank 0: {e}")
            
            # Set model to training mode
            self.model.train()
            
            # Configure training arguments
            self.logger.info("Configuring training arguments...")
            
            # Don't overwrite output directory if resuming from checkpoint
            overwrite_output = self.training_config['resume_from_checkpoint'] is None
            
            training_args = TrainingArguments(
                run_name=self.run_name,
                output_dir=self.pretrain_save_to,
                overwrite_output_dir=overwrite_output,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                num_train_epochs=self.training_config['epochs'],
                learning_rate=self.training_config['learning_rate'],
                weight_decay=self.training_config['weight_decay'],
                per_device_train_batch_size=self.training_config['train_batch_size'],
                per_device_eval_batch_size=self.training_config['valid_batch_size'],
                gradient_accumulation_steps=self.training_config['gradient_accumulation_steps'],
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                disable_tqdm=True,
                logging_steps=1,
                dataloader_num_workers=self.training_config['dataloader_num_workers'],
                fp16=True,
                remove_unused_columns=False,
                include_inputs_for_metrics=False,
                save_safetensors=False,
                local_rank=self.local_rank,
                ddp_backend="nccl",
                ddp_find_unused_parameters=True
            )
            
            # Determine training mode based on configuration
            pchembl_only_training = (
                not self.model_config['train_encoder_model'] and 
                not self.model_config['train_decoder_model'] and 
                self.model_config['train_pchembl_head']
            )
            
            if pchembl_only_training:
                self.logger.info("Training mode: pChEMBL head only (Stage 1)")
            else:
                self.logger.info("Training mode: Full training (Stage 2)")

            # Initialize trainer
            self.logger.info("Initializing trainer...")
            self.trainer = GPT2_w_crs_attn_Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_data,
                eval_dataset=self.test_data,
                compute_metrics=self.compute_metrics,
                preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
                pchembl_only_mode=pchembl_only_training,
                ignore_mismatched_optimizer=self.training_config.get('ignore_mismatched_optimizer', False)
            )
            
            self.logger.info(f"Building trainer on device: {training_args.device} with {training_args.n_gpu} GPUs")
            
            # Execute training
            if self.training_config['resume_from_checkpoint']:
                self.logger.info(f"Resuming training from checkpoint: {self.training_config['resume_from_checkpoint']}")
                self.trainer.train(resume_from_checkpoint=self.training_config['resume_from_checkpoint'])
            else:
                self.logger.info("Starting training from scratch...")
                self.trainer.train()
            self.logger.info("Training finished successfully")
            
            # Evaluate model
            self.logger.info("Evaluating model...")
            eval_results = self.trainer.evaluate()
            perplexity = math.exp(eval_results['eval_loss'])
            self.logger.info(f"Perplexity: {perplexity:.2f}")
            
            # Save model
            self.logger.info(f"Saving model to {self.pretrain_save_to}")
            self.trainer.save_model(self.pretrain_save_to)
            self.logger.info("Model saved successfully")
            
            # Finish wandb session on rank 0
            if self.global_rank == 0:
                try:
                    wandb.finish()
                    self.logger.info("Wandb session finished successfully")
                except Exception as e:
                    self.logger.warning(f"Failed to finish wandb session: {e}")
            
            return eval_results
            
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}", exc_info=True)
            # Cleanup wandb on rank 0 if there's an error
            if self.global_rank == 0:
                try:
                    wandb.finish()
                    self.logger.info("Wandb session finished due to error")
                except Exception as wandb_error:
                    self.logger.warning(f"Failed to finish wandb session during error cleanup: {wandb_error}")
            raise

def parse_arguments():
    """Parse and validate command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Train a Prot2Mol model for protein-to-molecule generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data configuration
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument(
        "--selfies_path",
        default="./data/papyrus/prot_comp_set_pchembl_6_protlen_1000_human_False.csv",
        help="Path to the SELFIES dataset"
    )

    # Model architecture
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument(
        "--prot_emb_model",
        default="saprot",
        choices=["prot_t5", "esm2", "saprot"],
        help="Protein embedding model to use"
    )
    model_group.add_argument(
        "--n_layer",
        type=int,
        default=1,
        help="Number of transformer layers"
    )
    model_group.add_argument(
        "--n_head",
        type=int,
        default=16,
        help="Number of attention heads"
    )
    model_group.add_argument(
        "--n_emb",
        type=int,
        default=1024,
        help="Embedding dimension"
    )
    model_group.add_argument(
        "--max_mol_len",
        type=int,
        default=256,
        help="Maximum molecule sequence length"
    )
    model_group.add_argument(
        "--prot_max_length",
        type=int,
        default=1024,
        help="Maximum protein sequence length"
    )

    # Training parameters
    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument(
        "--epoch",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    training_group.add_argument(
        "--learning_rate",
        type=float,
        default=1.0e-5,
        help="Learning rate for training"
    )
    training_group.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size for training"
    )
    training_group.add_argument(
        "--valid_batch_size",
        type=int,
        default=4,
        help="Batch size for validation"
    )
    training_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps"
    )
    training_group.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers"
    )
    training_group.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for optimization"
    )
    training_group.add_argument(
        "--train_encoder_model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to train the protein encoder model."
    )
    training_group.add_argument(
        "--train_decoder_model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to train the molecule decoder model."
    )
    training_group.add_argument(
        "--train_pchembl_head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to train the pChEMBL prediction head."
    )
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        "--save_dir",
        default="/gpfs/projects/etur29/atabey/saved_models",
        help="Directory to save trained models"
    )
    output_group.add_argument(
        "--run_name_suffix",
        type=str,
        default=None,
        help="Optional override for the final component of the save directory name"
    )
    output_group.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )
    output_group.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from (continues training state)"
    )
    output_group.add_argument(
        "--load_pretrained_model",
        type=str,
        default=None,
        help="Path to pretrained model to load weights from (for fine-tuning, resets training state)"
    )
    output_group.add_argument(
        "--ignore_mismatched_optimizer",
        action="store_true",
        default=False,
        help="Skip loading optimizer state if it doesn't match the model (useful when architecture changed)"
    )

    return parser.parse_args()

def _resolve_run_suffix(config) -> str:
    """Determine the suffix used for naming run directories.
    
    Preference order:
        1. User provided --run_name_suffix CLI argument
        2. PROT2MOL_RUN_ID environment variable override
        3. Cluster/job identifiers (e.g., SLURM, PBS, LSF, TORCHELASTIC)
        4. Current date (YYYYMMDD)
    """
    if getattr(config, "run_name_suffix", None):
        return config.run_name_suffix
    
    manual_env_override = os.environ.get("PROT2MOL_RUN_ID")
    if manual_env_override:
        return manual_env_override
    
    job_id_candidates = [
        os.environ.get("SLURM_JOB_ID"),
        os.environ.get("PBS_JOBID"),
        os.environ.get("LSB_JOBID"),
        os.environ.get("JOB_ID"),
        os.environ.get("TORCHELASTIC_RUN_ID"),
    ]
    
    normalized_job_id = None
    for candidate in job_id_candidates:
        if candidate and candidate.lower() not in {"default", "none"}:
            normalized_job_id = candidate
            break
    
    date_component = datetime.datetime.now().strftime("%Y%m%d")
    if normalized_job_id:
        return f"{date_component}_{normalized_job_id}"
    
    return date_component

def validate_and_process_paths(config):
    """Validate input paths and process dataset information.
    
    Args:
        config: Parsed command line arguments
        
    Returns:
        str: Dataset name extracted from the path
        
    Raises:
        FileNotFoundError: If the SELFIES dataset path doesn't exist
        FileNotFoundError: If the checkpoint path doesn't exist when resuming
        FileNotFoundError: If the pretrained model path doesn't exist when loading
    """
    # Validate SELFIES dataset path
    if not os.path.exists(config.selfies_path):
        raise FileNotFoundError(f"SELFIES dataset not found at: {config.selfies_path}")
    
    # Validate checkpoint path if resuming
    if config.resume_from_checkpoint:
        if not os.path.exists(config.resume_from_checkpoint):
            raise FileNotFoundError(f"Checkpoint directory not found at: {config.resume_from_checkpoint}")
        
        # Check if it's a valid checkpoint directory (should contain trainer_state.json)
        trainer_state_path = os.path.join(config.resume_from_checkpoint, "trainer_state.json")
        if not os.path.exists(trainer_state_path):
            raise FileNotFoundError(f"Invalid checkpoint directory: {config.resume_from_checkpoint}. "
                                   f"Missing trainer_state.json file.")
    
    # Validate pretrained model path if loading for fine-tuning
    if config.load_pretrained_model:
        if not os.path.exists(config.load_pretrained_model):
            raise FileNotFoundError(f"Pretrained model directory not found at: {config.load_pretrained_model}")
    
    # Extract dataset name from path
    dataset_name = os.path.splitext(os.path.basename(config.selfies_path))[0]
    
    # Create necessary directories
    os.makedirs(config.save_dir, exist_ok=True)
    
    return dataset_name

def create_run_name(config, dataset_name):
    """Create a unique run name based on configuration parameters.
    
    Args:
        config: Parsed command line arguments
        dataset_name: Name of the dataset being used
        
    Returns:
        str: Unique run name for this training session
    """
    # Create base run name
    run_components = [
        dataset_name,
        f"emb_{config.prot_emb_model}",
        f"enc_{config.train_encoder_model}",
        f"dec_{config.train_decoder_model}",
        f"pchembl_{config.train_pchembl_head}",
        f"n_layer_{config.n_layer}",
        f"n_head_{config.n_head}",
        f"n_emb_{config.n_emb}",
        f"max_mol_len_{config.max_mol_len}",
        f"prot_max_length_{config.prot_max_length}",
        f"lr_{config.learning_rate}",
        f"bs_{config.train_batch_size}"
    ]
    

    
    # Add model architecture info
    run_components.append(f"layers_{config.n_layer}")
    run_components.append(f"heads_{config.n_head}")
    
    # Create suffix that is shared across distributed processes
    run_suffix = _resolve_run_suffix(config)
    
    # Combine all components
    run_name = "_".join(run_components + [run_suffix])
    
    return run_name

def setup_logging(log_level):
    """Configure logging for the training process.
    
    Args:
        log_level: The logging level to use
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'training_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=handlers
    )
    
    # Reduce verbosity of transformers, datasets, and other libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

def main():
    """Main entry point for training."""
    
    config = parse_arguments()
    
    # Set up logging
    setup_logging(config.log_level)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Prot2Mol training")
    
    # Validate paths and create directories
    dataset_name = validate_and_process_paths(config)
    run_name = create_run_name(config, dataset_name)
    
    save_dir = os.path.join(config.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize and run training
    trainer = TrainingScript(
        config=config,
        selfies_path=config.selfies_path,
        pretrain_save_to=save_dir,
        dataset_name=dataset_name,
        run_name=run_name,
    )
    trainer.ddp_setup()
    trainer.model_training()
    destroy_process_group()

if __name__ == "__main__":
    main()


              
