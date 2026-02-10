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
from typing import List, Dict, Optional, Tuple

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import GenerationConfig

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prot2mol.model import Prot2MolModel
from prot2mol.protein_encoders import get_protein_tokenizer, format_protein_sequence
from prot2mol.hf_utils import load_molgen_tokenizer
from prot2mol.utils import metrics_calculation, canonicalize_smiles_list, decode_selfies_list
import selfies as sf
from rdkit import RDLogger

# Suppress warnings and logs
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

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
        
        # Load model and tokenizers
        self._load_components()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f'molecule_generation_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def _load_components(self):
        """Load tokenizers and model."""
        self.logger.info("Loading tokenizers and model...")
        
        # Load molecule tokenizer
        self.logger.info("Loading molecule tokenizer...")
        models_base = os.environ.get('MODELS_BASE_PATH', '/home/hu/hu544211/Prot2Mol/models')
        self.mol_tokenizer = load_molgen_tokenizer(models_base=models_base, padding_side="left")
        
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
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model state dict
        with tqdm(desc=f"Loading model weights from {os.path.basename(model_path)}", unit="MB") as pbar:
            model_state = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=self.device)
            pbar.update(1)
            
        # Create model config
        model_config = {
            'prot_emb_model': self.config.prot_emb_model,
            'n_layer': getattr(self.config, 'n_layer', 1),
            'n_head': getattr(self.config, 'n_head', 16),
            'n_emb': getattr(self.config, 'n_emb', 1024),
            'max_mol_len': getattr(self.config, 'max_mol_len', 256),
            'prot_max_length': getattr(self.config, 'prot_max_length', 1024),
            'train_encoder_model': False,
            'mol_tokenizer': self.mol_tokenizer
        }
        
        # Create and load model
        model = Prot2MolModel(model_config)
        model.load_state_dict(model_state, strict=False)
        model.to(self.device)
        return model
    
    def _load_prediction_data(self) -> pd.DataFrame:
        """
        Load data for prediction mode.
        """
        self.logger.info(f"Loading molecules for prediction from {self.config.input_molecules}")
        
        if not os.path.exists(self.config.input_molecules):
             raise FileNotFoundError(f"Input molecules file not found: {self.config.input_molecules}")
             
        df = pd.read_csv(self.config.input_molecules)
        
        # Normalize column names lower case for check
        cols = [c.lower() for c in df.columns]
        
        if 'smiles' not in cols and 'selfies' not in cols and 'compound_smiles' not in cols and 'compound_selfies' not in cols and 'generated_selfies' not in cols and 'generated_smiles' not in cols:
             # Try to find a column that might contain molecules
             potential_cols = [c for c in df.columns if 'smiles' in c.lower() or 'selfies' in c.lower()]
             if not potential_cols:
                  raise ValueError("Input file must contain a column with SMILES or SELFIES (e.g., 'smiles', 'Compound_SMILES', 'selfies', 'Compound_SELFIES', 'Generated_SELFIES')")
        
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
        
        # Add SELFIES alphabet to tokenizer from both datasets
        all_selfies = []
        if 'Compound_SELFIES' in train_data.columns:
            all_selfies.extend(train_data['Compound_SELFIES'].tolist())
        if 'Compound_SELFIES' in test_data.columns:
            all_selfies.extend(test_data['Compound_SELFIES'].tolist())
        
        if all_selfies:
            self.logger.info("Adding SELFIES alphabet to tokenizer...")
            alphabet = list(sf.get_alphabet_from_selfies(all_selfies))
            self.mol_tokenizer.add_tokens(alphabet)
            self.logger.info(f"Added {len(alphabet)} SELFIES tokens to tokenizer")
        
        self.logger.info(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples")
        
        return train_data, train_vec, test_data
    
    def _get_protein_embeddings(self, protein_sequence: str) -> torch.Tensor:
        """
        Get protein embeddings for a given sequence.
        
        Args:
            protein_sequence: Protein sequence string
            
        Returns:
            Protein embeddings tensor
        """
        # Prepare sequence for tokenization
        formatted_sequence = format_protein_sequence(protein_sequence, self.config.prot_emb_model)
        
        # Tokenize protein sequence
        prot_tokens = self.prot_tokenizer.encode_plus(
            formatted_sequence,
            add_special_tokens=True,
            max_length=self.config.prot_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return prot_tokens['input_ids'].to(self.device), prot_tokens['attention_mask'].to(self.device)
    
    def _generate_molecules_batch(self, prot_input_ids: torch.Tensor, prot_attention_mask: torch.Tensor, 
                                 num_samples: int) -> List[str]:
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
        
        # Generate in batches to manage memory
        batch_size = min(self.config.batch_size, num_samples)
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Generating molecules"):
                current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
                
                # Repeat protein embeddings for the batch
                batch_prot_input_ids = prot_input_ids.repeat(current_batch_size, 1)
                batch_prot_attention_mask = prot_attention_mask.repeat(current_batch_size, 1)
                
                # Generate molecules
                try:
                    generated_tokens = self.generation_model.generate(
                        prot_input_ids=batch_prot_input_ids,
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
                    
                except Exception as e:
                    self.logger.error(f"Error generating batch {batch_idx}: {str(e)}")
                    # Add empty strings for failed generations
                    generated_molecules.extend([''] * current_batch_size)
        
        return generated_molecules
    
    def _predict_pchembl_batch(self, prot_input_ids: torch.Tensor, prot_attention_mask: torch.Tensor, 
                               mol_input_ids: torch.Tensor, mol_attention_mask: torch.Tensor) -> np.ndarray:
        """
        Predict pChEMBL values for a batch of molecules.
        """
        with torch.no_grad():
            outputs = self.prediction_model(
                mol_input_ids=mol_input_ids,
                prot_input_ids=prot_input_ids,
                prot_attention_mask=prot_attention_mask,
                train_lm=False # No need to calculate LM loss during inference
            )
            predictions = outputs['pchembl_predictions'].cpu().numpy()
            
            # Denormalize
            denormalized_preds = predictions * self.config.pchembl_std + self.config.pchembl_mean
            return denormalized_preds

    def predict_pchembl(self, protein_sequence: str, molecules_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict pChEMBL values for a dataframe of molecules against a target protein.
        """
        self.logger.info(f"Predicting pChEMBL for {len(molecules_df)} molecules...")
        
        # Identify molecule column
        mol_col = None
        is_selfies = False
        
        # Prioritize SELFIES
        for col in molecules_df.columns:
            if 'selfies' in col.lower():
                mol_col = col
                is_selfies = True
                break
        
        if not mol_col:
            for col in molecules_df.columns:
                if 'smiles' in col.lower():
                    mol_col = col
                    break
        
        if not mol_col:
            raise ValueError("Could not find SMILES or SELFIES column")
            
        # Get protein embeddings (prepare once)
        prot_input_ids, prot_attention_mask = self._get_protein_embeddings(protein_sequence)
        
        # Batch processing
        batch_size = self.config.batch_size
        all_preds = []
        
        # Convert to SELFIES if strings are SMILES
        molecules_list = molecules_df[mol_col].tolist()
        selfies_list = []
        
        if not is_selfies:
            self.logger.info("Converting SMILES to SELFIES for prediction...")
            for smi in tqdm(molecules_list, desc="SMILES->SELFIES"):
                try:
                    s = sf.encoder(smi)
                    selfies_list.append(s if s else "[nop]")
                except:
                    selfies_list.append("[nop]")
        else:
            selfies_list = molecules_list

        # Run prediction in batches
        num_batches = (len(selfies_list) + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Predicting pChEMBL"):
            batch_selfies = selfies_list[i*batch_size : (i+1)*batch_size]
            current_batch_len = len(batch_selfies)
            
            # Tokenize molecules
            mol_tokens = self.mol_tokenizer.batch_encode_plus(
                batch_selfies,
                add_special_tokens=True,
                truncation=True,
                max_length=self.config.max_mol_len,
                padding='max_length',
                return_tensors='pt'
            )
            
            batch_mol_ids = mol_tokens['input_ids'].to(self.device)
            batch_mol_mask = mol_tokens['attention_mask'].to(self.device)
            
            # Expand protein to batch size
            batch_prot_ids = prot_input_ids.repeat(current_batch_len, 1)
            batch_prot_mask = prot_attention_mask.repeat(current_batch_len, 1)
            
            # Predict
            preds = self._predict_pchembl_batch(batch_prot_ids, batch_prot_mask, batch_mol_ids, batch_mol_mask)
            all_preds.extend(preds)
            
        # Add predictions to dataframe
        molecules_df['Predicted_pChEMBL'] = all_preds
        return molecules_df

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
        
        # Get protein embeddings
        prot_input_ids, prot_attention_mask = self._get_protein_embeddings(protein_sequence)
        
        # Generate molecules
        generated_selfies = self._generate_molecules_batch(
            prot_input_ids, prot_attention_mask, num_samples
        )
        
        # Calcuate pChEMBL for generated molecules
        self.logger.info("Predicting pChEMBL for generated molecules...")
        
        # Reuse prediction batch logic but we already have SELFIES
        # Prepare batches
        batch_size = self.config.batch_size
        all_preds = []
        
        num_batches = (len(generated_selfies) + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Predicting pChEMBL"):
            batch_selfies = generated_selfies[i*batch_size : (i+1)*batch_size]
            current_batch_len = len(batch_selfies)
            
            # Handle empty/invalid generation by replacing with padding token equivalent or simple "[nop]"
            # But tokenizer will handle strings.
            clean_batch = [s if s else "[nop]" for s in batch_selfies]
            
            mol_tokens = self.mol_tokenizer.batch_encode_plus(
                clean_batch,
                add_special_tokens=True,
                truncation=True,
                max_length=self.config.max_mol_len,
                padding='max_length',
                return_tensors='pt'
            )
            
            batch_mol_ids = mol_tokens['input_ids'].to(self.device)
            batch_mol_mask = mol_tokens['attention_mask'].to(self.device)
             # Expand protein to batch size
            batch_prot_ids = prot_input_ids.repeat(current_batch_len, 1)
            batch_prot_mask = prot_attention_mask.repeat(current_batch_len, 1)
            
            preds = self._predict_pchembl_batch(batch_prot_ids, batch_prot_mask, batch_mol_ids, batch_mol_mask)
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
        
        try:
            metrics, results_df = metrics_calculation(
                predictions=generated_df['Generated_SELFIES'].tolist(),
                references=reference_smiles,
                train_data=self.train_data,
                train_vec=self.train_vec,
                training=False
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
            if 'sa_score' in results_df.columns:
                generated_df['SA_Score'] = results_df['sa_score'].tolist()
            if 'qed_score' in results_df.columns:
                generated_df['QED_Score'] = results_df['qed_score'].tolist()
            if 'logp_score' in results_df.columns:
                generated_df['LogP_Score'] = results_df['logp_score'].tolist()
            if 'test_sim' in results_df.columns:
                generated_df['Test_Similarity'] = results_df['test_sim'].tolist()
            if 'train_sim' in results_df.columns:
                generated_df['Train_Similarity'] = results_df['train_sim'].tolist()
            
            self.logger.info("Metrics calculated successfully")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"{key}: {value:.4f}")
                else:
                    self.logger.info(f"{key}: {value}")
            
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
        
        # Verify both SELFIES and SMILES are present
        has_selfies = 'Generated_SELFIES' in generated_df.columns
        has_smiles = 'Generated_SMILES' in generated_df.columns
        
        if has_selfies and has_smiles:
            self.logger.info("✅ Both SELFIES and SMILES representations will be saved")
            # Count valid molecules
            valid_selfies = generated_df['Generated_SELFIES'].notna().sum()
            valid_smiles = (generated_df['Generated_SMILES'].notna() & 
                          (generated_df['Generated_SMILES'] != "")).sum()
            self.logger.info(f"📊 Valid SELFIES: {valid_selfies}/{len(generated_df)}")
            self.logger.info(f"📊 Valid SMILES: {valid_smiles}/{len(generated_df)}")
        elif has_selfies:
            self.logger.warning("⚠️ Only SELFIES representation available")
        elif has_smiles:
            self.logger.warning("⚠️ Only SMILES representation available")
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
             
             # Determine Protein ID if not provided
             if not self.config.prot_id:
                 if 'Protein_ID' in molecules_df.columns:
                     self.config.prot_id = molecules_df.iloc[0]['Protein_ID']
                     self.logger.info(f"Inferred Protein ID from input file: {self.config.prot_id}")
                 elif 'Target_CHEMBL_ID' in molecules_df.columns:
                     self.config.prot_id = molecules_df.iloc[0]['Target_CHEMBL_ID']
                     self.logger.info(f"Inferred Protein ID from input file: {self.config.prot_id}")
             
             protein_sequence = None
             
             # Strategy 1: Look up FASTA from dataset files using prot_id
             if self.config.prot_id:
                 try:
                     self.logger.info(f"Attempting to fetch sequence for {self.config.prot_id} from dataset...")
                     # _load_dataset uses self.config.prot_id to filter
                     _, _, test_data = self._load_dataset()
                     if len(test_data) > 0:
                         protein_sequence = test_data.iloc[0]['Target_FASTA']
                         self.logger.info("✅ Successfully retrieved protein sequence from dataset")
                 except Exception as e:
                     self.logger.warning(f"Could not load protein sequence from dataset files: {e}")
            
             # Strategy 2: Look for FASTA in the input dataframe
             if not protein_sequence:
                 if 'Target_FASTA' in molecules_df.columns:
                     protein_sequence = molecules_df.iloc[0]['Target_FASTA']
                     self.logger.info("✅ Non-standard: Found protein sequence in input file")
                 else:
                     raise ValueError(f"Could not find protein sequence! \n"
                                      f"1. Ensure 'prot_id' is provided or in the input file ('Protein_ID').\n"
                                      f"2. Ensure the dataset at '{self.config.selfies_path}' contains data for this protein.\n"
                                      f"3. Or provide 'Target_FASTA' in the input file.")

             # Step 2: Predict
             results_df = self.predict_pchembl(protein_sequence, molecules_df)
             
             # Step 3: Save
             self.save_results(results_df, {}, self.config.output_file)
             
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
                overall_pbar.update(1)
                
                overall_pbar.set_description("✅ Pipeline completed")
            
            self.logger.info("Molecule generation pipeline completed successfully!")
            
            return generated_df, metrics


def parse_arguments() -> argparse.Namespace:
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
        "--prot_id",
        required=False,
        help="Target protein CHEMBL ID (required for generation, optional for prediction if provided in input file)"
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
    
    return parser.parse_args()


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
