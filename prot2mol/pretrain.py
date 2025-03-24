# Standard library imports
import os
import sys
import math
import argparse
import logging
import datetime
import re
from typing import Dict, List, Tuple, Any, Optional

# Third-party library imports
import numpy as np
import torch
import torch.distributed
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    BartTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    T5Tokenizer,
    AutoTokenizer
)

# Local application imports
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing import train_val_test
from prot2mol.trainer import GPT2_w_crs_attn_Trainer
from prot2mol.utils import metrics_calculation
from prot2mol.protein_encoders import get_protein_encoder, get_protein_tokenizer, get_encoder_size
from prot2mol.molecule_decoders import get_molecule_decoder

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "false"

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
        model: The GPT2 model for generation
        encoder_model: The protein encoder model
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
        
        # Organize configurations into logical groups
        self.model_config = {
            'prot_emb_model': config.prot_emb_model,
            'n_layer': config.n_layer,
            'n_head': config.n_head,
            'n_emb': config.n_emb,
            'max_mol_len': config.max_mol_len,
            'prot_max_length': config.prot_max_length,
            'train_encoder_model': config.train_encoder_model
        }
        
        self.training_config = {
            'train_batch_size': config.train_batch_size,
            'valid_batch_size': config.valid_batch_size,
            'epochs': config.epoch,
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay
        }

        self.selfies_path = selfies_path
        self.pretrain_save_to = pretrain_save_to
        self.run_name = run_name
        
        # Load training vectors for similarity calculation
        self._load_training_vectors()
        
        # Initialize tokenizers and models
        self._init_tokenizers()
        self._init_models()
        
        self.logger.info(f"Model parameter count: {self.model.num_parameters():,}")

    def _load_training_vectors(self):
        """Load training vectors for similarity calculation if available."""
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        train_vecs_path = os.path.join(data_dir, "train_vecs.npy")
        
        if not os.path.exists(train_vecs_path):
            self.logger.warning(f"Training vectors file not found at {train_vecs_path}")
            self.logger.warning("You need to generate train_vecs.npy to calculate training similarity")
            self.training_vec = None
        else:
            self.training_vec = np.load(train_vecs_path)

    def _init_tokenizers(self):
        """Initialize tokenizers for proteins and molecules."""
        self.logger.info("Initializing tokenizers...")
        self.mol_tokenizer = BartTokenizer.from_pretrained("zjunlp/MolGen-large", padding_side="left")
        self.prot_tokenizer = get_protein_tokenizer(self.model_config['prot_emb_model'])

    def _init_models(self):
        """Initialize the GPT2 model and protein encoder."""
        self.logger.info("Initializing models...")
        self.configuration = GPT2Config(
            add_cross_attention=True, 
            is_decoder=True,
            n_embd=get_encoder_size(self.model_config['prot_emb_model']), 
            n_head=self.model_config['n_head'], 
            vocab_size=len(self.mol_tokenizer.added_tokens_decoder), 
            n_positions=256, 
            n_layer=self.model_config['n_layer'], 
            bos_token_id=self.mol_tokenizer.bos_token_id,
            eos_token_id=self.mol_tokenizer.eos_token_id
        )
        self.model = GPT2LMHeadModel(self.configuration)
        self.encoder_model = get_protein_encoder(
            self.model_config['prot_emb_model'], 
            active = self.model_config['train_encoder_model']
        )

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
            # Tokenize the SELFIES strings
            ids = self.mol_tokenizer.batch_encode_plus(
                batch["Compound_SELFIES"], 
                add_special_tokens=True, 
                truncation=True,
                max_length=self.model_config['max_mol_len'],
                padding="max_length"
            )
            
            return {
                'mol_input_ids': ids['input_ids'],
                'mol_attention_mask': ids['attention_mask']
            }
        except Exception as e:
            self.logger.error(f"Error in molecule tokenization: {str(e)}")
            raise

    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics for the model.
        
        Args:
            eval_pred: Tuple of (logits, labels)
            
        Returns:
            dict: Dictionary with computed metrics
        """
        try:
            logits = eval_pred.predictions
            labels = eval_pred.label_ids    
            # Convert logits to predictions
            predictions = np.argmax(logits, axis=-1)
            # Replace -100 tokens with pad_token_id for decoding
            labels = np.where(labels != -100, labels, self.mol_tokenizer.pad_token_id)
            
            
            # Decode predictions and labels
            decoded_preds = self.mol_tokenizer.batch_decode(
                predictions, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )
            decoded_labels = self.mol_tokenizer.batch_decode(
                labels, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )
            # Calculate metrics - explicitly set training=True to ensure it returns only metrics dict
            return metrics_calculation(
                predictions=decoded_preds, 
                references=decoded_labels, 
                train_data=self.train_data, 
                train_vec=self.training_vec,
                training=True
            )
        except Exception as e:
            self.logger.error(f"Error computing metrics: {str(e)}")
            # Return empty dict to avoid breaking the training loop
            return {}

    def model_training(self):
        """Execute the model training process.
        
        This method:
        1. Loads and preprocesses the dataset
        2. Configures the training arguments
        3. Initializes the trainer
        4. Executes the training
        5. Evaluates the model
        6. Saves the model
        
        Returns:
            None
        """
        self.logger.info(f"Starting training process for run: {self.run_name}")
        
        try:
            # Load the dataset
            self.logger.info("Loading dataset...")
            dataset = load_dataset("csv", data_files=self.selfies_path)
            
            # Process protein sequences
            self.logger.info("Tokenizing protein sequences...")
            dataset = dataset.map(
                self.tokenize_prot_function,
                batched=True,
                num_proc=1,
                batch_size=100,
                desc="Tokenizing protein sequences"
            )
            
            # Process molecule sequences
            self.logger.info("Tokenizing molecule sequences...")
            dataset = dataset.map(
                self.tokenize_mol_function,
                batched=True,
                num_proc=1,
                batch_size=100,
                desc="Tokenizing molecule sequences"
            )
            
            # Split dataset
            self.logger.info("Splitting dataset into train and test sets...")
            dataset = dataset["train"].train_test_split(test_size=0.1)
            self.train_data = dataset["train"]
            self.test_data = dataset["test"]
            
            # Set model to training mode
            self.model.train()
            
            # Configure training arguments
            self.logger.info("Configuring training arguments...")
            training_args = TrainingArguments(
                run_name=self.run_name,
                output_dir=self.pretrain_save_to,
                overwrite_output_dir=True,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                num_train_epochs=self.training_config['epochs'],
                learning_rate=self.training_config['learning_rate'],
                weight_decay=self.training_config['weight_decay'],
                per_device_train_batch_size=self.training_config['train_batch_size'],
                per_device_eval_batch_size=self.training_config['valid_batch_size'],
                save_total_limit=1,
                disable_tqdm=True,
                logging_steps=10,
                dataloader_num_workers=10,
                fp16=True,
                ddp_find_unused_parameters=False,
                remove_unused_columns=False
            )
            
            # Initialize trainer
            self.logger.info("Initializing trainer...")
            trainer = GPT2_w_crs_attn_Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_data,
                eval_dataset=self.test_data,
                compute_metrics=self.compute_metrics,
                encoder_model=self.encoder_model,
                train_encoder_model=self.model_config['train_encoder_model'] #flag of training the encoder
            )
            trainer.args._n_gpu = 1
            
            self.logger.info(f"Building trainer on device: {training_args.device} with {training_args.n_gpu} GPUs")
            
            # Execute training
            self.logger.info("Starting training...")
            trainer.train()
            self.logger.info("Training finished successfully")
            
            # Evaluate model
            self.logger.info("Evaluating model...")
            eval_results = trainer.evaluate()
            perplexity = math.exp(eval_results['eval_loss'])
            self.logger.info(f"Perplexity: {perplexity:.2f}")
            
            # Save model
            self.logger.info(f"Saving model to {self.pretrain_save_to}")
            trainer.save_model(self.pretrain_save_to)
            self.logger.info("Model saved successfully")
            
            return eval_results
            
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}", exc_info=True)
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
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data/papyrus/papyrus_20data.csv"),
        help="Path to the SELFIES dataset"
    )
    data_group.add_argument(
        "--full_set",
        action="store_true",
        default=True,
        help="Use full dataset instead of single protein"
    )
    data_group.add_argument(
        "--prot_ID",
        default="CHEMBL4282",
        help="Protein ID for single protein training"
    )

    # Model architecture
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument(
        "--prot_emb_model",
        default="esm2",
        choices=["prot_t5", "esm2", "saprot"],
        help="Protein embedding model to use"
    )
    model_group.add_argument(
        "--mol_decoder_model",
        default="gpt",
        choices=["gpt", "chemgpt"],
        help="Molecule decoder model to use"
    )
    
    # Training parameters
    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument(
        "--learning_rate",
        type=float,
        default=1.0e-5,
        help="Learning rate for training"
    )
    training_group.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size for training"
    )
    training_group.add_argument(
        "--valid_batch_size",
        type=int,
        default=1,
        help="Batch size for validation"
    )
    training_group.add_argument(
        "--epoch",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    training_group.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for optimization"
    )
    training_group.add_argument(
        "--n_layer",
        type=int,
        default=2,
        help="Number of transformer layers"
    )
    training_group.add_argument(
        "--n_head",
        type=int,
        default=16,
        help="Number of attention heads"
    )
    training_group.add_argument(
        "--n_emb",
        type=int,
        default=1024,
        help="Embedding dimension"
    )
    training_group.add_argument(
        "--max_mol_len",
        type=int,
        default=256,
        help="Maximum molecule sequence length"
    )
    training_group.add_argument(
        "--prot_max_length",
        type=int,
        default=1024,
        help="Maximum protein sequence length"
    )
    training_group.add_argument(
        "--train_encoder_model",
        action="store_true",
        default=True,
        help="Whether to train the protein encoder model"
    )
    training_group.add_argument(
        "--train_decoder_model",
        action="store_true",
        default=False,
        help="Whether to train the molecule decoder model"
    )
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        "--save_dir",
        default="../saved_models",
        help="Directory to save trained models"
    )
    output_group.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )

    return parser.parse_args()

def validate_and_process_paths(config):
    """Validate input paths and process dataset information.
    
    Args:
        config: Parsed command line arguments
        
    Returns:
        str: Dataset name extracted from the path
        
    Raises:
        FileNotFoundError: If the SELFIES dataset path doesn't exist
    """
    # Validate SELFIES dataset path
    if not os.path.exists(config.selfies_path):
        raise FileNotFoundError(f"SELFIES dataset not found at: {config.selfies_path}")
    
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
        f"dec_{config.mol_decoder_model}",
        f"lr_{config.learning_rate}",
        f"bs_{config.train_batch_size}"
    ]
    
    # Add conditional components
    if not config.full_set:
        run_components.append(f"prot_{config.prot_ID}")
    
    # Add model architecture info
    run_components.append(f"layers_{config.n_layer}")
    run_components.append(f"heads_{config.n_head}")
    
    # Create timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Combine all components
    run_name = "_".join(run_components + [timestamp])
    
    return run_name

def setup_logging(log_level):
    """Configure logging for the training process.
    
    Args:
        log_level: The logging level to use
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'training_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    
    # Reduce verbosity of transformers, datasets, and other libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

def main():
    """Main entry point for training.
    
    This function:
    1. Sets up logging
    2. Parses command line arguments
    3. Validates paths and creates run name
    4. Initializes and runs the training
    
    Returns:
        int: 0 for success, 1 for failure
    """
    try:
        # Parse arguments
        config = parse_arguments()
        
        # Set up logging
        setup_logging(config.log_level)
        logger = logging.getLogger(__name__)
        logger.info("Starting Prot2Mol training")
        
        # Show system info
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Torch version: {torch.__version__}")
        
        # Validate paths and create run name
        logger.info("Validating paths and creating run name...")
        dataset_name = validate_and_process_paths(config)
        run_name = create_run_name(config, dataset_name)
        
        logger.info(f"Starting training run: {run_name}")
        
        # Set up save directory
        save_dir = os.path.join(config.save_dir, run_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize and run training
        trainer = TrainingScript(
            config=config,
            selfies_path=config.selfies_path,
            pretrain_save_to=save_dir,
            dataset_name=dataset_name,
            run_name=run_name
        )
        
        # Run training
        eval_results = trainer.model_training()
        
        # Log results
        logger.info("Training completed successfully!")
        logger.info(f"Final evaluation results: {eval_results}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())


              
