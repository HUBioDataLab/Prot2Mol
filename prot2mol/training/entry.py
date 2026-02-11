import argparse
import datetime
import os
import logging
import sys

from ..io.config import parse_args_with_config


def parse_arguments(argv=None):
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Prot2Mol model for protein-to-molecule generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--selfies_path",
        default="./data/papyrus/prot_comp_set_pchembl_6_protlen_1000_human_False.csv",
        help="Path to the SELFIES dataset",
    )

    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument(
        "--prot_emb_model",
        default="saprot",
        choices=["prot_t5", "esm2", "saprot"],
        help="Protein embedding model to use",
    )
    model_group.add_argument("--n_layer", type=int, default=1, help="Number of transformer layers")
    model_group.add_argument("--n_head", type=int, default=16, help="Number of attention heads")
    model_group.add_argument("--n_emb", type=int, default=1024, help="Embedding dimension")
    model_group.add_argument("--max_mol_len", type=int, default=256, help="Maximum molecule sequence length")
    model_group.add_argument("--prot_max_length", type=int, default=1024, help="Maximum protein sequence length")

    training_group = parser.add_argument_group("Training Parameters")
    training_group.add_argument("--epoch", type=int, default=50, help="Number of training epochs")
    training_group.add_argument("--learning_rate", type=float, default=1.0e-5, help="Learning rate for training")
    training_group.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    training_group.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    training_group.add_argument(
        "--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps"
    )
    training_group.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of dataloader workers")
    training_group.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimization")
    training_group.add_argument(
        "--training_mode",
        type=str,
        default="auto",
        choices=["auto", "single_gpu", "multi_gpu", "multi_node"],
        help=(
            "Training execution mode. "
            "'auto' infers from launch environment; "
            "'single_gpu' runs one process; "
            "'multi_gpu' expects single-node torchrun; "
            "'multi_node' expects multi-node torchrun."
        ),
    )
    training_group.add_argument(
        "--eval_split",
        type=str,
        default="random",
        choices=["random", "aid"],
        help="Validation split strategy: random or AID hold-out",
    )
    training_group.add_argument(
        "--eval_split_ratio",
        type=float,
        default=0.01,
        help="Fraction of data (or AIDs) held out for validation",
    )
    training_group.add_argument("--split_seed", type=int, default=42, help="Random seed for data split reproducibility")
    training_group.add_argument(
        "--pchembl_huber_delta",
        type=float,
        default=1.0,
        help="Huber delta for pChEMBL loss (raw pChEMBL units)",
    )
    training_group.add_argument(
        "--train_encoder_model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to train the protein encoder model.",
    )
    training_group.add_argument(
        "--train_decoder_model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to train the molecule decoder model.",
    )
    training_group.add_argument(
        "--train_pchembl_head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to train the pChEMBL prediction head.",
    )
    training_group.add_argument(
        "--stop_pchembl_gradients",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If True, pChEMBL loss will not backpropagate into encoder/decoder.",
    )

    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--save_dir",
        default="/gpfs/projects/etur29/atabey/saved_models",
        help="Directory to save trained models",
    )
    output_group.add_argument(
        "--run_name_suffix",
        type=str,
        default=None,
        help="Optional override for the final component of the save directory name",
    )
    output_group.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level",
    )
    output_group.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from (continues training state)",
    )
    output_group.add_argument(
        "--load_pretrained_model",
        type=str,
        default=None,
        help="Path to pretrained model to load weights from (for fine-tuning, resets training state)",
    )
    output_group.add_argument(
        "--ignore_mismatched_optimizer",
        action="store_true",
        default=False,
        help="Skip loading optimizer state if it doesn't match the model (useful when architecture changed)",
    )

    return parse_args_with_config(parser, section="train", argv=argv)


def _resolve_run_suffix(config) -> str:
    """Determine the suffix used for naming run directories."""
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
    """Validate input paths and create output directories."""
    if not os.path.exists(config.selfies_path):
        raise FileNotFoundError(f"SELFIES dataset not found at: {config.selfies_path}")

    if config.resume_from_checkpoint:
        if not os.path.exists(config.resume_from_checkpoint):
            raise FileNotFoundError(f"Checkpoint directory not found at: {config.resume_from_checkpoint}")
        trainer_state_path = os.path.join(config.resume_from_checkpoint, "trainer_state.json")
        if not os.path.exists(trainer_state_path):
            raise FileNotFoundError(
                f"Invalid checkpoint directory: {config.resume_from_checkpoint}. Missing trainer_state.json file."
            )

    if config.load_pretrained_model and not os.path.exists(config.load_pretrained_model):
        raise FileNotFoundError(f"Pretrained model directory not found at: {config.load_pretrained_model}")

    dataset_name = os.path.splitext(os.path.basename(config.selfies_path))[0]
    os.makedirs(config.save_dir, exist_ok=True)
    return dataset_name


def create_run_name(config, dataset_name):
    """Create a unique run name based on configuration parameters."""
    run_components = [
        dataset_name,
        f"emb_{config.prot_emb_model}",
        f"enc_{config.train_encoder_model}",
        f"dec_{config.train_decoder_model}",
        f"pchembl_{config.train_pchembl_head}",
        f"stop_pchembl_grad_{config.stop_pchembl_gradients}",
        f"n_layer_{config.n_layer}",
        f"n_head_{config.n_head}",
        f"n_emb_{config.n_emb}",
        f"max_mol_len_{config.max_mol_len}",
        f"prot_max_length_{config.prot_max_length}",
        f"lr_{config.learning_rate}",
        f"bs_{config.train_batch_size}",
        f"mode_{config.training_mode}",
    ]
    run_components.append(f"layers_{config.n_layer}")
    run_components.append(f"heads_{config.n_head}")
    run_suffix = _resolve_run_suffix(config)
    return "_".join(run_components + [run_suffix])


def setup_logging(log_level):
    """Configure logging for the training process."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ]
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=handlers,
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
