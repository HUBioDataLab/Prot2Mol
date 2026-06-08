import argparse
import datetime
import hashlib
import os
import logging
import re
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
        default="single_gpu",
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
        "--training_stage",
        type=str,
        default="pchembl_only",
        choices=["lm_only", "pchembl_only", "multitask"],
        help="Objective family to optimize during training.",
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
    training_group.add_argument(
        "--pchembl_tf_hidden_dim",
        type=int,
        default=768,
        help="Hidden dimension used by the FusionDTI-style token-fusion pChEMBL head.",
    )
    training_group.add_argument(
        "--pchembl_tf_num_heads",
        type=int,
        default=8,
        help="Number of attention heads in the token-fusion pChEMBL head.",
    )
    training_group.add_argument(
        "--pchembl_tf_group_size",
        type=int,
        default=1,
        help="Token grouping size used before token fusion in the pChEMBL head.",
    )
    training_group.add_argument(
        "--pchembl_tf_agg_mode",
        type=str,
        default="mean",
        choices=["cls", "mean", "mean_all_tok"],
        help="Aggregation mode for fused protein and molecule tokens.",
    )
    training_group.add_argument(
        "--pchembl_tf_dropout",
        type=float,
        default=0.1,
        help="Dropout applied in the FusionDTI-style pChEMBL regression MLP.",
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


def _slugify_run_component(value, max_length=None):
    """Normalize a run-name component for filesystem-safe directory names."""
    text = str(value).strip().replace(os.sep, "-")
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("._-")
    if not text:
        text = "na"
    if max_length is not None and len(text) > max_length:
        text = text[:max_length].rstrip("._-")
    return text or "na"


def create_run_name(config, dataset_name):
    """Create a compact unique run name that stays below filesystem limits."""
    run_suffix = _resolve_run_suffix(config)
    full_descriptor = "|".join(
        [
            dataset_name,
            str(config.prot_emb_model),
            str(config.training_stage),
            str(config.train_encoder_model),
            str(config.train_decoder_model),
            str(config.train_pchembl_head),
            str(config.stop_pchembl_gradients),
            str(config.pchembl_tf_hidden_dim),
            str(config.pchembl_tf_num_heads),
            str(config.pchembl_tf_group_size),
            str(config.pchembl_tf_agg_mode),
            str(config.n_layer),
            str(config.n_head),
            str(config.n_emb),
            str(config.max_mol_len),
            str(config.prot_max_length),
            str(config.learning_rate),
            str(config.train_batch_size),
            str(config.training_mode),
            str(run_suffix),
        ]
    )
    run_hash = hashlib.sha1(full_descriptor.encode("utf-8")).hexdigest()[:10]

    run_components = [
        _slugify_run_component(dataset_name, max_length=40),
        f"emb-{_slugify_run_component(config.prot_emb_model, max_length=12)}",
        f"stg-{_slugify_run_component(config.training_stage, max_length=16)}",
        f"enc{int(bool(config.train_encoder_model))}",
        f"dec{int(bool(config.train_decoder_model))}",
        f"pc{int(bool(config.train_pchembl_head))}",
        f"spg{int(bool(config.stop_pchembl_gradients))}",
        f"tf{config.pchembl_tf_hidden_dim}h{config.pchembl_tf_num_heads}g{config.pchembl_tf_group_size}-{_slugify_run_component(config.pchembl_tf_agg_mode, max_length=10)}",
        f"L{config.n_layer}",
        f"H{config.n_head}",
        f"E{config.n_emb}",
        f"ml{config.max_mol_len}",
        f"pl{config.prot_max_length}",
        f"lr{_slugify_run_component(config.learning_rate, max_length=12)}",
        f"bs{config.train_batch_size}",
        f"mode-{_slugify_run_component(config.training_mode, max_length=10)}",
        _slugify_run_component(run_suffix, max_length=24),
        run_hash,
    ]
    run_name = "_".join(run_components)
    if len(run_name) <= 200:
        return run_name

    fallback_components = [
        _slugify_run_component(dataset_name, max_length=24),
        f"emb-{_slugify_run_component(config.prot_emb_model, max_length=10)}",
        f"stg-{_slugify_run_component(config.training_stage, max_length=12)}",
        f"enc{int(bool(config.train_encoder_model))}",
        f"dec{int(bool(config.train_decoder_model))}",
        f"pc{int(bool(config.train_pchembl_head))}",
        f"tf{config.pchembl_tf_hidden_dim}h{config.pchembl_tf_num_heads}",
        f"lr{_slugify_run_component(config.learning_rate, max_length=10)}",
        f"bs{config.train_batch_size}",
        _slugify_run_component(run_suffix, max_length=20),
        run_hash,
    ]
    return "_".join(fallback_components)


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
        force=True,
        handlers=handlers,
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
