import json
import os
from typing import Dict, Optional, Sequence

import torch

MODEL_CONFIG_KEYS = (
    "prot_emb_model",
    "protein_model_id",
    "decoder_model_id",
    "conditioning_dropout",
    "max_mol_len",
    "prot_max_length",
    "train_encoder_model",
    "train_projection_model",
    "train_decoder_model",
    "dataset_name",
    "dataset_source_path",
    "dataset_total_samples",
    "train_samples",
    "eval_samples",
    "train_unique_proteins",
    "eval_unique_proteins",
    "train_unique_molecules",
    "eval_unique_molecules",
    "split_seed",
    "generation_eval_proteins",
    "generation_samples_per_protein",
)


def resolve_model_path(model_name: str, models_base: Optional[str] = None, fallback_bases: Optional[Sequence[str]] = None) -> str:
    """
    Resolve a local HuggingFace cache path for a given model name.

    The function prefers snapshot directories when present and falls back to the
    model directory. If no base is provided, it uses MODELS_BASE_PATH or ./models.
    """
    if os.path.exists(model_name):
        return model_name

    canonical_name = model_name
    if "/" not in canonical_name and "--" in canonical_name:
        namespace, repository = canonical_name.split("--", 1)
        canonical_name = f"{namespace}/{repository}"

    bases = []
    if models_base:
        bases.append(models_base)

    env_base = os.environ.get("MODELS_BASE_PATH")
    if env_base and env_base not in bases:
        bases.append(env_base)

    if fallback_bases:
        for base in fallback_bases:
            if base and base not in bases:
                bases.append(base)

    if not bases:
        bases.append("./models")

    for base in bases:
        base = os.path.expanduser(base)
        cache_name = canonical_name.replace("/", "--")
        model_dir = os.path.join(base, f"models--{cache_name}")
        snapshots_dir = os.path.join(model_dir, "snapshots")
        if os.path.isdir(snapshots_dir):
            main_ref = os.path.join(model_dir, "refs", "main")
            if os.path.isfile(main_ref):
                with open(main_ref, "r", encoding="utf-8") as handle:
                    revision = handle.read().strip()
                referenced_snapshot = os.path.join(snapshots_dir, revision)
                if revision and os.path.isdir(referenced_snapshot):
                    return referenced_snapshot
            snapshots = sorted(os.listdir(snapshots_dir))
            if snapshots:
                snapshots.sort(
                    key=lambda name: os.path.getmtime(os.path.join(snapshots_dir, name)),
                    reverse=True,
                )
                return os.path.join(snapshots_dir, snapshots[0])
        if os.path.isdir(model_dir):
            return model_dir

    # A nonexistent local-looking path prevents Hugging Face from recognizing a
    # valid repository id. Fall back to the canonical Hub id so HF_HOME and the
    # standard cache/download behavior work as intended.
    return canonical_name


def load_molgen_tokenizer(
    models_base: Optional[str] = None,
    fallback_bases: Optional[Sequence[str]] = None,
    padding_side: str = "right",
    model_id: str = "zjunlp/MolGen-large",
):
    """Load the tokenizer from the same MolGen checkpoint as the decoder."""
    from transformers import AutoTokenizer

    model_path = resolve_model_path(
        model_id,
        models_base=models_base,
        fallback_bases=fallback_bases,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = padding_side
    return tokenizer


def find_model_config_path(model_path: str) -> Optional[str]:
    """Find config.json in a checkpoint directory or its parents."""
    model_path = os.path.abspath(model_path)
    candidate_dirs = [model_path]
    parent = os.path.dirname(model_path)
    if parent and parent not in candidate_dirs:
        candidate_dirs.append(parent)
    grandparent = os.path.dirname(parent)
    if grandparent and grandparent not in candidate_dirs:
        candidate_dirs.append(grandparent)

    for directory in candidate_dirs:
        config_path = os.path.join(directory, "config.json")
        if os.path.exists(config_path):
            return config_path
    return None


def load_saved_model_config(model_path: str, logger=None) -> Dict[str, object]:
    """Load the persisted Prot2Mol architecture config for a checkpoint if present."""
    config_path = find_model_config_path(model_path)
    if config_path is None:
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            raw_config = json.load(handle)
    except Exception as exc:
        if logger is not None:
            logger.warning("Failed to load config.json from %s: %s", config_path, exc)
        return {}

    if not isinstance(raw_config, dict):
        if logger is not None:
            logger.warning("Ignoring non-mapping config.json at %s", config_path)
        return {}

    return {key: raw_config[key] for key in MODEL_CONFIG_KEYS if key in raw_config}


def save_model_config(output_dir: str, model_config: Dict[str, object], logger=None) -> str:
    """Persist the architecture config alongside saved checkpoints."""
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.json")
    serializable_config = {key: model_config[key] for key in MODEL_CONFIG_KEYS if key in model_config}
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(serializable_config, handle, indent=2, sort_keys=True)
    if logger is not None:
        logger.info("Saved model config to %s", config_path)
    return config_path


def _resolve_checkpoint_file(model_path: str) -> str:
    """Resolve checkpoint file path in a model directory."""
    if os.path.isfile(model_path):
        if model_path.endswith((".bin", ".safetensors")):
            return model_path
        raise ValueError(f"Unsupported checkpoint file: {model_path}")
    pytorch_path = os.path.join(model_path, "pytorch_model.bin")
    safetensors_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(pytorch_path):
        return pytorch_path
    if os.path.exists(safetensors_path):
        return safetensors_path
    raise FileNotFoundError(
        f"Could not find checkpoint file in {model_path}. "
        "Expected 'pytorch_model.bin' or 'model.safetensors'."
    )


def load_prot2mol_inference_model(
    model_path: str,
    device: torch.device,
    mol_tokenizer,
    prot_emb_model: str,
    max_mol_len: int,
    prot_max_length: int,
    protein_model_id: Optional[str] = None,
    decoder_model_id: str = "zjunlp/MolGen-large",
    conditioning_dropout: float = 0.1,
    strict: bool = True,
    allow_strict_fallback: bool = False,
    logger=None,
):
    """
    Build a Prot2MolModel and load weights for inference.

    Returns:
        Loaded Prot2MolModel on the target device.
    """
    from ..core.model import Prot2MolModel

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    checkpoint_file = _resolve_checkpoint_file(model_path)
    if checkpoint_file.endswith(".safetensors"):
        from safetensors.torch import load_file
        model_state = load_file(checkpoint_file)
    else:
        model_state = torch.load(checkpoint_file, map_location=device)

    saved_model_config = load_saved_model_config(model_path, logger=logger)

    model_config = {
        "prot_emb_model": prot_emb_model,
        "protein_model_id": protein_model_id,
        "decoder_model_id": decoder_model_id,
        "conditioning_dropout": conditioning_dropout,
        "max_mol_len": max_mol_len,
        "prot_max_length": prot_max_length,
        "train_encoder_model": False,
        "train_projection_model": False,
        "train_decoder_model": False,
        "mol_tokenizer": mol_tokenizer,
    }
    model_config.update(saved_model_config)
    model_config["train_encoder_model"] = False
    model_config["train_projection_model"] = False
    model_config["train_decoder_model"] = False
    model_config["mol_tokenizer"] = mol_tokenizer

    model = Prot2MolModel(model_config)
    try:
        model.load_state_dict(model_state, strict=strict)
    except RuntimeError as exc:
        if not (strict and allow_strict_fallback):
            raise
        if logger is not None:
            logger.warning("Strict checkpoint load failed, retrying with strict=False: %s", exc)
        model.load_state_dict(model_state, strict=False)

    model.to(device)
    return model
