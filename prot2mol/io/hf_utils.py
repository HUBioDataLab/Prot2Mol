import os
from typing import Optional, Sequence

import torch


def resolve_model_path(model_name: str, models_base: Optional[str] = None, fallback_bases: Optional[Sequence[str]] = None) -> str:
    """
    Resolve a local HuggingFace cache path for a given model name.

    The function prefers snapshot directories when present and falls back to the
    model directory. If no base is provided, it uses MODELS_BASE_PATH or ./models.
    """
    if os.path.exists(model_name):
        return model_name

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
        model_dir = os.path.join(base, f"models--{model_name}")
        snapshots_dir = os.path.join(model_dir, "snapshots")
        if os.path.isdir(snapshots_dir):
            snapshots = sorted(os.listdir(snapshots_dir))
            if snapshots:
                return os.path.join(snapshots_dir, snapshots[0])
        if os.path.isdir(model_dir):
            return model_dir

    return os.path.join(os.path.expanduser(bases[0]), f"models--{model_name}")


def load_molgen_tokenizer(models_base: Optional[str] = None, fallback_bases: Optional[Sequence[str]] = None, padding_side: str = "left"):
    """Load the MolGen tokenizer from a local cache."""
    from transformers import BartTokenizer

    model_path = resolve_model_path("zjunlp--MolGen-large", models_base=models_base, fallback_bases=fallback_bases)
    return BartTokenizer.from_pretrained(model_path, padding_side=padding_side)


def _resolve_checkpoint_file(model_path: str) -> str:
    """Resolve checkpoint file path in a model directory."""
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
    n_layer: int,
    n_head: int,
    n_emb: int,
    max_mol_len: int,
    prot_max_length: int,
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

    model_config = {
        "prot_emb_model": prot_emb_model,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_emb": n_emb,
        "max_mol_len": max_mol_len,
        "prot_max_length": prot_max_length,
        "train_encoder_model": False,
        "train_decoder_model": False,
        "train_pchembl_head": True,
        "mol_tokenizer": mol_tokenizer,
    }

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
