import os
from typing import Optional, Sequence


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
