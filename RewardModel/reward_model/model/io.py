from __future__ import annotations

import json
import os
from typing import Dict, Optional

import torch

from .config import RewardModelConfig
from .encoders import LoadedEncoder

CONFIG_FILENAME = "config.json"
PYTORCH_WEIGHTS_FILENAME = "pytorch_model.bin"
SAFE_WEIGHTS_FILENAME = "model.safetensors"


def find_model_config_path(model_path: str) -> Optional[str]:
    model_path = os.path.abspath(model_path)
    candidate_dirs = [model_path]
    parent = os.path.dirname(model_path)
    if parent and parent not in candidate_dirs:
        candidate_dirs.append(parent)
    grandparent = os.path.dirname(parent)
    if grandparent and grandparent not in candidate_dirs:
        candidate_dirs.append(grandparent)

    for directory in candidate_dirs:
        config_path = os.path.join(directory, CONFIG_FILENAME)
        if os.path.exists(config_path):
            return config_path
    return None


def _resolve_checkpoint_file(model_path: str) -> str:
    pytorch_path = os.path.join(model_path, PYTORCH_WEIGHTS_FILENAME)
    safetensors_path = os.path.join(model_path, SAFE_WEIGHTS_FILENAME)
    if os.path.exists(pytorch_path):
        return pytorch_path
    if os.path.exists(safetensors_path):
        return safetensors_path
    raise FileNotFoundError(
        f"Could not find checkpoint file in {model_path}. "
        f"Expected {PYTORCH_WEIGHTS_FILENAME!r} or {SAFE_WEIGHTS_FILENAME!r}."
    )


def save_reward_model_config(output_dir: str, config: RewardModelConfig) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, CONFIG_FILENAME)
    config.save_json(path)
    return path


def load_reward_model_config(model_path: str) -> RewardModelConfig:
    config_path = find_model_config_path(model_path)
    if config_path is None:
        raise FileNotFoundError(f"Could not find {CONFIG_FILENAME!r} in or above {model_path}")
    return RewardModelConfig.load_json(config_path)


def save_reward_model(model, output_dir: str, use_safetensors: bool = False) -> str:
    os.makedirs(output_dir, exist_ok=True)
    save_reward_model_config(output_dir, model.config)

    if use_safetensors:
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise ImportError("safetensors is not available but use_safetensors=True was requested") from exc
        weights_path = os.path.join(output_dir, SAFE_WEIGHTS_FILENAME)
        save_file(model.state_dict(), weights_path)
        return weights_path

    weights_path = os.path.join(output_dir, PYTORCH_WEIGHTS_FILENAME)
    torch.save(model.state_dict(), weights_path)
    return weights_path


def load_reward_model(
    model_path: str,
    device: Optional[torch.device] = None,
    strict: bool = True,
    config_overrides: Optional[Dict[str, object]] = None,
    protein_bundle: Optional[LoadedEncoder] = None,
    molecule_bundle: Optional[LoadedEncoder] = None,
):
    from .core import RewardModel

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    checkpoint_file = _resolve_checkpoint_file(model_path)
    config = load_reward_model_config(model_path)
    if config_overrides:
        merged = config.to_dict()
        merged.update(config_overrides)
        config = RewardModelConfig.from_dict(merged)

    if checkpoint_file.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(checkpoint_file)
    else:
        load_device = device if device is not None else torch.device("cpu")
        state_dict = torch.load(checkpoint_file, map_location=load_device)

    model = RewardModel(
        config=config,
        protein_bundle=protein_bundle,
        molecule_bundle=molecule_bundle,
    )
    model.load_state_dict(state_dict, strict=strict)
    if device is not None:
        model.to(device)
    return model
