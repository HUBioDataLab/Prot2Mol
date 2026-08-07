from __future__ import annotations

import os
from typing import Optional, Sequence

import torch

from ..model import load_reward_model
from ..training import (
    get_tokenized_split_dataset_paths,
    load_reward_training_config,
    load_tokenized_example_dataset,
)
from .ranking_head import (
    load_assay_manifest,
    select_complete_assays,
    write_assay_manifest,
)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def load_model_and_training_config(
    *,
    checkpoint: str,
    config_path: str,
    device: torch.device,
):
    training_config = load_reward_training_config(config_path)
    model = load_reward_model(checkpoint, device=device)
    model.eval()
    return model, training_config


def load_selected_split(
    *,
    training_config,
    split: str,
    max_assays: Optional[int],
    seed: int,
    manifest_path: Optional[str] = None,
    output_manifest_path: Optional[str] = None,
):
    paths = get_tokenized_split_dataset_paths(
        training_config.data.tokenized_dataset_dir
    )
    dataset_path = paths[str(split)]
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Tokenized {split} dataset does not exist: {dataset_path}"
        )
    dataset = load_tokenized_example_dataset(dataset_path)
    requested_assays: Optional[Sequence[str]] = None
    if manifest_path is not None:
        requested_assays = load_assay_manifest(manifest_path, split=split)
    selected, assay_ids = select_complete_assays(
        dataset,
        max_assays=max_assays,
        seed=seed,
        min_size=3,
        min_pchembl_span=training_config.data.ranking_min_pchembl_span,
        assay_ids=requested_assays,
    )
    if len(selected) == 0:
        raise ValueError(f"No eligible assays were selected from split {split!r}")
    if output_manifest_path is not None:
        write_assay_manifest(
            output_manifest_path,
            split=split,
            assay_ids=assay_ids,
            dataset=selected,
            seed=seed,
        )
    return selected, assay_ids
