"""Standalone reward-model package for protein-molecule scoring."""

from .data import RewardDataStore, group_rows_by_group_id, load_curated_rows
from .data_processing import ChemblPreprocessConfig, preprocess_chembl_sqlite
from .model.config import RewardModelConfig

__all__ = [
    "ChemblPreprocessConfig",
    "RewardDataStore",
    "RewardModelConfig",
    "group_rows_by_group_id",
    "load_curated_rows",
    "preprocess_chembl_sqlite",
]

try:
    from .model import (
        LoadedEncoder,
        RewardModel,
        RewardModelOutput,
        batch_encode_texts,
        load_encoder_bundle,
        load_reward_model,
        load_reward_model_config,
        load_tokenizer,
        save_reward_model,
        save_reward_model_config,
    )
except Exception:
    pass
else:
    __all__.extend(
        [
            "LoadedEncoder",
            "RewardModel",
            "RewardModelOutput",
            "batch_encode_texts",
            "load_encoder_bundle",
            "load_reward_model",
            "load_reward_model_config",
            "load_tokenizer",
            "save_reward_model",
            "save_reward_model_config",
        ]
    )
