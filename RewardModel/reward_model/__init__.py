"""Standalone reward-model package for protein-molecule scoring."""

from .config import RewardModelConfig
from .encoders import LoadedEncoder, batch_encode_texts, load_encoder_bundle
from .io import load_reward_model, load_reward_model_config, save_reward_model, save_reward_model_config
from .model import RewardModel
from .outputs import RewardModelOutput

__all__ = [
    "LoadedEncoder",
    "RewardModel",
    "RewardModelConfig",
    "RewardModelOutput",
    "batch_encode_texts",
    "load_encoder_bundle",
    "load_reward_model",
    "load_reward_model_config",
    "save_reward_model",
    "save_reward_model_config",
]
