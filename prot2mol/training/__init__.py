"""Training utilities and orchestration components for Prot2Mol."""

from .entry import create_run_name, parse_arguments, setup_logging, validate_and_process_paths
from .grpo import (
    GRPOConfig,
    GRPOLossOutput,
    GRPORollout,
    GRPOStepOutput,
    GRPOTrainer,
    RewardMoleculeRepresentation,
    RewardModelProbabilityScorer,
    compute_grpo_loss,
    grouped_advantages,
    selfies_to_smiles,
)
from .metrics import compute_conditional_generation_metrics, compute_generation_metrics
from .pretrain import TrainingScript
from .trainer import Prot2MolTrainer
from .training_runner import TrainingRunner

__all__ = [
    "create_run_name",
    "parse_arguments",
    "setup_logging",
    "validate_and_process_paths",
    "compute_generation_metrics",
    "compute_conditional_generation_metrics",
    "GRPOConfig",
    "GRPOLossOutput",
    "GRPORollout",
    "GRPOStepOutput",
    "GRPOTrainer",
    "RewardMoleculeRepresentation",
    "RewardModelProbabilityScorer",
    "compute_grpo_loss",
    "grouped_advantages",
    "selfies_to_smiles",
    "TrainingScript",
    "Prot2MolTrainer",
    "TrainingRunner",
]
