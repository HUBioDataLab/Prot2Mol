"""Group Relative Policy Optimization for protein-conditioned molecule generation."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, ContextManager, Literal, Sequence

import selfies as sf
import torch
import torch.nn as nn

from ..chem.utils import canonic_smiles, molecular_property_summary
from ..rewards.diversity import internal_diversity_factors


RewardFunction = Callable[[Sequence[str], Sequence[str]], torch.Tensor | Sequence[float]]
RewardMoleculeRepresentation = Literal["smiles", "selfies"]
GRPOLossType = Literal["grpo", "bnpo"]
GenerationStartMode = Literal["tokenizer_bos", "legacy_pad"]
ValidityFunction = Callable[[str], bool]


@dataclass(frozen=True)
class GRPOConfig:
    group_size: int = 8
    clip_epsilon: float = 0.2
    kl_beta: float = 0.01
    advantage_epsilon: float = 1.0e-6
    max_length: int = 256
    min_length: int = 3
    temperature: float = 1.0
    top_p: float = 1.0
    num_iterations: int = 2
    loss_type: GRPOLossType = "grpo"
    max_grad_norm: float = 1.0
    reward_min: float = 0.0
    reward_max: float = 1.0
    reward_saturation_threshold: float = 0.01
    activity_probability_threshold: float = 0.5
    require_eos_for_reward: bool = True
    diversity_reward_shaping: bool = False
    diversity_reward_weight: float = 0.5
    diversity_mean_similarity_weight: float = 0.5
    diversity_duplicate_penalty_factor: float = 0.0
    diversity_morgan_radius: int = 2
    diversity_morgan_bits: int = 2048
    precision: Literal["fp32", "bf16"] = "fp32"
    generation_start_mode: GenerationStartMode = "tokenizer_bos"

    def __post_init__(self) -> None:
        if self.group_size < 2:
            raise ValueError("GRPO group_size must be at least 2")
        if not 0.0 < self.clip_epsilon < 1.0:
            raise ValueError("clip_epsilon must be in (0, 1)")
        if self.kl_beta < 0.0:
            raise ValueError("kl_beta must be nonnegative")
        if self.advantage_epsilon <= 0.0:
            raise ValueError("advantage_epsilon must be positive")
        if self.max_length < 3:
            raise ValueError("max_length must leave room for generated tokens")
        if self.min_length < 2 or self.min_length > self.max_length:
            raise ValueError("min_length must be in [2, max_length]")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.num_iterations < 1:
            raise ValueError("num_iterations must be positive")
        if self.loss_type not in {"grpo", "bnpo"}:
            raise ValueError("loss_type must be 'grpo' or 'bnpo'")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if self.reward_min >= self.reward_max:
            raise ValueError("reward_min must be smaller than reward_max")
        if not 0.0 <= self.diversity_reward_weight <= 1.0:
            raise ValueError("diversity_reward_weight must be in [0, 1]")
        if not 0.0 <= self.diversity_mean_similarity_weight <= 1.0:
            raise ValueError("diversity_mean_similarity_weight must be in [0, 1]")
        if not 0.0 <= self.diversity_duplicate_penalty_factor <= 1.0:
            raise ValueError("diversity_duplicate_penalty_factor must be in [0, 1]")
        if self.diversity_morgan_radius < 1 or self.diversity_morgan_bits < 8:
            raise ValueError("Morgan diversity fingerprint settings are invalid")
        reward_range = self.reward_max - self.reward_min
        if not 0.0 <= self.reward_saturation_threshold < reward_range / 2:
            raise ValueError(
                "reward_saturation_threshold must be nonnegative and smaller "
                "than half the reward range"
            )
        if not 0.0 < self.activity_probability_threshold < 1.0:
            raise ValueError("activity_probability_threshold must be in (0, 1)")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("GRPO precision must be 'fp32' or 'bf16'")
        if self.generation_start_mode not in {"tokenizer_bos", "legacy_pad"}:
            raise ValueError(
                "generation_start_mode must be 'tokenizer_bos' or 'legacy_pad'"
            )


@dataclass
class GRPORollout:
    protein_input_ids: torch.Tensor
    protein_attention_mask: torch.Tensor
    protein_embeddings: torch.Tensor | None
    generated_ids: torch.Tensor
    old_log_probs: torch.Tensor
    reference_log_probs: torch.Tensor
    action_mask: torch.Tensor
    rewards: torch.Tensor
    reward_diagnostics: dict[str, torch.Tensor]
    advantages: torch.Tensor
    generated_selfies: list[str]
    generated_smiles: list[str]
    reward_protein_sequences: list[str]
    reward_molecule_representation: RewardMoleculeRepresentation
    chemically_valid_mask: torch.Tensor
    terminated_mask: torch.Tensor
    valid_mask: torch.Tensor
    batch_size: int
    group_size: int


@dataclass
class GRPOLossOutput:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl: torch.Tensor
    metrics: dict[str, float]


@dataclass
class GRPOStepOutput:
    metrics: dict[str, float]
    rollout: GRPORollout


class RewardModelProbabilityScorer:
    """Freeze a RewardModel and expose its SMILES activity probability as reward."""

    protein_representation = "sequence"
    molecule_representation = "smiles"

    def __init__(self, reward_model: nn.Module):
        self.reward_model = reward_model
        self.reward_model.requires_grad_(False)
        self.reward_model.eval()

    def __call__(
        self,
        protein_sequences: Sequence[str],
        molecule_sequences: Sequence[str],
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.reward_model.score_pairs(
                protein_sequences=protein_sequences,
                molecule_sequences=molecule_sequences,
            )
        return outputs.activity_probability.detach().float().cpu()


def valid_selfies(value: str) -> bool:
    try:
        return bool(value and sf.decoder(value))
    except Exception:
        return False


def selfies_to_smiles(value: str) -> str:
    try:
        decoded = sf.decoder(value) if value else ""
    except Exception:
        return ""
    return canonic_smiles(decoded) or ""


def grouped_advantages(
    rewards: torch.Tensor,
    *,
    batch_size: int,
    group_size: int,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if rewards.numel() != batch_size * group_size:
        raise ValueError("reward count must equal batch_size * group_size")
    grouped = rewards.float().reshape(batch_size, group_size)
    means = grouped.mean(dim=1, keepdim=True)
    standard_deviations = grouped.std(dim=1, unbiased=False, keepdim=True)
    normalized = (grouped - means) / standard_deviations.clamp_min(epsilon)
    normalized = torch.where(
        standard_deviations > epsilon,
        normalized,
        torch.zeros_like(normalized),
    )
    return normalized.reshape(-1), means.squeeze(1), standard_deviations.squeeze(1)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError("values and mask must have the same shape")
    denominator = mask.sum()
    if int(denominator.detach().cpu()) == 0:
        raise ValueError("GRPO batch contains no generated action tokens")
    return (values * mask.to(dtype=values.dtype)).sum() / denominator


def _sequence_normalized_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Average tokens within each completion, then average completions equally."""

    if values.shape != mask.shape:
        raise ValueError("values and mask must have the same shape")
    lengths = mask.sum(dim=1)
    if lengths.numel() == 0 or bool(lengths.eq(0).any().detach().cpu()):
        raise ValueError("every GRPO completion must contain an action token")
    per_sequence = (values * mask.to(dtype=values.dtype)).sum(dim=1) / lengths.to(
        dtype=values.dtype
    )
    return per_sequence.mean()


def compute_grpo_loss(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    action_mask: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_epsilon: float,
    kl_beta: float,
    loss_type: GRPOLossType = "grpo",
) -> GRPOLossOutput:
    if not (
        current_log_probs.shape
        == old_log_probs.shape
        == reference_log_probs.shape
        == action_mask.shape
    ):
        raise ValueError("all token tensors must have the same shape")
    if advantages.ndim != 1 or advantages.numel() != current_log_probs.size(0):
        raise ValueError("advantages must contain one scalar per generated sequence")
    if loss_type not in {"grpo", "bnpo"}:
        raise ValueError("loss_type must be 'grpo' or 'bnpo'")

    loss_reducer = _sequence_normalized_mean if loss_type == "grpo" else _masked_mean

    log_ratio = (current_log_probs - old_log_probs).clamp(min=-20.0, max=20.0)
    ratio = log_ratio.exp()
    token_advantages = advantages.to(current_log_probs).unsqueeze(1)
    unclipped = ratio * token_advantages
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * token_advantages
    policy_loss = -loss_reducer(torch.minimum(unclipped, clipped), action_mask)

    reference_log_ratio = (reference_log_probs - current_log_probs).clamp(
        min=-20.0,
        max=20.0,
    )
    per_token_kl = reference_log_ratio.exp() - reference_log_ratio - 1.0
    kl = loss_reducer(per_token_kl, action_mask)
    loss = policy_loss + kl_beta * kl

    clipped_tokens = ratio.sub(1.0).abs().gt(clip_epsilon)
    sampled_nll = -_masked_mean(current_log_probs, action_mask)
    metrics = {
        "grpo/policy_loss": float(policy_loss.detach().cpu()),
        "grpo/kl": float(kl.detach().cpu()),
        "grpo/total_loss": float(loss.detach().cpu()),
        "grpo/clip_fraction": float(
            _masked_mean(clipped_tokens.float(), action_mask).detach().cpu()
        ),
        "grpo/ratio_mean": float(_masked_mean(ratio, action_mask).detach().cpu()),
        "grpo/ratio_std": float(
            torch.sqrt(
                _masked_mean(
                    (ratio - _masked_mean(ratio, action_mask)).square(),
                    action_mask,
                )
            )
            .detach()
            .cpu()
        ),
        "grpo/old_policy_approx_kl": float(
            (0.5 * _masked_mean(log_ratio.square(), action_mask)).detach().cpu()
        ),
        "grpo/sampled_token_nll": float(sampled_nll.detach().cpu()),
        "grpo/mean_sequence_log_prob": float(
            (current_log_probs * action_mask).sum(dim=1).mean().detach().cpu()
        ),
        "grpo/sequence_normalized_loss": float(loss_type == "grpo"),
    }
    return GRPOLossOutput(
        loss=loss,
        policy_loss=policy_loss,
        kl=kl,
        metrics=metrics,
    )


class GRPOTrainer:
    """Small online GRPO trainer with explicit rollout and metric boundaries."""

    def __init__(
        self,
        *,
        policy: nn.Module,
        reference_policy: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        tokenizer,
        reward_function: RewardFunction,
        config: GRPOConfig | None = None,
        validity_function: ValidityFunction = valid_selfies,
        reward_molecule_representation: RewardMoleculeRepresentation | None = None,
    ):
        if policy is reference_policy:
            raise ValueError("policy and reference_policy must be distinct models")
        self.policy = policy
        self.reference_policy = reference_policy
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.reward_function = reward_function
        self.config = config or GRPOConfig()
        self.validity_function = validity_function
        inferred_representation = getattr(
            reward_function,
            "molecule_representation",
            "smiles",
        )
        self.reward_molecule_representation = (
            reward_molecule_representation or inferred_representation
        )
        if self.reward_molecule_representation not in {"smiles", "selfies"}:
            raise ValueError(
                "reward_molecule_representation must be 'smiles' or 'selfies'"
            )
        self.global_step = 0
        self.optimization_step = 0

        policy_device = next(self.policy.parameters()).device
        reference_device = next(self.reference_policy.parameters()).device
        if policy_device != reference_device:
            raise ValueError("policy and reference_policy must be on the same device")
        if self.config.precision == "bf16" and policy_device.type not in {
            "cpu",
            "cuda",
        }:
            raise ValueError("GRPO BF16 autocast supports CPU and CUDA devices")
        if (
            self.config.precision == "bf16"
            and policy_device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            raise ValueError("the selected CUDA device does not support BF16")
        self.reference_policy.requires_grad_(False)
        self.reference_policy.eval()
        conditioning_modules = [
            getattr(self.policy, "protein_encoder", None),
            getattr(self.policy, "conditioning_projection", None),
        ]
        self._reuse_policy_protein_embeddings = all(
            module is not None
            and all(not parameter.requires_grad for parameter in module.parameters())
            for module in conditioning_modules
        )
        self._reuse_reference_protein_embeddings = (
            self._reuse_policy_protein_embeddings
            and getattr(self.policy, "protein_encoder", None)
            is getattr(self.reference_policy, "protein_encoder", None)
            and getattr(self.policy, "conditioning_projection", None)
            is getattr(self.reference_policy, "conditioning_projection", None)
        )

    def _autocast_context(self, device: torch.device) -> ContextManager:
        if self.config.precision == "bf16":
            return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        return nullcontext()

    def _decode(self, generated_ids: torch.Tensor) -> list[str]:
        decoded = self.tokenizer.batch_decode(
            generated_ids.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [str(value).replace(" ", "") for value in decoded]

    def _score_rewards(
        self,
        repeated_protein_sequences: Sequence[str],
        generated_selfies: Sequence[str],
        generated_smiles: Sequence[str],
        valid_mask: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten().tolist()
        rewards = torch.zeros(len(generated_smiles), dtype=torch.float32, device=device)
        diagnostics: dict[str, torch.Tensor] = {}
        reward_molecules = (
            generated_selfies
            if self.reward_molecule_representation == "selfies"
            else generated_smiles
        )
        if valid_indices:
            valid_proteins = [repeated_protein_sequences[index] for index in valid_indices]
            valid_molecules = [reward_molecules[index] for index in valid_indices]
            scored = torch.as_tensor(
                self.reward_function(valid_proteins, valid_molecules),
                dtype=torch.float32,
                device=device,
            ).reshape(-1)
            if scored.numel() != len(valid_indices):
                raise ValueError("reward function returned the wrong number of scores")
            if not torch.isfinite(scored).all():
                raise ValueError("reward function returned non-finite scores")
            if (scored < self.config.reward_min).any() or (
                scored > self.config.reward_max
            ).any():
                raise ValueError(
                    "reward function returned values outside the configured reward range"
                )
            rewards[torch.tensor(valid_indices, device=device)] = scored
            diagnostics_provider = getattr(
                self.reward_function,
                "last_diagnostics",
                None,
            )
            if callable(diagnostics_provider):
                valid_diagnostics = diagnostics_provider()
                for name, values in valid_diagnostics.items():
                    valid_values = torch.as_tensor(
                        values,
                        dtype=torch.float32,
                        device=device,
                    ).reshape(-1)
                    if valid_values.numel() != len(valid_indices):
                        raise ValueError(
                            f"Reward diagnostic {name!r} returned the wrong number "
                            "of values"
                        )
                    if not torch.isfinite(valid_values).all():
                        raise ValueError(
                            f"Reward diagnostic {name!r} returned non-finite values"
                        )
                    aligned = torch.zeros(
                        len(generated_smiles),
                        dtype=torch.float32,
                        device=device,
                    )
                    aligned[torch.tensor(valid_indices, device=device)] = valid_values
                    diagnostics[name] = aligned
            if "activity_probability" not in diagnostics:
                aligned_activity = torch.zeros(
                    len(generated_smiles),
                    dtype=torch.float32,
                    device=device,
                )
                aligned_activity[torch.tensor(valid_indices, device=device)] = scored
                diagnostics["activity_probability"] = aligned_activity
        if diagnostics or self.config.diversity_reward_shaping:
            diagnostics.setdefault(
                "property_shaped_reward",
                rewards.detach().clone(),
            )
        if self.config.diversity_reward_shaping:
            diversity = internal_diversity_factors(
                generated_smiles,
                valid_mask.detach().cpu().tolist(),
                group_size=self.config.group_size,
                reward_weight=self.config.diversity_reward_weight,
                mean_similarity_weight=(
                    self.config.diversity_mean_similarity_weight
                ),
                duplicate_penalty_factor=(
                    self.config.diversity_duplicate_penalty_factor
                ),
                morgan_radius=self.config.diversity_morgan_radius,
                morgan_bits=self.config.diversity_morgan_bits,
            )
            for name, values in diversity.as_dict().items():
                diagnostics[name] = torch.as_tensor(
                    values,
                    dtype=torch.float32,
                    device=device,
                )
            rewards = rewards * diagnostics["diversity_penalty_factor"]
        if diagnostics:
            diagnostics["final_reward"] = rewards.detach().clone()
        return rewards, diagnostics

    def collect_rollouts(
        self,
        *,
        protein_input_ids: torch.Tensor,
        protein_attention_mask: torch.Tensor,
        protein_sequences: Sequence[str],
        reward_protein_sequences: Sequence[str] | None = None,
    ) -> GRPORollout:
        batch_size = protein_input_ids.size(0)
        if protein_attention_mask.shape != protein_input_ids.shape:
            raise ValueError("protein input ids and attention mask must align")
        if len(protein_sequences) != batch_size:
            raise ValueError("protein_sequences must align with the protein tensor batch")
        reward_sequences = (
            protein_sequences
            if reward_protein_sequences is None
            else reward_protein_sequences
        )
        if len(reward_sequences) != batch_size:
            raise ValueError(
                "reward_protein_sequences must align with the protein tensor batch"
            )
        if any(not isinstance(sequence, str) or not sequence for sequence in reward_sequences):
            raise ValueError("reward protein sequences must be non-empty strings")

        device = next(self.policy.parameters()).device
        protein_input_ids = protein_input_ids.to(device)
        protein_attention_mask = protein_attention_mask.to(device)
        group_size = self.config.group_size
        repeated_input_ids = protein_input_ids.repeat_interleave(group_size, dim=0)
        repeated_attention_mask = protein_attention_mask.repeat_interleave(
            group_size,
            dim=0,
        )
        repeated_reward_sequences = [
            sequence for sequence in reward_sequences for _ in range(group_size)
        ]

        was_training = self.policy.training
        self.policy.eval()
        self.reference_policy.eval()
        with torch.no_grad(), self._autocast_context(device):
            protein_embeddings = self.policy.encode_protein(
                protein_input_ids,
                protein_attention_mask,
            ).repeat_interleave(group_size, dim=0)
            generated_ids = self.policy.generate_from_protein_embeddings(
                protein_embeddings,
                repeated_attention_mask,
                max_length=self.config.max_length,
                min_length=self.config.min_length,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                bos_token_id=(
                    int(self.policy.config.pad_token_id)
                    if self.config.generation_start_mode == "legacy_pad"
                    else int(self.policy.config.bos_token_id)
                ),
            )
            old_log_probs, action_mask = self.policy.generated_token_log_probs(
                generated_ids,
                repeated_input_ids,
                repeated_attention_mask,
                temperature=self.config.temperature,
                protein_embeddings=protein_embeddings,
                pad_token_is_termination=(
                    self.config.generation_start_mode != "legacy_pad"
                ),
            )
            reference_log_probs, reference_mask = (
                self.reference_policy.generated_token_log_probs(
                    generated_ids,
                    repeated_input_ids,
                    repeated_attention_mask,
                    temperature=self.config.temperature,
                    protein_embeddings=(
                        protein_embeddings
                        if self._reuse_reference_protein_embeddings
                        else None
                    ),
                    pad_token_is_termination=(
                        self.config.generation_start_mode != "legacy_pad"
                    ),
                )
            )
        if was_training:
            self.policy.train()
        if not torch.equal(action_mask, reference_mask):
            raise RuntimeError("policy and reference action masks differ")

        generated_selfies = self._decode(generated_ids)
        generated_smiles = [selfies_to_smiles(value) for value in generated_selfies]
        chemically_valid_mask = torch.tensor(
            [
                self.validity_function(value) and bool(generated_smiles[index])
                for index, value in enumerate(generated_selfies)
            ],
            dtype=torch.bool,
            device=device,
        )
        eos_id = int(self.policy.config.eos_token_id)
        terminated_mask = generated_ids[:, 1:].eq(eos_id).any(dim=1)
        valid_mask = chemically_valid_mask & (
            terminated_mask
            if self.config.require_eos_for_reward
            else torch.ones_like(terminated_mask)
        )
        rewards, reward_diagnostics = self._score_rewards(
            repeated_reward_sequences,
            generated_selfies,
            generated_smiles,
            valid_mask,
            device,
        )
        advantages, _, _ = grouped_advantages(
            rewards,
            batch_size=batch_size,
            group_size=group_size,
            epsilon=self.config.advantage_epsilon,
        )
        return GRPORollout(
            protein_input_ids=repeated_input_ids,
            protein_attention_mask=repeated_attention_mask,
            protein_embeddings=(
                protein_embeddings.detach()
                if self._reuse_policy_protein_embeddings
                else None
            ),
            generated_ids=generated_ids,
            old_log_probs=old_log_probs,
            reference_log_probs=reference_log_probs,
            action_mask=action_mask,
            rewards=rewards,
            reward_diagnostics=reward_diagnostics,
            advantages=advantages,
            generated_selfies=generated_selfies,
            generated_smiles=generated_smiles,
            reward_protein_sequences=repeated_reward_sequences,
            reward_molecule_representation=self.reward_molecule_representation,
            chemically_valid_mask=chemically_valid_mask,
            terminated_mask=terminated_mask,
            valid_mask=valid_mask,
            batch_size=batch_size,
            group_size=group_size,
        )

    def _rollout_metrics(self, rollout: GRPORollout) -> dict[str, float]:
        grouped_rewards = rollout.rewards.reshape(rollout.batch_size, rollout.group_size)
        group_means = grouped_rewards.mean(dim=1)
        group_standard_deviations = grouped_rewards.std(dim=1, unbiased=False)
        group_unique_fractions = []
        for start in range(0, len(rollout.generated_smiles), rollout.group_size):
            group = rollout.generated_smiles[start : start + rollout.group_size]
            group_unique_fractions.append(len(set(group)) / len(group))
        action_lengths = rollout.action_mask.sum(dim=1).float()
        eos_fraction = rollout.terminated_mask.float().mean()
        valid_rewards = rollout.rewards[rollout.valid_mask]
        saturation_threshold = self.config.reward_saturation_threshold
        low_saturation_limit = self.config.reward_min + saturation_threshold
        high_saturation_limit = self.config.reward_max - saturation_threshold
        valid_smiles = [
            value
            for value, is_valid in zip(
                rollout.generated_smiles,
                rollout.valid_mask.detach().cpu().tolist(),
            )
            if is_valid
        ]
        property_metrics = molecular_property_summary(valid_smiles)
        metrics = {
            "grpo/precision_bf16": float(self.config.precision == "bf16"),
            "grpo/legacy_pad_generation": float(
                self.config.generation_start_mode == "legacy_pad"
            ),
            "grpo/reused_policy_protein_embeddings": float(
                rollout.protein_embeddings is not None
            ),
            "grpo/reused_reference_protein_embeddings": float(
                self._reuse_reference_protein_embeddings
            ),
            "grpo/reward_mean": float(rollout.rewards.mean().detach().cpu()),
            "grpo/reward_std": float(
                rollout.rewards.std(unbiased=False).detach().cpu()
            ),
            "grpo/reward_min": float(rollout.rewards.min().detach().cpu()),
            "grpo/reward_max": float(rollout.rewards.max().detach().cpu()),
            "grpo/group_reward_mean": float(group_means.mean().detach().cpu()),
            "grpo/group_reward_mean_std": float(
                group_means.std(unbiased=False).detach().cpu()
            ),
            "grpo/valid_reward_mean": float(
                valid_rewards.mean().detach().cpu() if valid_rewards.numel() else 0.0
            ),
            "grpo/valid_reward_low_saturation_fraction": float(
                valid_rewards.le(low_saturation_limit).float().mean().detach().cpu()
                if valid_rewards.numel()
                else 0.0
            ),
            "grpo/valid_reward_high_saturation_fraction": float(
                valid_rewards.ge(high_saturation_limit).float().mean().detach().cpu()
                if valid_rewards.numel()
                else 0.0
            ),
            "grpo/group_reward_std_mean": float(
                group_standard_deviations.mean().detach().cpu()
            ),
            "grpo/zero_variance_group_fraction": float(
                group_standard_deviations.le(self.config.advantage_epsilon)
                .float()
                .mean()
                .detach()
                .cpu()
            ),
            "grpo/advantage_mean": float(rollout.advantages.mean().detach().cpu()),
            "grpo/advantage_std": float(
                rollout.advantages.std(unbiased=False).detach().cpu()
            ),
            "grpo/nonzero_advantage_fraction": float(
                rollout.advantages.ne(0).float().mean().detach().cpu()
            ),
            "grpo/valid_fraction": float(rollout.valid_mask.float().mean().cpu()),
            "grpo/chemical_valid_fraction": float(
                rollout.chemically_valid_mask.float().mean().cpu()
            ),
            "grpo/valid_and_terminated_fraction": float(
                (rollout.chemically_valid_mask & rollout.terminated_mask)
                .float()
                .mean()
                .cpu()
            ),
            "grpo/terminated_fraction": float(eos_fraction.cpu()),
            "grpo/truncated_fraction": float((1.0 - eos_fraction).cpu()),
            "grpo/selfies_unique_fraction": len(set(rollout.generated_selfies))
            / len(rollout.generated_selfies),
            "grpo/unique_fraction": len(set(rollout.generated_smiles))
            / len(rollout.generated_smiles),
            "grpo/valid_unique_fraction": (
                len(set(valid_smiles)) / len(valid_smiles) if valid_smiles else 0.0
            ),
            "grpo/group_unique_fraction": sum(group_unique_fractions)
            / len(group_unique_fractions),
            "grpo/sequence_length_mean": float(action_lengths.mean().cpu()),
            "grpo/sequence_length_max": float(action_lengths.max().cpu()),
            "grpo/eos_fraction": float(eos_fraction.cpu()),
            "grpo/num_sequences": float(rollout.rewards.numel()),
            "grpo/num_action_tokens": float(rollout.action_mask.sum().cpu()),
        }
        if rollout.reward_diagnostics:
            valid_diagnostics = {
                name: values[rollout.valid_mask]
                for name, values in rollout.reward_diagnostics.items()
            }

            def diagnostic_mean(name: str) -> float:
                values = valid_diagnostics.get(name)
                if values is None or not values.numel():
                    return 0.0
                return float(values.mean().detach().cpu())

            def diagnostic_max(name: str) -> float:
                values = valid_diagnostics.get(name)
                if values is None or not values.numel():
                    return 0.0
                return float(values.max().detach().cpu())

            activity = valid_diagnostics.get("activity_probability")
            metrics.update(
                {
                    "grpo/valid_activity_probability_mean": diagnostic_mean(
                        "activity_probability"
                    ),
                    "grpo/valid_activity_optimization_reward_mean": diagnostic_mean(
                        "activity_optimization_reward"
                    ),
                    "grpo/valid_activity_probability_active_fraction": (
                        float(
                            activity.ge(self.config.activity_probability_threshold)
                            .float()
                            .mean()
                            .detach()
                            .cpu()
                        )
                        if activity is not None and activity.numel()
                        else 0.0
                    ),
                    "grpo/valid_activity_probability_high_saturation_fraction": (
                        float(
                            activity.ge(high_saturation_limit)
                            .float()
                            .mean()
                            .detach()
                            .cpu()
                        )
                        if activity is not None and activity.numel()
                        else 0.0
                    ),
                    "grpo/property_penalty_factor_mean": diagnostic_mean(
                        "property_penalty_factor"
                    ),
                    "grpo/logp_penalty_factor_mean": diagnostic_mean(
                        "logp_penalty_factor"
                    ),
                    "grpo/sas_penalty_factor_mean": diagnostic_mean(
                        "sas_penalty_factor"
                    ),
                    "grpo/heavy_atom_penalty_factor_mean": diagnostic_mean(
                        "heavy_atom_penalty_factor"
                    ),
                    "grpo/logp_violation_fraction": diagnostic_mean(
                        "logp_violation"
                    ),
                    "grpo/sas_violation_fraction": diagnostic_mean(
                        "sas_violation"
                    ),
                    "grpo/heavy_atom_violation_fraction": diagnostic_mean(
                        "heavy_atom_violation"
                    ),
                    "grpo/logp_excess_z_mean": diagnostic_mean("logp_excess_z"),
                    "grpo/logp_excess_z_max": diagnostic_max("logp_excess_z"),
                    "grpo/sas_excess_z_mean": diagnostic_mean("sas_excess_z"),
                    "grpo/sas_excess_z_max": diagnostic_max("sas_excess_z"),
                    "grpo/heavy_atom_excess_z_mean": diagnostic_mean(
                        "heavy_atom_excess_z"
                    ),
                    "grpo/heavy_atom_excess_z_max": diagnostic_max(
                        "heavy_atom_excess_z"
                    ),
                    "grpo/valid_property_shaped_reward_mean": diagnostic_mean(
                        "property_shaped_reward"
                    ),
                }
            )
            if "diversity_penalty_factor" in rollout.reward_diagnostics:
                comparable_mask = rollout.valid_mask & rollout.reward_diagnostics[
                    "diversity_comparable"
                ].bool()
                has_comparable = bool(comparable_mask.any())

                def comparable_mean(name: str) -> float:
                    if not has_comparable:
                        return 0.0
                    values = rollout.reward_diagnostics[name]
                    return float(values[comparable_mask].mean().detach().cpu())

                def comparable_max(name: str) -> float:
                    if not has_comparable:
                        return 0.0
                    values = rollout.reward_diagnostics[name]
                    return float(values[comparable_mask].max().detach().cpu())

                metrics.update(
                    {
                        "grpo/diversity_penalty_factor_mean": diagnostic_mean(
                            "diversity_penalty_factor"
                        ),
                        "grpo/internal_diversity_mean": (
                            1.0 - comparable_mean("mean_tanimoto_similarity")
                            if has_comparable
                            else 0.0
                        ),
                        "grpo/mean_tanimoto_similarity": comparable_mean(
                            "mean_tanimoto_similarity"
                        ),
                        "grpo/nearest_neighbor_tanimoto_similarity_mean": (
                            comparable_mean("max_tanimoto_similarity")
                        ),
                        "grpo/max_tanimoto_similarity": comparable_max(
                            "max_tanimoto_similarity"
                        ),
                        "grpo/combined_tanimoto_similarity_mean": comparable_mean(
                            "combined_tanimoto_similarity"
                        ),
                        "grpo/diversity_score_mean": comparable_mean(
                            "diversity_score"
                        ),
                        "grpo/exact_duplicate_fraction": diagnostic_mean(
                            "exact_duplicate"
                        ),
                        "grpo/diversity_comparable_fraction": diagnostic_mean(
                            "diversity_comparable"
                        ),
                    }
                )
        metrics.update(
            {
                f"grpo/{'property_count' if name == 'count' else name}": value
                for name, value in property_metrics.items()
            }
        )
        return metrics

    def step(
        self,
        *,
        protein_input_ids: torch.Tensor,
        protein_attention_mask: torch.Tensor,
        protein_sequences: Sequence[str],
        reward_protein_sequences: Sequence[str] | None = None,
    ) -> GRPOStepOutput:
        rollout = self.collect_rollouts(
            protein_input_ids=protein_input_ids,
            protein_attention_mask=protein_attention_mask,
            protein_sequences=protein_sequences,
            reward_protein_sequences=reward_protein_sequences,
        )
        metrics = self._rollout_metrics(rollout)
        iteration_metrics: list[dict[str, float]] = []
        last_grad_norm = 0.0
        restore_training_mode = self.policy.training

        # Keep dropout disabled while recomputing likelihoods. GRPO ratios must
        # compare the same deterministic policy distributions; eval mode still
        # permits gradients and parameter updates.
        self.policy.eval()
        for _ in range(self.config.num_iterations):
            with self._autocast_context(next(self.policy.parameters()).device):
                current_log_probs, action_mask = self.policy.generated_token_log_probs(
                    rollout.generated_ids,
                    rollout.protein_input_ids,
                    rollout.protein_attention_mask,
                    temperature=self.config.temperature,
                    protein_embeddings=rollout.protein_embeddings,
                    pad_token_is_termination=(
                        self.config.generation_start_mode != "legacy_pad"
                    ),
                )
                if not torch.equal(action_mask, rollout.action_mask):
                    raise RuntimeError("policy action mask changed during optimization")
                loss_output = compute_grpo_loss(
                    current_log_probs,
                    rollout.old_log_probs,
                    rollout.reference_log_probs,
                    rollout.action_mask,
                    rollout.advantages,
                    clip_epsilon=self.config.clip_epsilon,
                    kl_beta=self.config.kl_beta,
                    loss_type=self.config.loss_type,
                )
            self.optimizer.zero_grad(set_to_none=True)
            loss_output.loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in self.policy.parameters() if parameter.requires_grad],
                self.config.max_grad_norm,
            )
            last_grad_norm = float(torch.as_tensor(grad_norm).detach().cpu())
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimization_step += 1
            iteration_metrics.append(loss_output.metrics)

        with torch.no_grad(), self._autocast_context(
            next(self.policy.parameters()).device
        ):
            updated_log_probs, _ = self.policy.generated_token_log_probs(
                rollout.generated_ids,
                rollout.protein_input_ids,
                rollout.protein_attention_mask,
                temperature=self.config.temperature,
                protein_embeddings=rollout.protein_embeddings,
                pad_token_is_termination=(
                    self.config.generation_start_mode != "legacy_pad"
                ),
            )
            per_sequence_change = (
                (updated_log_probs - rollout.old_log_probs) * rollout.action_mask
            ).sum(dim=1) / rollout.action_mask.sum(dim=1).clamp_min(1)
            aligned_change = (per_sequence_change * rollout.advantages).mean()
            post_update_loss = compute_grpo_loss(
                updated_log_probs,
                rollout.old_log_probs,
                rollout.reference_log_probs,
                rollout.action_mask,
                rollout.advantages,
                clip_epsilon=self.config.clip_epsilon,
                kl_beta=self.config.kl_beta,
                loss_type=self.config.loss_type,
            )

        for key in iteration_metrics[0]:
            metrics[key] = sum(item[key] for item in iteration_metrics) / len(
                iteration_metrics
            )
        metrics.update(
            {
                "grpo/grad_norm": last_grad_norm,
                "grpo/post_update_kl": post_update_loss.metrics["grpo/kl"],
                "grpo/post_update_clip_fraction": post_update_loss.metrics[
                    "grpo/clip_fraction"
                ],
                "grpo/post_update_ratio_mean": post_update_loss.metrics[
                    "grpo/ratio_mean"
                ],
                "grpo/post_update_old_policy_approx_kl": post_update_loss.metrics[
                    "grpo/old_policy_approx_kl"
                ],
                "grpo/advantage_weighted_logprob_change": float(
                    aligned_change.detach().cpu()
                ),
                "grpo/mean_abs_logprob_change": float(
                    per_sequence_change.abs().mean().detach().cpu()
                ),
                "grpo/optimization_iterations": float(self.config.num_iterations),
                "grpo/optimizer_step": float(self.optimization_step),
                "grpo/step": float(self.global_step + 1),
            }
        )
        self.global_step += 1
        if restore_training_mode:
            self.policy.train()
        return GRPOStepOutput(metrics=metrics, rollout=rollout)
