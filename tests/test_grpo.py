import copy
import math
import os
from types import MethodType, SimpleNamespace

import pytest
import torch

from prot2mol.core import model as model_module
from prot2mol.rewards.fusiondti import (
    FusionDTIActivityHead,
    FusionDTIActivityScorer,
    FusionDTISelfiesTokenizer,
)
from prot2mol.training.grpo import (
    GRPOConfig,
    GRPOTrainer,
    RewardModelProbabilityScorer,
    compute_grpo_loss,
    grouped_advantages,
)


class SelfiesTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    vocab_size = 16

    def __len__(self):
        return self.vocab_size

    def batch_decode(self, rows, skip_special_tokens=True, **kwargs):
        token_to_selfies = {3: "[C]", 4: "[O]", 5: "[N]", 6: "[F]"}
        return [
            "".join(token_to_selfies.get(token, "") for token in row)
            for row in rows
        ]


def _tiny_gpt2_policy(monkeypatch):
    from conftest import DummyProteinEncoder

    monkeypatch.setattr(
        model_module,
        "get_protein_encoder",
        lambda model_name, model_id, active: DummyProteinEncoder(hidden_size=8),
    )
    tokenizer = SelfiesTokenizer()
    policy = model_module.create_prot2mol_model(
        {
            "prot_emb_model": "esm2",
            "protein_model_id": "dummy/protein",
            "decoder_type": "gpt2",
            "decoder_model_id": "from-scratch",
            "n_layer": 1,
            "n_head": 2,
            "n_emb": 8,
            "max_mol_len": 8,
            "conditioning_dropout": 0.0,
            "train_encoder_model": False,
            "train_projection_model": False,
            "train_decoder_model": True,
            "mol_tokenizer": tokenizer,
        }
    )
    return policy, tokenizer


def test_grouped_advantages_are_normalized_within_each_protein_group():
    advantages, means, standard_deviations = grouped_advantages(
        torch.tensor([0.0, 1.0, 2.0, 3.0, 0.5, 0.5, 0.5, 0.5]),
        batch_size=2,
        group_size=4,
        epsilon=1.0e-6,
    )

    assert means.tolist() == pytest.approx([1.5, 0.5])
    assert standard_deviations.tolist() == pytest.approx([math.sqrt(1.25), 0.0])
    assert advantages[:4].mean().item() == pytest.approx(0.0, abs=1.0e-6)
    assert advantages[:4].std(unbiased=False).item() == pytest.approx(1.0)
    assert torch.equal(advantages[4:], torch.zeros(4))


def test_grpo_gradient_favors_positive_advantage_and_suppresses_negative():
    current_log_probs = torch.nn.Parameter(torch.zeros(2, 1))
    old_log_probs = torch.zeros(2, 1)
    output = compute_grpo_loss(
        current_log_probs,
        old_log_probs,
        old_log_probs,
        torch.ones(2, 1, dtype=torch.bool),
        torch.tensor([1.0, -1.0]),
        clip_epsilon=0.2,
        kl_beta=0.0,
    )

    output.loss.backward()

    assert current_log_probs.grad[0, 0] < 0.0
    assert current_log_probs.grad[1, 0] > 0.0


def test_grpo_reports_clipped_out_of_range_policy_ratios():
    current_log_probs = torch.tensor([[math.log(1.5)], [math.log(0.5)]])
    zeros = torch.zeros_like(current_log_probs)
    output = compute_grpo_loss(
        current_log_probs,
        zeros,
        zeros,
        torch.ones_like(current_log_probs, dtype=torch.bool),
        torch.tensor([1.0, -1.0]),
        clip_epsilon=0.2,
        kl_beta=0.0,
    )

    assert output.policy_loss.item() == pytest.approx(-0.2)
    assert output.metrics["grpo/clip_fraction"] == pytest.approx(1.0)
    assert output.metrics["grpo/old_policy_approx_kl"] > 0.0


def test_real_grpo_normalizes_each_completion_before_batch_mean():
    zeros = torch.zeros(2, 3)
    action_mask = torch.tensor(
        [[True, False, False], [True, True, True]],
    )
    advantages = torch.tensor([1.0, -1.0])

    grpo = compute_grpo_loss(
        zeros,
        zeros,
        zeros,
        action_mask,
        advantages,
        clip_epsilon=0.2,
        kl_beta=0.0,
        loss_type="grpo",
    )
    token_normalized = compute_grpo_loss(
        zeros,
        zeros,
        zeros,
        action_mask,
        advantages,
        clip_epsilon=0.2,
        kl_beta=0.0,
        loss_type="bnpo",
    )

    assert grpo.policy_loss.item() == pytest.approx(0.0)
    assert token_normalized.policy_loss.item() == pytest.approx(0.5)
    assert grpo.metrics["grpo/sequence_normalized_loss"] == 1.0


def test_reward_probability_scorer_freezes_reward_model():
    class FakeRewardModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def score_pairs(self, protein_sequences, molecule_sequences):
            assert protein_sequences == ["AAAA", "BBBB"]
            assert molecule_sequences == ["C", "O"]
            return SimpleNamespace(activity_probability=torch.tensor([0.25, 0.75]))

    reward_model = FakeRewardModel()
    scorer = RewardModelProbabilityScorer(reward_model)
    rewards = scorer(["AAAA", "BBBB"], ["C", "O"])

    assert rewards.tolist() == pytest.approx([0.25, 0.75])
    assert reward_model.training is False
    assert all(not parameter.requires_grad for parameter in reward_model.parameters())


def test_fake_gpt2_grpo_step_runs_rollout_reward_and_policy_update(monkeypatch):
    torch.manual_seed(7)
    policy, tokenizer = _tiny_gpt2_policy(monkeypatch)
    reference_policy = copy.deepcopy(policy)
    fixed_generations = torch.tensor(
        [
            [1, 3, 2, 0, 0],
            [1, 4, 2, 0, 0],
            [1, 3, 4, 2, 0],
            [1, 5, 2, 0, 0],
            [1, 6, 2, 0, 0],
            [1, 4, 4, 2, 0],
            [1, 3, 3, 2, 0],
            [1, 5, 3, 2, 0],
            [1, 3, 2, 0, 0],
            [1, 4, 2, 0, 0],
            [1, 3, 4, 2, 0],
            [1, 5, 2, 0, 0],
            [1, 6, 2, 0, 0],
            [1, 4, 4, 2, 0],
            [1, 3, 3, 2, 0],
            [1, 5, 3, 2, 0],
        ],
        dtype=torch.long,
    )

    def deterministic_generate(self, protein_embeddings, prot_attention_mask, **kwargs):
        assert kwargs["do_sample"] is True
        assert kwargs["max_length"] == 5
        assert protein_embeddings.size(0) == fixed_generations.size(0)
        return fixed_generations.to(protein_embeddings.device)

    policy.generate_from_protein_embeddings = MethodType(deterministic_generate, policy)
    score_by_smiles = {
        "C": 0.2,
        "O": 0.9,
        "CO": 0.6,
        "N": 0.4,
        "F": 0.3,
        "OO": 0.7,
        "CC": 0.5,
        "CN": 0.8,
    }

    def fake_reward(proteins, molecules):
        assert len(proteins) == len(molecules) == 16
        return [score_by_smiles[molecule] for molecule in molecules]

    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=1.0e-2,
    )
    trainer = GRPOTrainer(
        policy=policy,
        reference_policy=reference_policy,
        optimizer=optimizer,
        tokenizer=tokenizer,
        reward_function=fake_reward,
        config=GRPOConfig(
            group_size=8,
            max_length=5,
            min_length=3,
            num_iterations=1,
            kl_beta=0.01,
        ),
    )
    policy_before = {
        name: parameter.detach().clone() for name, parameter in policy.named_parameters()
    }
    reference_before = {
        name: parameter.detach().clone()
        for name, parameter in reference_policy.named_parameters()
    }

    result = trainer.step(
        protein_input_ids=torch.tensor([[1, 2, 3, 0], [4, 5, 6, 0]]),
        protein_attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]]),
        protein_sequences=["AAAA", "BBBB"],
    )

    assert result.rollout.generated_selfies == [
        "[C]",
        "[O]",
        "[C][O]",
        "[N]",
        "[F]",
        "[O][O]",
        "[C][C]",
        "[N][C]",
        "[C]",
        "[O]",
        "[C][O]",
        "[N]",
        "[F]",
        "[O][O]",
        "[C][C]",
        "[N][C]",
    ]
    assert result.rollout.generated_smiles == [
        "C",
        "O",
        "CO",
        "N",
        "F",
        "OO",
        "CC",
        "CN",
        "C",
        "O",
        "CO",
        "N",
        "F",
        "OO",
        "CC",
        "CN",
    ]
    assert result.rollout.rewards.tolist() == pytest.approx(
        [
            0.2,
            0.9,
            0.6,
            0.4,
            0.3,
            0.7,
            0.5,
            0.8,
            0.2,
            0.9,
            0.6,
            0.4,
            0.3,
            0.7,
            0.5,
            0.8,
        ]
    )
    assert result.rollout.advantages.reshape(2, 8).mean(dim=1).tolist() == pytest.approx(
        [0.0, 0.0], abs=1.0e-6
    )
    assert all(math.isfinite(value) for value in result.metrics.values())
    assert result.metrics["grpo/num_sequences"] == 16.0
    assert result.metrics["grpo/valid_fraction"] == 1.0
    assert result.metrics["grpo/group_unique_fraction"] == 1.0
    assert result.metrics["grpo/terminated_fraction"] == 1.0
    assert result.metrics["grpo/truncated_fraction"] == 0.0
    assert result.metrics["grpo/property_count"] == 16.0
    assert 0.0 <= result.metrics["grpo/qed_mean"] <= 1.0
    assert result.metrics["grpo/sas_mean"] > 0.0
    assert math.isfinite(result.metrics["grpo/logp_mean"])
    assert result.metrics["grpo/zero_variance_group_fraction"] == 0.0
    assert result.metrics["grpo/grad_norm"] > 0.0
    assert result.metrics["grpo/mean_abs_logprob_change"] > 0.0
    assert result.metrics["grpo/advantage_weighted_logprob_change"] > 0.0
    assert result.metrics["grpo/post_update_kl"] >= 0.0
    assert any(
        not torch.equal(parameter, policy_before[name])
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    )
    assert all(
        torch.equal(parameter, reference_before[name])
        for name, parameter in reference_policy.named_parameters()
    )
    assert all(
        not parameter.requires_grad for parameter in reference_policy.parameters()
    )


def test_grpo_requires_eos_before_a_chemically_valid_generation_can_be_rewarded(
    monkeypatch,
):
    policy, tokenizer = _tiny_gpt2_policy(monkeypatch)
    reference_policy = copy.deepcopy(policy)
    fixed_generations = torch.tensor(
        [
            [1, 3, 2, 0, 0],
            [1, 3, 3, 3, 3],
        ],
        dtype=torch.long,
    )

    def deterministic_generate(self, protein_embeddings, prot_attention_mask, **kwargs):
        return fixed_generations.to(protein_embeddings.device)

    policy.generate_from_protein_embeddings = MethodType(deterministic_generate, policy)
    reward_calls = []

    def reward(proteins, molecules):
        reward_calls.append((list(proteins), list(molecules)))
        return [0.8]

    trainer = GRPOTrainer(
        policy=policy,
        reference_policy=reference_policy,
        optimizer=torch.optim.AdamW(
            [parameter for parameter in policy.parameters() if parameter.requires_grad],
            lr=1.0e-2,
        ),
        tokenizer=tokenizer,
        reward_function=reward,
        config=GRPOConfig(group_size=2, max_length=5, min_length=3),
    )

    rollout = trainer.collect_rollouts(
        protein_input_ids=torch.tensor([[1, 2, 3, 0]]),
        protein_attention_mask=torch.tensor([[1, 1, 1, 0]]),
        protein_sequences=["AAAA"],
    )
    metrics = trainer._rollout_metrics(rollout)

    assert rollout.chemically_valid_mask.tolist() == [True, True]
    assert rollout.terminated_mask.tolist() == [True, False]
    assert rollout.valid_mask.tolist() == [True, False]
    assert rollout.rewards.tolist() == pytest.approx([0.8, 0.0])
    assert reward_calls == [(["AAAA"], ["C"])]
    assert metrics["grpo/chemical_valid_fraction"] == 1.0
    assert metrics["grpo/valid_and_terminated_fraction"] == 0.5
    assert metrics["grpo/truncated_fraction"] == 0.5


def test_fusiondti_ablation_runs_structure_aware_reward_feedback_to_gpt2(monkeypatch):
    class StructureAwareTokenizer:
        def __init__(self):
            self.calls = []

        def __call__(self, sequences, *, max_length, **kwargs):
            self.calls.append(list(sequences))
            rows = []
            masks = []
            for sequence in sequences:
                ids = [1]
                ids.extend((ord(character) % 31) + 3 for character in sequence)
                ids.append(2)
                ids = ids[:max_length]
                mask = [1] * len(ids)
                padding = max_length - len(ids)
                rows.append(ids + [0] * padding)
                masks.append(mask + [0] * padding)
            return {
                "input_ids": torch.tensor(rows, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }

    class ProteinLogitEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(64, 6)

        def forward(self, input_ids, attention_mask, return_dict=True):
            return SimpleNamespace(logits=self.embedding(input_ids))

    class MoleculeHiddenStateEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 5)

        def forward(self, input_ids, attention_mask, return_dict=True):
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    torch.manual_seed(17)
    policy, tokenizer = _tiny_gpt2_policy(monkeypatch)
    reference_policy = copy.deepcopy(policy)
    fixed_generations = torch.tensor(
        [
            [1, 3, 2, 0, 0],
            [1, 4, 2, 0, 0],
            [1, 5, 2, 0, 0],
            [1, 6, 2, 0, 0],
            [1, 3, 4, 2, 0],
            [1, 4, 4, 2, 0],
            [1, 3, 3, 2, 0],
            [1, 5, 3, 2, 0],
        ],
        dtype=torch.long,
    )

    def deterministic_generate(self, protein_embeddings, prot_attention_mask, **kwargs):
        return fixed_generations.to(protein_embeddings.device)

    policy.generate_from_protein_embeddings = MethodType(deterministic_generate, policy)
    protein_tokenizer = StructureAwareTokenizer()
    molecule_tokenizer = FusionDTISelfiesTokenizer(
        vocab={
            "<unk>": 0,
            "<s>": 1,
            "</s>": 2,
            "<pad>": 3,
            "C": 4,
            "O": 5,
            "N": 6,
            "F": 7,
        },
        special_tokens={
            "unk_token": "<unk>",
            "cls_token": "<s>",
            "sep_token": "</s>",
            "pad_token": "<pad>",
        },
    )
    reward_scorer = FusionDTIActivityScorer(
        protein_encoder=ProteinLogitEncoder(),
        molecule_encoder=MoleculeHiddenStateEncoder(),
        activity_head=FusionDTIActivityHead(
            protein_dim=6,
            molecule_dim=5,
            hidden_dim=8,
            num_heads=2,
        ),
        protein_tokenizer=protein_tokenizer,
        molecule_tokenizer=molecule_tokenizer,
        max_length=8,
        batch_size=8,
        device="cpu",
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=1.0e-2,
    )
    trainer = GRPOTrainer(
        policy=policy,
        reference_policy=reference_policy,
        optimizer=optimizer,
        tokenizer=tokenizer,
        reward_function=reward_scorer,
        config=GRPOConfig(
            group_size=8,
            max_length=5,
            min_length=3,
            num_iterations=1,
            precision="bf16",
        ),
    )
    policy_before = {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    }

    result = trainer.step(
        protein_input_ids=torch.tensor([[1, 2, 3, 0]]),
        protein_attention_mask=torch.tensor([[1, 1, 1, 0]]),
        protein_sequences=["ACDE"],
        reward_protein_sequences=["MdEvLp"],
    )

    assert result.rollout.reward_molecule_representation == "selfies"
    assert result.rollout.reward_protein_sequences == ["MdEvLp"] * 8
    assert protein_tokenizer.calls == [["MdEvLp"]]
    assert result.rollout.rewards.std(unbiased=False) > 0.0
    assert result.metrics["grpo/num_sequences"] == 8.0
    assert result.metrics["grpo/precision_bf16"] == 1.0
    assert result.metrics["grpo/nonzero_advantage_fraction"] > 0.0
    assert result.metrics["grpo/grad_norm"] > 0.0
    assert 0.0 <= result.metrics[
        "grpo/valid_reward_low_saturation_fraction"
    ] <= 1.0
    assert 0.0 <= result.metrics[
        "grpo/valid_reward_high_saturation_fraction"
    ] <= 1.0
    assert any(
        not torch.equal(parameter, policy_before[name])
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    )
    assert all(not parameter.requires_grad for parameter in reward_scorer.parameters())


@pytest.mark.skipif(
    os.environ.get("PROT2MOL_RUN_FUSIONDTI_LIVE") != "1",
    reason="set PROT2MOL_RUN_FUSIONDTI_LIVE=1 to load published encoders",
)
def test_published_fusiondti_probabilities_drive_real_gpt2_grpo_update(monkeypatch):
    torch.manual_seed(29)
    policy, tokenizer = _tiny_gpt2_policy(monkeypatch)
    reference_policy = copy.deepcopy(policy)
    fixed_generations = torch.tensor(
        [
            [1, 3, 2, 0, 0],
            [1, 4, 2, 0, 0],
            [1, 5, 2, 0, 0],
            [1, 6, 2, 0, 0],
            [1, 3, 4, 2, 0],
            [1, 4, 4, 2, 0],
            [1, 3, 3, 2, 0],
            [1, 5, 3, 2, 0],
        ],
        dtype=torch.long,
    )

    def deterministic_generate(self, protein_embeddings, prot_attention_mask, **kwargs):
        return fixed_generations.to(protein_embeddings.device)

    policy.generate_from_protein_embeddings = MethodType(deterministic_generate, policy)
    reward_scorer = FusionDTIActivityScorer.from_pretrained(
        dataset="BindingDB",
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=8,
        max_length=32,
        local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=1.0e-2,
    )
    trainer = GRPOTrainer(
        policy=policy,
        reference_policy=reference_policy,
        optimizer=optimizer,
        tokenizer=tokenizer,
        reward_function=reward_scorer,
        config=GRPOConfig(group_size=8, max_length=5, min_length=3),
    )
    policy_before = {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    }

    result = trainer.step(
        protein_input_ids=torch.tensor([[1, 2, 3, 0]]),
        protein_attention_mask=torch.tensor([[1, 1, 1, 0]]),
        protein_sequences=["ACDE"],
        reward_protein_sequences=["MdEvLp"],
    )

    assert result.rollout.rewards.std(unbiased=False) > 0.0
    assert result.metrics["grpo/nonzero_advantage_fraction"] > 0.0
    assert result.metrics["grpo/grad_norm"] > 0.0
    assert result.metrics["grpo/mean_abs_logprob_change"] > 0.0
    assert result.metrics["grpo/advantage_weighted_logprob_change"] > 0.0
    assert any(
        not torch.equal(parameter, policy_before[name])
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    )
