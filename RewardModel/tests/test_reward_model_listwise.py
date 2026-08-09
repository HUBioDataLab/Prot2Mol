import copy
import math
from contextlib import contextmanager

import pytest
import torch
from accelerate.data_loader import BatchSamplerShard
from datasets import Dataset
from torch.utils.data import BatchSampler, DataLoader

from conftest import DummyEncoder, DummyTokenizer
from reward_model.model import (
    DEFAULT_CONTRASTIVE_ACTIVE_THRESHOLD,
    DEFAULT_RANKING_AFFINITY_MARGIN,
    LoadedEncoder,
    RewardModel,
    RewardModelConfig,
    ligunity_bidirectional_contrastive_loss,
    ligunity_listwise_loss,
)
from reward_model.training import (
    AssayListEpochSampler,
    RewardAssayListCollator,
    RewardAssayListDataset,
    RewardEvaluationDataset,
    RewardModelTrainer,
    RewardTrainerConfig,
    build_complete_coverage_ranking_partitions,
    compute_contrastive_evaluation_loss,
    compute_joint_evaluation_metrics,
    create_training_arguments,
)


def _tokenized_rows(group_sizes=(40, 20, 3, 1)):
    rows = []
    example_id = 0
    for group_index, group_size in enumerate(group_sizes):
        group_id = f"T{group_index}__A{group_index}"
        for member_index in range(group_size):
            if group_index == 2:
                pchembl = 6.0 + 0.1 * (member_index % 3)
            else:
                pchembl = 4.0 + 4.0 * member_index / max(1, group_size - 1)
            rows.append(
                {
                    "example_id": example_id,
                    "group_id": group_id,
                    "target_chembl_id": f"T{group_index}",
                    "assay_id": f"A{group_index}",
                    "compound_id": f"M{example_id}",
                    "pchembl_value": pchembl,
                    "binary_label": int(pchembl >= 6.0),
                    "protein_input_ids": [group_index + 1, 2, 0],
                    "protein_attention_mask": [1, 1, 0],
                    "protein_length": 2,
                    "molecule_input_ids": [member_index + 1, 3, 0, 0],
                    "molecule_attention_mask": [1, 1, 0, 0],
                    "molecule_length": 2,
                }
            )
            example_id += 1
    return Dataset.from_list(rows)


def _dummy_model():
    return RewardModel(
        config=RewardModelConfig(
            protein_model_name_or_path="protein/dummy",
            molecule_model_name_or_path="molecule/dummy",
            fusion_hidden_dim=10,
            fusion_num_heads=2,
            dropout=0.0,
            ranking_temperature=1.0,
            ranking_min_pchembl_span=0.5,
        ),
        protein_bundle=LoadedEncoder(
            name_or_path="protein/dummy",
            tokenizer=DummyTokenizer(),
            model=DummyEncoder(hidden_size=6),
            hidden_size=6,
        ),
        molecule_bundle=LoadedEncoder(
            name_or_path="molecule/dummy",
            tokenizer=DummyTokenizer(),
            model=DummyEncoder(hidden_size=8),
            hidden_size=8,
        ),
    )


def _reference_unique_ligunity_loss(
    scores,
    targets,
    temperature=1.0,
    affinity_margin=DEFAULT_RANKING_AFFINITY_MARGIN,
):
    order = torch.argsort(targets, descending=True)
    ordered_scores = scores[order] / temperature
    ordered_targets = targets[order]
    n = ordered_scores.numel()
    terms = []
    for index in range(n):
        weight = 1.0 / (math.sqrt(n) * math.log(index + 2.0))
        weaker_scores = ordered_scores[
            ordered_targets < ordered_targets[index] - affinity_margin
        ]
        candidates = torch.cat([ordered_scores[index : index + 1], weaker_scores])
        terms.append(weight * (torch.logsumexp(candidates, dim=0) - ordered_scores[index]))
    return torch.stack(terms).sum()


def _reference_ligunity_contrastive_loss(
    scores,
    pchembl_values,
    ligand_group_ids,
    group_target_ids,
    molecule_identity_ids,
    active_threshold=DEFAULT_CONTRASTIVE_ACTIVE_THRESHOLD,
):
    """Literal CPU/device-neutral translation of LigUnity's released loops."""
    scores = scores.float() if scores.dtype in (torch.float16, torch.bfloat16) else scores
    num_groups = scores.shape[0]
    masked_scores = scores.clone()
    for group_index in range(num_groups):
        own_molecules = molecule_identity_ids[ligand_group_ids == group_index]
        for other_group_index in range(num_groups):
            if other_group_index == group_index:
                continue
            other_indices = torch.nonzero(
                ligand_group_ids == other_group_index,
                as_tuple=False,
            ).flatten()
            if group_target_ids[group_index] == group_target_ids[other_group_index]:
                masked_scores[group_index, other_indices] += -1e9
            for ligand_index_tensor in other_indices:
                ligand_index = int(ligand_index_tensor.item())
                if (own_molecules == molecule_identity_ids[ligand_index]).any():
                    masked_scores[group_index, ligand_index] += -1e9

    protein_to_molecule_terms = []
    molecule_to_protein_terms = []
    for group_index in range(num_groups):
        own_indices = torch.nonzero(
            ligand_group_ids == group_index,
            as_tuple=False,
        ).flatten()
        group_size = int(own_indices.numel())
        for ligand_index_tensor in own_indices:
            ligand_index = int(ligand_index_tensor.item())
            mask = torch.zeros_like(masked_scores[group_index])
            mask[own_indices] = -1e9
            mask[ligand_index] = 0.0
            loss = torch.nn.functional.nll_loss(
                torch.nn.functional.log_softmax(
                    mask + masked_scores[group_index],
                    dim=-1,
                ),
                torch.tensor(ligand_index, device=scores.device),
                reduction="sum",
            )
            if (
                group_size > 1
                and pchembl_values[ligand_index] < active_threshold
            ):
                continue
            protein_to_molecule_terms.append(loss / math.sqrt(group_size))

    ligand_to_group = ligand_group_ids.long()
    molecule_to_protein_per_ligand = torch.nn.functional.nll_loss(
        torch.nn.functional.log_softmax(masked_scores.transpose(0, 1), dim=-1),
        ligand_to_group,
        reduction="none",
    )
    for group_index in range(num_groups):
        own = ligand_group_ids == group_index
        molecule_to_protein_terms.append(
            molecule_to_protein_per_ligand[own].sum()
            / math.sqrt(int(own.sum().item()))
        )

    protein_to_molecule = torch.stack(protein_to_molecule_terms).sum() / num_groups
    molecule_to_protein = torch.stack(molecule_to_protein_terms).sum() / num_groups
    return (
        protein_to_molecule + molecule_to_protein,
        protein_to_molecule,
        molecule_to_protein,
    )


def _contrastive_reference_inputs(*, requires_grad=False, dtype=torch.float64):
    scores = torch.tensor(
        [
            [1.2, 0.7, -0.4, 2.5, 0.1, -0.3, 0.4],
            [-0.2, 0.3, 1.1, 1.4, 0.8, 0.2, -0.5],
            [0.4, -0.6, 0.2, -0.1, 0.7, 1.3, 0.9],
        ],
        dtype=dtype,
        requires_grad=requires_grad,
    )
    return (
        scores,
        torch.tensor([7.0, 4.5, 8.0, 6.5, 4.0, 7.5, 5.5], dtype=dtype),
        torch.tensor([0, 0, 1, 1, 1, 2, 2]),
        torch.tensor([10, 10, 20]),
        torch.tensor([100, 101, 102, 100, 103, 104, 105]),
    )


def test_ligunity_contrastive_loss_matches_released_reference_and_gradients():
    inputs = _contrastive_reference_inputs(requires_grad=True)
    actual = ligunity_bidirectional_contrastive_loss(*inputs)
    expected = _reference_ligunity_contrastive_loss(*inputs)

    actual_gradient = torch.autograd.grad(actual[0], inputs[0], retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected[0], inputs[0])[0]

    for actual_value, expected_value in zip(actual, expected):
        assert torch.allclose(actual_value, expected_value, rtol=1e-12, atol=1e-12)
    assert torch.allclose(
        actual_gradient,
        expected_gradient,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("seed", [3, 7, 19, 31])
def test_ligunity_contrastive_randomized_values_and_gradients_match_reference(seed):
    generator = torch.Generator().manual_seed(seed)
    scores = torch.randn(
        (3, 7),
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    pchembl = 4.0 + 4.0 * torch.rand(
        7,
        generator=generator,
        dtype=torch.float64,
    )
    owners = torch.tensor([0, 1, 1, 2, 2, 2, 2])
    target_ids = torch.tensor([10, 20, 10])
    molecule_ids = torch.tensor([100, 101, 102, 100, 103, 104, 105])

    actual = ligunity_bidirectional_contrastive_loss(
        scores,
        pchembl,
        owners,
        target_ids,
        molecule_ids,
    )
    expected = _reference_ligunity_contrastive_loss(
        scores,
        pchembl,
        owners,
        target_ids,
        molecule_ids,
    )
    actual_gradient = torch.autograd.grad(actual[0], scores, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected[0], scores)[0]

    for actual_value, expected_value in zip(actual, expected):
        assert torch.allclose(actual_value, expected_value, rtol=1e-12, atol=1e-12)
    assert torch.allclose(
        actual_gradient,
        expected_gradient,
        rtol=1e-12,
        atol=1e-12,
    )


def test_ligunity_contrastive_masks_same_target_and_duplicate_molecule_scores():
    inputs = _contrastive_reference_inputs()
    baseline = ligunity_bidirectional_contrastive_loss(*inputs)
    changed_scores = inputs[0].clone()
    # Groups 0 and 1 share a target, so every cross-assay entry is masked.
    changed_scores[0, 2:5] = 1e6
    changed_scores[1, 0:2] = -1e6
    # Molecule 100 occurs in group 0 column 0 and group 1 column 3.
    changed_scores[0, 3] = -1e6
    changed_scores[1, 0] = 1e6
    changed = ligunity_bidirectional_contrastive_loss(changed_scores, *inputs[1:])

    for baseline_value, changed_value in zip(baseline, changed):
        assert torch.allclose(baseline_value, changed_value, rtol=0.0, atol=1e-12)


def test_ligunity_contrastive_active_filter_is_directionally_asymmetric():
    scores = torch.tensor(
        [[1.0, -0.5, 0.3, 0.2], [0.1, 0.4, 1.2, -0.7]],
        dtype=torch.float64,
    )
    pchembl = torch.tensor([7.0, 4.0, 8.0, 6.0], dtype=torch.float64)
    owners = torch.tensor([0, 0, 1, 1])
    target_ids = torch.tensor([10, 20])
    molecule_ids = torch.tensor([100, 101, 102, 103])
    baseline = ligunity_bidirectional_contrastive_loss(
        scores,
        pchembl,
        owners,
        target_ids,
        molecule_ids,
    )
    changed_scores = scores.clone()
    changed_scores[0, 1] = 20.0
    changed = ligunity_bidirectional_contrastive_loss(
        changed_scores,
        pchembl,
        owners,
        target_ids,
        molecule_ids,
    )

    assert changed[1] == pytest.approx(baseline[1].item(), abs=1e-12)
    assert changed[2] != pytest.approx(baseline[2].item(), abs=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ligunity_contrastive_promotes_low_precision_math(dtype):
    inputs = list(_contrastive_reference_inputs(requires_grad=True, dtype=dtype))
    loss, protein_to_molecule, molecule_to_protein = (
        ligunity_bidirectional_contrastive_loss(*inputs)
    )
    loss.backward()

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert torch.isfinite(protein_to_molecule)
    assert torch.isfinite(molecule_to_protein)
    assert inputs[0].grad is not None
    assert torch.isfinite(inputs[0].grad).all()


def test_ligunity_loss_matches_equation_reference_for_unique_affinities():
    scores = torch.tensor([0.3, -0.2, 1.2, 0.7], dtype=torch.float64)
    targets = torch.tensor([7.0, 5.0, 8.0, 6.0], dtype=torch.float64)
    group_ids = torch.zeros(4, dtype=torch.long)

    actual = ligunity_listwise_loss(
        scores,
        targets,
        group_ids,
        temperature=0.7,
        min_pchembl_span=0.5,
    )
    expected = _reference_unique_ligunity_loss(scores, targets, temperature=0.7)

    assert actual == pytest.approx(expected.item(), rel=1e-12, abs=1e-12)


def test_ligunity_loss_matches_reference_with_mixed_affinity_gaps():
    scores = torch.tensor([0.4, -0.7, 1.1, 0.2], dtype=torch.float64)
    targets = torch.tensor([8.0, 7.8, 7.0, 6.9], dtype=torch.float64)
    groups = torch.zeros(4, dtype=torch.long)

    actual = ligunity_listwise_loss(
        scores,
        targets,
        groups,
        temperature=0.8,
        min_pchembl_span=0.0,
    )
    expected = _reference_unique_ligunity_loss(
        scores,
        targets,
        temperature=0.8,
    )

    assert actual == pytest.approx(expected.item(), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    ("seed", "list_size", "temperature", "affinity_margin"),
    [
        (3, 3, 1.0, 0.0),
        (7, 5, 0.3, DEFAULT_RANKING_AFFINITY_MARGIN),
        (11, 9, 1.7, 0.25),
        (19, 16, 0.8, 0.75),
    ],
)
def test_ligunity_loss_randomized_values_and_gradients_match_reference(
    seed,
    list_size,
    temperature,
    affinity_margin,
):
    generator = torch.Generator().manual_seed(seed)
    scores = torch.randn(
        list_size,
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    targets = 5.0 + 3.0 * torch.rand(
        list_size,
        generator=generator,
        dtype=torch.float64,
    )
    groups = torch.zeros(list_size, dtype=torch.long)

    actual = ligunity_listwise_loss(
        scores,
        targets,
        groups,
        temperature=temperature,
        min_pchembl_span=0.0,
        affinity_margin=affinity_margin,
    )
    expected = _reference_unique_ligunity_loss(
        scores,
        targets,
        temperature=temperature,
        affinity_margin=affinity_margin,
    )
    actual_gradient = torch.autograd.grad(actual, scores, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, scores)[0]

    assert torch.allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert torch.allclose(
        actual_gradient,
        expected_gradient,
        rtol=1e-12,
        atol=1e-12,
    )


def test_ligunity_margin_excludes_threefold_or_smaller_affinity_gaps():
    margin = DEFAULT_RANKING_AFFINITY_MARGIN
    scores = torch.tensor([0.3, -0.2, 1.4], dtype=torch.float64, requires_grad=True)
    at_margin_targets = torch.tensor(
        [8.0, 8.0 - margin, 8.0 - margin],
        dtype=torch.float64,
    )
    groups = torch.zeros(3, dtype=torch.long)

    at_margin_loss = ligunity_listwise_loss(
        scores,
        at_margin_targets,
        groups,
        min_pchembl_span=0.0,
    )
    at_margin_loss.backward()

    assert at_margin_loss.item() == pytest.approx(0.0, abs=1e-15)
    assert scores.grad is not None
    assert scores.grad.abs().sum().item() == pytest.approx(0.0, abs=1e-15)

    scores.grad = None
    beyond_margin_targets = at_margin_targets.clone()
    beyond_margin_targets[1:] -= 1.0e-6
    beyond_margin_loss = ligunity_listwise_loss(
        scores,
        beyond_margin_targets,
        groups,
        min_pchembl_span=0.0,
    )
    beyond_margin_loss.backward()

    assert beyond_margin_loss.item() > 0.0
    assert scores.grad is not None
    assert scores.grad.abs().sum().item() > 0.0


def test_ligunity_margin_removes_all_near_affinity_ranking_signal():
    scores = torch.tensor(
        [-100.0, 0.0, 100.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    targets = torch.tensor([8.0, 7.8, 7.7], dtype=torch.float64)
    groups = torch.zeros(3, dtype=torch.long)

    loss = ligunity_listwise_loss(
        scores,
        targets,
        groups,
        min_pchembl_span=0.0,
    )
    loss.backward()

    assert loss.item() == pytest.approx(0.0, abs=1e-15)
    assert scores.grad is not None
    assert scores.grad.abs().sum().item() == pytest.approx(0.0, abs=1e-15)


def test_ligunity_loss_is_tie_permutation_invariant():
    scores = torch.tensor([1.1, -0.4, 0.2, 0.7], dtype=torch.float64)
    targets = torch.tensor([8.0, 8.0, 6.0, 5.0], dtype=torch.float64)
    groups = torch.zeros(4, dtype=torch.long)
    permutation = torch.tensor([1, 0, 2, 3])

    original = ligunity_listwise_loss(scores, targets, groups)
    permuted = ligunity_listwise_loss(
        scores[permutation], targets[permutation], groups[permutation]
    )

    assert original == pytest.approx(permuted.item(), rel=1e-12, abs=1e-12)


def test_ligunity_loss_extreme_logits_have_finite_loss_and_gradients():
    scores = torch.tensor(
        [1.0e4, -1.0e4, 5.0e3, -5.0e3],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor([8.0, 7.0, 6.0, 5.0])
    groups = torch.zeros(4, dtype=torch.long)

    loss = ligunity_listwise_loss(scores, targets, groups, temperature=0.1)
    loss.backward()

    assert torch.isfinite(loss)
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_ligunity_loss_promotes_fp16_math_to_avoid_overflow():
    scores = torch.tensor(
        [-6.0e4, 6.0e4, -3.0e4, 3.0e4],
        dtype=torch.float16,
        requires_grad=True,
    )
    targets = torch.tensor([8.0, 7.0, 6.0, 5.0], dtype=torch.float32)
    groups = torch.zeros(4, dtype=torch.long)

    loss = ligunity_listwise_loss(scores, targets, groups, temperature=0.1)
    loss.backward()

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


@pytest.mark.parametrize(
    ("scores", "targets", "message"),
    [
        ([float("nan"), 0.0], [7.0, 5.0], "ranking_scores"),
        ([0.0, 1.0], [7.0, float("inf")], "pchembl_values"),
    ],
)
def test_ligunity_loss_rejects_nonfinite_inputs(scores, targets, message):
    with pytest.raises(ValueError, match=message):
        ligunity_listwise_loss(
            torch.tensor(scores),
            torch.tensor(targets),
            torch.zeros(2, dtype=torch.long),
        )


@pytest.mark.parametrize("affinity_margin", [-0.1, float("nan"), float("inf")])
def test_ligunity_loss_rejects_invalid_affinity_margin(affinity_margin):
    with pytest.raises(ValueError, match="affinity_margin"):
        ligunity_listwise_loss(
            torch.tensor([0.0, 1.0, 2.0]),
            torch.tensor([8.0, 7.0, 6.0]),
            torch.zeros(3, dtype=torch.long),
            affinity_margin=affinity_margin,
        )


def test_validation_partitions_are_balanced_complete_and_deterministic():
    pchembl_values = torch.cat(
        [
            torch.linspace(4.0, 8.0, 17),
            torch.linspace(4.0, 8.0, 33),
            torch.tensor([4.0, 8.0]),
            torch.tensor([6.0, 6.1, 6.2, 6.3]),
        ]
    )
    assay_ids = torch.cat(
        [
            torch.full((17,), 0, dtype=torch.long),
            torch.full((33,), 1, dtype=torch.long),
            torch.full((2,), 2, dtype=torch.long),
            torch.full((4,), 3, dtype=torch.long),
        ]
    )

    partitions = build_complete_coverage_ranking_partitions(
        pchembl_values=pchembl_values,
        ranking_group_ids=assay_ids,
        max_list_size=16,
        num_partitions=3,
        seed=73,
        min_pchembl_span=0.5,
    )
    replica = build_complete_coverage_ranking_partitions(
        pchembl_values=pchembl_values,
        ranking_group_ids=assay_ids,
        max_list_size=16,
        num_partitions=3,
        seed=73,
        min_pchembl_span=0.5,
    )

    assert len(partitions) == 3
    assert all(torch.equal(left, right) for left, right in zip(partitions, replica))
    assert not torch.equal(partitions[0], partitions[1])
    for partition_ids in partitions:
        assert (partition_ids[:50] >= 0).all()
        assert (partition_ids[50:] < 0).all()
        ranked_ids = partition_ids[partition_ids >= 0]
        list_sizes = torch.bincount(ranked_ids).tolist()
        assert sorted(list_sizes) == [8, 9, 11, 11, 11]
        assert sum(list_sizes) == 50
        assert max(list_sizes) <= 16
        assert min(list_sizes) >= 3
        for list_id in torch.unique(ranked_ids):
            original_assays = torch.unique(assay_ids[partition_ids == list_id])
            assert original_assays.numel() == 1


def test_validation_partition_seed_changes_only_the_fixed_grouping():
    pchembl_values = torch.linspace(4.0, 8.0, 40)
    assay_ids = torch.zeros(40, dtype=torch.long)
    first = build_complete_coverage_ranking_partitions(
        pchembl_values=pchembl_values,
        ranking_group_ids=assay_ids,
        seed=11,
    )
    second = build_complete_coverage_ranking_partitions(
        pchembl_values=pchembl_values,
        ranking_group_ids=assay_ids,
        seed=29,
    )

    assert not torch.equal(first[0], second[0])
    assert (first[0] >= 0).all()
    assert (second[0] >= 0).all()
    assert sorted(torch.bincount(first[0]).tolist()) == [13, 13, 14]
    assert sorted(torch.bincount(second[0]).tolist()) == [13, 13, 14]


@pytest.mark.parametrize(
    "assay_size",
    [3, 4, 5, 15, 16, 17, 31, 32, 33, 47, 48, 49, 100, 257],
)
def test_validation_partitions_cover_assays_across_boundary_sizes(assay_size):
    partitions = build_complete_coverage_ranking_partitions(
        pchembl_values=torch.linspace(4.0, 8.0, assay_size),
        ranking_group_ids=torch.zeros(assay_size, dtype=torch.long),
        max_list_size=16,
        num_partitions=2,
        seed=101,
    )

    for partition_ids in partitions:
        assert (partition_ids >= 0).all()
        list_sizes = torch.bincount(partition_ids).tolist()
        assert sum(list_sizes) == assay_size
        assert max(list_sizes) <= 16
        assert min(list_sizes) >= 3
        assert max(list_sizes) - min(list_sizes) <= 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_list_size": 4}, "max_list_size"),
        ({"num_partitions": 0}, "num_partitions"),
        ({"min_pchembl_span": -0.1}, "min_pchembl_span"),
    ],
)
def test_validation_partitions_reject_invalid_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_complete_coverage_ranking_partitions(
            pchembl_values=torch.tensor([8.0, 7.0, 6.0]),
            ranking_group_ids=torch.zeros(3, dtype=torch.long),
            **kwargs,
        )


def test_contrastive_evaluation_matches_full_coverage_ligunity_loss():
    protein_embeddings = torch.tensor(
        [[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3,
    )
    molecule_embeddings = torch.nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 0.2],
                [0.8, 0.4],
                [0.6, 0.8],
                [0.2, 1.0],
                [0.4, 0.8],
                [0.8, 0.6],
            ]
        ),
        dim=-1,
    )
    pchembl_values = torch.tensor([7.0, 6.0, 5.0, 7.5, 6.5, 5.5])
    assay_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    target_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    molecule_ids = torch.arange(6)
    temperature = 0.1

    actual = compute_contrastive_evaluation_loss(
        normalized_protein_embeddings=protein_embeddings,
        normalized_molecule_embeddings=molecule_embeddings,
        pchembl_values=pchembl_values,
        ranking_group_ids=assay_ids,
        target_identity_ids=target_ids,
        molecule_identity_ids=molecule_ids,
        temperature=temperature,
        active_threshold=5.0,
        assay_batch_size=2,
        ranking_max_ligands=16,
        ranking_num_partitions=2,
        ranking_partition_seed=17,
        ranking_min_pchembl_span=0.5,
    )
    expected, _, _ = ligunity_bidirectional_contrastive_loss(
        torch.matmul(
            protein_embeddings[[0, 3]],
            molecule_embeddings.transpose(0, 1),
        )
        / temperature,
        pchembl_values,
        assay_ids,
        torch.tensor([0, 1]),
        molecule_ids,
        active_threshold=5.0,
    )

    assert actual == pytest.approx(expected.item())


def test_joint_evaluation_averages_complete_coverage_partition_losses():
    pchembl_values = torch.linspace(5.0, 8.0, 20)
    ranking_scores = torch.linspace(-1.0, 1.0, 20)
    assay_ids = torch.zeros(20, dtype=torch.long)
    partitions = build_complete_coverage_ranking_partitions(
        pchembl_values=pchembl_values,
        ranking_group_ids=assay_ids,
        max_list_size=16,
        num_partitions=2,
        seed=17,
        min_pchembl_span=0.5,
    )
    expected_loss = torch.stack(
        [
            ligunity_listwise_loss(
                ranking_scores,
                pchembl_values,
                partition_ids,
                min_pchembl_span=0.0,
            )
            for partition_ids in partitions
        ]
    ).mean()

    metrics, assay_records = compute_joint_evaluation_metrics(
        activity_logits=torch.linspace(-2.0, 2.0, 20),
        ranking_scores=ranking_scores,
        activity_labels=(pchembl_values >= 6.0).float(),
        pchembl_values=pchembl_values,
        ranking_group_ids=assay_ids,
        group_id_names=["T0__A0"],
        classification_loss_weight=1.0,
        ranking_loss_weight=1.0,
        bce_pos_weight=1.0,
        ranking_temperature=1.0,
        ranking_affinity_margin=DEFAULT_RANKING_AFFINITY_MARGIN,
        ranking_min_pchembl_span=0.5,
        ranking_max_ligands=16,
        ranking_num_partitions=2,
        ranking_partition_seed=17,
    )

    assert metrics["eval_ranking_loss"] == pytest.approx(expected_loss.item())
    assert metrics["eval_num_ranking_groups"] == pytest.approx(1.0)
    assert metrics["eval_num_ranking_lists"] == pytest.approx(2.0)
    assert metrics["eval_num_ranked_examples"] == pytest.approx(20.0)
    assert metrics["eval_ranking_partitions"] == pytest.approx(2.0)
    assert len(assay_records) == 1


def test_assay_list_dataset_has_exact_coverage_and_dynamic_nonoverlapping_lists():
    examples = _tokenized_rows()
    dataset = RewardAssayListDataset(
        examples,
        seed=42,
        ranking_max_ligands=16,
        ranking_opportunity_divisor=32,
        ranking_min_pchembl_span=0.5,
        max_classification_only_per_item=4,
        item_count_multiple=4,
    )

    assert dataset.stats.num_examples == 64
    assert dataset.stats.num_assays == 4
    assert dataset.stats.num_eligible_assays == 2
    assert dataset.stats.num_ranking_lists == 3
    assert dataset.stats.num_ranked_examples == 48
    assert dataset.stats.num_classification_only_examples == 16
    assert dataset.stats.num_dataset_items % 4 == 0
    assert sorted(dataset.epoch_example_indices()) == list(range(64))
    assert len(set(dataset.epoch_ranking_indices())) == 48

    epoch_zero_ranking = set(dataset.epoch_ranking_indices())
    dataset.set_epoch(1)
    epoch_one_ranking = set(dataset.epoch_ranking_indices())
    assert sorted(dataset.epoch_example_indices()) == list(range(64))
    assert len(epoch_one_ranking) == 48
    assert epoch_zero_ranking != epoch_one_ranking

    replica = RewardAssayListDataset(
        examples,
        seed=42,
        ranking_max_ligands=16,
        ranking_opportunity_divisor=32,
        ranking_min_pchembl_span=0.5,
        max_classification_only_per_item=4,
        item_count_multiple=4,
    )
    replica.set_epoch(1)
    assert replica.epoch_ranking_indices() == dataset.epoch_ranking_indices()


def test_ranking_only_dataset_omits_classification_only_examples():
    examples = _tokenized_rows()
    dataset = RewardAssayListDataset(
        examples,
        seed=42,
        ranking_max_ligands=16,
        ranking_opportunity_divisor=32,
        ranking_min_pchembl_span=0.5,
        max_classification_only_per_item=0,
    )

    assert dataset.stats.num_ranking_lists == 3
    assert dataset.stats.num_ranked_examples == 48
    assert dataset.stats.num_classification_only_examples == 0
    assert len(dataset) == dataset.stats.num_ranking_lists
    assert sorted(dataset.epoch_example_indices()) == sorted(
        dataset.epoch_ranking_indices()
    )
    assert len(dataset.epoch_example_indices()) == 48

    batch = RewardAssayListCollator()(
        [dataset[index] for index in range(len(dataset))]
    )
    assert (batch["ranking_group_ids"] >= 0).all()


def test_assay_metadata_scan_selects_only_the_two_required_columns():
    examples = _tokenized_rows(group_sizes=(4,))
    selected_columns = []
    original_select_columns = examples.select_columns

    def _tracked_select_columns(columns):
        selected_columns.append(list(columns))
        return original_select_columns(columns)

    examples.select_columns = _tracked_select_columns
    dataset = RewardAssayListDataset(examples)

    assert selected_columns == [["group_id", "pchembl_value"]]
    assert dataset.stats.num_examples == 4


def test_assay_list_collator_marks_only_ranked_rows_and_keeps_all_labels():
    dataset = RewardAssayListDataset(
        _tokenized_rows(group_sizes=(20, 1)),
        seed=7,
        max_classification_only_per_item=8,
    )
    collator = RewardAssayListCollator()
    batch = collator([dataset[index] for index in range(len(dataset))])

    assert batch["activity_labels"].numel() == 21
    assert batch["pchembl_values"].numel() == 21
    assert batch["num_ranking_lists"].item() == 1
    assert batch["num_ranked_examples"].item() == 16
    assert (batch["ranking_group_ids"] >= 0).sum().item() == 16
    assert (batch["ranking_group_ids"] < 0).sum().item() == 5
    assert sorted(batch["evaluation_example_indices"].tolist()) == list(range(21))
    assert batch["contrastive_target_ids"].shape == (21,)
    assert batch["contrastive_molecule_ids"].shape == (21,)
    assert torch.unique(batch["contrastive_molecule_ids"]).numel() == 21
    for group_id in torch.unique(
        batch["ranking_group_ids"][batch["ranking_group_ids"] >= 0]
    ):
        group_targets = batch["contrastive_target_ids"][
            batch["ranking_group_ids"] == group_id
        ]
        assert torch.unique(group_targets).numel() == 1


def test_joint_model_backward_reaches_both_heads_and_shared_trunk():
    model = _dummy_model()
    dataset = RewardAssayListDataset(
        _tokenized_rows(group_sizes=(8, 3)),
        seed=11,
        max_classification_only_per_item=8,
    )
    batch = RewardAssayListCollator()(
        [dataset[index] for index in range(len(dataset))]
    )

    outputs = model(
        protein_input_ids=batch["protein_input_ids"],
        protein_attention_mask=batch["protein_attention_mask"],
        molecule_input_ids=batch["molecule_input_ids"],
        molecule_attention_mask=batch["molecule_attention_mask"],
        activity_labels=batch["activity_labels"],
        pchembl_values=batch["pchembl_values"],
        ranking_group_ids=batch["ranking_group_ids"],
    )
    outputs.loss.backward()

    assert outputs.classification_loss is not None
    assert outputs.ranking_loss is not None
    assert torch.isfinite(outputs.loss)
    gradient_targets = [
        model.protein_encoder.proj.weight.grad,
        model.molecule_encoder.proj.weight.grad,
        model.fusion.query_p.weight.grad,
        model.ranking_head.fc1.weight.grad,
        model.classification_head.fc1.weight.grad,
    ]
    assert all(gradient is not None for gradient in gradient_targets)
    assert all(torch.isfinite(gradient).all() for gradient in gradient_targets)
    assert all(gradient.abs().sum() > 0 for gradient in gradient_targets)


def test_classification_only_batch_has_zero_ranking_loss_and_valid_backpropagation():
    model = _dummy_model()
    examples = _tokenized_rows(group_sizes=(1, 1))
    evaluation_dataset = RewardEvaluationDataset(examples)
    batch = RewardAssayListCollator()(
        [evaluation_dataset[index] for index in range(len(evaluation_dataset))]
    )

    outputs = model(
        protein_input_ids=batch["protein_input_ids"],
        protein_attention_mask=batch["protein_attention_mask"],
        molecule_input_ids=batch["molecule_input_ids"],
        molecule_attention_mask=batch["molecule_attention_mask"],
        activity_labels=batch["activity_labels"],
        pchembl_values=batch["pchembl_values"],
        ranking_group_ids=batch["ranking_group_ids"],
    )
    outputs.loss.backward()

    assert outputs.ranking_loss.item() == pytest.approx(0.0)
    assert model.classification_head.fc1.weight.grad is not None
    assert model.classification_head.fc1.weight.grad.abs().sum() > 0
    ranking_grad = model.ranking_head.fc1.weight.grad
    assert ranking_grad is None or ranking_grad.abs().sum() == 0


def test_evaluation_protein_shuffle_is_deterministic_and_keeps_original_ligands():
    examples = _tokenized_rows(group_sizes=(3, 3, 3))
    first = RewardEvaluationDataset(examples, ranking_partition_seed=17)
    second = RewardEvaluationDataset(examples, ranking_partition_seed=17)

    for index in range(len(examples)):
        first_item = first[index]
        second_item = second[index]
        original = first_item["rows"][0]
        shuffled = first_item["protein_shuffled_rows"][0]
        assert shuffled["target_chembl_id"] != original["target_chembl_id"]
        assert (
            first_item["protein_shuffled_rows"][0]["protein_input_ids"]
            == second_item["protein_shuffled_rows"][0]["protein_input_ids"]
        )

    batch = RewardAssayListCollator()([first[index] for index in range(len(first))])
    assert torch.equal(
        batch["molecule_input_ids"],
        torch.tensor(examples["molecule_input_ids"])[:, :2],
    )
    assert not torch.equal(
        batch["protein_input_ids"],
        batch["protein_shuffled_input_ids"],
    )


def test_single_target_evaluation_omits_protein_shuffle_pass():
    examples = _tokenized_rows(group_sizes=(3,))
    dataset = RewardEvaluationDataset(examples)
    assert "protein_shuffled_rows" not in dataset[0]
    batch = RewardAssayListCollator()([dataset[index] for index in range(len(dataset))])
    assert "protein_shuffled_input_ids" not in batch


def test_protein_shuffle_sensitivity_can_be_disabled_without_extra_tokens():
    examples = _tokenized_rows(group_sizes=(3, 3))
    dataset = RewardEvaluationDataset(
        examples,
        protein_shuffle_sensitivity=False,
    )
    batch = RewardAssayListCollator()([dataset[index] for index in range(len(dataset))])
    assert "protein_shuffled_input_ids" not in batch


def test_two_ligand_assay_is_classification_only():
    dataset = RewardAssayListDataset(_tokenized_rows(group_sizes=(2,)))
    batch = RewardAssayListCollator()([dataset[index] for index in range(len(dataset))])

    assert dataset.stats.num_eligible_assays == 0
    assert dataset.stats.num_ranking_lists == 0
    assert batch["num_ranking_lists"].item() == 0
    assert (batch["ranking_group_ids"] < 0).all()


def test_model_skips_two_ligand_listwise_group():
    model = _dummy_model()
    scores = torch.tensor([0.1, 0.2], requires_grad=True)
    loss = model._compute_ranking_loss(
        scores,
        torch.tensor([6.0, 6.2]),
        torch.tensor([0, 0]),
    )

    assert loss.item() == pytest.approx(0.0)
    loss.backward()
    assert scores.grad is not None
    assert scores.grad.abs().sum() == 0.0


def test_model_ranks_three_ligand_list():
    model = _dummy_model()
    scores = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    loss = model._compute_ranking_loss(
        scores,
        torch.tensor([6.0, 6.6, 7.2]),
        torch.tensor([0, 0, 0]),
    )

    assert loss.item() > 0.0
    loss.backward()
    assert scores.grad is not None
    assert scores.grad.abs().sum() > 0.0


def test_epoch_sampler_is_reproducible_and_changes_epoch_order():
    dataset = RewardAssayListDataset(
        _tokenized_rows(),
        seed=17,
        max_classification_only_per_item=4,
    )
    sampler = AssayListEpochSampler(
        dataset,
        seed=17,
        length_bucketing=False,
        batch_size=2,
        bucket_size_multiplier=2,
    )
    epoch_indices_zero = list(sampler)
    sampler.set_epoch(1)
    epoch_indices_one = list(sampler)
    assert {epoch for epoch, _ in epoch_indices_zero} == {0}
    assert {epoch for epoch, _ in epoch_indices_one} == {1}
    order_zero = [index for _, index in epoch_indices_zero]
    order_one = [index for _, index in epoch_indices_one]
    assert sorted(order_zero) == list(range(len(dataset)))
    assert sorted(order_one) == list(range(len(dataset)))
    assert order_zero != order_one


def test_epoch_tag_updates_an_independent_dataloader_worker_copy():
    main_dataset = RewardAssayListDataset(
        _tokenized_rows(),
        seed=29,
        max_classification_only_per_item=4,
    )
    worker_copy = copy.deepcopy(main_dataset)
    sampler = AssayListEpochSampler(
        main_dataset,
        seed=29,
        length_bucketing=False,
        batch_size=2,
        bucket_size_multiplier=2,
    )
    sampler.set_epoch(1)
    tagged_index = next(iter(sampler))

    assert worker_copy.epoch == 0
    worker_item = worker_copy[tagged_index]
    assert worker_copy.epoch == 1
    assert worker_item["example_indices"] == main_dataset[tagged_index]["example_indices"]


def test_multiworker_dataloader_receives_the_current_epoch():
    dataset = RewardAssayListDataset(
        _tokenized_rows(),
        seed=41,
        max_classification_only_per_item=4,
    )
    sampler = AssayListEpochSampler(
        dataset,
        seed=41,
        length_bucketing=False,
        batch_size=2,
        bucket_size_multiplier=2,
    )

    def _worker_ranked_indices(epoch):
        sampler.set_epoch(epoch)
        loader = DataLoader(
            dataset,
            batch_size=2,
            sampler=sampler,
            collate_fn=list,
            num_workers=2,
        )
        return sorted(
            example_index
            for features in loader
            for feature in features
            for example_index in feature["example_indices"][
                : sum(feature["ranking_group_sizes"])
            ]
        )

    epoch_zero = _worker_ranked_indices(0)
    epoch_one = _worker_ranked_indices(1)
    assert epoch_zero == sorted(
        RewardAssayListDataset(
            _tokenized_rows(),
            seed=41,
            max_classification_only_per_item=4,
        ).epoch_ranking_indices()
    )
    expected_epoch_one = RewardAssayListDataset(
        _tokenized_rows(),
        seed=41,
        max_classification_only_per_item=4,
    )
    expected_epoch_one.set_epoch(1)
    assert epoch_one == sorted(expected_epoch_one.epoch_ranking_indices())
    assert epoch_zero != epoch_one


def test_accelerate_ddp_shards_cover_each_example_once_without_padding_duplicates():
    world_size = 4
    per_device_batch_size = 2
    datasets = [
        RewardAssayListDataset(
            _tokenized_rows(),
            seed=19,
            max_classification_only_per_item=1,
            item_count_multiple=world_size * per_device_batch_size,
        )
        for _ in range(world_size)
    ]
    assert len(datasets[0]) % (world_size * per_device_batch_size) == 0

    rank_example_sets = []
    for rank, dataset in enumerate(datasets):
        sampler = AssayListEpochSampler(
            dataset,
            seed=19,
            length_bucketing=False,
            batch_size=per_device_batch_size,
            bucket_size_multiplier=2,
        )
        batches = BatchSamplerShard(
            BatchSampler(sampler, per_device_batch_size, drop_last=False),
            num_processes=world_size,
            process_index=rank,
            split_batches=False,
            even_batches=True,
        )
        item_indices = [index for batch in batches for index in batch]
        rank_examples = {
            example_index
            for item_index in item_indices
            for example_index in dataset[item_index]["example_indices"]
        }
        rank_example_sets.append(rank_examples)

    assert set.union(*rank_example_sets) == set(range(len(datasets[0].example_dataset)))
    for left_rank in range(world_size):
        for right_rank in range(left_rank + 1, world_size):
            assert rank_example_sets[left_rank].isdisjoint(rank_example_sets[right_rank])


def test_duplicate_free_ddp_sharding_rejects_a_dataset_smaller_than_one_global_batch():
    with pytest.raises(ValueError, match="too small for duplicate-free distributed"):
        RewardAssayListDataset(
            _tokenized_rows(group_sizes=(1,)),
            item_count_multiple=8,
        )


def test_cpu_trainer_smoke_trains_and_evaluates_joint_objective(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    examples = _tokenized_rows(group_sizes=(8, 6))
    train_dataset = RewardAssayListDataset(
        examples,
        seed=23,
        max_classification_only_per_item=4,
        item_count_multiple=2,
    )
    eval_dataset = RewardEvaluationDataset(examples, ranking_min_pchembl_span=0.5)
    model = _dummy_model()
    initial_state = copy.deepcopy(model.state_dict())
    trainer = RewardModelTrainer(
        model=model,
        args=create_training_arguments(
            RewardTrainerConfig(
                output_dir=str(tmp_path / "output"),
                num_train_epochs=1,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=4,
                logging_steps=1,
                learning_rate=1.0e-3,
                fp16=False,
                optim="adamw_torch",
                training_mode="single_gpu",
            )
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=RewardAssayListCollator(),
    )

    result = trainer.train()
    metrics = trainer.evaluate()

    assert math.isfinite(result.training_loss)
    assert math.isfinite(metrics["eval_loss"])
    assert math.isfinite(metrics["eval_classification_loss"])
    assert math.isfinite(metrics["eval_ranking_loss"])
    assert metrics["eval_num_examples"] == len(examples)
    assert metrics["eval_num_ranking_groups"] == 2
    assert metrics["eval_num_ranking_lists"] == 2
    assert metrics["eval_num_ranked_examples"] == len(examples)
    assert metrics["eval_ranking_partitions"] == 3
    assert any(
        not torch.equal(initial_state[name], value)
        for name, value in model.state_dict().items()
    )


def test_cpu_validation_is_deterministic_and_scores_each_row_once(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    examples = _tokenized_rows(group_sizes=(17, 18))
    train_dataset = RewardAssayListDataset(
        examples,
        seed=31,
        max_classification_only_per_item=4,
    )
    eval_dataset = RewardEvaluationDataset(examples)
    model = _dummy_model()
    forward_calls = []
    hook = model.register_forward_hook(lambda *args: forward_calls.append(1))
    trainer = RewardModelTrainer(
        model=model,
        args=create_training_arguments(
            RewardTrainerConfig(
                output_dir=str(tmp_path / "output"),
                num_train_epochs=1,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=4,
                fp16=False,
                optim="adamw_torch",
                training_mode="single_gpu",
            )
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=RewardAssayListCollator(),
    )
    autocast_context_calls = []

    @contextmanager
    def _tracked_autocast_context():
        autocast_context_calls.append(1)
        yield

    monkeypatch.setattr(
        trainer,
        "compute_loss_context_manager",
        _tracked_autocast_context,
    )

    first = trainer.evaluate()
    second = trainer.evaluate()
    hook.remove()

    # Each evaluation scores the real protein and a deterministically shuffled
    # protein for every molecule.
    assert len(forward_calls) == 4 * math.ceil(len(examples) / 4)
    assert len(autocast_context_calls) == len(forward_calls)
    for key in (
        "eval_loss",
        "eval_classification_loss",
        "eval_ranking_loss",
        "eval_spearman",
        "eval_macro_spearman",
        "eval_weighted_spearman",
        "eval_protein_shuffle_macro_rank_stability",
    ):
        assert first[key] == pytest.approx(second[key])
    assert first["eval_num_examples"] == len(examples)
    assert first["eval_num_ranked_examples"] == len(examples)
    assert first["eval_num_ranking_lists"] == 4
    assert first["eval_ranking_partitions"] == 3


def test_cpu_validation_loss_is_independent_of_batch_partitioning(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    examples = _tokenized_rows(group_sizes=(8, 6))
    train_dataset = RewardAssayListDataset(
        examples,
        seed=37,
        max_classification_only_per_item=4,
    )
    eval_dataset = RewardEvaluationDataset(examples)
    model = _dummy_model()

    metrics_by_batch_size = {}
    for eval_batch_size in (3, 8):
        trainer = RewardModelTrainer(
            model=model,
            args=create_training_arguments(
                RewardTrainerConfig(
                    output_dir=str(tmp_path / f"output_{eval_batch_size}"),
                    num_train_epochs=1,
                    per_device_train_batch_size=2,
                    per_device_eval_batch_size=eval_batch_size,
                    fp16=False,
                    optim="adamw_torch",
                    training_mode="single_gpu",
                )
            ),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=RewardAssayListCollator(),
        )
        metrics_by_batch_size[eval_batch_size] = trainer.evaluate()

    for key in (
        "eval_loss",
        "eval_classification_loss",
        "eval_ranking_loss",
        "eval_spearman",
    ):
        assert metrics_by_batch_size[3][key] == pytest.approx(
            metrics_by_batch_size[8][key]
        )
