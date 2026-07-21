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
    LoadedEncoder,
    RewardModel,
    RewardModelConfig,
    ligunity_listwise_loss,
)
from reward_model.training import (
    AssayListEpochSampler,
    RewardAssayListCollator,
    RewardAssayListDataset,
    RewardEvaluationDataset,
    RewardModelTrainer,
    RewardTrainerConfig,
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


def _reference_unique_ligunity_loss(scores, targets, temperature=1.0):
    order = torch.argsort(targets, descending=True)
    ordered_scores = scores[order] / temperature
    n = ordered_scores.numel()
    terms = []
    for index in range(n):
        weight = 1.0 / (math.sqrt(n) * math.log(index + 2.0))
        terms.append(
            weight
            * (torch.logsumexp(ordered_scores[index:], dim=0) - ordered_scores[index])
        )
    return torch.stack(terms).sum()


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


def test_model_uses_every_list_from_an_assay_eligible_at_the_dataset_level():
    model = _dummy_model()
    scores = torch.tensor([0.1, 0.2], requires_grad=True)
    loss = model._compute_ranking_loss(
        scores,
        torch.tensor([6.0, 6.2]),
        torch.tensor([0, 0]),
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
    assert any(
        not torch.equal(initial_state[name], value)
        for name, value in model.state_dict().items()
    )


def test_cpu_validation_is_deterministic_and_scores_each_row_once(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    examples = _tokenized_rows(group_sizes=(8, 6))
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

    assert len(forward_calls) == 2 * math.ceil(len(examples) / 4)
    assert len(autocast_context_calls) == len(forward_calls)
    for key in (
        "eval_loss",
        "eval_classification_loss",
        "eval_ranking_loss",
        "eval_spearman",
    ):
        assert first[key] == pytest.approx(second[key])
    assert first["eval_num_examples"] == len(examples)


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
