from __future__ import annotations

import math
import os
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from datasets import Dataset, load_from_disk
from torch.utils.data import Dataset as TorchDataset

from ..data import load_curated_rows
from ..model import RewardModelConfig
from ..model.encoders import batch_encode_texts, load_tokenizer
from .config import RewardTrainingDataConfig

PAIR_RULE_LOWER_PCHEMBL = 5.0
PAIR_RULE_UPPER_PCHEMBL = 8.0
PAIR_RULE_LOWER_FOLD_CHANGE = 10.0
PAIR_RULE_UPPER_FOLD_CHANGE = 2.0


@dataclass
class TokenizedExamplesArtifacts:
    dataset_path: str
    num_examples: int
    num_groups: int


@dataclass
class ExampleSplitStats:
    train_examples: int
    eval_examples: int
    train_groups: int
    eval_groups: int


@dataclass
class PairBuildStats:
    num_groups: int
    num_groups_with_pairs: int
    num_pairs: int


def build_group_id(target_chembl_id: str, assay_chembl_id: str) -> str:
    return f"{target_chembl_id}__{assay_chembl_id}"


def required_fold_change_for_positive_pchembl(positive_pchembl: float) -> float:
    """Return the clamped fold-change requirement for a positive inside one target+assay group."""
    if positive_pchembl <= PAIR_RULE_LOWER_PCHEMBL:
        return PAIR_RULE_LOWER_FOLD_CHANGE
    if positive_pchembl >= PAIR_RULE_UPPER_PCHEMBL:
        return PAIR_RULE_UPPER_FOLD_CHANGE

    slope = (
        (PAIR_RULE_UPPER_FOLD_CHANGE - PAIR_RULE_LOWER_FOLD_CHANGE)
        / (PAIR_RULE_UPPER_PCHEMBL - PAIR_RULE_LOWER_PCHEMBL)
    )
    return PAIR_RULE_LOWER_FOLD_CHANGE + slope * (positive_pchembl - PAIR_RULE_LOWER_PCHEMBL)


def required_pchembl_margin_for_positive(positive_pchembl: float) -> float:
    """Convert the positive-dependent fold-change rule into the required pChEMBL margin."""
    return math.log10(required_fold_change_for_positive_pchembl(positive_pchembl))


def is_valid_negative_for_positive(positive_pchembl: float, negative_pchembl: float) -> bool:
    """Pair legality rule: same group only, and negative must satisfy the positive-dependent threshold."""
    required_margin = required_pchembl_margin_for_positive(positive_pchembl)
    negative_threshold = positive_pchembl - required_margin
    return negative_pchembl <= negative_threshold


def _prepare_example_rows(curated_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    prepared_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(curated_rows):
        prepared = dict(row)
        prepared["example_id"] = int(index)
        prepared["group_id"] = build_group_id(
            str(prepared["target_chembl_id"]),
            str(prepared["assay_chembl_id"]),
        )
        prepared_rows.append(prepared)
    return prepared_rows


def prepare_tokenized_example_dataset(
    data_config: RewardTrainingDataConfig,
    model_config: RewardModelConfig,
) -> TokenizedExamplesArtifacts:
    curated_rows = load_curated_rows(data_config.curated_data_path)
    prepared_rows = _prepare_example_rows(curated_rows)
    dataset = Dataset.from_list(prepared_rows)

    protein_tokenizer = load_tokenizer(
        model_config.protein_tokenizer_name_or_path or model_config.protein_model_name_or_path
    )
    molecule_tokenizer = load_tokenizer(
        model_config.molecule_tokenizer_name_or_path or model_config.molecule_model_name_or_path
    )

    def _tokenize_batch(batch: Mapping[str, Sequence[str]]) -> Dict[str, List[List[int]]]:
        protein_batch = batch_encode_texts(
            tokenizer=protein_tokenizer,
            texts=batch["protein_sequence"],
            max_length=model_config.protein_max_length,
        )
        molecule_batch = batch_encode_texts(
            tokenizer=molecule_tokenizer,
            texts=batch["compound_selfies"],
            max_length=model_config.molecule_max_length,
        )
        return {
            "protein_input_ids": protein_batch["input_ids"].tolist(),
            "protein_attention_mask": protein_batch["attention_mask"].tolist(),
            "molecule_input_ids": molecule_batch["input_ids"].tolist(),
            "molecule_attention_mask": molecule_batch["attention_mask"].tolist(),
        }

    tokenized = dataset.map(
        _tokenize_batch,
        batched=True,
        batch_size=data_config.tokenization_batch_size,
        desc="Tokenizing reward-model examples",
    )

    output_dir = os.path.abspath(data_config.tokenized_dataset_dir)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    tokenized.save_to_disk(output_dir)

    return TokenizedExamplesArtifacts(
        dataset_path=output_dir,
        num_examples=len(tokenized),
        num_groups=len(set(tokenized["group_id"])),
    )


def load_tokenized_example_dataset(dataset_path: str) -> Dataset:
    return load_from_disk(os.path.abspath(dataset_path))


def split_tokenized_examples_by_group(
    dataset: Dataset,
    eval_split_ratio: float,
    split_seed: int,
) -> Tuple[Dataset, Dataset, ExampleSplitStats]:
    group_to_indices: Dict[str, List[int]] = defaultdict(list)
    for index, group_id in enumerate(dataset["group_id"]):
        group_to_indices[str(group_id)].append(index)

    unique_group_ids = sorted(group_to_indices)
    if len(unique_group_ids) < 2:
        raise ValueError("Need at least two groups to create a temporary train/eval holdout")

    rng = random.Random(split_seed)
    rng.shuffle(unique_group_ids)

    eval_group_count = min(
        len(unique_group_ids) - 1,
        max(1, int(len(unique_group_ids) * eval_split_ratio)),
    )
    eval_group_ids = set(unique_group_ids[:eval_group_count])
    train_group_ids = set(unique_group_ids[eval_group_count:])

    train_indices: List[int] = []
    eval_indices: List[int] = []
    for group_id, indices in group_to_indices.items():
        if group_id in eval_group_ids:
            eval_indices.extend(indices)
        elif group_id in train_group_ids:
            train_indices.extend(indices)

    train_dataset = dataset.select(sorted(train_indices))
    eval_dataset = dataset.select(sorted(eval_indices))
    stats = ExampleSplitStats(
        train_examples=len(train_dataset),
        eval_examples=len(eval_dataset),
        train_groups=len(train_group_ids),
        eval_groups=len(eval_group_ids),
    )
    return train_dataset, eval_dataset, stats


def build_pair_records(dataset: Dataset) -> Tuple[List[Dict[str, Any]], PairBuildStats]:
    grouped: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    example_ids = dataset["example_id"]
    group_ids = dataset["group_id"]
    pchembl_values = dataset["pchembl_value"]

    for example_id, group_id, pchembl_value in zip(example_ids, group_ids, pchembl_values):
        grouped[str(group_id)].append((int(example_id), float(pchembl_value)))

    pair_records: List[Dict[str, Any]] = []
    groups_with_pairs = 0
    for group_id, members in grouped.items():
        sorted_members = sorted(members, key=lambda item: (item[1], item[0]))
        group_pair_count = 0
        for positive_index, (positive_example_id, positive_pchembl) in enumerate(sorted_members):
            for negative_example_id, negative_pchembl in sorted_members[:positive_index]:
                if not is_valid_negative_for_positive(
                    positive_pchembl=float(positive_pchembl),
                    negative_pchembl=float(negative_pchembl),
                ):
                    continue
                pair_records.append(
                    {
                        "positive_example_id": int(positive_example_id),
                        "negative_example_id": int(negative_example_id),
                        "group_id": group_id,
                        "positive_pchembl": float(positive_pchembl),
                        "negative_pchembl": float(negative_pchembl),
                    }
                )
                group_pair_count += 1
        if group_pair_count > 0:
            groups_with_pairs += 1

    stats = PairBuildStats(
        num_groups=len(grouped),
        num_groups_with_pairs=groups_with_pairs,
        num_pairs=len(pair_records),
    )
    return pair_records, stats


class RewardPairDataset(TorchDataset):
    def __init__(self, example_dataset: Dataset, pair_records: Sequence[Mapping[str, Any]]):
        self.example_dataset = example_dataset
        self.pair_records = [dict(record) for record in pair_records]
        self.example_id_to_index = {
            int(example_id): index
            for index, example_id in enumerate(example_dataset["example_id"])
        }

    def __len__(self) -> int:
        return len(self.pair_records)

    def _get_example(self, example_id: int) -> Dict[str, Any]:
        return dict(self.example_dataset[self.example_id_to_index[int(example_id)]])

    def __getitem__(self, index: int) -> Dict[str, Any]:
        pair_record = self.pair_records[index]
        return {
            "positive": self._get_example(int(pair_record["positive_example_id"])),
            "negative": self._get_example(int(pair_record["negative_example_id"])),
            "pair": dict(pair_record),
        }


class RewardPairCollator:
    def __call__(self, features: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ValueError("RewardPairCollator received an empty batch")

        positive_rows = [dict(feature["positive"]) for feature in features]
        negative_rows = [dict(feature["negative"]) for feature in features]
        all_rows = positive_rows + negative_rows
        num_pairs = len(features)

        return {
            "protein_input_ids": torch.tensor(
                [row["protein_input_ids"] for row in all_rows],
                dtype=torch.long,
            ),
            "protein_attention_mask": torch.tensor(
                [row["protein_attention_mask"] for row in all_rows],
                dtype=torch.long,
            ),
            "molecule_input_ids": torch.tensor(
                [row["molecule_input_ids"] for row in all_rows],
                dtype=torch.long,
            ),
            "molecule_attention_mask": torch.tensor(
                [row["molecule_attention_mask"] for row in all_rows],
                dtype=torch.long,
            ),
            "activity_labels": torch.tensor(
                [float(row["activity_label"]) for row in all_rows],
                dtype=torch.float32,
            ),
            "positive_indices": torch.arange(0, num_pairs, dtype=torch.long),
            "negative_indices": torch.arange(num_pairs, num_pairs * 2, dtype=torch.long),
            "group_ids": [str(feature["pair"]["group_id"]) for feature in features],
            "pchembl_values": torch.tensor(
                [float(row["pchembl_value"]) for row in all_rows],
                dtype=torch.float32,
            ),
            "num_pairs": torch.tensor(num_pairs, dtype=torch.long),
        }
