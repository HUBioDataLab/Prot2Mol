from __future__ import annotations

import math
import os
import shutil
from array import array
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Mapping, Sequence, Tuple

import torch
from datasets import Dataset, Features, Value, load_from_disk
from torch.utils.data import Dataset as TorchDataset

from ..model import RewardModelConfig
from ..model.losses import MIN_LISTWISE_LIGANDS
from ..model.encoders import batch_encode_texts, load_tokenizer
from .config import RewardTrainingDataConfig

PAIR_RULE_LOWER_PCHEMBL = 5.0
PAIR_RULE_UPPER_PCHEMBL = 8.0
PAIR_RULE_LOWER_FOLD_CHANGE = 10.0
PAIR_RULE_UPPER_FOLD_CHANGE = 2.0

SOURCE_PARQUET_COLUMNS = (
    "target_chembl_id",
    "protein_sequence",
    "assay_id",
    "compound_id",
    "compound_selfies",
    "pchembl_value",
    "binary_label",
)

TOKENIZED_DATASET_COLUMNS = (
    "example_id",
    "group_id",
    "target_chembl_id",
    "assay_id",
    "compound_id",
    "pchembl_value",
    "binary_label",
    "activity_type",
    "protein_input_ids",
    "protein_attention_mask",
    "protein_length",
    "molecule_input_ids",
    "molecule_attention_mask",
    "molecule_length",
)

PAIR_DATASET_COLUMNS = (
    "positive_example_id",
    "negative_example_id",
    "group_id",
    "positive_pchembl",
    "negative_pchembl",
)

PAIR_DATASET_FEATURES = Features(
    {
        "positive_example_id": Value("int64"),
        "negative_example_id": Value("int64"),
        "group_id": Value("string"),
        "positive_pchembl": Value("float64"),
        "negative_pchembl": Value("float64"),
    }
)


@dataclass
class TokenizedSplitArtifacts:
    base_dir: str
    train_dataset_path: str
    val_dataset_path: str
    test_dataset_path: str
    train_examples: int
    val_examples: int
    test_examples: int
    train_groups: int
    val_groups: int
    test_groups: int


@dataclass
class PairBuildStats:
    num_groups: int
    num_groups_with_pairs: int
    num_pairs: int


@dataclass
class SavedPairDatasetArtifacts:
    dataset_path: str
    num_groups: int
    num_groups_with_pairs: int
    num_pairs: int


@dataclass(frozen=True)
class AssayListSamplingStats:
    num_examples: int
    num_assays: int
    num_eligible_assays: int
    num_ranking_lists: int
    num_ranked_examples: int
    num_classification_only_examples: int
    num_dataset_items: int


def build_group_id(target_chembl_id: str, assay_id: str) -> str:
    return f"{target_chembl_id}__{assay_id}"


def get_tokenized_split_dataset_paths(tokenized_dataset_dir: str) -> Dict[str, str]:
    base_dir = os.path.abspath(tokenized_dataset_dir)
    return {
        "train": os.path.join(base_dir, "train_examples"),
        "val": os.path.join(base_dir, "val_examples"),
        "test": os.path.join(base_dir, "test_examples"),
    }


def get_saved_pair_dataset_paths(tokenized_dataset_dir: str) -> Dict[str, str]:
    base_dir = os.path.abspath(tokenized_dataset_dir)
    return {
        "train": os.path.join(base_dir, "train_pairs"),
        "val": os.path.join(base_dir, "val_pairs"),
        "test": os.path.join(base_dir, "test_pairs"),
    }


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


def _load_split_parquet_rows(path: str) -> List[Dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required to load split parquet training inputs. "
            "Use the reward_model environment."
        ) from exc

    resolved_path = os.path.abspath(path)
    try:
        import pyarrow.parquet as parquet

        available_columns = set(parquet.ParquetFile(resolved_path).schema.names)
        missing_columns = [
            column
            for column in SOURCE_PARQUET_COLUMNS
            if column not in available_columns
        ]
        if missing_columns:
            raise ValueError(
                f"Split parquet at {resolved_path} is missing required columns: "
                f"{missing_columns}"
            )
        selected_columns = list(SOURCE_PARQUET_COLUMNS)
        if "activity_type" in available_columns:
            selected_columns.append("activity_type")
        frame = pd.read_parquet(resolved_path, columns=selected_columns)
    except ImportError:
        frame = pd.read_parquet(resolved_path)
        available_columns = set(frame.columns)
    missing_columns = [column for column in SOURCE_PARQUET_COLUMNS if column not in available_columns]
    if missing_columns:
        raise ValueError(
            f"Split parquet at {resolved_path} is missing required columns: {missing_columns}"
        )

    minimal_frame = frame.loc[:, list(SOURCE_PARQUET_COLUMNS)].copy()
    minimal_frame["activity_type"] = (
        frame["activity_type"].fillna("").astype(str)
        if "activity_type" in frame.columns
        else "Unknown"
    )
    return minimal_frame.to_dict(orient="records")


def _prepare_split_rows(
    split_rows: Sequence[Mapping[str, Any]],
    *,
    activity_threshold: float = 6.0,
) -> List[Dict[str, Any]]:
    prepared_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(split_rows):
        target_chembl_id = str(row["target_chembl_id"])
        assay_id = str(row["assay_id"])
        pchembl_value = float(row["pchembl_value"])
        binary_label = int(row["binary_label"])
        expected_label = int(pchembl_value >= activity_threshold)
        if binary_label != expected_label:
            raise ValueError(
                "binary_label does not match the configured pChEMBL activity threshold: "
                f"row={index}, pchembl_value={pchembl_value}, "
                f"binary_label={binary_label}, expected={expected_label}"
            )
        prepared_rows.append(
            {
                "example_id": int(index),
                "group_id": build_group_id(target_chembl_id, assay_id),
                "target_chembl_id": target_chembl_id,
                "protein_sequence": str(row["protein_sequence"]),
                "assay_id": assay_id,
                "compound_id": str(row["compound_id"]),
                "compound_selfies": str(row["compound_selfies"]),
                "pchembl_value": pchembl_value,
                "binary_label": binary_label,
                "activity_type": str(row.get("activity_type", "Unknown")),
            }
        )
    return prepared_rows


def _select_tokenized_columns(dataset: Dataset) -> Dataset:
    if hasattr(dataset, "select_columns"):
        return dataset.select_columns(list(TOKENIZED_DATASET_COLUMNS))

    removable = [column for column in dataset.column_names if column not in TOKENIZED_DATASET_COLUMNS]
    return dataset.remove_columns(removable)


def _measure_tokenized_lengths(
    tokenizer: Any,
    texts: Sequence[str],
) -> List[int]:
    encode_kwargs = {
        "add_special_tokens": True,
        "padding": "longest",
        "truncation": False,
        "return_tensors": "pt",
    }
    text_list = list(texts)

    if callable(tokenizer):
        encoded = tokenizer(text_list, **encode_kwargs)
    else:
        encoded = tokenizer.batch_encode_plus(text_list, **encode_kwargs)

    return [int(length) for length in encoded["attention_mask"].sum(dim=1).tolist()]


def _filter_rows_that_fit_max_lengths(
    dataset: Dataset,
    *,
    split_name: str,
    data_config: RewardTrainingDataConfig,
    model_config: RewardModelConfig,
    protein_tokenizer: Any,
    molecule_tokenizer: Any,
) -> Dataset:
    def _mark_rows_that_fit(batch: Mapping[str, Sequence[str]]) -> Dict[str, List[bool]]:
        protein_lengths = _measure_tokenized_lengths(protein_tokenizer, batch["protein_sequence"])
        molecule_lengths = _measure_tokenized_lengths(molecule_tokenizer, batch["compound_selfies"])
        return {
            "_fits_max_lengths": [
                protein_length <= model_config.protein_max_length
                and molecule_length <= model_config.molecule_max_length
                for protein_length, molecule_length in zip(protein_lengths, molecule_lengths)
            ]
        }

    filtered = dataset.map(
        _mark_rows_that_fit,
        batched=True,
        batch_size=data_config.tokenization_batch_size,
        desc=f"Filtering {split_name} reward-model examples by max token length",
    )
    filtered = filtered.filter(
        lambda row: bool(row["_fits_max_lengths"]),
        desc=f"Dropping overlong {split_name} reward-model examples",
    )
    filtered = filtered.remove_columns("_fits_max_lengths")
    filtered = filtered.remove_columns("example_id")
    return filtered.add_column("example_id", list(range(len(filtered))))


def _tokenize_example_rows(
    prepared_rows: Sequence[Mapping[str, Any]],
    *,
    split_name: str,
    data_config: RewardTrainingDataConfig,
    model_config: RewardModelConfig,
    protein_tokenizer: Any,
    molecule_tokenizer: Any,
) -> Dataset:
    dataset = Dataset.from_list([dict(row) for row in prepared_rows])
    dataset = _filter_rows_that_fit_max_lengths(
        dataset,
        split_name=split_name,
        data_config=data_config,
        model_config=model_config,
        protein_tokenizer=protein_tokenizer,
        molecule_tokenizer=molecule_tokenizer,
    )

    def _tokenize_batch(batch: Mapping[str, Sequence[str]]) -> Dict[str, List[Any]]:
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
            "protein_length": protein_batch["attention_mask"].sum(dim=1).tolist(),
            "molecule_input_ids": molecule_batch["input_ids"].tolist(),
            "molecule_attention_mask": molecule_batch["attention_mask"].tolist(),
            "molecule_length": molecule_batch["attention_mask"].sum(dim=1).tolist(),
        }

    tokenized = dataset.map(
        _tokenize_batch,
        batched=True,
        batch_size=data_config.tokenization_batch_size,
        desc=f"Tokenizing {split_name} reward-model examples",
    )
    return _select_tokenized_columns(tokenized)


def prepare_tokenized_split_datasets(
    data_config: RewardTrainingDataConfig,
    model_config: RewardModelConfig,
) -> TokenizedSplitArtifacts:
    split_paths = get_tokenized_split_dataset_paths(data_config.tokenized_dataset_dir)
    output_dir = os.path.abspath(data_config.tokenized_dataset_dir)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    protein_tokenizer = load_tokenizer(
        model_config.protein_tokenizer_name_or_path or model_config.protein_model_name_or_path
    )
    molecule_tokenizer = load_tokenizer(
        model_config.molecule_tokenizer_name_or_path or model_config.molecule_model_name_or_path
    )

    stats: Dict[str, Tuple[int, int]] = {}
    source_paths = {
        "train": data_config.train_parquet_path,
        "val": data_config.val_parquet_path,
        "test": data_config.test_parquet_path,
    }
    for split_name, parquet_path in source_paths.items():
        prepared_rows = _prepare_split_rows(
            _load_split_parquet_rows(parquet_path),
            activity_threshold=model_config.activity_threshold,
        )
        tokenized = _tokenize_example_rows(
            prepared_rows,
            split_name=split_name,
            data_config=data_config,
            model_config=model_config,
            protein_tokenizer=protein_tokenizer,
            molecule_tokenizer=molecule_tokenizer,
        )
        tokenized.save_to_disk(split_paths[split_name])
        stats[split_name] = (len(tokenized), len(set(tokenized["group_id"])))

    return TokenizedSplitArtifacts(
        base_dir=output_dir,
        train_dataset_path=split_paths["train"],
        val_dataset_path=split_paths["val"],
        test_dataset_path=split_paths["test"],
        train_examples=stats["train"][0],
        val_examples=stats["val"][0],
        test_examples=stats["test"][0],
        train_groups=stats["train"][1],
        val_groups=stats["val"][1],
        test_groups=stats["test"][1],
    )


def load_tokenized_example_dataset(dataset_path: str) -> Dataset:
    return load_from_disk(os.path.abspath(dataset_path))


class RewardAssayListDataset(TorchDataset):
    """Epoch-aware joint classification and assay-list ranking dataset.

    Every source observation appears exactly once as a classification example
    per epoch. Assays with at least three ligands and sufficient affinity span
    contribute ``ceil(n / opportunity_divisor)`` non-overlapping lists of at
    most ``ranking_max_ligands`` observations. The selected ranking observations
    change deterministically with ``seed + epoch``.
    """

    def __init__(
        self,
        example_dataset: Dataset,
        *,
        seed: int = 42,
        ranking_max_ligands: int = 16,
        ranking_opportunity_divisor: int = 32,
        ranking_min_pchembl_span: float = 0.5,
        max_classification_only_per_item: int = 16,
        item_count_multiple: int = 1,
    ):
        if ranking_max_ligands <= 1:
            raise ValueError("ranking_max_ligands must be > 1")
        if ranking_opportunity_divisor < ranking_max_ligands:
            raise ValueError(
                "ranking_opportunity_divisor must be >= ranking_max_ligands"
            )
        if ranking_min_pchembl_span < 0.0:
            raise ValueError("ranking_min_pchembl_span must be >= 0")
        if max_classification_only_per_item <= 0:
            raise ValueError("max_classification_only_per_item must be > 0")
        if item_count_multiple <= 0:
            raise ValueError("item_count_multiple must be > 0")

        required_columns = {
            "group_id",
            "pchembl_value",
            "binary_label",
            "protein_input_ids",
            "protein_attention_mask",
            "molecule_input_ids",
            "molecule_attention_mask",
        }
        missing_columns = sorted(required_columns.difference(example_dataset.column_names))
        if missing_columns:
            raise ValueError(
                f"example_dataset is missing required columns: {missing_columns}"
            )

        self.example_dataset = example_dataset
        self.seed = int(seed)
        self.ranking_max_ligands = int(ranking_max_ligands)
        self.ranking_opportunity_divisor = int(ranking_opportunity_divisor)
        self.ranking_min_pchembl_span = float(ranking_min_pchembl_span)
        self.max_classification_only_per_item = int(max_classification_only_per_item)
        self.item_count_multiple = int(item_count_multiple)
        self._group_members, self._pchembl_values = self._build_group_metadata()
        self._eligible_group_ids = {
            group_id
            for group_id, indices in self._group_members.items()
            if len(indices) >= MIN_LISTWISE_LIGANDS
            and (
                max(self._pchembl_values[index] for index in indices)
                - min(self._pchembl_values[index] for index in indices)
            )
            >= self.ranking_min_pchembl_span
        }

        num_ranking_lists = sum(
            math.ceil(len(self._group_members[group_id]) / self.ranking_opportunity_divisor)
            for group_id in self._eligible_group_ids
        )
        num_ranked_examples = sum(
            min(
                len(self._group_members[group_id]),
                math.ceil(
                    len(self._group_members[group_id]) / self.ranking_opportunity_divisor
                )
                * self.ranking_max_ligands,
            )
            for group_id in self._eligible_group_ids
        )
        num_classification_only = len(example_dataset) - num_ranked_examples
        desired_items = max(
            1,
            num_ranking_lists,
            math.ceil(
                num_classification_only / self.max_classification_only_per_item
            ),
        )
        max_nonempty_items = num_ranking_lists + num_classification_only
        if self.item_count_multiple > 1 and max_nonempty_items < self.item_count_multiple:
            raise ValueError(
                "Training dataset is too small for duplicate-free distributed sharding: "
                f"at most {max_nonempty_items} non-empty items are available, but "
                f"world_size * per_device_train_batch_size is {self.item_count_multiple}"
            )
        if desired_items >= self.item_count_multiple:
            num_items = (
                desired_items // self.item_count_multiple
            ) * self.item_count_multiple
        elif self.item_count_multiple > 1:
            num_items = self.item_count_multiple
        else:
            num_items = desired_items
        num_items = max(1, min(num_items, max_nonempty_items))
        if self.item_count_multiple > 1 and num_items % self.item_count_multiple != 0:
            raise RuntimeError("distributed item count must be exactly shardable")

        self.stats = AssayListSamplingStats(
            num_examples=len(example_dataset),
            num_assays=len(self._group_members),
            num_eligible_assays=len(self._eligible_group_ids),
            num_ranking_lists=num_ranking_lists,
            num_ranked_examples=num_ranked_examples,
            num_classification_only_examples=num_classification_only,
            num_dataset_items=num_items,
        )
        self._epoch: int | None = None
        self._bundles: list[
            tuple[list[tuple[str, tuple[int, ...]]], tuple[int, ...]]
        ] = []
        self._example_lengths: array | None = None
        self.set_epoch(0)

    def _build_group_metadata(self) -> tuple[Dict[str, List[int]], array]:
        grouped: Dict[str, List[int]] = defaultdict(list)
        pchembl_values = array("d")
        metadata_columns = ["group_id", "pchembl_value"]
        metadata_dataset = (
            self.example_dataset.select_columns(metadata_columns)
            if hasattr(self.example_dataset, "select_columns")
            else self.example_dataset
        )
        scan_batch_size = 65536
        for start in range(0, len(metadata_dataset), scan_batch_size):
            stop = min(start + scan_batch_size, len(metadata_dataset))
            rows = metadata_dataset[start:stop]
            for offset, (group_id, pchembl_value) in enumerate(
                zip(rows["group_id"], rows["pchembl_value"])
            ):
                value = float(pchembl_value)
                if not math.isfinite(value):
                    raise ValueError("pchembl_value must contain only finite values")
                index = start + offset
                grouped[str(group_id)].append(index)
                pchembl_values.append(value)
        return dict(grouped), pchembl_values

    @staticmethod
    def _permuted(values: Sequence[int], generator: torch.Generator) -> list[int]:
        if len(values) <= 1:
            return [int(value) for value in values]
        order = torch.randperm(len(values), generator=generator).tolist()
        return [int(values[index]) for index in order]

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if self._epoch == epoch:
            return
        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch)

        ranking_lists: list[tuple[str, tuple[int, ...]]] = []
        classification_only: list[int] = []
        for group_id, members in self._group_members.items():
            if group_id not in self._eligible_group_ids:
                classification_only.extend(members)
                continue
            shuffled = self._permuted(members, generator)
            list_count = math.ceil(len(members) / self.ranking_opportunity_divisor)
            ranked_count = min(
                len(members),
                list_count * self.ranking_max_ligands,
            )
            selected = shuffled[:ranked_count]
            classification_only.extend(shuffled[ranked_count:])
            for start in range(0, ranked_count, self.ranking_max_ligands):
                ranking_lists.append(
                    (
                        group_id,
                        tuple(selected[start : start + self.ranking_max_ligands]),
                    )
                )

        ranking_lists = [
            ranking_lists[index]
            for index in torch.randperm(
                len(ranking_lists), generator=generator
            ).tolist()
        ] if ranking_lists else []
        classification_only = self._permuted(classification_only, generator)

        num_items = self.stats.num_dataset_items
        assigned_ranking: list[list[tuple[str, tuple[int, ...]]]] = [
            [] for _ in range(num_items)
        ]
        for list_index, ranking_list in enumerate(ranking_lists):
            assigned_ranking[list_index % num_items].append(ranking_list)

        classification_counts = [
            len(classification_only) // num_items for _ in range(num_items)
        ]
        for item_index in range(len(classification_only) % num_items):
            classification_counts[item_index] += 1
        classification_item_order = sorted(
            range(num_items),
            key=lambda index: (bool(assigned_ranking[index]), index),
        )
        assigned_classification: list[tuple[int, ...]] = [tuple() for _ in range(num_items)]
        cursor = 0
        for count_index, item_index in enumerate(classification_item_order):
            count = classification_counts[count_index]
            assigned_classification[item_index] = tuple(
                classification_only[cursor : cursor + count]
            )
            cursor += count

        self._bundles = [
            (assigned_ranking[index], assigned_classification[index])
            for index in range(num_items)
        ]
        if any(not ranking and not classification for ranking, classification in self._bundles):
            raise RuntimeError("assay-list sampler produced an empty dataset item")
        if sum(len(indices) for _, indices in ranking_lists) != self.stats.num_ranked_examples:
            raise RuntimeError("ranked example count changed while constructing the epoch")
        if cursor != self.stats.num_classification_only_examples:
            raise RuntimeError("classification-only example count changed while constructing the epoch")
        self._epoch = epoch

    @property
    def epoch(self) -> int:
        return int(self._epoch or 0)

    def __len__(self) -> int:
        return self.stats.num_dataset_items

    def _rows_for_indices(self, indices: Sequence[int]) -> list[Dict[str, Any]]:
        if not indices:
            return []
        columns = self.example_dataset[list(indices)]
        return [
            {column: columns[column][row_index] for column in columns}
            for row_index in range(len(indices))
        ]

    def __getitem__(self, index: int | tuple[int, int]) -> Dict[str, Any]:
        if isinstance(index, tuple):
            epoch, index = index
            self.set_epoch(int(epoch))
        ranking_lists, classification_indices = self._bundles[int(index)]
        all_indices: list[int] = []
        ranking_group_sizes: list[int] = []
        ranking_assay_ids: list[str] = []
        for group_id, ranking_indices in ranking_lists:
            all_indices.extend(ranking_indices)
            ranking_group_sizes.append(len(ranking_indices))
            ranking_assay_ids.append(group_id)
        all_indices.extend(classification_indices)
        return {
            "rows": self._rows_for_indices(all_indices),
            "example_indices": all_indices,
            "ranking_group_sizes": ranking_group_sizes,
            "ranking_assay_ids": ranking_assay_ids,
        }

    def epoch_example_indices(self) -> list[int]:
        indices: list[int] = []
        for ranking_lists, classification_indices in self._bundles:
            for _, ranking_indices in ranking_lists:
                indices.extend(ranking_indices)
            indices.extend(classification_indices)
        return indices

    def epoch_ranking_indices(self) -> list[int]:
        indices: list[int] = []
        for ranking_lists, _ in self._bundles:
            for _, ranking_indices in ranking_lists:
                indices.extend(ranking_indices)
        return indices

    def get_item_sequence_lengths(self) -> Sequence[int]:
        if self._example_lengths is None:
            lengths = array("I")
            columns = set(self.example_dataset.column_names)
            if {"protein_length", "molecule_length"}.issubset(columns):
                lengths.extend(
                    int(protein) + int(molecule)
                    for protein, molecule in zip(
                        self.example_dataset["protein_length"],
                        self.example_dataset["molecule_length"],
                    )
                )
            else:
                for protein_mask, molecule_mask in zip(
                    self.example_dataset["protein_attention_mask"],
                    self.example_dataset["molecule_attention_mask"],
                ):
                    lengths.append(
                        sum(int(value) for value in protein_mask)
                        + sum(int(value) for value in molecule_mask)
                    )
            self._example_lengths = lengths

        item_lengths = array("I")
        for ranking_lists, classification_indices in self._bundles:
            indices = [
                index
                for _, ranking_indices in ranking_lists
                for index in ranking_indices
            ]
            indices.extend(classification_indices)
            item_lengths.append(max(self._example_lengths[index] for index in indices))
        return item_lengths


class RewardEvaluationDataset(TorchDataset):
    """One-pass evaluation view with stable complete-coverage list settings."""

    def __init__(
        self,
        example_dataset: Dataset,
        *,
        ranking_min_pchembl_span: float = 0.5,
        ranking_max_ligands: int = 16,
        ranking_num_partitions: int = 3,
        ranking_partition_seed: int = 42,
        protein_shuffle_sensitivity: bool = True,
    ):
        if ranking_min_pchembl_span < 0.0 or not math.isfinite(
            float(ranking_min_pchembl_span)
        ):
            raise ValueError("ranking_min_pchembl_span must be finite and >= 0")
        if ranking_max_ligands < 5:
            raise ValueError("ranking_max_ligands must be >= 5")
        if ranking_num_partitions <= 0:
            raise ValueError("ranking_num_partitions must be > 0")
        self.example_dataset = example_dataset
        self.ranking_min_pchembl_span = float(ranking_min_pchembl_span)
        self.ranking_max_ligands = int(ranking_max_ligands)
        self.ranking_num_partitions = int(ranking_num_partitions)
        self.ranking_partition_seed = int(ranking_partition_seed)
        self.protein_shuffle_sensitivity = bool(protein_shuffle_sensitivity)
        group_ids = [str(group_id) for group_id in example_dataset["group_id"]]
        self.group_id_to_index = {
            group_id: index for index, group_id in enumerate(sorted(set(group_ids)))
        }
        self.group_ids = [None] * len(self.group_id_to_index)
        for group_id, index in self.group_id_to_index.items():
            self.group_ids[index] = group_id
        self._row_group_indices = array(
            "I", (self.group_id_to_index[group_id] for group_id in group_ids)
        )
        target_ids = [str(value) for value in example_dataset["target_chembl_id"]]
        target_representatives: Dict[str, int] = {}
        for row_index, target_id in enumerate(target_ids):
            target_representatives.setdefault(target_id, row_index)
        unique_targets = sorted(target_representatives)
        self._protein_shuffle_indices: array | None = None
        if self.protein_shuffle_sensitivity and len(unique_targets) > 1:
            shift = 1 + (self.ranking_partition_seed % (len(unique_targets) - 1))
            shuffled_target = {
                target_id: unique_targets[(index + shift) % len(unique_targets)]
                for index, target_id in enumerate(unique_targets)
            }
            self._protein_shuffle_indices = array(
                "I",
                (
                    target_representatives[shuffled_target[target_id]]
                    for target_id in target_ids
                ),
            )

    def __len__(self) -> int:
        return len(self.example_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        resolved_index = int(index)
        feature = {
            "rows": [dict(self.example_dataset[resolved_index])],
            "example_indices": [int(index)],
            "ranking_group_sizes": [],
            "ranking_assay_ids": [],
            "evaluation_group_indices": [int(self._row_group_indices[resolved_index])],
        }
        if self._protein_shuffle_indices is not None:
            shuffled_index = int(self._protein_shuffle_indices[resolved_index])
            feature["protein_shuffled_rows"] = [
                dict(self.example_dataset[shuffled_index])
            ]
        return feature


def load_saved_pair_dataset(dataset_path: str) -> Dataset:
    dataset = load_from_disk(os.path.abspath(dataset_path))
    if len(dataset) > 0 and int(dataset[0]["positive_example_id"]) < 0:
        dataset = dataset.filter(lambda row: int(row["positive_example_id"]) >= 0)
    return dataset


def _build_grouped_example_metadata(dataset: Dataset) -> Dict[str, List[Tuple[int, float]]]:
    grouped: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    example_ids = dataset["example_id"]
    group_ids = dataset["group_id"]
    pchembl_values = dataset["pchembl_value"]

    for example_id, group_id, pchembl_value in zip(example_ids, group_ids, pchembl_values):
        grouped[str(group_id)].append((int(example_id), float(pchembl_value)))

    return grouped


def _iter_valid_group_pairs(
    group_id: str,
    members: Sequence[Tuple[int, float]],
) -> Iterator[Dict[str, Any]]:
    sorted_members = sorted(members, key=lambda item: (item[1], item[0]))
    for positive_index, (positive_example_id, positive_pchembl) in enumerate(sorted_members):
        for negative_example_id, negative_pchembl in sorted_members[:positive_index]:
            if not is_valid_negative_for_positive(
                positive_pchembl=float(positive_pchembl),
                negative_pchembl=float(negative_pchembl),
            ):
                continue
            yield {
                "positive_example_id": int(positive_example_id),
                "negative_example_id": int(negative_example_id),
                "group_id": str(group_id),
                "positive_pchembl": float(positive_pchembl),
                "negative_pchembl": float(negative_pchembl),
            }


def _compute_pair_build_stats(
    grouped_metadata: Mapping[str, Sequence[Tuple[int, float]]],
) -> PairBuildStats:
    num_pairs = 0
    groups_with_pairs = 0
    for group_id, members in grouped_metadata.items():
        group_pair_count = sum(1 for _ in _iter_valid_group_pairs(group_id, members))
        if group_pair_count > 0:
            groups_with_pairs += 1
            num_pairs += group_pair_count

    return PairBuildStats(
        num_groups=len(grouped_metadata),
        num_groups_with_pairs=groups_with_pairs,
        num_pairs=num_pairs,
    )


def _empty_pair_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "positive_example_id": [-1],
            "negative_example_id": [-1],
            "group_id": [""],
            "positive_pchembl": [0.0],
            "negative_pchembl": [0.0],
        },
        features=PAIR_DATASET_FEATURES,
    )


def save_pair_dataset_from_example_dataset(
    example_dataset: Dataset,
    output_path: str,
    *,
    split_name: str,
) -> SavedPairDatasetArtifacts:
    grouped_metadata = _build_grouped_example_metadata(example_dataset)
    stats = _compute_pair_build_stats(grouped_metadata)
    resolved_output_path = os.path.abspath(output_path)
    cache_dir = os.path.join(
        os.path.dirname(resolved_output_path),
        f".{os.path.basename(resolved_output_path)}_cache",
    )

    for path in (resolved_output_path, cache_dir):
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)

    try:
        if stats.num_pairs == 0:
            pair_dataset = _empty_pair_dataset()
        else:
            def _generator() -> Iterator[Dict[str, Any]]:
                for group_id, members in grouped_metadata.items():
                    yield from _iter_valid_group_pairs(group_id, members)

            pair_dataset = Dataset.from_generator(
                _generator,
                features=PAIR_DATASET_FEATURES,
                cache_dir=cache_dir,
                keep_in_memory=False,
            )

        pair_dataset.save_to_disk(resolved_output_path)
    finally:
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
    return SavedPairDatasetArtifacts(
        dataset_path=resolved_output_path,
        num_groups=stats.num_groups,
        num_groups_with_pairs=stats.num_groups_with_pairs,
        num_pairs=stats.num_pairs,
    )


def build_pair_records(dataset: Dataset) -> Tuple[List[Dict[str, Any]], PairBuildStats]:
    grouped = _build_grouped_example_metadata(dataset)
    pair_records: List[Dict[str, Any]] = []
    for group_id, members in grouped.items():
        pair_records.extend(_iter_valid_group_pairs(group_id, members))

    stats = _compute_pair_build_stats(grouped)
    return pair_records, stats


class RewardPairDataset(TorchDataset):
    def __init__(self, example_dataset: Dataset, pair_dataset: Dataset):
        self.example_dataset = example_dataset
        self.pair_dataset = pair_dataset
        self.example_id_to_index = {
            int(example_id): index
            for index, example_id in enumerate(example_dataset["example_id"])
        }
        self._example_sequence_lengths: Tuple[array, array] | None = None
        self._pair_sequence_lengths: array | None = None

    def __len__(self) -> int:
        return len(self.pair_dataset)

    def _get_example(self, example_id: int) -> Dict[str, Any]:
        return dict(self.example_dataset[self.example_id_to_index[int(example_id)]])

    def __getitem__(self, index: int) -> Dict[str, Any]:
        pair_record = dict(self.pair_dataset[index])
        return {
            "positive": self._get_example(int(pair_record["positive_example_id"])),
            "negative": self._get_example(int(pair_record["negative_example_id"])),
            "pair": dict(pair_record),
        }

    def _load_example_sequence_lengths(self) -> Tuple[array, array]:
        if self._example_sequence_lengths is not None:
            return self._example_sequence_lengths

        columns = set(self.example_dataset.column_names)
        if {"protein_length", "molecule_length"}.issubset(columns):
            protein_lengths = array(
                "I",
                (int(value) for value in self.example_dataset["protein_length"]),
            )
            molecule_lengths = array(
                "I",
                (int(value) for value in self.example_dataset["molecule_length"]),
            )
        else:
            protein_lengths = array("I")
            molecule_lengths = array("I")
            scan_batch_size = 4096
            for start in range(0, len(self.example_dataset), scan_batch_size):
                stop = min(start + scan_batch_size, len(self.example_dataset))
                rows = self.example_dataset[start:stop]
                protein_lengths.extend(
                    max(1, sum(int(value) for value in mask))
                    for mask in rows["protein_attention_mask"]
                )
                molecule_lengths.extend(
                    max(1, sum(int(value) for value in mask))
                    for mask in rows["molecule_attention_mask"]
                )

        self._example_sequence_lengths = (protein_lengths, molecule_lengths)
        return self._example_sequence_lengths

    def get_pair_sequence_lengths(self) -> Sequence[int]:
        """Return a scalar length proxy for grouping similarly sized pair batches."""
        if self._pair_sequence_lengths is not None:
            return self._pair_sequence_lengths

        protein_lengths, molecule_lengths = self._load_example_sequence_lengths()
        pair_lengths = array("I")
        scan_batch_size = 4096
        for start in range(0, len(self.pair_dataset), scan_batch_size):
            stop = min(start + scan_batch_size, len(self.pair_dataset))
            rows = self.pair_dataset[start:stop]
            for positive_id, negative_id in zip(
                rows["positive_example_id"],
                rows["negative_example_id"],
            ):
                positive_index = self.example_id_to_index[int(positive_id)]
                negative_index = self.example_id_to_index[int(negative_id)]
                pair_lengths.append(
                    max(protein_lengths[positive_index], protein_lengths[negative_index])
                    + max(molecule_lengths[positive_index], molecule_lengths[negative_index])
                )

        self._pair_sequence_lengths = pair_lengths
        return self._pair_sequence_lengths


class RewardPairCollator:
    def __init__(
        self,
        *,
        dynamic_padding: bool = True,
        protein_pad_token_id: int | None = 0,
        molecule_pad_token_id: int | None = 0,
    ):
        self.dynamic_padding = bool(dynamic_padding)
        self.protein_pad_token_id = int(protein_pad_token_id or 0)
        self.molecule_pad_token_id = int(molecule_pad_token_id or 0)

    def _collate_tokens(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        input_key: str,
        mask_key: str,
        pad_token_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_width = max(len(row[input_key]) for row in rows)
        if max_width <= 0:
            raise ValueError(f"{input_key} rows must contain at least one token")
        if any(len(row[input_key]) != len(row[mask_key]) for row in rows):
            raise ValueError(f"{input_key} and {mask_key} rows must have matching lengths")

        row_widths = {len(row[input_key]) for row in rows}
        if len(row_widths) == 1:
            input_ids = torch.tensor(
                [row[input_key] for row in rows],
                dtype=torch.long,
            )
            attention_mask = torch.tensor(
                [row[mask_key] for row in rows],
                dtype=torch.long,
            )
        else:
            input_ids = torch.full(
                (len(rows), max_width),
                fill_value=pad_token_id,
                dtype=torch.long,
            )
            attention_mask = torch.zeros((len(rows), max_width), dtype=torch.long)
            for row_index, row in enumerate(rows):
                row_input_ids = row[input_key]
                row_attention_mask = row[mask_key]
                row_width = len(row_input_ids)
                input_ids[row_index, :row_width] = torch.as_tensor(row_input_ids, dtype=torch.long)
                attention_mask[row_index, :row_width] = torch.as_tensor(
                    row_attention_mask,
                    dtype=torch.long,
                )

        if not self.dynamic_padding:
            return input_ids, attention_mask

        active_columns = attention_mask.bool().any(dim=0).nonzero(as_tuple=False).flatten()
        if active_columns.numel() == 0:
            return input_ids[:, :1], attention_mask[:, :1]
        last_active = int(active_columns[-1].item()) + 1
        return (
            input_ids[:, :last_active],
            attention_mask[:, :last_active],
        )

    def collate_example_tokens(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        protein_input_ids, protein_attention_mask = self._collate_tokens(
            rows,
            input_key="protein_input_ids",
            mask_key="protein_attention_mask",
            pad_token_id=self.protein_pad_token_id,
        )
        molecule_input_ids, molecule_attention_mask = self._collate_tokens(
            rows,
            input_key="molecule_input_ids",
            mask_key="molecule_attention_mask",
            pad_token_id=self.molecule_pad_token_id,
        )
        return {
            "protein_input_ids": protein_input_ids,
            "protein_attention_mask": protein_attention_mask,
            "molecule_input_ids": molecule_input_ids,
            "molecule_attention_mask": molecule_attention_mask,
        }

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ValueError("RewardPairCollator received an empty batch")

        positive_rows = [dict(feature["positive"]) for feature in features]
        negative_rows = [dict(feature["negative"]) for feature in features]
        all_rows = positive_rows + negative_rows
        num_pairs = len(features)
        token_batch = self.collate_example_tokens(all_rows)

        return {
            **token_batch,
            "activity_labels": torch.tensor(
                [float(row["binary_label"]) for row in all_rows],
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


class RewardAssayListCollator(RewardPairCollator):
    """Flatten assay-list items while retaining list boundaries for ranking."""

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ValueError("RewardAssayListCollator received an empty batch")
        if "rows" not in features[0]:
            return super().__call__(features)

        all_rows: list[Dict[str, Any]] = []
        ranking_group_ids: list[int] = []
        ranking_assay_ids: list[str] = []
        evaluation_group_indices: list[int] = []
        evaluation_example_indices: list[int] = []
        protein_shuffled_rows: list[Dict[str, Any]] = []
        protein_shuffle_presence: list[bool] = []
        next_group_id = 0
        for feature in features:
            rows = [dict(row) for row in feature["rows"]]
            group_sizes = [int(size) for size in feature.get("ranking_group_sizes", [])]
            assay_ids = [str(value) for value in feature.get("ranking_assay_ids", [])]
            if len(group_sizes) != len(assay_ids):
                raise ValueError(
                    "ranking_group_sizes and ranking_assay_ids must have the same length"
                )
            ranked_count = sum(group_sizes)
            if ranked_count > len(rows):
                raise ValueError("ranking group sizes exceed the number of feature rows")

            local_ranking_ids: list[int] = []
            for group_size, assay_id in zip(group_sizes, assay_ids):
                if group_size < MIN_LISTWISE_LIGANDS:
                    raise ValueError(
                        "ranking lists must contain at least three observations"
                    )
                local_ranking_ids.extend([next_group_id] * group_size)
                ranking_assay_ids.append(assay_id)
                next_group_id += 1
            local_ranking_ids.extend([-1] * (len(rows) - ranked_count))
            ranking_group_ids.extend(local_ranking_ids)

            feature_eval_groups = feature.get("evaluation_group_indices")
            if feature_eval_groups is None:
                evaluation_group_indices.extend([-1] * len(rows))
            else:
                if len(feature_eval_groups) != len(rows):
                    raise ValueError(
                        "evaluation_group_indices must align with feature rows"
                    )
                evaluation_group_indices.extend(int(value) for value in feature_eval_groups)
            feature_example_indices = feature.get("example_indices")
            if feature_example_indices is None:
                evaluation_example_indices.extend([-1] * len(rows))
            else:
                if len(feature_example_indices) != len(rows):
                    raise ValueError("example_indices must align with feature rows")
                evaluation_example_indices.extend(
                    int(value) for value in feature_example_indices
                )
            all_rows.extend(rows)
            feature_shuffled_rows = feature.get("protein_shuffled_rows")
            protein_shuffle_presence.append(feature_shuffled_rows is not None)
            if feature_shuffled_rows is not None:
                if len(feature_shuffled_rows) != len(rows):
                    raise ValueError(
                        "protein_shuffled_rows must align with feature rows"
                    )
                protein_shuffled_rows.extend(
                    dict(row) for row in feature_shuffled_rows
                )

        if any(protein_shuffle_presence) and not all(protein_shuffle_presence):
            raise ValueError(
                "protein_shuffled_rows must be present for every feature or none"
            )

        if not all_rows:
            raise ValueError("RewardAssayListCollator received no example rows")
        token_batch = self.collate_example_tokens(all_rows)
        ranked_examples = sum(group_id >= 0 for group_id in ranking_group_ids)
        batch = {
            **token_batch,
            "activity_labels": torch.tensor(
                [float(row["binary_label"]) for row in all_rows],
                dtype=torch.float32,
            ),
            "pchembl_values": torch.tensor(
                [float(row["pchembl_value"]) for row in all_rows],
                dtype=torch.float32,
            ),
            "ranking_group_ids": torch.tensor(ranking_group_ids, dtype=torch.long),
            "evaluation_group_indices": torch.tensor(
                evaluation_group_indices,
                dtype=torch.long,
            ),
            "evaluation_example_indices": torch.tensor(
                evaluation_example_indices,
                dtype=torch.long,
            ),
            "ranking_assay_ids": ranking_assay_ids,
            "num_examples": torch.tensor(len(all_rows), dtype=torch.long),
            "num_ranking_lists": torch.tensor(next_group_id, dtype=torch.long),
            "num_ranked_examples": torch.tensor(ranked_examples, dtype=torch.long),
        }
        if all(protein_shuffle_presence):
            shuffled_tokens = self.collate_example_tokens(protein_shuffled_rows)
            batch["protein_shuffled_input_ids"] = shuffled_tokens[
                "protein_input_ids"
            ]
            batch["protein_shuffled_attention_mask"] = shuffled_tokens[
                "protein_attention_mask"
            ]
        return batch
