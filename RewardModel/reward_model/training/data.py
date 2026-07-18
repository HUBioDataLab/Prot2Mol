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
    frame = pd.read_parquet(resolved_path)
    missing_columns = [column for column in SOURCE_PARQUET_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            f"Split parquet at {resolved_path} is missing required columns: {missing_columns}"
        )

    minimal_frame = frame.loc[:, list(SOURCE_PARQUET_COLUMNS)]
    return minimal_frame.to_dict(orient="records")


def _prepare_split_rows(split_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    prepared_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(split_rows):
        target_chembl_id = str(row["target_chembl_id"])
        assay_id = str(row["assay_id"])
        prepared_rows.append(
            {
                "example_id": int(index),
                "group_id": build_group_id(target_chembl_id, assay_id),
                "target_chembl_id": target_chembl_id,
                "protein_sequence": str(row["protein_sequence"]),
                "assay_id": assay_id,
                "compound_id": str(row["compound_id"]),
                "compound_selfies": str(row["compound_selfies"]),
                "pchembl_value": float(row["pchembl_value"]),
                "binary_label": int(row["binary_label"]),
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
        prepared_rows = _prepare_split_rows(_load_split_parquet_rows(parquet_path))
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
