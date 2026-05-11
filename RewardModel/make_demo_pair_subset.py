#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil

from datasets import load_from_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a small demo subset of saved RewardModel pair datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Directory containing train_examples/val_examples/test_examples and train_pairs/val_pairs/test_pairs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where the demo subset datasets will be written",
    )
    parser.add_argument(
        "--train-pairs",
        type=int,
        default=50000,
        help="Number of random train pair rows to keep",
    )
    parser.add_argument(
        "--val-pairs",
        type=int,
        default=5000,
        help="Number of random val pair rows to keep",
    )
    parser.add_argument(
        "--test-pairs",
        type=int,
        default=0,
        help="If > 0, also sample this many test pair rows; otherwise copy all test pairs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for pair sampling",
    )
    return parser.parse_args()


def split_paths(base_dir: str) -> dict[str, str]:
    resolved = os.path.abspath(base_dir)
    return {
        "train_examples": os.path.join(resolved, "train_examples"),
        "val_examples": os.path.join(resolved, "val_examples"),
        "test_examples": os.path.join(resolved, "test_examples"),
        "train_pairs": os.path.join(resolved, "train_pairs"),
        "val_pairs": os.path.join(resolved, "val_pairs"),
        "test_pairs": os.path.join(resolved, "test_pairs"),
    }


def link_or_copy_dir(source_path: str, target_path: str) -> str:
    if os.path.lexists(target_path):
        if os.path.islink(target_path) or os.path.isfile(target_path):
            os.remove(target_path)
        else:
            shutil.rmtree(target_path)
    try:
        os.symlink(source_path, target_path, target_is_directory=True)
        return "symlink"
    except OSError:
        shutil.copytree(source_path, target_path)
        return "copy"


def sample_dataset(dataset_path: str, sample_size: int, seed: int):
    dataset = load_from_disk(dataset_path)
    if sample_size <= 0 or sample_size >= len(dataset):
        return dataset, len(dataset), len(dataset), False
    sampled = dataset.shuffle(seed=seed).select(range(sample_size))
    return sampled, len(dataset), len(sampled), True


def main() -> None:
    args = parse_args()
    source_dir = os.path.abspath(args.source_dir)
    output_dir = os.path.abspath(args.output_dir)

    source_paths = split_paths(source_dir)
    missing = [path for path in source_paths.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing source datasets: {missing}")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    output_paths = split_paths(output_dir)
    example_modes: dict[str, str] = {}
    for split_name in ("train_examples", "val_examples", "test_examples"):
        example_modes[split_name] = link_or_copy_dir(
            source_paths[split_name],
            output_paths[split_name],
        )

    train_pairs, train_source_count, train_subset_count, train_was_sampled = sample_dataset(
        source_paths["train_pairs"],
        args.train_pairs,
        args.seed,
    )
    val_pairs, val_source_count, val_subset_count, val_was_sampled = sample_dataset(
        source_paths["val_pairs"],
        args.val_pairs,
        args.seed + 1,
    )
    test_pairs, test_source_count, test_subset_count, test_was_sampled = sample_dataset(
        source_paths["test_pairs"],
        args.test_pairs,
        args.seed + 2,
    )

    train_pairs.save_to_disk(output_paths["train_pairs"])
    val_pairs.save_to_disk(output_paths["val_pairs"])
    test_pairs.save_to_disk(output_paths["test_pairs"])

    summary = {
        "source_dir": source_dir,
        "output_dir": output_dir,
        "seed": args.seed,
        "train_examples_path": output_paths["train_examples"],
        "val_examples_path": output_paths["val_examples"],
        "test_examples_path": output_paths["test_examples"],
        "train_pairs_path": output_paths["train_pairs"],
        "val_pairs_path": output_paths["val_pairs"],
        "test_pairs_path": output_paths["test_pairs"],
        "train_pairs_source_count": train_source_count,
        "train_pairs_subset_count": train_subset_count,
        "train_pairs_was_sampled": train_was_sampled,
        "val_pairs_source_count": val_source_count,
        "val_pairs_subset_count": val_subset_count,
        "val_pairs_was_sampled": val_was_sampled,
        "test_pairs_source_count": test_source_count,
        "test_pairs_subset_count": test_subset_count,
        "test_pairs_was_sampled": test_was_sampled,
        "example_reuse_mode": example_modes,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
