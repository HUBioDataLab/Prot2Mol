#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

from reward_model.training import train_reward_model_from_config


def parse_args() -> argparse.Namespace:
    default_config = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "configs",
        "reward_train.yaml",
    )
    parser = argparse.ArgumentParser(
        description="Train the RewardModel from tokenized split parquet examples and in-memory pair tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help="Path to the RewardModel training YAML config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_reward_model_from_config(args.config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
