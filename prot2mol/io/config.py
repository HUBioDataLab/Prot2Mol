import argparse
import os
from typing import Any, Dict, Optional

import yaml


def _normalize_config_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize YAML keys to argparse-style destination names."""
    normalized = {}
    for key, value in data.items():
        if not isinstance(key, str):
            key = str(key)
        normalized[key.replace("-", "_")] = value
    return normalized


def load_yaml_config(config_path: str, section: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML config and optionally resolve a section key."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping at top-level: {config_path}")

    if section and section in raw:
        section_value = raw[section]
        if not isinstance(section_value, dict):
            raise ValueError(
                f"Config section '{section}' must be a mapping in file: {config_path}"
            )
        raw = section_value

    return _normalize_config_keys(raw)


def parse_args_with_config(parser: argparse.ArgumentParser, section: Optional[str] = None, argv=None):
    """
    Parse args with optional YAML config support.

    Precedence: CLI arguments > YAML values > parser defaults.
    """
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file.",
    )

    pre_args, _ = pre_parser.parse_known_args(argv)

    if pre_args.config:
        config_values = load_yaml_config(pre_args.config, section=section)

        valid_keys = {action.dest for action in parser._actions}
        unknown_keys = sorted(k for k in config_values.keys() if k not in valid_keys)
        if unknown_keys:
            raise ValueError(
                f"Unknown config keys for section '{section or 'root'}': {unknown_keys}. "
                f"Expected one of: {sorted(valid_keys)}"
            )

        # Required CLI options can be satisfied by config values.
        for action in parser._actions:
            if getattr(action, "required", False):
                if action.dest in config_values and config_values[action.dest] is not None:
                    action.required = False

        parser.set_defaults(**config_values)

    if not any(action.dest == "config" for action in parser._actions):
        parser.add_argument(
            "--config",
            type=str,
            default=pre_args.config,
            help=(
                "Path to YAML config file. "
                f"If file contains a '{section}' mapping, that section is used."
                if section
                else "Path to YAML config file."
            ),
        )

    return parser.parse_args(argv)
