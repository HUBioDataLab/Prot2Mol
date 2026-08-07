#!/usr/bin/env python3
"""Print compact ranking-score diagnostics from a live or completed run."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_path",
        type=Path,
        help="Path to ranking_score_diagnostics.jsonl",
    )
    parser.add_argument("--tail", type=int, default=20, help="Rows to show initially")
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Keep printing new records until interrupted",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds with --follow",
    )
    args = parser.parse_args()
    if args.tail <= 0:
        parser.error("--tail must be > 0")
    if args.interval <= 0.0:
        parser.error("--interval must be > 0")
    return args


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Diagnostic log does not exist: {path}")
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
    return records


def _metric(record: dict[str, Any], key: str) -> Any:
    split = str(record.get("split", "train"))
    prefix = "" if split == "train" else f"{split}_"
    return record.get("metrics", {}).get(f"{prefix}{key}")


def _format(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4g}"


def _print_records(records: Iterable[dict[str, Any]]) -> None:
    headers = (
        "step",
        "split",
        "rank_loss",
        "spearman",
        "grad_norm",
        "scale",
        "cos_mean",
        "cos_std",
        "entropy",
        "margin_acc",
        "gap_p50",
        "class_bias",
    )
    rows = []
    for record in records:
        rows.append(
            (
                str(record.get("global_step", "-")),
                str(record.get("split", "-")),
                _format(_metric(record, "ranking_loss")),
                _format(_metric(record, "spearman")),
                _format(_metric(record, "grad_norm")),
                _format(_metric(record, "cosine_scale")),
                _format(_metric(record, "ranking_cosine_mean")),
                _format(_metric(record, "ranking_cosine_std")),
                _format(_metric(record, "ranking_list_normalized_entropy")),
                _format(_metric(record, "ranking_margin_pair_accuracy")),
                _format(_metric(record, "ranking_margin_pair_gap_p50")),
                _format(_metric(record, "classification_logit_bias")),
            )
        )
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    for row in rows:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))


def main() -> None:
    args = _parse_args()
    records = _read_records(args.log_path)
    _print_records(records[-args.tail :])
    if not args.follow:
        return

    seen = len(records)
    try:
        while True:
            time.sleep(args.interval)
            records = _read_records(args.log_path)
            if len(records) > seen:
                _print_records(records[seen:])
                seen = len(records)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
