#!/usr/bin/env python3
"""Compare prediction-analysis artifacts from multiple checkpoints."""

from __future__ import annotations

import argparse
import html
import json
import os

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        metavar="LABEL=ANALYSIS_DIR",
        help="Two or more outputs from inspect_ranking_predictions.py",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if len(args.runs) < 2:
        parser.error("--runs requires at least two checkpoint analyses")
    return args


def _parse_run(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"Run must use LABEL=ANALYSIS_DIR syntax: {value!r}")
    label, directory = value.split("=", 1)
    if not label or not directory:
        raise ValueError(f"Run must use LABEL=ANALYSIS_DIR syntax: {value!r}")
    return label, os.path.abspath(directory)


def compare_runs(run_specs: list[tuple[str, str]]):
    metric_records = []
    assay_frames = []
    for label, directory in run_specs:
        summary_path = os.path.join(directory, "prediction_analysis_summary.json")
        with open(summary_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for split, split_record in payload.items():
            metrics = dict(split_record["metrics"])
            metrics["checkpoint"] = label
            metric_records.append(metrics)
            assay_path = os.path.join(directory, f"{split}_assay_summary.csv")
            assays = pd.read_csv(assay_path)
            assays.insert(0, "checkpoint", label)
            assay_frames.append(assays)
    return pd.DataFrame.from_records(metric_records), pd.concat(
        assay_frames, ignore_index=True
    )


def main() -> None:
    args = _parse_args()
    run_specs = [_parse_run(value) for value in args.runs]
    labels = [label for label, _ in run_specs]
    if len(set(labels)) != len(labels):
        raise ValueError("Checkpoint labels must be unique")
    metrics, assays = compare_runs(run_specs)
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.abspath(
        os.path.join(args.output_dir, "checkpoint_metrics.csv")
    )
    assays_path = os.path.abspath(
        os.path.join(args.output_dir, "checkpoint_assay_metrics.csv")
    )
    report_path = os.path.abspath(
        os.path.join(args.output_dir, "checkpoint_comparison.html")
    )
    metrics.to_csv(metrics_path, index=False)
    assays.to_csv(assays_path, index=False)
    selected_columns = [
        column
        for column in (
            "checkpoint",
            "split",
            "weighted_spearman",
            "margin_pair_accuracy",
            "mean_normalized_entropy",
            "mean_score_std",
            "within_assay_score_activity_logit_spearman",
            "mean_top1_pchembl_regret",
        )
        if column in metrics.columns
    ]
    table = metrics[selected_columns].to_html(index=False, escape=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(
            "<!doctype html><meta charset='utf-8'><title>Checkpoint comparison</title>"
            "<style>body{font-family:system-ui;margin:24px}"
            "table{border-collapse:collapse}th,td{border:1px solid #ddd;padding:5px}"
            "</style><h1>Ranking checkpoint comparison</h1>"
            f"<p>{html.escape(', '.join(labels))}</p>{table}"
        )
    print(
        json.dumps(
            {
                "metrics": metrics_path,
                "assays": assays_path,
                "report": report_path,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
