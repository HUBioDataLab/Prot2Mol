#!/usr/bin/env python3
"""Analyze MMseqs-split ChEMBL data against the Reward Model ranking contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


DEFAULT_DATASET_DIR = Path("dataset/processed/chembl_37/protein_cluster_50")
DEFAULT_TRAINING_CONFIG = Path("RewardModel/configs/reward_train.yaml")
DEFAULT_AFFINITY_MARGIN = math.log10(3.0)
DEFAULT_MIN_LIGANDS = 3
DEFAULT_MIN_PCHEMBL_SPAN = 0.5
SPLITS = ("train", "val", "test")
REQUIRED_COLUMNS = {
    "assay_group_id",
    "assay_id",
    "target_chembl_id",
    "compound_id",
    "protein_sequence",
    "compound_selfies",
    "pchembl_value",
    "binary_label",
    "activity_type",
}
QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
QUANTILE_NAMES = ("min", "p25", "median", "p75", "p90", "p95", "p99", "max")


def _json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    series = pd.Series(values, dtype="float64").dropna()
    if series.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            **{name: None for name in QUANTILE_NAMES},
        }
    quantiles = series.quantile(QUANTILES)
    return {
        "count": int(len(series)),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
        **{
            name: float(quantiles.loc[quantile])
            for name, quantile in zip(QUANTILE_NAMES, QUANTILES)
        },
    }


def count_comparable_pairs(values: Sequence[float], affinity_margin: float) -> int:
    """Count unordered ligand pairs separated by strictly more than the margin."""
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.size < 2:
        return 0
    lower_counts = np.searchsorted(
        ordered,
        ordered - float(affinity_margin),
        side="left",
    )
    return int(lower_counts.sum())


def build_assay_statistics(
    frame: pd.DataFrame,
    *,
    boundary: str,
    split: str,
    min_ligands: int,
    min_pchembl_span: float,
    affinity_margin: float,
) -> pd.DataFrame:
    duplicate_count = int(frame.duplicated(["assay_group_id", "compound_id"]).sum())
    if duplicate_count:
        raise ValueError(
            f"{boundary}/{split} contains {duplicate_count} duplicate assay-group/compound rows"
        )

    records: list[dict[str, object]] = []
    for assay_group_id, group in frame.groupby("assay_group_id", sort=False):
        pchembl_values = group["pchembl_value"].to_numpy(dtype=np.float64)
        ligand_count = int(group["compound_id"].nunique())
        pchembl_min = float(pchembl_values.min())
        pchembl_max = float(pchembl_values.max())
        pchembl_span = pchembl_max - pchembl_min
        comparable_pairs = count_comparable_pairs(pchembl_values, affinity_margin)
        records.append(
            {
                "boundary": boundary,
                "split": split,
                "assay_group_id": str(assay_group_id),
                "assay_id": str(group["assay_id"].iloc[0]),
                "target_chembl_id": str(group["target_chembl_id"].iloc[0]),
                "ligands": ligand_count,
                "pchembl_min": pchembl_min,
                "pchembl_max": pchembl_max,
                "pchembl_span": pchembl_span,
                "comparable_comparisons": comparable_pairs,
                "eligible_ranking_assay": (
                    ligand_count >= min_ligands
                    and pchembl_span >= min_pchembl_span
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _activity_type_statistics(
    frame: pd.DataFrame,
    *,
    boundary: str,
    split: str,
) -> list[dict[str, object]]:
    row_count = len(frame)
    memberships: dict[str, int] = {}
    composite_counts: dict[str, int] = {}
    mixed_rows = 0
    for raw_value in frame["activity_type"].fillna("").astype(str):
        activity_types = sorted({item.strip() for item in raw_value.split("|") if item.strip()})
        composite_key = "|".join(activity_types) if activity_types else "<missing>"
        composite_counts[composite_key] = composite_counts.get(composite_key, 0) + 1
        if len(activity_types) > 1:
            mixed_rows += 1
        for activity_type in activity_types or ["<missing>"]:
            memberships[activity_type] = memberships.get(activity_type, 0) + 1

    records: list[dict[str, object]] = []
    for activity_type, count in sorted(memberships.items(), key=lambda item: (-item[1], item[0])):
        records.append(
            {
                "boundary": boundary,
                "split": split,
                "view": "exploded_membership",
                "activity_type": activity_type,
                "rows": int(count),
                "row_percent": 100.0 * count / row_count if row_count else 0.0,
            }
        )
    for activity_type, count in sorted(composite_counts.items(), key=lambda item: (-item[1], item[0])):
        records.append(
            {
                "boundary": boundary,
                "split": split,
                "view": "exact_composite",
                "activity_type": activity_type,
                "rows": int(count),
                "row_percent": 100.0 * count / row_count if row_count else 0.0,
            }
        )
    records.append(
        {
            "boundary": boundary,
            "split": split,
            "view": "mixed_rows",
            "activity_type": "<multiple activity types>",
            "rows": int(mixed_rows),
            "row_percent": 100.0 * mixed_rows / row_count if row_count else 0.0,
        }
    )
    return records


def _distribution_records(
    frame: pd.DataFrame,
    assay_stats: pd.DataFrame,
    *,
    boundary: str,
    split: str,
) -> list[dict[str, object]]:
    eligible_groups = set(
        assay_stats.loc[assay_stats["eligible_ranking_assay"], "assay_group_id"]
    )
    eligible_rows = frame.loc[frame["assay_group_id"].isin(eligible_groups)]
    eligible_assays = assay_stats.loc[assay_stats["eligible_ranking_assay"]]
    distributions = {
        ("all_assays", "ligands_per_assay"): assay_stats["ligands"],
        ("eligible_assays", "ligands_per_assay"): eligible_assays["ligands"],
        ("all_assays", "pchembl_span"): assay_stats["pchembl_span"],
        ("eligible_assays", "pchembl_span"): eligible_assays["pchembl_span"],
        ("all_rows", "pchembl_value"): frame["pchembl_value"],
        ("eligible_assay_rows", "pchembl_value"): eligible_rows["pchembl_value"],
        ("all_assays", "comparable_comparisons_per_assay"): assay_stats[
            "comparable_comparisons"
        ],
        ("eligible_assays", "comparable_comparisons_per_assay"): eligible_assays[
            "comparable_comparisons"
        ],
    }
    records: list[dict[str, object]] = []
    for (scope, metric), values in distributions.items():
        records.append(
            {
                "boundary": boundary,
                "split": split,
                "scope": scope,
                "metric": metric,
                **_distribution(values),
            }
        )
    return records


def _summary_record(
    frame: pd.DataFrame,
    assay_stats: pd.DataFrame,
    *,
    boundary: str,
    split: str,
) -> dict[str, object]:
    eligible = assay_stats.loc[assay_stats["eligible_ranking_assay"]]
    ligand_distribution = _distribution(assay_stats["ligands"])
    span_distribution = _distribution(assay_stats["pchembl_span"])
    pchembl_distribution = _distribution(frame["pchembl_value"])
    return {
        "boundary": boundary,
        "split": split,
        "rows": int(len(frame)),
        "assays": int(len(assay_stats)),
        "eligible_ranking_assays": int(len(eligible)),
        "eligible_assay_percent": 100.0 * len(eligible) / len(assay_stats) if len(assay_stats) else 0.0,
        "ligands_median": ligand_distribution["median"],
        "ligands_p90": ligand_distribution["p90"],
        "ligands_max": ligand_distribution["max"],
        "pchembl_span_median": span_distribution["median"],
        "pchembl_span_p90": span_distribution["p90"],
        "pchembl_span_max": span_distribution["max"],
        "comparable_comparisons_all_assays": int(assay_stats["comparable_comparisons"].sum()),
        "comparable_comparisons_eligible_assays": int(eligible["comparable_comparisons"].sum()),
        "pchembl_mean": pchembl_distribution["mean"],
        "pchembl_median": pchembl_distribution["median"],
        "pchembl_p10": float(frame["pchembl_value"].quantile(0.1)) if len(frame) else None,
        "pchembl_p90": pchembl_distribution["p90"],
        "positive_rows": int(frame["binary_label"].sum()),
        "positive_percent": 100.0 * float(frame["binary_label"].mean()) if len(frame) else 0.0,
    }


def analyze_boundary(
    frames: Mapping[str, pd.DataFrame],
    *,
    boundary: str,
    min_ligands: int,
    min_pchembl_span: float,
    affinity_margin: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_records: list[dict[str, object]] = []
    assay_frames: list[pd.DataFrame] = []
    distribution_records: list[dict[str, object]] = []
    activity_records: list[dict[str, object]] = []

    combined = pd.concat(
        [frame.assign(split=split) for split, frame in frames.items()],
        ignore_index=True,
    )
    scoped_frames = {"all": combined, **frames}
    for split, frame in scoped_frames.items():
        assay_stats = build_assay_statistics(
            frame,
            boundary=boundary,
            split=split,
            min_ligands=min_ligands,
            min_pchembl_span=min_pchembl_span,
            affinity_margin=affinity_margin,
        )
        summary_records.append(
            _summary_record(frame, assay_stats, boundary=boundary, split=split)
        )
        assay_frames.append(assay_stats)
        distribution_records.extend(
            _distribution_records(
                frame,
                assay_stats,
                boundary=boundary,
                split=split,
            )
        )
        activity_records.extend(
            _activity_type_statistics(frame, boundary=boundary, split=split)
        )
    return (
        pd.DataFrame.from_records(summary_records),
        pd.concat(assay_frames, ignore_index=True),
        pd.DataFrame.from_records(distribution_records),
        pd.DataFrame.from_records(activity_records),
    )


def _load_split_frames(dataset_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        path = dataset_dir / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing MMseqs split: {path}")
        frame = pd.read_parquet(path)
        missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        frames[split] = frame
    return frames


def _measure_lengths(
    tokenizer,
    texts: Sequence[str],
    *,
    batch_size: int,
    label: str,
) -> dict[str, int]:
    unique_texts = list(dict.fromkeys(str(text) for text in texts))
    lengths: dict[str, int] = {}
    for start in range(0, len(unique_texts), batch_size):
        batch = unique_texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            padding="longest",
            truncation=False,
            return_tensors="np",
        )
        batch_lengths = np.asarray(encoded["attention_mask"]).sum(axis=1)
        lengths.update(zip(batch, (int(value) for value in batch_lengths)))
        if start == 0 or start + batch_size >= len(unique_texts) or (start // batch_size) % 500 == 0:
            print(
                f"Measured {min(start + batch_size, len(unique_texts)):,}/"
                f"{len(unique_texts):,} unique {label} token lengths",
                flush=True,
            )
    return lengths


def token_filter_frames(
    frames: Mapping[str, pd.DataFrame],
    *,
    training_config_path: Path,
    cache_dir: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    from transformers import AutoTokenizer

    payload = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
    model_config = payload["model"]
    data_config = payload["data"]
    protein_tokenizer_name = model_config.get("protein_tokenizer_name_or_path") or model_config[
        "protein_model_name_or_path"
    ]
    molecule_tokenizer_name = model_config.get("molecule_tokenizer_name_or_path") or model_config[
        "molecule_model_name_or_path"
    ]
    protein_max_length = int(model_config["protein_max_length"])
    molecule_max_length = int(model_config["molecule_max_length"])
    batch_size = int(data_config["tokenization_batch_size"])

    cache_dir.mkdir(parents=True, exist_ok=True)
    protein_tokenizer = AutoTokenizer.from_pretrained(
        protein_tokenizer_name,
        cache_dir=cache_dir,
        clean_up_tokenization_spaces=False,
    )
    molecule_tokenizer = AutoTokenizer.from_pretrained(
        molecule_tokenizer_name,
        cache_dir=cache_dir,
        clean_up_tokenization_spaces=False,
    )

    combined = pd.concat(frames.values(), ignore_index=True)
    protein_lengths = _measure_lengths(
        protein_tokenizer,
        combined["protein_sequence"].drop_duplicates().tolist(),
        batch_size=batch_size,
        label="protein",
    )
    molecule_lengths = _measure_lengths(
        molecule_tokenizer,
        combined["compound_selfies"].drop_duplicates().tolist(),
        batch_size=batch_size,
        label="molecule",
    )

    filtered: dict[str, pd.DataFrame] = {}
    split_stats: dict[str, dict[str, int]] = {}
    for split, frame in frames.items():
        protein_length = frame["protein_sequence"].map(protein_lengths)
        molecule_length = frame["compound_selfies"].map(molecule_lengths)
        keep = protein_length.le(protein_max_length) & molecule_length.le(molecule_max_length)
        filtered[split] = frame.loc[keep].copy()
        split_stats[split] = {
            "input_rows": int(len(frame)),
            "retained_rows": int(keep.sum()),
            "dropped_rows": int((~keep).sum()),
            "dropped_for_protein_length": int(protein_length.gt(protein_max_length).sum()),
            "dropped_for_molecule_length": int(molecule_length.gt(molecule_max_length).sum()),
        }
    return filtered, {
        "protein_tokenizer": protein_tokenizer_name,
        "molecule_tokenizer": molecule_tokenizer_name,
        "protein_max_length": protein_max_length,
        "molecule_max_length": molecule_max_length,
        "tokenization_batch_size": batch_size,
        "split_stats": split_stats,
    }


def _format_number(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.3f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    labels = [column.replace("_", " ") for column in columns]
    lines = [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join("---" for _ in labels) + " |",
    ]
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format_number(value) for value in row) + " |")
    return "\n".join(lines)


def _write_markdown_report(
    path: Path,
    summary: pd.DataFrame,
    distributions: pd.DataFrame,
    activity_types: pd.DataFrame,
    metadata: Mapping[str, object],
) -> None:
    summary_columns = [
        "boundary",
        "split",
        "rows",
        "assays",
        "eligible_ranking_assays",
        "eligible_assay_percent",
        "ligands_median",
        "ligands_p90",
        "pchembl_span_median",
        "pchembl_span_p90",
        "comparable_comparisons_eligible_assays",
        "pchembl_median",
    ]
    total_activity = activity_types.loc[
        activity_types["split"].eq("all")
        & activity_types["view"].eq("exploded_membership")
    ]
    activity_columns = ["boundary", "activity_type", "rows", "row_percent"]
    selected_distributions = distributions.loc[
        distributions["split"].eq("all")
        & distributions["metric"].isin(
            ["ligands_per_assay", "pchembl_span", "pchembl_value"]
        )
    ]
    distribution_columns = [
        "boundary",
        "scope",
        "metric",
        "count",
        "mean",
        "std",
        "min",
        "p25",
        "median",
        "p75",
        "p90",
        "p95",
        "p99",
        "max",
    ]
    text = "\n".join(
        [
            "# ChEMBL 37 MMseqs50 Reward Model dataset statistics",
            "",
            "## Contract",
            "",
            f"- Eligible ranking assay: at least {metadata['min_ligands']} ligands and pChEMBL span >= {metadata['min_pchembl_span']}.",
            f"- Comparable comparison: an unordered within-assay ligand pair with absolute pChEMBL difference > {metadata['affinity_margin']}.",
            "- Activity-type membership explodes pipe-delimited types; percentages may sum above 100% when a curated row combines types.",
            "",
            "## Dataset summary",
            "",
            _markdown_table(summary, summary_columns),
            "",
            "## Full-dataset distributions",
            "",
            _markdown_table(selected_distributions, distribution_columns),
            "",
            "## Full-dataset activity types",
            "",
            _markdown_table(total_activity, activity_columns),
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate Reward Model ranking statistics for MMseqs-split ChEMBL data."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--hf-cache-dir", type=Path, default=Path("dataset/cache/huggingface"))
    parser.add_argument("--skip-token-filter", action="store_true")
    parser.add_argument("--min-ligands", type=int, default=DEFAULT_MIN_LIGANDS)
    parser.add_argument("--min-pchembl-span", type=float, default=DEFAULT_MIN_PCHEMBL_SPAN)
    parser.add_argument("--affinity-margin", type=float, default=DEFAULT_AFFINITY_MARGIN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_ligands < 2:
        raise ValueError("--min-ligands must be >= 2")
    if args.min_pchembl_span < 0.0 or not math.isfinite(args.min_pchembl_span):
        raise ValueError("--min-pchembl-span must be finite and >= 0")
    if args.affinity_margin < 0.0 or not math.isfinite(args.affinity_margin):
        raise ValueError("--affinity-margin must be finite and >= 0")

    dataset_dir = args.dataset_dir.resolve()
    output_dir = (args.output_dir or dataset_dir / "statistics" / "reward_ranking").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = _load_split_frames(dataset_dir)

    summaries: list[pd.DataFrame] = []
    assay_statistics: list[pd.DataFrame] = []
    distributions: list[pd.DataFrame] = []
    activity_types: list[pd.DataFrame] = []
    mmseq_results = analyze_boundary(
        frames,
        boundary="mmseq50_curated",
        min_ligands=args.min_ligands,
        min_pchembl_span=args.min_pchembl_span,
        affinity_margin=args.affinity_margin,
    )
    for destination, result in zip(
        (summaries, assay_statistics, distributions, activity_types), mmseq_results
    ):
        destination.append(result)

    token_filter_metadata: dict[str, object] | None = None
    if not args.skip_token_filter:
        token_filtered, token_filter_metadata = token_filter_frames(
            frames,
            training_config_path=args.training_config.resolve(),
            cache_dir=args.hf_cache_dir.resolve(),
        )
        token_results = analyze_boundary(
            token_filtered,
            boundary="reward_token_filtered",
            min_ligands=args.min_ligands,
            min_pchembl_span=args.min_pchembl_span,
            affinity_margin=args.affinity_margin,
        )
        for destination, result in zip(
            (summaries, assay_statistics, distributions, activity_types), token_results
        ):
            destination.append(result)

    summary_frame = pd.concat(summaries, ignore_index=True)
    assay_frame = pd.concat(assay_statistics, ignore_index=True)
    distribution_frame = pd.concat(distributions, ignore_index=True)
    activity_frame = pd.concat(activity_types, ignore_index=True)
    summary_frame.to_csv(output_dir / "dataset_summary.csv", index=False)
    assay_frame.to_parquet(output_dir / "assay_statistics.parquet", index=False)
    distribution_frame.to_csv(output_dir / "distribution_statistics.csv", index=False)
    activity_frame.to_csv(output_dir / "activity_type_counts.csv", index=False)

    metadata = {
        "dataset_dir": dataset_dir,
        "training_config": args.training_config.resolve(),
        "min_ligands": args.min_ligands,
        "min_pchembl_span": args.min_pchembl_span,
        "affinity_margin": args.affinity_margin,
        "affinity_margin_fold_change": 10**args.affinity_margin,
        "comparison_operator": "absolute pChEMBL difference > affinity_margin",
        "token_filter": token_filter_metadata,
    }
    payload = {
        "metadata": metadata,
        "summary": summary_frame.to_dict(orient="records"),
        "distributions": distribution_frame.to_dict(orient="records"),
        "activity_types": activity_frame.to_dict(orient="records"),
    }
    (output_dir / "statistics.json").write_text(
        json.dumps(payload, indent=2, default=_json_value),
        encoding="utf-8",
    )
    _write_markdown_report(
        output_dir / "report.md",
        summary_frame,
        distribution_frame,
        activity_frame,
        metadata,
    )
    print(json.dumps({"output_dir": str(output_dir), "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
