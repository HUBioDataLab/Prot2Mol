#!/usr/bin/env python3
"""Quantify drug and target prior tendency on a Prot2Mol CSV dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.stats import binom, chi2


def _stable_digest(value: str) -> bytes:
    return hashlib.blake2b(value.encode("utf-8"), digest_size=16).digest()


def _format_entity_value(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip()


def _rounded_tendency_bins(z_values: np.ndarray) -> np.ndarray:
    rounded = np.floor(np.clip(z_values, 0.0, 1.0) * 10.0 + 0.5).astype(np.int16)
    return np.clip(rounded, 0, 10)


def _overall_prior_tendency(z_values: np.ndarray) -> float:
    # The user-provided equation omits the averaging term; we include it so Z stays in [0.5, 1.0].
    return float(np.mean(np.abs(z_values - 0.5)) + 0.5)


def _safe_p_value(value: float) -> float:
    if not np.isfinite(value):
        return 1.0
    return float(max(min(value, 1.0), 0.0))


@dataclass
class EntityPriorTendencyResult:
    entity_label: str
    entity_count: int
    overall_prior_tendency_Z: float
    observed_statistic_T: float
    chi_square_statistic: float
    asymptotic_p_value: float
    monte_carlo_p_value: float | None
    monte_carlo_iterations: int
    monte_carlo_null: str | None
    min_occurrence: int
    median_occurrence: float
    mean_occurrence: float
    max_occurrence: int
    tendency_bin_frequencies: dict[str, float]


@dataclass
class PriorTendencyAnalysis:
    dataset_path: str
    label_column: str
    positive_threshold: float
    positive_label_rate_g: float
    row_count: int
    drug_identifier_column: str
    target_identifier_column: str
    z_definition: str
    drug: EntityPriorTendencyResult
    target: EntityPriorTendencyResult


def _counts_to_arrays(entity_counts: dict[bytes, list[int]]) -> tuple[np.ndarray, np.ndarray]:
    counts = np.fromiter((value[0] for value in entity_counts.values()), dtype=np.int32, count=len(entity_counts))
    positives = np.fromiter((value[1] for value in entity_counts.values()), dtype=np.int32, count=len(entity_counts))
    return counts, positives


def _stream_binary_entity_counts(
    csv_path: str | Path,
    label_column: str,
    positive_threshold: float,
    drug_identifier_column: str,
    target_identifier_column: str,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    csv_path = Path(csv_path)
    total_rows = 0
    total_positives = 0
    drug_counts: dict[bytes, list[int]] = {}
    target_counts: dict[bytes, list[int]] = {}

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} does not have a header row.")

        required = {label_column, drug_identifier_column, target_identifier_column}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise KeyError(f"Dataset is missing required columns: {sorted(missing)}")

        for row in reader:
            total_rows += 1
            label_value = float(row[label_column])
            is_positive = int(label_value >= positive_threshold)
            total_positives += is_positive

            drug_key = _stable_digest(_format_entity_value(row[drug_identifier_column]))
            target_key = _stable_digest(_format_entity_value(row[target_identifier_column]))

            drug_entry = drug_counts.setdefault(drug_key, [0, 0])
            drug_entry[0] += 1
            drug_entry[1] += is_positive

            target_entry = target_counts.setdefault(target_key, [0, 0])
            target_entry[0] += 1
            target_entry[1] += is_positive

    drug_n, drug_pos = _counts_to_arrays(drug_counts)
    target_n, target_pos = _counts_to_arrays(target_counts)
    return total_rows, total_positives, drug_n, drug_pos, target_n, target_pos


def _simulate_statistic_under_null(
    counts: np.ndarray,
    g: float,
    iterations: int,
    seed: int,
) -> tuple[float, str] | tuple[None, None]:
    if iterations <= 0:
        return None, None

    rng = np.random.default_rng(seed)
    observed = counts.shape[0]

    # Fast path for large, low-count entity spaces like compounds in this dataset.
    if observed > 100_000 and int(counts.max()) <= 512:
        unique_counts, multiplicities = np.unique(counts, return_counts=True)
        pmf_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for n in unique_counts.tolist():
            k_values = np.arange(n + 1)
            pmf = binom.pmf(k_values, n, g)
            pmf = pmf / pmf.sum()
            weights = n * ((k_values / max(n, 1)) - g) ** 2
            pmf_cache[int(n)] = (pmf.astype(np.float64, copy=False), weights.astype(np.float64, copy=False))

        stats = np.empty(iterations, dtype=np.float64)
        for idx in range(iterations):
            total = 0.0
            for n, multiplicity in zip(unique_counts.tolist(), multiplicities.tolist()):
                pmf, weights = pmf_cache[int(n)]
                sampled = rng.multinomial(int(multiplicity), pmf)
                total += float(np.dot(sampled, weights))
            stats[idx] = total
        return stats, "bernoulli_monte_carlo_grouped_by_occurrence"

    stats = np.empty(iterations, dtype=np.float64)
    counts_float = counts.astype(np.float64, copy=False)
    expected = counts_float * g
    for idx in range(iterations):
        sampled = rng.binomial(counts, g).astype(np.float64, copy=False)
        stats[idx] = float(np.sum(((sampled - expected) ** 2) / counts_float))
    return stats, "bernoulli_monte_carlo_per_entity"


def _compute_entity_result(
    entity_label: str,
    counts: np.ndarray,
    positives: np.ndarray,
    g: float,
    monte_carlo_iterations: int,
    seed: int,
) -> EntityPriorTendencyResult:
    counts_float = counts.astype(np.float64, copy=False)
    positives_float = positives.astype(np.float64, copy=False)
    z_values = positives_float / counts_float
    rounded_bins = _rounded_tendency_bins(z_values)
    frequency = np.bincount(rounded_bins, minlength=11).astype(np.float64) / max(len(z_values), 1)

    observed_statistic = float(np.sum(counts_float * np.square(z_values - g)))
    chi_square_statistic = float(observed_statistic / max(g * (1.0 - g), 1e-12))
    asymptotic_p_value = _safe_p_value(chi2.sf(chi_square_statistic, df=max(len(z_values) - 1, 1)))

    null_stats, monte_carlo_null = _simulate_statistic_under_null(
        counts=counts,
        g=g,
        iterations=monte_carlo_iterations,
        seed=seed,
    )
    monte_carlo_p_value = None
    if null_stats is not None:
        monte_carlo_p_value = float((1 + np.count_nonzero(null_stats >= observed_statistic)) / (1 + monte_carlo_iterations))

    return EntityPriorTendencyResult(
        entity_label=entity_label,
        entity_count=int(len(z_values)),
        overall_prior_tendency_Z=_overall_prior_tendency(z_values),
        observed_statistic_T=observed_statistic,
        chi_square_statistic=chi_square_statistic,
        asymptotic_p_value=asymptotic_p_value,
        monte_carlo_p_value=monte_carlo_p_value,
        monte_carlo_iterations=int(monte_carlo_iterations),
        monte_carlo_null=monte_carlo_null,
        min_occurrence=int(counts.min()),
        median_occurrence=float(np.median(counts_float)),
        mean_occurrence=float(counts_float.mean()),
        max_occurrence=int(counts.max()),
        tendency_bin_frequencies={f"{bin_idx / 10:.1f}": float(value) for bin_idx, value in enumerate(frequency.tolist())},
    )


def run_prior_tendency_analysis(
    csv_path: str | Path,
    label_column: str = "pchembl_value_Median",
    positive_threshold: float = 6.0,
    drug_identifier_column: str = "Compound_SMILES",
    target_identifier_column: str = "Target_FASTA",
    monte_carlo_iterations: int = 1000,
    seed: int = 42,
) -> PriorTendencyAnalysis:
    total_rows, total_positives, drug_n, drug_pos, target_n, target_pos = _stream_binary_entity_counts(
        csv_path=csv_path,
        label_column=label_column,
        positive_threshold=positive_threshold,
        drug_identifier_column=drug_identifier_column,
        target_identifier_column=target_identifier_column,
    )
    g = total_positives / max(total_rows, 1)

    return PriorTendencyAnalysis(
        dataset_path=str(Path(csv_path).resolve()),
        label_column=label_column,
        positive_threshold=float(positive_threshold),
        positive_label_rate_g=float(g),
        row_count=int(total_rows),
        drug_identifier_column=drug_identifier_column,
        target_identifier_column=target_identifier_column,
        z_definition="mean_i(|z_i - 0.5|) + 0.5",
        drug=_compute_entity_result(
            entity_label="Drug",
            counts=drug_n,
            positives=drug_pos,
            g=g,
            monte_carlo_iterations=monte_carlo_iterations,
            seed=seed,
        ),
        target=_compute_entity_result(
            entity_label="Target",
            counts=target_n,
            positives=target_pos,
            g=g,
            monte_carlo_iterations=monte_carlo_iterations,
            seed=seed + 1,
        ),
    )


def _write_histogram_csv(path: Path, analysis: PriorTendencyAnalysis) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["prior_tendency_bin", "drug_frequency", "target_frequency"])
        for bin_idx in range(11):
            key = f"{bin_idx / 10:.1f}"
            writer.writerow(
                [
                    key,
                    analysis.drug.tendency_bin_frequencies[key],
                    analysis.target.tendency_bin_frequencies[key],
                ]
            )


def _write_overall_z_csv(path: Path, analysis: PriorTendencyAnalysis) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "entity",
                "overall_prior_tendency_Z",
                "asymptotic_p_value",
                "monte_carlo_p_value",
                "entity_count",
            ]
        )
        writer.writerow(
            [
                analysis.drug.entity_label,
                analysis.drug.overall_prior_tendency_Z,
                analysis.drug.asymptotic_p_value,
                analysis.drug.monte_carlo_p_value,
                analysis.drug.entity_count,
            ]
        )
        writer.writerow(
            [
                analysis.target.entity_label,
                analysis.target.overall_prior_tendency_Z,
                analysis.target.asymptotic_p_value,
                analysis.target.monte_carlo_p_value,
                analysis.target.entity_count,
            ]
        )


def _format_p_value_for_plot(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def _preferred_p_value(result: EntityPriorTendencyResult) -> float | None:
    if result.monte_carlo_p_value is not None:
        return result.monte_carlo_p_value
    return result.asymptotic_p_value


def _render_figure(path: Path, analysis: PriorTendencyAnalysis) -> None:
    mpl_dir = path.parent / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

    import matplotlib.pyplot as plt

    bins = np.arange(11)
    labels = [f"{value / 10:.1f}" for value in bins]
    drug_freq = np.array([analysis.drug.tendency_bin_frequencies[label] for label in labels], dtype=np.float64)
    target_freq = np.array([analysis.target.tendency_bin_frequencies[label] for label in labels], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.38

    drug_fill, drug_edge = "#8CB3E6", "#1F4AA8"
    target_fill, target_edge = "#F09AB8", "#A11A52"

    axes[0].bar(x - width / 2, drug_freq, width=width, color=drug_fill, edgecolor=drug_edge, linewidth=2, label="Drug")
    axes[0].bar(x + width / 2, target_freq, width=width, color=target_fill, edgecolor=target_edge, linewidth=2, label="Target")
    axes[0].set_xticks(x, labels)
    axes[0].set_xlabel(r"Prior Tendency $z_i$")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Papyrus Prior Tendency")
    axes[0].legend(loc="upper right", frameon=True)
    axes[0].text(
        0.97,
        0.58,
        "\n".join(
            [
                f"P_d = {_format_p_value_for_plot(_preferred_p_value(analysis.drug))}",
                f"P_t = {_format_p_value_for_plot(_preferred_p_value(analysis.target))}",
                f"g = {analysis.positive_label_rate_g:.3f}",
            ]
        ),
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=14,
    )

    z_values = [analysis.drug.overall_prior_tendency_Z, analysis.target.overall_prior_tendency_Z]
    axes[1].bar(
        [0, 1],
        z_values,
        width=0.6,
        color=[drug_fill, target_fill],
        edgecolor=[drug_edge, target_edge],
        linewidth=2,
    )
    axes[1].set_xticks([0, 1], ["Drug", "Target"])
    axes[1].set_ylabel("Overall Prior Tendency Z")
    axes[1].set_ylim(0.5, min(1.0, max(z_values) + 0.08))
    axes[1].set_title("Overall Prior Tendency")
    for idx, value in enumerate(z_values):
        axes[1].text(idx, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=12)

    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _to_jsonable(analysis: PriorTendencyAnalysis) -> dict[str, object]:
    payload = asdict(analysis)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze drug/target prior tendency on a Prot2Mol CSV.")
    parser.add_argument("csv_path", help="Path to the CSV dataset.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for summary tables and figure.")
    parser.add_argument("--label-column", default="pchembl_value_Median", help="Continuous label column to binarize.")
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=6.0,
        help="Rows with label >= threshold are treated as positive interactions.",
    )
    parser.add_argument("--drug-column", default="Compound_SMILES", help="Column used as the drug entity identifier.")
    parser.add_argument("--target-column", default="Target_FASTA", help="Column used as the target entity identifier.")
    parser.add_argument("--monte-carlo-iterations", type=int, default=1000, help="Monte Carlo iterations for null simulation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for null simulation.")
    args = parser.parse_args()

    analysis = run_prior_tendency_analysis(
        csv_path=args.csv_path,
        label_column=args.label_column,
        positive_threshold=args.positive_threshold,
        drug_identifier_column=args.drug_column,
        target_identifier_column=args.target_column,
        monte_carlo_iterations=args.monte_carlo_iterations,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    histogram_path = args.output_dir / "prior_tendency_histogram.csv"
    overall_z_path = args.output_dir / "overall_prior_tendency.csv"
    figure_path = args.output_dir / "prior_tendency_figure.png"

    summary_path.write_text(json.dumps(_to_jsonable(analysis), indent=2), encoding="utf-8")
    _write_histogram_csv(histogram_path, analysis)
    _write_overall_z_csv(overall_z_path, analysis)
    _render_figure(figure_path, analysis)

    print(f"Saved summary to {summary_path}")
    print(f"Saved histogram to {histogram_path}")
    print(f"Saved Z summary to {overall_z_path}")
    print(f"Saved figure to {figure_path}")


if __name__ == "__main__":
    main()
