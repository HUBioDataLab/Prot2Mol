from __future__ import annotations

import html
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

from ..model.fusion import masked_pool
from ..training.data import RewardPairCollator


PROTEIN_TOKEN_COLUMNS = ("protein_input_ids", "protein_attention_mask")
MOLECULE_TOKEN_COLUMNS = ("molecule_input_ids", "molecule_attention_mask")
PREDICTION_COLUMNS = (
    "example_id",
    "group_id",
    "target_chembl_id",
    "assay_id",
    "compound_id",
    "pchembl_value",
    "binary_label",
)


def _safe_correlation(
    left: Sequence[float],
    right: Sequence[float],
    *,
    method: str,
) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left_array) & np.isfinite(right_array)
    left_array = left_array[finite]
    right_array = right_array[finite]
    if left_array.size < 2:
        return float("nan")
    if np.unique(left_array).size < 2 or np.unique(right_array).size < 2:
        return float("nan")
    result = (
        spearmanr(left_array, right_array)
        if method == "spearman"
        else pearsonr(left_array, right_array)
    )
    return float(getattr(result, "statistic", result[0]))


class _FenwickTree:
    def __init__(self, size: int):
        self.values = [0] * (int(size) + 1)

    def add(self, index: int, value: int = 1) -> None:
        position = int(index) + 1
        while position < len(self.values):
            self.values[position] += int(value)
            position += position & -position

    def prefix_sum(self, end_exclusive: int) -> int:
        total = 0
        position = int(end_exclusive)
        while position > 0:
            total += self.values[position]
            position -= position & -position
        return total


def margin_pair_counts(
    scores: Sequence[float],
    targets: Sequence[float],
    *,
    affinity_margin: float,
) -> tuple[float, int]:
    """Count correct eligible pairs exactly in O(n log n) time."""
    score_array = np.asarray(scores, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    if score_array.shape != target_array.shape:
        raise ValueError("scores and targets must have the same shape")
    if score_array.ndim != 1:
        raise ValueError("scores and targets must be one-dimensional")
    if not np.isfinite(score_array).all() or not np.isfinite(target_array).all():
        raise ValueError("scores and targets must be finite")
    if affinity_margin < 0.0 or not math.isfinite(float(affinity_margin)):
        raise ValueError("affinity_margin must be finite and >= 0")
    if score_array.size < 2:
        return 0.0, 0

    target_order = np.argsort(target_array, kind="stable")
    score_levels = np.unique(score_array)
    score_ranks = np.searchsorted(score_levels, score_array)
    tree = _FenwickTree(len(score_levels))
    eligible_cursor = 0
    inserted_count = 0
    correct_count = 0.0
    pair_count = 0

    for current_position, current_index in enumerate(target_order):
        current_target = target_array[current_index]
        threshold = current_target - float(affinity_margin)
        while eligible_cursor < current_position:
            candidate_index = target_order[eligible_cursor]
            if not target_array[candidate_index] < threshold:
                break
            tree.add(int(score_ranks[candidate_index]))
            inserted_count += 1
            eligible_cursor += 1

        current_rank = int(score_ranks[current_index])
        lower_scores = tree.prefix_sum(current_rank)
        equal_scores = tree.prefix_sum(current_rank + 1) - lower_scores
        correct_count += lower_scores + 0.5 * equal_scores
        pair_count += inserted_count

    return float(correct_count), int(pair_count)


def select_complete_assays(
    dataset,
    *,
    max_assays: Optional[int] = None,
    seed: int = 42,
    min_size: int = 3,
    min_pchembl_span: float = 0.5,
    assay_ids: Optional[Sequence[str]] = None,
):
    """Select complete eligible assays, never partial rows from an assay."""
    if max_assays is not None and max_assays <= 0:
        raise ValueError("max_assays must be > 0 when provided")
    if min_size < 2:
        raise ValueError("min_size must be >= 2")
    group_ids = np.asarray(dataset["group_id"], dtype=object)
    targets = np.asarray(dataset["pchembl_value"], dtype=np.float64)

    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids.tolist()):
        grouped_indices[str(group_id)].append(index)

    eligible = []
    for group_id in sorted(grouped_indices):
        indices = grouped_indices[group_id]
        group_targets = targets[indices]
        if len(indices) < min_size:
            continue
        if float(group_targets.max() - group_targets.min()) < min_pchembl_span:
            continue
        eligible.append(group_id)

    if assay_ids is not None:
        requested = [str(group_id) for group_id in assay_ids]
        missing = sorted(set(requested) - set(eligible))
        if missing:
            raise ValueError(
                "Assay manifest contains groups absent or ineligible in this split: "
                f"{missing[:5]}"
            )
        selected_groups = requested
    elif max_assays is not None and len(eligible) > max_assays:
        generator = np.random.default_rng(int(seed))
        selected_positions = generator.choice(
            len(eligible),
            size=int(max_assays),
            replace=False,
        )
        selected_groups = [eligible[index] for index in sorted(selected_positions)]
    else:
        selected_groups = eligible

    selected_set = set(selected_groups)
    selected_indices = [
        index for index, group_id in enumerate(group_ids) if str(group_id) in selected_set
    ]
    return dataset.select(selected_indices), selected_groups


def write_assay_manifest(
    path: str,
    *,
    split: str,
    assay_ids: Sequence[str],
    dataset,
    seed: int,
) -> str:
    counts: dict[str, int] = defaultdict(int)
    for group_id in dataset["group_id"]:
        counts[str(group_id)] += 1
    payload = {
        "split": str(split),
        "seed": int(seed),
        "num_assays": len(assay_ids),
        "num_examples": len(dataset),
        "assays": [
            {"group_id": str(group_id), "num_examples": counts[str(group_id)]}
            for group_id in assay_ids
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return os.path.abspath(path)


def load_assay_manifest(path: str, *, split: Optional[str] = None) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if split is not None and str(payload.get("split")) != str(split):
        raise ValueError(
            f"Manifest split {payload.get('split')!r} does not match {split!r}"
        )
    assays = payload.get("assays")
    if not isinstance(assays, list) or not assays:
        raise ValueError("Assay manifest must contain a non-empty assays list")
    return [str(record["group_id"]) for record in assays]


def _hybrid_row(
    metadata_row: Mapping[str, Any],
    protein_row: Mapping[str, Any],
    molecule_row: Mapping[str, Any],
) -> dict[str, Any]:
    row = dict(metadata_row)
    for key in PROTEIN_TOKEN_COLUMNS:
        row[key] = protein_row[key]
    for key in MOLECULE_TOKEN_COLUMNS:
        row[key] = molecule_row[key]
    return row


def score_dataset_pairs(
    model,
    dataset,
    *,
    batch_size: int,
    device: torch.device,
    protein_source_indices: Optional[Sequence[int]] = None,
    molecule_source_indices: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Score arbitrary protein/molecule row pairings with original row metadata."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    row_count = len(dataset)
    protein_indices = (
        np.arange(row_count, dtype=np.int64)
        if protein_source_indices is None
        else np.asarray(protein_source_indices, dtype=np.int64)
    )
    molecule_indices = (
        np.arange(row_count, dtype=np.int64)
        if molecule_source_indices is None
        else np.asarray(molecule_source_indices, dtype=np.int64)
    )
    if protein_indices.shape != (row_count,) or molecule_indices.shape != (row_count,):
        raise ValueError("source index arrays must have one entry per dataset row")
    if row_count and (
        protein_indices.min() < 0
        or molecule_indices.min() < 0
        or protein_indices.max() >= row_count
        or molecule_indices.max() >= row_count
    ):
        raise ValueError("source indices must refer to rows in the selected dataset")

    collator = RewardPairCollator(
        dynamic_padding=True,
        protein_pad_token_id=getattr(
            getattr(model, "protein_tokenizer", None), "pad_token_id", 0
        ),
        molecule_pad_token_id=getattr(
            getattr(model, "molecule_tokenizer", None), "pad_token_id", 0
        ),
    )
    was_training = model.training
    model.eval()
    records: list[dict[str, Any]] = []
    try:
        for start in range(0, row_count, batch_size):
            end = min(start + batch_size, row_count)
            rows = []
            metadata_rows = []
            protein_source_rows = []
            molecule_source_rows = []
            for index in range(start, end):
                metadata_row = dict(dataset[index])
                protein_row = dict(dataset[int(protein_indices[index])])
                molecule_row = dict(dataset[int(molecule_indices[index])])
                rows.append(_hybrid_row(metadata_row, protein_row, molecule_row))
                metadata_rows.append(metadata_row)
                protein_source_rows.append(protein_row)
                molecule_source_rows.append(molecule_row)
            token_batch = {
                key: value.to(device)
                for key, value in collator.collate_example_tokens(rows).items()
            }
            with torch.no_grad():
                outputs = model(**token_batch, return_dict=True)
            scores = outputs.ranking_score.detach().float().cpu().numpy()
            logits = outputs.activity_logits.detach().float().cpu().numpy()
            probabilities = outputs.activity_probability.detach().float().cpu().numpy()
            embedding_norms = (
                outputs.joint_embedding.detach().float().norm(dim=-1).cpu().numpy()
            )
            for local_index, metadata_row in enumerate(metadata_rows):
                protein_source_row = protein_source_rows[local_index]
                molecule_source_row = molecule_source_rows[local_index]
                record = {
                    column: metadata_row[column] for column in PREDICTION_COLUMNS
                }
                record.update(
                    {
                        "ranking_score": float(scores[local_index]),
                        "activity_logit": float(logits[local_index]),
                        "activity_probability": float(probabilities[local_index]),
                        "joint_embedding_norm": float(embedding_norms[local_index]),
                        "protein_source_index": int(protein_indices[start + local_index]),
                        "molecule_source_index": int(molecule_indices[start + local_index]),
                        "protein_source_group_id": str(
                            protein_source_row["group_id"]
                        ),
                        "molecule_source_group_id": str(
                            molecule_source_row["group_id"]
                        ),
                        "molecule_source_compound_id": str(
                            molecule_source_row["compound_id"]
                        ),
                    }
                )
                records.append(record)
    finally:
        model.train(was_training)
    return pd.DataFrame.from_records(records)


def score_fusion_cosines(
    model,
    dataset,
    *,
    batch_size: int,
    device: torch.device,
    protein_source_indices: Optional[Sequence[int]] = None,
    molecule_source_indices: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Measure cosine similarity before fusion and after fusion plus residuals."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    row_count = len(dataset)
    protein_indices = (
        np.arange(row_count, dtype=np.int64)
        if protein_source_indices is None
        else np.asarray(protein_source_indices, dtype=np.int64)
    )
    molecule_indices = (
        np.arange(row_count, dtype=np.int64)
        if molecule_source_indices is None
        else np.asarray(molecule_source_indices, dtype=np.int64)
    )
    if protein_indices.shape != (row_count,) or molecule_indices.shape != (row_count,):
        raise ValueError("source index arrays must have one entry per dataset row")
    if row_count and (
        protein_indices.min() < 0
        or molecule_indices.min() < 0
        or protein_indices.max() >= row_count
        or molecule_indices.max() >= row_count
    ):
        raise ValueError("source indices must refer to rows in the selected dataset")

    pooling_type = str(getattr(getattr(model, "config", None), "pooling_type", "mean"))
    collator = RewardPairCollator(
        dynamic_padding=True,
        protein_pad_token_id=getattr(
            getattr(model, "protein_tokenizer", None), "pad_token_id", 0
        ),
        molecule_pad_token_id=getattr(
            getattr(model, "molecule_tokenizer", None), "pad_token_id", 0
        ),
    )
    was_training = model.training
    model.eval()
    records: list[dict[str, Any]] = []
    try:
        for start in range(0, row_count, batch_size):
            end = min(start + batch_size, row_count)
            rows = []
            metadata_rows = []
            protein_source_rows = []
            molecule_source_rows = []
            for index in range(start, end):
                metadata_row = dict(dataset[index])
                protein_row = dict(dataset[int(protein_indices[index])])
                molecule_row = dict(dataset[int(molecule_indices[index])])
                rows.append(_hybrid_row(metadata_row, protein_row, molecule_row))
                metadata_rows.append(metadata_row)
                protein_source_rows.append(protein_row)
                molecule_source_rows.append(molecule_row)
            token_batch = {
                key: value.to(device)
                for key, value in collator.collate_example_tokens(rows).items()
            }
            with torch.no_grad():
                outputs = model(
                    **token_batch,
                    return_token_embeddings=True,
                    return_dict=True,
                )
            required_outputs = {
                "protein_token_embeddings": outputs.protein_token_embeddings,
                "molecule_token_embeddings": outputs.molecule_token_embeddings,
                "fused_protein_tokens": outputs.fused_protein_tokens,
                "fused_molecule_tokens": outputs.fused_molecule_tokens,
                "protein_attention_mask": outputs.protein_attention_mask,
                "molecule_attention_mask": outputs.molecule_attention_mask,
            }
            missing = [name for name, value in required_outputs.items() if value is None]
            if missing:
                raise ValueError(
                    "Model did not return representations required for fusion cosine "
                    f"analysis: {missing}"
                )

            pre_protein = masked_pool(
                outputs.protein_token_embeddings,
                outputs.protein_attention_mask,
                pooling_type,
            ).float()
            pre_molecule = masked_pool(
                outputs.molecule_token_embeddings,
                outputs.molecule_attention_mask,
                pooling_type,
            ).float()
            post_protein = masked_pool(
                outputs.fused_protein_tokens,
                outputs.protein_attention_mask,
                pooling_type,
            ).float()
            post_molecule = masked_pool(
                outputs.fused_molecule_tokens,
                outputs.molecule_attention_mask,
                pooling_type,
            ).float()

            pre_cosines = F.cosine_similarity(pre_protein, pre_molecule, dim=-1)
            post_cosines = F.cosine_similarity(post_protein, post_molecule, dim=-1)
            tensors = {
                "pre_fusion_cosine": pre_cosines,
                "post_fusion_cosine": post_cosines,
                "pre_protein_norm": pre_protein.norm(dim=-1),
                "pre_molecule_norm": pre_molecule.norm(dim=-1),
                "post_protein_norm": post_protein.norm(dim=-1),
                "post_molecule_norm": post_molecule.norm(dim=-1),
            }
            arrays = {
                name: value.detach().cpu().numpy() for name, value in tensors.items()
            }
            for local_index, metadata_row in enumerate(metadata_rows):
                protein_source_row = protein_source_rows[local_index]
                molecule_source_row = molecule_source_rows[local_index]
                record = {
                    column: metadata_row[column] for column in PREDICTION_COLUMNS
                }
                record.update(
                    {
                        name: float(values[local_index])
                        for name, values in arrays.items()
                    }
                )
                record.update(
                    {
                        "protein_source_index": int(
                            protein_indices[start + local_index]
                        ),
                        "molecule_source_index": int(
                            molecule_indices[start + local_index]
                        ),
                        "protein_source_group_id": str(
                            protein_source_row["group_id"]
                        ),
                        "molecule_source_group_id": str(
                            molecule_source_row["group_id"]
                        ),
                        "molecule_source_compound_id": str(
                            molecule_source_row["compound_id"]
                        ),
                    }
                )
                records.append(record)
    finally:
        model.train(was_training)
    return pd.DataFrame.from_records(records)


def _normalized_entropy(scores: np.ndarray, temperature: float) -> float:
    scaled = scores.astype(np.float64) / float(temperature)
    scaled -= scaled.max()
    probabilities = np.exp(scaled)
    probabilities /= probabilities.sum()
    entropy = -float(np.sum(probabilities * np.log(np.clip(probabilities, 1e-300, None))))
    return entropy / math.log(len(scores)) if len(scores) > 1 else float("nan")


def analyze_predictions(
    scored_rows: pd.DataFrame,
    *,
    split: str,
    affinity_margin: float,
    temperature: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if scored_rows.empty:
        raise ValueError("scored_rows must not be empty")
    predictions = scored_rows.copy()
    predictions.insert(0, "split", str(split))
    grouped = predictions.groupby("group_id", sort=True, group_keys=False)
    predictions["ranking_score_centered"] = grouped["ranking_score"].transform(
        lambda values: values - values.mean()
    )
    predictions["pchembl_centered"] = grouped["pchembl_value"].transform(
        lambda values: values - values.mean()
    )
    predictions["true_rank"] = grouped["pchembl_value"].rank(
        method="average", ascending=False
    )
    predictions["predicted_rank"] = grouped["ranking_score"].rank(
        method="average", ascending=False
    )
    predictions["rank_error"] = predictions["predicted_rank"] - predictions["true_rank"]
    predictions["absolute_rank_error"] = predictions["rank_error"].abs()
    predictions["true_rank_percentile"] = grouped["true_rank"].transform(
        lambda values: (values - 1.0) / max(len(values) - 1, 1)
    )
    predictions["predicted_rank_percentile"] = grouped["predicted_rank"].transform(
        lambda values: (values - 1.0) / max(len(values) - 1, 1)
    )

    strength_probabilities = np.zeros(len(predictions), dtype=np.float64)
    assay_records: list[dict[str, Any]] = []
    for group_id, group in predictions.groupby("group_id", sort=True):
        indices = group.index.to_numpy()
        scores = group["ranking_score"].to_numpy(dtype=np.float64)
        targets = group["pchembl_value"].to_numpy(dtype=np.float64)
        scaled = scores / float(temperature)
        probabilities = np.exp(scaled - scaled.max())
        probabilities /= probabilities.sum()
        strength_probabilities[indices] = probabilities
        correct_pairs, pair_count = margin_pair_counts(
            scores,
            targets,
            affinity_margin=affinity_margin,
        )
        predicted_top = group.loc[group["ranking_score"].idxmax()]
        max_target = float(group["pchembl_value"].max())
        assay_records.append(
            {
                "split": str(split),
                "group_id": str(group_id),
                "target_chembl_id": str(group["target_chembl_id"].iloc[0]),
                "assay_id": str(group["assay_id"].iloc[0]),
                "num_examples": int(len(group)),
                "pchembl_span": float(targets.max() - targets.min()),
                "num_unique_pchembl": int(np.unique(targets).size),
                "score_mean": float(scores.mean()),
                "score_std": float(scores.std()),
                "score_range": float(scores.max() - scores.min()),
                "normalized_entropy": _normalized_entropy(scores, temperature),
                "spearman": _safe_correlation(scores, targets, method="spearman"),
                "pearson": _safe_correlation(scores, targets, method="pearson"),
                "score_activity_logit_spearman": _safe_correlation(
                    scores,
                    group["activity_logit"],
                    method="spearman",
                ),
                "score_activity_logit_pearson": _safe_correlation(
                    scores,
                    group["activity_logit"],
                    method="pearson",
                ),
                "margin_pair_correct": float(correct_pairs),
                "margin_pair_count": int(pair_count),
                "margin_pair_accuracy": (
                    float(correct_pairs / pair_count) if pair_count else float("nan")
                ),
                "classification_accuracy": float(
                    np.mean(
                        (group["activity_probability"].to_numpy() >= 0.5)
                        == (group["binary_label"].to_numpy() >= 0.5)
                    )
                ),
                "predicted_top_compound_id": str(predicted_top["compound_id"]),
                "predicted_top_pchembl": float(predicted_top["pchembl_value"]),
                "top1_pchembl_regret": max_target
                - float(predicted_top["pchembl_value"]),
                "top1_correct": float(
                    math.isclose(float(predicted_top["pchembl_value"]), max_target)
                ),
            }
        )
    predictions["pl_strength_probability"] = strength_probabilities
    assay_summary = pd.DataFrame.from_records(assay_records)

    finite_spearman = assay_summary[np.isfinite(assay_summary["spearman"])]
    total_margin_pairs = int(assay_summary["margin_pair_count"].sum())
    split_summary = {
        "split": str(split),
        "num_examples": int(len(predictions)),
        "num_assays": int(len(assay_summary)),
        "macro_spearman": (
            float(finite_spearman["spearman"].mean())
            if not finite_spearman.empty
            else float("nan")
        ),
        "weighted_spearman": (
            float(
                np.average(
                    finite_spearman["spearman"],
                    weights=finite_spearman["num_examples"],
                )
            )
            if not finite_spearman.empty
            else float("nan")
        ),
        "margin_pair_accuracy": (
            float(assay_summary["margin_pair_correct"].sum() / total_margin_pairs)
            if total_margin_pairs
            else float("nan")
        ),
        "margin_pair_count": total_margin_pairs,
        "within_assay_score_pchembl_spearman": _safe_correlation(
            predictions["ranking_score_centered"],
            predictions["pchembl_centered"],
            method="spearman",
        ),
        "within_assay_score_activity_logit_spearman": _safe_correlation(
            predictions["ranking_score_centered"],
            predictions.groupby("group_id")["activity_logit"].transform(
                lambda values: values - values.mean()
            ),
            method="spearman",
        ),
        "mean_normalized_entropy": float(assay_summary["normalized_entropy"].mean()),
        "mean_score_std": float(assay_summary["score_std"].mean()),
        "mean_top1_pchembl_regret": float(
            assay_summary["top1_pchembl_regret"].mean()
        ),
        "top1_accuracy": float(assay_summary["top1_correct"].mean()),
    }
    return predictions, assay_summary, split_summary


def _scatter_svg(group: pd.DataFrame, *, width: int = 520, height: int = 260) -> str:
    pad = 38
    if len(group) > 2000:
        ordered = group.sort_values(["pchembl_value", "ranking_score"])
        display_positions = np.linspace(0, len(ordered) - 1, 2000, dtype=int)
        group = ordered.iloc[display_positions]
    x_values = group["pchembl_value"].to_numpy(dtype=np.float64)
    y_values = group["ranking_score_centered"].to_numpy(dtype=np.float64)
    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_min, y_max = float(y_values.min()), float(y_values.max())
    x_span = max(x_max - x_min, 1e-8)
    y_span = max(y_max - y_min, 1e-8)
    circles = []
    for (_, row), x_value, y_value in zip(group.iterrows(), x_values, y_values):
        x = pad + (x_value - x_min) / x_span * (width - 2 * pad)
        y = height - pad - (y_value - y_min) / y_span * (height - 2 * pad)
        probability = float(row["activity_probability"])
        red = int(round(220 * probability))
        blue = int(round(220 * (1.0 - probability)))
        title = html.escape(
            f"{row['compound_id']}: pChEMBL={x_value:.3f}, score={row['ranking_score']:.3f}"
        )
        circles.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" '
            f'fill="rgb({red},70,{blue})"><title>{title}</title></circle>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img">'
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#555"/>'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#555"/>'
        + "".join(circles)
        + f'<text x="{width/2:.0f}" y="{height-5}" text-anchor="middle">pChEMBL</text>'
        + (
            f'<text x="12" y="{height/2:.0f}" '
            f'transform="rotate(-90 12 {height/2:.0f})" '
            'text-anchor="middle">centered score</text>'
        )
        + "</svg>"
    )


def _write_html_report(
    path: str,
    predictions: pd.DataFrame,
    assay_summary: pd.DataFrame,
    split_summary: Mapping[str, Any],
) -> None:
    ordered = assay_summary.sort_values("spearman", na_position="last")
    representative_ids: list[str] = []
    if not ordered.empty:
        positions = sorted(set([0, len(ordered) // 2, len(ordered) - 1]))
        representative_ids.extend(
            str(ordered.iloc[position]["group_id"]) for position in positions
        )
    worst_errors = predictions.nlargest(20, "absolute_rank_error")
    cards = []
    for group_id in representative_ids:
        group = predictions[predictions["group_id"] == group_id]
        assay = assay_summary[assay_summary["group_id"] == group_id].iloc[0]
        cards.append(
            f"<h3>{html.escape(group_id)}; Spearman={assay['spearman']:.3f}</h3>"
            + _scatter_svg(group)
        )
    error_columns = [
        "group_id",
        "compound_id",
        "pchembl_value",
        "ranking_score",
        "true_rank",
        "predicted_rank",
        "absolute_rank_error",
    ]
    error_table = worst_errors[error_columns].to_html(index=False, escape=True)
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Ranking-head inspection</title>
<style>
body{{font-family:system-ui;margin:24px;max-width:1200px}}
table{{border-collapse:collapse;font-size:12px}}
th,td{{border:1px solid #ddd;padding:4px}}
svg{{border:1px solid #ddd;max-width:100%}}
</style>
</head><body><h1>Ranking-head inspection: {html.escape(str(split_summary['split']))}</h1>
<pre>{html.escape(json.dumps(dict(split_summary), indent=2, sort_keys=True))}</pre>
{''.join(cards)}
<h2>Largest rank errors</h2>
{error_table}
</body></html>"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(document)


def write_prediction_analysis(
    output_dir: str,
    *,
    predictions: pd.DataFrame,
    assay_summary: pd.DataFrame,
    split_summary: Mapping[str, Any],
) -> dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    split = str(split_summary["split"])
    paths = {
        "predictions": os.path.abspath(
            os.path.join(output_dir, f"{split}_ranking_predictions.parquet")
        ),
        "assay_summary": os.path.abspath(
            os.path.join(output_dir, f"{split}_assay_summary.csv")
        ),
        "largest_rank_errors": os.path.abspath(
            os.path.join(output_dir, f"{split}_largest_rank_errors.csv")
        ),
        "summary": os.path.abspath(os.path.join(output_dir, f"{split}_summary.json")),
        "html_report": os.path.abspath(
            os.path.join(output_dir, f"{split}_report.html")
        ),
    }
    predictions.to_parquet(paths["predictions"], index=False)
    assay_summary.to_csv(paths["assay_summary"], index=False)
    predictions.nlargest(200, "absolute_rank_error").to_csv(
        paths["largest_rank_errors"], index=False
    )
    with open(paths["summary"], "w", encoding="utf-8") as handle:
        json.dump(dict(split_summary), handle, indent=2, sort_keys=True)
    _write_html_report(
        paths["html_report"],
        predictions,
        assay_summary,
        split_summary,
    )
    return paths


def _weighted_group_spearman(
    frame: pd.DataFrame,
    left_column: str,
    right_column: str,
) -> float:
    correlations = []
    weights = []
    for _, group in frame.groupby("group_id", sort=True):
        correlation = _safe_correlation(
            group[left_column], group[right_column], method="spearman"
        )
        if math.isfinite(correlation):
            correlations.append(correlation)
            weights.append(len(group))
    return (
        float(np.average(correlations, weights=weights))
        if correlations
        else float("nan")
    )


def _cyclic_derangement(size: int, generator: np.random.Generator) -> np.ndarray:
    if size < 2:
        raise ValueError("A shuffle sensitivity test requires at least two items")
    cycle = generator.permutation(size)
    sources = np.empty(size, dtype=np.int64)
    sources[cycle] = np.roll(cycle, 1)
    if np.any(sources == np.arange(size)):
        raise RuntimeError("Failed to build a derangement")
    return sources


def run_input_sensitivity(
    model,
    dataset,
    *,
    split: str,
    batch_size: int,
    device: torch.device,
    num_shuffles: int = 3,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if num_shuffles <= 0:
        raise ValueError("num_shuffles must be > 0")
    baseline = score_dataset_pairs(
        model,
        dataset,
        batch_size=batch_size,
        device=device,
    )
    group_ids = [str(value) for value in dataset["group_id"]]
    unique_groups = sorted(set(group_ids))
    if len(unique_groups) < 2:
        raise ValueError("Protein shuffling requires at least two assays")
    first_group_index = {}
    for index, group_id in enumerate(group_ids):
        first_group_index.setdefault(group_id, index)
    generator = np.random.default_rng(int(seed))
    frames = []
    repeat_records = []

    for repeat in range(int(num_shuffles)):
        group_sources = _cyclic_derangement(len(unique_groups), generator)
        source_group_by_group = {
            unique_groups[index]: unique_groups[int(group_sources[index])]
            for index in range(len(unique_groups))
        }
        protein_indices = np.asarray(
            [first_group_index[source_group_by_group[group_id]] for group_id in group_ids],
            dtype=np.int64,
        )
        molecule_indices = _cyclic_derangement(len(dataset), generator)
        variants = {
            "protein_shuffled": score_dataset_pairs(
                model,
                dataset,
                batch_size=batch_size,
                device=device,
                protein_source_indices=protein_indices,
            ),
            "ligand_shuffled": score_dataset_pairs(
                model,
                dataset,
                batch_size=batch_size,
                device=device,
                molecule_source_indices=molecule_indices,
            ),
        }
        for perturbation, variant in variants.items():
            frame = baseline[list(PREDICTION_COLUMNS)].copy()
            frame.insert(0, "split", str(split))
            frame.insert(1, "repeat", repeat)
            frame.insert(2, "perturbation", perturbation)
            frame["baseline_score"] = baseline["ranking_score"].to_numpy()
            frame["perturbed_score"] = variant["ranking_score"].to_numpy()
            frame["raw_score_delta"] = (
                frame["perturbed_score"] - frame["baseline_score"]
            )
            frame["baseline_centered_score"] = frame.groupby("group_id")[
                "baseline_score"
            ].transform(lambda values: values - values.mean())
            frame["perturbed_centered_score"] = frame.groupby("group_id")[
                "perturbed_score"
            ].transform(lambda values: values - values.mean())
            frame["centered_score_delta"] = (
                frame["perturbed_centered_score"]
                - frame["baseline_centered_score"]
            )
            frame["protein_source_index"] = variant[
                "protein_source_index"
            ].to_numpy()
            frame["molecule_source_index"] = variant[
                "molecule_source_index"
            ].to_numpy()
            frame["protein_source_group_id"] = variant[
                "protein_source_group_id"
            ].to_numpy()
            frame["molecule_source_group_id"] = variant[
                "molecule_source_group_id"
            ].to_numpy()
            frame["molecule_source_compound_id"] = variant[
                "molecule_source_compound_id"
            ].to_numpy()
            frames.append(frame)
            baseline_target_spearman = _weighted_group_spearman(
                frame,
                "baseline_score",
                "pchembl_value",
            )
            perturbed_target_spearman = _weighted_group_spearman(
                frame,
                "perturbed_score",
                "pchembl_value",
            )
            repeat_records.append(
                {
                    "split": str(split),
                    "repeat": repeat,
                    "perturbation": perturbation,
                    "raw_score_mae": float(frame["raw_score_delta"].abs().mean()),
                    "centered_score_mae": float(
                        frame["centered_score_delta"].abs().mean()
                    ),
                    "baseline_perturbed_pearson": _safe_correlation(
                        frame["baseline_score"],
                        frame["perturbed_score"],
                        method="pearson",
                    ),
                    "within_assay_rank_stability": _weighted_group_spearman(
                        frame,
                        "baseline_score",
                        "perturbed_score",
                    ),
                    "baseline_target_spearman": baseline_target_spearman,
                    "perturbed_target_spearman": perturbed_target_spearman,
                    "target_spearman_delta": (
                        perturbed_target_spearman - baseline_target_spearman
                    ),
                }
            )

    sensitivity_rows = pd.concat(frames, ignore_index=True)
    by_repeat = pd.DataFrame.from_records(repeat_records)
    aggregate = {}
    for perturbation, group in by_repeat.groupby("perturbation", sort=True):
        aggregate[str(perturbation)] = {
            column: {
                "mean": float(group[column].mean()),
                "std": float(group[column].std(ddof=0)),
            }
            for column in (
                "raw_score_mae",
                "centered_score_mae",
                "baseline_perturbed_pearson",
                "within_assay_rank_stability",
                "baseline_target_spearman",
                "perturbed_target_spearman",
                "target_spearman_delta",
            )
        }
    protein_centered = aggregate["protein_shuffled"]["centered_score_mae"]["mean"]
    ligand_centered = aggregate["ligand_shuffled"]["centered_score_mae"]["mean"]
    summary = {
        "split": str(split),
        "num_examples": len(dataset),
        "num_assays": len(unique_groups),
        "num_shuffles": int(num_shuffles),
        "seed": int(seed),
        "perturbations": aggregate,
        "protein_to_ligand_centered_sensitivity_ratio": float(
            protein_centered / ligand_centered
        )
        if ligand_centered > 0.0
        else float("nan"),
    }
    return sensitivity_rows, by_repeat, summary


def write_sensitivity_analysis(
    output_dir: str,
    *,
    split: str,
    sensitivity_rows: pd.DataFrame,
    by_repeat: pd.DataFrame,
    summary: Mapping[str, Any],
) -> dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "predictions": os.path.abspath(
            os.path.join(output_dir, f"{split}_input_sensitivity.parquet")
        ),
        "by_repeat": os.path.abspath(
            os.path.join(output_dir, f"{split}_input_sensitivity_by_repeat.csv")
        ),
        "summary": os.path.abspath(
            os.path.join(output_dir, f"{split}_input_sensitivity_summary.json")
        ),
    }
    sensitivity_rows.to_parquet(paths["predictions"], index=False)
    by_repeat.to_csv(paths["by_repeat"], index=False)
    with open(paths["summary"], "w", encoding="utf-8") as handle:
        json.dump(dict(summary), handle, indent=2, sort_keys=True)
    return paths


def _distribution_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p01": float("nan"),
            "p10": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }
    quantiles = np.quantile(array, [0.01, 0.10, 0.50, 0.90, 0.99])
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p01": float(quantiles[0]),
        "p10": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p90": float(quantiles[3]),
        "p99": float(quantiles[4]),
        "max": float(array.max()),
    }


def _mean_within_assay_std(frame: pd.DataFrame, column: str) -> float:
    values = frame.groupby("group_id", sort=True)[column].agg(
        lambda group: float(np.asarray(group, dtype=np.float64).std())
    )
    return float(values.mean()) if len(values) else float("nan")


def _safe_ratio(numerator: float, denominator: float) -> float:
    return (
        float(numerator / denominator)
        if math.isfinite(float(numerator))
        and math.isfinite(float(denominator))
        and float(denominator) != 0.0
        else float("nan")
    )


def run_fusion_cosine_sensitivity(
    model,
    dataset,
    *,
    split: str,
    batch_size: int,
    device: torch.device,
    num_shuffles: int = 3,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compare pre- and post-fusion cosine behavior under input shuffles."""
    if num_shuffles <= 0:
        raise ValueError("num_shuffles must be > 0")
    baseline = score_fusion_cosines(
        model,
        dataset,
        batch_size=batch_size,
        device=device,
    )
    group_ids = [str(value) for value in dataset["group_id"]]
    unique_groups = sorted(set(group_ids))
    if len(unique_groups) < 2:
        raise ValueError("Protein shuffling requires at least two assays")
    first_group_index = {}
    for index, group_id in enumerate(group_ids):
        first_group_index.setdefault(group_id, index)

    generator = np.random.default_rng(int(seed))
    frames = []
    repeat_records = []
    distribution_records = []
    stage_columns = {
        "pre_fusion": "pre_fusion_cosine",
        "post_fusion": "post_fusion_cosine",
    }
    for stage, column in stage_columns.items():
        distribution_records.append(
            {
                "split": str(split),
                "repeat": -1,
                "stage": stage,
                "pairing": "correct",
                **_distribution_summary(baseline[column]),
            }
        )

    for repeat in range(int(num_shuffles)):
        group_sources = _cyclic_derangement(len(unique_groups), generator)
        source_group_by_group = {
            unique_groups[index]: unique_groups[int(group_sources[index])]
            for index in range(len(unique_groups))
        }
        protein_indices = np.asarray(
            [first_group_index[source_group_by_group[group_id]] for group_id in group_ids],
            dtype=np.int64,
        )
        molecule_indices = _cyclic_derangement(len(dataset), generator)
        variants = {
            "protein_shuffled": score_fusion_cosines(
                model,
                dataset,
                batch_size=batch_size,
                device=device,
                protein_source_indices=protein_indices,
            ),
            "ligand_shuffled": score_fusion_cosines(
                model,
                dataset,
                batch_size=batch_size,
                device=device,
                molecule_source_indices=molecule_indices,
            ),
        }
        for perturbation, variant in variants.items():
            frame = baseline[list(PREDICTION_COLUMNS)].copy()
            frame.insert(0, "split", str(split))
            frame.insert(1, "repeat", repeat)
            frame.insert(2, "perturbation", perturbation)
            for source_column in (
                "protein_source_index",
                "molecule_source_index",
                "protein_source_group_id",
                "molecule_source_group_id",
                "molecule_source_compound_id",
            ):
                frame[source_column] = variant[source_column].to_numpy()

            for stage, source_column in stage_columns.items():
                baseline_column = f"baseline_{stage}_cosine"
                perturbed_column = f"perturbed_{stage}_cosine"
                delta_column = f"{stage}_cosine_delta"
                baseline_centered_column = f"baseline_{stage}_centered_cosine"
                perturbed_centered_column = f"perturbed_{stage}_centered_cosine"
                centered_delta_column = f"{stage}_centered_cosine_delta"
                frame[baseline_column] = baseline[source_column].to_numpy()
                frame[perturbed_column] = variant[source_column].to_numpy()
                frame[delta_column] = frame[perturbed_column] - frame[baseline_column]
                frame[baseline_centered_column] = frame.groupby("group_id")[
                    baseline_column
                ].transform(lambda values: values - values.mean())
                frame[perturbed_centered_column] = frame.groupby("group_id")[
                    perturbed_column
                ].transform(lambda values: values - values.mean())
                frame[centered_delta_column] = (
                    frame[perturbed_centered_column]
                    - frame[baseline_centered_column]
                )

                baseline_target_spearman = _weighted_group_spearman(
                    frame,
                    baseline_column,
                    "pchembl_value",
                )
                perturbed_target_spearman = _weighted_group_spearman(
                    frame,
                    perturbed_column,
                    "pchembl_value",
                )
                repeat_records.append(
                    {
                        "split": str(split),
                        "repeat": repeat,
                        "stage": stage,
                        "perturbation": perturbation,
                        "cosine_mae": float(frame[delta_column].abs().mean()),
                        "centered_cosine_mae": float(
                            frame[centered_delta_column].abs().mean()
                        ),
                        "baseline_perturbed_pearson": _safe_correlation(
                            frame[baseline_column],
                            frame[perturbed_column],
                            method="pearson",
                        ),
                        "within_assay_rank_stability": _weighted_group_spearman(
                            frame,
                            baseline_column,
                            perturbed_column,
                        ),
                        "baseline_target_spearman": baseline_target_spearman,
                        "perturbed_target_spearman": perturbed_target_spearman,
                        "target_spearman_delta": (
                            perturbed_target_spearman - baseline_target_spearman
                        ),
                        "baseline_within_assay_cosine_std_mean": (
                            _mean_within_assay_std(frame, baseline_column)
                        ),
                        "perturbed_within_assay_cosine_std_mean": (
                            _mean_within_assay_std(frame, perturbed_column)
                        ),
                    }
                )
                distribution_records.append(
                    {
                        "split": str(split),
                        "repeat": repeat,
                        "stage": stage,
                        "pairing": perturbation,
                        **_distribution_summary(frame[perturbed_column]),
                    }
                )
            frames.append(frame)

    sensitivity_rows = pd.concat(frames, ignore_index=True)
    by_repeat = pd.DataFrame.from_records(repeat_records)
    distributions = pd.DataFrame.from_records(distribution_records)
    metric_columns = (
        "cosine_mae",
        "centered_cosine_mae",
        "baseline_perturbed_pearson",
        "within_assay_rank_stability",
        "baseline_target_spearman",
        "perturbed_target_spearman",
        "target_spearman_delta",
        "baseline_within_assay_cosine_std_mean",
        "perturbed_within_assay_cosine_std_mean",
    )
    stages: dict[str, Any] = {}
    for stage, source_column in stage_columns.items():
        stage_repeats = by_repeat[by_repeat["stage"] == stage]
        perturbations = {}
        for perturbation, group in stage_repeats.groupby("perturbation", sort=True):
            perturbation_rows = sensitivity_rows[
                sensitivity_rows["perturbation"] == perturbation
            ]
            perturbations[str(perturbation)] = {
                column: {
                    "mean": float(group[column].mean()),
                    "std": float(group[column].std(ddof=0)),
                }
                for column in metric_columns
            }
            perturbations[str(perturbation)]["distribution"] = _distribution_summary(
                perturbation_rows[f"perturbed_{stage}_cosine"]
            )

        protein_centered = perturbations["protein_shuffled"][
            "centered_cosine_mae"
        ]["mean"]
        ligand_centered = perturbations["ligand_shuffled"][
            "centered_cosine_mae"
        ]["mean"]
        stages[stage] = {
            "correct": {
                "distribution": _distribution_summary(baseline[source_column]),
                "target_spearman": _weighted_group_spearman(
                    baseline,
                    source_column,
                    "pchembl_value",
                ),
                "within_assay_cosine_std_mean": _mean_within_assay_std(
                    baseline,
                    source_column,
                ),
            },
            "perturbations": perturbations,
            "protein_to_ligand_centered_sensitivity_ratio": _safe_ratio(
                protein_centered,
                ligand_centered,
            ),
        }

    pre_correct = stages["pre_fusion"]["correct"]
    post_correct = stages["post_fusion"]["correct"]
    pre_perturbations = stages["pre_fusion"]["perturbations"]
    post_perturbations = stages["post_fusion"]["perturbations"]
    fusion_effect = {
        "correct_pre_post_pearson": _safe_correlation(
            baseline["pre_fusion_cosine"],
            baseline["post_fusion_cosine"],
            method="pearson",
        ),
        "correct_pre_post_within_assay_rank_stability": _weighted_group_spearman(
            baseline,
            "pre_fusion_cosine",
            "post_fusion_cosine",
        ),
        "correct_target_spearman_delta": float(
            post_correct["target_spearman"] - pre_correct["target_spearman"]
        ),
        "correct_cosine_std_ratio_post_to_pre": _safe_ratio(
            post_correct["distribution"]["std"],
            pre_correct["distribution"]["std"],
        ),
        "correct_within_assay_std_ratio_post_to_pre": _safe_ratio(
            post_correct["within_assay_cosine_std_mean"],
            pre_correct["within_assay_cosine_std_mean"],
        ),
        "protein_shuffle_rank_stability_change": float(
            post_perturbations["protein_shuffled"][
                "within_assay_rank_stability"
            ]["mean"]
            - pre_perturbations["protein_shuffled"][
                "within_assay_rank_stability"
            ]["mean"]
        ),
        "ligand_shuffle_rank_stability_change": float(
            post_perturbations["ligand_shuffled"][
                "within_assay_rank_stability"
            ]["mean"]
            - pre_perturbations["ligand_shuffled"][
                "within_assay_rank_stability"
            ]["mean"]
        ),
        "protein_centered_sensitivity_ratio_post_to_pre": _safe_ratio(
            post_perturbations["protein_shuffled"]["centered_cosine_mae"][
                "mean"
            ],
            pre_perturbations["protein_shuffled"]["centered_cosine_mae"][
                "mean"
            ],
        ),
        "ligand_centered_sensitivity_ratio_post_to_pre": _safe_ratio(
            post_perturbations["ligand_shuffled"]["centered_cosine_mae"][
                "mean"
            ],
            pre_perturbations["ligand_shuffled"]["centered_cosine_mae"][
                "mean"
            ],
        ),
    }
    summary = {
        "split": str(split),
        "num_examples": len(dataset),
        "num_assays": len(unique_groups),
        "num_shuffles": int(num_shuffles),
        "seed": int(seed),
        "pooling_type": str(
            getattr(getattr(model, "config", None), "pooling_type", "mean")
        ),
        "fusion_residual": bool(
            getattr(getattr(model, "config", None), "fusion_residual", False)
        ),
        "stages": stages,
        "fusion_effect": fusion_effect,
    }
    return sensitivity_rows, by_repeat, distributions, summary


def write_fusion_cosine_analysis(
    output_dir: str,
    *,
    split: str,
    sensitivity_rows: pd.DataFrame,
    by_repeat: pd.DataFrame,
    distributions: pd.DataFrame,
    summary: Mapping[str, Any],
) -> dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "predictions": os.path.abspath(
            os.path.join(output_dir, f"{split}_fusion_cosine_sensitivity.parquet")
        ),
        "by_repeat": os.path.abspath(
            os.path.join(output_dir, f"{split}_fusion_cosine_by_repeat.csv")
        ),
        "distributions": os.path.abspath(
            os.path.join(output_dir, f"{split}_fusion_cosine_distributions.csv")
        ),
        "summary": os.path.abspath(
            os.path.join(output_dir, f"{split}_fusion_cosine_summary.json")
        ),
    }
    sensitivity_rows.to_parquet(paths["predictions"], index=False)
    by_repeat.to_csv(paths["by_repeat"], index=False)
    distributions.to_csv(paths["distributions"], index=False)
    with open(paths["summary"], "w", encoding="utf-8") as handle:
        json.dump(dict(summary), handle, indent=2, sort_keys=True)
    return paths


@dataclass
class _ActivationMoments:
    count: int = 0
    total: float = 0.0
    total_square: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    near_zero: int = 0
    nonfinite: int = 0

    def update(self, tensor: torch.Tensor) -> None:
        values = tensor.detach().float()
        finite = torch.isfinite(values)
        self.nonfinite += int((~finite).sum().item())
        values = values[finite]
        if values.numel() == 0:
            return
        self.count += int(values.numel())
        self.total += float(values.sum().item())
        self.total_square += float(values.square().sum().item())
        self.minimum = min(self.minimum, float(values.min().item()))
        self.maximum = max(self.maximum, float(values.max().item()))
        self.near_zero += int((values.abs() < 1e-6).sum().item())


class ActivationCollector:
    """Streaming activation checks for MLP heads when they are configured."""

    def __init__(self, model):
        self.model = model
        self.moments: dict[str, _ActivationMoments] = {}
        self.handles = []

    def __enter__(self):
        for head_name in ("ranking_head", "classification_head"):
            head = getattr(self.model, head_name, None)
            if not isinstance(head, torch.nn.Module):
                continue
            for module_name, module in head.named_modules():
                if not isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)):
                    continue
                name = f"{head_name}.{module_name}"
                self.moments[name] = _ActivationMoments()

                def _hook(_module, _inputs, output, *, key=name):
                    self.moments[key].update(output)

                self.handles.append(module.register_forward_hook(_hook))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def to_frame(self) -> pd.DataFrame:
        records = []
        for name, moments in sorted(self.moments.items()):
            mean = moments.total / moments.count if moments.count else float("nan")
            variance = (
                max(moments.total_square / moments.count - mean * mean, 0.0)
                if moments.count
                else float("nan")
            )
            records.append(
                {
                    "module": name,
                    "count": moments.count,
                    "mean": mean,
                    "std": math.sqrt(variance),
                    "min": moments.minimum if moments.count else float("nan"),
                    "max": moments.maximum if moments.count else float("nan"),
                    "near_zero_fraction": (
                        moments.near_zero / moments.count
                        if moments.count
                        else float("nan")
                    ),
                    "nonfinite_count": moments.nonfinite,
                }
            )
        return pd.DataFrame.from_records(records)


def capture_head_activations(
    model,
    dataset,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    with ActivationCollector(model) as collector:
        predictions = score_dataset_pairs(
            model,
            dataset,
            batch_size=batch_size,
            device=device,
        )
    return predictions, collector.to_frame()
