from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data import RewardPairCollator, RewardPairDataset


def compute_classification_metrics(
    probabilities: Sequence[float],
    labels: Sequence[float],
    *,
    threshold: float = 0.5,
) -> Dict[str, float]:
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    predicted_labels = (probabilities_array >= threshold).astype(np.int64)
    return {
        "eval_mcc": float(matthews_corrcoef(labels_array, predicted_labels)),
        "eval_f1": float(f1_score(labels_array, predicted_labels)),
        "eval_roc_auc": float(roc_auc_score(labels_array, probabilities_array)),
        "eval_precision": float(precision_score(labels_array, predicted_labels)),
        "eval_recall": float(recall_score(labels_array, predicted_labels)),
        "eval_accuracy": float(accuracy_score(labels_array, predicted_labels)),
    }


def compute_pairwise_accuracy(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
) -> float:
    positive_array = np.asarray(positive_scores, dtype=np.float64)
    negative_array = np.asarray(negative_scores, dtype=np.float64)
    return float(np.mean(positive_array > negative_array))


def compute_groupwise_spearman(
    *,
    group_ids: Sequence[str],
    ranking_scores: Sequence[float],
    pchembl_values: Sequence[float],
    min_group_size: int = 3,
) -> Dict[str, float]:
    grouped_scores: Dict[str, list[float]] = defaultdict(list)
    grouped_pchembl: Dict[str, list[float]] = defaultdict(list)
    for group_id, score, pchembl in zip(group_ids, ranking_scores, pchembl_values):
        grouped_scores[str(group_id)].append(float(score))
        grouped_pchembl[str(group_id)].append(float(pchembl))

    correlations: list[float] = []
    for group_id in sorted(grouped_scores):
        scores = grouped_scores[group_id]
        pchembls = grouped_pchembl[group_id]
        if len(scores) < min_group_size:
            continue
        if len(set(pchembls)) == 1:
            continue
        result = spearmanr(scores, pchembls)
        correlations.append(float(getattr(result, "statistic", result[0])))

    if not correlations:
        return {
            "eval_spearman": float("nan"),
            "eval_spearman_num_groups": 0.0,
        }
    return {
        "eval_spearman": float(np.mean(correlations)),
        "eval_spearman_num_groups": float(len(correlations)),
    }


def _collate_example_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
    return {
        "protein_input_ids": torch.tensor(
            [row["protein_input_ids"] for row in rows],
            dtype=torch.long,
        ),
        "protein_attention_mask": torch.tensor(
            [row["protein_attention_mask"] for row in rows],
            dtype=torch.long,
        ),
        "molecule_input_ids": torch.tensor(
            [row["molecule_input_ids"] for row in rows],
            dtype=torch.long,
        ),
        "molecule_attention_mask": torch.tensor(
            [row["molecule_attention_mask"] for row in rows],
            dtype=torch.long,
        ),
    }


def _score_example_dataset(
    trainer,
    model,
    example_dataset,
    *,
    batch_size: int,
) -> Dict[str, Sequence[float] | Sequence[str]]:
    ranking_scores: list[float] = []
    activity_probabilities: list[float] = []
    labels: list[float] = []
    group_ids: list[str] = []
    pchembl_values: list[float] = []

    for start in range(0, len(example_dataset), batch_size):
        rows = [dict(example_dataset[index]) for index in range(start, min(start + batch_size, len(example_dataset)))]
        model_inputs = trainer._prepare_inputs(_collate_example_rows(rows))
        with torch.no_grad():
            outputs = model(return_dict=True, **model_inputs)
        ranking_scores.extend(outputs.ranking_score.detach().cpu().tolist())
        activity_probabilities.extend(outputs.activity_probability.detach().cpu().tolist())
        labels.extend(float(row["binary_label"]) for row in rows)
        group_ids.extend(str(row["group_id"]) for row in rows)
        pchembl_values.extend(float(row["pchembl_value"]) for row in rows)

    return {
        "ranking_scores": ranking_scores,
        "activity_probabilities": activity_probabilities,
        "labels": labels,
        "group_ids": group_ids,
        "pchembl_values": pchembl_values,
    }


def _score_pair_dataset(
    trainer,
    model,
    eval_dataset: RewardPairDataset,
    *,
    batch_size: int,
) -> Dict[str, float]:
    collator = RewardPairCollator()
    positive_scores: list[float] = []
    negative_scores: list[float] = []

    for start in range(0, len(eval_dataset), batch_size):
        features = [
            eval_dataset[index]
            for index in range(start, min(start + batch_size, len(eval_dataset)))
        ]
        batch = collator(features)
        model_inputs = trainer._prepare_inputs(trainer._build_model_inputs(batch))
        with torch.no_grad():
            outputs = model(**model_inputs)
        ranking_score = outputs.ranking_score.detach().cpu()
        positive_indices = batch["positive_indices"]
        negative_indices = batch["negative_indices"]
        positive_scores.extend(ranking_score[positive_indices].tolist())
        negative_scores.extend(ranking_score[negative_indices].tolist())

    return {
        "eval_pairwise_accuracy": compute_pairwise_accuracy(positive_scores, negative_scores),
    }


def compute_reward_model_eval_metrics(
    trainer,
    model,
    eval_dataset: RewardPairDataset,
) -> Dict[str, float]:
    batch_size = int(trainer.args.per_device_eval_batch_size)
    example_metrics_source = _score_example_dataset(
        trainer,
        model,
        eval_dataset.example_dataset,
        batch_size=batch_size,
    )
    metrics = compute_classification_metrics(
        example_metrics_source["activity_probabilities"],
        example_metrics_source["labels"],
        threshold=0.5,
    )
    metrics.update(
        compute_groupwise_spearman(
            group_ids=example_metrics_source["group_ids"],
            ranking_scores=example_metrics_source["ranking_scores"],
            pchembl_values=example_metrics_source["pchembl_values"],
            min_group_size=3,
        )
    )
    metrics.update(
        _score_pair_dataset(
            trainer,
            model,
            eval_dataset,
            batch_size=batch_size,
        )
    )
    return metrics
