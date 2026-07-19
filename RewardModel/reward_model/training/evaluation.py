from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
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
from ..model.losses import ligunity_listwise_loss

ASSAY_SPEARMAN_LOG_FILENAME = "eval_assay_spearman.jsonl"


def compute_classification_metrics(
    probabilities: Sequence[float],
    labels: Sequence[float],
    *,
    threshold: float = 0.5,
) -> Dict[str, float]:
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    predicted_labels = (probabilities_array >= threshold).astype(np.int64)
    metrics = {
        "eval_mcc": float(matthews_corrcoef(labels_array, predicted_labels)),
        "eval_f1": float(f1_score(labels_array, predicted_labels, zero_division=0)),
        "eval_precision": float(
            precision_score(labels_array, predicted_labels, zero_division=0)
        ),
        "eval_recall": float(recall_score(labels_array, predicted_labels, zero_division=0)),
        "eval_accuracy": float(accuracy_score(labels_array, predicted_labels)),
    }
    metrics["eval_roc_auc"] = (
        float(roc_auc_score(labels_array, probabilities_array))
        if len(np.unique(labels_array)) >= 2
        else float("nan")
    )
    return metrics


def compute_joint_evaluation_metrics(
    *,
    activity_logits: torch.Tensor,
    ranking_scores: torch.Tensor,
    activity_labels: torch.Tensor,
    pchembl_values: torch.Tensor,
    ranking_group_ids: torch.Tensor,
    group_id_names: Sequence[str],
    classification_loss_weight: float,
    ranking_loss_weight: float,
    bce_pos_weight: float,
    ranking_temperature: float,
    ranking_min_pchembl_span: float,
) -> tuple[Dict[str, float], list[Dict[str, float | int | str]]]:
    """Compute deterministic full-dataset losses and metrics from one scoring pass."""
    logits = activity_logits.reshape(-1).detach().cpu().float()
    scores = ranking_scores.reshape(-1).detach().cpu().float()
    labels = activity_labels.reshape(-1).detach().cpu().float()
    pchembl = pchembl_values.reshape(-1).detach().cpu().float()
    group_indices = ranking_group_ids.reshape(-1).detach().cpu().long()
    if not (
        logits.numel()
        == scores.numel()
        == labels.numel()
        == pchembl.numel()
        == group_indices.numel()
    ):
        raise ValueError("evaluation tensors must contain the same number of observations")

    pos_weight = torch.tensor(float(bce_pos_weight), dtype=logits.dtype)
    classification_loss = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=pos_weight,
    )
    ranking_loss = ligunity_listwise_loss(
        scores,
        pchembl,
        group_indices,
        temperature=ranking_temperature,
        min_pchembl_span=ranking_min_pchembl_span,
    )
    total_loss = (
        float(classification_loss_weight) * classification_loss
        + float(ranking_loss_weight) * ranking_loss
    )

    probabilities = torch.sigmoid(logits).numpy()
    metrics = compute_classification_metrics(probabilities, labels.numpy())
    string_group_ids = [group_id_names[int(index)] for index in group_indices.tolist()]
    spearman_metrics, assay_records = _compute_groupwise_spearman_with_records(
        group_ids=string_group_ids,
        ranking_scores=scores.tolist(),
        pchembl_values=pchembl.tolist(),
        min_group_size=2,
        min_pchembl_span=ranking_min_pchembl_span,
    )
    metrics.update(spearman_metrics)
    metrics.update(
        {
            "eval_loss": float(total_loss.item()),
            "eval_total_loss": float(total_loss.item()),
            "eval_classification_loss": float(classification_loss.item()),
            "eval_ranking_loss": float(ranking_loss.item()),
            "eval_num_examples": float(logits.numel()),
            "eval_num_ranking_groups": float(len(assay_records)),
        }
    )
    return metrics, assay_records


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
    metrics, _ = _compute_groupwise_spearman_with_records(
        group_ids=group_ids,
        ranking_scores=ranking_scores,
        pchembl_values=pchembl_values,
        min_group_size=min_group_size,
    )
    return metrics


def _split_group_id(group_id: str) -> tuple[str, str]:
    target_chembl_id, assay_id = group_id.split("__", 1)
    return target_chembl_id, assay_id


def _compute_groupwise_spearman_with_records(
    *,
    group_ids: Sequence[str],
    ranking_scores: Sequence[float],
    pchembl_values: Sequence[float],
    min_group_size: int = 3,
    min_pchembl_span: float = 0.0,
) -> tuple[Dict[str, float], list[Dict[str, float | int | str]]]:
    grouped_scores: Dict[str, list[float]] = defaultdict(list)
    grouped_pchembl: Dict[str, list[float]] = defaultdict(list)
    for group_id, score, pchembl in zip(group_ids, ranking_scores, pchembl_values):
        grouped_scores[str(group_id)].append(float(score))
        grouped_pchembl[str(group_id)].append(float(pchembl))

    assay_records: list[Dict[str, float | int | str]] = []
    for group_id in sorted(grouped_scores):
        scores = grouped_scores[group_id]
        pchembls = grouped_pchembl[group_id]
        if len(scores) < min_group_size:
            continue
        if len(set(pchembls)) == 1:
            continue
        if max(pchembls) - min(pchembls) < min_pchembl_span:
            continue
        result = spearmanr(scores, pchembls)
        target_chembl_id, assay_id = _split_group_id(group_id)
        assay_records.append(
            {
                "group_id": group_id,
                "target_chembl_id": target_chembl_id,
                "assay_id": assay_id,
                "num_examples": len(scores),
                "spearman": float(getattr(result, "statistic", result[0])),
            }
        )

    if not assay_records:
        return (
            {
                "eval_spearman": float("nan"),
                "eval_spearman_num_groups": 0.0,
            },
            assay_records,
        )

    weighted_sum = sum(
        float(record["spearman"]) * int(record["num_examples"])
        for record in assay_records
    )
    total_weight = sum(int(record["num_examples"]) for record in assay_records)
    return (
        {
            "eval_spearman": float(weighted_sum / total_weight),
            "eval_spearman_num_groups": float(len(assay_records)),
        },
        assay_records,
    )


def _append_assay_spearman_log(
    trainer,
    metrics: Mapping[str, float],
    assay_records: Sequence[Mapping[str, float | int | str]],
    *,
    metric_key_prefix: str,
) -> None:
    if not trainer.is_world_process_zero():
        return

    os.makedirs(trainer.args.output_dir, exist_ok=True)
    filename = (
        ASSAY_SPEARMAN_LOG_FILENAME
        if metric_key_prefix == "eval"
        else f"{metric_key_prefix}_assay_spearman.jsonl"
    )
    path = os.path.join(trainer.args.output_dir, filename)
    record = {
        "global_step": int(trainer.state.global_step),
        "epoch": trainer.state.epoch,
        "weighted_spearman": metrics[f"{metric_key_prefix}_spearman"],
        "num_eligible_groups": metrics[f"{metric_key_prefix}_spearman_num_groups"],
        "assays": list(assay_records),
    }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _with_metric_prefix(metrics: Mapping[str, float], metric_key_prefix: str) -> Dict[str, float]:
    if metric_key_prefix == "eval":
        return dict(metrics)
    return {
        key.replace("eval_", f"{metric_key_prefix}_", 1): value
        for key, value in metrics.items()
    }


def _collate_example_rows(
    rows: Sequence[Mapping[str, Any]],
    collator: RewardPairCollator,
) -> Dict[str, torch.Tensor]:
    return collator.collate_example_tokens(rows)


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
    collator = (
        trainer.data_collator
        if isinstance(trainer.data_collator, RewardPairCollator)
        else RewardPairCollator()
    )

    for start in range(0, len(example_dataset), batch_size):
        rows = [dict(example_dataset[index]) for index in range(start, min(start + batch_size, len(example_dataset)))]
        model_inputs = trainer._prepare_inputs(_collate_example_rows(rows, collator))
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
    collator = (
        trainer.data_collator
        if isinstance(trainer.data_collator, RewardPairCollator)
        else RewardPairCollator()
    )
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
    *,
    metric_key_prefix: str = "eval",
    pairwise_accuracy: float | None = None,
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
    spearman_metrics, assay_records = _compute_groupwise_spearman_with_records(
        group_ids=example_metrics_source["group_ids"],
        ranking_scores=example_metrics_source["ranking_scores"],
        pchembl_values=example_metrics_source["pchembl_values"],
        min_group_size=3,
    )
    metrics.update(spearman_metrics)
    if pairwise_accuracy is None:
        metrics.update(
            _score_pair_dataset(
                trainer,
                model,
                eval_dataset,
                batch_size=batch_size,
            )
        )
    else:
        metrics["eval_pairwise_accuracy"] = float(pairwise_accuracy)
    metrics = _with_metric_prefix(metrics, metric_key_prefix)
    _append_assay_spearman_log(
        trainer,
        metrics,
        assay_records,
        metric_key_prefix=metric_key_prefix,
    )
    return metrics
