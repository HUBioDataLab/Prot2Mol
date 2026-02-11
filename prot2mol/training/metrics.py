from typing import Optional

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ..chem.utils import metrics_calculation, canonicalize_smiles_list, decode_selfies_list


def preprocess_logits_for_metrics(logits, labels, logger=None):
    """
    Convert model logits into token-id predictions expected by HF metric pipeline.
    """
    try:
        if isinstance(logits, dict) and "logits" in logits:
            raw_logits = logits["logits"]
        elif isinstance(logits, (tuple, list)) and len(logits) > 1:
            raw_logits = logits[1] if isinstance(logits[1], torch.Tensor) and len(logits[1].shape) >= 2 else logits[0]
        elif isinstance(logits, (tuple, list)) and len(logits) > 0:
            raw_logits = logits[0] if isinstance(logits[0], torch.Tensor) else logits
        elif isinstance(logits, torch.Tensor):
            raw_logits = logits
        else:
            if logger is not None:
                logger.error("Unexpected logits format: %s", type(logits))
            return None

        pred_ids = torch.argmax(raw_logits, dim=-1)
        if labels is not None and labels.shape != pred_ids.shape:
            if logger is not None:
                logger.warning("Shape mismatch: labels %s vs pred_ids %s", labels.shape, pred_ids.shape)
            if pred_ids.numel() == labels.numel():
                pred_ids = pred_ids.reshape(labels.shape)
            else:
                return None
        return pred_ids
    except Exception as exc:
        if logger is not None:
            logger.error("Error preprocessing logits: %s", exc, exc_info=True)
        return None


def compute_lm_metrics(
    predictions,
    labels,
    mol_tokenizer,
    eval_reference_smiles,
    train_smiles_list,
    training_vec,
    global_rank: int = 0,
    logger=None,
):
    """Compute language-model generation metrics."""
    try:
        if predictions is None:
            if logger is not None:
                logger.error("LM predictions are None, skipping LM metrics")
            return {}

        valid_labels = (labels != -100).sum()
        if valid_labels == 0:
            if logger is not None:
                logger.info("Rank %s: No valid LM labels (all negative samples)", global_rank)
            return {}

        if predictions.shape != labels.shape:
            if logger is not None:
                logger.error("Shape mismatch: predictions %s vs labels %s", predictions.shape, labels.shape)
            if predictions.size == labels.size:
                predictions = predictions.reshape(labels.shape)
            else:
                return {}

        labels_for_decoding = np.where(labels != -100, labels, mol_tokenizer.pad_token_id)
        predictions_for_decoding = np.where(labels != -100, predictions, mol_tokenizer.pad_token_id)

        decoded_preds = mol_tokenizer.batch_decode(
            predictions_for_decoding,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        decoded_labels = mol_tokenizer.batch_decode(
            labels_for_decoding,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        reference_smiles = eval_reference_smiles
        if not reference_smiles:
            fallback_decoded = decode_selfies_list(decoded_labels)
            reference_smiles = canonicalize_smiles_list(fallback_decoded, drop_invalid=True)

        lm_metrics = metrics_calculation(
            predictions=decoded_preds,
            references=reference_smiles,
            train_data=train_smiles_list,
            train_vec=training_vec,
            training=True,
        )
        return {f"lm_{k}": v for k, v in lm_metrics.items()}
    except Exception as exc:
        if logger is not None:
            logger.error("Error computing LM metrics: %s", exc, exc_info=True)
        return {}


def _pairwise_accuracy(y_pred, y_true, group_ids, max_pairs_per_group=200):
    """Estimate pairwise ranking accuracy within groups."""
    correct = 0
    total = 0
    for g in np.unique(group_ids):
        idx = np.where(group_ids == g)[0]
        if idx.size < 2:
            continue
        np.random.shuffle(idx)
        half = idx.size // 2
        if half == 0:
            continue
        a = idx[:half]
        b = idx[half:half * 2]
        if max_pairs_per_group and a.size > max_pairs_per_group:
            a = a[:max_pairs_per_group]
            b = b[:max_pairs_per_group]
        dy_true = y_true[a] - y_true[b]
        dy_pred = y_pred[a] - y_pred[b]
        non_zero = dy_true != 0
        if not np.any(non_zero):
            continue
        dy_true = dy_true[non_zero]
        dy_pred = dy_pred[non_zero]
        correct += np.sum(np.sign(dy_true) == np.sign(dy_pred))
        total += dy_true.size
    if total == 0:
        return None
    return float(correct / total)


def _group_spearman(y_pred, y_true, group_ids, min_group_size=3):
    """Compute macro-average Spearman correlation within groups."""
    try:
        from scipy.stats import spearmanr
    except Exception:
        return None

    correlations = []
    for g in np.unique(group_ids):
        idx = np.where(group_ids == g)[0]
        if idx.size < min_group_size:
            continue
        corr, _ = spearmanr(y_true[idx], y_pred[idx])
        if np.isfinite(corr):
            correlations.append(corr)
    if not correlations:
        return None
    return float(np.mean(correlations))


def compute_pchembl_metrics(
    pchembl_predictions,
    pchembl_targets,
    pchembl_mean: float,
    pchembl_std: float,
    group_ids: Optional[np.ndarray] = None,
    logger=None,
):
    """Compute regression/ranking metrics for pChEMBL predictions."""
    try:
        if pchembl_predictions is None or pchembl_targets is None:
            if logger is not None:
                logger.warning("pChEMBL predictions or targets are None")
            return {}

        if isinstance(pchembl_predictions, torch.Tensor):
            pchembl_predictions = pchembl_predictions.detach().cpu().numpy()
        if isinstance(pchembl_targets, torch.Tensor):
            pchembl_targets = pchembl_targets.detach().cpu().numpy()

        valid_mask = ~(np.isnan(pchembl_predictions) | np.isnan(pchembl_targets))
        if not valid_mask.any():
            if logger is not None:
                logger.warning("No valid pChEMBL predictions/targets found (all NaN)")
            return {}

        valid_preds = pchembl_predictions[valid_mask]
        valid_true = pchembl_targets[valid_mask]
        pred_raw = valid_preds * (pchembl_std + 1e-8) + pchembl_mean
        true_raw = valid_true * (pchembl_std + 1e-8) + pchembl_mean

        mse = mean_squared_error(valid_true, valid_preds)
        mae = mean_absolute_error(valid_true, valid_preds)
        rmse = np.sqrt(mse)
        mse_raw = mean_squared_error(true_raw, pred_raw)
        mae_raw = mean_absolute_error(true_raw, pred_raw)
        rmse_raw = np.sqrt(mse_raw)
        try:
            r2 = r2_score(valid_true, valid_preds)
        except ValueError:
            r2 = float("nan")

        metrics = {
            "pchembl_mse": mse,
            "pchembl_mae": mae,
            "pchembl_rmse": rmse,
            "pchembl_r2": r2,
            "pchembl_valid_count": len(valid_preds),
            "pchembl_mean_pred": np.mean(valid_preds),
            "pchembl_std_pred": np.std(valid_preds),
            "pchembl_mean_true": np.mean(valid_true),
            "pchembl_std_true": np.std(valid_true),
            "pchembl_mse_raw": mse_raw,
            "pchembl_mae_raw": mae_raw,
            "pchembl_rmse_raw": rmse_raw,
        }

        try:
            from scipy.stats import pearsonr, spearmanr

            pearson_corr, _ = pearsonr(true_raw, pred_raw)
            spearman_corr, _ = spearmanr(true_raw, pred_raw)
            metrics["pchembl_pearson_raw"] = float(pearson_corr)
            metrics["pchembl_spearman_raw"] = float(spearman_corr)
        except Exception as exc:
            if logger is not None:
                logger.warning("Could not compute Pearson/Spearman correlations: %s", exc)

        if group_ids is not None:
            try:
                group_ids = np.asarray(group_ids)
                group_ids = group_ids[valid_mask]
                valid_group_mask = group_ids >= 0
                if valid_group_mask.any():
                    group_ids = group_ids[valid_group_mask]
                    pred_raw_g = pred_raw[valid_group_mask]
                    true_raw_g = true_raw[valid_group_mask]
                    pair_acc = _pairwise_accuracy(pred_raw_g, true_raw_g, group_ids)
                    if pair_acc is not None:
                        metrics["pchembl_pairwise_acc"] = pair_acc
                    group_spearman = _group_spearman(pred_raw_g, true_raw_g, group_ids)
                    if group_spearman is not None:
                        metrics["pchembl_group_spearman"] = group_spearman
            except Exception as exc:
                if logger is not None:
                    logger.warning("Could not compute group-based metrics: %s", exc)

        return metrics
    except Exception as exc:
        if logger is not None:
            logger.error("Error computing pChEMBL metrics: %s", exc, exc_info=True)
        return {}
