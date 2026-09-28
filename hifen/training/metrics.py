from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    coverage_error,
    f1_score,
    label_ranking_average_precision_score,
    label_ranking_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


LEVELS = ("ec1", "ec2", "ec3", "ec4")


def _as_binary(y_true: np.ndarray) -> np.ndarray:
    return (np.asarray(y_true) > 0).astype(np.int64)


def _as_prob(y_prob: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(y_prob, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)


def _safe_float(value: float | int | np.floating) -> float:
    value = float(value)
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return value


def _safe_metric(fn: Callable[[], float], default: float = 0.0) -> float:
    try:
        return _safe_float(fn())
    except Exception:
        return float(default)


def _valid_auc_columns(y_true: np.ndarray) -> np.ndarray:
    positives = y_true.sum(axis=0)
    negatives = y_true.shape[0] - positives
    return (positives > 0) & (negatives > 0)


def _valid_ap_columns(y_true: np.ndarray) -> np.ndarray:
    return y_true.sum(axis=0) > 0


def _safe_ratio(numerator: float | int | np.integer, denominator: float | int | np.integer) -> float:
    denominator = float(denominator)
    return 0.0 if denominator == 0.0 else float(numerator) / denominator


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=np.float64), where=denominator != 0)


def _threshold_bin_counts(bins: np.ndarray, positives: np.ndarray, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    grid_size = int(grid_size)
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    bins = np.asarray(bins, dtype=np.int64).reshape(-1)
    positives = np.asarray(positives, dtype=bool).reshape(-1)
    if positives.shape[0] != bins.shape[0]:
        raise ValueError("positives and bins must have the same length")
    valid = (bins >= 0) & (bins < grid_size)
    valid_bins = np.ascontiguousarray(bins[valid], dtype=np.int64)
    positive_bins = np.ascontiguousarray(bins[valid & positives], dtype=np.int64)
    predicted_by_bin = np.bincount(valid_bins, minlength=grid_size)
    positive_by_bin = np.bincount(positive_bins, minlength=grid_size)
    return predicted_by_bin, positive_by_bin


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, axis: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=bool)
    pred = np.asarray(y_pred, dtype=bool)
    tp = np.count_nonzero(true & pred, axis=axis)
    fp = np.count_nonzero(~true & pred, axis=axis)
    fn = np.count_nonzero(true & ~pred, axis=axis)
    tn = np.count_nonzero(~true & ~pred, axis=axis)
    return tp, fp, fn, tn


def _nanmean_or_zero(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    selected = np.asarray(values if mask is None else values[mask], dtype=np.float64)
    selected = selected[np.isfinite(selected)]
    return float(selected.mean()) if selected.size else 0.0


def _per_class_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    valid = _valid_ap_columns(y_true)
    values = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    for idx in np.flatnonzero(valid):
        values[idx] = _safe_metric(lambda col=idx: average_precision_score(y_true[:, col], y_prob[:, col]))
    return values


def _per_class_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    valid = _valid_auc_columns(y_true)
    values = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    for idx in np.flatnonzero(valid):
        values[idx] = _safe_metric(lambda col=idx: roc_auc_score(y_true[:, col], y_prob[:, col]), default=0.5)
    return values


def _threshold_values(threshold: float | Sequence[float] | np.ndarray, n_classes: int) -> np.ndarray:
    values = np.asarray(threshold, dtype=np.float64)
    if values.ndim == 0:
        return np.full((1, int(n_classes)), float(values), dtype=np.float64)
    values = values.reshape(-1)
    if values.size == 1:
        return np.full((1, int(n_classes)), float(values[0]), dtype=np.float64)
    if values.size != int(n_classes):
        raise ValueError(f"threshold has {values.size} value(s), expected 1 or {n_classes}")
    return values.reshape(1, int(n_classes))


def _same_threshold(
    cached: object,
    threshold: float | Sequence[float] | np.ndarray,
    n_classes: int,
) -> bool:
    try:
        return np.array_equal(_threshold_values(cached, n_classes), _threshold_values(threshold, n_classes))
    except Exception:
        return False


def optimize_global_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    steps: int = 101,
    default_threshold: float = 0.5,
    min_threshold: float = 0.0,
    max_threshold: float = 1.0,
    beta: float = 1.0,
) -> float:
    y_true = _as_binary(y_true).astype(bool).ravel()
    y_prob = _as_prob(y_prob).ravel()
    min_threshold = float(min_threshold)
    max_threshold = float(max_threshold)
    if not 0.0 <= min_threshold <= max_threshold <= 1.0:
        raise ValueError("threshold range must satisfy 0 <= min_threshold <= max_threshold <= 1")
    beta = float(beta)
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if not np.any(y_true):
        return float(default_threshold)
    grid = np.linspace(min_threshold, max_threshold, int(steps))
    # Reserve bucket zero for scores below the configured threshold floor.
    # This keeps every index passed to np.bincount non-negative.  The previous
    # ``searchsorted(...) - 1`` representation could surface a -1 underflow
    # bucket during long cross-fitted calibration runs.
    bins = np.searchsorted(grid, y_prob, side="right")
    predicted_with_underflow, positive_with_underflow = _threshold_bin_counts(
        bins,
        y_true,
        grid.size + 1,
    )
    predicted_by_bin = predicted_with_underflow[1:]
    positive_by_bin = positive_with_underflow[1:]
    predicted = np.cumsum(predicted_by_bin[::-1])[::-1]
    true_positive = np.cumsum(positive_by_bin[::-1])[::-1]
    beta_squared = beta**2
    fbeta = _safe_divide(
        (1.0 + beta_squared) * true_positive,
        predicted + beta_squared * int(y_true.sum()),
    )
    return float(grid[int(np.argmax(fbeta))])


def optimize_per_class_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    steps: int = 101,
    default_threshold: float = 0.5,
    min_threshold: float = 0.0,
    max_threshold: float = 1.0,
    unseen_fallback_threshold: float | Sequence[float] | np.ndarray | None = None,
    beta: float | Sequence[float] | np.ndarray = 1.0,
) -> np.ndarray:
    y_true = _as_binary(y_true)
    y_prob = _as_prob(y_prob)
    if y_prob.ndim != 2:
        raise ValueError("y_prob must be a 2D array")
    min_threshold = float(min_threshold)
    max_threshold = float(max_threshold)
    if not 0.0 <= min_threshold <= max_threshold <= 1.0:
        raise ValueError("threshold range must satisfy 0 <= min_threshold <= max_threshold <= 1")
    class_count = int(y_prob.shape[1])
    beta_values = np.asarray(beta, dtype=np.float64)
    if beta_values.ndim == 0 or beta_values.size == 1:
        beta_values = np.full(class_count, float(beta_values.reshape(-1)[0]), dtype=np.float64)
    else:
        beta_values = beta_values.reshape(-1)
        if beta_values.size != class_count:
            raise ValueError(f"beta has {beta_values.size} value(s), expected 1 or {class_count}")
    if np.any(beta_values <= 0.0):
        raise ValueError("beta values must be positive")
    fallback_values = None
    if unseen_fallback_threshold is not None:
        fallback_values = np.asarray(unseen_fallback_threshold, dtype=np.float64)
        if fallback_values.ndim == 0 or fallback_values.size == 1:
            fallback_values = np.full(class_count, float(fallback_values.reshape(-1)[0]), dtype=np.float64)
        else:
            fallback_values = fallback_values.reshape(-1)
            if fallback_values.size != class_count:
                raise ValueError(
                    f"unseen_fallback_threshold has {fallback_values.size} value(s), "
                    f"expected 1 or {class_count}"
                )
    thresholds = np.full(class_count, float(default_threshold), dtype=np.float64)
    grid = np.linspace(min_threshold, max_threshold, int(steps))
    for col in range(class_count):
        true_col = y_true[:, col].astype(bool)
        positives = int(true_col.sum())
        if positives == 0:
            thresholds[col] = 1.0 if fallback_values is None else float(fallback_values[col])
            continue
        scores = y_prob[:, col]
        # Keep the underflow bucket explicit instead of encoding scores below
        # min_threshold as -1.  np.bincount only accepts non-negative indices;
        # dropping bucket zero after counting preserves the original threshold
        # semantics without ever sending a negative bin to the counter.
        bins = np.searchsorted(grid, scores, side="right")
        predicted_with_underflow, positive_with_underflow = _threshold_bin_counts(
            bins,
            true_col,
            grid.size + 1,
        )
        predicted_by_bin = predicted_with_underflow[1:]
        positive_by_bin = positive_with_underflow[1:]
        predicted = np.cumsum(predicted_by_bin[::-1])[::-1]
        true_positive = np.cumsum(positive_by_bin[::-1])[::-1]
        beta_squared = float(beta_values[col]) ** 2
        fbeta = _safe_divide(
            (1.0 + beta_squared) * true_positive,
            predicted + beta_squared * positives,
        )
        thresholds[col] = float(grid[int(np.argmax(fbeta))])
    return thresholds


def apply_hierarchy_threshold_gate(
    thresholds: dict[str, float | Sequence[float] | np.ndarray],
    parent_child_maps: dict[str, list[tuple[int, int]]],
) -> dict[str, float | np.ndarray]:
    adjusted: dict[str, float | np.ndarray] = {}
    for level, values in thresholds.items():
        array = np.asarray(values, dtype=np.float64)
        adjusted[level] = float(array) if array.ndim == 0 else array.reshape(-1).copy()
    for parent_level, child_level, key in [
        ("ec1", "ec2", "ec1_to_ec2"),
        ("ec2", "ec3", "ec2_to_ec3"),
        ("ec3", "ec4", "ec3_to_ec4"),
    ]:
        pairs = parent_child_maps.get(key, [])
        if not pairs or parent_level not in adjusted or child_level not in adjusted:
            continue
        parent_values = adjusted[parent_level]
        child_values = adjusted[child_level]
        for parent_idx, child_idx in pairs:
            parent_threshold = float(parent_values if np.isscalar(parent_values) else parent_values[parent_idx])
            if np.isscalar(child_values):
                child_values = max(float(child_values), parent_threshold)
            elif child_idx < child_values.size:
                child_values[child_idx] = max(float(child_values[child_idx]), parent_threshold)
        adjusted[child_level] = child_values
    return adjusted


def apply_hierarchy_probability_gate(
    probs: dict[str, np.ndarray],
    parent_child_maps: dict[str, list[tuple[int, int]]],
    *,
    mode: str = "min_parent",
) -> dict[str, np.ndarray]:
    if mode in {"", "none", "off", "false"}:
        return {level: _as_prob(values).copy() for level, values in probs.items()}
    adjusted = {level: _as_prob(values).copy() for level, values in probs.items()}
    for parent_level, child_level, key in [
        ("ec1", "ec2", "ec1_to_ec2"),
        ("ec2", "ec3", "ec2_to_ec3"),
        ("ec3", "ec4", "ec3_to_ec4"),
    ]:
        pairs = parent_child_maps.get(key, [])
        if not pairs or parent_level not in adjusted or child_level not in adjusted:
            continue
        for parent_idx, child_idx in pairs:
            if parent_idx >= adjusted[parent_level].shape[1] or child_idx >= adjusted[child_level].shape[1]:
                continue
            parent_prob = adjusted[parent_level][:, parent_idx]
            if mode == "multiply_parent":
                adjusted[child_level][:, child_idx] *= parent_prob
            elif mode == "mask_parent":
                adjusted[child_level][:, child_idx] *= parent_prob >= 0.5
            else:
                adjusted[child_level][:, child_idx] = np.minimum(adjusted[child_level][:, child_idx], parent_prob)
    return adjusted


def build_metric_cache(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | Sequence[float] | np.ndarray = 0.5,
) -> dict[str, np.ndarray | float]:
    y_true = _as_binary(y_true)
    y_prob = _as_prob(y_prob)
    threshold_values = _threshold_values(threshold, y_prob.shape[1])
    y_pred = (y_prob >= threshold_values).astype(np.int64)
    tp, fp, fn, tn = _confusion_counts(y_true, y_pred, axis=0)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    return {
        "threshold": threshold_values.reshape(-1) if threshold_values.size > 1 else float(threshold_values[0, 0]),
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auprc": _per_class_average_precision(y_true, y_prob),
        "auroc": _per_class_roc_auc(y_true, y_prob),
    }


def _metric_cache(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | Sequence[float] | np.ndarray,
    metric_cache: dict[str, np.ndarray | float] | None,
) -> dict[str, np.ndarray | float]:
    if metric_cache is not None and _same_threshold(metric_cache.get("threshold"), threshold, np.asarray(y_prob).shape[1]):
        return metric_cache
    return build_metric_cache(y_true, y_prob, threshold=threshold)


def _macro_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return _nanmean_or_zero(_per_class_average_precision(y_true, y_prob))


def _macro_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return _nanmean_or_zero(_per_class_roc_auc(y_true, y_prob))


def _micro_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.sum() == 0:
        return 0.0
    return _safe_metric(lambda: average_precision_score(y_true.ravel(), y_prob.ravel()))


def _micro_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    flat = y_true.ravel()
    if flat.size == 0 or flat.min() == flat.max():
        return 0.0
    return _safe_metric(lambda: roc_auc_score(flat, y_prob.ravel()), default=0.5)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 15) -> float:
    y_true = _as_binary(y_true).ravel()
    y_prob = _as_prob(y_prob).ravel()
    if y_prob.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    ece = 0.0
    for idx in range(len(edges) - 1):
        lo, hi = edges[idx], edges[idx + 1]
        if idx == len(edges) - 2:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        confidence = float(y_prob[mask].mean())
        accuracy = float(y_true[mask].mean())
        ece += float(mask.mean()) * abs(accuracy - confidence)
    return float(ece)


def fmax_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    thresholds: Sequence[float] | None = None,
    steps: int = 101,
) -> dict[str, float]:
    y_true = _as_binary(y_true)
    y_prob = _as_prob(y_prob)
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, int(steps))
    best = {"fmax": 0.0, "fmax_threshold": 0.0, "fmax_precision": 0.0, "fmax_recall": 0.0}
    true_bool = y_true.astype(bool)
    positives = int(true_bool.sum())
    for threshold in thresholds:
        pred_bool = y_prob >= float(threshold)
        true_positive = int(np.count_nonzero(pred_bool & true_bool))
        precision = _safe_ratio(true_positive, int(np.count_nonzero(pred_bool)))
        recall = _safe_ratio(true_positive, positives)
        denom = precision + recall
        f1 = 0.0 if denom == 0 else 2.0 * precision * recall / denom
        if f1 > best["fmax"]:
            best = {
                "fmax": float(f1),
                "fmax_threshold": float(threshold),
                "fmax_precision": float(precision),
                "fmax_recall": float(recall),
            }
    return best


def topk_metrics(y_true: np.ndarray, y_prob: np.ndarray, ks: Sequence[int] = (1, 3, 5)) -> dict[str, float]:
    y_true = _as_binary(y_true)
    y_prob = _as_prob(y_prob)
    if y_prob.ndim != 2 or y_prob.shape[0] == 0 or y_prob.shape[1] == 0:
        return {f"top{k}_{name}": 0.0 for k in ks for name in ("hit", "precision", "recall", "full_recall")} | {
            f"top{k}_exact_match": 0.0 for k in ks
        }
    metrics: dict[str, float] = {}
    positives_per_row = y_true.sum(axis=1)
    rows = np.arange(y_true.shape[0])[:, None]
    for k_value in ks:
        k = min(int(k_value), y_prob.shape[1])
        if k <= 0:
            continue
        topk = np.argpartition(-y_prob, kth=k - 1, axis=1)[:, :k]
        hits = y_true[rows, topk].sum(axis=1)
        has_positive = positives_per_row > 0
        metrics[f"top{k_value}_hit"] = float(np.mean(hits[has_positive] > 0)) if has_positive.any() else 0.0
        metrics[f"top{k_value}_precision"] = float(np.mean(hits / float(k)))
        row_recall = np.divide(hits, positives_per_row, out=np.zeros_like(hits, dtype=np.float64), where=positives_per_row > 0)
        metrics[f"top{k_value}_recall"] = float(row_recall[has_positive].mean()) if has_positive.any() else 0.0
        metrics[f"top{k_value}_full_recall"] = float(np.mean((hits >= positives_per_row)[has_positive])) if has_positive.any() else 0.0
        metrics[f"top{k_value}_exact_match"] = float(np.mean((hits == positives_per_row) & (positives_per_row == k)))
    return metrics


def single_label_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    y_true = _as_binary(y_true)
    y_prob = _as_prob(y_prob)
    sample_count = int(y_true.shape[0]) if y_true.ndim == 2 else 0
    eligible = y_true.sum(axis=1) == 1 if y_true.ndim == 2 else np.zeros(0, dtype=bool)
    single_count = int(eligible.sum())
    metrics = {
        "single_label_sample_count": float(single_count),
        "single_label_sample_fraction": _safe_ratio(single_count, sample_count),
        "single_label_excluded_sample_count": float(sample_count - single_count),
        "single_label_class_count": 0.0,
        "single_label_accuracy": 0.0,
        "single_label_balanced_accuracy": 0.0,
        "single_label_precision_macro": 0.0,
        "single_label_precision_micro": 0.0,
        "single_label_precision_weighted": 0.0,
        "single_label_recall_macro": 0.0,
        "single_label_recall_micro": 0.0,
        "single_label_recall_weighted": 0.0,
        "single_label_f1_macro": 0.0,
        "single_label_f1_micro": 0.0,
        "single_label_f1_weighted": 0.0,
        "single_label_mcc": 0.0,
        "single_label_mean_reciprocal_rank": 0.0,
        "single_label_cross_entropy": 0.0,
        "single_label_brier_score": 0.0,
        "single_label_auprc_macro": 0.0,
        "single_label_auprc_micro": 0.0,
        "single_label_auroc_macro": 0.0,
        "single_label_auroc_micro": 0.0,
    }
    for k_value in ks:
        metrics[f"single_label_top{k_value}_accuracy"] = 0.0
    if single_count == 0 or y_prob.ndim != 2 or y_prob.shape[1] == 0:
        return metrics

    true_index = np.argmax(y_true[eligible], axis=1)
    probabilities = y_prob[eligible]
    predicted_index = np.argmax(probabilities, axis=1)
    observed_labels = np.unique(true_index)
    normalized = probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    zero_rows = probabilities.sum(axis=1) <= 0
    if zero_rows.any():
        normalized[zero_rows] = 1.0 / probabilities.shape[1]
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[true_index]
    ranks = np.argsort(-probabilities, axis=1)
    true_rank = np.argmax(ranks == true_index[:, None], axis=1) + 1
    accuracy = float(np.mean(predicted_index == true_index))
    balanced_accuracy = _safe_metric(lambda: balanced_accuracy_score(true_index, predicted_index)) if observed_labels.size > 1 else accuracy
    mcc = _safe_metric(lambda: matthews_corrcoef(true_index, predicted_index)) if np.unique(np.concatenate((true_index, predicted_index))).size > 1 else 0.0

    metrics.update(
        {
            "single_label_class_count": float(observed_labels.size),
            "single_label_accuracy": accuracy,
            "single_label_balanced_accuracy": balanced_accuracy,
            "single_label_precision_macro": _safe_metric(
                lambda: precision_score(true_index, predicted_index, labels=observed_labels, average="macro", zero_division=0)
            ),
            "single_label_precision_micro": _safe_metric(
                lambda: precision_score(true_index, predicted_index, average="micro", zero_division=0)
            ),
            "single_label_precision_weighted": _safe_metric(
                lambda: precision_score(true_index, predicted_index, labels=observed_labels, average="weighted", zero_division=0)
            ),
            "single_label_recall_macro": _safe_metric(
                lambda: recall_score(true_index, predicted_index, labels=observed_labels, average="macro", zero_division=0)
            ),
            "single_label_recall_micro": _safe_metric(
                lambda: recall_score(true_index, predicted_index, average="micro", zero_division=0)
            ),
            "single_label_recall_weighted": _safe_metric(
                lambda: recall_score(true_index, predicted_index, labels=observed_labels, average="weighted", zero_division=0)
            ),
            "single_label_f1_macro": _safe_metric(
                lambda: f1_score(true_index, predicted_index, labels=observed_labels, average="macro", zero_division=0)
            ),
            "single_label_f1_micro": _safe_metric(
                lambda: f1_score(true_index, predicted_index, average="micro", zero_division=0)
            ),
            "single_label_f1_weighted": _safe_metric(
                lambda: f1_score(true_index, predicted_index, labels=observed_labels, average="weighted", zero_division=0)
            ),
            "single_label_mcc": mcc,
            "single_label_mean_reciprocal_rank": float(np.mean(1.0 / true_rank)),
            "single_label_cross_entropy": float(-np.mean(np.log(np.maximum(normalized[np.arange(single_count), true_index], 1e-12)))),
            "single_label_brier_score": float(np.mean(np.sum((normalized - one_hot) ** 2, axis=1))),
            "single_label_auprc_macro": _macro_average_precision(one_hot, normalized),
            "single_label_auprc_micro": _micro_average_precision(one_hot, normalized),
            "single_label_auroc_macro": _macro_roc_auc(one_hot, normalized),
            "single_label_auroc_micro": _micro_roc_auc(one_hot, normalized),
        }
    )
    for k_value in ks:
        k = min(int(k_value), probabilities.shape[1])
        metrics[f"single_label_top{k_value}_accuracy"] = (
            float(np.mean(np.any(ranks[:, :k] == true_index[:, None], axis=1))) if k > 0 else 0.0
        )
    return metrics


def closed_set_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_support: np.ndarray,
    threshold: float | Sequence[float] | np.ndarray = 0.5,
    *,
    fmax_steps: int = 101,
    calibration_bins: int = 15,
) -> dict[str, float]:
    y_true = _as_binary(y_true)
    y_prob = _as_prob(y_prob)
    support = np.asarray(class_support, dtype=np.float64).reshape(-1)
    if support.size != y_true.shape[1]:
        raise ValueError("class_support must match the number of prediction classes")
    seen = support > 0
    unseen = ~seen
    unseen_positive_by_row = y_true[:, unseen].sum(axis=1) if unseen.any() else np.zeros(y_true.shape[0], dtype=np.int64)
    metrics = {
        "train_seen_class_count": float(seen.sum()),
        "train_unseen_class_count": float(unseen.sum()),
        "zero_shot_positive_count": float(unseen_positive_by_row.sum()),
        "zero_shot_sample_count": float(np.count_nonzero(unseen_positive_by_row)),
        "closed_set_sample_count": float(y_true.shape[0] - np.count_nonzero(unseen_positive_by_row)),
    }
    if not seen.any():
        return metrics
    threshold_values = _threshold_values(threshold, y_true.shape[1]).reshape(-1)[seen]
    closed = multilabel_metrics(
        y_true[:, seen],
        y_prob[:, seen],
        threshold=threshold_values,
        fmax_steps=fmax_steps,
        calibration_bins=calibration_bins,
    )
    for key in ("f1_macro", "f1_micro", "precision_macro", "precision_micro", "recall_macro", "recall_micro", "mcc_micro", "auprc_macro", "auprc_micro", "auroc_macro", "auroc_micro", "fmax"):
        metrics[f"closed_{key}"] = closed[key]
    return metrics


def multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | Sequence[float] | np.ndarray = 0.5,
    *,
    fmax_steps: int = 101,
    calibration_bins: int = 15,
    metric_cache: dict[str, np.ndarray | float] | None = None,
) -> dict[str, float]:
    cache = _metric_cache(y_true, y_prob, threshold, metric_cache)
    y_true = cache["y_true"]
    y_prob = cache["y_prob"]
    y_pred = cache["y_pred"]
    flat_true = y_true.ravel()
    flat_pred = y_pred.ravel()
    tp, fp, fn, tn = _confusion_counts(y_true, y_pred, axis=None)
    precision_micro = _safe_ratio(tp, tp + fp)
    recall_micro = _safe_ratio(tp, tp + fn)
    f1_micro = _safe_ratio(2.0 * precision_micro * recall_micro, precision_micro + recall_micro)
    mcc_denom = math.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    observed = y_true.sum(axis=0) > 0
    unobserved = ~observed
    unobserved_label_fp_count = int(y_pred[:, unobserved].sum()) if unobserved.any() else 0
    predicted_positive_count = int(y_pred.sum())
    return {
        "f1_macro": float(np.mean(cache["f1"])) if y_true.shape[1] else 0.0,
        "f1_macro_observed": float(np.mean(cache["f1"][observed])) if observed.any() else 0.0,
        "observed_class_count": float(observed.sum()),
        "unobserved_class_count": float(unobserved.sum()),
        "class_coverage": _safe_ratio(observed.sum(), y_true.shape[1]),
        "unobserved_label_fp_count": float(unobserved_label_fp_count),
        "unobserved_label_fp_rate": _safe_ratio(unobserved_label_fp_count, predicted_positive_count),
        "unobserved_label_prediction_per_sample": _safe_ratio(unobserved_label_fp_count, y_true.shape[0]),
        "f1_micro": f1_micro,
        "precision_macro": float(np.mean(cache["precision"])) if y_true.shape[1] else 0.0,
        "precision_micro": precision_micro,
        "recall_macro": float(np.mean(cache["recall"])) if y_true.shape[1] else 0.0,
        "recall_micro": recall_micro,
        "mcc_micro": 0.0 if mcc_denom == 0.0 else float((tp * tn - fp * fn) / mcc_denom),
        "subset_accuracy": float(np.mean(np.equal(y_true, y_pred).all(axis=1))) if y_true.shape[0] else 0.0,
        "hamming_loss": float(np.mean(flat_true != flat_pred)) if flat_true.size else 0.0,
        "auprc_micro": _micro_average_precision(y_true, y_prob),
        "auprc_macro": _nanmean_or_zero(cache["auprc"]),
        "auroc_micro": _micro_roc_auc(y_true, y_prob),
        "auroc_macro": _nanmean_or_zero(cache["auroc"]),
        "brier_score": float(np.mean((y_prob - y_true) ** 2)) if y_true.size else 0.0,
        "ece": expected_calibration_error(y_true, y_prob, bins=calibration_bins),
        "coverage_error": _safe_metric(lambda: coverage_error(y_true, y_prob)),
        "label_ranking_loss": _safe_metric(lambda: label_ranking_loss(y_true, y_prob)),
        "label_ranking_average_precision": _safe_metric(lambda: label_ranking_average_precision_score(y_true, y_prob)),
        **fmax_score(y_true, y_prob, steps=fmax_steps),
    }


def topk_hit_rate(y_true: np.ndarray, y_prob: np.ndarray, k: int = 5) -> float:
    return topk_metrics(y_true, y_prob, ks=(k,)).get(f"top{k}_hit", 0.0)


def class_frequency_bucket_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_support: np.ndarray | None = None,
    *,
    threshold: float | Sequence[float] | np.ndarray = 0.5,
    low_max_support: int = 50,
    metric_cache: dict[str, np.ndarray | float] | None = None,
) -> dict[str, float]:
    cache = _metric_cache(y_true, y_prob, threshold, metric_cache)
    y_true = cache["y_true"]
    y_prob = cache["y_prob"]
    y_pred = cache["y_pred"]
    observed = y_true.sum(axis=0) > 0
    support = np.asarray(class_support if class_support is not None else y_true.sum(axis=0), dtype=np.float64)
    buckets = {
        "low": (support > 0) & (support <= int(low_max_support)),
        "high": support > int(low_max_support),
    }
    metrics: dict[str, float] = {}
    for name, mask in buckets.items():
        class_count = int(mask.sum())
        observed_mask = mask & observed
        observed_class_count = int(observed_mask.sum())
        metrics[f"{name}_class_count"] = float(class_count)
        metrics[f"{name}_observed_class_count"] = float(observed_class_count)
        metrics[f"{name}_class_coverage"] = _safe_ratio(observed_class_count, class_count)
        if not mask.any():
            metrics[f"{name}_f1_macro"] = 0.0
            metrics[f"{name}_f1_macro_observed"] = 0.0
            metrics[f"{name}_f1_micro"] = 0.0
            metrics[f"{name}_precision_macro"] = 0.0
            metrics[f"{name}_precision_macro_observed"] = 0.0
            metrics[f"{name}_precision_micro"] = 0.0
            metrics[f"{name}_recall_macro"] = 0.0
            metrics[f"{name}_recall_macro_observed"] = 0.0
            metrics[f"{name}_recall_micro"] = 0.0
            metrics[f"{name}_auprc_macro"] = 0.0
            metrics[f"{name}_auprc_micro"] = 0.0
            metrics[f"{name}_auroc_macro"] = 0.0
            continue
        tp, fp, fn, _ = _confusion_counts(y_true[:, mask], y_pred[:, mask], axis=None)
        precision_micro = _safe_ratio(tp, tp + fp)
        recall_micro = _safe_ratio(tp, tp + fn)
        metrics[f"{name}_f1_macro"] = float(np.mean(cache["f1"][mask]))
        metrics[f"{name}_f1_macro_observed"] = (
            float(np.mean(cache["f1"][observed_mask])) if observed_mask.any() else 0.0
        )
        metrics[f"{name}_precision_macro"] = float(np.mean(cache["precision"][mask]))
        metrics[f"{name}_precision_macro_observed"] = (
            float(np.mean(cache["precision"][observed_mask])) if observed_mask.any() else 0.0
        )
        metrics[f"{name}_precision_micro"] = precision_micro
        metrics[f"{name}_recall_macro"] = float(np.mean(cache["recall"][mask]))
        metrics[f"{name}_recall_macro_observed"] = (
            float(np.mean(cache["recall"][observed_mask])) if observed_mask.any() else 0.0
        )
        metrics[f"{name}_recall_micro"] = recall_micro
        metrics[f"{name}_f1_micro"] = _safe_ratio(
            2.0 * precision_micro * recall_micro,
            precision_micro + recall_micro,
        )
        metrics[f"{name}_auprc_macro"] = _nanmean_or_zero(cache["auprc"], mask)
        metrics[f"{name}_auprc_micro"] = _micro_average_precision(y_true[:, mask], y_prob[:, mask])
        metrics[f"{name}_auroc_macro"] = _nanmean_or_zero(cache["auroc"], mask)
    return metrics


def _expand_ancestors(binary_by_level: dict[str, np.ndarray], parent_child_maps: dict[str, list[tuple[int, int]]]) -> dict[str, np.ndarray]:
    expanded = {level: _as_binary(values).copy() for level, values in binary_by_level.items()}
    for parent_level, child_level, key in [
        ("ec3", "ec4", "ec3_to_ec4"),
        ("ec2", "ec3", "ec2_to_ec3"),
        ("ec1", "ec2", "ec1_to_ec2"),
    ]:
        pairs = parent_child_maps.get(key, [])
        if not pairs or parent_level not in expanded or child_level not in expanded:
            continue
        for parent_idx, child_idx in pairs:
            if parent_idx < expanded[parent_level].shape[1] and child_idx < expanded[child_level].shape[1]:
                expanded[parent_level][:, parent_idx] = np.maximum(
                    expanded[parent_level][:, parent_idx],
                    expanded[child_level][:, child_idx],
                )
    return expanded


def hierarchy_violation_rate(
    probs: dict[str, np.ndarray],
    parent_child_maps: dict[str, list[tuple[int, int]]],
    threshold: float | dict[str, float | Sequence[float] | np.ndarray] = 0.5,
) -> float:
    violations = 0
    active_children = 0
    for parent_level, child_level, key in [
        ("ec1", "ec2", "ec1_to_ec2"),
        ("ec2", "ec3", "ec2_to_ec3"),
        ("ec3", "ec4", "ec3_to_ec4"),
    ]:
        pairs = parent_child_maps.get(key, [])
        if not pairs or parent_level not in probs or child_level not in probs:
            continue
        parent_prob = _as_prob(probs[parent_level])
        child_prob = _as_prob(probs[child_level])
        if isinstance(threshold, dict):
            parent_threshold = _threshold_values(threshold.get(parent_level, 0.5), parent_prob.shape[1])
            child_threshold = _threshold_values(threshold.get(child_level, 0.5), child_prob.shape[1])
        else:
            parent_threshold = _threshold_values(threshold, parent_prob.shape[1])
            child_threshold = _threshold_values(threshold, child_prob.shape[1])
        parent_pred = parent_prob >= parent_threshold
        child_pred = child_prob >= child_threshold
        for parent_idx, child_idx in pairs:
            if parent_idx >= parent_pred.shape[1] or child_idx >= child_pred.shape[1]:
                continue
            active = child_pred[:, child_idx]
            active_children += int(active.sum())
            violations += int((active & ~parent_pred[:, parent_idx]).sum())
    return float(violations / active_children) if active_children else 0.0


def parent_child_consistency_rate(
    probs: dict[str, np.ndarray],
    parent_child_maps: dict[str, list[tuple[int, int]]],
    threshold: float | dict[str, float | Sequence[float] | np.ndarray] = 0.5,
) -> float:
    return 1.0 - hierarchy_violation_rate(probs, parent_child_maps, threshold=threshold)


def hierarchical_metrics(
    targets: dict[str, np.ndarray],
    probs: dict[str, np.ndarray],
    parent_child_maps: dict[str, list[tuple[int, int]]],
    threshold: float | dict[str, float | Sequence[float] | np.ndarray] = 0.5,
    known_level_masks: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    target_expanded = _expand_ancestors(targets, parent_child_maps)
    pred_binary = {}
    for level, values in probs.items():
        y_prob = _as_prob(values)
        level_threshold = threshold.get(level, 0.5) if isinstance(threshold, dict) else threshold
        pred_binary[level] = (y_prob >= _threshold_values(level_threshold, y_prob.shape[1])).astype(np.int64)
    pred_expanded = _expand_ancestors(pred_binary, parent_child_maps)
    true_parts = []
    pred_parts = []
    for level in LEVELS:
        if level not in target_expanded or level not in pred_expanded:
            continue
        level_true = target_expanded[level]
        level_pred = pred_expanded[level]
        if known_level_masks is not None and level in known_level_masks:
            known = np.asarray(known_level_masks[level], dtype=bool).reshape(-1)
            if known.size != level_true.shape[0]:
                raise ValueError(
                    f"known_level_masks[{level!r}] has {known.size} rows, "
                    f"expected {level_true.shape[0]}"
                )
            level_true = level_true[known]
            level_pred = level_pred[known]
        if level_true.size == 0:
            continue
        true_parts.append(level_true.reshape(-1))
        pred_parts.append(level_pred.reshape(-1))
    if not true_parts:
        return {
            "hierarchical_precision": 0.0,
            "hierarchical_recall": 0.0,
            "hierarchical_f1": 0.0,
            "hierarchy_violation_rate": 0.0,
            "hierarchy_consistency": 1.0,
        }
    y_true = np.concatenate(true_parts, axis=0)
    y_pred = np.concatenate(pred_parts, axis=0)
    tp, fp, fn, _ = _confusion_counts(y_true, y_pred, axis=None)
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    denom = precision + recall
    return {
        "hierarchical_precision": float(precision),
        "hierarchical_recall": float(recall),
        "hierarchical_f1": 0.0 if denom == 0 else float(2.0 * precision * recall / denom),
        "hierarchy_violation_rate": hierarchy_violation_rate(probs, parent_child_maps, threshold=threshold),
        "hierarchy_consistency": parent_child_consistency_rate(probs, parent_child_maps, threshold=threshold),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    resamples: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float]:
    y_true = _as_binary(y_true)
    y_prob = _as_prob(y_prob)
    if y_true.shape[0] == 0 or resamples <= 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(resamples)):
        indices = rng.integers(0, y_true.shape[0], size=y_true.shape[0])
        values.append(float(metric_fn(y_true[indices], y_prob[indices])))
    alpha = (1.0 - float(confidence)) / 2.0
    return (
        _safe_float(np.quantile(values, alpha)),
        _safe_float(np.quantile(values, 1.0 - alpha)),
    )


def binary_histogram_counts(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    bins: int = 200,
) -> dict[str, np.ndarray | int]:
    true = _as_binary(y_true).reshape(-1)
    prob = _as_prob(y_prob).reshape(-1)
    if true.size != prob.size:
        raise ValueError("y_true and y_prob must have the same number of values")
    indices = np.minimum((prob * int(bins)).astype(np.int64), int(bins) - 1)
    positive = np.bincount(indices[true > 0], minlength=int(bins)).astype(np.int64)
    negative = np.bincount(indices[true <= 0], minlength=int(bins)).astype(np.int64)
    return {"positive": positive, "negative": negative, "count": int(true.size)}


def binary_metrics_from_histogram(
    positive: np.ndarray,
    negative: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    positive = np.asarray(positive, dtype=np.int64).reshape(-1)
    negative = np.asarray(negative, dtype=np.int64).reshape(-1)
    if positive.size != negative.size or positive.size == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "auprc": 0.0, "count": 0.0}
    bins = positive.size
    tp_curve = np.cumsum(positive[::-1])
    fp_curve = np.cumsum(negative[::-1])
    total_positive = int(positive.sum())
    threshold_bin = min(max(int(float(threshold) * bins), 0), bins - 1)
    curve_index = bins - 1 - threshold_bin
    tp = int(tp_curve[curve_index])
    fp = int(fp_curve[curve_index])
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, total_positive)
    denom = precision + recall
    curve_precision = _safe_divide(tp_curve, tp_curve + fp_curve)
    curve_recall = _safe_divide(tp_curve, np.full_like(tp_curve, total_positive))
    recall_delta = np.diff(np.concatenate(([0.0], curve_recall)))
    auprc = float(np.sum(curve_precision * recall_delta)) if total_positive else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": 0.0 if denom == 0.0 else float(2.0 * precision * recall / denom),
        "auprc": auprc,
        "count": float(positive.sum() + negative.sum()),
    }
