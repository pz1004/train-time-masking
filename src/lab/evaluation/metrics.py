from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def binary_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 10) -> dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=int)
    y_prob_array = np.asarray(y_prob, dtype=float)
    return {
        "auroc": _safe_round(_safe_auroc(y_true_array, y_prob_array)),
        "brier": _safe_round(float(brier_score_loss(y_true_array, y_prob_array))),
        "log_loss": _safe_round(float(log_loss(y_true_array, y_prob_array, labels=[0, 1]))),
        "ece": _safe_round(expected_calibration_error(y_true_array, y_prob_array, n_bins=n_bins)),
    }


def thresholded_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, *, threshold: float) -> dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=int)
    y_prob_array = np.asarray(y_prob, dtype=float)
    y_pred_array = (y_prob_array >= float(threshold)).astype(int)

    true_positive = int(np.sum((y_true_array == 1) & (y_pred_array == 1)))
    true_negative = int(np.sum((y_true_array == 0) & (y_pred_array == 0)))
    false_positive = int(np.sum((y_true_array == 0) & (y_pred_array == 1)))
    false_negative = int(np.sum((y_true_array == 1) & (y_pred_array == 0)))

    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    specificity = _safe_divide(true_negative, true_negative + false_positive)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    accuracy = _safe_divide(true_positive + true_negative, len(y_true_array))
    balanced_accuracy = 0.5 * (recall + specificity)

    return {
        "threshold": _safe_round(float(threshold)),
        "accuracy": _safe_round(accuracy),
        "balanced_accuracy": _safe_round(balanced_accuracy),
        "precision": _safe_round(precision),
        "recall": _safe_round(recall),
        "specificity": _safe_round(specificity),
        "f1": _safe_round(f1),
        "tp": float(true_positive),
        "tn": float(true_negative),
        "fp": float(false_positive),
        "fn": float(false_negative),
    }


def select_threshold_by_validation_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_prob_array = np.asarray(y_prob, dtype=float)
    if len(y_prob_array) == 0:
        return 0.5
    candidates = np.unique(np.clip(y_prob_array, 0.0, 1.0))
    if len(candidates) > 512:
        candidates = np.quantile(candidates, np.linspace(0.0, 1.0, 512))
    candidates = np.unique(np.concatenate(([0.5], candidates)))
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        score = thresholded_classification_metrics(y_true, y_prob_array, threshold=float(threshold))["f1"]
        if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)):
            best_score = float(score)
            best_threshold = float(threshold)
    return float(best_threshold)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 10) -> float:
    y_true_array = np.asarray(y_true, dtype=float)
    y_prob_array = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total_count = len(y_true_array)
    ece = 0.0

    for index in range(n_bins):
        left_edge = bin_edges[index]
        right_edge = bin_edges[index + 1]
        if index == n_bins - 1:
            mask = (y_prob_array >= left_edge) & (y_prob_array <= right_edge)
        else:
            mask = (y_prob_array >= left_edge) & (y_prob_array < right_edge)

        if not np.any(mask):
            continue

        bin_accuracy = float(y_true_array[mask].mean())
        bin_confidence = float(y_prob_array[mask].mean())
        bin_weight = float(mask.mean())
        ece += abs(bin_accuracy - bin_confidence) * bin_weight

    return float(ece)


def _safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return math.nan
    return float(roc_auc_score(y_true, y_prob))


def _safe_round(value: float) -> float:
    if math.isnan(value):
        return value
    return round(float(value), 6)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)
