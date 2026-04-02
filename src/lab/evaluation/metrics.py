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
