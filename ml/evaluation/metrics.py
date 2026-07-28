"""
Evaluation metrics for MHBAP.

Supports both classification and regression targets per output head.
Used by both training loop (Phase 6-7) and evaluation framework (Phase 11).

Targets
-------
emotion     : 8-class classification → accuracy, macro-F1, per-class F1
stress      : regression (0-10) + 3-class → RMSE, MAE, r², accuracy
engagement  : regression (0-1) + binary → RMSE, MAE, AUC
attention   : regression (0-1) → RMSE, MAE
fatigue     : regression + binary (alert/not) → RMSE, AUC
"""
from __future__ import annotations

from typing import Dict, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-8))


def emotion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """y_pred: (N, 8) logits or probabilities; y_true: (N,) int labels."""
    pred_cls = np.argmax(y_pred, axis=1) if y_pred.ndim == 2 else y_pred
    return {
        "accuracy": float(accuracy_score(y_true, pred_cls)),
        "macro_f1": float(f1_score(y_true, pred_cls, average="macro", zero_division=0)),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Shared regression metrics for stress / engagement / attention / fatigue."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r_squared(y_true, y_pred),
    }


def compute_all_metrics(
    targets: Dict[str, np.ndarray],
    predictions: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for all 5 output heads.

    Parameters
    ----------
    targets     : {"emotion": ..., "stress": ..., ...}
    predictions : same keys

    Returns
    -------
    Nested dict of metric results per target.
    """
    results: Dict[str, Dict[str, float]] = {}

    if "emotion" in targets and "emotion" in predictions:
        results["emotion"] = emotion_metrics(targets["emotion"], predictions["emotion"])

    for head in ("stress", "engagement", "attention", "fatigue"):
        if head in targets and head in predictions:
            results[head] = regression_metrics(targets[head], predictions[head])

    return results
