"""
Evaluation metrics for MHBAP — real-data evaluation.

Outputs per head:
  emotion     : accuracy, macro-F1, per-class F1, confusion_matrix, ROC-AUC (OvR)
  stress      : RMSE, MAE, R²
  engagement  : RMSE, MAE, R²
  attention   : RMSE, MAE, R²
  fatigue     : RMSE, MAE, R²
"""
from __future__ import annotations
from typing import Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, roc_auc_score, confusion_matrix,
)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-8))

def emotion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """y_pred: (N, C) logits; y_true: (N,) int labels."""
    pred_cls = np.argmax(y_pred, axis=1) if y_pred.ndim == 2 else y_pred.astype(int)
    classes  = sorted(np.unique(y_true).tolist())
    per_f1   = f1_score(y_true, pred_cls, labels=classes, average=None, zero_division=0)
    cm       = confusion_matrix(y_true, pred_cls, labels=classes).tolist()
    # ROC-AUC one-vs-rest (needs probabilities; use softmax if logits)
    roc_auc  = None
    if y_pred.ndim == 2 and len(classes) > 1:
        try:
            from scipy.special import softmax as _sfmx
            probs   = _sfmx(y_pred, axis=1)
            roc_auc = float(roc_auc_score(
                y_true, probs[:, :len(classes)], multi_class="ovr",
                labels=classes, average="macro"))
        except Exception:
            pass
    out = {
        "accuracy":        float(accuracy_score(y_true, pred_cls)),
        "macro_f1":        float(f1_score(y_true, pred_cls, average="macro", zero_division=0)),
        "per_class_f1":    {str(c): float(f) for c, f in zip(classes, per_f1)},
        "confusion_matrix": cm,
    }
    if roc_auc is not None:
        out["roc_auc_ovr"] = roc_auc
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae":  mae(y_true, y_pred),
        "r2":   r_squared(y_true, y_pred),
    }


def compute_all_metrics(
    targets: Dict[str, np.ndarray],
    predictions: Dict[str, np.ndarray],
) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    if "emotion" in targets and "emotion" in predictions:
        results["emotion"] = emotion_metrics(targets["emotion"], predictions["emotion"])
    for head in ("stress", "engagement", "attention", "fatigue"):
        if head in targets and head in predictions:
            results[head] = regression_metrics(targets[head], predictions[head])
    return results
