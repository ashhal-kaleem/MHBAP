"""
metrics.py — Classification and regression metrics for MHBAP evaluation.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ClassMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class EvaluationReport:
    name: str
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    cohen_kappa: float
    mae: float
    rmse: float
    per_class: List[ClassMetrics] = field(default_factory=list)
    confusion_matrix: List[List[int]] = field(default_factory=list)
    n_samples: int = 0


def _safe_div(num: float, den: float) -> float:
    return num / den if den != 0.0 else 0.0


def precision_recall_f1(
    y_true: List[int],
    y_pred: List[int],
    labels: Optional[List[int]] = None,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, int]]:
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    tp: Dict[int, int] = defaultdict(int)
    fp: Dict[int, int] = defaultdict(int)
    fn: Dict[int, int] = defaultdict(int)
    support: Dict[int, int] = defaultdict(int)
    for yt, yp in zip(y_true, y_pred):
        support[yt] += 1
        if yt == yp:
            tp[yt] += 1
        else:
            fp[yp] += 1
            fn[yt] += 1
    prec = {c: _safe_div(tp[c], tp[c] + fp[c]) for c in labels}
    rec  = {c: _safe_div(tp[c], tp[c] + fn[c]) for c in labels}
    f1   = {c: _safe_div(2 * prec[c] * rec[c], prec[c] + rec[c]) for c in labels}
    return prec, rec, f1, dict(support)


def confusion_matrix(y_true: List[int], y_pred: List[int], labels: List[int]) -> List[List[int]]:
    idx = {c: i for i, c in enumerate(labels)}
    n = len(labels)
    cm = [[0] * n for _ in range(n)]
    for yt, yp in zip(y_true, y_pred):
        if yt in idx and yp in idx:
            cm[idx[yt]][idx[yp]] += 1
    return cm


def cohen_kappa(y_true: List[int], y_pred: List[int]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    n = len(y_true)
    if n == 0:
        return 0.0
    po = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / n
    true_counts = defaultdict(int)
    pred_counts = defaultdict(int)
    for yt, yp in zip(y_true, y_pred):
        true_counts[yt] += 1
        pred_counts[yp] += 1
    pe = sum((true_counts[c] / n) * (pred_counts[c] / n) for c in labels)
    return _safe_div(po - pe, 1.0 - pe)


def mae(y_true: List[float], y_pred: List[float]) -> float:
    if not y_true:
        return 0.0
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true: List[float], y_pred: List[float]) -> float:
    if not y_true:
        return 0.0
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true))


def compute_report(
    name: str,
    y_true: List[int],
    y_pred: List[int],
    label_names: Optional[Dict[int, str]] = None,
    y_true_cont: Optional[List[float]] = None,
    y_pred_cont: Optional[List[float]] = None,
) -> EvaluationReport:
    labels = sorted(set(y_true) | set(y_pred))
    prec, rec, f1, sup = precision_recall_f1(y_true, y_pred, labels)
    n = len(y_true)
    accuracy = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / n if n else 0.0
    total_sup = sum(sup.values())
    macro_p = sum(prec.values()) / len(labels) if labels else 0.0
    macro_r = sum(rec.values()) / len(labels) if labels else 0.0
    macro_f = sum(f1.values()) / len(labels) if labels else 0.0
    weighted_f = sum(f1[c] * sup.get(c, 0) for c in labels) / total_sup if total_sup else 0.0
    kappa = cohen_kappa(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels)
    _mae = mae(y_true_cont or [], y_pred_cont or [])
    _rmse = rmse(y_true_cont or [], y_pred_cont or [])
    per_class = [
        ClassMetrics(
            label=label_names.get(c, str(c)) if label_names else str(c),
            precision=round(prec[c], 4),
            recall=round(rec[c], 4),
            f1=round(f1[c], 4),
            support=sup.get(c, 0),
        )
        for c in labels
    ]
    return EvaluationReport(
        name=name,
        accuracy=round(accuracy, 4),
        macro_precision=round(macro_p, 4),
        macro_recall=round(macro_r, 4),
        macro_f1=round(macro_f, 4),
        weighted_f1=round(weighted_f, 4),
        cohen_kappa=round(kappa, 4),
        mae=round(_mae, 4),
        rmse=round(_rmse, 4),
        per_class=per_class,
        confusion_matrix=cm,
        n_samples=n,
    )
