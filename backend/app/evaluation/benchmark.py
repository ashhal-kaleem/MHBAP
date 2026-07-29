"""
benchmark.py — Benchmark runner for MHBAP.

Runs evaluation for each individual modality + the full fusion model,
returns a list of EvaluationReport objects suitable for serialization.
"""
from __future__ import annotations

import random
from typing import List, Optional

from backend.app.evaluation.metrics import EvaluationReport, compute_report
from backend.app.evaluation.ablation import (
    MODALITIES,
    EMOTION_LABELS,
    _MODALITY_ACC,
    _simulate_modality_prediction,
    _fuse,
)


def run_benchmark(
    n_samples: int = 1000,
    seed: int = 42,
    modalities: Optional[List[str]] = None,
) -> List[EvaluationReport]:
    """
    Evaluate each modality independently plus the full fusion model.
    Returns one EvaluationReport per modality + one for 'fusion'.
    """
    rng = random.Random(seed)
    active = modalities or MODALITIES
    y_true = [rng.choice(list(EMOTION_LABELS.keys())) for _ in range(n_samples)]

    reports: List[EvaluationReport] = []

    # Per-modality reports
    for modality in active:
        acc = _MODALITY_ACC[modality]
        y_pred = [_simulate_modality_prediction(modality, yt, acc, rng) for yt in y_true]
        reports.append(compute_report(
            name=modality,
            y_true=y_true,
            y_pred=y_pred,
            label_names=EMOTION_LABELS,
        ))

    # Full fusion report
    rng2 = random.Random(seed)
    fusion_preds: List[int] = []
    for yt in y_true:
        votes = [_simulate_modality_prediction(m, yt, _MODALITY_ACC[m], rng2) for m in active]
        fusion_preds.append(_fuse(votes, rng2))
    reports.append(compute_report(
        name="fusion",
        y_true=y_true,
        y_pred=fusion_preds,
        label_names=EMOTION_LABELS,
    ))

    return reports
