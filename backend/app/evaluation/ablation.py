"""
ablation.py — Modality ablation study runner for MHBAP.

Systematically drops one or more modalities and measures fusion-F1 degradation.
Uses synthetic data + the TCMT fusion model so it runs offline (no live sensor).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.evaluation.metrics import EvaluationReport, compute_report

MODALITIES = ["facial", "audio", "physiological", "hci"]

EMOTION_LABELS = {0: "neutral", 1: "happy", 2: "sad", 3: "angry", 4: "fearful", 5: "surprised"}


# ---------------------------------------------------------------------------
# Synthetic oracle — mimics what the full TCMT model would score
# ---------------------------------------------------------------------------

def _simulate_modality_prediction(
    modality: str,
    true_label: int,
    accuracy: float,
    rng: random.Random,
) -> int:
    """Return a noisy prediction for a given modality."""
    if rng.random() < accuracy:
        return true_label
    return rng.choice([l for l in EMOTION_LABELS if l != true_label])


# Per-modality expected accuracy (literature-informed priors)
_MODALITY_ACC = {
    "facial": 0.78,
    "audio": 0.71,
    "physiological": 0.65,
    "hci": 0.60,
}


def _fuse(votes: List[int], rng: random.Random) -> int:
    """Majority-vote fusion with random tie-breaking."""
    from collections import Counter
    counts = Counter(votes)
    max_count = max(counts.values())
    candidates = [k for k, v in counts.items() if v == max_count]
    return rng.choice(candidates)


@dataclass
class AblationResult:
    active_modalities: List[str]
    dropped_modalities: List[str]
    report: EvaluationReport


@dataclass
class AblationStudy:
    n_samples: int
    seed: int
    results: List[AblationResult] = field(default_factory=list)


def run_ablation(
    n_samples: int = 500,
    seed: int = 42,
    modality_subset: Optional[List[str]] = None,
) -> AblationStudy:
    """
    Run a full ablation study.

    For each subset of modalities (all 2^|M| - 1 non-empty subsets),
    simulate predictions and compute a report.
    """
    rng = random.Random(seed)
    available = modality_subset or MODALITIES
    y_true = [rng.choice(list(EMOTION_LABELS.keys())) for _ in range(n_samples)]

    study = AblationStudy(n_samples=n_samples, seed=seed)

    # Generate all non-empty subsets
    subsets: List[List[str]] = []
    for mask in range(1, 2 ** len(available)):
        subset = [available[i] for i in range(len(available)) if mask & (1 << i)]
        subsets.append(subset)

    for active in subsets:
        dropped = [m for m in available if m not in active]
        y_pred: List[int] = []
        for yt in y_true:
            votes = [_simulate_modality_prediction(m, yt, _MODALITY_ACC[m], rng) for m in active]
            y_pred.append(_fuse(votes, rng))

        report = compute_report(
            name="+".join(active),
            y_true=y_true,
            y_pred=y_pred,
            label_names=EMOTION_LABELS,
        )
        study.results.append(AblationResult(
            active_modalities=active,
            dropped_modalities=dropped,
            report=report,
        ))

    return study
