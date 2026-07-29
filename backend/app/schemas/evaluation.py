"""Pydantic schemas for evaluation API responses."""
from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel


class ClassMetricsSchema(BaseModel):
    label: str
    precision: float
    recall: float
    f1: float
    support: int


class EvaluationReportSchema(BaseModel):
    name: str
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    cohen_kappa: float
    mae: float
    rmse: float
    per_class: List[ClassMetricsSchema]
    confusion_matrix: List[List[int]]
    n_samples: int


class BenchmarkResponse(BaseModel):
    reports: List[EvaluationReportSchema]
    n_samples: int
    seed: int


class AblationResultSchema(BaseModel):
    active_modalities: List[str]
    dropped_modalities: List[str]
    accuracy: float
    macro_f1: float
    weighted_f1: float
    cohen_kappa: float


class AblationStudyResponse(BaseModel):
    n_samples: int
    seed: int
    results: List[AblationResultSchema]
    baseline_f1: float          # full-fusion F1 for comparison
