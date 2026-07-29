"""
evaluation.py — REST endpoints for benchmarks and ablation studies.

GET /api/v1/evaluation/benchmark   — run per-modality + fusion benchmark
GET /api/v1/evaluation/ablation    — run full ablation study
"""
from __future__ import annotations

from fastapi import APIRouter, Query

from backend.app.evaluation.benchmark import run_benchmark
from backend.app.evaluation.ablation import run_ablation, MODALITIES
from backend.app.schemas.evaluation import (
    BenchmarkResponse,
    EvaluationReportSchema,
    ClassMetricsSchema,
    AblationStudyResponse,
    AblationResultSchema,
)

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


def _to_report_schema(report) -> EvaluationReportSchema:
    return EvaluationReportSchema(
        name=report.name,
        accuracy=report.accuracy,
        macro_precision=report.macro_precision,
        macro_recall=report.macro_recall,
        macro_f1=report.macro_f1,
        weighted_f1=report.weighted_f1,
        cohen_kappa=report.cohen_kappa,
        mae=report.mae,
        rmse=report.rmse,
        per_class=[
            ClassMetricsSchema(
                label=c.label,
                precision=c.precision,
                recall=c.recall,
                f1=c.f1,
                support=c.support,
            )
            for c in report.per_class
        ],
        confusion_matrix=report.confusion_matrix,
        n_samples=report.n_samples,
    )


@router.get("/benchmark", response_model=BenchmarkResponse)
def benchmark(
    n_samples: int = Query(1000, ge=100, le=10000),
    seed: int = Query(42),
):
    """Run per-modality + fusion benchmark and return metrics."""
    reports = run_benchmark(n_samples=n_samples, seed=seed)
    return BenchmarkResponse(
        reports=[_to_report_schema(r) for r in reports],
        n_samples=n_samples,
        seed=seed,
    )


@router.get("/ablation", response_model=AblationStudyResponse)
def ablation(
    n_samples: int = Query(500, ge=100, le=5000),
    seed: int = Query(42),
):
    """Run full modality ablation study."""
    study = run_ablation(n_samples=n_samples, seed=seed)

    # Baseline = all modalities active
    full_key = "+".join(MODALITIES)
    full_result = next(
        (r for r in study.results if r.report.name == full_key),
        study.results[-1],
    )
    baseline_f1 = full_result.report.macro_f1

    results = [
        AblationResultSchema(
            active_modalities=r.active_modalities,
            dropped_modalities=r.dropped_modalities,
            accuracy=r.report.accuracy,
            macro_f1=r.report.macro_f1,
            weighted_f1=r.report.weighted_f1,
            cohen_kappa=r.report.cohen_kappa,
        )
        for r in study.results
    ]
    return AblationStudyResponse(
        n_samples=study.n_samples,
        seed=study.seed,
        results=results,
        baseline_f1=baseline_f1,
    )
