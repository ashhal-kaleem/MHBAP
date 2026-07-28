"""
Health check endpoints.
GET /api/v1/health/       — liveness probe
GET /api/v1/health/ready  — readiness probe (checks DB + Redis in Phase 2)
"""
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


class ReadinessResponse(BaseModel):
    status: str
    database: str
    redis: str


@router.get("/", response_model=HealthResponse, summary="Liveness probe")
async def liveness() -> HealthResponse:
    """Returns 200 if the application process is alive."""
    from backend.app.core.config import settings
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
    )


@router.get("/ready", response_model=ReadinessResponse, summary="Readiness probe")
async def readiness() -> ReadinessResponse:
    """
    Returns 200 only if all downstream dependencies are reachable.
    Phase 2 will add real DB + Redis checks.
    """
    return ReadinessResponse(
        status="ok",
        database="not_checked",  # updated in Phase 2
        redis="not_checked",     # updated in Phase 2
    )
