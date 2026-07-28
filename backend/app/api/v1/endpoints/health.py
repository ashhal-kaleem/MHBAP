"""
Health check endpoints.
GET /api/v1/health/       — liveness probe
GET /api/v1/health/ready  — readiness probe (checks DB + Redis in Phase 2)
"""
from fastapi import APIRouter, Response
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
async def readiness(response: Response) -> ReadinessResponse:
    """Returns 200 only if all downstream dependencies are reachable, else 503."""
    from backend.app.core.redis import check_redis_connection
    from backend.app.db.session import check_db_connection

    db_ok = await check_db_connection()
    redis_ok = await check_redis_connection()
    if not (db_ok and redis_ok):
        response.status_code = 503

    return ReadinessResponse(
        status="ok" if (db_ok and redis_ok) else "degraded",
        database="ok" if db_ok else "unreachable",
        redis="ok" if redis_ok else "unreachable",
    )
