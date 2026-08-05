"""
Unit tests for health endpoints.
These run without any external services (no DB, no Redis).
"""
import pytest
from httpx import AsyncClient, ASGITransport

from app.Main import app


@pytest.mark.asyncio
async def test_liveness_returns_ok() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/v1/health/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "environment" in body


@pytest.mark.asyncio
async def test_readiness_reports_dependency_shape() -> None:
    """
    Readiness performs REAL DB + Redis checks (Phase 2). Locally, without
    docker-compose services running, this legitimately returns 503/degraded —
    that's correct behaviour, not a test failure. CI runs Postgres + Redis
    as services (see .github/workflows/ci.yml) so it exercises the "ok" path.
    We assert the response *shape* here; the full-stack "ok" path is covered
    by tests/integration/test_health_integration.py, which skips without a DB.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/v1/health/ready")
    assert response.status_code in (200, 503)
    body = response.json()
    assert body["status"] in ("ok", "degraded")
    assert body["database"] in ("ok", "unreachable")
    assert body["redis"] in ("ok", "unreachable")
