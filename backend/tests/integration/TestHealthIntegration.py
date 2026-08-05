"""
Integration tests — require a real Postgres/TimescaleDB + Redis
(docker-compose up db redis, or CI services). Auto-skip otherwise
so `pytest` still passes on a laptop with no services running.
"""
from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.db.Session import check_db_connection
from app.Main import app


async def _db_available() -> bool:
    return await check_db_connection()


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_readiness_is_ok_with_live_services(client: AsyncClient) -> None:
    if not await _db_available():
        pytest.skip("Postgres not reachable — start `docker compose up db redis` to run this")

    response = await client.get("/api/v1/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok", "database": "ok", "redis": "ok"}
