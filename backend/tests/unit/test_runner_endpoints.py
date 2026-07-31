"""
test_runner_endpoints.py — Unit tests for SessionRunner API endpoints.
"""
from __future__ import annotations

import asyncio
import sys
import types
import pytest
from httpx import AsyncClient, ASGITransport
from uuid import uuid4


# 1. Create and register the MockSessionRunner in sys.modules BEFORE any backend imports
class MockSessionRunner:
    def __init__(self, session_id: str, **kwargs) -> None:
        self.session_id = session_id
        self._stop = asyncio.Event()

    async def __aenter__(self) -> MockSessionRunner:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def stop(self) -> None:
        self._stop.set()

    @property
    def running(self) -> bool:
        return not self._stop.is_set()

    async def run_until_stopped(self) -> None:
        try:
            await self._stop.wait()
        except asyncio.CancelledError:
            pass


# Stub out the ml.session_runner module completely
fake_sr_mod = types.ModuleType("ml.session_runner")
fake_sr_mod.SessionRunner = MockSessionRunner  # type: ignore[attr-defined]
sys.modules["ml.session_runner"] = fake_sr_mod

# Now import the app and runner endpoint internals
from app.main import app
from app.api.v1.endpoints.runner import _active_runners, _runner_objects


@pytest.fixture(autouse=True)
def clean_runner_state() -> None:
    _active_runners.clear()
    _runner_objects.clear()
    yield
    for task in list(_active_runners.values()):
        task.cancel()
    _active_runners.clear()
    _runner_objects.clear()


@pytest.mark.asyncio
async def test_runner_start_success() -> None:
    session_id = str(uuid4())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(f"/api/v1/runner/session/{session_id}/start")
        assert response.status_code == 202
        body = response.json()
        assert body["session_id"] == session_id
        assert body["running"] is True

        status_resp = await ac.get(f"/api/v1/runner/session/{session_id}/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["running"] is True


@pytest.mark.asyncio
async def test_runner_duplicate_start_returns_409() -> None:
    session_id = str(uuid4())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response1 = await ac.post(f"/api/v1/runner/session/{session_id}/start")
        assert response1.status_code == 202

        response2 = await ac.post(f"/api/v1/runner/session/{session_id}/start")
        assert response2.status_code == 409
        assert "already active" in response2.json()["detail"]


@pytest.mark.asyncio
async def test_runner_stop_success() -> None:
    session_id = str(uuid4())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        await ac.post(f"/api/v1/runner/session/{session_id}/start")
        await asyncio.sleep(0.01)

        response = await ac.post(f"/api/v1/runner/session/{session_id}/stop")
        assert response.status_code == 200
        body = response.json()
        assert body["session_id"] == session_id
        assert body["running"] is False

        status_resp = await ac.get(f"/api/v1/runner/session/{session_id}/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["running"] is False


@pytest.mark.asyncio
async def test_runner_stop_non_existent_runner_is_noop() -> None:
    session_id = str(uuid4())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(f"/api/v1/runner/session/{session_id}/stop")
        assert response.status_code == 200
        assert response.json()["running"] is False
