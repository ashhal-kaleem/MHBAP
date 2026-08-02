"""
test_runner_endpoints.py — Unit tests for SessionRunner API endpoints.

Covers:
  - C-05: Authentication (401 for missing/invalid/revoked tokens)
  - C-07: Role-based authorization (403 for disallowed roles)
  - Functional: start/stop/status/duplicate/noop with authorized user
  - Ownership: 403 when an authorized role tries to manage another user's session
"""
from __future__ import annotations

import asyncio
import sys
import types
import time
import uuid
from typing import Tuple
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ─── 1. Stub ml.session_runner BEFORE any backend imports ───────────────────
class MockSessionRunner:
    def __init__(self, session_id: str, **kwargs) -> None:
        self.session_id = session_id
        self._stop = asyncio.Event()

    async def __aenter__(self) -> "MockSessionRunner":
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


fake_sr_mod = types.ModuleType("ml.session_runner")
fake_sr_mod.SessionRunner = MockSessionRunner  # type: ignore[attr-defined]
sys.modules["ml.session_runner"] = fake_sr_mod

# ─── 2. Import app components ────────────────────────────────────────────────
from app.api.v1.endpoints.runner import _active_runners, _runner_objects, router as runner_router
from app.api.dependencies import (
    get_db,
    get_current_user_with_role,
    require_roles,
    RUNNER_ALLOWED_ROLES,
)
from app.core.security import create_access_token, decode_access_token, blacklist_token, clear_blacklist
from tests.unit.fake_redis import FakeRedis

_fake_redis = FakeRedis()


# ─── 3. Helpers ─────────────────────────────────────────────────────────────

def _make_app(
    owner_id: str,
    role: str = "participant",
    mock_session_owner_id: str | None = None,
) -> FastAPI:
    """
    Build a minimal FastAPI app with the runner router.

    - Overrides get_db with a no-op async generator.
    - Overrides get_current_user_with_role to return (owner_id, role).
    - Patches session_service.get_session so ownership checks work.
    """
    app = FastAPI()
    app.include_router(runner_router, prefix="/runner")

    async def _noop_db():
        yield AsyncMock()

    app.dependency_overrides[get_db] = _noop_db

    effective_owner = mock_session_owner_id or owner_id

    async def _fake_auth() -> Tuple[str, str]:
        return owner_id, role

    app.dependency_overrides[get_current_user_with_role] = _fake_auth

    mock_session = MagicMock()
    mock_session.user_id = uuid.UUID(effective_owner)

    import app.services.session_service as _ss
    _ss.get_session = AsyncMock(return_value=mock_session)  # type: ignore[assignment]

    return app


@pytest.fixture(autouse=True)
def clean_runner_state() -> None:
    _active_runners.clear()
    _runner_objects.clear()
    yield
    for task in list(_active_runners.values()):
        task.cancel()
    _active_runners.clear()
    _runner_objects.clear()


@pytest.fixture(autouse=True)
def patch_redis():
    with patch("app.core.security.get_redis", return_value=_fake_redis):
        yield


@pytest.fixture(autouse=True)
async def clear_redis(patch_redis):
    _fake_redis.clear()
    await clear_blacklist()
    yield
    _fake_redis.clear()


# ─── 4. Authentication tests (C-05) ─────────────────────────────────────────

class TestRunnerAuthentication:
    """
    Endpoints must return 401 for missing/invalid/revoked tokens.
    These tests use the real auth dependency (no override).
    """

    def _make_bare_app(self) -> FastAPI:
        """App with runner router but NO dependency overrides → real auth runs."""
        from app.api.v1.endpoints.runner import router as r
        from app.api.dependencies import get_db as _gdb

        app = FastAPI()
        app.include_router(r, prefix="/runner")

        async def _noop_db():
            yield AsyncMock()

        app.dependency_overrides[_gdb] = _noop_db
        return app

    def test_unauthenticated_start_returns_401(self):
        with TestClient(self._make_bare_app(), raise_server_exceptions=False) as c:
            r = c.post(f"/runner/session/{uuid.uuid4()}/start")
        assert r.status_code == 401

    def test_unauthenticated_stop_returns_401(self):
        with TestClient(self._make_bare_app(), raise_server_exceptions=False) as c:
            r = c.post(f"/runner/session/{uuid.uuid4()}/stop")
        assert r.status_code == 401

    def test_unauthenticated_status_returns_401(self):
        with TestClient(self._make_bare_app(), raise_server_exceptions=False) as c:
            r = c.get(f"/runner/session/{uuid.uuid4()}/status")
        assert r.status_code == 401

    def test_invalid_token_returns_401(self):
        with TestClient(self._make_bare_app(), raise_server_exceptions=False) as c:
            r = c.post(
                f"/runner/session/{uuid.uuid4()}/start",
                headers={"Authorization": "Bearer this.is.garbage"},
            )
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_revoked_token_returns_401(self):
        token = create_access_token(str(uuid.uuid4()))
        payload = decode_access_token(token)
        await blacklist_token(payload["jti"], time.time() + 3600)

        with TestClient(self._make_bare_app(), raise_server_exceptions=False) as c:
            r = c.post(
                f"/runner/session/{uuid.uuid4()}/start",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert r.status_code == 401


# ─── 5. Role-based authorization tests (C-07) ───────────────────────────────

class TestRunnerRoleAuthorization:
    """
    Endpoints must return 403 for authenticated users with disallowed roles.
    Allowed roles: participant, educator, researcher.
    """

    def _app_with_role(self, role: str) -> FastAPI:
        return _make_app(owner_id=str(uuid.uuid4()), role=role)

    def test_participant_can_start_runner(self):
        owner_id = str(uuid.uuid4())
        app = _make_app(owner_id=owner_id, role="participant")
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.post(f"/runner/session/{uuid.uuid4()}/start")
        assert r.status_code == 202

    def test_educator_can_start_runner(self):
        owner_id = str(uuid.uuid4())
        app = _make_app(owner_id=owner_id, role="educator")
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.post(f"/runner/session/{uuid.uuid4()}/start")
        assert r.status_code == 202

    def test_researcher_can_start_runner(self):
        owner_id = str(uuid.uuid4())
        app = _make_app(owner_id=owner_id, role="researcher")
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.post(f"/runner/session/{uuid.uuid4()}/start")
        assert r.status_code == 202

    def test_unknown_role_start_returns_403(self):
        app = self._app_with_role("guest")
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.post(f"/runner/session/{uuid.uuid4()}/start")
        assert r.status_code == 403

    def test_unknown_role_stop_returns_403(self):
        app = self._app_with_role("admin_external")
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.post(f"/runner/session/{uuid.uuid4()}/stop")
        assert r.status_code == 403

    def test_unknown_role_status_returns_403(self):
        app = self._app_with_role("viewer")
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get(f"/runner/session/{uuid.uuid4()}/status")
        assert r.status_code == 403

    def test_empty_role_returns_403(self):
        app = self._app_with_role("")
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.post(f"/runner/session/{uuid.uuid4()}/start")
        assert r.status_code == 403


# ─── 6. Ownership tests ───────────────────────────────────────────────────────

class TestRunnerOwnership:
    """Authorized roles must still only manage their own sessions (HTTP 403)."""

    def test_non_owner_start_returns_403(self):
        owner_id = str(uuid.uuid4())
        other_id = str(uuid.uuid4())
        # other_id is authenticated but session belongs to owner_id
        app = _make_app(owner_id=other_id, role="participant", mock_session_owner_id=owner_id)
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.post(f"/runner/session/{uuid.uuid4()}/start")
        assert r.status_code == 403

    def test_non_owner_stop_returns_403(self):
        owner_id = str(uuid.uuid4())
        other_id = str(uuid.uuid4())
        app = _make_app(owner_id=other_id, role="participant", mock_session_owner_id=owner_id)
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.post(f"/runner/session/{uuid.uuid4()}/stop")
        assert r.status_code == 403

    def test_non_owner_status_returns_403(self):
        owner_id = str(uuid.uuid4())
        other_id = str(uuid.uuid4())
        app = _make_app(owner_id=other_id, role="participant", mock_session_owner_id=owner_id)
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get(f"/runner/session/{uuid.uuid4()}/status")
        assert r.status_code == 403

    def test_researcher_non_owner_returns_403(self):
        """Researcher cannot manage another user's session."""
        owner_id = str(uuid.uuid4())
        researcher_id = str(uuid.uuid4())
        app = _make_app(owner_id=researcher_id, role="researcher", mock_session_owner_id=owner_id)
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.post(f"/runner/session/{uuid.uuid4()}/start")
        assert r.status_code == 403


# ─── 7. Functional tests (authorized owner) ──────────────────────────────────

class TestRunnerFunctional:
    """Full start/stop/status lifecycle for an authenticated owner."""

    def _app(self, owner_id: str, role: str = "participant") -> FastAPI:
        return _make_app(owner_id=owner_id, role=role)

    @pytest.mark.asyncio
    async def test_start_returns_202_and_running(self):
        owner_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        app = self._app(owner_id)
        with TestClient(app, raise_server_exceptions=True) as c:
            r = c.post(f"/runner/session/{session_id}/start")
        assert r.status_code == 202
        body = r.json()
        assert body["session_id"] == session_id
        assert body["running"] is True

    @pytest.mark.asyncio
    async def test_status_reflects_running_task(self):
        owner_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        app = self._app(owner_id)
        with TestClient(app, raise_server_exceptions=True) as c:
            c.post(f"/runner/session/{session_id}/start")
            r = c.get(f"/runner/session/{session_id}/status")
        assert r.status_code == 200
        assert r.json()["running"] is True

    @pytest.mark.asyncio
    async def test_duplicate_start_returns_409(self):
        owner_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        app = self._app(owner_id)
        with TestClient(app, raise_server_exceptions=False) as c:
            r1 = c.post(f"/runner/session/{session_id}/start")
            assert r1.status_code == 202
            r2 = c.post(f"/runner/session/{session_id}/start")
        assert r2.status_code == 409
        assert "already active" in r2.json()["detail"]

    @pytest.mark.asyncio
    async def test_stop_returns_200_and_not_running(self):
        owner_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        app = self._app(owner_id)
        with TestClient(app, raise_server_exceptions=True) as c:
            c.post(f"/runner/session/{session_id}/start")
            r = c.post(f"/runner/session/{session_id}/stop")
        assert r.status_code == 200
        assert r.json()["running"] is False

    @pytest.mark.asyncio
    async def test_stop_non_existent_runner_is_noop(self):
        owner_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        app = self._app(owner_id)
        with TestClient(app, raise_server_exceptions=True) as c:
            r = c.post(f"/runner/session/{session_id}/stop")
        assert r.status_code == 200
        assert r.json()["running"] is False

    @pytest.mark.asyncio
    async def test_educator_can_run_own_session(self):
        owner_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        app = self._app(owner_id, role="educator")
        with TestClient(app, raise_server_exceptions=True) as c:
            r = c.post(f"/runner/session/{session_id}/start")
        assert r.status_code == 202

    @pytest.mark.asyncio
    async def test_researcher_can_run_own_session(self):
        owner_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        app = self._app(owner_id, role="researcher")
        with TestClient(app, raise_server_exceptions=True) as c:
            r = c.post(f"/runner/session/{session_id}/start")
        assert r.status_code == 202
