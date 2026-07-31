"""
test_auth_endpoints.py — Unit tests for POST /auth/register, /login, /refresh, GET /me.

All DB and security calls are mocked; no live database or Redis required.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.endpoints.auth import router
from app.core.rate_limit import rate_limit


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_app() -> FastAPI:
    """Minimal FastAPI app with the auth router and a no-op DB override."""
    from app.api.dependencies import get_db

    app = FastAPI()
    app.include_router(router, prefix="/auth")

    async def _noop_db():
        db = AsyncMock()
        yield db

    app.dependency_overrides[get_db] = _noop_db
    return app


def _noop_rate_limit(limit=10, window_seconds=60):
    """Return a dependency that always allows the request through."""
    async def _dep():
        return None
    return _dep


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    with patch(
        "app.api.v1.endpoints.auth.rate_limit",
        side_effect=_noop_rate_limit,
    ):
        app = _make_app()
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


FAKE_UID = str(uuid.uuid4())


def _fake_user(email: str = "alice@example.com") -> MagicMock:
    u = MagicMock()
    u.id = uuid.UUID(FAKE_UID)
    u.email = email
    u.display_name = "Alice"
    u.role = "participant"
    u.is_active = True
    u.hashed_password = "hashed"
    return u


# ── register ─────────────────────────────────────────────────────────────────

class TestRegister:
    def test_success_returns_token(self, client):
        with (
            patch("app.api.v1.endpoints.auth.get_user_by_email", new=AsyncMock(return_value=None)),
            patch("app.api.v1.endpoints.auth.create_user_with_password", new=AsyncMock(return_value=_fake_user())),
            patch("app.api.v1.endpoints.auth.create_access_token", return_value="tok.abc.xyz"),
        ):
            r = client.post("/auth/register", json={"email": "alice@example.com", "password": "Secret1!"})
        assert r.status_code == 201
        body = r.json()
        assert body["access_token"] == "tok.abc.xyz"
        assert body["token_type"] == "bearer"
        assert body["user_id"] == FAKE_UID

    def test_conflict_on_duplicate_email(self, client):
        with patch(
            "app.api.v1.endpoints.auth.get_user_by_email",
            new=AsyncMock(return_value=_fake_user()),
        ):
            r = client.post("/auth/register", json={"email": "alice@example.com", "password": "Secret1!"})
        assert r.status_code == 409

    def test_missing_email_is_422(self, client):
        r = client.post("/auth/register", json={"password": "pw"})
        assert r.status_code == 422


# ── login ─────────────────────────────────────────────────────────────────────

class TestLogin:
    def test_success_returns_token(self, client):
        with (
            patch("app.api.v1.endpoints.auth.get_user_by_email", new=AsyncMock(return_value=_fake_user())),
            patch("app.api.v1.endpoints.auth.verify_password", return_value=True),
            patch("app.api.v1.endpoints.auth.create_access_token", return_value="tok.login"),
        ):
            r = client.post("/auth/login", json={"email": "alice@example.com", "password": "Secret1!"})
        assert r.status_code == 200
        assert r.json()["access_token"] == "tok.login"

    def test_wrong_password_returns_401(self, client):
        with (
            patch("app.api.v1.endpoints.auth.get_user_by_email", new=AsyncMock(return_value=_fake_user())),
            patch("app.api.v1.endpoints.auth.verify_password", return_value=False),
        ):
            r = client.post("/auth/login", json={"email": "alice@example.com", "password": "wrong"})
        assert r.status_code == 401

    def test_inactive_user_returns_401(self, client):
        inactive = _fake_user()
        inactive.is_active = False
        with patch("app.api.v1.endpoints.auth.get_user_by_email", new=AsyncMock(return_value=inactive)):
            r = client.post("/auth/login", json={"email": "alice@example.com", "password": "pw"})
        assert r.status_code == 401

    def test_unknown_email_returns_401(self, client):
        with patch("app.api.v1.endpoints.auth.get_user_by_email", new=AsyncMock(return_value=None)):
            r = client.post("/auth/login", json={"email": "nobody@example.com", "password": "pw"})
        assert r.status_code == 401
    def test_missing_email_is_422(self, client):
        r = client.post("/auth/login", json={"password": "pw"})
        assert r.status_code == 422


# ── refresh ───────────────────────────────────────────────────────────────────

class TestRefresh:
    def test_refresh_with_valid_token(self):
        """Refresh endpoint reads user_id from the JWT dependency — mock it."""
        from app.api.dependencies import get_current_user

        with patch(
            "app.api.v1.endpoints.auth.rate_limit",
            side_effect=_noop_rate_limit,
        ):
            app = _make_app()
        app.dependency_overrides[get_current_user] = lambda: FAKE_UID

        with (
            patch("app.api.v1.endpoints.auth.create_access_token", return_value="tok.refresh"),
            TestClient(app) as c,
        ):
            r = c.post("/auth/refresh", headers={"Authorization": "Bearer dummy"})
        assert r.status_code == 200
        assert r.json()["access_token"] == "tok.refresh"


# ── me ────────────────────────────────────────────────────────────────────────

class TestMe:
    def test_me_returns_user_profile(self):
        from app.api.dependencies import get_current_user

        with patch(
            "app.api.v1.endpoints.auth.rate_limit",
            side_effect=_noop_rate_limit,
        ):
            app = _make_app()
        app.dependency_overrides[get_current_user] = lambda: FAKE_UID

        with (
            patch("app.api.v1.endpoints.auth.get_user", new=AsyncMock(return_value=_fake_user())),
            TestClient(app) as c,
        ):
            r = c.get("/auth/me", headers={"Authorization": "Bearer dummy"})
        assert r.status_code == 200
        body = r.json()
        assert body["email"] == "alice@example.com"
        assert body["display_name"] == "Alice"
        assert body["role"] == "participant"

    def test_me_404_when_user_deleted(self):
        from app.api.dependencies import get_current_user

        with patch(
            "app.api.v1.endpoints.auth.rate_limit",
            side_effect=_noop_rate_limit,
        ):
            app = _make_app()
        app.dependency_overrides[get_current_user] = lambda: FAKE_UID

        with (
            patch("app.api.v1.endpoints.auth.get_user", new=AsyncMock(return_value=None)),
            TestClient(app) as c,
        ):
            r = c.get("/auth/me", headers={"Authorization": "Bearer dummy"})
        assert r.status_code == 404


# ── migration sanity ──────────────────────────────────────────────────────────

class TestMigration0002:
    def test_revision_attributes(self):
        from app.db.migrations.versions import (
            _0002_add_auth_fields_to_users as m,
        )
        assert m.revision == "0002"
        assert m.down_revision == "0001"

    def test_upgrade_callable(self):
        from app.db.migrations.versions import (
            _0002_add_auth_fields_to_users as m,
        )
        assert callable(m.upgrade)

    def test_downgrade_callable(self):
        from app.db.migrations.versions import (
            _0002_add_auth_fields_to_users as m,
        )
        assert callable(m.downgrade)
