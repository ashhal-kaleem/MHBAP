"""
test_security_phase_c.py — Phase C security hardening tests.

Covers:
  - Token blacklist (blacklist_token / is_token_blacklisted / clear_blacklist)
  - Password strength validator on RegisterRequest
  - Account lockout (record_failed_login / check_account_lockout / clear_failed_logins)
  - POST /auth/logout endpoint
  - SecurityHeadersMiddleware
  - ContentSizeLimitMiddleware
"""
from __future__ import annotations

import time
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import Response
from starlette.testclient import TestClient as StarletteTestClient

from app.core.security import (
    blacklist_token,
    clear_blacklist,
    create_access_token,
    decode_access_token,
    is_token_blacklisted,
)
from app.core.rate_limit import (
    check_account_lockout,
    clear_failed_logins,
    record_failed_login,
    LOCKOUT_MAX_ATTEMPTS,
)
from app.core.security_headers import SecurityHeadersMiddleware
from app.core.content_size import ContentSizeLimitMiddleware


class FakeRedis:
    def __init__(self):
        self.store = {}
    async def set(self, key, value, ex=None):
        self.store[key] = (value, time.time() + (ex or 0))
    async def exists(self, key):
        if key in self.store:
            val, exp = self.store[key]
            if exp >= time.time():
                return 1
            else:
                del self.store[key]
        return 0
    async def keys(self, pattern):
        # Only handling blacklist:* pattern for tests
        prefix = pattern.replace('*', '')
        return [k for k in self.store.keys() if k.startswith(prefix)]
    async def delete(self, *keys):
        for k in keys:
            self.store.pop(k, None)

_fake_redis = FakeRedis()

# ── Token blacklist ────────────────────────────────────────────────────────────

class TestTokenBlacklist:
    @pytest.fixture(autouse=True)
    def patch_redis(self):
        with patch("app.core.security.get_redis", return_value=_fake_redis):
            yield

    @pytest.fixture(autouse=True)
    async def setup_teardown(self, patch_redis):
        await clear_blacklist()
        yield
        await clear_blacklist()

    @pytest.mark.asyncio
    async def test_fresh_jti_not_blacklisted(self):
        jti = str(uuid.uuid4())
        assert await is_token_blacklisted(jti) is False

    @pytest.mark.asyncio
    async def test_blacklisted_jti_is_detected(self):
        jti = str(uuid.uuid4())
        await blacklist_token(jti, time.time() + 3600)
        assert await is_token_blacklisted(jti) is True

    @pytest.mark.asyncio
    async def test_expired_entry_auto_evicted(self):
        jti = str(uuid.uuid4())
        await blacklist_token(jti, time.time() - 1)  # expired, redis sets ttl=1
        import asyncio
        await asyncio.sleep(1.1) # wait for redis to evict
        assert await is_token_blacklisted(jti) is False

    @pytest.mark.asyncio
    async def test_empty_jti_ignored(self):
        await blacklist_token("", time.time() + 3600)
        assert await is_token_blacklisted("") is False
        assert await is_token_blacklisted(None) is False

    @pytest.mark.asyncio
    async def test_clear_blacklist_wipes_all(self):
        jti = str(uuid.uuid4())
        await blacklist_token(jti, time.time() + 3600)
        await clear_blacklist()
        assert await is_token_blacklisted(jti) is False

    def test_token_includes_jti_claim(self):
        token = create_access_token("user1")
        payload = decode_access_token(token)
        assert "jti" in payload
        assert len(payload["jti"]) > 0


# ── Password strength ─────────────────────────────────────────────────────────

class TestPasswordStrength:
    def _make_client(self):
        from app.api.v1.endpoints.auth import router
        from app.api.dependencies import get_db

        def _noop_rl(limit=10, window_seconds=60):
            async def _dep(): return None
            return _dep

        app = FastAPI()
        app.include_router(router, prefix="/auth")

        async def _noop_db():
            yield AsyncMock()

        app.dependency_overrides[get_db] = _noop_db
        with patch("app.api.v1.endpoints.auth.rate_limit", side_effect=_noop_rl):
            return TestClient(app, raise_server_exceptions=True)

    def test_short_password_rejected(self):
        c = self._make_client()
        r = c.post("/auth/register", json={"email": "a@b.com", "password": "abc"})
        assert r.status_code == 422

    def test_all_alpha_no_digit_rejected(self):
        c = self._make_client()
        r = c.post("/auth/register", json={"email": "a@b.com", "password": "abcdefgh"})
        assert r.status_code == 422

    def test_digit_satisfies_requirement(self):
        c = self._make_client()
        with (
            patch("app.api.v1.endpoints.auth.get_user_by_email", new=AsyncMock(return_value=None)),
            patch("app.api.v1.endpoints.auth.create_user_with_password", new=AsyncMock(return_value=_fake_user())),
        ):
            r = c.post("/auth/register", json={"email": "a@b.com", "password": "password1"})
        assert r.status_code == 201

    def test_special_char_satisfies_requirement(self):
        c = self._make_client()
        with (
            patch("app.api.v1.endpoints.auth.get_user_by_email", new=AsyncMock(return_value=None)),
            patch("app.api.v1.endpoints.auth.create_user_with_password", new=AsyncMock(return_value=_fake_user())),
        ):
            r = c.post("/auth/register", json={"email": "a@b.com", "password": "password!"})
        assert r.status_code == 201


# ── Account lockout ───────────────────────────────────────────────────────────

class TestAccountLockout:
    def setup_method(self):
        clear_failed_logins("test@example.com")

    def test_no_lockout_initially(self):
        check_account_lockout("test@example.com")  # should not raise

    def test_lockout_after_max_attempts(self):
        from fastapi import HTTPException
        for _ in range(LOCKOUT_MAX_ATTEMPTS):
            record_failed_login("test@example.com")
        with pytest.raises(HTTPException) as exc_info:
            check_account_lockout("test@example.com")
        assert exc_info.value.status_code == 429

    def test_clear_resets_lockout(self):
        for _ in range(LOCKOUT_MAX_ATTEMPTS):
            record_failed_login("test@example.com")
        clear_failed_logins("test@example.com")
        check_account_lockout("test@example.com")  # should not raise


# ── Logout endpoint ────────────────────────────────────────────────────────────

class TestLogoutEndpoint:
    def _make_app_with_user(self, user_id: str) -> FastAPI:
        from app.api.v1.endpoints.auth import router
        from app.api.dependencies import get_db, get_current_user

        def _noop_rl(limit=10, window_seconds=60):
            async def _dep(): return None
            return _dep

        app = FastAPI()
        app.include_router(router, prefix="/auth")

        async def _noop_db():
            yield AsyncMock()

        app.dependency_overrides[get_db] = _noop_db
        app.dependency_overrides[get_current_user] = lambda: user_id
        return app

    @pytest.mark.asyncio
    async def test_logout_blacklists_token(self):
        with patch("app.core.security.get_redis", return_value=_fake_redis):
            await clear_blacklist()
            uid = str(uuid.uuid4())
            token = create_access_token(uid)
            payload = decode_access_token(token)
            jti = payload["jti"]

            with patch("app.api.v1.endpoints.auth.rate_limit",
                       side_effect=lambda **kw: (lambda: None)):
                app = self._make_app_with_user(uid)

            with TestClient(app) as c:
                r = c.post("/auth/logout", headers={"Authorization": f"Bearer {token}"})
            assert r.status_code == 204
            assert await is_token_blacklisted(jti) is True

    def test_logout_requires_auth(self):
        uid = str(uuid.uuid4())
        with patch("app.api.v1.endpoints.auth.rate_limit",
                   side_effect=lambda **kw: (lambda: None)):
            app = self._make_app_with_user(uid)
            # remove override so real auth runs
            from app.api.dependencies import get_current_user
            app.dependency_overrides.pop(get_current_user, None)

        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.post("/auth/logout")
        assert r.status_code == 401


# ── Security headers middleware ───────────────────────────────────────────────

class TestSecurityHeadersMiddleware:
    def _make_app(self, production: bool = False) -> FastAPI:
        app = FastAPI()
        app.add_middleware(SecurityHeadersMiddleware, production=production)

        @app.get("/ping")
        async def ping():
            return {"ok": True}

        return app

    def test_x_content_type_options(self):
        with TestClient(self._make_app()) as c:
            r = c.get("/ping")
        assert r.headers["x-content-type-options"] == "nosniff"

    def test_x_frame_options(self):
        with TestClient(self._make_app()) as c:
            r = c.get("/ping")
        assert r.headers["x-frame-options"] == "DENY"

    def test_referrer_policy(self):
        with TestClient(self._make_app()) as c:
            r = c.get("/ping")
        assert "referrer-policy" in r.headers

    def test_csp_present(self):
        with TestClient(self._make_app()) as c:
            r = c.get("/ping")
        assert "content-security-policy" in r.headers

    def test_hsts_absent_in_dev(self):
        with TestClient(self._make_app(production=False)) as c:
            r = c.get("/ping")
        assert "strict-transport-security" not in r.headers

    def test_hsts_present_in_production(self):
        with TestClient(self._make_app(production=True)) as c:
            r = c.get("/ping")
        assert "strict-transport-security" in r.headers


# ── Content size middleware ───────────────────────────────────────────────────

class TestContentSizeLimitMiddleware:
    def _make_app(self, max_bytes: int = 100) -> FastAPI:
        app = FastAPI()
        app.add_middleware(ContentSizeLimitMiddleware, max_bytes=max_bytes)

        @app.post("/upload")
        async def upload(request: Request):
            body = await request.body()
            return {"size": len(body)}

        return app

    def test_small_body_allowed(self):
        with TestClient(self._make_app(max_bytes=1000)) as c:
            r = c.post("/upload", content=b"hello", headers={"Content-Length": "5"})
        assert r.status_code == 200

    def test_oversized_body_rejected(self):
        with TestClient(self._make_app(max_bytes=10)) as c:
            r = c.post(
                "/upload",
                content=b"x" * 20,
                headers={"Content-Length": "20"},
            )
        assert r.status_code == 413

    def test_bad_content_length_rejected(self):
        with TestClient(self._make_app(max_bytes=100)) as c:
            r = c.post("/upload", content=b"hi", headers={"Content-Length": "notanumber"})
        assert r.status_code == 400

    def test_no_content_length_passes_through(self):
        with TestClient(self._make_app(max_bytes=10)) as c:
            # Starlette test client typically sets Content-Length; use chunked
            r = c.post("/upload", content=b"hi")
        # Either passes or not — just must not be 413
        assert r.status_code != 413


# ── helpers ──────────────────────────────────────────────────────────────────

def _fake_user(email: str = "a@b.com"):
    from unittest.mock import MagicMock
    u = MagicMock()
    u.id = uuid.uuid4()
    u.email = email
    u.display_name = "Test"
    u.role = "participant"
    u.is_active = True
    u.hashed_password = "hashed"
    return u
