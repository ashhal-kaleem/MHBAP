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


from tests.unit.fake_redis import FakeRedis

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
    @pytest.fixture(autouse=True)
    def patch_redis(self):
        with patch("app.core.rate_limit.get_redis", return_value=_fake_redis):
            yield

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
    @pytest.fixture(autouse=True)
    def patch_redis(self):
        with patch("app.core.rate_limit.get_redis", return_value=_fake_redis):
            yield

    @pytest.fixture(autouse=True)
    async def setup_test(self):
        _fake_redis.clear()
        await clear_failed_logins("test@example.com")

    @pytest.mark.asyncio
    async def test_no_lockout_initially(self):
        await check_account_lockout("test@example.com")  # should not raise

    @pytest.mark.asyncio
    async def test_lockout_after_max_attempts(self):
        from fastapi import HTTPException
        for _ in range(LOCKOUT_MAX_ATTEMPTS):
            await record_failed_login("test@example.com")
        with pytest.raises(HTTPException) as exc_info:
            await check_account_lockout("test@example.com")
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_clear_resets_lockout(self):
        for _ in range(LOCKOUT_MAX_ATTEMPTS):
            await record_failed_login("test@example.com")
        await clear_failed_logins("test@example.com")
        await check_account_lockout("test@example.com")  # should not raise


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


# ── WebSocket Authentication ───────────────────────────────────────────────────

class TestWebSocketAuth:
    def _make_app(self) -> FastAPI:
        from app.api.v1.endpoints.stream import router
        app = FastAPI()
        app.include_router(router, prefix="/stream")
        return app

    @pytest.fixture(autouse=True)
    def patch_redis(self):
        with patch("app.core.security.get_redis", return_value=_fake_redis):
            yield

    @pytest.fixture(autouse=True)
    def patch_redis_bus(self):
        # Prevent redis_stream_bus from making real connections in stream.py
        from contextlib import asynccontextmanager
        @asynccontextmanager
        async def fake_subscribe(*args, **kwargs):
            class FakeQueue:
                async def get(self):
                    import asyncio
                    await asyncio.sleep(0.1)
                    return {"type": "session_end"}
            yield FakeQueue()
            
        with patch("app.api.v1.endpoints.stream.redis_subscribe", new=fake_subscribe):
            yield

    @pytest.fixture(autouse=True)
    async def setup_test(self, patch_redis):
        await clear_blacklist()
        yield
        await clear_blacklist()

    def test_missing_token_rejected(self):
        with TestClient(self._make_app()) as c:
            from fastapi import WebSocketDisconnect
            with pytest.raises(WebSocketDisconnect) as exc:
                with c.websocket_connect("/stream/demo"):
                    pass
            assert exc.value.code == 1008

    def test_invalid_token_rejected(self):
        with TestClient(self._make_app()) as c:
            from fastapi import WebSocketDisconnect
            with pytest.raises(WebSocketDisconnect) as exc:
                with c.websocket_connect("/stream/demo?access_token=invalid_token"):
                    pass
            assert exc.value.code == 1008

    @pytest.mark.asyncio
    async def test_revoked_token_rejected(self):
        token = create_access_token(str(uuid.uuid4()))
        payload = decode_access_token(token)
        await blacklist_token(payload["jti"], time.time() + 3600)
        
        with TestClient(self._make_app()) as c:
            from fastapi import WebSocketDisconnect
            with pytest.raises(WebSocketDisconnect) as exc:
                with c.websocket_connect(f"/stream/demo?access_token={token}"):
                    pass
            assert exc.value.code == 1008

    def test_valid_token_query_accepted(self):
        token = create_access_token(str(uuid.uuid4()))
        with TestClient(self._make_app()) as c:
            with c.websocket_connect(f"/stream/demo?access_token={token}") as ws:
                data = ws.receive_json()
                assert data["type"] == "session_start"

    def test_valid_token_header_accepted(self):
        token = create_access_token(str(uuid.uuid4()))
        with TestClient(self._make_app()) as c:
            with c.websocket_connect("/stream/demo", headers={"Authorization": f"Bearer {token}"}) as ws:
                data = ws.receive_json()
                assert data["type"] == "session_start"

    def test_expired_token_rejected(self):
        with patch("app.core.security.ACCESS_TOKEN_EXPIRE_MINUTES", -1):
            token = create_access_token(str(uuid.uuid4()))
            
        with TestClient(self._make_app()) as c:
            from fastapi import WebSocketDisconnect
            with pytest.raises(WebSocketDisconnect) as exc:
                with c.websocket_connect(f"/stream/demo?access_token={token}"):
                    pass
            assert exc.value.code == 1008

# ── C-05 & C-06 Authentication and Authorization ───────────────────────────────────────────

class TestResourceOwnership:
    @pytest.fixture(autouse=True)
    def patch_redis(self):
        """Patch Redis so get_current_user's is_token_blacklisted doesn't hit real Redis."""
        with patch("app.core.security.get_redis", return_value=_fake_redis):
            yield

    @pytest.fixture(autouse=True)
    async def clear_redis(self, patch_redis):
        _fake_redis.clear()
        await clear_blacklist()
        yield
        _fake_redis.clear()

    def _make_app(self) -> FastAPI:
        from app.api.v1.endpoints.sessions import router as sessions_router
        from app.api.v1.endpoints.predictions import router as predictions_router
        from app.api.v1.endpoints.analytics import router as analytics_router
        from app.api.v1.endpoints.runner import router as runner_router
        from app.api.v1.endpoints.evaluation import router as evaluation_router
        from app.api.dependencies import get_db

        app = FastAPI()
        app.include_router(sessions_router, prefix="/sessions")
        app.include_router(predictions_router, prefix="/predictions")
        app.include_router(analytics_router, prefix="/analytics")
        app.include_router(runner_router, prefix="/runner")
        app.include_router(evaluation_router, prefix="/evaluation")

        async def _noop_db():
            yield AsyncMock()

        app.dependency_overrides[get_db] = _noop_db
        return app

    def _get_token(self, user_id: str) -> str:
        return create_access_token(user_id)

    def test_unauthenticated_request_rejected(self):
        with TestClient(self._make_app(), raise_server_exceptions=False) as c:
            r = c.get(f"/sessions/user/{uuid.uuid4()}")
        assert r.status_code == 401

    def test_invalid_token_rejected(self):
        with TestClient(self._make_app(), raise_server_exceptions=False) as c:
            r = c.get(f"/sessions/user/{uuid.uuid4()}", headers={"Authorization": "Bearer invalid"})
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_revoked_token_rejected(self):
        token = self._get_token(str(uuid.uuid4()))
        payload = decode_access_token(token)
        await blacklist_token(payload["jti"], time.time() + 3600)

        with TestClient(self._make_app(), raise_server_exceptions=False) as c:
            r = c.get(f"/sessions/user/{uuid.uuid4()}", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 401

    def test_analytics_owner_access(self):
        owner_id = str(uuid.uuid4())
        token = self._get_token(owner_id)
        with patch("app.services.analytics_service.get_user_analytics", new=AsyncMock(return_value={"test": "data"})):
            with TestClient(self._make_app(), raise_server_exceptions=False) as c:
                r = c.get(f"/analytics/user/{owner_id}", headers={"Authorization": f"Bearer {token}"})
            # Expected to pass auth, maybe fail validation because return_value is dict instead of UserAnalytics, but status shouldn't be 403 or 401
            assert r.status_code not in (401, 403)

    def test_analytics_non_owner_access(self):
        owner_id = str(uuid.uuid4())
        other_id = str(uuid.uuid4())
        token = self._get_token(other_id)
        with TestClient(self._make_app(), raise_server_exceptions=False) as c:
            r = c.get(f"/analytics/user/{owner_id}", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 403

    def test_session_owner_access(self):
        owner_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        token = self._get_token(owner_id)

        mock_session = AsyncMock()
        mock_session.user_id = uuid.UUID(owner_id)
        
        with patch("app.services.session_service.get_session", new=AsyncMock(return_value=mock_session)):
            with TestClient(self._make_app(), raise_server_exceptions=False) as c:
                r = c.get(f"/sessions/{session_id}", headers={"Authorization": f"Bearer {token}"})
            assert r.status_code not in (401, 403)

    def test_session_non_owner_access(self):
        owner_id = str(uuid.uuid4())
        other_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        token = self._get_token(other_id)

        mock_session = AsyncMock()
        mock_session.user_id = uuid.UUID(owner_id)
        
        with patch("app.services.session_service.get_session", new=AsyncMock(return_value=mock_session)):
            with TestClient(self._make_app(), raise_server_exceptions=False) as c:
                r = c.get(f"/sessions/{session_id}", headers={"Authorization": f"Bearer {token}"})
            assert r.status_code == 403

    def test_session_delete_non_owner(self):
        owner_id = str(uuid.uuid4())
        other_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        token = self._get_token(other_id)

        mock_session = AsyncMock()
        mock_session.user_id = uuid.UUID(owner_id)
        
        with patch("app.services.session_service.get_session", new=AsyncMock(return_value=mock_session)):
            with TestClient(self._make_app(), raise_server_exceptions=False) as c:
                r = c.delete(f"/sessions/{session_id}", headers={"Authorization": f"Bearer {token}"})
            assert r.status_code == 403

    def test_runner_non_owner(self):
        owner_id = str(uuid.uuid4())
        other_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())

        mock_session = AsyncMock()
        mock_session.user_id = uuid.UUID(owner_id)

        # Runner now uses require_roles → get_current_user_with_role (needs DB lookup).
        # Override it directly so the test stays a pure unit test.
        from app.api.dependencies import get_current_user_with_role as _gcuwr
        app = self._make_app()
        app.dependency_overrides[_gcuwr] = lambda: (other_id, "participant")

        with patch("app.services.session_service.get_session", new=AsyncMock(return_value=mock_session)):
            with TestClient(app, raise_server_exceptions=False) as c:
                r = c.post(f"/runner/session/{session_id}/start")
        assert r.status_code == 403

    def test_evaluation_authenticated(self):
        token = self._get_token(str(uuid.uuid4()))
        with patch("app.api.v1.endpoints.evaluation.run_benchmark", new=lambda **kw: []):
            with TestClient(self._make_app(), raise_server_exceptions=False) as c:
                r = c.get("/evaluation/benchmark", headers={"Authorization": f"Bearer {token}"})
            assert r.status_code not in (401, 403)
