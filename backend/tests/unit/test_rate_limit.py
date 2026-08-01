"""Tests for rate_limit.py — sliding-window limiter."""
from __future__ import annotations

import pytest
from unittest.mock import patch
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from app.core.rate_limit import _sliding_window_check, rate_limit
from tests.unit.fake_redis import FakeRedis

_fake_redis = FakeRedis()

@pytest.fixture(autouse=True)
def patch_redis():
    with patch("app.core.rate_limit.get_redis", return_value=_fake_redis):
        yield

def _clear_store():
    _fake_redis.clear()


# ── _sliding_window_check ─────────────────────────────────────────────────────

class TestSlidingWindowCheck:
    @pytest.fixture(autouse=True)
    def setup_test(self):
        _clear_store()

    @pytest.mark.asyncio
    async def test_allows_under_limit(self):
        remaining, _ = await _sliding_window_check("k1", limit=5, window_seconds=60)
        assert remaining == 4

    @pytest.mark.asyncio
    async def test_counts_requests(self):
        for _ in range(3):
            await _sliding_window_check("k2", limit=5, window_seconds=60)
        remaining, _ = await _sliding_window_check("k2", limit=5, window_seconds=60)
        assert remaining == 1

    @pytest.mark.asyncio
    async def test_raises_429_at_limit(self):
        from fastapi import HTTPException
        for _ in range(5):
            await _sliding_window_check("k3", limit=5, window_seconds=60)
        with pytest.raises(HTTPException) as exc_info:
            await _sliding_window_check("k3", limit=5, window_seconds=60)
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_different_keys_independent(self):
        for _ in range(5):
            await _sliding_window_check("ka", limit=5, window_seconds=60)
        # Different key should still be allowed
        remaining, _ = await _sliding_window_check("kb", limit=5, window_seconds=60)
        assert remaining == 4


# ── rate_limit dependency via TestClient ──────────────────────────────────────

def _make_app(limit: int = 3):
    app = FastAPI()

    @app.get("/limited", dependencies=[Depends(rate_limit(limit=limit, window_seconds=60))])
    async def limited():
        return {"ok": True}

    return app


class TestRateLimitDependency:
    @pytest.fixture(autouse=True)
    def setup_test(self):
        _clear_store()

    def test_allows_up_to_limit(self):
        client = TestClient(_make_app(limit=3))
        for _ in range(3):
            r = client.get("/limited")
            assert r.status_code == 200

    def test_blocks_beyond_limit(self):
        client = TestClient(_make_app(limit=2))
        client.get("/limited")
        client.get("/limited")
        r = client.get("/limited")
        assert r.status_code == 429

    def test_retry_after_header_present(self):
        client = TestClient(_make_app(limit=1))
        client.get("/limited")
        r = client.get("/limited")
        assert "Retry-After" in r.headers
