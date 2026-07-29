"""Tests for rate_limit.py — sliding-window limiter."""
from __future__ import annotations

import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from app.core.rate_limit import _sliding_window_check, _store, rate_limit


def _clear_store():
    _store.clear()


# ── _sliding_window_check ─────────────────────────────────────────────────────

class TestSlidingWindowCheck:
    def setup_method(self):
        _clear_store()

    def test_allows_under_limit(self):
        remaining, _ = _sliding_window_check("k1", limit=5, window_seconds=60)
        assert remaining == 4

    def test_counts_requests(self):
        for _ in range(3):
            _sliding_window_check("k2", limit=5, window_seconds=60)
        remaining, _ = _sliding_window_check("k2", limit=5, window_seconds=60)
        assert remaining == 1

    def test_raises_429_at_limit(self):
        from fastapi import HTTPException
        for _ in range(5):
            _sliding_window_check("k3", limit=5, window_seconds=60)
        with pytest.raises(HTTPException) as exc_info:
            _sliding_window_check("k3", limit=5, window_seconds=60)
        assert exc_info.value.status_code == 429

    def test_different_keys_independent(self):
        for _ in range(5):
            _sliding_window_check("ka", limit=5, window_seconds=60)
        # Different key should still be allowed
        remaining, _ = _sliding_window_check("kb", limit=5, window_seconds=60)
        assert remaining == 4


# ── rate_limit dependency via TestClient ──────────────────────────────────────

def _make_app(limit: int = 3):
    app = FastAPI()

    @app.get("/limited", dependencies=[Depends(rate_limit(limit=limit, window_seconds=60))])
    async def limited():
        return {"ok": True}

    return app


class TestRateLimitDependency:
    def setup_method(self):
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
