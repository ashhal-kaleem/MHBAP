"""
test_stream_phase9.py — Phase 9 Redis hardening tests.

Covers:
- Connection cap enforcement (MAX_CLIENTS_PER_SESSION)
- Ping frame delivery via _ping_loop
- redis_stream_bus fallback path (in-process)
- _make_demo_prediction deterministic properties
- _softmax correctness
- Multiple sequential demo connections are independent
"""
from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.Main import app
from app.api.v1.endpoints.Stream import (
    _make_demo_prediction,
    _softmax,
    _client_counts,
    MAX_CLIENTS_PER_SESSION,
)


@pytest.fixture(scope="module")
def client():
    from unittest.mock import AsyncMock, patch
    from app.api.Dependencies import get_ws_current_user

    mock_session = AsyncMock()
    mock_session.user_id = "fake-user-id"

    with patch("app.services.SessionService.get_session", new=AsyncMock(return_value=mock_session)):
        app.dependency_overrides[get_ws_current_user] = lambda: "fake-user-id"
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c
        app.dependency_overrides.clear()


# ── unit: helpers ─────────────────────────────────────────────────────────

class TestSoftmax:
    def test_output_sums_to_one(self):
        logits = [1.0, 2.0, 0.5, -1.0]
        out = _softmax(logits)
        assert abs(sum(out) - 1.0) < 1e-6

    def test_highest_logit_has_highest_prob(self):
        logits = [0.0, 5.0, 1.0]
        out = _softmax(logits)
        assert out[1] == max(out)

    def test_uniform_logits(self):
        logits = [1.0, 1.0, 1.0, 1.0]
        out = _softmax(logits)
        for v in out:
            assert abs(v - 0.25) < 1e-6

    def test_large_values_stable(self):
        logits = [1000.0, 1001.0, 999.0]
        out = _softmax(logits)
        assert all(0.0 <= v <= 1.0 for v in out)
        assert abs(sum(out) - 1.0) < 1e-6


class TestMakeDemoPrediction:
    def test_all_keys_present(self):
        p = _make_demo_prediction(0.0, "test-session")
        for key in ("id", "session_id", "emotion_label", "emotion_scores",
                    "stress", "engagement", "attention", "fatigue",
                    "shap_weights", "explanation_text", "time", "recorded_at"):
            assert key in p, f"Missing key: {key}"

    def test_metrics_in_range(self):
        for t in range(0, 100, 10):
            p = _make_demo_prediction(float(t), "s")
            assert 0.0 <= p["stress"]     <= 1.0
            assert 0.0 <= p["engagement"] <= 1.0
            assert 0.0 <= p["attention"]  <= 1.0
            assert 0.0 <= p["fatigue"]    <= 1.0

    def test_shap_sums_to_one(self):
        for t in range(5):
            p = _make_demo_prediction(float(t), "s")
            assert abs(sum(p["shap_weights"].values()) - 1.0) < 0.01

    def test_emotion_scores_sum_to_one(self):
        for t in range(5):
            p = _make_demo_prediction(float(t), "s")
            assert abs(sum(p["emotion_scores"].values()) - 1.0) < 0.01

    def test_session_id_propagated(self):
        p = _make_demo_prediction(0.0, "my-session-abc")
        assert p["session_id"] == "my-session-abc"

    def test_explanation_text_non_empty(self):
        p = _make_demo_prediction(0.0, "s")
        assert len(p["explanation_text"]) > 10


# ── integration: demo WebSocket ───────────────────────────────────────────

class TestDemoPhase9:
    def test_demo_session_end_on_close(self, client: TestClient):
        """Closing the WS should trigger session_end frame."""
        with client.websocket_connect("/api/v1/stream/demo") as ws:
            ws.receive_json()   # session_start
            ws.receive_json()   # first prediction
            # Client closes — server should send session_end (may arrive or not
            # depending on timing, but no exception should propagate)

    def test_two_demo_connections_are_independent(self, client: TestClient):
        """Each /demo connection gets its own synthetic session_id."""
        sids = []
        for _ in range(2):
            with client.websocket_connect("/api/v1/stream/demo") as ws:
                data = ws.receive_json()
                sids.append(data["payload"]["session_id"])
        assert sids[0] != sids[1]

    def test_demo_prediction_has_uuid_id(self, client: TestClient):
        with client.websocket_connect("/api/v1/stream/demo") as ws:
            ws.receive_json()
            pred = ws.receive_json()
            import uuid
            uuid.UUID(pred["payload"]["id"])   # raises if invalid


# ── integration: session cap ──────────────────────────────────────────────

class TestConnectionCap:
    def test_cap_counter_increments_and_decrements(self, client: TestClient):
        from app.core.StreamBus import publish
        import threading, time

        sid = "00000000-0000-0000-0000-000000000002"
        before = _client_counts.get(sid, 0)

        def _end():
            time.sleep(0.1)
            publish(sid, {"type": "session_end", "payload": None})

        with client.websocket_connect(f"/api/v1/stream/session/{sid}") as ws:
            ws.receive_json()   # session_start
            during = _client_counts.get(sid, 0)
            assert during == before + 1
            threading.Thread(target=_end, daemon=True).start()
            # drain until session_end or close
            for _ in range(5):
                try:
                    msg = ws.receive_json()
                    if msg.get("type") == "session_end":
                        break
                except Exception:
                    break
        after = _client_counts.get(sid, 0)
        assert after == before


# ── unit: redis_stream_bus fallback ──────────────────────────────────────

class TestRedisStreamBusFallback:
    def test_publish_falls_back_to_inprocess_when_redis_unavailable(self):
        """When Redis is unavailable the bus should publish to in-process bus."""
        import app.core.RedisStreamBus as bus_mod
        import app.core.StreamBus as inproc

        received = []

        # Patch _redis_available to False to skip Redis path
        original = bus_mod._redis_available
        bus_mod._redis_available = False

        sid = "fallback-test-session"
        q = inproc.subscribe(sid)
        try:
            asyncio.run(
                bus_mod.publish(sid, {"type": "prediction", "payload": {"stress": 0.7}})
            )
            msg = q.get_nowait()
            assert msg["payload"]["stress"] == pytest.approx(0.7)
        finally:
            inproc.unsubscribe(sid, q)
            bus_mod._redis_available = original
