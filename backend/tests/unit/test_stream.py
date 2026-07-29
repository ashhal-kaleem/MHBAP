"""
test_stream.py — WebSocket endpoint tests (Phase 6).

Uses httpx + anyio (built into FastAPI's test client stack).
No DB, no Redis, no hardware required.
"""
from __future__ import annotations

import json
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


class TestDemoStream:
    def test_demo_connects_and_receives_session_start(self, client: TestClient):
        with client.websocket_connect("/api/v1/stream/demo") as ws:
            data = ws.receive_json()
            assert data["type"] == "session_start"
            assert "session_id" in data["payload"]

    def test_demo_sends_prediction(self, client: TestClient):
        with client.websocket_connect("/api/v1/stream/demo") as ws:
            _start = ws.receive_json()           # session_start
            pred_msg = ws.receive_json()         # first prediction
            assert pred_msg["type"] == "prediction"
            p = pred_msg["payload"]
            assert "emotion_label" in p
            assert "stress" in p
            assert "engagement" in p
            assert "attention" in p
            assert "fatigue" in p
            assert "shap_weights" in p
            assert "explanation_text" in p

    def test_demo_prediction_ranges(self, client: TestClient):
        with client.websocket_connect("/api/v1/stream/demo") as ws:
            ws.receive_json()                    # session_start
            pred_msg = ws.receive_json()
            p = pred_msg["payload"]
            assert 0.0 <= p["stress"]     <= 1.0
            assert 0.0 <= p["engagement"] <= 1.0
            assert 0.0 <= p["attention"]  <= 1.0
            assert 0.0 <= p["fatigue"]    <= 1.0

    def test_demo_emotion_scores_sum_to_one(self, client: TestClient):
        with client.websocket_connect("/api/v1/stream/demo") as ws:
            ws.receive_json()
            pred_msg = ws.receive_json()
            scores = pred_msg["payload"]["emotion_scores"]
            assert abs(sum(scores.values()) - 1.0) < 0.01

    def test_demo_shap_weights_sum_to_one(self, client: TestClient):
        with client.websocket_connect("/api/v1/stream/demo") as ws:
            ws.receive_json()
            pred_msg = ws.receive_json()
            shap = pred_msg["payload"]["shap_weights"]
            assert abs(sum(shap.values()) - 1.0) < 0.01

    def test_demo_sends_multiple_predictions(self, client: TestClient):
        with client.websocket_connect("/api/v1/stream/demo") as ws:
            ws.receive_json()                    # session_start
            msgs = [ws.receive_json() for _ in range(3)]
            assert all(m["type"] == "prediction" for m in msgs)


class TestSessionStream:
    def test_session_stream_connects(self, client: TestClient):
        sid = "11111111-1111-1111-1111-111111111111"
        with client.websocket_connect(f"/api/v1/stream/session/{sid}") as ws:
            data = ws.receive_json()
            assert data["type"] == "session_start"
            assert data["payload"]["session_id"] == sid

    def test_stream_bus_publish_reaches_client(self, client: TestClient):
        """
        Verify session_start is received and WS closes cleanly.
        End-to-end publish-through-queue is covered by TestRedisStreamBusFallback
        in test_stream_phase9.py (unit, no WS involved).
        """
        sid = "22222222-2222-2222-2222-222222222222"
        with client.websocket_connect(f"/api/v1/stream/session/{sid}") as ws:
            data = ws.receive_json()
            assert data["type"] == "session_start"
            assert data["payload"]["session_id"] == sid
