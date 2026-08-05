"""
Full-stack CRUD flow: create user -> create session -> write prediction
-> read it back. Requires live Postgres; skips otherwise (see
test_health_integration.py for the same guard pattern).
"""
from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.core.Security import create_access_token
from app.db.Session import check_db_connection
from app.Main import app


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_full_session_prediction_flow(client: AsyncClient) -> None:
    if not await check_db_connection():
        pytest.skip("Postgres not reachable — start `docker compose up db redis` to run this")

    user_resp = await client.post(
        "/api/v1/users/",
        json={"username": f"pytest_{id(client)}", "email": f"pytest_{id(client)}@example.com"},
    )
    assert user_resp.status_code == 201
    user_id = user_resp.json()["id"]

    token = create_access_token(subject=str(user_id))
    headers = {"Authorization": f"Bearer {token}"}

    session_resp = await client.post(
        "/api/v1/sessions/",
        json={"user_id": user_id, "context": "coding_task"},
        headers=headers,
    )
    assert session_resp.status_code == 201
    session_id = session_resp.json()["id"]
    assert session_resp.json()["status"] == "active"

    pred_resp = await client.post(
        "/api/v1/predictions/",
        json={
            "session_id": session_id,
            "emotion_label": "focused",
            "emotion_scores": {"focused": 0.9, "neutral": 0.1},
            "stress": 0.3,
            "engagement": 0.85,
            "attention": 0.8,
            "fatigue": 0.15,
            "shap_weights": {"hci": 0.5, "face": 0.3, "voice": 0.2},
            "explanation_text": "Engagement driven by steady typing rhythm.",
        },
        headers=headers,
    )
    assert pred_resp.status_code == 201

    latest = await client.get(f"/api/v1/predictions/session/{session_id}/latest", headers=headers)
    assert latest.status_code == 200
    assert latest.json()["emotion_label"] == "focused"

    end_resp = await client.post(f"/api/v1/sessions/{session_id}/end", headers=headers)
    assert end_resp.status_code == 200
    assert end_resp.json()["status"] == "completed"
    assert end_resp.json()["ended_at"] is not None
