"""
WebSocket streaming endpoint — SKELETON ONLY.

Full implementation (Phase 9) will subscribe to a Redis pub/sub channel
keyed by session_id, fed by the inference service, and push each new
Prediction as JSON the moment it's written. For now this just accepts
the connection and echoes a heartbeat so the frontend can be built
against a stable contract before the real pipeline exists.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

router = APIRouter()


@router.websocket("/session/{session_id}")
async def stream_session(websocket: WebSocket, session_id: uuid.UUID) -> None:
    await websocket.accept()
    logger.info(f"WebSocket connected for session {session_id}")
    try:
        await websocket.send_json({"type": "connected", "session_id": str(session_id)})
        while True:
            # Phase 9: replace with `await redis_pubsub.get_message()`
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
