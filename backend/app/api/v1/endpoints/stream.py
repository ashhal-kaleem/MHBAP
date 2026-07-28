"""
stream.py — WebSocket streaming endpoint.

Routes
------
GET /api/v1/stream/session/{session_id}
    Real-time prediction stream for a running session.
    Clients subscribe via StreamBus; SessionRunner publishes each tick.

GET /api/v1/stream/demo
    Synthetic prediction stream — no hardware, no DB required.
    Sends one WsMessage per second with smoothly animated fake values.
    Perfect for frontend development and demos.

WsMessage wire format
---------------------
{
  "type": "prediction" | "ping" | "session_start" | "session_end" | "error",
  "payload": <Prediction-shaped dict> | {"message": str} | null
}

Prediction payload fields match PredictionRead schema PLUS a
"recorded_at" alias for the "time" field (frontend compat).
"""
from __future__ import annotations

import asyncio
import math
import random
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from backend.app.core.stream_bus import subscribe, unsubscribe

router = APIRouter()

# ── helpers ────────────────────────────────────────────────────────────────

EMOTION_LABELS = [
    "neutral", "happy", "sad", "angry",
    "surprised", "fearful", "disgusted", "contemptuous",
]
MODALITIES = ["face", "gaze", "pose", "voice", "hci"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _softmax(logits: list) -> list:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _make_demo_prediction(t: float, session_id: str) -> dict:
    """Generate one synthetic prediction at time offset t (seconds)."""
    # Smoothly oscillating signals so the charts look alive
    stress     = 0.4 + 0.25 * math.sin(t / 8.0)
    engagement = 0.6 + 0.20 * math.cos(t / 11.0)
    attention  = 0.55 + 0.22 * math.sin(t / 6.0 + 1.0)
    fatigue    = 0.3 + 0.15 * math.sin(t / 20.0 + 2.0)

    # Clamp to [0, 1]
    stress     = max(0.0, min(1.0, stress))
    engagement = max(0.0, min(1.0, engagement))
    attention  = max(0.0, min(1.0, attention))
    fatigue    = max(0.0, min(1.0, fatigue))

    # Emotion — mostly neutral with gentle drift
    raw_logits = [random.gauss(0, 0.3) for _ in EMOTION_LABELS]
    raw_logits[0] += 1.5          # bias toward neutral
    probs = _softmax(raw_logits)
    top_idx = probs.index(max(probs))

    # SHAP weights (sum to 1.0)
    raw_shap = [abs(random.gauss(0, 1)) for _ in MODALITIES]
    shap_sum = sum(raw_shap) or 1.0
    shap = {m: round(raw_shap[i] / shap_sum, 4) for i, m in enumerate(MODALITIES)}

    # NL explanation
    top_mod = max(shap, key=lambda k: shap[k])
    mod_phrase = {
        "face": "facial expression cues", "gaze": "gaze and blink patterns",
        "pose": "body posture signals",   "voice": "vocal prosody features",
        "hci":  "keyboard and mouse dynamics",
    }.get(top_mod, "multimodal signals")
    emotion_lbl = EMOTION_LABELS[top_idx]
    stress_lv = "low" if stress < 0.35 else ("moderate" if stress < 0.65 else "high")
    explanation = (
        f"The user appears {emotion_lbl} with {stress_lv} stress. "
        f"Prediction primarily driven by {mod_phrase} ({int(shap[top_mod]*100)}% attribution)."
    )

    now = _now_iso()
    return {
        "id": str(uuid.uuid4()),
        "session_id": session_id,
        "time": now,
        "recorded_at": now,          # alias for frontend compat
        "emotion_label": emotion_lbl,
        "emotion_scores": {EMOTION_LABELS[i]: round(probs[i], 4) for i in range(len(EMOTION_LABELS))},
        "stress":     round(stress, 3),
        "engagement": round(engagement, 3),
        "attention":  round(attention, 3),
        "fatigue":    round(fatigue, 3),
        "shap_weights":    shap,
        "explanation_text": explanation,
    }


# ── WebSocket: live session ────────────────────────────────────────────────

@router.websocket("/session/{session_id}")
async def stream_session(websocket: WebSocket, session_id: uuid.UUID) -> None:
    """
    Stream live predictions for an active session.

    The SessionRunner calls stream_bus.publish(session_id, msg) each tick.
    This endpoint subscribes a per-client queue and forwards messages.
    """
    sid = str(session_id)
    await websocket.accept()
    logger.info(f"WS connected: session={sid}")

    q = subscribe(sid)
    try:
        await websocket.send_json({"type": "session_start", "payload": {"session_id": sid}})

        while True:
            # Wait for next message with a short timeout so we stay responsive
            try:
                msg = await asyncio.wait_for(q.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send a ping to keep the connection alive
                await websocket.send_json({"type": "ping", "payload": None})
                continue
            await websocket.send_json(msg)

    except WebSocketDisconnect:
        logger.info(f"WS disconnected: session={sid}")
    except Exception as exc:
        logger.error(f"WS error session={sid}: {exc}")
    finally:
        unsubscribe(sid, q)


# ── WebSocket: demo (no hardware / no DB) ─────────────────────────────────

@router.websocket("/demo")
async def stream_demo(websocket: WebSocket) -> None:
    """
    Synthetic prediction stream.
    Connect to ws://localhost:8000/api/v1/stream/demo (or via Vite proxy /ws/demo).
    """
    demo_session_id = "demo-session-001"
    await websocket.accept()
    logger.info("WS demo connected")
    t0 = 0.0
    try:
        await websocket.send_json({"type": "session_start", "payload": {"session_id": demo_session_id}})
        while True:
            pred = _make_demo_prediction(t0, demo_session_id)
            await websocket.send_json({"type": "prediction", "payload": pred})
            t0 += 1.0
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        logger.info("WS demo disconnected")
    except Exception as exc:
        logger.error(f"WS demo error: {exc}")
