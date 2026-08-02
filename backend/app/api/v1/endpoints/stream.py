"""
stream.py — WebSocket streaming endpoint (Phase 9: Redis-hardened).

Routes
------
GET /api/v1/stream/session/{session_id}
    Real-time prediction stream for a running session.
    Uses Redis pub/sub (falls back to in-process bus if Redis is down).

GET /api/v1/stream/demo
    Synthetic prediction stream — no hardware, no DB required.
    Sends one WsMessage/s with smoothly animated fake values.

WsMessage wire format
---------------------
{
  "type": "prediction" | "ping" | "session_start" | "session_end" | "error",
  "payload": <Prediction-shaped dict> | {"message": str} | null
}

Hardening (Phase 9)
-------------------
- Redis pub/sub via redis_stream_bus; auto-falls-back to in-process
- Per-session connection cap (MAX_CLIENTS_PER_SESSION = 8)
- 30-second ping keepalive to prevent proxy timeouts
- Graceful error → "error" frame before closing
- Structured logging (loguru) on connect/disconnect/error
"""
from __future__ import annotations

import asyncio
import math
import random
import uuid
from collections import defaultdict
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from loguru import logger

from app.core.redis_stream_bus import subscribe as redis_subscribe, publish as redis_publish
from app.api.dependencies import get_ws_current_user

router = APIRouter()

# ── constants ─────────────────────────────────────────────────────────────
MAX_CLIENTS_PER_SESSION = 8
PING_INTERVAL_SECONDS   = 25
def _queue_recv_timeout() -> float:
    import os
    return float(os.environ.get("MHBAP_WS_RECV_TIMEOUT", "60"))

# Track active client count per session for the cap
_client_counts: dict[str, int] = defaultdict(int)

# ── helpers ───────────────────────────────────────────────────────────────
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
    stress     = max(0.0, min(1.0, 0.4 + 0.25 * math.sin(t / 8.0)))
    engagement = max(0.0, min(1.0, 0.6 + 0.20 * math.cos(t / 11.0)))
    attention  = max(0.0, min(1.0, 0.55 + 0.22 * math.sin(t / 6.0 + 1.0)))
    fatigue    = max(0.0, min(1.0, 0.3 + 0.15 * math.sin(t / 20.0 + 2.0)))

    raw_logits = [random.gauss(0, 0.3) for _ in EMOTION_LABELS]
    raw_logits[0] += 1.5
    probs   = _softmax(raw_logits)
    top_idx = probs.index(max(probs))

    raw_shap  = [abs(random.gauss(0, 1)) for _ in MODALITIES]
    shap_sum  = sum(raw_shap) or 1.0
    shap      = {m: round(raw_shap[i] / shap_sum, 4) for i, m in enumerate(MODALITIES)}
    top_mod   = max(shap, key=lambda k: shap[k])

    mod_phrase = {
        "face": "facial expression cues", "gaze": "gaze and blink patterns",
        "pose": "body posture signals",   "voice": "vocal prosody features",
        "hci":  "keyboard and mouse dynamics",
    }.get(top_mod, "multimodal signals")
    emotion_lbl = EMOTION_LABELS[top_idx]
    stress_lv   = "low" if stress < 0.35 else ("moderate" if stress < 0.65 else "high")
    explanation = (
        f"The user appears {emotion_lbl} with {stress_lv} stress. "
        f"Prediction primarily driven by {mod_phrase} ({int(shap[top_mod]*100)}% attribution)."
    )
    now = _now_iso()
    return {
        "id": str(uuid.uuid4()),
        "session_id": session_id,
        "time": now,
        "recorded_at": now,
        "emotion_label": emotion_lbl,
        "emotion_scores": {EMOTION_LABELS[i]: round(probs[i], 4) for i in range(len(EMOTION_LABELS))},
        "stress":     round(stress, 3),
        "engagement": round(engagement, 3),
        "attention":  round(attention, 3),
        "fatigue":    round(fatigue, 3),
        "shap_weights":    shap,
        "explanation_text": explanation,
    }


async def _send_frame(ws: WebSocket, frame_type: str, payload) -> None:
    """Send a WsMessage frame; swallow any send errors."""
    try:
        await ws.send_json({"type": frame_type, "payload": payload})
    except Exception:
        pass


# ── WebSocket: live session stream ────────────────────────────────────────

@router.websocket("/session/{session_id}")
async def ws_session_stream(
    websocket: WebSocket,
    session_id: str,
    user_id: str = Depends(get_ws_current_user),
) -> None:
    """
    Subscribe to a running session's prediction stream.
    Closes with code 1008 (policy violation) if the per-session cap is hit.
    Sends ping frames every PING_INTERVAL_SECONDS to prevent proxy timeouts.
    """
    # Connection cap
    if _client_counts[session_id] >= MAX_CLIENTS_PER_SESSION:
        await websocket.close(code=1008, reason="Connection limit reached")
        logger.warning(f"WS rejected (cap hit) session={session_id}")
        return

    await websocket.accept()
    _client_counts[session_id] += 1
    client_id = str(uuid.uuid4())[:8]
    logger.info(f"WS connected session={session_id} client={client_id} "
                f"total={_client_counts[session_id]}")

    await _send_frame(websocket, "session_start", {"session_id": session_id})

    try:
        async with redis_subscribe(session_id) as q:
            ping_task = asyncio.create_task(_ping_loop(websocket))
            try:
                while True:
                    try:
                        msg = await asyncio.wait_for(q.get(), timeout=_queue_recv_timeout())
                    except asyncio.TimeoutError:
                        # No data in 60 s — send keep-alive and wait again
                        await _send_frame(websocket, "ping", {"ts": _now_iso()})
                        continue

                    msg_type = msg.get("type", "prediction")
                    await _send_frame(websocket, msg_type, msg.get("payload", msg))

                    if msg_type == "session_end":
                        break
            finally:
                ping_task.cancel()
                try:
                    await ping_task
                except asyncio.CancelledError:
                    pass

    except WebSocketDisconnect:
        logger.info(f"WS disconnected session={session_id} client={client_id}")
    except Exception as exc:
        logger.error(f"WS error session={session_id} client={client_id}: {exc}")
        await _send_frame(websocket, "error", {"message": str(exc)})
    finally:
        _client_counts[session_id] = max(0, _client_counts[session_id] - 1)
        if _client_counts[session_id] == 0:
            del _client_counts[session_id]
        logger.info(f"WS cleanup session={session_id} client={client_id}")


async def _ping_loop(ws: WebSocket) -> None:
    """Send a ping frame every PING_INTERVAL_SECONDS."""
    while True:
        await asyncio.sleep(PING_INTERVAL_SECONDS)
        await _send_frame(ws, "ping", {"ts": _now_iso()})


# ── WebSocket: demo stream ────────────────────────────────────────────────

@router.websocket("/demo")
async def ws_demo_stream(
    websocket: WebSocket,
    user_id: str = Depends(get_ws_current_user),
) -> None:
    """
    Synthetic prediction stream for UI development / demos.
    No DB, no hardware, no Redis required.
    Emits one prediction per second with sinusoidal variation.
    """
    await websocket.accept()
    session_id = f"demo-{str(uuid.uuid4())[:8]}"
    logger.info(f"WS demo connected session={session_id}")

    await _send_frame(websocket, "session_start", {"session_id": session_id})

    t = 0.0
    try:
        while True:
            pred = _make_demo_prediction(t, session_id)
            await _send_frame(websocket, "prediction", pred)
            t += 1.0
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        logger.info(f"WS demo disconnected session={session_id}")
    except Exception as exc:
        logger.error(f"WS demo error: {exc}")
        await _send_frame(websocket, "error", {"message": str(exc)})
    finally:
        await _send_frame(websocket, "session_end", {"session_id": session_id})
        logger.info(f"WS demo ended session={session_id}")
