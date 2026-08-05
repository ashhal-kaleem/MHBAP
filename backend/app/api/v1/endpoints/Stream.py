"""
stream.py — WebSocket streaming endpoint (Phase 9: Redis-hardened).

Routes
------
GET /api/v1/Stream/Session/{session_id}
    Real-time prediction stream for a running session.
    Uses Redis pub/sub (falls back to in-process bus if Redis is down).

GET /api/v1/Stream/demo
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

from app.core.Redis_stream_bus import subscribe as redis_subscribe, publish as redis_publish
from app.api.Dependencies import get_ws_current_user  # used by live session stream only
from app.db.Session import get_session_factory
from app.services import session_service

router = APIRouter()

# ── constants ─────────────────────────────────────────────────────────────
MAX_CLIENTS_PER_SESSION = 8
PING_INTERVAL_SECONDS   = 25
# Cache timeout at module load — avoids env lookup on every WS receive iteration
import os as _os
_QUEUE_RECV_TIMEOUT: float = float(_os.environ.get("MHBAP_WS_RECV_TIMEOUT", "60"))
del _os


def _queue_recv_timeout() -> float:
    return _QUEUE_RECV_TIMEOUT

# Track active client count per session for the cap
# In-process fallback used when Redis unavailable (single-worker only)
_client_counts: dict[str, int] = defaultdict(int)

_WS_CAP_PREFIX = "mhbap:ws:cap"
_WS_CAP_TTL    = 3600  # 1 hour — auto-expire stale keys if worker crashes


async def _ws_count_incr(session_id: str) -> int:
    """Atomically increment WS connection count. Returns new count."""
    if await _probe_redis():
        try:
            from app.core.Redis import get_redis
            r = get_redis()
            key = f"{_WS_CAP_PREFIX}:{session_id}"
            count = await r.incr(key)
            await r.expire(key, _WS_CAP_TTL)
            return int(count)
        except Exception:
            pass  # fall through to in-process
    _client_counts[session_id] += 1
    return _client_counts[session_id]


async def _ws_count_decr(session_id: str) -> int:
    """Atomically decrement WS connection count. Returns new count (min 0)."""
    if await _probe_redis():
        try:
            from app.core.Redis import get_redis
            r = get_redis()
            key = f"{_WS_CAP_PREFIX}:{session_id}"
            count = await r.decr(key)
            count = max(0, int(count))
            if count == 0:
                await r.delete(key)
            return count
        except Exception:
            pass
    _client_counts[session_id] = max(0, _client_counts[session_id] - 1)
    if _client_counts[session_id] == 0:
        del _client_counts[session_id]
    return _client_counts[session_id]


async def _ws_count_get(session_id: str) -> int:
    """Get current WS connection count."""
    if await _probe_redis():
        try:
            from app.core.Redis import get_redis
            val = await get_redis().get(f"{_WS_CAP_PREFIX}:{session_id}")
            return int(val) if val else 0
        except Exception:
            pass
    return _client_counts[session_id]

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

@router.websocket("/Session/{session_id}")
async def ws_session_stream(
    websocket: WebSocket,
    session_id: str,
    user_id: str = Depends(get_ws_current_user),
) -> None:
    """
    Subscribe to a running session's prediction stream.
    Validates session_id as UUID and enforces ownership before accepting.
    Closes with code 1008 (policy violation) if cap hit or not authorized.
    """
    # Validate session_id is a real UUID — prevents arbitrary Redis channel injection
    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        await websocket.close(code=1008, reason="Invalid session_id")
        return

    # Ownership check BEFORE accept() — fetch session from DB
    async with get_session_factory()() as db:
        session = await session_service.get_session(db, session_uuid)
    if session is None:
        await websocket.close(code=1008, reason="Session not found")
        return
    if str(session.user_id) != user_id:
        await websocket.close(code=1008, reason="Not authorized")
        logger.warning(f"WS rejected (ownership) session={session_id} user={user_id}")
        return

    # Connection cap — Redis-backed for multi-worker correctness
    current = await _ws_count_get(session_id)
    if current >= MAX_CLIENTS_PER_SESSION:
        await websocket.close(code=1008, reason="Connection limit reached")
        logger.warning(f"WS rejected (cap hit) session={session_id} count={current}")
        return

    await websocket.accept()
    total = await _ws_count_incr(session_id)
    client_id = str(uuid.uuid4())[:8]
    logger.info(f"WS connected session={session_id} client={client_id} total={total}")

    try:
        await _send_frame(websocket, "session_start", {"session_id": session_id})
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
        remaining = await _ws_count_decr(session_id)
        logger.info(f"WS cleanup session={session_id} client={client_id} remaining={remaining}")


async def _ping_loop(ws: WebSocket) -> None:
    """Send a ping frame every PING_INTERVAL_SECONDS."""
    while True:
        await asyncio.sleep(PING_INTERVAL_SECONDS)
        await _send_frame(ws, "ping", {"ts": _now_iso()})


# ── WebSocket: demo stream ────────────────────────────────────────────────

@router.websocket("/demo")
async def ws_demo_stream(
    websocket: WebSocket,
) -> None:
    # NOTE: No auth required — this endpoint emits purely synthetic data
    # (no DB, no real sessions, no PII).  The live session stream at
    # /Stream/Session/{id} retains its get_ws_current_user guard.
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
