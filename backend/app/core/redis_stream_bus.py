"""
redis_stream_bus.py — Redis pub/sub backed message bus for WebSocket streaming.

Architecture
------------
Publisher (SessionRunner):
    await publish(session_id, message)
    → serialises to JSON → Redis PUBLISH mhbap:stream:{session_id}

Subscriber (WebSocket handler):
    async with subscribe(session_id) as q:
        async for msg in q:
            await ws.send_json(msg)
    → each WS handler runs its own asyncio task draining a Redis subscriber

Fallback
--------
If Redis is unreachable at startup, or a publish/subscribe call fails,
the bus transparently falls back to the in-process queue bus so the app
works without Redis (dev / CI).

Channel naming
--------------
  mhbap:stream:{session_id}    — per-session prediction stream
  mhbap:stream:*               — wildcard used by psubscribe

"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)

CHANNEL_PREFIX = "mhbap:stream"
_redis_available: bool | None = None   # None = not yet probed
_last_probe_time: float = 0.0
_PROBE_INTERVAL = 60.0  # re-probe every 60s so recovered Redis is detected


def _channel(session_id: str) -> str:
    return f"{CHANNEL_PREFIX}:{session_id}"


async def _probe_redis() -> bool:
    global _redis_available, _last_probe_time
    import time
    now = time.monotonic()
    # Re-probe if: never probed, or last probe was >60s ago
    if _redis_available is not None and (now - _last_probe_time) < _PROBE_INTERVAL:
        return _redis_available
    try:
        from app.core.redis import get_redis
        await get_redis().ping()
        if _redis_available is not True:
            logger.info("Redis available — using Redis pub/sub bus")
        _redis_available = True
    except Exception as exc:
        if _redis_available is not False:
            logger.warning(f"Redis unavailable ({exc}) — falling back to in-process bus")
        _redis_available = False
    _last_probe_time = now
    return _redis_available


# ── publish ───────────────────────────────────────────────────────────────

async def publish(session_id: str, message: Any) -> None:
    """Publish a WsMessage dict to all subscribers of this session."""
    if await _probe_redis():
        try:
            from app.core.redis import get_redis
            await get_redis().publish(_channel(session_id), json.dumps(message))
            return
        except Exception as exc:
            logger.error(f"Redis publish failed ({exc}), falling back to in-process")
            # Force re-probe next time instead of staying stuck on broken connection
            global _last_probe_time
            _last_probe_time = 0.0
    # fallback
    from app.core import stream_bus
    stream_bus.publish(session_id, message)


# ── subscribe (async context manager) ────────────────────────────────────

@asynccontextmanager
async def subscribe(session_id: str, maxsize: int = 128) -> AsyncIterator[asyncio.Queue]:
    """
    Async context manager that yields an asyncio.Queue of decoded messages.

    Usage:
        async with subscribe(session_id) as q:
            msg = await asyncio.wait_for(q.get(), timeout=30)
    """
    if await _probe_redis():
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        task = asyncio.create_task(_redis_drain(session_id, q))
        try:
            yield q
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    else:
        # in-process fallback
        from app.core import stream_bus
        q = stream_bus.subscribe(session_id, maxsize=maxsize)
        try:
            yield q
        finally:
            stream_bus.unsubscribe(session_id, q)


async def _redis_drain(session_id: str, q: asyncio.Queue) -> None:
    """Background task: subscribe to Redis channel and push decoded msgs to queue."""
    from app.core.redis import get_redis_pool
    import redis.asyncio as redis

    channel = _channel(session_id)
    while True:
        try:
            client = redis.Redis(connection_pool=get_redis_pool())
            pubsub = client.pubsub()
            await pubsub.subscribe(channel)
            logger.debug(f"Redis drain started: {channel}")
            async for raw_msg in pubsub.listen():
                if raw_msg["type"] == "message":
                    try:
                        decoded = json.loads(raw_msg["data"])
                    except json.JSONDecodeError:
                        continue
                    try:
                        q.put_nowait(decoded)
                    except asyncio.QueueFull:
                        # slow consumer — drop oldest frame
                        try:
                            q.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        q.put_nowait(decoded)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.warning(f"Redis drain error ({exc}), reconnecting in 2s")
            await asyncio.sleep(2)
