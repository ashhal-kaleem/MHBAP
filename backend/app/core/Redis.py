"""
Redis connection pool.

Used as: (1) message queue between webcam/audio capture and the
inference service (Phase 5+), and (2) pub/sub backbone for the
WebSocket streaming endpoint (Phase 9).
"""
from __future__ import annotations

import redis.asyncio as redis

from app.core.Config import settings

_pool: redis.ConnectionPool | None = None
_client: redis.Redis | None = None


def get_redis_pool() -> redis.ConnectionPool:
    global _pool
    if _pool is None:
        _pool = redis.ConnectionPool.from_url(
            settings.REDIS_URL, decode_responses=True, max_connections=50
        )
    return _pool


def get_redis() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis(connection_pool=get_redis_pool())
    return _client


async def check_redis_connection() -> bool:
    """Used by the readiness probe. Returns False instead of raising."""
    try:
        await get_redis().ping()
        return True
    except Exception:
        return False


async def close_redis() -> None:
    """Called on app shutdown."""
    global _client, _pool
    if _client is not None:
        await _client.aclose()
        _client = None
    if _pool is not None:
        await _pool.disconnect()
        _pool = None
