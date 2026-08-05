"""
rate_limit.py — Redis-backed sliding-window rate limiter.

Designed to work with Redis for production readiness in a distributed setup.
"""
from __future__ import annotations

import time
import uuid
from typing import Tuple

from fastapi import Request, HTTPException, status

from app.core.redis import get_redis

# ── Account lockout constants ──────────────────────────────────────────────────

LOCKOUT_MAX_ATTEMPTS = 5        # failed attempts before lockout
LOCKOUT_WINDOW_SECONDS = 300    # 5-minute sliding window
LOCKOUT_DURATION_SECONDS = 900  # 15-minute lockout after threshold


async def record_failed_login(identifier: str) -> None:
    """Record a failed login attempt for *identifier* (email or IP)."""
    now = time.time()
    key = f"lockout:{identifier}"
    redis = get_redis()
    member = f"{now}:{uuid.uuid4()}"
    pipe = redis.pipeline()
    pipe.zadd(key, {member: now})
    pipe.expire(key, LOCKOUT_WINDOW_SECONDS)
    await pipe.execute()


async def clear_failed_logins(identifier: str) -> None:
    """Clear failed-login history on successful authentication."""
    key = f"lockout:{identifier}"
    redis = get_redis()
    await redis.delete(key)


async def check_account_lockout(identifier: str) -> None:
    """Raise HTTP 429 if *identifier* is currently locked out."""
    now = time.time()
    cutoff = now - LOCKOUT_WINDOW_SECONDS
    key = f"lockout:{identifier}"
    redis = get_redis()
    
    pipe = redis.pipeline()
    pipe.zremrangebyscore(key, 0, cutoff)
    pipe.zcard(key)
    pipe.zrange(key, 0, 0, withscores=True)
    results = await pipe.execute()
    
    count = results[1]
    if count >= LOCKOUT_MAX_ATTEMPTS:
        oldest_records = results[2]
        retry_after = LOCKOUT_WINDOW_SECONDS
        if oldest_records:
            _, oldest_time = oldest_records[0]
            retry_after = int(LOCKOUT_WINDOW_SECONDS - (now - oldest_time)) + 1
        if retry_after < 0:
            retry_after = 0
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Account temporarily locked. Retry after {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )


async def _sliding_window_check(key: str, limit: int, window_seconds: int) -> Tuple[int, int]:
    """
    Check the sliding window in Redis.
    Returns (remaining_requests, retry_after_seconds).
    Raises HTTPException 429 if limit exceeded.
    """
    now = time.time()
    cutoff = now - window_seconds
    redis = get_redis()
    
    pipe = redis.pipeline()
    pipe.zremrangebyscore(key, 0, cutoff)
    pipe.zcard(key)
    pipe.zrange(key, 0, 0, withscores=True)
    results = await pipe.execute()
    
    count = results[1]
    if count >= limit:
        oldest_records = results[2]
        retry_after = 1
        if oldest_records:
            _, oldest_time = oldest_records[0]
            retry_after = int(oldest_time - cutoff) + 1
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Retry after {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )
        
    member = f"{now}:{uuid.uuid4()}"
    pipe = redis.pipeline()
    pipe.zadd(key, {member: now})
    pipe.expire(key, window_seconds)
    await pipe.execute()
    
    return limit - count - 1, 0


# ── Dependency factories ──────────────────────────────────────────────────────

def _get_client_ip(request: Request) -> str:
    """Extract real client IP, honouring X-Forwarded-For / X-Real-IP proxy headers."""
    # X-Forwarded-For: client, proxy1, proxy2 — take leftmost (real client)
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else "unknown"

def rate_limit(limit: int = 60, window_seconds: int = 60):
    """
    FastAPI dependency: rate-limit by real client IP (proxy-aware).
    """
    async def _dep(request: Request):
        ip = _get_client_ip(request)
        key = f"rl:{ip}:{request.url.path}"
        remaining, _ = await _sliding_window_check(key, limit, window_seconds)
        request.state.ratelimit_remaining = remaining

    return _dep


def rate_limit_user(limit: int = 120, window_seconds: int = 60):
    """
    FastAPI dependency: rate-limit by authenticated user_id (falls back to real IP).
    Must be placed after an auth dependency that sets request.state.user_id.
    """
    async def _dep(request: Request):
        user_id = getattr(request.state, "user_id", None)
        key_part = f"user:{user_id}" if user_id else f"ip:{_get_client_ip(request)}"
        key = f"rl:{key_part}:{request.url.path}"
        await _sliding_window_check(key, limit, window_seconds)

    return _dep
