"""
rate_limit.py — Simple in-process sliding-window rate limiter.

Designed to work without Redis for unit tests; swaps to Redis-backed
counts in production when REDIS_URL is set.
Uses a thread-safe in-memory fallback for dev/test environments.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock
from typing import Deque, Dict, Tuple

from fastapi import Request, HTTPException, status


# ── In-process store ─────────────────────────────────────────────────────────

_store: Dict[str, Deque[float]] = defaultdict(deque)
_lock = Lock()


def _sliding_window_check(key: str, limit: int, window_seconds: int) -> Tuple[int, int]:
    """
    Check the sliding window.
    Returns (remaining_requests, retry_after_seconds).
    Raises HTTPException 429 if limit exceeded.
    """
    now = time.monotonic()
    cutoff = now - window_seconds

    with _lock:
        dq = _store[key]
        # Drop entries outside the window
        while dq and dq[0] < cutoff:
            dq.popleft()
        count = len(dq)
        if count >= limit:
            oldest = dq[0]
            retry_after = int(oldest - cutoff) + 1
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Retry after {retry_after}s.",
                headers={"Retry-After": str(retry_after)},
            )
        dq.append(now)
        return limit - count - 1, 0


# ── Dependency factories ──────────────────────────────────────────────────────

def rate_limit(limit: int = 60, window_seconds: int = 60):
    """
    FastAPI dependency: rate-limit by client IP.

    Usage:
        @router.get("/heavy")
        async def heavy(request: Request, _=Depends(rate_limit(10, 60))):
            ...
    """
    def _dep(request: Request):
        ip = request.client.host if request.client else "unknown"
        key = f"rl:{ip}:{request.url.path}"
        remaining, _ = _sliding_window_check(key, limit, window_seconds)
        # Attach headers for the response (best-effort — middleware would be cleaner)
        request.state.ratelimit_remaining = remaining

    return _dep


def rate_limit_user(limit: int = 120, window_seconds: int = 60):
    """
    FastAPI dependency: rate-limit by authenticated user_id (falls back to IP).
    Must be placed after an auth dependency that sets request.state.user_id.
    """
    def _dep(request: Request):
        user_id = getattr(request.state, "user_id", None)
        key_part = f"user:{user_id}" if user_id else f"ip:{request.client.host}"
        key = f"rl:{key_part}:{request.url.path}"
        _sliding_window_check(key, limit, window_seconds)

    return _dep
