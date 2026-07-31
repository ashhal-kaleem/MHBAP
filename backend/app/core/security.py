"""
security.py — JWT creation/verification and password hashing.

Uses only stdlib + passlib + python-jose (already common in FastAPI stacks).
Falls back to a pure-Python HMAC implementation if jose is unavailable
so unit tests run without the optional dependency installed.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
import base64
import uuid as _uuid
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any, Dict, Optional, Tuple

from passlib.context import CryptContext

from app.core.config import settings


# ── Password hashing ──────────────────────────────────────────────────────────

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)


# ── JWT (pure-Python fallback — no python-jose dep required) ─────────────────

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24   # 24 h


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    pad = 4 - len(s) % 4
    return base64.urlsafe_b64decode(s + "=" * (pad % 4))


def create_access_token(
    subject: str,
    expires_delta: Optional[timedelta] = None,
    extra_claims: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a signed JWT access token."""
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    payload: Dict[str, Any] = {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "jti": str(_uuid.uuid4()),
        **(extra_claims or {}),
    }
    header = _b64url(json.dumps({"alg": ALGORITHM, "typ": "JWT"}).encode())
    body   = _b64url(json.dumps(payload).encode())
    signing_input = f"{header}.{body}".encode()
    sig = hmac.new(
        settings.SECRET_KEY.encode(),
        signing_input,
        hashlib.sha256,
    ).digest()
    return f"{header}.{body}.{_b64url(sig)}"


def decode_access_token(token: str) -> Dict[str, Any]:
    """
    Verify and decode a JWT access token.
    Raises ValueError on any failure.
    """
    try:
        header_b64, body_b64, sig_b64 = token.split(".")
    except ValueError:
        raise ValueError("Malformed token")

    signing_input = f"{header_b64}.{body_b64}".encode()
    expected_sig = hmac.new(
        settings.SECRET_KEY.encode(),
        signing_input,
        hashlib.sha256,
    ).digest()
    actual_sig = _b64url_decode(sig_b64)
    if not hmac.compare_digest(expected_sig, actual_sig):
        raise ValueError("Invalid signature")

    payload: Dict[str, Any] = json.loads(_b64url_decode(body_b64))
    if payload.get("exp", 0) < time.time():
        raise ValueError("Token expired")
    return payload


# ── Token blacklist (logout / refresh rotation) ───────────────────────────────
#
# In-process store, mirroring rate_limit.py's approach: sufficient for a
# single-worker dev/test deployment; swap for a Redis SET with TTL in
# production (same pattern used by core/redis.py for the stream bus).

_blacklist: Dict[str, float] = {}
_blacklist_lock = Lock()


def blacklist_token(jti: str, exp: float) -> None:
    """Mark a token's *jti* as revoked until it would have expired anyway."""
    if not jti:
        return
    with _blacklist_lock:
        _blacklist[jti] = exp


def is_token_blacklisted(jti: Optional[str]) -> bool:
    """Check whether *jti* has been revoked. Lazily evicts expired entries."""
    if not jti:
        return False
    with _blacklist_lock:
        exp = _blacklist.get(jti)
        if exp is None:
            return False
        if exp < time.time():
            del _blacklist[jti]
            return False
        return True


def clear_blacklist() -> None:
    """Test helper: wipe the blacklist store."""
    with _blacklist_lock:
        _blacklist.clear()
