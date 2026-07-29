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
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from passlib.context import CryptContext

from backend.app.core.config import settings


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
