"""
auth.py — Authentication endpoints.

POST /api/v1/auth/register   — create account, return token
POST /api/v1/auth/login      — verify credentials, return token
POST /api/v1/auth/refresh    — re-issue token (valid token required)
GET  /api/v1/auth/me         — return current user info
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr

from backend.app.core.security import create_access_token, hash_password, verify_password
from backend.app.core.rate_limit import rate_limit
from backend.app.api.dependencies import get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    display_name: str = ""


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str


class MeResponse(BaseModel):
    user_id: str
    email: str


# ── In-memory user store (replace with DB service in production) ──────────────
# Keyed by email → {user_id, hashed_password, display_name}
_users: dict[str, dict] = {}
_counter: list[int] = [0]


def _next_id() -> str:
    _counter[0] += 1
    return f"u_{_counter[0]:06d}"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(
    body: RegisterRequest,
    _rl=Depends(rate_limit(limit=10, window_seconds=60)),
):
    """Register a new user and return an access token."""
    if body.email in _users:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
    user_id = _next_id()
    _users[body.email] = {
        "user_id": user_id,
        "hashed_password": hash_password(body.password),
        "display_name": body.display_name or body.email.split("@")[0],
        "email": body.email,
    }
    token = create_access_token(subject=user_id)
    return TokenResponse(access_token=token, user_id=user_id)


@router.post("/login", response_model=TokenResponse)
def login(
    body: LoginRequest,
    _rl=Depends(rate_limit(limit=20, window_seconds=60)),
):
    """Authenticate and return an access token."""
    record = _users.get(body.email)
    if not record or not verify_password(body.password, record["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    token = create_access_token(subject=record["user_id"])
    return TokenResponse(access_token=token, user_id=record["user_id"])


@router.post("/refresh", response_model=TokenResponse)
def refresh(user_id: str = Depends(get_current_user)):
    """Re-issue a fresh token for an already-authenticated user."""
    token = create_access_token(subject=user_id)
    return TokenResponse(access_token=token, user_id=user_id)


@router.get("/me", response_model=MeResponse)
def me(user_id: str = Depends(get_current_user)):
    """Return the current user's id and email."""
    # Find email by user_id (linear scan — fine for prototype; use DB index in prod)
    for email, record in _users.items():
        if record["user_id"] == user_id:
            return MeResponse(user_id=user_id, email=email)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
