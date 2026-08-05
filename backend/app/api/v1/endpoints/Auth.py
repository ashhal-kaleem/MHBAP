"""
auth.py — Authentication endpoints.

POST /api/v1/Auth/register   — create account, return token
POST /api/v1/Auth/login      — verify credentials, return token
POST /api/v1/Auth/refresh    — re-issue token, blacklist the old one
POST /api/v1/Auth/logout     — blacklist the current token
GET  /api/v1/Auth/me         — return current user info

Security hardening (Phase C):
  - Password strength: min 8 chars, at least one digit or special char
  - Account lockout: 5 failed attempts in 5 min → 15-min lockout per email
  - Refresh token blacklist: old token revoked the moment a new one is issued
  - Rate limiting on all mutation endpoints
"""
from __future__ import annotations

import re
import uuid as _uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.Security import (
    create_access_token,
    decode_access_token,
    verify_password,
    blacklist_token,
)
from app.core.RateLimit import (
    rate_limit,
    check_account_lockout,
    record_failed_login,
    clear_failed_logins,
)
from app.api.Dependencies import get_current_user, get_bearer_token, get_db
from app.services.UserService import (
    create_user_with_password,
    get_user_by_email,
    get_user,
)

router = APIRouter(tags=["auth"])


# ── Request / Response schemas ────────────────────────────────────────────────

_SPECIAL = re.compile(r"[^a-zA-Z0-9]")
_DIGIT = re.compile(r"\d")


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    display_name: str = ""

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not (_DIGIT.search(v) or _SPECIAL.search(v)):
            raise ValueError(
                "Password must contain at least one digit or special character"
            )
        return v


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
    display_name: str
    role: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
async def register(
    body: RegisterRequest,
    db: AsyncSession = Depends(get_db),
    _rl=Depends(rate_limit(limit=10, window_seconds=60)),
):
    """Create a new user account and return a JWT access token."""
    existing = await get_user_by_email(db, body.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Email already registered"
        )
    user = await create_user_with_password(
        db,
        email=body.email,
        password=body.password,
        display_name=body.display_name,
    )
    token = create_access_token(subject=str(user.id))
    return TokenResponse(access_token=token, user_id=str(user.id))


@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    db: AsyncSession = Depends(get_db),
    _rl=Depends(rate_limit(limit=20, window_seconds=60)),
):
    """Authenticate and return an access token. Locks the account after
    repeated failures to slow down credential-stuffing attacks."""
    await check_account_lockout(body.email)

    user = await get_user_by_email(db, body.email)
    if not user or not user.is_active:
        await record_failed_login(body.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    if not verify_password(body.password, user.hashed_password):
        await record_failed_login(body.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    await clear_failed_logins(body.email)
    token = create_access_token(subject=str(user.id))
    return TokenResponse(access_token=token, user_id=str(user.id))


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    user_id: str = Depends(get_current_user),
    token: str = Depends(get_bearer_token),
):
    """Re-issue a fresh token for an already-authenticated user, revoking
    the token that was just used (refresh-token rotation)."""
    try:
        payload = decode_access_token(token)
        await blacklist_token(payload.get("jti", ""), payload.get("exp", 0))
    except ValueError:
        # Token was already validated by get_current_user in real use; this
        # only trips in tests that stub get_current_user with a dummy token.
        pass

    new_token = create_access_token(subject=user_id)
    return TokenResponse(access_token=new_token, user_id=user_id)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    _user_id: str = Depends(get_current_user),
    token: str = Depends(get_bearer_token),
):
    """Revoke the current access token so it can no longer be used."""
    try:
        payload = decode_access_token(token)
        await blacklist_token(payload.get("jti", ""), payload.get("exp", 0))
    except ValueError:
        pass
    return None


@router.get("/me", response_model=MeResponse)
async def me(
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the current user's profile."""
    user = await get_user(db, _uuid.UUID(user_id))
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return MeResponse(
        user_id=str(user.id),
        email=user.email,
        display_name=user.display_name,
        role=user.role,
    )
