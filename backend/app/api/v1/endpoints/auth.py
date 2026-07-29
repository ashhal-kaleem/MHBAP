"""
auth.py — Authentication endpoints.

POST /api/v1/auth/register   — create account, return token
POST /api/v1/auth/login      — verify credentials, return token
POST /api/v1/auth/refresh    — re-issue token (valid token required)
GET  /api/v1/auth/me         — return current user info

Auth is now fully DB-backed: passwords are bcrypt-hashed and persisted
in the PostgreSQL users table via user_service.  No in-memory state.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.security import create_access_token, verify_password
from backend.app.core.rate_limit import rate_limit
from backend.app.api.dependencies import get_current_user, get_db
from backend.app.services.user_service import (
    create_user_with_password,
    get_user_by_email,
    get_user,
)

router = APIRouter(tags=["auth"])


# ── Request / Response schemas ────────────────────────────────────────────────

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
    display_name: str
    role: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest,
    db: AsyncSession = Depends(get_db),
    _rl=Depends(rate_limit(limit=10, window_seconds=60)),
):
    """Register a new user and return an access token."""
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
    """Authenticate and return an access token."""
    user = await get_user_by_email(db, body.email)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    if not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    token = create_access_token(subject=str(user.id))
    return TokenResponse(access_token=token, user_id=str(user.id))


@router.post("/refresh", response_model=TokenResponse)
async def refresh(user_id: str = Depends(get_current_user)):
    """Re-issue a fresh token for an already-authenticated user."""
    token = create_access_token(subject=user_id)
    return TokenResponse(access_token=token, user_id=user_id)


@router.get("/me", response_model=MeResponse)
async def me(
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the current user's profile."""
    import uuid as _uuid
    user = await get_user(db, _uuid.UUID(user_id))
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return MeResponse(
        user_id=str(user.id),
        email=user.email,
        display_name=user.display_name,
        role=user.role,
    )
