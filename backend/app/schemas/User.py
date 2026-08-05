"""Pydantic schemas for the User resource."""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserCreate(BaseModel):
    """Legacy schema used by unauthenticated POST /users/ (Phase 2).
    hashed_password / role / is_active intentionally excluded —
    those are server-assigned only.
    """
    username: str = Field(..., min_length=3, max_length=64)
    email: EmailStr
    display_name: str = ""


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    username: str
    email: EmailStr
    display_name: str
    created_at: datetime
    # role and is_active intentionally omitted from public read schema
