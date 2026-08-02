"""Pydantic schemas for the User resource."""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr


class UserCreate(BaseModel):
    username: str
    email: EmailStr
    hashed_password: str = ""
    display_name: str = ""
    role: str = "participant"
    is_active: bool = True


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    username: str
    email: EmailStr
    display_name: str
    role: str
    is_active: bool
    created_at: datetime
