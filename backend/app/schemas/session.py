"""Pydantic schemas for the Session resource."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class SessionCreate(BaseModel):
    user_id: uuid.UUID
    context: str = "unspecified"
    consent_recording: bool = False
    faces_blurred: bool = False
    session_metadata: Optional[dict] = None


class SessionUpdate(BaseModel):
    status: Optional[str] = None
    ended_at: Optional[datetime] = None


class SessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    context: str
    status: str
    consent_recording: bool
    faces_blurred: bool
    session_metadata: Optional[dict]
    started_at: datetime
    ended_at: Optional[datetime]
