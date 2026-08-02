"""
Application configuration via environment variables.
All settings have safe defaults for local development.
Production values must be injected via .env or container environment.
"""
from __future__ import annotations

import json
from typing import List

from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────
    APP_ENV: str = "development"
    APP_NAME: str = "MHBAP"
    APP_VERSION: str = "0.1.0"
    SECRET_KEY: str = "dev-secret-key-replace-in-production"
    DEBUG: bool = True

    # ── Database ─────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://mhbap:mhbap@localhost:5432/mhbap"
    DATABASE_SYNC_URL: str = "postgresql://mhbap:mhbp2625@postgres:5432/mhbap"
    # ── Redis ─────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── ML ────────────────────────────────────────────
    DEVICE: str = "cpu"
    MODEL_WEIGHTS_DIR: str = "ml/models/weights"
    INFERENCE_FPS: int = 1
    WINDOW_SECONDS: int = 30

    # ── CORS ─────────────────────────────────────────
    CORS_ORIGINS: List[AnyHttpUrl] = []

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors(cls, v: str | list) -> list:
        if isinstance(v, str):
            return json.loads(v)
        return v


settings = Settings()
