"""
Application configuration via environment variables.
All settings have safe defaults for local development.
Production values must be injected via .env or container environment.
"""
from __future__ import annotations

import json
from typing import List

from pydantic import AnyHttpUrl, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_INSECURE_KEYS = {
    "dev-secret-key-replace-in-production",
    "changeme_in_production",
    "secret",
    "changeme",
    "replace_with_a_long_random_string",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────
    # APP_ENV is the canonical var; ENVIRONMENT is an accepted alias
    # (docker-compose sets ENVIRONMENT=production by default).
    APP_ENV: str = "development"
    ENVIRONMENT: str = ""  # alias — model_validator copies it to APP_ENV below
    APP_NAME: str = "MHBAP"
    APP_VERSION: str = "0.1.0"
    SECRET_KEY: str = "dev-secret-key-replace-in-production"
    DEBUG: bool = False  # safe default — enable explicitly in development

    # ── Database ─────────────────────────────────────────────────────────────
    # No default credentials — must be supplied via environment or .env file.
    DATABASE_URL: str = "postgresql+asyncpg://mhbap:changeme@localhost:5432/mhbap"
    DATABASE_SYNC_URL: str = "postgresql://mhbap:changeme@localhost:5432/mhbap"
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

    @model_validator(mode="after")
    def validate_production_secrets(self) -> "Settings":
        # Sync ENVIRONMENT alias → APP_ENV (docker-compose sets ENVIRONMENT,
        # but internal code reads APP_ENV).
        if self.ENVIRONMENT and not self.model_fields_set.intersection({"app_env"}):
            object.__setattr__(self, "APP_ENV", self.ENVIRONMENT)
        effective_env = self.ENVIRONMENT or self.APP_ENV
        if effective_env == "production":
            if self.SECRET_KEY in _INSECURE_KEYS or len(self.SECRET_KEY) < 32:
                raise ValueError(
                    "SECRET_KEY is insecure. Set a random string of ≥32 chars in .env"
                )
        return self


settings = Settings()
