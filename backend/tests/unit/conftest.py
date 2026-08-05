"""
Pytest fixtures shared across unit tests.
Provides an async HTTP client and overrides settings for test environment.
"""
import os
import pytest
from httpx import AsyncClient, ASGITransport

from app.core.Config import settings
from app.main import app

# Short WS queue timeout so session-stream tests don't stall for 60 s
os.environ.setdefault("MHBAP_WS_RECV_TIMEOUT", "2")


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force test environment for all unit tests."""
    monkeypatch.setattr(settings, "APP_ENV", "testing")
    monkeypatch.setattr(settings, "DEBUG", True)
    monkeypatch.setenv("MHBAP_WS_RECV_TIMEOUT", "2")


@pytest.fixture
async def client() -> AsyncClient:
    """Async HTTP client for FastAPI app (no real server needed)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
