"""
Pytest fixtures shared across unit tests.
Provides an async HTTP client and overrides settings for test environment.
"""
import pytest
from httpx import AsyncClient, ASGITransport

from backend.app.core.config import settings
from backend.app.main import app


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force test environment for all unit tests."""
    monkeypatch.setattr(settings, "APP_ENV", "testing")
    monkeypatch.setattr(settings, "DEBUG", True)


@pytest.fixture
async def client() -> AsyncClient:
    """Async HTTP client for FastAPI app (no real server needed)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
