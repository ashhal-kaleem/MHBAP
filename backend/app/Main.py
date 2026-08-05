"""
MHBAP FastAPI application entry point.

Lifespan context manager handles startup/shutdown of:
  - database connection pool
  - Redis connection
  - ML model loading (Phase 5+)
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.core.Config import settings
from app.core.Logging import setup_logging
from app.core.SecurityHeaders import SecurityHeadersMiddleware
from app.core.ContentSize import ContentSizeLimitMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup → yield → shutdown."""
    setup_logging()
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    from app.db.Session import get_engine
    from app.core.Redis import get_redis

    engine = get_engine()  # opens the pool lazily; first real query connects
    from app.db.Base import Base
    import app.db.models  # noqa: F401
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    get_redis()

    # Phase 5: pre-load ML model weights here

    yield

    logger.info("Shutting down MHBAP")
    from app.core.Redis import close_redis
    from app.db.Session import dispose_engine

    await dispose_engine()
    await close_redis()


def create_app() -> FastAPI:
    """Application factory — enables testing with different configs."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Multimodal Human Behavior Analysis Platform API",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    # Security headers
    app.add_middleware(
        SecurityHeadersMiddleware,
        production=getattr(settings, "APP_ENV", "development") == "production",
    )

    # Content-size guard (10 MB default)
    app.add_middleware(ContentSizeLimitMiddleware)

    # CORS
    # allow_credentials=True is incompatible with allow_origins=["*"] in
    # Starlette 0.30+ and is unnecessary for token-based auth (Bearer via
    # query param).  Only enable it when explicit origins are configured.
    _cors_origins = [str(o) for o in settings.CORS_ORIGINS]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins or ["*"],
        allow_credentials=bool(_cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers — added phase by phase
    from app.api.v1 import router as v1_router
    app.include_router(v1_router, prefix="/api/v1")

    # Global exception handler — never leak stack traces to clients
    from fastapi import Request as _Request
    from fastapi.responses import JSONResponse as _JSONResponse

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: _Request, exc: Exception) -> _JSONResponse:
        logger.exception(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
        return _JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return app


app = create_app()
