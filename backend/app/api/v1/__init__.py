"""API v1 router — aggregates all endpoint modules."""
from fastapi import APIRouter
from backend.app.api.v1.endpoints import analytics, health, predictions, sessions, stream, users

router = APIRouter()
router.include_router(health.router, prefix="/health", tags=["health"])
router.include_router(users.router, prefix="/users", tags=["users"])
router.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
router.include_router(predictions.router, prefix="/predictions", tags=["predictions"])
router.include_router(stream.router, prefix="/stream", tags=["streaming"])
router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
