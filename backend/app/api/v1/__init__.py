"""API v1 router — aggregates all endpoint modules."""
from fastapi import APIRouter
from app.api.v1.endpoints import (
    Analytics as analytics,
    Auth as auth,
    Evaluation as evaluation,
    Health as health,
    Predictions as predictions,
    Runner as runner,
    Sessions as sessions,
    Stream as stream,
    Users as users,
)

router = APIRouter()
router.include_router(health.router, prefix="/health", tags=["health"])
router.include_router(auth.router, prefix="/auth", tags=["auth"])
router.include_router(users.router, prefix="/users", tags=["users"])
router.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
router.include_router(predictions.router, prefix="/predictions", tags=["predictions"])
router.include_router(stream.router, prefix="/stream", tags=["streaming"])
router.include_router(runner.router, prefix="/runner", tags=["runner"])
router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
router.include_router(evaluation.router, prefix="/evaluation", tags=["evaluation"])
