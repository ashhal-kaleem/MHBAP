"""
runner.py — API endpoints to start/stop the real-time SessionRunner.

Routes
------
POST /api/v1/Runner/Session/{session_id}/start
    Launch a SessionRunner background task for the given session.
    The runner captures camera/mic/HCI, runs all pipelines, performs TCMT
    inference, and publishes predictions to stream_bus so the WebSocket
    handler at /api/v1/Stream/Session/{session_id} receives them.

POST /api/v1/Runner/Session/{session_id}/stop
    Signal the running task to stop and wait for cleanup.

GET  /api/v1/Runner/Session/{session_id}/status
    Returns {"session_id": ..., "running": bool}.

Design notes
------------
- Uses an asyncio background task so uvicorn's event loop drives both the
  runner _tick() loop and the WebSocket handler concurrently.
- DataWriter (DB persistence) errors are non-fatal — the runner continues
  streaming predictions even when Postgres is unavailable.
- At most one runner per session_id.  A second POST /start returns 409.
- Cleanup is registered in the task's done callback so crash/cancellation
  is always handled.
"""
from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.Session import get_db
from app.api.Dependencies import require_roles, RUNNER_ALLOWED_ROLES
from app.services import SessionService as session_service

logger = logging.getLogger(__name__)
router = APIRouter()

# session_id (str) → asyncio.Task
_active_runners: dict[str, asyncio.Task] = {}
# session_id (str) → SessionRunner instance (for stop signal)
_runner_objects: dict[str, object] = {}


# ── response schema ────────────────────────────────────────────────────────

class RunnerStatus(BaseModel):
    session_id: str
    running: bool


# ── helpers ────────────────────────────────────────────────────────────────

async def _run_session(session_id: str) -> None:
    """Entry-point for the background task.  Wraps SessionRunner lifecycle."""
    from uuid import UUID as _UUID
    
    def _import_session_runner():
        from ml.SessionRunner import SessionRunner
        return SessionRunner

    SessionRunner = await asyncio.to_thread(_import_session_runner)

    sid = _UUID(session_id)
    logger.info("SessionRunner starting for session=%s", session_id)
    try:
        async with SessionRunner(session_id=sid) as runner:
            _runner_objects[session_id] = runner
            await runner.run_until_stopped()
    except asyncio.CancelledError:
        logger.info("SessionRunner task cancelled for session=%s", session_id)
    except Exception as exc:
        logger.error("SessionRunner error for session=%s: %s", session_id, exc, exc_info=True)
    finally:
        _runner_objects.pop(session_id, None)
        logger.info("SessionRunner stopped for session=%s", session_id)


def _task_done_callback(session_id: str, task: asyncio.Task) -> None:
    """Remove from tracking dict when the task finishes for any reason."""
    _active_runners.pop(session_id, None)
    if task.cancelled():
        logger.debug("Runner task for session=%s was cancelled", session_id)
    elif task.exception():
        logger.error("Runner task for session=%s raised: %s", session_id, task.exception())


# ── endpoints ──────────────────────────────────────────────────────────────

@router.post("/Session/{session_id}/start", response_model=RunnerStatus, status_code=202)
@router.post("/session/{session_id}/start", response_model=RunnerStatus, status_code=202, include_in_schema=False)
async def start_runner(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
    auth: tuple = Depends(require_roles(list(RUNNER_ALLOWED_ROLES))),
) -> RunnerStatus:
    """
    Launch a SessionRunner background task for ``session_id``.

    Returns 409 if a runner is already active for this session.
    Returns 202 Accepted — the task is started asynchronously; predictions
    will appear on the WebSocket stream within the first pipeline tick (~67 ms
    at 15 fps).
    """
    current_user_id, _role = auth
    sid = str(session_id)

    session = await session_service.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if str(session.user_id) != current_user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this resource")

    if sid in _active_runners and not _active_runners[sid].done():
        raise HTTPException(
            status_code=409,
            detail=f"Runner already active for session {sid}",
        )

    task = asyncio.create_task(_run_session(sid), name=f"runner-{sid[:8]}")
    task.add_done_callback(lambda t: _task_done_callback(sid, t))
    _active_runners[sid] = task

    logger.info("Runner task created for session=%s", sid)
    return RunnerStatus(session_id=sid, running=True)


@router.post("/Session/{session_id}/stop", response_model=RunnerStatus)
@router.post("/session/{session_id}/stop", response_model=RunnerStatus, include_in_schema=False)
async def stop_runner(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
    auth: tuple = Depends(require_roles(list(RUNNER_ALLOWED_ROLES))),
) -> RunnerStatus:
    """
    Signal the SessionRunner for ``session_id`` to stop cleanly.

    Calls ``runner.stop()`` (sets the internal asyncio.Event) then cancels
    the background task.  Returns immediately; cleanup happens asynchronously.
    Returns 404 if no runner is active.
    """
    current_user_id, _role = auth
    sid = str(session_id)

    session = await session_service.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if str(session.user_id) != current_user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this resource")

    runner = _runner_objects.get(sid)
    if runner is not None:
        # Signal the run_until_stopped loop to exit gracefully
        runner.stop()  # type: ignore[attr-defined]

    task = _active_runners.get(sid)
    if task is None or task.done():
        # Already stopped — idempotent OK
        return RunnerStatus(session_id=sid, running=False)

    task.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=3.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass  # expected — task was cancelled

    return RunnerStatus(session_id=sid, running=False)


@router.get("/Session/{session_id}/status", response_model=RunnerStatus)
@router.get("/session/{session_id}/status", response_model=RunnerStatus, include_in_schema=False)
async def runner_status(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
    auth: tuple = Depends(require_roles(list(RUNNER_ALLOWED_ROLES))),
) -> RunnerStatus:
    """Return whether a runner task is currently active for ``session_id``."""
    current_user_id, _role = auth
    sid = str(session_id)

    session = await session_service.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if str(session.user_id) != current_user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this resource")

    task = _active_runners.get(sid)
    running = task is not None and not task.done()
    return RunnerStatus(session_id=sid, running=running)
