"""
Async DB writer — persists ModalityFeatures rows from pipeline output dicts.

Usage
-----
writer = DataWriter()
await writer.write(session_id, modality="face", features={"au_jaw_drop": 0.3, ...})
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.session import get_db
from backend.app.db.models.modality_feature import ModalityFeature

logger = logging.getLogger(__name__)


class DataWriter:
    """
    Wraps async DB insertion with a bounded in-memory queue so the
    capture/pipeline threads don't block on I/O.
    """

    def __init__(self, queue_size: int = 256) -> None:
        self._queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=queue_size)
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    async def start(self) -> None:
        self._task = asyncio.create_task(self._flush_loop(), name="data-writer")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def write(
        self,
        session_id: UUID,
        modality: str,
        features: Dict[str, float],
    ) -> None:
        """Enqueue a feature row — non-blocking; drops if queue is full."""
        try:
            self._queue.put_nowait({
                "session_id": session_id,
                "modality": modality,
                "features": features,
            })
        except asyncio.QueueFull:
            logger.warning("DataWriter queue full — dropping %s frame", modality)

    # ------------------------------------------------------------------
    async def _flush_loop(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                async for db in get_db():
                    row = ModalityFeature(
                        session_id=item["session_id"],
                        modality=item["modality"],
                        features=item["features"],
                    )
                    db.add(row)
                    await db.commit()
            except Exception as exc:
                logger.error("DataWriter flush error: %s", exc)
            finally:
                self._queue.task_done()
