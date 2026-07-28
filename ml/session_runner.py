"""
SessionRunner — orchestrates capture threads, pipeline calls, and DB writes.

Flow per tick (default 15 fps)
-------------------------------
1. CameraCapture.get_frame()   → FacePipeline, GazePipeline, PosePipeline
2. MicrophoneCapture.get_chunk() → VoicePipeline  (when chunk ready)
3. HCIListener.get_events()    → HCIPipeline
4. All feature dicts → DataWriter.write()

Usage (async context)
---------------------
async with SessionRunner(session_id=...) as runner:
    await runner.run_until_stopped()
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional
from uuid import UUID

from ml.capture.camera import CameraCapture
from ml.capture.microphone import MicrophoneCapture
from ml.capture.hci_listener import HCIListener
from ml.pipelines.face.pipeline import FacePipeline
from ml.pipelines.gaze.pipeline import GazePipeline
from ml.pipelines.pose.pipeline import PosePipeline
from ml.pipelines.voice.pipeline import VoicePipeline
from ml.pipelines.hci.pipeline import HCIPipeline
from ml.data_writer import DataWriter

logger = logging.getLogger(__name__)

_TICK_HZ = 15  # pipeline invocation rate


class SessionRunner:
    def __init__(self, session_id: UUID, fps: int = _TICK_HZ) -> None:
        self.session_id = session_id
        self._fps = fps
        self._stop = asyncio.Event()

        # Capture
        self._cam = CameraCapture(fps=fps)
        self._mic = MicrophoneCapture()
        self._hci = HCIListener()

        # Pipelines
        self._face = FacePipeline()
        self._gaze = GazePipeline()
        self._pose = PosePipeline()
        self._voice = VoicePipeline()
        self._hci_pipe = HCIPipeline()

        # Writer
        self._writer = DataWriter()

    # ------------------------------------------------------------------
    async def __aenter__(self) -> "SessionRunner":
        await self._writer.start()
        self._cam.start()
        self._mic.start()
        self._hci.start()
        return self

    async def __aexit__(self, *_) -> None:
        self._stop.set()
        self._cam.stop()
        self._mic.stop()
        self._hci.stop()
        self._face.close()
        self._gaze.close()
        self._pose.close()
        await self._writer.stop()

    def stop(self) -> None:
        """Signal run_until_stopped() to exit cleanly."""
        self._stop.set()

    # ------------------------------------------------------------------
    async def run_until_stopped(self) -> None:
        interval = 1.0 / self._fps
        while not self._stop.is_set():
            t0 = asyncio.get_event_loop().time()
            await self._tick()
            elapsed = asyncio.get_event_loop().time() - t0
            await asyncio.sleep(max(0.0, interval - elapsed))

    async def _tick(self) -> None:
        frame = self._cam.get_frame()

        # Vision pipelines (all consume the same frame)
        face_feats = self._face.process(frame)
        gaze_feats = self._gaze.process(frame)
        pose_feats = self._pose.process(frame)

        # Voice (consume whatever chunk is ready)
        audio_chunk = self._mic.get_chunk()
        voice_feats = self._voice.process(audio_chunk)

        # HCI (drain event buffer)
        hci_events  = self._hci.get_events()
        hci_feats   = self._hci_pipe.process(hci_events)

        # Persist
        for modality, feats in [
            ("face",  face_feats),
            ("gaze",  gaze_feats),
            ("pose",  pose_feats),
            ("voice", voice_feats),
            ("hci",   hci_feats),
        ]:
            await self._writer.write(self.session_id, modality, feats)
