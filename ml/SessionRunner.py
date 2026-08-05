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
import uuid as _uuid_mod
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from ml.capture.Camera import CameraCapture
from ml.capture.Microphone import MicrophoneCapture
from ml.capture.HciListener import HCIListener
from ml.pipelines.face.Pipeline import FacePipeline
from ml.pipelines.gaze.Pipeline import GazePipeline
from ml.pipelines.pose.Pipeline import PosePipeline
from ml.pipelines.voice.Pipeline import VoicePipeline
from ml.pipelines.hci.Pipeline import HCIPipeline
from ml.DataWriter import DataWriter
from ml.fusion.Predictor import BehaviourPredictor, PredictionResult
from ml.xai.ShapExplainer import SHAPExplainer
from ml.xai.NlExplainer import generate_explanation
from app.core.Redis_stream_bus import publish as _bus_publish

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

        # Writer + predictor + XAI
        self._writer    = DataWriter()
        self._predictor = BehaviourPredictor()
        self._explainer = SHAPExplainer(self._predictor._model)
        self.latest_prediction: Optional[PredictionResult] = None
        self.latest_shap: dict = {}
        self.latest_explanation: str = ""

    # ------------------------------------------------------------------
    async def __aenter__(self) -> "SessionRunner":
        await self._writer.start()
        for name, cap in [("camera", self._cam), ("mic", self._mic), ("hci", self._hci)]:
            try:
                cap.start()
            except Exception as exc:
                logger.warning(f"SessionRunner capture '{name}' startup notice: {exc}")
        return self

    async def __aexit__(self, *_) -> None:
        self._stop.set()
        for cap in (self._cam, self._mic, self._hci):
            try:
                cap.stop()
            except Exception:
                pass
        for pipe in (self._face, self._gaze, self._pose):
            try:
                pipe.close()
            except Exception:
                pass
        try:
            await self._writer.stop()
        except Exception:
            pass

    def stop(self) -> None:
        """Signal run_until_stopped() to exit cleanly."""
        self._stop.set()

    @property
    def running(self) -> bool:
        """Whether the runner is currently active."""
        return not self._stop.is_set()

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
        mouse_events, key_events = self._hci.drain()
        hci_events = (mouse_events, key_events)
        hci_feats   = self._hci_pipe.process(hci_events)

        # Fuse + predict + explain
        feature_dicts = {
            "face": face_feats, "gaze": gaze_feats, "pose": pose_feats,
            "voice": voice_feats, "hci": hci_feats,
        }
        prediction = self._predictor.predict(feature_dicts, bgr_frame=frame)
        if prediction.FeatureVector is not None:
            shap = self._explainer.explain(prediction.FeatureVector)
            self.latest_shap = shap.get("stress", {})
            self.latest_explanation = generate_explanation(
                prediction, self.latest_shap, head="stress"
            )
        self.latest_prediction = prediction

        # Push to any connected WebSocket clients
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()
        await _bus_publish(str(self.session_id), {
            "type": "prediction",
            "payload": {
                "id": str(_uuid_mod.uuid4()),
                "session_id": str(self.session_id),
                "time": now_iso,
                "recorded_at": now_iso,
                "emotion_label": prediction.emotion,
                "emotion_scores": prediction.emotion_scores,
                "stress":     prediction.stress,
                "engagement": prediction.engagement,
                "attention":  prediction.attention,
                "fatigue":    prediction.fatigue,
                "shap_weights":     self.latest_shap,
                "explanation_text": self.latest_explanation,
            },
        })

        # Persist
        for modality, feats in [
            ("face",  face_feats),
            ("gaze",  gaze_feats),
            ("pose",  pose_feats),
            ("voice", voice_feats),
            ("hci",   hci_feats),
        ]:
            await self._writer.write(self.session_id, modality, feats, timestamp=now_dt)
