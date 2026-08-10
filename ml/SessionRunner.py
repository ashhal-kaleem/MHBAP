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
import uuid as _uuid_mod
from datetime import datetime, timezone
from typing import Optional, Tuple
from uuid import UUID
import numpy as np

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
from app.core.RedisStreamBus import publish as _bus_publish

from loguru import logger

_TICK_HZ = 15  # pipeline invocation rate


def _crop_face(
    frame: np.ndarray,
    bbox_norm: Tuple[float, float, float, float],
    padding: float = 0.25,
) -> Optional[np.ndarray]:
    """
    Expand the normalized bbox by `padding` fraction, clamp to frame bounds,
    and return the BGR crop. Returns None if the crop is degenerate (<8px).
    """
    if frame is None or bbox_norm is None:
        return None

    h, w = frame.shape[:2]
    nx0, ny0, nx1, ny1 = bbox_norm

    x0 = int(nx0 * w)
    y0 = int(ny0 * h)
    x1 = int(nx1 * w)
    y1 = int(ny1 * h)

    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)

    px = int(bw * padding)
    py = int(bh * padding)

    cx0 = max(0, x0 - px)
    cy0 = max(0, y0 - py)
    cx1 = min(w, x1 + px)
    cy1 = min(h, y1 + py)

    if (cx1 - cx0) < 64 or (cy1 - cy0) < 64:
        # Crop too small – likely a partial face or detection error.
        logger.warning(
            "SessionRunner: crop too small ({}x{}) after clamping – skipping emotion prediction.",
            cx1 - cx0,
            cy1 - cy0,
        )
        return None

    return frame[cy0:cy1, cx0:cx1]


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
        logger.info("SessionRunner.__aenter__: starting capture threads for session={}", self.session_id)
        await self._writer.start()
        for name, cap in [("camera", self._cam), ("mic", self._mic), ("hci", self._hci)]:
            try:
                cap.start()
                logger.info("SessionRunner: capture '{}' started", name)
            except Exception as exc:
                logger.warning("SessionRunner capture '{}' startup notice: {}", name, exc)
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

    _tick_count: int = 0  # class-level counter shared across instances is wrong; use instance

    async def _tick(self) -> None:
        if not hasattr(self, "_tick_n"):
            self._tick_n = 0
        self._tick_n += 1

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

        # ── face crop for EmotionRecognizer ──────────────────────────────
        face_crop = None
        if frame is not None and getattr(self._face, "last_face_bbox", None) is not None:
            face_crop = _crop_face(frame, self._face.last_face_bbox)
        # ────────────────────────────────────────────────────────────────

        prediction = self._predictor.predict(feature_dicts, bgr_frame=face_crop)
        if prediction.feature_vector is not None:
            shap = self._explainer.explain(prediction.feature_vector)
            self.latest_shap = shap.get("stress", {})
            self.latest_explanation = generate_explanation(
                prediction, self.latest_shap, head="stress"
            )
        self.latest_prediction = prediction

        # ── End-to-end pipeline diagnostic (every ~1 s) ──────────────────
        top_emo = max(prediction.emotion_scores, key=prediction.emotion_scores.get) if prediction.emotion_scores else "?"
        top_score = prediction.emotion_scores.get(top_emo, 0.0)

        if self._tick_n <= 30:
            # Detailed per‑frame diagnostic (first 30 frames)
            logger.info(
                "DIAGNOSTIC tick={} | crop={} | probs={} | top={:.3f}",
                self._tick_n,
                face_crop.shape if face_crop is not None else "NONE",
                prediction.emotion_scores,
                top_score,
            )
        if self._tick_n % 15 == 0:
            face_nonzero = sum(1 for v in face_feats.values() if v != 0.0) if face_feats else 0
            shap_str = ", ".join(f"{k}={v:.3f}" for k, v in self.latest_shap.items()) if self.latest_shap else "(empty)"
            logger.info(
                "PIPELINE tick={} | frame={} | crop={} | face_AU_nonzero={}/{} | "
                "emotion={}({:.3f}) src={} | stress={:.3f} eng={:.3f} att={:.3f} fat={:.3f} | "
                "SHAP_stress=[{}]",
                self._tick_n,
                "real" if frame is not None else "NONE",
                face_crop.shape if face_crop is not None else "NONE",
                face_nonzero,
                len(face_feats) if face_feats else 0,
                prediction.emotion, top_score, prediction.emotion_source,
                prediction.stress, prediction.engagement, prediction.attention, prediction.fatigue,
                shap_str,
            )

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
