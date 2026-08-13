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

# Prediction persistence: write one DB row per second (every _PERSIST_EVERY ticks).
# At 15 fps this means ~1 write/s — sufficient for analytics without thrashing Postgres.
_PERSIST_EVERY = 15  # ticks between Prediction DB inserts

# (Preview constants removed — frame publishing disabled; the frontend
#  renders a privacy-safe animated face-mesh driven by prediction payloads.)

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

        # Writer
        self._writer = DataWriter()

        # Fix A: reuse the singleton predictor/explainer pre-loaded at startup
        # (Main.py lifespan). Falls back to constructing a new one if the
        # singleton was not initialised (e.g. test environments).
        try:
            import ml._singleton as _s
            if _s.predictor is not None:
                self._predictor = _s.predictor
                self._explainer = _s.explainer
                logger.info("SessionRunner: reusing pre-loaded ML singleton for session={}", session_id)
            else:
                raise AttributeError("singleton not ready")
        except Exception:
            logger.warning("SessionRunner: ML singleton unavailable — constructing BehaviourPredictor locally")
            self._predictor = BehaviourPredictor()
            self._explainer = SHAPExplainer(self._predictor._model)

        self.latest_prediction: Optional[PredictionResult] = None
        self.latest_shap: dict = {}
        self.latest_explanation: str = ""

        # Cache last non-zero voice features so ticks where no new audio chunk
        # is ready (mic produces a chunk every 250ms but _tick runs at 15fps)
        # still contribute real voice data instead of all-zeros.
        self._last_voice_feats: Optional[dict] = None

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

        # Warn immediately if HCI listener failed — avoids silent all-zero HCI features.
        if not self._hci.started:
            logger.warning(
                "SessionRunner: HCIListener did not start (pynput unavailable or permission denied). "
                "HCI features will be zero for this session. "
                "Fix: ensure pynput is installed (`uv add pynput`) and that the process has "
                "accessibility/input-monitoring permissions."
            )
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

        # Voice — consume whatever chunk is ready.
        # MicrophoneCapture produces one chunk every 250ms (CHUNK_MS) but _tick
        # runs at 15fps (~67ms). On the ~3 ticks between chunks, get_chunk()
        # returns None and VoicePipeline would return all-zeros, making voice
        # contribute nothing to the predictor on those ticks.
        # Fix: cache the last non-zero voice feature dict and reuse it until a
        # fresh chunk arrives — voice state changes slowly relative to 250ms anyway.
        audio_chunk = self._mic.get_chunk()
        if audio_chunk is not None:
            fresh_feats = self._voice.process(audio_chunk)
            # Only update cache when the chunk produced real features (not silent noise)
            if any(v != 0.0 for v in fresh_feats.values()):
                self._last_voice_feats = fresh_feats
            voice_feats = fresh_feats
        elif self._last_voice_feats is not None:
            # Reuse last valid voice features — no new audio chunk this tick
            voice_feats = self._last_voice_feats
        else:
            # No chunk ever received yet (start of session)
            voice_feats = self._voice.process(None)

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
            face_nonzero  = sum(1 for v in face_feats.values() if v != 0.0) if face_feats else 0
            voice_nonzero = sum(1 for v in voice_feats.values() if v != 0.0) if voice_feats else 0
            hci_nonzero   = sum(1 for v in hci_feats.values()  if v != 0.0) if hci_feats  else 0
            voice_src     = "fresh" if audio_chunk is not None else ("cached" if self._last_voice_feats else "zero")
            hci_src       = "live"  if self._hci.started else "DEGRADED(pynput failed)"
            shap_str = ", ".join(f"{k}={v:.3f}" for k, v in self.latest_shap.items()) if self.latest_shap else "(empty)"
            logger.info(
                "PIPELINE tick={} | frame={} | crop={} | face_AU={}/{} | "
                "voice={}/{} [{}] | hci={}/{} [{}] | "
                "emotion={}({:.3f}) src={} | stress={:.3f} eng={:.3f} att={:.3f} fat={:.3f} | "
                "SHAP_stress=[{}]",
                self._tick_n,
                "real" if frame is not None else "NONE",
                face_crop.shape if face_crop is not None else "NONE",
                face_nonzero,  len(face_feats)  if face_feats  else 0,
                voice_nonzero, len(voice_feats) if voice_feats else 0, voice_src,
                hci_nonzero,   len(hci_feats)   if hci_feats   else 0, hci_src,
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

        # Camera preview frames are no longer sent — the frontend renders a
        # privacy-safe animated face-mesh visualisation driven by the live
        # prediction payload above.  JPEG encode + Redis publish removed.

        # Persist Prediction row (once per second — every _PERSIST_EVERY ticks).
        # Deferred imports keep this module importable in ML-only test envs.
        if self._tick_n % _PERSIST_EVERY == 0:
            try:
                from app.db.Session import get_session_factory
                from app.services.PredictionService import create_prediction
                from app.schemas.Prediction import PredictionCreate
                _pdata = PredictionCreate(
                    session_id=self.session_id,
                    emotion_label=prediction.emotion,
                    emotion_scores=prediction.emotion_scores,
                    stress=prediction.stress,
                    engagement=prediction.engagement,
                    attention=prediction.attention,
                    fatigue=prediction.fatigue,
                    shap_weights=self.latest_shap,
                    explanation_text=self.latest_explanation,
                )
                async with get_session_factory()() as _db:
                    await create_prediction(_db, _pdata)
            except Exception as _exc:
                logger.warning("SessionRunner: Prediction persist failed (non-fatal): {}", _exc)

        # Persist modality features
        for modality, feats in [
            ("face",  face_feats),
            ("gaze",  gaze_feats),
            ("pose",  pose_feats),
            ("voice", voice_feats),
            ("hci",   hci_feats),
        ]:
            await self._writer.write(self.session_id, modality, feats, timestamp=now_dt)
