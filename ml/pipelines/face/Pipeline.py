"""
Face pipeline — MediaPipe FaceMesh → Action Unit proxy features.

Outputs a dict of 12 AU-proxy floats derived from landmark geometry,
normalised to [0, 1]. Falls back to zeros if frame is None or
MediaPipe import fails (CI / no-camera environments).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
from loguru import logger

from ml.pipelines.Base import BasePipeline

_FEATURE_KEYS = [
    "au_brow_raise_l", "au_brow_raise_r",
    "au_brow_furrow",
    "au_eye_open_l", "au_eye_open_r",
    "au_nose_wrinkle",
    "au_lip_corner_pull_l", "au_lip_corner_pull_r",
    "au_lip_press",
    "au_jaw_drop",
    "au_cheek_raise",
    "au_chin_raise",
]

def _dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


class FacePipeline(BasePipeline):
    """
    Processes a BGR frame → AU proxy feature dict.

    Parameters
    ----------
    min_detection_confidence : float
    min_tracking_confidence  : float
    """

    MODALITY = "face"

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._det_conf = min_detection_confidence
        self._trk_conf = min_tracking_confidence
        self._mesh = None  # lazy-init on first call
        self._mp_available: Optional[bool] = None  # None = not yet checked
        self._mp_drawing = None
        self._frame_count = 0
        self._first_frame_logged = False
        # Normalized face bounding box from the most recent detection.
        # Set to (x_min, y_min, x_max, y_max) in [0, 1] when a face is found,
        # or None when no face is detected.  Read by SessionRunner to crop the
        # face region before passing it to EmotionRecognizer.
        self.last_face_bbox: Optional[Tuple[float, float, float, float]] = None

    # ------------------------------------------------------------------
    def _ensure_mesh(self) -> bool:
        if self._mesh is not None:
            return True
        # Already tried and failed — don't retry every frame, but DO log loudly once
        if self._mp_available is False:
            return False
        try:
            import os
            import pathlib
            import urllib.request
            import mediapipe as mp  # type: ignore
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            weights_dir = pathlib.Path(__file__).parent.parent.parent / "models" / "weights"
            task_path = weights_dir / "face_landmarker.task"
            
            if not task_path.exists():
                logger.info("FacePipeline: Downloading MediaPipe face_landmarker.task...")
                weights_dir.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                    str(task_path)
                )

            base_options = python.BaseOptions(model_asset_path=str(task_path))
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=self._det_conf,
                min_tracking_confidence=self._trk_conf,
            )
            self._mesh = vision.FaceLandmarker.create_from_options(options)
            
            self._mp_available = True
            logger.info(
                "FacePipeline: MediaPipe Tasks FaceLandmarker initialised OK "
                "(det_conf={}, trk_conf={})",
                self._det_conf, self._trk_conf,
            )
            return True
        except ImportError as exc:
            self._mp_available = False
            logger.error(
                "FacePipeline: mediapipe NOT INSTALLED — face AU features will be "
                "ALL ZEROS every frame. Install with: uv add mediapipe  (error: {})",
                exc,
            )
            return False
        except Exception as exc:
            self._mp_available = False
            logger.error("FacePipeline: FaceMesh init failed unexpectedly: {}", exc)
            return False

    def process(self, frame: Optional[np.ndarray]) -> Dict[str, float]:
        zeros = {k: 0.0 for k in _FEATURE_KEYS}
        if frame is None:
            return zeros

        if not self._ensure_mesh():
            return zeros

        # Log frame properties on the first call so we can verify shape/dtype/channels
        if not self._first_frame_logged:
            self._first_frame_logged = True
            logger.info(
                "FacePipeline: FIRST FRAME received — shape={} dtype={} "
                "channels={} min={} max={}",
                frame.shape,
                frame.dtype,
                frame.shape[2] if frame.ndim == 3 else "N/A",
                int(frame.min()),
                int(frame.max()),
            )

        import cv2  # type: ignore
        import mediapipe as mp
        # Frame is BGR (from OpenCV VideoCapture); MediaPipe needs RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._mesh.detect(mp_image)

        self._frame_count += 1
        detected = bool(results.face_landmarks)

        # Log every frame for the first 30 frames, then every 30 frames
        if self._frame_count <= 30 or self._frame_count % 30 == 0:
            logger.info(
                "FacePipeline: frame={} shape={} face_detected={}",
                self._frame_count,
                frame.shape,
                detected,
            )

        if not detected:
            self.last_face_bbox = None
            return zeros

        lm = results.face_landmarks[0]

        # ── Compute normalized face bounding box from all 468 landmarks ──────
        # Using all points is more stable than a fixed subset.  The FaceLandmarker
        # returns x/y in [0, 1] normalized image coordinates.
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        nx0, ny0, nx1, ny1 = min(xs), min(ys), max(xs), max(ys)
        self.last_face_bbox = (nx0, ny0, nx1, ny1)

        # ── Input-quality warning: face near or beyond frame edge ─────────────
        # MediaPipe can return landmark coords slightly outside [0, 1] when the
        # face is at the webcam boundary.  The downstream _crop_face() will clamp
        # safely, but the crop will be geometrically truncated (partial face).
        if nx0 < 0.0 or ny0 < 0.0 or nx1 > 1.0 or ny1 > 1.0:
            logger.warning(
                "FacePipeline: face bbox extends beyond frame boundary on frame={} "
                "bbox_norm=({:.3f},{:.3f},{:.3f},{:.3f}) — face may be partially "
                "outside the webcam view; emotion predictions may be degraded. "
                "Centre your face in the webcam for best accuracy.",
                self._frame_count, nx0, ny0, nx1, ny1,
            )
        # ──────────────────────────────────────────────────────────────────────

        # Landmark indices per MediaPipe FaceMesh 468-point map
        # Brow raise: distance between brow tip and eye corner
        left_brow   = lm[105]; left_eye_top   = lm[159]
        right_brow  = lm[334]; right_eye_top  = lm[386]
        brow_mid_l  = lm[107]; brow_mid_r     = lm[336]
        nose_tip    = lm[4]
        upper_lip   = lm[13]; lower_lip = lm[14]
        lip_l       = lm[61]; lip_r     = lm[291]
        jaw         = lm[152]; chin      = lm[175]
        l_eye_top   = lm[159]; l_eye_bot = lm[145]
        r_eye_top   = lm[386]; r_eye_bot = lm[374]

        face_h = _dist(lm[10], lm[152]) or 1e-6  # normalisation ref

        feats = {
            "au_brow_raise_l":      min(1.0, _dist(left_brow,  left_eye_top)  / face_h * 5),
            "au_brow_raise_r":      min(1.0, _dist(right_brow, right_eye_top) / face_h * 5),
            "au_brow_furrow":       1.0 - min(1.0, _dist(brow_mid_l, brow_mid_r) / face_h * 4),
            "au_eye_open_l":        min(1.0, _dist(l_eye_top, l_eye_bot) / face_h * 10),
            "au_eye_open_r":        min(1.0, _dist(r_eye_top, r_eye_bot) / face_h * 10),
            "au_nose_wrinkle":      min(1.0, abs(nose_tip.z) * 3),
            "au_lip_corner_pull_l": min(1.0, max(0.0, lip_l.x - lm[0].x) / face_h * 8),
            "au_lip_corner_pull_r": min(1.0, max(0.0, lm[0].x - lip_r.x) / face_h * 8),
            "au_lip_press":         1.0 - min(1.0, _dist(upper_lip, lower_lip) / face_h * 12),
            "au_jaw_drop":          min(1.0, _dist(jaw, chin) / face_h * 6),
            "au_cheek_raise":       min(1.0, abs(lm[117].z + lm[346].z) * 2),
            "au_chin_raise":        min(1.0, abs(lm[199].z) * 4),
        }

        # Log the AU values on the first 5 detections and every 30 frames after
        if self._frame_count <= 5 or (self._frame_count % 30 == 0 and detected):
            nonzero = sum(1 for v in feats.values() if v > 0.0)
            x0, y0, x1, y1 = self.last_face_bbox
            logger.info(
                "FacePipeline: face DETECTED on frame={} nonzero_AUs={}/12 "
                "eye_open_l={:.3f} eye_open_r={:.3f} brow_furrow={:.3f} "
                "bbox_norm=({:.3f},{:.3f},{:.3f},{:.3f})",
                self._frame_count, nonzero,
                feats["au_eye_open_l"],
                feats["au_eye_open_r"],
                feats["au_brow_furrow"],
                x0, y0, x1, y1,
            )
        return feats

    def warm_up(self) -> None:
        self._ensure_mesh()
        self._loaded = True

    def extract(self, data):  # type: ignore[override]
        return self.process(data)

    def close(self) -> None:
        if self._mesh:
            self._mesh.close()
            self._mesh = None
