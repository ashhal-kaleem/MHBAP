"""
Face pipeline — MediaPipe FaceMesh → Action Unit proxy features.

Outputs a dict of 12 AU-proxy floats derived from landmark geometry,
normalised to [0, 1]. Falls back to zeros if frame is None or
MediaPipe import fails (CI / no-camera environments).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

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
        self._mp_drawing = None

    # ------------------------------------------------------------------
    def _ensure_mesh(self) -> bool:
        if self._mesh is not None:
            return True
        try:
            import mediapipe as mp  # type: ignore
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self._det_conf,
                min_tracking_confidence=self._trk_conf,
            )
            return True
        except ImportError:
            return False

    def process(self, frame: Optional[np.ndarray]) -> Dict[str, float]:
        zeros = {k: 0.0 for k in _FEATURE_KEYS}
        if frame is None or not self._ensure_mesh():
            return zeros

        import cv2  # type: ignore
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(rgb)

        if not results.multi_face_landmarks:
            return zeros

        lm = results.multi_face_landmarks[0].landmark
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

        return {
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

    def warm_up(self) -> None:
        self._ensure_mesh()
        self._loaded = True

    def extract(self, data):  # type: ignore[override]
        return self.process(data)

    def close(self) -> None:
        if self._mesh:
            self._mesh.close()
            self._mesh = None
