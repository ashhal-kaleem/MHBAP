"""
Gaze pipeline — MediaPipe iris landmarks → gaze vector + blink.

Features
--------
gaze_x, gaze_y   : horizontal / vertical gaze offset [-1, 1]
blink_l, blink_r  : blink probability [0, 1]
fixation_stability: inverse of gaze displacement over last N frames [0, 1]
"""
from __future__ import annotations

import collections
import math
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from ml.pipelines.Base import BasePipeline

_HISTORY = 10  # frames for fixation stability


class GazePipeline(BasePipeline):
    MODALITY = "gaze"

    def __init__(self) -> None:
        self._mesh = None
        self._history: Deque[Tuple[float, float]] = collections.deque(maxlen=_HISTORY)

    def _ensure_mesh(self) -> bool:
        if self._mesh is not None:
            return True
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
                min_face_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._mesh = vision.FaceLandmarker.create_from_options(options)
            return True
        except ImportError:
            return False

    def process(self, frame: Optional[np.ndarray]) -> Dict[str, float]:
        zeros = {"gaze_x": 0.0, "gaze_y": 0.0,
                 "blink_l": 0.0, "blink_r": 0.0,
                 "fixation_stability": 1.0}
        if frame is None or not self._ensure_mesh():
            return zeros

        import cv2  # type: ignore
        import mediapipe as mp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._mesh.detect(mp_image)
        
        if not results.face_landmarks:
            return zeros

        lm = results.face_landmarks[0]

        # Iris centres (refine_landmarks indices)
        l_iris = lm[468]; r_iris = lm[473]

        # Eye corners for reference width
        l_inner = lm[133]; l_outer = lm[33]
        r_inner = lm[362]; r_outer = lm[263]

        l_eye_w = max(abs(l_inner.x - l_outer.x), 1e-6)
        r_eye_w = max(abs(r_inner.x - r_outer.x), 1e-6)

        gaze_x = ((l_iris.x - (l_inner.x + l_outer.x) / 2) / l_eye_w
                  + (r_iris.x - (r_inner.x + r_outer.x) / 2) / r_eye_w) / 2
        gaze_y = ((l_iris.y - (lm[159].y + lm[145].y) / 2)
                  + (r_iris.y - (lm[386].y + lm[374].y) / 2)) / 2

        gaze_x = float(np.clip(gaze_x * 4, -1.0, 1.0))
        gaze_y = float(np.clip(gaze_y * 8, -1.0, 1.0))

        # Blink: eye aspect ratio
        def _ear(top_idx, bot_idx, inner_idx, outer_idx):
            h = abs(lm[top_idx].y - lm[bot_idx].y)
            w = max(abs(lm[inner_idx].x - lm[outer_idx].x), 1e-6)
            return h / w

        blink_l = float(np.clip(1.0 - _ear(159, 145, 133, 33) * 6, 0.0, 1.0))
        blink_r = float(np.clip(1.0 - _ear(386, 374, 362, 263) * 6, 0.0, 1.0))

        # Fixation stability
        self._history.append((gaze_x, gaze_y))
        if len(self._history) > 1:
            xs = [p[0] for p in self._history]
            ys = [p[1] for p in self._history]
            spread = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
            fixation_stability = float(np.clip(1.0 - spread, 0.0, 1.0))
        else:
            fixation_stability = 1.0

        return {"gaze_x": gaze_x, "gaze_y": gaze_y,
                "blink_l": blink_l, "blink_r": blink_r,
                "fixation_stability": fixation_stability}

    def warm_up(self) -> None:
        self._ensure_mesh()
        self._loaded = True

    def extract(self, data):  # type: ignore[override]
        return self.process(data)

    def close(self) -> None:
        if self._mesh:
            self._mesh.close()
            self._mesh = None
