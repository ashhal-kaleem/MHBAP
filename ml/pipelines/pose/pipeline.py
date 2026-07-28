"""
Pose pipeline — MediaPipe Pose → upper-body keypoints + posture features.

Features (11 floats)
--------------------
shoulder_slope      : tilt angle of shoulder line [-1, 1]
head_forward_tilt   : forward lean proxy [0, 1]
spine_curvature     : hunch proxy [0, 1]
left_arm_raise      : how high left wrist is relative to shoulder [0, 1]
right_arm_raise     : same for right
body_sway_x         : lateral sway normalised [0, 1]
body_sway_y         : vertical sway normalised [0, 1]
head_tilt_x         : left-right head tilt [-1, 1]
head_tilt_y         : up-down head tilt [-1, 1]
torso_rotation      : shoulder vs hip rotation delta [-1, 1]
visibility_score    : mean visibility of key landmarks [0, 1]
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from ml.pipelines.base import BasePipeline


class PosePipeline(BasePipeline):
    MODALITY = "pose"

    def __init__(self) -> None:
        self._pose = None

    def _ensure_pose(self) -> bool:
        if self._pose is not None:
            return True
        try:
            import mediapipe as mp  # type: ignore
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return True
        except ImportError:
            return False

    def process(self, frame: Optional[np.ndarray]) -> Dict[str, float]:
        zeros = {k: 0.0 for k in [
            "shoulder_slope", "head_forward_tilt", "spine_curvature",
            "left_arm_raise", "right_arm_raise",
            "body_sway_x", "body_sway_y",
            "head_tilt_x", "head_tilt_y",
            "torso_rotation", "visibility_score",
        ]}
        if frame is None or not self._ensure_pose():
            return zeros

        import cv2  # type: ignore
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)
        if not results.pose_landmarks:
            return zeros

        lm = results.pose_landmarks.landmark
        # MediaPipe Pose landmark indices
        L_SHOULDER, R_SHOULDER = 11, 12
        L_HIP, R_HIP           = 23, 24
        L_EAR, R_EAR           = 7, 8
        NOSE                    = 0
        L_WRIST, R_WRIST       = 15, 16

        ls = lm[L_SHOULDER]; rs = lm[R_SHOULDER]
        lh = lm[L_HIP];      rh = lm[R_HIP]
        le = lm[L_EAR];      re = lm[R_EAR]
        nose = lm[NOSE]
        lw = lm[L_WRIST];    rw = lm[R_WRIST]

        shoulder_slope  = float(np.clip((rs.y - ls.y) * 10, -1.0, 1.0))
        head_forward_tilt = float(np.clip(
            (((ls.x + rs.x) / 2) - nose.x) * 5, 0.0, 1.0))
        spine_curvature = float(np.clip(
            abs(((ls.x + rs.x) / 2) - ((lh.x + rh.x) / 2)) * 8, 0.0, 1.0))

        shoulder_y = (ls.y + rs.y) / 2
        left_arm_raise  = float(np.clip((shoulder_y - lw.y) * 4, 0.0, 1.0))
        right_arm_raise = float(np.clip((shoulder_y - rw.y) * 4, 0.0, 1.0))

        mid_x = (ls.x + rs.x + lh.x + rh.x) / 4
        mid_y = (ls.y + rs.y + lh.y + rh.y) / 4
        body_sway_x = float(np.clip(abs(mid_x - 0.5) * 4, 0.0, 1.0))
        body_sway_y = float(np.clip(abs(mid_y - 0.5) * 4, 0.0, 1.0))

        head_tilt_x = float(np.clip((le.y - re.y) * 10, -1.0, 1.0))
        head_tilt_y = float(np.clip(
            (nose.y - (le.y + re.y) / 2) * 8, -1.0, 1.0))

        torso_rotation = float(np.clip(
            ((ls.z - rs.z) - (lh.z - rh.z)) * 3, -1.0, 1.0))

        vis_keys = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, NOSE]
        visibility_score = float(np.mean([lm[i].visibility for i in vis_keys]))

        return {
            "shoulder_slope": shoulder_slope, "head_forward_tilt": head_forward_tilt,
            "spine_curvature": spine_curvature, "left_arm_raise": left_arm_raise,
            "right_arm_raise": right_arm_raise, "body_sway_x": body_sway_x,
            "body_sway_y": body_sway_y, "head_tilt_x": head_tilt_x,
            "head_tilt_y": head_tilt_y, "torso_rotation": torso_rotation,
            "visibility_score": visibility_score,
        }

    def warm_up(self) -> None:
        self._ensure_pose()
        self._loaded = True

    def extract(self, data):  # type: ignore[override]
        return self.process(data)

    def close(self) -> None:
        if self._pose:
            self._pose.close()
            self._pose = None
