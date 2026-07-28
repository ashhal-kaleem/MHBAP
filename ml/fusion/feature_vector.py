"""feature_vector.py — canonical feature layout for MHBAP fusion.

Total 58 dims:  face[0:12] gaze[12:17] pose[17:28] voice[28:48] hci[48:58]
"""
from __future__ import annotations
from typing import Dict, List
import numpy as np

FACE_KEYS: List[str] = [
    "au_brow_raise_l","au_brow_raise_r","au_brow_furrow",
    "au_eye_open_l","au_eye_open_r","au_nose_wrinkle",
    "au_lip_corner_pull_l","au_lip_corner_pull_r","au_lip_press",
    "au_jaw_drop","au_cheek_raise","au_chin_raise",
]
GAZE_KEYS: List[str] = ["gaze_x","gaze_y","blink_l","blink_r","fixation_stability"]
POSE_KEYS: List[str] = [
    "shoulder_slope","head_forward_tilt","spine_curvature",
    "left_arm_raise","right_arm_raise","body_sway_x","body_sway_y",
    "head_tilt_x","head_tilt_y","torso_rotation","visibility_score",
]
VOICE_KEYS: List[str] = (
    [f"mfcc_{i}" for i in range(1, 14)]
    + ["pitch_mean","pitch_std","energy","zcr","spectral_centroid","speaking_rate_proxy"]
)
HCI_KEYS: List[str] = [
    "mouse_speed","mouse_acceleration","click_rate","scroll_intensity",
    "keystroke_rate","dwell_time","error_rate_proxy","typing_rhythm_std",
    "mouse_pause_ratio","interaction_entropy",
]
MODALITY_KEYS: Dict[str, List[str]] = {
    "face": FACE_KEYS, "gaze": GAZE_KEYS, "pose": POSE_KEYS,
    "voice": VOICE_KEYS, "hci": HCI_KEYS,
}
FEATURE_DIM = sum(len(v) for v in MODALITY_KEYS.values())  # 58
