"""
DaiseeDataset.py — real temporal benchmark data for MHBAP (DAiSEE).

DAiSEE (Gupta et al., 2016, arXiv:1609.01885): 9,068 ten-second webcam clips
from 112 subjects, each labelled 0-3 (very low .. very high) for four
affective states: Boredom, Engagement, Confusion, Frustration.
Labels are crowd-annotated and checked against a psychologist gold standard.

Unlike ml/training/RealDataset.py (static images + random-noise modalities),
every feature here comes from real video frames, and every label is a real
annotation. We use the OFFICIAL Train / Validation / Test splits, which are
subject-disjoint, so test results are not inflated by identity leakage.

Expected layout after downloading and unzipping DAiSEE:

    DAiSEE/
      DataSet/{Train,Validation,Test}/<subject>/<clip_id>/<clip_id>.avi|.mp4
      Labels/{TrainLabels,ValidationLabels,TestLabels}.csv

Visual modalities only (DAiSEE has no usable audio or HCI streams):
    face (12) + gaze (5) + pose (11) = 28 dims per frame.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ml.fusion.FeatureVector import FACE_KEYS, GAZE_KEYS, POSE_KEYS

AFFECTIVE_STATES: List[str] = ["Boredom", "Engagement", "Confusion", "Frustration"]
N_LEVELS = 4
SPLITS: Dict[str, str] = {
    "train": "Train",
    "val": "Validation",
    "test": "Test",
}
LABEL_FILES: Dict[str, str] = {
    "train": "TrainLabels.csv",
    "val": "ValidationLabels.csv",
    "test": "TestLabels.csv",
}

VISUAL_MODALITIES: Dict[str, List[str]] = {
    "face": FACE_KEYS,
    "gaze": GAZE_KEYS,
    "pose": POSE_KEYS,
}
VISUAL_SLICES: Dict[str, Tuple[int, int]] = {}
_o = 0
for _m, _k in VISUAL_MODALITIES.items():
    VISUAL_SLICES[_m] = (_o, _o + len(_k))
    _o += len(_k)
VISUAL_DIM = _o  # 28


# ── labels ──────────────────────────────────────────────────────────────────

def read_labels(csv_path: Path) -> Dict[str, np.ndarray]:
    """Parse a DAiSEE label CSV -> {clip_stem: int array (4,)}.

    Handles the trailing-space header ("Frustration ") in the official files.
    """
    out: Dict[str, np.ndarray] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip() for h in (reader.fieldnames or [])]
        for row in reader:
            row = {k.strip(): v for k, v in row.items() if k is not None}
            clip = Path(row["ClipID"].strip()).stem
            out[clip] = np.array([int(row[s]) for s in AFFECTIVE_STATES], dtype=np.int64)
    return out


def find_clips(split_dir: Path) -> Dict[str, Tuple[str, Path]]:
    """Map clip_stem -> (subject_id, video_path) for one split directory."""
    clips: Dict[str, Tuple[str, Path]] = {}
    for ext in ("*.avi", "*.mp4"):
        for p in split_dir.glob(f"*/*/{ext}"):
            subject = p.parent.parent.name
            clips[p.stem] = (subject, p)
    return clips


# ── frame sampling + feature extraction ────────────────────────────────────

def sample_frame_indices(n_frames: int, t: int) -> np.ndarray:
    """T indices spread uniformly over the clip (centre of each segment)."""
    if n_frames <= 0:
        return np.zeros(t, dtype=np.int64)
    edges = np.linspace(0, n_frames, t + 1)
    return np.clip(((edges[:-1] + edges[1:]) / 2).astype(np.int64), 0, n_frames - 1)


def read_frames(video_path: Path, t: int) -> List[Optional[np.ndarray]]:
    """Read T uniformly sampled BGR frames; None where decoding fails."""
    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: List[Optional[np.ndarray]] = []
    for idx in sample_frame_indices(n, t):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        frames.append(frame if ok else None)
    cap.release()
    return frames


class VisualExtractor:
    """Runs the SAME face/gaze/pose pipelines used at inference time,
    so the benchmark measures the deployed feature extractor."""

    def __init__(self) -> None:
        from ml.pipelines.face.Pipeline import FacePipeline
        from ml.pipelines.gaze.Pipeline import GazePipeline
        from ml.pipelines.pose.Pipeline import PosePipeline

        self.face = FacePipeline()
        self.gaze = GazePipeline()
        self.pose = PosePipeline()

    def reset(self) -> None:
        # Gaze keeps a fixation-history buffer; clear it between clips.
        hist = getattr(self.gaze, "_history", None)
        if hist is not None:
            hist.clear()

    def frame_features(self, frame: Optional[np.ndarray]) -> Tuple[np.ndarray, bool]:
        vec = np.zeros(VISUAL_DIM, dtype=np.float32)
        if frame is None:
            return vec, False
        dicts = {
            "face": self.face.process(frame),
            "gaze": self.gaze.process(frame),
            "pose": self.pose.process(frame),
        }
        for mod, keys in VISUAL_MODALITIES.items():
            s, _ = VISUAL_SLICES[mod]
            for i, k in enumerate(keys):
                vec[s + i] = float(dicts[mod].get(k, 0.0))
        face_found = bool(np.any(vec[slice(*VISUAL_SLICES["face"])] != 0))
        return vec, face_found

    def clip_features(self, video_path: Path, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """-> X (T, 28), valid (T,) bool (face detected in frame)."""
        self.reset()
        X = np.zeros((t, VISUAL_DIM), dtype=np.float32)
        valid = np.zeros(t, dtype=bool)
        for i, fr in enumerate(read_frames(video_path, t)):
            X[i], valid[i] = self.frame_features(fr)
        return X, valid


def extract_split(
    root: Path,
    split: str,
    t: int = 32,
    limit: Optional[int] = None,
    extractor: Optional[VisualExtractor] = None,
) -> Dict[str, np.ndarray]:
    """Extract features for one official split. Returns arrays ready for np.savez."""
    labels = read_labels(root / "Labels" / LABEL_FILES[split])
    clips = find_clips(root / "DataSet" / SPLITS[split])
    ids = sorted(set(labels) & set(clips))
    if limit:
        ids = ids[:limit]
    ex = extractor or VisualExtractor()
    X = np.zeros((len(ids), t, VISUAL_DIM), dtype=np.float32)
    V = np.zeros((len(ids), t), dtype=bool)
    Y = np.zeros((len(ids), len(AFFECTIVE_STATES)), dtype=np.int64)
    subj: List[str] = []
    for n, cid in enumerate(ids):
        s, path = clips[cid]
        X[n], V[n] = ex.clip_features(path, t)
        Y[n] = labels[cid]
        subj.append(s)
        if (n + 1) % 100 == 0:
            print(f"[daisee:{split}] {n + 1}/{len(ids)}", flush=True)
    missing = len(labels) - len(set(labels) & set(clips))
    if missing:
        print(f"[daisee:{split}] WARN {missing} labelled clips not found on disk", flush=True)
    return {
        "X": X, "valid": V, "y": Y,
        "clip_ids": np.array(ids), "subjects": np.array(subj),
    }


# ── normalisation + torch dataset ───────────────────────────────────────────

def fit_normaliser(X: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std over valid frames of the TRAIN split only."""
    flat = X[valid] if valid.any() else X.reshape(-1, X.shape[-1])
    mu = flat.mean(0)
    sd = flat.std(0)
    sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_normaliser(X: np.ndarray, valid: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    Z = (X - mu) / sd
    Z[~valid] = 0.0  # invalid frames -> zero (and masked in the model)
    return Z.astype(np.float32)


def class_weights(y: np.ndarray, n_levels: int = N_LEVELS) -> np.ndarray:
    """Inverse-frequency weights per (state, level) -> (4, n_levels).
    DAiSEE is heavily imbalanced (very few 'very low engagement' clips)."""
    w = np.zeros((y.shape[1], n_levels), dtype=np.float32)
    for s in range(y.shape[1]):
        counts = np.bincount(y[:, s], minlength=n_levels).astype(np.float32)
        counts[counts == 0] = 1.0
        w[s] = counts.sum() / (n_levels * counts)
    return w


def synthetic_split(n: int, t: int = 16, seed: int = 0) -> Dict[str, np.ndarray]:
    """Tiny synthetic split with a learnable temporal signal (tests only)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, t, VISUAL_DIM)).astype(np.float32)
    y = rng.integers(0, N_LEVELS, size=(n, len(AFFECTIVE_STATES)))
    # Engagement level is encoded as a trend in gaze_x over time.
    gs = VISUAL_SLICES["gaze"][0]
    ramp = np.linspace(-1, 1, t, dtype=np.float32)
    X[:, :, gs] += (y[:, 1:2] - 1.5) * ramp[None, :]
    return {
        "X": X, "valid": np.ones((n, t), dtype=bool), "y": y.astype(np.int64),
        "clip_ids": np.array([f"c{i}" for i in range(n)]),
        "subjects": np.array([f"s{i % 7}" for i in range(n)]),
    }
