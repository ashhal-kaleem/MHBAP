"""
dataset.py — Synthetic labelled dataset for TCMT training.

Labels are computed from deterministic feature rules so the model can actually
learn signal (unlike pure random labels).  Meant for offline training with no
live sensors.

Label rules
-----------
emotion  : AU pattern → happy/neutral/sad/angry (simplified 4-way)
stress   : voice energy + HCI error_rate_proxy + brow_furrow
engagement : gaze fixation + speaking_rate + keystroke_rate
attention  : blink rate (low=attentive) + head_tilt stability
fatigue    : mouse_pause_ratio + dwell_time + low energy
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Optional

from ml.fusion.feature_vector import FEATURE_DIM, MODALITY_KEYS
from ml.fusion.feature_utils import modality_slice


EMOTION_CLASSES = 4   # simplified; full 8-class via mapping
EMOTION_MAP = {0: "neutral", 1: "happy", 2: "sad", 3: "angry"}


def _make_sample(rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
    """Return (feature_vector, label_dict) with consistent signal."""
    x = rng.uniform(0.0, 1.0, FEATURE_DIM).astype(np.float32)

    fs, fe = modality_slice("face")
    gs, ge = modality_slice("gaze")
    ps, pe = modality_slice("pose")
    vs, ve = modality_slice("voice")
    hs, he = modality_slice("hci")

    # Feature indices within each modality slice
    # face: au_lip_corner_pull [6,7], au_jaw_drop[9], au_brow_furrow[2]
    lip_pull  = float(x[fs + 6] + x[fs + 7]) / 2.0
    jaw_drop  = float(x[fs + 9])
    brow_frow = float(x[fs + 2])

    # gaze: blink_l[2], blink_r[3], fixation_stability[4]
    blink_avg = float(x[gs + 2] + x[gs + 3]) / 2.0
    fix_stab  = float(x[gs + 4])

    # voice: energy=index 15 within voice slice (mfcc_1..13, pitch_mean=13,
    #        pitch_std=14, energy=15, zcr=16, spectral_centroid=17, speaking_rate=18)
    energy     = float(x[vs + 15])
    speak_rate = float(x[vs + 18])

    # hci: error_rate_proxy[6], mouse_pause_ratio[8], dwell_time[5], keystroke_rate[4]
    err_rate   = float(x[hs + 6])
    pause_rat  = float(x[hs + 8])
    dwell_time = float(x[hs + 5])
    ksr        = float(x[hs + 4])

    # ── emotion (4-class) ───────────────────────────────────────────────────
    happy_score  = lip_pull - brow_frow + 0.1 * rng.uniform(-1, 1)
    sad_score    = brow_frow - lip_pull + low_energy_term(energy)
    angry_score  = brow_frow + err_rate - lip_pull + 0.1 * rng.uniform(-1, 1)
    neutral_score= 1.0 - abs(happy_score) - abs(sad_score) * 0.5 + 0.05 * rng.uniform(-1, 1)
    scores = [neutral_score, happy_score, sad_score, angry_score]
    emotion = int(np.argmax(scores))

    # ── stress (0-1, then scaled) ────────────────────────────────────────
    stress = float(np.clip(
        0.4 * energy + 0.3 * err_rate + 0.2 * brow_frow + 0.1 * rng.uniform(0, 1),
        0.0, 1.0
    ))

    # ── engagement (0-1) ────────────────────────────────────────────────
    engagement = float(np.clip(
        0.35 * fix_stab + 0.30 * speak_rate + 0.25 * ksr + 0.1 * rng.uniform(0, 1),
        0.0, 1.0
    ))

    # ── attention (0-1) ─────────────────────────────────────────────────
    attention = float(np.clip(
        0.5 * (1.0 - blink_avg) + 0.35 * fix_stab + 0.15 * rng.uniform(0, 1),
        0.0, 1.0
    ))

    # ── fatigue (0-1) ────────────────────────────────────────────────────
    fatigue = float(np.clip(
        0.4 * pause_rat + 0.3 * dwell_time + 0.2 * (1.0 - energy) + 0.1 * rng.uniform(0, 1),
        0.0, 1.0
    ))

    labels = {
        "emotion":    emotion,
        "stress":     stress,
        "engagement": engagement,
        "attention":  attention,
        "fatigue":    fatigue,
    }
    return x, labels


def low_energy_term(e: float) -> float:
    return max(0.0, 0.5 - e)


def make_dataset(
    n_samples: int = 2000,
    seed: int = 42,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> Tuple[dict, dict, dict]:
    """
    Returns (train, val, test) each a dict with keys:
        X         : np.ndarray (N, FEATURE_DIM)
        emotion   : np.ndarray (N,) int
        stress    : np.ndarray (N,)
        engagement: np.ndarray (N,)
        attention : np.ndarray (N,)
        fatigue   : np.ndarray (N,)
    """
    rng = np.random.default_rng(seed)
    samples = [_make_sample(rng) for _ in range(n_samples)]
    X = np.stack([s[0] for s in samples]).astype(np.float32)
    Y = {k: np.array([s[1][k] for s in samples]) for k in samples[0][1]}
    Y["emotion"] = Y["emotion"].astype(np.int64)

    n_val  = int(n_samples * val_frac)
    n_test = int(n_samples * test_frac)
    n_train = n_samples - n_val - n_test

    def split(arr):
        return arr[:n_train], arr[n_train:n_train+n_val], arr[n_train+n_val:]

    def pack(X_split, Y_splits):
        d = {"X": X_split}
        for k, (tr, va, te) in Y_splits.items():
            pass
        return d

    X_tr, X_va, X_te = split(X)
    Y_splits = {k: split(v) for k, v in Y.items()}

    def make_part(X_part, idx):
        d = {"X": X_part}
        for k, parts in Y_splits.items():
            d[k] = parts[idx]
        return d

    return make_part(X_tr, 0), make_part(X_va, 1), make_part(X_te, 2)
