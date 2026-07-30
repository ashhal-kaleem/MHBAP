"""
real_dataset.py — Real public dataset loader for MHBAP TCMT training.

Verified datasets (all downloadable via HuggingFace datasets library):
  1. clip-benchmark/wds_fer2013  → emotion (7-class → 4-class)
     keys: cls (int 0-6), jpg (PIL Image)
  2. deanngkl/raf-db-7emotions   → emotion (7-class → 4-class, supplement)
     keys: image (PIL), bbox, label (int 1-7)
  3. LouisSimon/wesad-parquet    → stress (0/1/2/3 int → 0.0-1.0 float)
     keys: bvp, eda, temp, acc_x/y/z, stress (int), user (int)

Since MHBAP operates on 58-dim feature vectors (face AUs, gaze, pose, voice, HCI)
extracted at runtime, we derive proxy feature vectors from dataset signals:
  - FER2013/RAF-DB: pixel statistics → 12-dim face AU proxies
  - WESAD: bvp/eda/temp → voice/HCI energy proxies; face/gaze conditioned on stress
  - engagement/attention/fatigue: no public labelled dataset available;
    derived via deterministic rules (documented in CHANGELOG).

All splits are reproducible via seed. Test set held-out, never seen during training.
"""
from __future__ import annotations
import os, warnings
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import numpy as np

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)

from ml.fusion.feature_vector import FEATURE_DIM, MODALITY_KEYS
from ml.fusion.feature_utils import modality_slice

# FER2013 7-class → MHBAP 4-class: 0=Angry,1=Disgust,2=Fear,3=Happy,4=Sad,5=Surprise,6=Neutral
FER_TO_MHBAP = {0: 3, 1: 3, 2: 2, 3: 1, 4: 2, 5: 1, 6: 0}
# RAF-DB 1-indexed: 1=Surprise,2=Fear,3=Disgust,4=Happy,5=Sad,6=Angry,7=Neutral
RAF_TO_MHBAP = {1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3, 7: 0}
# WESAD: 0=undef,1=baseline,2=stress,3=amusement → 0-1 float
WESAD_STRESS_MAP = {0: 0.3, 1: 0.1, 2: 0.9, 3: 0.15}


def _img_to_face_features(img_array: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pixel-region statistics → 12-dim AU proxy features."""
    gray = img_array.mean(axis=2) if img_array.ndim == 3 else img_array.astype(float)
    H, W = gray.shape
    g = gray / 255.0

    def _r(r0, r1, c0, c1):
        a = g[int(H*r0):int(H*r1), int(W*c0):int(W*c1)]
        return float(a.mean()) if a.size > 0 else 0.5

    brow = _r(0.10, 0.30, 0.20, 0.80)
    el   = _r(0.25, 0.45, 0.10, 0.45)
    er   = _r(0.25, 0.45, 0.55, 0.90)
    nose = _r(0.40, 0.60, 0.30, 0.70)
    mth  = _r(0.60, 0.85, 0.25, 0.75)
    jaw  = _r(0.75, 0.95, 0.20, 0.80)
    n    = rng.uniform

    return np.array([
        float(np.clip(brow * 1.5, 0, 1)),
        float(np.clip(brow * 1.5 + n(-0.05, 0.05), 0, 1)),
        float(np.clip(1.0 - brow * 1.2, 0, 1)),
        float(np.clip(1.0 - el, 0, 1)),
        float(np.clip(1.0 - er, 0, 1)),
        float(np.clip(nose * 0.8, 0, 1)),
        float(np.clip(mth * 1.3, 0, 1)),
        float(np.clip(mth * 1.3 + n(-0.05, 0.05), 0, 1)),
        float(np.clip(1.0 - mth * 1.5, 0, 1)),
        float(np.clip(jaw * 1.5, 0, 1)),
        float(np.clip(mth * 0.8, 0, 1)),
        float(np.clip(jaw * 0.9, 0, 1)),
    ], dtype=np.float32)


def _emotion_biases(emotion: int) -> dict:
    """Feature biases per MHBAP emotion class (0=neutral,1=happy,2=sad,3=angry)."""
    if emotion == 1:
        return dict(gaze=[0.2,0.2,0.7,0.7,0.8], v_energy=0.6, v_rate=0.7, hci_err=0.05)
    if emotion == 2:
        return dict(gaze=[0.6,0.6,0.5,0.5,0.4], v_energy=0.3, v_rate=0.3, hci_err=0.15)
    if emotion == 3:
        return dict(gaze=[0.3,0.3,0.6,0.6,0.5], v_energy=0.8, v_rate=0.8, hci_err=0.30)
    return dict(gaze=[0.3,0.3,0.6,0.6,0.65], v_energy=0.5, v_rate=0.5, hci_err=0.10)


def _build_feature(face_aus: np.ndarray, stress_val: Optional[float],
                   emotion: int, rng: np.random.Generator) -> np.ndarray:
    """Assemble 58-dim feature: real face AUs + conditioned proxies."""
    x = rng.uniform(0.15, 0.85, FEATURE_DIM).astype(np.float32)
    fs, fe = modality_slice("face")
    gs, ge = modality_slice("gaze")
    vs, ve = modality_slice("voice")
    hs, he = modality_slice("hci")

    x[fs:fe] = face_aus
    b = _emotion_biases(emotion)
    x[gs:ge] = np.clip(
        np.array(b["gaze"], dtype=np.float32) + rng.uniform(-0.1, 0.1, ge-gs).astype(np.float32), 0, 1)
    x[vs+15] = float(np.clip(b["v_energy"] + rng.uniform(-0.1, 0.1), 0, 1))
    x[vs+18] = float(np.clip(b["v_rate"]   + rng.uniform(-0.1, 0.1), 0, 1))
    x[hs+6]  = float(np.clip(b["hci_err"]  + rng.uniform(-0.05, 0.05), 0, 1))

    if stress_val is not None:
        x[vs+15] = float(np.clip(stress_val*0.8 + rng.uniform(-0.05, 0.05), 0, 1))
        x[hs+6]  = float(np.clip(stress_val*0.6 + rng.uniform(-0.05, 0.05), 0, 1))
        x[fs+2]  = float(np.clip(stress_val*0.7 + rng.uniform(-0.05, 0.05), 0, 1))
    return x


def _derive_labels(emotion: int, stress_val: Optional[float],
                   x: np.ndarray, rng: np.random.Generator) -> dict:
    """Derive all 5 supervision labels."""
    fs, _ = modality_slice("face")
    gs, _ = modality_slice("gaze")
    vs, _ = modality_slice("voice")
    hs, _ = modality_slice("hci")

    energy    = float(x[vs+15]);  speak_rt = float(x[vs+18])
    fix_stab  = float(x[gs+4]);   blink_a  = float(x[gs+2]+x[gs+3])/2
    err_rate  = float(x[hs+6]);   pause_r  = float(x[hs+8])
    dwell_t   = float(x[hs+5]);   ksr      = float(x[hs+4])
    brow_fw   = float(x[fs+2])

    stress = float(np.clip(stress_val + rng.uniform(-0.05,0.05), 0, 1)) \
        if stress_val is not None else \
        float(np.clip(0.4*energy + 0.3*err_rate + 0.2*brow_fw + 0.1*rng.uniform(0,1), 0, 1))
    engagement = float(np.clip(0.35*fix_stab + 0.30*speak_rt + 0.25*ksr + 0.1*rng.uniform(0,1), 0, 1))
    attention  = float(np.clip(0.5*(1-blink_a) + 0.35*fix_stab + 0.15*rng.uniform(0,1), 0, 1))
    fatigue    = float(np.clip(0.4*pause_r + 0.3*dwell_t + 0.2*(1-energy) + 0.1*rng.uniform(0,1), 0, 1))
    return dict(emotion=emotion, stress=stress, engagement=engagement,
                attention=attention, fatigue=fatigue)


def load_fer2013(max_samples: int = 6000, seed: int = 42) -> List[dict]:
    """Load FER2013 from HuggingFace. Returns list of label dicts with 'X' key."""
    print("[real_dataset] FER2013 (clip-benchmark/wds_fer2013)...", flush=True)
    from datasets import load_dataset
    rng = np.random.default_rng(seed)
    records: List[dict] = []
    per_split = max_samples // 2
    for split in ("train", "test"):
        try:
            ds = load_dataset("clip-benchmark/wds_fer2013", split=split,
                              streaming=True, trust_remote_code=False)
            count = 0
            for item in ds:
                if count >= per_split:
                    break
                fer_lbl = int(item.get("cls", 6))
                emo = FER_TO_MHBAP.get(fer_lbl, 0)
                img = item.get("jpg")
                if img is None:
                    continue
                img_arr = np.array(img.convert("L").resize((48, 48))) \
                    if hasattr(img, "convert") else np.array(img)
                aus = _img_to_face_features(img_arr, rng)
                x   = _build_feature(aus, None, emo, rng)
                lbl = _derive_labels(emo, None, x, rng)
                lbl["X"] = x
                records.append(lbl)
                count += 1
            print(f"  FER2013 [{split}] {count} samples", flush=True)
        except Exception as e:
            print(f"  FER2013 [{split}] WARN: {e}", flush=True)
    return records


def load_rafdb(max_samples: int = 3000, seed: int = 42) -> List[dict]:
    """Load RAF-DB from HuggingFace. Returns list of label dicts with 'X' key."""
    print("[real_dataset] RAF-DB (deanngkl/raf-db-7emotions)...", flush=True)
    from datasets import load_dataset
    rng = np.random.default_rng(seed + 1)
    records: List[dict] = []
    per_split = max_samples // 2
    for split in ("train", "test"):
        try:
            ds = load_dataset("deanngkl/raf-db-7emotions", split=split,
                              streaming=True, trust_remote_code=False)
            count = 0
            for item in ds:
                if count >= per_split:
                    break
                raf_lbl = int(item.get("label", 7))
                emo = RAF_TO_MHBAP.get(raf_lbl, 0)
                img = item.get("image")
                if img is None:
                    continue
                img_arr = np.array(img.convert("L").resize((48, 48))) \
                    if hasattr(img, "convert") else np.array(img)
                aus = _img_to_face_features(img_arr, rng)
                x   = _build_feature(aus, None, emo, rng)
                lbl = _derive_labels(emo, None, x, rng)
                lbl["X"] = x
                records.append(lbl)
                count += 1
            print(f"  RAF-DB [{split}] {count} samples", flush=True)
        except Exception as e:
            print(f"  RAF-DB [{split}] WARN: {e}", flush=True)
    return records


def load_wesad(max_samples: int = 2000, seed: int = 42) -> List[dict]:
    """
    Load WESAD physiological stress data from HuggingFace.
    keys: bvp, eda, temp, acc_x/y/z, stress (int 0-3), user (int)
    Derives stress-conditioned 58-dim feature vectors.
    """
    print("[real_dataset] WESAD (LouisSimon/wesad-parquet)...", flush=True)
    from datasets import load_dataset
    rng = np.random.default_rng(seed + 2)
    records: List[dict] = []
    try:
        ds = load_dataset("LouisSimon/wesad-parquet", split="train",
                          streaming=True, trust_remote_code=False)
        count = 0
        for item in ds:
            if count >= max_samples:
                break
            stress_raw = item.get("stress", 0)
            stress_lbl = int(stress_raw[0]) if isinstance(stress_raw, (list, np.ndarray)) else int(stress_raw)
            stress_val = WESAD_STRESS_MAP.get(stress_lbl, 0.3)
            # Physio signals as voice/HCI proxies (fields may be lists/arrays)
            def _scalar(v, default=0.0):
                if isinstance(v, (list, np.ndarray)):
                    return float(v[0]) if len(v) > 0 else default
                return float(v) if v is not None else default
            bvp  = _scalar(item.get("bvp",  0))
            eda  = _scalar(item.get("eda",  0))
            temp = _scalar(item.get("temp", 36))
            # Normalise crudely (clamp to plausible ranges)
            bvp_n  = float(np.clip((bvp + 500) / 1000, 0, 1))
            eda_n  = float(np.clip(eda / 20, 0, 1))
            temp_n = float(np.clip((temp - 30) / 12, 0, 1))
            # Neutral emotion assumed for stress-only data
            emo = 0
            aus = np.full(12, 0.5, dtype=np.float32)
            aus[2] = float(np.clip(stress_val * 0.7, 0, 1))  # brow furrow
            x = _build_feature(aus, stress_val, emo, rng)
            # Override with real physio signals
            vs, _ = modality_slice("voice")
            hs, _ = modality_slice("hci")
            x[vs+15] = bvp_n
            x[vs+16] = eda_n
            x[vs+17] = temp_n
            x[hs+6]  = float(np.clip(stress_val * 0.6 + rng.uniform(-0.05, 0.05), 0, 1))
            lbl = _derive_labels(emo, stress_val, x, rng)
            lbl["X"] = x
            records.append(lbl)
            count += 1
        print(f"  WESAD {count} samples", flush=True)
    except Exception as e:
        print(f"  WESAD WARN: {e}", flush=True)
    return records


def make_real_dataset(
    fer_samples: int = 6000,
    raf_samples: int = 3000,
    wesad_samples: int = 2000,
    seed: int = 42,
    val_frac: float = 0.10,
    test_frac: float = 0.15,
) -> Tuple[dict, dict, dict]:
    """
    Download real datasets, combine, split into train/val/test.
    Returns (train_split, val_split, test_split) each a dict with keys:
      X, emotion, stress, engagement, attention, fatigue (all np.ndarray)
    """
    print("[real_dataset] Loading real datasets...", flush=True)
    records: List[dict] = []
    records.extend(load_fer2013(max_samples=fer_samples, seed=seed))
    records.extend(load_rafdb(max_samples=raf_samples, seed=seed))
    records.extend(load_wesad(max_samples=wesad_samples, seed=seed))
    print(f"[real_dataset] Total records: {len(records)}", flush=True)

    if len(records) == 0:
        raise RuntimeError("No records loaded from any dataset.")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    records = [records[i] for i in idx]

    N = len(records)
    n_test = int(N * test_frac)
    n_val  = int(N * val_frac)
    n_train = N - n_test - n_val

    def _pack(recs: list) -> dict:
        return {
            "X":          np.stack([r["X"] for r in recs]).astype(np.float32),
            "emotion":    np.array([r["emotion"] for r in recs], dtype=np.int64),
            "stress":     np.array([r["stress"] for r in recs], dtype=np.float32),
            "engagement": np.array([r["engagement"] for r in recs], dtype=np.float32),
            "attention":  np.array([r["attention"] for r in recs], dtype=np.float32),
            "fatigue":    np.array([r["fatigue"] for r in recs], dtype=np.float32),
        }

    train = _pack(records[:n_train])
    val   = _pack(records[n_train:n_train+n_val])
    test  = _pack(records[n_train+n_val:])

    print(f"[real_dataset] Split: train={len(train['X'])} "
          f"val={len(val['X'])} test={len(test['X'])}", flush=True)
    return train, val, test
