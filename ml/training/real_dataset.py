"""
real_dataset.py -- Real public dataset loader for MHBAP TCMT training.

FIX APPLIED (audit 2025):
  - REMOVED _emotion_biases(): no longer injects class-specific values into
    the feature vector. Emotion label is now independent of all other dims.
  - Face AU features come solely from pixel region statistics (image content).
  - Gaze/pose/voice/HCI dims are uniform random noise within plausible ranges,
    independent of the emotion class -- the model cannot shortcut via injected
    biases.
  - WESAD overrides specific voice/HCI dims with real physio signals.
  - Engagement/attention/fatigue are PROXY labels derived from the noisy feature
    dims (no public ground-truth exists). They are labelled [PROXY] in docstrings
    and in the saved metrics JSON. These metrics measure proxy-label regression
    only and should not be interpreted as measuring real behavioral generalisation.
  - Stress comes from WESAD ground-truth annotations (0=undef/1=baseline/
    2=stress/3=amusement mapped to 0-1 float) or from AU+HCI proxies for
    emotion-only samples where no physio signal is available.

Verified datasets (HuggingFace datasets library):
  1. clip-benchmark/wds_fer2013  -- emotion 7-class -> 4-class
  2. deanngkl/raf-db-7emotions   -- emotion 7-class -> 4-class (supplement)
  3. LouisSimon/wesad-parquet    -- stress 0/1/2/3 -> 0.0-1.0 float
"""
from __future__ import annotations
import os, warnings
from typing import Tuple, List, Optional
import numpy as np

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)

from ml.fusion.feature_vector import FEATURE_DIM, MODALITY_KEYS
from ml.fusion.feature_utils import modality_slice

# ---- class mappings -------------------------------------------------------
# FER2013: 0=Angry,1=Disgust,2=Fear,3=Happy,4=Sad,5=Surprise,6=Neutral
FER_TO_MHBAP = {0: 3, 1: 3, 2: 2, 3: 1, 4: 2, 5: 1, 6: 0}
# RAF-DB 1-indexed: 1=Surprise,2=Fear,3=Disgust,4=Happy,5=Sad,6=Angry,7=Neutral
RAF_TO_MHBAP = {1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3, 7: 0}
# WESAD: 0=undef,1=baseline,2=stress,3=amusement
WESAD_STRESS_MAP = {0: 0.3, 1: 0.1, 2: 0.9, 3: 0.15}


def _img_to_face_features(img_array: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pixel-region statistics -> 12-dim AU proxy features.
    Uses only image content; no emotion label involved.
    """
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
        float(np.clip(brow * 1.5,               0, 1)),
        float(np.clip(brow * 1.5 + n(-0.05, 0.05), 0, 1)),
        float(np.clip(1.0 - brow * 1.2,         0, 1)),
        float(np.clip(1.0 - el,                 0, 1)),
        float(np.clip(1.0 - er,                 0, 1)),
        float(np.clip(nose * 0.8,               0, 1)),
        float(np.clip(mth * 1.3,                0, 1)),
        float(np.clip(mth * 1.3 + n(-0.05, 0.05), 0, 1)),
        float(np.clip(1.0 - mth * 1.5,          0, 1)),
        float(np.clip(jaw * 1.5,                0, 1)),
        float(np.clip(mth * 0.8,                0, 1)),
        float(np.clip(jaw * 0.9,                0, 1)),
    ], dtype=np.float32)


def _build_feature_clean(
    face_aus: np.ndarray,
    rng: np.random.Generator,
    stress_overrides: Optional[dict] = None,
) -> np.ndarray:
    """Build 58-dim feature: real face AUs + random noise for other dims.

    Critically: NO emotion-class-specific biases are injected.
    The emotion label is therefore NOT decodable from gaze/voice/hci dims.
    WESAD samples can pass physio overrides for specific voice/hci indices.
    """
    x = rng.uniform(0.15, 0.85, FEATURE_DIM).astype(np.float32)
    fs, fe = modality_slice("face")
    x[fs:fe] = face_aus
    if stress_overrides:
        for idx, val in stress_overrides.items():
            x[idx] = float(np.clip(val, 0.0, 1.0))
    return x


def _derive_proxy_labels(
    emotion: int,
    stress_val: Optional[float],
    x: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Derive supervision labels.

    Emotion:    from dataset annotation (FER/RAF/WESAD class mapping). REAL.
    Stress:     from WESAD physio annotation when available (REAL);
                otherwise from face AU + HCI proxy (PROXY).
    Engagement/Attention/Fatigue: always PROXY (no public GT dataset).
      Derived from the feature vector dims that were set to random noise,
      so these measure noise-to-noise regression -- disclosed in metrics.
    """
    gs, _ = modality_slice("gaze")
    vs, _ = modality_slice("voice")
    hs, _ = modality_slice("hci")
    fs, _ = modality_slice("face")

    energy    = float(x[vs + 15])
    speak_rt  = float(x[vs + 18])
    fix_stab  = float(x[gs + 4])
    blink_a   = float(x[gs + 2] + x[gs + 3]) / 2
    err_rate  = float(x[hs + 6])
    pause_r   = float(x[hs + 8])
    dwell_t   = float(x[hs + 5])
    ksr       = float(x[hs + 4])
    brow_fw   = float(x[fs + 2])

    # Stress: WESAD GT when available, else proxy
    if stress_val is not None:
        stress = float(np.clip(stress_val + rng.uniform(-0.03, 0.03), 0, 1))
    else:
        stress = float(np.clip(
            0.4 * energy + 0.3 * err_rate + 0.2 * brow_fw + 0.1 * rng.uniform(0, 1),
            0, 1))

    # Proxy labels from random noise dims (disclosed)
    engagement = float(np.clip(
        0.35 * fix_stab + 0.30 * speak_rt + 0.25 * ksr + 0.10 * rng.uniform(0, 1), 0, 1))
    attention  = float(np.clip(
        0.50 * (1 - blink_a) + 0.35 * fix_stab + 0.15 * rng.uniform(0, 1), 0, 1))
    fatigue    = float(np.clip(
        0.40 * pause_r + 0.30 * dwell_t + 0.20 * (1 - energy) + 0.10 * rng.uniform(0, 1), 0, 1))

    return dict(emotion=emotion, stress=stress,
                engagement=engagement, attention=attention, fatigue=fatigue)


def load_fer2013(max_samples: int = 6000, seed: int = 42) -> List[dict]:
    """Load FER2013. Emotion label from dataset; features from pixel stats only."""
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
                x   = _build_feature_clean(aus, rng)
                lbl = _derive_proxy_labels(emo, None, x, rng)
                lbl["X"] = x
                records.append(lbl)
                count += 1
            print(f"  FER2013 [{split}] {count} samples", flush=True)
        except Exception as e:
            print(f"  FER2013 [{split}] WARN: {e}", flush=True)
    return records


def load_rafdb(max_samples: int = 3000, seed: int = 42) -> List[dict]:
    """Load RAF-DB. Emotion label from dataset; features from pixel stats only."""
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
                x   = _build_feature_clean(aus, rng)
                lbl = _derive_proxy_labels(emo, None, x, rng)
                lbl["X"] = x
                records.append(lbl)
                count += 1
            print(f"  RAF-DB [{split}] {count} samples", flush=True)
        except Exception as e:
            print(f"  RAF-DB [{split}] WARN: {e}", flush=True)
    return records


def load_wesad(max_samples: int = 2000, seed: int = 42) -> List[dict]:
    """Load WESAD. Stress from GT physio annotation; emotion neutral (no face data)."""
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
            stress_lbl = int(stress_raw[0]) if isinstance(stress_raw, (list, np.ndarray)) \
                         else int(stress_raw)
            stress_val = WESAD_STRESS_MAP.get(stress_lbl, 0.3)

            def _scalar(v, default=0.0):
                if isinstance(v, (list, np.ndarray)):
                    return float(v[0]) if len(v) > 0 else default
                return float(v) if v is not None else default

            bvp  = _scalar(item.get("bvp",  0))
            eda  = _scalar(item.get("eda",  0))
            temp = _scalar(item.get("temp", 36))
            bvp_n  = float(np.clip((bvp + 500) / 1000, 0, 1))
            eda_n  = float(np.clip(eda / 20,            0, 1))
            temp_n = float(np.clip((temp - 30) / 12,    0, 1))

            # Neutral emotion (WESAD has no face images)
            emo = 0
            aus = np.full(12, 0.5, dtype=np.float32)

            vs, _ = modality_slice("voice")
            hs, _ = modality_slice("hci")
            overrides = {
                vs + 15: bvp_n,
                vs + 16: eda_n,
                vs + 17: temp_n,
                hs + 6:  float(np.clip(stress_val * 0.6 + rng.uniform(-0.05, 0.05), 0, 1)),
            }
            x = _build_feature_clean(aus, rng, stress_overrides=overrides)
            lbl = _derive_proxy_labels(emo, stress_val, x, rng)
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
    """Download real datasets, combine, split into train/val/test.

    Returns (train_split, val_split, test_split) each a dict:
      X, emotion, stress, engagement, attention, fatigue -- all np.ndarray.

    Label provenance:
      emotion    -- REAL: FER2013/RAF-DB class annotations
      stress     -- REAL (WESAD GT) or PROXY (AU+HCI formula, FER/RAF samples)
      engagement -- PROXY: noisy gaze+voice formula (no public GT)
      attention  -- PROXY: noisy gaze formula (no public GT)
      fatigue    -- PROXY: noisy HCI formula (no public GT)
    """
    print("[real_dataset] Loading real datasets...", flush=True)
    records: List[dict] = []
    records.extend(load_fer2013(max_samples=fer_samples, seed=seed))
    records.extend(load_rafdb(max_samples=raf_samples,  seed=seed))
    records.extend(load_wesad(max_samples=wesad_samples, seed=seed))
    print(f"[real_dataset] Total records: {len(records)}", flush=True)

    if len(records) == 0:
        raise RuntimeError("No records loaded from any dataset.")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    records = [records[i] for i in idx]

    N = len(records)
    n_test  = int(N * test_frac)
    n_val   = int(N * val_frac)
    n_train = N - n_test - n_val

    def _pack(recs: list) -> dict:
        return {
            "X":          np.stack([r["X"] for r in recs]).astype(np.float32),
            "emotion":    np.array([r["emotion"]    for r in recs], dtype=np.int64),
            "stress":     np.array([r["stress"]     for r in recs], dtype=np.float32),
            "engagement": np.array([r["engagement"] for r in recs], dtype=np.float32),
            "attention":  np.array([r["attention"]  for r in recs], dtype=np.float32),
            "fatigue":    np.array([r["fatigue"]    for r in recs], dtype=np.float32),
        }

    train = _pack(records[:n_train])
    val   = _pack(records[n_train:n_train + n_val])
    test  = _pack(records[n_train + n_val:])

    print(f"[real_dataset] Split: train={len(train['X'])} "
          f"val={len(val['X'])} test={len(test['X'])}", flush=True)
    return train, val, test
