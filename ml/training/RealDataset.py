"""
real_dataset.py -- Real public dataset loader for MHBAP TCMT training.

FIX APPLIED (post-retrain, class-collapse investigation):
  - load_fer2013()/load_rafdb() previously took the first N items from the
    HF streaming iterator with NO shuffle. Both webdataset sources are shard-
    ordered by class, so this produced a heavily skewed sample (e.g. 54% of
    FER2013+RAF-DB samples landing in class 3 "angry/disgust"), which is why
    the emotion head collapsed to predicting only classes 0 and 3 (F1=0.0 on
    classes 1 and 2). Fixed by adding ds.shuffle(seed=..., buffer_size=4000)
    before truncating to per_split -- see verification in AUDIT_REPORT.md.

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
import math

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)

from ml.fusion.FeatureVector import FEATURE_DIM, MODALITY_KEYS
from ml.fusion.FeatureUtils import modality_slice

# ---- class mappings -------------------------------------------------------
# MHBAP 4-class schema: 0=neutral, 1=happy/positive, 2=sad/fearful, 3=angry/negative
#
# FER2013 7-class: 0=Angry,1=Disgust,2=Fear,3=Happy,4=Sad,5=Surprise,6=Neutral
# Mapping rationale:
#   Angry(0)   → 3 (angry/negative)
#   Disgust(1) → 2 (was 3 -- caused double-counting of class 3; disgust is
#                   negative but not aggressive, closer to sad/fearful)
#   Fear(2)    → 2 (sad/fearful)
#   Happy(3)   → 1 (happy/positive)
#   Sad(4)     → 2 (sad/fearful)
#   Surprise(5)→ 1 (positive/aroused, treated as happy-adjacent)
#   Neutral(6) → 0 (neutral)
FER_TO_MHBAP = {0: 3, 1: 2, 2: 2, 3: 1, 4: 2, 5: 1, 6: 0}
# RAF-DB 1-indexed: 1=Surprise,2=Fear,3=Disgust,4=Happy,5=Sad,6=Angry,7=Neutral
# Mapping rationale: same schema; Disgust(3)→2 for consistency with FER fix
RAF_TO_MHBAP = {1: 1, 2: 2, 3: 2, 4: 1, 5: 2, 6: 3, 7: 0}
# WESAD: 0=undef,1=baseline,2=stress,3=amusement
WESAD_STRESS_MAP = {0: 0.3, 1: 0.1, 2: 0.9, 3: 0.15}


_GLOBAL_FACE_MESH = None

def _get_face_mesh():
    global _GLOBAL_FACE_MESH
    if _GLOBAL_FACE_MESH is None:
        try:
            import mediapipe as mp # type: ignore
            _GLOBAL_FACE_MESH = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
        except ImportError:
            _GLOBAL_FACE_MESH = "MISSING"
    return _GLOBAL_FACE_MESH


def _dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _img_to_face_features(img_array: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """MediaPipe FaceMesh -> 12-dim AU proxy features.
    Matches FacePipeline inference logic. Falls back to zeros if no face.
    Uses only image content; no emotion label involved.
    """
    mesh = _get_face_mesh()
    zeros = np.zeros(12, dtype=np.float32)
    if mesh == "MISSING" or mesh is None:
        return zeros

    results = mesh.process(img_array)
    if not results.multi_face_landmarks:
        return zeros

    lm = results.multi_face_landmarks[0].landmark
    left_brow   = lm[105]; left_eye_top   = lm[159]
    right_brow  = lm[334]; right_eye_top  = lm[386]
    brow_mid_l  = lm[107]; brow_mid_r     = lm[336]
    nose_tip    = lm[4]
    upper_lip   = lm[13]; lower_lip = lm[14]
    lip_l       = lm[61]; lip_r     = lm[291]
    jaw         = lm[152]; chin      = lm[175]
    l_eye_top   = lm[159]; l_eye_bot = lm[145]
    r_eye_top   = lm[386]; r_eye_bot = lm[374]

    face_h = _dist(lm[10], lm[152]) or 1e-6

    return np.array([
        min(1.0, _dist(left_brow,  left_eye_top)  / face_h * 5),
        min(1.0, _dist(right_brow, right_eye_top) / face_h * 5),
        1.0 - min(1.0, _dist(brow_mid_l, brow_mid_r) / face_h * 4),
        min(1.0, _dist(l_eye_top, l_eye_bot) / face_h * 10),
        min(1.0, _dist(r_eye_top, r_eye_bot) / face_h * 10),
        min(1.0, abs(nose_tip.z) * 3),
        min(1.0, max(0.0, lip_l.x - lm[0].x) / face_h * 8),
        min(1.0, max(0.0, lm[0].x - lip_r.x) / face_h * 8),
        1.0 - min(1.0, _dist(upper_lip, lower_lip) / face_h * 12),
        min(1.0, _dist(jaw, chin) / face_h * 6),
        min(1.0, abs(lm[117].z + lm[346].z) * 2),
        min(1.0, abs(lm[199].z) * 4),
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


def load_fer2013(max_samples: int = 6000, seed: int = 42, split: str = "train") -> List[dict]:
    """Load FER2013. Emotion label from dataset; features from pixel stats only."""
    print(f"[real_dataset] FER2013 (clip-benchmark/wds_fer2013) split={split}...", flush=True)
    from datasets import load_dataset
    rng = np.random.default_rng(seed)
    records: List[dict] = []
    try:
        ds = load_dataset("clip-benchmark/wds_fer2013", split=split,
                          streaming=True, trust_remote_code=False)
        ds = ds.shuffle(seed=seed, buffer_size=4000)
        count = 0
        for item in ds:
            if count >= max_samples:
                break
            fer_lbl = int(item.get("cls", 6))
            emo = FER_TO_MHBAP.get(fer_lbl, 0)
            img = item.get("jpg")
            if img is None:
                continue
            img_arr = np.array(img.convert("RGB")) \
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


def load_rafdb(max_samples: int = 3000, seed: int = 42, split: str = "train") -> List[dict]:
    """Load RAF-DB. Emotion label from dataset; features from pixel stats only."""
    print(f"[real_dataset] RAF-DB (deanngkl/raf-db-7emotions) split={split}...", flush=True)
    from datasets import load_dataset
    rng = np.random.default_rng(seed + 1)
    records: List[dict] = []
    try:
        ds = load_dataset("deanngkl/raf-db-7emotions", split=split,
                          streaming=True, trust_remote_code=False)
        ds = ds.shuffle(seed=seed + 1, buffer_size=4000)
        count = 0
        for item in ds:
            if count >= max_samples:
                break
            raf_lbl = int(item.get("label", 7))
            emo = RAF_TO_MHBAP.get(raf_lbl, 0)
            img = item.get("image")
            if img is None:
                continue
            img_arr = np.array(img.convert("RGB")) \
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
    train_records: List[dict] = []
    test_records: List[dict] = []
    
    # Calculate samples for splits based on total desired samples
    fer_test = int(fer_samples * (val_frac + test_frac))
    fer_train = fer_samples - fer_test
    
    raf_test = int(raf_samples * (val_frac + test_frac))
    raf_train = raf_samples - raf_test
    
    wesad_test = int(wesad_samples * (val_frac + test_frac))
    wesad_train = wesad_samples - wesad_test

    train_records.extend(load_fer2013(max_samples=fer_train, seed=seed, split="train"))
    test_records.extend(load_fer2013(max_samples=fer_test, seed=seed, split="test"))
    
    raf_all = load_rafdb(max_samples=raf_samples, seed=seed, split="train")
    rng_r = np.random.default_rng(seed + 1)
    idx_r = np.arange(len(raf_all))
    rng_r.shuffle(idx_r)
    raf_all = [raf_all[i] for i in idx_r]
    train_records.extend(raf_all[:raf_train])
    test_records.extend(raf_all[raf_train:])
    
    wesad_all = load_wesad(max_samples=wesad_samples, seed=seed)
    
    # WESAD doesn't have a test split in LouisSimon/wesad-parquet usually, 
    # so we'll just slice it. Since it's not face images, it's less of an issue, 
    # but still better to keep them separate.
    rng_w = np.random.default_rng(seed)
    idx_w = np.arange(len(wesad_all))
    rng_w.shuffle(idx_w)
    wesad_all = [wesad_all[i] for i in idx_w]
    train_records.extend(wesad_all[:wesad_train])
    test_records.extend(wesad_all[wesad_train:])

    print(f"[real_dataset] Total train records: {len(train_records)}, test records: {len(test_records)}", flush=True)

    if len(train_records) == 0:
        raise RuntimeError("No train records loaded from any dataset.")

    rng = np.random.default_rng(seed)
    idx_train = np.arange(len(train_records))
    rng.shuffle(idx_train)
    train_records = [train_records[i] for i in idx_train]
    
    idx_test = np.arange(len(test_records))
    rng.shuffle(idx_test)
    test_records = [test_records[i] for i in idx_test]

    N_test_pool = len(test_records)
    # the test pool contains val_frac + test_frac portion of samples.
    # we need to split it proportionally into val and test.
    val_ratio_in_test_pool = val_frac / (val_frac + test_frac)
    n_val = int(N_test_pool * val_ratio_in_test_pool)

    def _pack(recs: list) -> dict:
        return {
            "X":          np.stack([r["X"] for r in recs]).astype(np.float32),
            "emotion":    np.array([r["emotion"]    for r in recs], dtype=np.int64),
            "stress":     np.array([r["stress"]     for r in recs], dtype=np.float32),
            "engagement": np.array([r["engagement"] for r in recs], dtype=np.float32),
            "attention":  np.array([r["attention"]  for r in recs], dtype=np.float32),
            "fatigue":    np.array([r["fatigue"]    for r in recs], dtype=np.float32),
        }

    train = _pack(train_records)
    val   = _pack(test_records[:n_val])
    test  = _pack(test_records[n_val:])

    print(f"[real_dataset] Split: train={len(train['X'])} "
          f"val={len(val['X'])} test={len(test['X'])}", flush=True)
    return train, val, test
