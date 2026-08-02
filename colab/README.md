# MHBAP — Colab Training

This folder contains everything needed to train TCMT on Google Colab GPU.

## Files

| File | Purpose |
|---|---|
| `TCMT_Train_Colab.ipynb` | Main training notebook |
| `sync_weights.py` | Post-training: copies downloaded weights into the repo |

## Workflow

### Step 1 — Upload notebook to Colab

Go to [colab.research.google.com](https://colab.research.google.com) →
File → Upload notebook → select `TCMT_Train_Colab.ipynb`.

### Step 2 — Set runtime to GPU

Runtime → Change runtime type → **T4 GPU** → Save.

### Step 3 — Run all cells

In order, top to bottom. Cell 4 clones the repo from GitHub.
Cell 6 resumes from a Drive checkpoint if one exists.

Estimated time on T4: **~25–40 min** for 75 epochs × 11,000 training samples.

### Step 4 — Download artifacts

Cell 12 triggers browser download of:
- `tcmt_trained.pt` (~1.6 MB)
- `tcmt_eval_metrics.json`

### Step 5 — Sync to local repo

```powershell
cd D:\MHBAP
python colab/sync_weights.py
```

Then commit and push:

```powershell
git add ml/models/weights/tcmt_trained.pt ml/models/weights/tcmt_eval_metrics.json
git commit -m "feat(ml): update TCMT weights from Colab GPU training"
git push origin feature/production-hardening
```

## Architecture (frozen — do not change)

| Param | Value |
|---|---|
| FEATURE_DIM | 58 |
| EMOTION_CLASSES | 4 (neutral / happy / sad / angry) |
| D_MODEL | 128 |
| N_HEADS | 4 |
| N_LAYERS | 3 |
| FFN_DIM | 256 |
| DROPOUT | 0.15 |

## Datasets

| Dataset | Source | Labels |
|---|---|---|
| FER2013 | `clip-benchmark/wds_fer2013` | emotion (REAL) |
| RAF-DB | `deanngkl/raf-db-7emotions` | emotion (REAL) |
| WESAD | `LouisSimon/wesad-parquet` | stress (REAL GT) |

Engagement / attention / fatigue are PROXY labels (no public GT).
This is documented in the saved metrics JSON.
