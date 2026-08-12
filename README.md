# MHBAP — Multimodal Human Behaviour Analysis Platform

A full-stack platform that captures webcam, microphone, and keyboard/mouse input,
extracts features from each, fuses them using a Temporal Cross-Modal Transformer (TCMT), and streams behaviour predictions live to a React dashboard.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture & Workflow](#architecture--workflow)
4. [ML Models & Components](#ml-models--components)
5. [Tech Stack](#tech-stack)
6. [Project Structure](#project-structure)
7. [Setup & Installation](#setup--installation)
8. [Testing](#testing)
9. [Evaluation](#evaluation)
10. [Privacy & Data Handling](#privacy--data-handling)
11. [References](#references)

---

## Overview

MHBAP reads five input modalities, extracts features from each, and fuses them
into per-frame behaviour predictions using the **Temporal Cross-Modal Transformer (TCMT)**.

| Modality | Source | Features |
|----------|--------|----------|
| **Face** | Webcam — MediaPipe FaceMesh | 12 Action Unit proxy intensities |
| **Gaze** | Webcam — landmark geometry | Gaze vector, blink L/R, fixation stability (5) |
| **Pose** | Webcam — MediaPipe Pose | Head tilt, shoulder slope, spine curvature, body sway (11) |
| **Voice** | Microphone | 13 MFCCs, pitch, energy, ZCR, spectral centroid, speaking rate (19) |
| **HCI** | OS keyboard/mouse hooks | Speed, acceleration, click rate, keystroke rhythm, interaction entropy (10) |

**Total: 57 features.** Each modality slice is projected to a shared 128-dim token space;
the TCMT produces a 4-class emotion prediction and four continuous indices
(stress, engagement, attention, and fatigue) per inference step.

A second, independently loaded model — **EmotiEffNet-B0** (EfficientNet-B0 fine-tuned
on AffectNet-8) — runs on the raw face crop and returns 8-class emotion probabilities
plus continuous valence and arousal estimates. It operates independently of the TCMT
and is not used as TCMT input.

---

## Key Features

- **Five-modality fusion** — per-modality Linear+LayerNorm projections with learnable
  modality-type embeddings; shared TransformerEncoder with CLS token aggregation.
- **Real-time WebSocket streaming** — Redis Streams pub/sub bus; per-session WebSocket
  push with ping keepalive and exponential back-off reconnection.
- **Explainability** — Integrated Gradients (captum) with a Gradient×Input fallback;
  per-modality importance scores served via REST and visualised in the frontend XAI panel.
- **Evaluation framework** — pure-Python metrics (no scikit-learn): accuracy, precision,
  recall, macro-F1, Cohen's κ, MAE, RMSE, R²; ablation runner over all 2ᴺ−1 modality
  subsets; configurable benchmark endpoint.
- **Auth** — bcrypt passwords, HS256 JWT with token blacklisting, account lockout after
  repeated failures, rate-limiting middleware, content-size limits, security-header middleware.
- **Analytics & export** — per-session statistics, historical prediction queries,
  full prediction history downloadable as CSV.
- **Docker stack** — `docker compose up --build` starts PostgreSQL (TimescaleDB), Redis,
  FastAPI, and React/nginx together.


---

## Architecture & Workflow

```
┌──────────────────────────────────────────────────────────────────┐
│  Browser  (React 18 · Vite · TypeScript · Tailwind)              │
│  Landing · Login · Dashboard · Analytics · Evaluation            │
│  WebSocket ← live gauges, emotion bars, XAI panel               │
└─────────────────────────┬────────────────────────────────────────┘
                          │  HTTP / WebSocket
┌─────────────────────────▼────────────────────────────────────────┐
│  FastAPI  (Python 3.9)                                           │
│  /auth  /users  /sessions  /predictions                          │
│  /stream  /analytics  /evaluation  /health                       │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  ML layer  (singleton, loaded at startup)                  │  │
│  │  FacePipeline · GazePipeline · PosePipeline               │  │
│  │  VoicePipeline · HciListener                              │  │
│  │      → FeatureVector (57 dims)                            │  │
│  │      → TCMT → emotion (4-class) + stress / engagement /   │  │
│  │                attention / fatigue                        │  │
│  │      → EmotiEffNet-B0 → emotion (8-class) + valence/arousal│  │
│  │      → IntegratedGradients → per-modality attributions    │  │
│  └────────────────────────────────────────────────────────────┘  │
│  Redis Streams  (pub/sub · keepalive · reconnect)                │
│  SQLAlchemy async  →  PostgreSQL / TimescaleDB                   │
└──────────────────────────────────────────────────────────────────┘
```

**Session lifecycle**

1. User starts a session from the Dashboard.
2. `SessionRunner` starts capture threads: camera, mic, HCI listener.
3. Each frame batch runs through the per-modality pipelines → 57-dim feature vector.
4. TCMT forward pass produces behaviour predictions; EmotiEffNet-B0 runs on the
   face crop separately.
5. Integrated Gradients attributes each TCMT prediction back to per-feature importance.
6. Results are published to Redis and pushed to the browser over WebSocket.
7. Predictions and feature vectors are persisted to PostgreSQL.

---

## ML Models & Components

### Temporal Cross-Modal Transformer (TCMT)

| Parameter | Value |
|-----------|-------|
| Input features | 57 (5 modality slices) |
| d\_model | 128 |
| Attention heads | 4 |
| Encoder layers | 3 |
| FFN dim | 256 |
| Dropout | 0.15 |
| Emotion classes | 4 (neutral / happy / sad / angry) |
| Regression heads | stress, engagement, attention, fatigue (sigmoid → [0, 1]) |

Each time step is projected per modality (Linear + LayerNorm + learnable modality-type
embedding), concatenated into a token sequence, prepended with a CLS token, and passed
through a shared TransformerEncoder. The CLS output feeds separate prediction heads.

When PyTorch is not installed, the TCMT is replaced by a stub that returns uniform
0.5 values across all heads and equal logits across emotion classes.

**Training data:** FER2013 (9 000 requested), RAF-DB (3 000 requested), and WESAD
(2 000 requested) — 12 500 samples actually loaded. The loaded data was split into
train (9 375) / val (1 250) / test (1 875). A synthetic dataset module
(`ml/training/Dataset.py`) is also included for offline development without sensors.

**Label provenance** (important for interpreting evaluation results):
- *Emotion* labels come from FER2013 and RAF-DB annotations, mapped to the 4-class scheme.
- *Stress* labels are WESAD ground-truth annotations for WESAD samples, and an
  AU+HCI proxy formula for FER/RAF samples where no physiological signal is available.
- *Engagement*, *attention*, and *fatigue* labels are **proxy labels** derived from
  noisy feature-vector dimensions — no public ground-truth dataset exists for these.
  Regression metrics for these three heads measure fit to the proxy labels only.

### EmotiEffNet-B0

Pre-trained EfficientNet-B0 checkpoint (`enet_b0_8_best_afew.pt`, ~16 MB) from the
HSE/asavchenko repo, trained on AffectNet-8. Outputs:

- 8-class softmax: anger, contempt, disgust, fear, happiness, neutral, sadness, surprise
- Continuous valence (−1 to +1) and arousal (0 to 1) from class-prior mappings

The checkpoint is not included in the repository. Download it manually before
running face-crop emotion inference. If the checkpoint is missing, EmotionRecognizer
logs a warning and skips inference for that frame (it does not fall back to random output).

### Explainability (XAI)

| Method | Module | Fallback |
|--------|--------|----------|
| Integrated Gradients | `captum.attr.IntegratedGradients` | Gradient×Input |
| GradCAM | `ml/xai/Gradcam.py` | — |
| Feature importance | Per-modality attribution aggregation | Feature-magnitude heuristic |

Attributions for a session are available at
`GET /api/v1/predictions/Session/{session_id}/xai` (requires authentication).

### Capture Pipelines

| Pipeline | Underlying library | Output dims |
|----------|--------------------|-------------|
| `FacePipeline` | MediaPipe FaceMesh | 12 |
| `GazePipeline` | MediaPipe FaceMesh | 5 |
| `PosePipeline` | MediaPipe Pose | 11 |
| `VoicePipeline` | sounddevice | 19 |
| `HciListener` | pynput | 10 |

All inherit from `BasePipeline`. If a sensor is unavailable (no camera, no mic,
headless CI), the pipeline emits a zero vector — inference keeps running.


---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.9, FastAPI, Uvicorn, SQLAlchemy 2 (async), Alembic |
| Database | PostgreSQL 16 + TimescaleDB |
| Cache / messaging | Redis 7 (Streams + pub/sub) |
| ML | PyTorch 2.3+, timm, MediaPipe, captum |
| Frontend | React 18, TypeScript, Vite, Tailwind CSS, Recharts, Framer Motion |
| Auth | passlib (bcrypt), HS256 JWT |
| Containers | Docker, Docker Compose (multi-stage builds) |
| Testing | pytest, TypeScript compiler |
| Dev tooling | uv / pip, pre-commit, loguru, Makefile |

---

## Project Structure

```
MHBAP/
├── backend/
│   ├── app/
│   │   ├── api/v1/endpoints/   # auth, health, users, sessions, predictions,
│   │   │                       # stream, analytics, evaluation, runner
│   │   ├── core/               # config, logging, Redis, rate limiting,
│   │   │                       # security, security headers
│   │   ├── db/                 # SQLAlchemy models, async session, Alembic migrations
│   │   ├── evaluation/         # Metrics.py, Ablation.py, Benchmark.py
│   │   ├── schemas/            # Pydantic request/response models
│   │   └── services/           # UserService, SessionService,
│   │                           # PredictionService, AnalyticsService
│   ├── tests/
│   │   ├── unit/               # ~20 modules — services, endpoints, XAI, streaming
│   │   └── integration/        # health + session flow (requires live DB)
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/         # MetricGauge, EmotionBar, XAIPanel,
│   │   │                       # TimeSeriesChart, StatusBadge
│   │   ├── hooks/              # useStream, useCurrentUser, useUserAnalytics,
│   │   │                       # useXAISummary, useHistoricalPredictions
│   │   ├── pages/              # Landing, Login, Signup, Dashboard,
│   │   │                       # Analytics, Evaluation
│   │   ├── services/           # api.ts
│   │   └── types/              # index.ts, evaluation.ts
│   ├── nginx.conf
│   └── Dockerfile
├── ml/
│   ├── capture/                # Camera.py, Microphone.py, HciListener.py
│   ├── fusion/                 # Tcmt.py, Predictor.py, FeatureVector.py
│   ├── models/
│   │   ├── EmotionRecognizer.py
│   │   └── weights/            # enet_b0_8_best_afew.pt (download separately)
│   ├── pipelines/              # face/, gaze/, pose/, voice/, hci/
│   ├── training/               # Dataset.py (synthetic), RealDataset.py
│   ├── xai/                    # ShapExplainer.py, Gradcam.py, NlExplainer.py
│   └── SessionRunner.py
├── scripts/                    # training, evaluation, download, diagnostic scripts
├── colab/                      # Colab training notebook + weight sync
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

---

## Setup & Installation

### Requirements

- Docker + Docker Compose, **or** Python 3.9 and Node.js 18+
- 4 GB RAM minimum; 8 GB recommended when running PyTorch inference
- Webcam and microphone for live capture — pipelines fall back to zero-vectors if absent

### Docker (recommended)

```bash
cp .env.example .env
# Set POSTGRES_PASSWORD and SECRET_KEY in .env

docker compose up --build
# Dashboard:  http://localhost
# API docs:   http://localhost:8000/api/docs
```

### Manual development setup

```bash
# Backend
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[ml,dev]"         # or: uv sync --extra ml --extra dev

docker compose up postgres redis -d
uvicorn app.main:app --reload      # http://localhost:8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev                        # http://localhost:5173
```

### EmotiEffNet checkpoint

The EmotiEffNet-B0 checkpoint (`enet_b0_8_best_afew.pt`, ~16 MB) is not included
in the repository. Download it separately and place it in `ml/models/weights/`
before starting the application. If the checkpoint is missing, a warning is
logged and face-crop emotion inference is skipped — everything else continues normally.

### Environment variables

| Variable | Required | Notes |
|----------|----------|-------|
| `POSTGRES_PASSWORD` | Yes | |
| `SECRET_KEY` | Yes | Long random string for JWT signing |
| `REDIS_URL` | No | Default: `redis://redis:6379/0` |
| `ENVIRONMENT` | No | `production` (default) or `development` |
| `LOG_LEVEL` | No | `info` (default), `debug`, `warning` |


---

## Testing

### Backend

```bash
cd backend
python -m pytest tests/ -v
```

Covers: auth, users, sessions, predictions, analytics, evaluation metrics,
streaming, XAI, rate limiting, security headers. Integration tests
(`tests/integration/`) require a live PostgreSQL instance and are skipped otherwise.

### Frontend

```bash
cd frontend
npx tsc --noEmit    # type check
npm run lint
```

### ML tests

```bash
# from repo root
python -m pytest ml/tests/ -v
```

---

## Evaluation

The benchmark and ablation endpoints run against live TCMT inference using the
held-out evaluation split stored in `eval_test_split.npz`. Both endpoints require
authentication. There is no silent fallback to synthetic evaluation data.

```
GET /api/v1/evaluation/benchmark?n_samples=1000   # per-modality + fusion metrics
GET /api/v1/evaluation/ablation?n_samples=500      # 2ᴺ−1 modality subset sweep
```

Metrics are implemented in `backend/app/evaluation/Metrics.py` with no scikit-learn
dependency: accuracy, precision, recall, macro-F1, Cohen's κ (classification);
MAE, RMSE, R² (regression).

**Held-out test set results** (n = 1 875; 9 375 training samples, 1 250 validation samples,
and 1 875 test samples from FER2013, RAF-DB, and WESAD; 75 epochs on a Tesla T4 via Colab):

| Head | Metric | Value |
|------|--------|-------|
| Emotion | Accuracy | 35.5 % |
| Emotion | Macro F1 | 0.315 |
| Emotion | ROC-AUC (OvR) | 0.662 |
| Stress | RMSE | 0.132 |
| Stress | R² | 0.476 |
| Engagement\* | RMSE | 0.073 |
| Engagement\* | R² | 0.553 |
| Attention\* | RMSE | 0.086 |
| Attention\* | R² | 0.357 |
| Fatigue\* | RMSE | 0.065 |
| Fatigue\* | R² | 0.645 |

\* Engagement, attention, and fatigue use **proxy labels** (no public ground-truth
dataset exists). Their regression metrics measure fit to a deterministic feature-derived
formula, not real behavioural ground truth — see `ml/training/RealDataset.py` for
the label logic and `ml/models/weights/tcmt_eval_metrics.json` for the raw numbers.

Emotion accuracy is 35.5 %; per-class F1 varies significantly
(0.11–0.64 across the four classes).

---

## Privacy & Data Handling

- **Local processing only.** Camera frames, audio, and OS events are processed on the
  host running the backend. No raw sensor data is sent to external services.
- **Raw media is not stored.** Only the extracted 57-dim feature vectors and aggregated
  predictions are written to the database. Frame data is discarded after extraction.
- **Auth.** All API endpoints — including `/evaluation` — require a valid JWT.
  The public exceptions are `/api/v1/health/` (liveness), `/api/v1/health/ready`
  (readiness), `/api/v1/auth/register`, and `/api/v1/auth/login`. Passwords are
  stored as bcrypt hashes; tokens are blacklisted on logout and on refresh.
- **Rate limiting** and security headers are applied globally via middleware.
- If you deploy this with real participants, an ethics/IRB review is required first.

---

## References

1. Tsai et al. (2019). *Multimodal Transformer for Unaligned Multimodal Language Sequences.* ACL 2019.
2. Savchenko, A. V. (2022). *HSEmotion: Efficient facial representations for emotion recognition.* ([HSE-asavchenko/face-emotion-recognition](https://github.com/HSE-asavchenko/face-emotion-recognition))
3. Ekman & Friesen (1978). *Facial Action Coding System.*
4. Zadeh et al. (2018). *Multi-attention Recurrent Network for Human Communication Comprehension.* AAAI 2018.
5. Poria et al. (2017). *A review of affective computing: From unimodal analysis to multimodal fusion.* Information Fusion.
6. Sundararajan et al. (2017). *Axiomatic Attribution for Deep Networks (Integrated Gradients).* ICML 2017.
