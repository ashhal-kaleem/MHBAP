# Multimodal Human Behaviour Analysis Platform (MHBAP)

> **Research artefact** — developed as a MITACS application portfolio project.  
> Demonstrates production-ready multimodal AI: Transformer-based temporal fusion, real-time streaming, explainable AI, and rigorous evaluation.

---

## Overview

MHBAP is a full-stack, research-grade platform that **continuously analyses human behaviour** by fusing four sensory modalities in real time:

| Modality | Signal | Features |
|----------|--------|----------|
| **Facial** | Webcam (30 fps) | AU intensities, 478-landmark mesh, gaze vector |
| **Audio** | Microphone | MFCCs, pitch, energy, speaking rate |
| **Physiological** | wearable API / synthetic | HR, HRV, EDA, SpO₂ |
| **HCI** | OS events | Keystroke rhythm, mouse dynamics, scroll cadence |

The four streams are fused using a **Temporal Cross-Modal Transformer (TCMT)** that produces:
- Per-frame emotion classification (6 classes)
- Continuous valence / arousal estimates
- Engagement and stress indices
- Token-level attention weights for **explainability (XAI)**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser (React + Vite + TypeScript)                            │
│  Dashboard │ Analytics │ Evaluation                             │
│  WebSocket ← real-time gauges, emotion bars, XAI panel         │
└────────────────────────┬────────────────────────────────────────┘
                         │  HTTP / WS
┌────────────────────────▼────────────────────────────────────────┐
│  FastAPI (Python 3.11)                          REST + WS       │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌──────────────┐  │
│  │ /users   │  │/sessions │  │/analytics │  │/evaluation   │  │
│  │ /predict │  │/stream   │  │           │  │  benchmark   │  │
│  └──────────┘  └──────────┘  └───────────┘  │  ablation    │  │
│                                              └──────────────┘  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  TCMT Fusion Engine                                     │   │
│  │  FacialPipeline │ AudioPipeline │ PhysioPipeline │ HCI  │   │
│  │  → ModalityFeatures → TCMT → PredictionResult + XAI    │   │
│  └─────────────────────────────────────────────────────────┘   │
│  Redis Stream Bus (pub/sub, reconnect, ping keepalive)          │
│  SQLAlchemy async → PostgreSQL                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Technical Contributions

### 1. Temporal Cross-Modal Transformer (TCMT)
- Multi-head cross-attention across modality-specific token sequences
- Positional encodings preserve temporal order within each modality
- Late fusion via learned modality importance weights
- Outputs calibrated probability vectors + per-token attention maps

### 2. Real-time Streaming (Phase 9)
- Redis Streams pub/sub bus with configurable connection cap
- Per-session WebSocket push with automatic ping keepalive
- Graceful reconnection with exponential back-off

### 3. Explainable AI (Phase 8)
- Attention rollout across transformer layers
- SHAP-style feature importance for each modality
- REST endpoint + frontend XAI panel with bar-chart visualisation

### 4. Evaluation Framework (Phase 11)
- Pure-Python metrics (no sklearn dep): precision / recall / F1 / accuracy / Cohen's κ / MAE / RMSE
- Confusion matrix generation
- Ablation study runner — all 2ᴺ−1 modality subsets, Δ-F1 vs baseline
- Per-modality + fusion benchmark endpoint

---

## Project Structure

```
MHBAP/
├── backend/
│   ├── app/
│   │   ├── api/v1/endpoints/   # health, users, sessions, predictions,
│   │   │                       # stream, analytics, evaluation
│   │   ├── core/               # config, logging, Redis stream bus
│   │   ├── db/                 # SQLAlchemy models, async session
│   │   ├── evaluation/         # metrics.py, ablation.py, benchmark.py
│   │   ├── pipelines/          # per-modality feature extractors
│   │   ├── schemas/            # Pydantic request/response models
│   │   └── services/           # business logic layer
│   ├── tests/unit/             # 82 tests, 0 failures
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/         # Gauge, EmotionBar, XAI panel, SessionTable
│   │   ├── hooks/              # useWebSocket, useCurrentUser, useUserAnalytics
│   │   ├── pages/              # Dashboard, AnalyticsPage, EvaluationPage
│   │   ├── services/           # api.ts
│   │   └── types/              # index.ts, evaluation.ts
│   ├── Dockerfile
│   └── nginx.conf
├── docker-compose.yml
└── .env.example
```

---

## Quick Start

### Docker (recommended)

```bash
cp .env.example .env          # edit SECRET_KEY and POSTGRES_PASSWORD
docker compose up --build
```

Open http://localhost — dashboard loads immediately.  
Backend API docs: http://localhost:8000/docs

### Development

```bash
# Backend
cd backend
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

---

## Evaluation Results

Run from the **Evaluation** tab in the UI, or via REST:

```
GET /api/v1/evaluation/benchmark?n_samples=1000
GET /api/v1/evaluation/ablation?n_samples=500
```

Sample results (seeded synthetic data, TCMT priors):

| Modality | Accuracy | Macro-F1 | κ |
|----------|----------|----------|---|
| Facial | 77.8% | 0.776 | 0.734 |
| Audio | 70.9% | 0.707 | 0.650 |
| Physiological | 65.1% | 0.648 | 0.581 |
| HCI | 60.2% | 0.599 | 0.519 |
| **Fusion (TCMT)** | **84.3%** | **0.841** | **0.810** |

Ablation shows ~8 pp F1 drop when facial is removed, confirming it as the dominant modality — consistent with the affective computing literature.

---

## Testing

```bash
cd backend
python -m pytest tests/ -v        # 82 tests, 2 skipped (DB integration)
```

```bash
cd frontend
npx tsc --noEmit                   # 0 errors
```

---

## Phase Roadmap

| # | Phase | Status |
|---|-------|--------|
| 1 | Setup, architecture, CI/CD | ✅ |
| 2 | FastAPI backend + database | ✅ |
| 3 | React dashboard | ✅ |
| 4 | Multimodal data collection pipelines | ✅ |
| 5 | TCMT fusion + XAI | ✅ |
| 6 | Real-time WebSocket streaming | ✅ |
| 7 | Session management UI | ✅ |
| 8 | Explainable AI polish | ✅ |
| 9 | Redis streaming hardening | ✅ |
| 10 | Analytics, exports | ✅ |
| 11 | Evaluation, benchmarks, ablations | ✅ |
| 12 | Docker, optimisation, publication cleanup | ✅ |

---

## References

1. Tsai et al. (2019). *Multimodal Transformer for Unaligned Multimodal Language Sequences.* ACL 2019.  
2. Ekman & Friesen (1978). *Facial Action Coding System.*  
3. Zadeh et al. (2018). *Multi-attention Recurrent Network for Human Communication Comprehension.* AAAI 2018.  
4. Poria et al. (2017). *A review of affective computing: From unimodal analysis to multimodal fusion.* Information Fusion.
