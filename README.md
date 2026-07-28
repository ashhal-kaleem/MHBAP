# MHBAP — Multimodal Human Behavior Analysis Platform

> Research-grade affective computing infrastructure for real-time estimation of
> **emotion · stress · engagement · attention · fatigue** from six synchronized
> modalities.  Designed for MITACS / international research internship publication.

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB)](https://reactjs.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Phase](https://img.shields.io/badge/Phase-1%20%E2%9C%85-brightgreen)](PROJECT_STATUS.md)

---

## Architecture overview

```
Webcam / Mic / HCI
       │
       ▼
┌──────────────────────────────────────────────────────┐
│  ML Pipeline (ml/)                                   │
│  face · gaze · head_pose · body_pose · voice · hci  │
│              │                                       │
│     Temporal Cross-Modal Transformer (TCMT)          │
│              │                                       │
│  5 output heads: emotion · stress · engagement ·    │
│                  attention · fatigue                 │
│              │                                       │
│     XAI: SHAP · GradCAM · attention maps            │
└──────────────────┬───────────────────────────────────┘
                   │ JSON/WebSocket
       ┌───────────▼────────────┐
       │  FastAPI backend       │
       │  PostgreSQL/Timescale  │
       │  Redis queue           │
       └───────────┬────────────┘
                   │ REST / WS
       ┌───────────▼────────────┐
       │  React Dashboard       │
       │  Live charts · XAI UI  │
       │  Educator alerts       │
       └────────────────────────┘
```

## Quick start

```bash
# 1. Clone & enter
git clone <repo_url>
cd MHBAP

# 2. Python environment
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"

# 3. Backend
cd backend && uvicorn app.main:app --reload

# 4. Frontend
cd frontend && npm install && npm run dev

# 5. Docker (all services)
docker compose up --build
```

## Project structure

```
MHBAP/
├── backend/          FastAPI application + DB models
├── frontend/         React 18 dashboard
├── ml/               AI pipelines, fusion, XAI, evaluation
│   ├── pipelines/    Per-modality feature extractors
│   ├── fusion/       Temporal Cross-Modal Transformer
│   ├── xai/          SHAP, GradCAM, NL explanations
│   └── evaluation/   Metrics, ablations, benchmarks
├── data/             Session recordings, exports, annotations
├── docs/             Architecture diagrams, API docs, research notes
├── scripts/          Setup, seed, migration helpers
├── docker/           Dockerfiles + compose
└── .github/          CI/CD workflows
```

## Publication targets

| Venue | Type | Fit |
|-------|------|-----|
| IEEE TAFFC | Journal | Fusion + XAI contribution |
| ACM CHI | Conference | XAI user study |
| ACII | Conference | Affective computing system |
| IEEE FG | Conference | Face/gaze pipeline |

## Phase status → see [PROJECT_STATUS.md](PROJECT_STATUS.md)
