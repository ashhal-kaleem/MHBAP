# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning: `MAJOR.PHASE.PATCH` — e.g. `0.1.0` = phase 1 complete.

---

## [Unreleased]

## [0.7.0] — 2026-07-30 — Phase G: Real Dataset Training & Evaluation

### Added
- `ml/training/real_dataset.py` — loaders for FER2013, RAF-DB, WESAD (HuggingFace)
  - FER2013 (clip-benchmark/wds_fer2013): 6000 samples, 7-class -> 4-class MHBAP mapping
  - RAF-DB (deanngkl/raf-db-7emotions): 1500 samples (train split only; test split unavailable)
  - WESAD (LouisSimon/wesad-parquet): 2000 samples, physiological stress labels
  - Total: 9500 real samples; train=7125 / val=950 / test=1425
- `ml/training/train_tcmt.py` — full training loop using real data, saves weights + metrics
- `ml/evaluation/metrics.py` — accuracy/F1/ROC-AUC for emotion; RMSE/MAE/R2 for regression
- `ml/models/weights/tcmt_trained.pt` — trained checkpoint (305 KB)
- `ml/models/weights/tcmt_eval_metrics.json` — held-out test metrics

### Real Experimental Results (held-out test set, n=1425)
**Emotion classification** (4-class: neutral/happy/sad/angry):
- Accuracy:  99.93%
- Macro F1:  99.86%
- Per-class F1: neutral=99.84%, happy=99.58%, sad=100.00%, angry=100.00%
- Confusion matrix: [[313,0,0,0],[1,119,0,0],[0,0,197,0],[0,0,0,795]]

**Stress regression** (WESAD ground-truth labels):
- RMSE: 0.0464  |  MAE: 0.0369  |  R2: 0.9433

**Engagement regression** (derived from HCI/gaze features):
- RMSE: 0.0584  |  MAE: 0.0478  |  R2: 0.6054

**Attention regression** (derived from gaze/blink features):
- RMSE: 0.0508  |  MAE: 0.0417  |  R2: 0.1730

**Fatigue regression** (derived from pause/dwell/energy features):
- RMSE: 0.0770  |  MAE: 0.0602  |  R2: 0.5205

### Notes
- Engagement/attention/fatigue have no public labelled test sets; labels are rule-derived
  from real signal features (documented design decision; R2 reflects label consistency).
- Attention R2 low (0.17) — blink/gaze proxies from FER image pixels are noisy;
  will improve when real gaze hardware data is available.
- Unicode arrow bug in print fixed (cp1252 codec on Windows).


## [0.3.0] — 2026-07-28 — Phase 3: React Dashboard

### Added
- `frontend/` — Vite 5 + React 18 + TypeScript + Tailwind CSS scaffold
- `src/types/index.ts` — shared TS types (Prediction, Session, WsMessage, MetricSeries)
- `src/hooks/useStream.ts` — WebSocket hook with auto-reconnect (3 s), 120-frame ring buffer
- `src/services/api.ts` — typed fetch wrappers for sessions, predictions, health endpoints
- `src/components/StatusBadge.tsx` — live/connecting/offline/error indicator with pulse animation
- `src/components/MetricGauge.tsx` — animated SVG radial gauge for stress/engagement/attention/fatigue
- `src/components/EmotionBar.tsx` — sorted horizontal bars for all emotion score probabilities
- `src/components/TimeSeriesChart.tsx` — Recharts line chart, 120-point rolling window
- `src/components/XAIPanel.tsx` — SHAP modality contribution bars + NL explanation text
- `src/pages/Dashboard.tsx` — full single-page dashboard wiring all components
- Vite proxy: `/api` → `localhost:8000`, `/ws` → `ws://localhost:8000`
- Manual chunk split: react / recharts / icons for better caching
- `.env.example` with `VITE_WS_URL` and `VITE_DEMO_SESSION_ID`
- `frontend/dist/` and `frontend/node_modules/` added to root `.gitignore`

## [0.2.0] — 2026-07-28 — Phase 2: Database Layer & API Skeleton

### Added
- `backend/app/db/base.py` — SQLAlchemy declarative base with constraint naming convention
- `backend/app/db/session.py` — async engine, session factory, `get_db` dependency, health helpers
- `backend/app/core/redis.py` — Redis connection pool, health check, graceful close
- ORM models: `User`, `Session`, `ModalityFeature` (TimescaleDB hypertable), `Prediction` (hypertable)
- Alembic migration `0001_initial_schema` with `create_hypertable()` calls for time-series tables
- `docker/init_timescale.sql` — enables TimescaleDB extension on DB init
- Pydantic v2 schemas for users, sessions, and predictions (with `[0,1]` validators on score fields)
- Service layer: `session_service`, `prediction_service`, `user_service` — full async CRUD
- API endpoints: `/users`, `/sessions`, `/predictions`, `/stream` (WebSocket skeleton)
- `main.py` lifespan wires DB + Redis init/dispose
- Unit tests for all services (8 tests, 0 warnings, 67% overall coverage)

## [0.1.0] — 2026-07-28 — Phase 1: Project Setup

### Added
- Full monorepo folder structure (backend / frontend / ml / data / docs / scripts / docker)
- `pyproject.toml` with all dependency groups (core, ml, dev, docs)
- `requirements.txt` pinned snapshot
- `.gitignore` for Python + Node + ML artifacts
- `Makefile` with dev shortcuts
- Pre-commit config (ruff, black, mypy, prettier)
- GitHub Actions CI workflow (lint + test on push/PR)
- `README.md`, `PROJECT_STATUS.md`, `CHANGELOG.md`, `TODO.md`
- Architecture documentation in `docs/architecture/`
- MIT License
