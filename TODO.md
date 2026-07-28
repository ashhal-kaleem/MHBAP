# MHBAP — TODO

## Phase 1 (current)
- [x] Folder structure
- [x] pyproject.toml
- [x] requirements.txt
- [x] .gitignore
- [x] Makefile
- [x] Pre-commit config
- [x] GitHub Actions CI
- [x] README / STATUS / CHANGELOG
- [x] Architecture docs
- [ ] Git init + first commit

## Phase 2
- [ ] FastAPI app skeleton (main.py, config, lifespan)
- [ ] SQLAlchemy models: Session, Prediction, ModalityFeature, User
- [ ] Alembic migrations
- [ ] TimescaleDB hypertable setup
- [ ] Redis connection pool
- [ ] Health check endpoint
- [ ] Session CRUD endpoints
- [ ] Prediction write/read endpoints
- [ ] WebSocket endpoint skeleton
- [ ] Unit tests (pytest)
- [ ] Postman/OpenAPI docs export

## Phase 3
- [ ] Vite + React 18 + TypeScript scaffold
- [ ] Tailwind CSS + shadcn/ui
- [ ] WebSocket hook
- [ ] Live signal chart (Recharts)
- [ ] XAI explanation card
- [ ] Session summary heatmap
- [ ] Educator alert list

## Phase 4
- [ ] Webcam capture module (OpenCV)
- [ ] Microphone capture (sounddevice / pyaudio)
- [ ] HCI logger (pynput keyboard + mouse)
- [ ] Frame queue + async producer/consumer
- [ ] Synchronisation timestamps

## Phase 5 — ML pipelines
- [ ] Face: EmotiEffNet / AffectNet fine-tune
- [ ] Head pose: 6D-RepNet
- [ ] Gaze: L2CSNet + blink detector
- [ ] Body pose: MediaPipe Holistic
- [ ] Voice: wav2vec 2.0 + openSMILE
- [ ] HCI: feature engineering module

## Phase 6
- [ ] TCMT architecture (PyTorch)
- [ ] Missing-modality masking
- [ ] Learned confidence weights
- [ ] Training loop + validation

## Phase 7
- [ ] 5 output heads
- [ ] Multi-task loss with correlation matrix
- [ ] Inference pipeline end-to-end

## Phase 8
- [ ] SHAP TreeExplainer / GradientExplainer
- [ ] GradCAM on face crops
- [ ] Attention map visualisation
- [ ] NL explanation generator

## Phase 9
- [ ] FastAPI WebSocket streaming (1 Hz predictions)
- [ ] React live update
- [ ] Latency profiling

## Phase 10
- [ ] TimescaleDB continuous aggregates
- [ ] Session export (CSV, JSON, HDF5)
- [ ] Annotation EMA widget

## Phase 11
- [ ] Per-modality ablation runner
- [ ] Cross-context generalisation test
- [ ] Benchmark comparison table

## Phase 12
- [ ] Docker Compose (all services)
- [ ] Dockerfile per service
- [ ] Performance profiling + optimisation
- [ ] Publication-ready code cleanup
- [ ] ArXiv draft skeleton
