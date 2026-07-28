# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning: `MAJOR.PHASE.PATCH` — e.g. `0.1.0` = phase 1 complete.

---

## [Unreleased]

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
