# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning: `MAJOR.PHASE.PATCH` — e.g. `0.1.0` = phase 1 complete.

---

## [Unreleased]

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
