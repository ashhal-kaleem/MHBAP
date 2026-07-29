#!/usr/bin/env python3
"""
scripts/setup.py — First-time MHBAP setup wizard.

Run this once after cloning:
    python scripts/setup.py

Does:
  1. Check Python version (>=3.9)
  2. Install Python deps  (pip install -e ".[dev]")
  3. Copy .env.example -> .env  (if missing)
  4. Check Docker + docker compose
  5. Start postgres + redis via docker compose
  6. Run alembic migrations
  7. Verify server starts (import check)
  8. Run test suite
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"


def run(cmd: str, cwd: Path = ROOT, check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {cmd}")
    return subprocess.run(cmd, shell=True, cwd=cwd, check=check)


def step(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main() -> None:
    step("1 / 8 — Python version check")
    if sys.version_info < (3, 9):
        sys.exit(f"Python 3.9+ required, got {sys.version}")
    print(f"  Python {sys.version.split()[0]} ✅")

    step("2 / 8 — Install Python dependencies")
    run("pip install -e .[dev] --quiet")

    step("3 / 8 — Create .env")
    env_file = ROOT / ".env"
    example  = ROOT / ".env.example"
    if not env_file.exists() and example.exists():
        shutil.copy(example, env_file)
        print("  Copied .env.example → .env")
        print("  ⚠️  Edit .env and set SECRET_KEY, DATABASE_URL, REDIS_URL before prod!")
    else:
        print("  .env already exists ✅")

    step("4 / 8 — Docker check")
    r = run("docker --version", check=False)
    if r.returncode != 0:
        print("  ❌ Docker not found. Install Docker Desktop then re-run this script.")
        print("     Windows: winget install Docker.DockerDesktop")
        sys.exit(1)
    run("docker compose version")

    step("5 / 8 — Start postgres + redis")
    run("docker compose up -d db redis")
    import time; time.sleep(5)   # give postgres a moment

    step("6 / 8 — Run Alembic migrations")
    run("python -m alembic upgrade head", cwd=BACKEND)

    step("7 / 8 — Verify app import")
    run('python -c "from backend.app.main import app; print(\'App import OK\')"')

    step("8 / 8 — Run test suite")
    run("python -m pytest backend/tests/ -q --tb=short")

    print("\n" + "="*60)
    print("  ✅  Setup complete!")
    print("  Start backend : uvicorn backend.app.main:app --reload")
    print("  Start frontend: cd frontend && npm run dev")
    print("  Full stack    : docker compose up --build")
    print("="*60)


if __name__ == "__main__":
    main()
