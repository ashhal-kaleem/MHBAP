# MHBAP — Makefile
# Usage: make <target>

.PHONY: help install test test-unit test-integration lint tsc docker-build docker-up docker-down clean

help:
	@echo "MHBAP dev commands:"
	@echo "  make install        Install backend deps (pip install -e .)"
	@echo "  make test           Run full test suite"
	@echo "  make test-unit      Unit tests only"
	@echo "  make test-int       Integration tests only"
	@echo "  make lint           Ruff + mypy"
	@echo "  make tsc            TypeScript type-check frontend"
	@echo "  make docker-build   Build all Docker images"
	@echo "  make docker-up      Start full stack (postgres+redis+backend+frontend)"
	@echo "  make docker-down    Stop and remove containers"
	@echo "  make clean          Remove __pycache__, .pytest_cache, dist"

install:
	pip install -e ".[dev]"

test:
	python -m pytest backend/tests/ -q

test-unit:
	python -m pytest backend/tests/unit/ -q

test-int:
	python -m pytest backend/tests/integration/ -q

lint:
	ruff check backend/ ml/
	mypy backend/app --ignore-missing-imports

tsc:
	cd frontend && npx tsc --noEmit

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null; \
	rm -rf dist/ build/ *.egg-info/
