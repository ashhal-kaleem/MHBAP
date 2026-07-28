.PHONY: help install dev-install lint fmt test coverage backend frontend docker-up docker-down clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n",$$1,$$2}'

install:  ## Install production dependencies
	pip install -e .

dev-install:  ## Install all dependency groups
	pip install -e ".[ml,dev,docs]"
	pre-commit install

lint:  ## Run ruff + mypy
	ruff check backend ml scripts
	mypy backend ml

fmt:  ## Auto-format with black + ruff
	black backend ml scripts
	ruff check --fix backend ml scripts

test:  ## Run pytest
	pytest

coverage:  ## Run tests with HTML coverage report
	pytest --cov=backend --cov=ml --cov-report=html
	start htmlcov/index.html

backend:  ## Start FastAPI dev server
	cd backend && uvicorn app.main:app --reload --port 8000

frontend:  ## Start Vite dev server
	cd frontend && npm run dev

docker-up:  ## Start all services via Docker Compose
	docker compose -f docker/docker-compose.yml up --build

docker-down:  ## Stop all Docker services
	docker compose -f docker/docker-compose.yml down -v

migrate:  ## Run Alembic migrations
	cd backend && alembic upgrade head

clean:  ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>nul || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>nul || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>nul || true
	find . -name "*.pyc" -delete 2>nul || true
