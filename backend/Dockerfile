# ── Stage 1: builder ────────────────────────────────────────────────────────
# Build context is the repo root so pyproject.toml is reachable.
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps required to compile native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the packaging manifest first to exploit Docker layer cache:
# re-installs only when pyproject.toml changes, not on every source edit.
COPY pyproject.toml .

# Install runtime dependencies declared in [project].dependencies.
# --no-build-isolation keeps the build reproducible without a network call
# for build-backend packages that are already present in the base image.
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir .

# ── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="MHBAP Backend"
LABEL org.opencontainers.image.description="Multimodal Human Behaviour Analysis Platform — FastAPI backend"

# Non-root user
RUN groupadd -r mhbap && useradd -r -g mhbap -s /sbin/nologin mhbap

# Runtime system libs (same set as builder, minus build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Pull installed packages from the builder stage
COPY --from=builder /install /usr/local

WORKDIR /app
# Build context is repo root; copy only the backend application source
COPY backend/ .

RUN chown -R mhbap:mhbap /app
USER mhbap

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health/')"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
