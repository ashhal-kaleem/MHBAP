# ── Stage 1: builder ────────────────────────────────────────────────────────
# Build context is the repo root so pyproject.toml is reachable.
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps: build tools + OpenCV runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging manifest first to exploit Docker layer cache.
COPY pyproject.toml .

# ── base deps (FastAPI, DB, auth, numpy) ──────────────────────────────────
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir .

# ── ML inference deps ─────────────────────────────────────────────────────
# torch CPU-only saves ~1.5 GB vs CUDA build.
# timm: EfficientNet-B0 backbone for EmotionRecognizer.
# scikit-learn + scipy: evaluation metrics (benchmark / ablation endpoints).
# opencv-python-headless: frame preprocessing without X11.
# datasets: HuggingFace loader used by cache_eval_test_split.py.
# shap: SHAP explainer in SHAPExplainer.
# Hardware-capture extras (sounddevice, pynput, mediapipe, opensmile) are
# excluded -- they require audio/video devices absent in the server container;
# SessionRunner detects missing capture hardware and falls back gracefully.
RUN pip install --prefix=/install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1+cpu \
    torchvision==0.18.1+cpu \
    timm==1.0.3 \
    "scikit-learn>=1.5.0" \
    "scipy>=1.13.0" \
    "pandas>=2.2.0" \
    "opencv-python-headless>=4.10.0" \
    "Pillow>=10.3.0" \
    "datasets>=2.20.0" \
    "shap>=0.45.0"


# ── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="MHBAP Backend"
LABEL org.opencontainers.image.description="Multimodal Human Behaviour Analysis Platform — FastAPI backend"

# Non-root user
RUN groupadd -r mhbap && useradd -r -g mhbap -s /sbin/nologin mhbap

# Runtime system libs (build tools excluded)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Pull ALL installed packages (base + ML) from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# FastAPI application source
COPY backend/ .

# ML package: SessionRunner, all pipelines, TCMT, EmotionRecognizer, weights.
# Placed at /ml so benchmark.py's _REPO_ROOT path resolution works:
#   /app/app/evaluation/benchmark.py  →  parent×4  =  /
#   _REPO_ROOT / "ml" / ...           =  /ml / ...
COPY ml/ /ml

RUN chown -R mhbap:mhbap /app /ml
USER mhbap

# PYTHONPATH=/ makes `import ml` resolve to /ml (top-level package).
# /app is also on the path so `import app` (FastAPI) keeps working.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    PYTHONPATH=/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health/')"

CMD ["sh", "-c", "alembic upgrade head && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2"]
