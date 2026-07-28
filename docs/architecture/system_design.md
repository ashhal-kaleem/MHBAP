# MHBAP System Architecture

## Design principles

1. **Modularity** — each modality pipeline is independently testable and replaceable.
2. **Graceful degradation** — system predicts with any subset of available modalities.
3. **Research-first** — every design decision exposes hooks for ablation/evaluation.
4. **Privacy by design** — no raw video stored server-side; on-device processing.
5. **Publication target** — code and experiments reproducible from config files alone.

---

## Layer breakdown

### Layer 0 — Data capture (`ml/capture/`)
- `webcam.py` — OpenCV frame producer, 30 FPS → async queue
- `microphone.py` — sounddevice chunk producer, 16 kHz mono
- `hci_logger.py` — pynput keyboard + mouse event logger → feature extractor

### Layer 1 — Per-modality feature extractors (`ml/pipelines/`)
| Module | Model | Output shape | FPS |
|--------|-------|-------------|-----|
| `face/` | EmotiEffNet-B0 | (8,) emotion logits + (2,) VA | 10 |
| `gaze/` | L2CSNet + blink detector | (3,) gaze dir + scalar blink_rate | 10 |
| `pose/` | 6D-RepNet | (6,) rotation matrix repr | 10 |
| `body/` | MediaPipe Holistic | (33*3,) joint coords | 30→1 avg |
| `voice/` | wav2vec-2.0 + openSMILE | (768,) + (88,) | async |
| `hci/` | Feature engineering | (12,) HCI features | 1 Hz |

### Layer 2 — Temporal encoder (inside TCMT)
- Per-modality 1D-TCN or Bi-LSTM over 30 s sliding window
- Output: (batch, modality, d_model=128) embeddings at 1 Hz

### Layer 3 — Cross-modal fusion: TCMT (`ml/fusion/`)
- Transformer encoder with cross-attention between modality tokens
- Learned confidence weights (softmax gating per sample)
- Missing-modality masking via padding + attention mask

### Layer 4 — Output heads (`ml/fusion/heads.py`)
- 5 independent MLP heads sharing fused representation
- Multi-task loss: `L = Σ_t w_t * L_t` with learned task weights

### Layer 5 — XAI (`ml/xai/`)
- `shap_explainer.py` — per-modality SHAP importance
- `gradcam.py` — spatial attention on face crops
- `nl_explainer.py` — top-K SHAP → natural language string

### Layer 6 — Backend (`backend/`)
- FastAPI + SQLAlchemy 2.0 async
- PostgreSQL + TimescaleDB for time-series predictions
- Redis for frame queue + pub/sub (inference results → WS clients)

### Layer 7 — Frontend (`frontend/`)
- React 18 + TypeScript + Vite
- WebSocket hook → live prediction stream
- Recharts for signal timelines
- Shadcn/ui for educator alert panel

---

## Data flow (inference path)

```
Webcam frame (33ms)
        │
        ▼  (async queue, Redis)
Face extractor ──► gaze extractor ──► pose extractor
        │                 │                 │
        └──────────────── temporal buffer ──┘
                               │  (30s window fills)
                               ▼
                    TCMT forward pass
                               │
                         5 output heads
                               │
                    XAI: SHAP + GradCAM (async, 5s)
                               │
                    FastAPI WS → React dashboard
                               │
                    TimescaleDB INSERT (1 row/s/session)
```

---

## Key research contributions

1. **TCMT with missing-modality masking** — novel variant of cross-modal transformer that tolerates incomplete sensor data at inference time without retraining.
2. **Unified multi-target XAI** — single explanation pipeline covering 5 correlated targets with per-modality SHAP + face GradCAM + NL generation.
3. **HCI as affective signal** — systematic feature engineering of keyboard/mouse dynamics as stress proxy; first integration with vision-audio fusion in an engagement context.
4. **EMA annotation paradigm** — 5-minute pulse-check prompts provide pseudo-ground-truth aligned temporally with model predictions.
