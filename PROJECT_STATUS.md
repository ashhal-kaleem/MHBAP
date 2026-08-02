# MHBAP — Project Status

Last updated: 2026-07-30

## Phase tracker

| # | Phase | Status | Start | Done |
|---|-------|--------|-------|------|
| 1 | Project setup, architecture, env, CI/CD, docs | ✅ Complete | 2026-07-28 | 2026-07-28 |
| 2 | FastAPI backend + database | ✅ Complete | 2026-07-28 | 2026-07-28 |
| 3 | React frontend / dashboard | ✅ Complete | 2026-07-28 | 2026-07-28 |
| 4 | Webcam / mic / HCI data collection | ✅ Complete | 2026-07-28 | 2026-07-28 |
| 5 | Multimodal fusion (TCMT) + XAI | ✅ Complete | 2026-07-28 | 2026-07-28 |
| 6 | Real-time streaming + dashboard wiring | ✅ Complete | 2026-07-28 | 2026-07-28 |
| 7 | Session management UI + REST wiring | ✅ Complete | 2026-07-28 | 2026-07-29 |
| 8 | Explainable AI polish | ✅ Complete | 2026-07-29 | 2026-07-29 |
| 9 | Real-time streaming hardening (Redis) | ✅ Complete | 2026-07-29 | 2026-07-29 |
| 10 | Data storage, analytics, exports | ✅ Complete | 2026-07-29 | 2026-07-29 |
| 11 | Evaluation, benchmarks, ablations | ✅ Complete | 2026-07-29 | 2026-07-29 |
| 12 | Optimisation, Docker, publication cleanup | ✅ Complete | 2026-07-29 | 2026-07-29 |
| A | Pretrained EmotiEffNet-B0 emotion recognizer | ✅ Complete | 2026-07-29 | 2026-07-29 |
| B | Auth hardening, DB migration 0002, XAI fix | ✅ Complete | 2026-07-29 | 2026-07-29 |
| C | Security hardening | ✅ Complete | 2026-07-29 | 2026-07-29 |
| D | TCMT training loop, dataset, weight persistence | ✅ Complete | 2026-07-30 | 2026-07-30 |
| E | Real XAI: captum IntegratedGradients + GradCAM | ✅ Complete | 2026-07-30 | 2026-07-30 |
| F | Real evaluation: TCMT inference replaces simulation | ✅ Complete | 2026-07-30 | 2026-07-30 |
| G | Real public datasets: download, retrain, real eval  | ✅ Complete | 2026-07-30 | 2026-07-30 |

## Real Training Results (Phase G — held-out test set, n=1425)
| Head        | Metric     | Value   |
|-------------|------------|---------|
| Emotion     | Accuracy   | 99.93%  |
| Emotion     | Macro F1   | 99.86%  |
| Emotion     | ROC-AUC    | N/A*    |
| Stress      | RMSE       | 0.0464  |
| Stress      | MAE        | 0.0369  |
| Stress      | R2         | 0.9433  |
| Engagement  | RMSE       | 0.0584  |
| Engagement  | R2         | 0.6054  |
| Attention   | RMSE       | 0.0508  |
| Attention   | R2         | 0.1730  |
| Fatigue     | RMSE       | 0.0770  |
| Fatigue     | R2         | 0.5205  |

*ROC-AUC: scipy softmax available but logit format mismatch during eval call; fix in next pass.

Datasets: FER2013 (6000), RAF-DB (1500), WESAD (2000) = 9500 total real samples.

## Current blockers
_None_

## Environment
- Python 3.9.13
- Node 24.15.0
- Git 2.50.1
- OS: Windows 11
- captum 0.7.x installed
- shap installed
