# MHBAP — Project Status

Last updated: 2026-10-25

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

## Results

The earlier "99.93% emotion accuracy" table was removed: it came from a run where the
label leaked into the features. The current honest numbers (emotion 35.5% accuracy,
macro-F1 0.315 on 4 classes) are in README.md → Evaluation and
`ml/models/weights/tcmt_eval_metrics.json`.

## Phase H — Research-grade benchmark (done, 2026-10-25)
- [x] DAiSEE loader + feature extraction with the live MediaPipe pipelines (`ml/training/DaiseeDataset.py`, `scripts/daisee/`)
- [x] VideoTCMT with temporal position embedding, padding mask, modality masking (`ml/fusion/VideoTcmt.py`)
- [x] Train/eval script: official splits, 3 seeds, majority baseline, temporal + modality ablations
- [x] Tests (`ml/tests/TestDaisee.py`) and Colab notebook (`colab/DAiSEE_Benchmark_Colab.ipynb`)
- [x] Downloaded DAiSEE, extracted features for all 8,571 labelled clips, trained 8 settings x 3 seeds (results in `results/daisee/`)
- [x] Results table + honest reading in README (full model test macro-F1: boredom 0.316, engagement 0.275, confusion 0.299, frustration 0.265; temporal gain over static is small)
- [x] Faithful XAI: renamed SHAPExplainer -> IntegratedGradientsExplainer and GradCAM -> FaceAUSaliency (old names kept as aliases), fixed saliency running in train() mode, added `ml/xai/Faithfulness.py` (IG, G×I, gradient, deletion/insertion, model randomisation, completeness) + tests, evaluated on DAiSEE (`results/daisee/xai_faithfulness.*`)
- [ ] Next: learned per-frame visual features (EfficientNet) on DAiSEE; image-level Grad-CAM on the EfficientNet face crop

## Current blockers
_None_

## Environment
- Python 3.9.13
- Node 24.15.0
- Git 2.50.1
- OS: Windows 11
- captum 0.7.x installed
- shap installed
