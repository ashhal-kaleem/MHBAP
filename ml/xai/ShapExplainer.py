"""Deprecated alias. The explainer computes Integrated Gradients, not SHAP.
Use ml.xai.IntegratedGradients.IntegratedGradientsExplainer."""
from ml.xai.IntegratedGradients import (  # noqa: F401
    IntegratedGradientsExplainer, SHAPExplainer, TARGET_HEADS, _CAPTUM_AVAILABLE, _TORCH_AVAILABLE,
)
