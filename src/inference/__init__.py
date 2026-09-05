"""Inference package for PatchCore Anomaly Detection."""

from __future__ import annotations

from .decision import classify_decision_and_severity
from .detector import AnomalyDetector
from .localization import (
    apply_heatmap_smoothing,
    compute_anomalous_area_ratio,
    create_heatmap_overlay_b64,
)

__all__ = [
    "AnomalyDetector",
    "apply_heatmap_smoothing",
    "compute_anomalous_area_ratio",
    "create_heatmap_overlay_b64",
    "classify_decision_and_severity",
]
