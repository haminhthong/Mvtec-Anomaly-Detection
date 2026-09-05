"""Evaluation package for PatchCore Anomaly Detection."""

from __future__ import annotations

from .aupro import compute_aupro
from .evaluator import evaluate_category
from .metrics import calculate_3tier_metrics

__all__ = [
    "compute_aupro",
    "calculate_3tier_metrics",
    "evaluate_category",
]
