"""Module tiện ích chứa các hàm hỗ trợ hệ thống (Utility Functions).

Re-export các hàm từ các subsystem chuyên biệt:
- set_seed từ src.training.trainer
- greedy_coreset từ src.model.coreset
- apply_heatmap_smoothing, create_heatmap_overlay_b64 từ src.inference.localization
- save_json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .inference.localization import (
    apply_heatmap_smoothing,
    compute_anomalous_area_ratio,
    create_heatmap_overlay_b64,
)
from .model.coreset import greedy_coreset
from .training.trainer import set_seed

LOGGER: logging.Logger = logging.getLogger("mvtec_anomaly_detection")


def save_json(path: str | Path, payload: dict) -> None:
    """Ghi dữ liệu dictionary ra tập tin JSON với định dạng utf-8 thụt lề sạch đẹp."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = [
    "set_seed",
    "save_json",
    "greedy_coreset",
    "apply_heatmap_smoothing",
    "compute_anomalous_area_ratio",
    "create_heatmap_overlay_b64",
    "LOGGER",
]
