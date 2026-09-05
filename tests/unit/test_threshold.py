"""Unit tests kiểm tra Held-out Normal Calibration và Dual-Thresholds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from src.training.calibration import calibrate_thresholds, split_normal_paths


def test_split_normal_paths_disjoint_and_reproducible(tmp_path: Path) -> None:
    """Tách normal paths phải tái lập được, không giao nhau và đủ kích thước."""
    paths = [tmp_path / f"{i:03d}.png" for i in range(100)]
    mem1, cal1 = split_normal_paths(paths, calibration_fraction=0.2, seed=42, min_calibration_samples=20)
    mem2, cal2 = split_normal_paths(paths, calibration_fraction=0.2, seed=42, min_calibration_samples=20)

    assert mem1 == mem2
    assert cal1 == cal2
    assert set(mem1).isdisjoint(cal1)
    assert len(cal1) == 20
    assert len(mem1) == 80


def test_calibrate_thresholds_ordering() -> None:
    """Kiểm tra quan hệ thứ tự: review_threshold (P95) < fail_threshold (P99)."""
    # 100 mẫu phân phối normal từ 1.0 đến 5.0
    scores = list(np.linspace(1.0, 5.0, 100))
    heatmaps = [np.full((28, 28), s, dtype=np.float32) for s in scores]

    review_th, fail_th, pixel_th = calibrate_thresholds(
        scores, heatmaps, review_quantile=0.95, fail_quantile=0.99, pixel_quantile=0.99
    )

    assert review_th < fail_th
    assert pixel_th > 0.0
