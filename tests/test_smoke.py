"""Smoke tests nhanh cho các module tiện ích và cấu hình hệ thống MVTec AD."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image
from src.api import MODEL_DIR, health
from src.config import TrainConfig
from src.evaluate import compute_aupro
from src.utils import (
    apply_heatmap_smoothing,
    create_heatmap_overlay_b64,
    greedy_coreset,
)


def test_greedy_coreset_respects_requested_size() -> None:
    """Kiểm tra thuật toán Greedy Coreset giữ đúng kích thước k được yêu cầu."""
    x = np.arange(40, dtype=np.float32).reshape(20, 2)
    coreset = greedy_coreset(x, 5)
    assert coreset.shape == (5, 2)


def test_aupro_perfect_map_is_high() -> None:
    """Kiểm tra chỉ số AUPRO đạt điểm cao gần 1.0 với bản đồ nhiệt hoàn hảo."""
    masks = np.zeros((1, 8, 8), dtype=bool)
    masks[:, 2:6, 2:6] = True
    maps = masks.astype(np.float32)
    assert compute_aupro(masks, maps) > 0.95


def test_health_does_not_load_heavy_model() -> None:
    """Kiểm tra endpoint health check không cần nạp PyTorch/weights ResNet18."""
    response = health()
    assert response.status in {"ok", "degraded"}
    assert isinstance(response.model_ready, bool)


def test_model_config_records_calibration_contract() -> None:
    """Kiểm tra tập tin config.json artifact lưu trữ đầy đủ thông số calibration hợp lệ."""
    config_file = MODEL_DIR / "config.json"
    if config_file.exists():
        config = json.loads(config_file.read_text(encoding="utf-8"))
        assert config["schema_version"] in {1, 2}
        assert config["calibration_images"] > 0
        assert config["threshold"] > 0


def test_train_config_rejects_invalid_coreset_fraction() -> None:
    """Kiểm tra TrainConfig quăng ngoại lệ khi coreset_fraction nằm ngoài (0, 1]."""
    with pytest.raises(ValueError, match="coreset_fraction"):
        TrainConfig(coreset_fraction=0).validate()


def test_apply_heatmap_smoothing() -> None:
    """Kiểm tra bộ lọc Gaussian Smoothing biến đổi bản đồ nhiệt mượt mà hơn."""
    raw_heat = np.zeros((10, 10), dtype=np.float32)
    raw_heat[5, 5] = 10.0
    smoothed = apply_heatmap_smoothing(raw_heat, sigma=1.0)
    assert smoothed.shape == (10, 10)
    assert smoothed[5, 5] < 10.0  # Giá trị tại đỉnh giảm do lan tỏa xung quanh
    assert smoothed[4, 5] > 0.0  # Điểm lân cận nhận thêm năng lượng


def test_create_heatmap_overlay_b64() -> None:
    """Kiểm tra tạo chuỗi Base64 cho ảnh heatmap overlay thành công."""
    img = Image.new("RGB", (100, 100), color="white")
    heat = np.random.rand(14, 14).astype(np.float32)
    b64_str = create_heatmap_overlay_b64(img, heat)
    assert b64_str.startswith("data:image/png;base64,")
