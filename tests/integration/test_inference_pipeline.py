"""Integration tests kiểm tra quy trình suy luận (Inference Pipeline) với Design B."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from src.inference.detector import AnomalyDetector


def test_inference_with_design_b_artifact(tmp_path: Path) -> None:
    """Kiểm tra AnomalyDetector nạp memory_bank.npy và trả về rich inspection payload."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Tạo dummy memory bank [20, 384]
    memory = np.random.randn(20, 384).astype(np.float32)
    np.save(model_dir / "memory_bank.npy", memory)

    # Tạo config.json
    config_data = {
        "schema_version": 3,
        "category": "test_box",
        "version": "test-v5",
        "smooth_sigma": 1.0,
        "threshold": 3.5,
        "review_threshold": 2.8,
        "pixel_threshold": 3.0,
        "preprocessing": {
            "image_size": [224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }
    (model_dir / "config.json").write_text(
        json.dumps(config_data), encoding="utf-8"
    )

    det = AnomalyDetector(model_dir=str(model_dir))
    assert det.threshold == 3.5
    assert det.review_threshold == 2.8
    assert det.pixel_threshold == 3.0
    assert det.memory_bank.size == 20

    img = Image.new("RGB", (200, 200), color="white")
    s, heat = det.score(img)
    assert isinstance(s, float)
    assert heat.shape == (28, 28)

    res = det.inspect(img, include_overlay=True)
    assert "inspection_id" in res
    assert "prediction" in res
    assert "localization" in res
    assert "model" in res
    assert res["prediction"]["decision"] in {"PASS", "REVIEW", "FAIL"}
    assert res["prediction"]["severity"] in {"PASS", "REVIEW", "FAIL_MINOR", "FAIL_MAJOR"}
    assert res["overlay_b64"].startswith("data:image/png;base64,")
