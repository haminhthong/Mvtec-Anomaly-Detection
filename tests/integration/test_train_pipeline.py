"""Integration tests kiểm tra quy trình huấn luyện tạo artifact Design B."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from src.config import TrainConfig
from src.training.trainer import train_patchcore


def test_train_pipeline_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tạo dummy MVTec AD dataset và chạy train_patchcore kiểm tra artifact Design B."""
    # Thiết lập thư mục data/raw/dummy_cat/train/good
    raw_dir = tmp_path / "data" / "raw"
    category = "dummy_cat"
    good_dir = raw_dir / category / "train" / "good"
    good_dir.mkdir(parents=True, exist_ok=True)

    # Sinh 25 ảnh mẫu normal
    for i in range(25):
        img = Image.new("RGB", (100, 100), color=(i * 10, i * 5, 200))
        img.save(good_dir / f"{i:03d}.png")

    # Monkeypatch find_category_root để trỏ tới thư mục tạm
    import src.training.trainer as trainer_mod
    monkeypatch.setattr(trainer_mod, "find_category_root", lambda **kwargs: raw_dir / category)

    # Đổi thư mục models về thư mục tạm
    models_dir = tmp_path / "models"
    monkeypatch.setattr(trainer_mod, "Path", lambda p: models_dir if str(p) == "models" else Path(p))

    cfg = TrainConfig(
        category=category,
        batch_size=4,
        calibration_fraction=0.2,
        min_calibration_samples=5,
        review_quantile=0.90,
        threshold_quantile=0.98,
        pixel_quantile=0.98,
        coreset_fraction=0.2,
        min_coreset_size=10,
        max_coreset_size=50,
        smooth_sigma=1.0,
    )

    payload = train_patchcore(cfg)
    assert payload["category"] == category
    assert payload["schema_version"] == 3
    assert payload["thresholds"]["review_threshold"] <= payload["thresholds"]["fail_threshold"]

    # Kiểm tra artifact Design B
    cat_dir = models_dir / category
    assert (cat_dir / "memory_bank.npy").exists()
    assert (cat_dir / "config.json").exists()

    loaded_mem = np.load(cat_dir / "memory_bank.npy")
    assert loaded_mem.ndim == 2
    assert loaded_mem.shape[1] == 384
