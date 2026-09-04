"""Bộ unit test kiểm tra toàn diện pipeline PatchCore (Feature Extractor, Dataset, Config, API Schemas)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image
from src.api import HealthResponse, InspectionResponse
from src.config import TrainConfig
from src.data import ImageFolderDataset
from src.features import FeatureExtractor
from src.train import split_normal_paths


def test_feature_extractor_output_shape() -> None:
    """Kiểm tra FeatureExtractor trích xuất đúng kích thước vector đặc trưng đa tầng ResNet18."""
    extractor = FeatureExtractor()
    dummy_input = torch.randn(2, 3, 224, 224)

    # 224x224 input qua Layer 2 (28x28) và Layer 3 (14x14 upsampled lên 28x28)
    # Channel Layer 2 (128) + Layer 3 (256) = 384 channels. Total patches per image = 28 * 28 = 784.
    patches = extractor(dummy_input)
    assert patches.shape == (2 * 784, 384)

    spatial_patches, (h, w) = extractor.extract_spatial_features(dummy_input)
    assert (h, w) == (28, 28)
    assert spatial_patches.shape == (2 * 784, 384)


def test_image_folder_dataset() -> None:
    """Kiểm tra ImageFolderDataset nạp ảnh và tiền xử lý tensor chính xác."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        img1_path = tmp_path / "test1.png"
        img2_path = tmp_path / "test2.png"

        Image.new("RGB", (100, 100), color="red").save(img1_path)
        Image.new("RGB", (100, 100), color="blue").save(img2_path)

        ds = ImageFolderDataset([img1_path, img2_path])
        assert len(ds) == 2

        tensor, path_str = ds[0]
        assert tensor.shape == (3, 224, 224)
        assert str(img1_path) in path_str


def test_train_config_validations() -> None:
    """Kiểm tra các trường hợp bắt lỗi trong TrainConfig."""
    with pytest.raises(ValueError, match="category"):
        TrainConfig(category="").validate()

    with pytest.raises(ValueError, match="batch_size"):
        TrainConfig(batch_size=0).validate()

    with pytest.raises(ValueError, match="calibration_fraction"):
        TrainConfig(calibration_fraction=0.8).validate()

    with pytest.raises(ValueError, match="coreset tối thiểu/tối đa"):
        TrainConfig(min_coreset_size=500, max_coreset_size=100).validate()


def test_random_calibration_split_is_reproducible_and_disjoint(tmp_path: Path) -> None:
    """Calibration phải tái lập được và không giao với memory bank."""
    paths = [tmp_path / f"{index:03d}.png" for index in range(20)]
    first_memory, first_calibration = split_normal_paths(paths, 0.2, seed=42)
    second_memory, second_calibration = split_normal_paths(paths, 0.2, seed=42)
    assert first_memory == second_memory
    assert first_calibration == second_calibration
    assert set(first_memory).isdisjoint(first_calibration)
    assert len(first_calibration) == 4


def test_api_pydantic_schemas() -> None:
    """Kiểm tra tính đúng đắn của các Pydantic Schemas trong API."""
    health_resp = HealthResponse(status="ok", model_ready=True, model_version="v3")
    assert health_resp.status == "ok"
    assert health_resp.model_ready is True

    inspect_resp = InspectionResponse(
        anomaly_score=0.45,
        threshold=0.50,
        decision="PASS",
        heatmap_shape=[28, 28],
        model_version="v3",
        overlay_b64="data:image/png;base64,sample",
    )
    assert inspect_resp.decision == "PASS"
    assert inspect_resp.anomaly_score < inspect_resp.threshold
