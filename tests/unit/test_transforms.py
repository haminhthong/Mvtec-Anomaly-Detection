"""Unit tests kiểm tra tính nhất quán của pipeline tiền xử lý (Transforms)."""

from __future__ import annotations

import torch
from PIL import Image
from src.data.transforms import PreprocessingConfig, build_transform


def test_build_transform_output() -> None:
    """Kiểm tra build_transform chuyển đổi PIL image thành Tensor đúng kích thước và kiểu."""
    cfg = PreprocessingConfig(image_size=(224, 224))
    tfm = build_transform(cfg)

    img = Image.new("RGB", (300, 400), color=(128, 64, 32))
    tensor = tfm(img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_train_and_inference_preprocessing_consistency() -> None:
    """Đảm bảo cùng cấu hình PreprocessingConfig sinh ra kết quả tensor hoàn toàn trùng khớp."""
    cfg = PreprocessingConfig(image_size=(224, 224))
    train_tfm = build_transform(cfg)
    infer_tfm = build_transform(cfg)

    img = Image.new("RGB", (250, 250), color=(100, 150, 200))
    tensor_train = train_tfm(img)
    tensor_infer = infer_tfm(img)

    assert torch.equal(tensor_train, tensor_infer)
