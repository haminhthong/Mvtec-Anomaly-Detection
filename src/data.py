"""Module quản lý dữ liệu hình ảnh và tiền xử lý cho dataset MVTec AD.

Cung cấp PreprocessingConfig, build_transform, ImageFolderDataset và find_category_root.
Tập tin này re-export từ package src.data để đảm bảo tính tương thích ngược hoàn toàn.
"""

from __future__ import annotations

from .data import (
    DEFAULT_PREPROCESSING_CONFIG,
    TFM,
    ImageFolderDataset,
    PreprocessingConfig,
    build_transform,
    find_category_root,
)

__all__ = [
    "ImageFolderDataset",
    "find_category_root",
    "PreprocessingConfig",
    "build_transform",
    "DEFAULT_PREPROCESSING_CONFIG",
    "TFM",
]
