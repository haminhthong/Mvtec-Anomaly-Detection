"""Module trích xuất đặc trưng PatchCore đa tầng (Multi-Layer Feature Extractor).

Re-export từ package src.model.patch_embedding để đảm bảo tính tương thích ngược hoàn toàn.
"""

from __future__ import annotations

from .model.patch_embedding import FeatureExtractor

__all__ = ["FeatureExtractor"]
