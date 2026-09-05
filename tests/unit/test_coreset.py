"""Unit tests kiểm tra thuật toán Greedy K-Center Coreset và MemoryBank."""

from __future__ import annotations

import numpy as np
import pytest
from src.model.coreset import greedy_coreset
from src.model.memory_bank import MemoryBank


def test_greedy_coreset_size_and_dimension() -> None:
    """Kiểm tra coreset giảm số điểm nhưng giữ nguyên chiều đặc trưng ban đầu (384D)."""
    features = np.random.randn(100, 384).astype(np.float32)
    selected = greedy_coreset(features, size=15, seed=42)

    assert selected.shape == (15, 384)
    # Các hàng được chọn phải nằm trong tập features ban đầu
    for row in selected:
        assert any(np.allclose(row, orig) for orig in features)


def test_greedy_coreset_invalid_size() -> None:
    """Kiểm tra quăng ValueError khi size coreset <= 0."""
    features = np.random.randn(20, 384).astype(np.float32)
    with pytest.raises(ValueError, match="Kích thước coreset"):
        greedy_coreset(features, size=0)


def test_memory_bank_nearest_neighbors() -> None:
    """Kiểm tra MemoryBank dựng 1-NN index và trả về khoảng cách chính xác."""
    vectors = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    bank = MemoryBank(vectors)

    query = np.array([[0.1, 0.0], [9.9, 10.0]], dtype=np.float32)
    distances, indices = bank.kneighbors(query)

    assert indices[0][0] == 0
    assert indices[1][0] == 1
    assert distances[0][0] < 0.2
