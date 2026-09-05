"""Module thuật toán Coreset K-Center Greedy tối ưu hóa kích thước Memory Bank.

Thực hiện:
Full Memory [N, 384]
    ↓
Random Projection (Johnson-Lindenstrauss) [N, 64]
    ↓
Greedy K-Center Selection
    ↓
Selected Indices
    ↓
Original 384D features at selected indices
    ↓
Final Compact Memory Bank [K, 384]

Đảm bảo phép chiếu ngẫu nhiên chỉ phục vụ chọn coreset; các tác vụ so khớp 1-NN
sau đó đều diễn ra trên không gian gốc 384 chiều.
"""

from __future__ import annotations

import numpy as np


def greedy_coreset(features: np.ndarray, size: int, seed: int = 42) -> np.ndarray:
    """Thuật toán Greedy K-Center Coreset giảm quy mô Memory Bank.

    Args:
        features: Mảng 2D chứa các vector patch đặc trưng gốc [N, Dim] (ví dụ: [N, 384]).
        size: Kích thước tập coreset cần giữ lại (K).
        seed: Seed sinh ma trận chiếu ngẫu nhiên và chọn điểm khởi tạo.

    Returns:
        np.ndarray: Mảng 2D tập coreset đặc trưng gốc [K, Dim] (ví dụ: [K, 384]).

    Raises:
        ValueError: Nếu kích thước coreset truyền vào không hợp lệ.
    """
    if size <= 0:
        raise ValueError("Kích thước coreset phải là một số nguyên dương > 0.")
    if len(features) <= size:
        return features

    rng = np.random.default_rng(seed)

    # 1. Chiếu ngẫu nhiên Johnson-Lindenstrauss xuống 64D để tăng tốc tính khoảng cách K-Center
    projected = features
    if features.shape[1] > 64:
        projection = rng.normal(size=(features.shape[1], 64)).astype(np.float32)
        projection /= np.sqrt(64.0)
        projected = features @ projection

    # 2. Khởi tạo điểm ngẫu nhiên đầu tiên
    selected: list[int] = [int(rng.integers(len(features)))]
    min_dist = np.full(len(features), np.inf, dtype=np.float32)

    # 3. Lần lượt chọn K-1 điểm tiếp theo có khoảng cách cực đại tới tập các tâm đã chọn
    for _ in range(1, size):
        center = projected[selected[-1]]
        distance = np.sum((projected - center) ** 2, axis=1)
        min_dist = np.minimum(min_dist, distance)
        selected.append(int(np.argmax(min_dist)))

    # 4. Trích xuất đúng các vector đặc trưng trong không gian gốc 384D tại các chỉ số được chọn
    return features[np.asarray(selected)]
