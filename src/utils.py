"""Module tiện ích chứa các hàm hỗ trợ hệ thống (Utility Functions).

Cung cấp các hàm đặt seed tái lập kết quả ngẫu nhiên, lưu JSON,
thuật toán lấy mẫu Coreset K-Center greedy tối ưu hóa bộ nhớ, và tiện ích xử lý
bản đồ nhiệt (Gaussian Smoothing & Heatmap Overlay Base64).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import random
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

LOGGER: logging.Logger = logging.getLogger("mvtec_anomaly_detection")


def set_seed(seed: int = 42) -> None:
    """Cố định seed ngẫu nhiên cho Python, NumPy và PyTorch để đảm bảo tính tái lập.

    Args:
        seed: Giá trị seed ngẫu nhiên (mặc định: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        LOGGER.debug("Không tìm thấy PyTorch; chỉ đặt seed cho random và NumPy.")


def save_json(path: str | Path, payload: dict) -> None:
    """Ghi dữ liệu dictionary ra tập tin JSON với định dạng utf-8 thụt lề thụt dòng sạch đẹp.

    Args:
        path: Đường dẫn tập tin đầu ra.
        payload: Dữ liệu dictionary cần lưu.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def greedy_coreset(features: np.ndarray, size: int, seed: int = 42) -> np.ndarray:
    """Thuật toán Greedy K-Center Coreset giảm quy mô Memory Bank.

    Thuật toán chọn tập con đại diện k điểm từ tập đặc trưng N điểm sao cho khoảng cách
    từ bất kỳ điểm nào tới tâm gần nhất được cực tiểu hóa (tối đa hóa độ phủ không gian).

    Để giảm chi phí tính toán khi chiều vector lớn (>64), áp dụng phép chiếu ngẫu nhiên
    Johnson-Lindenstrauss (Random Projection) xuống 64 chiều.

    Args:
        features: Mảng 2D chứa các vector patch đặc trưng [N, Dim].
        size: Kích thước tập coreset cần giữ lại (k).
        seed: Seed sinh ma trận chiếu ngẫu nhiên.

    Returns:
        np.ndarray: Mảng 2D tập coreset đặc trưng đã được rút gọn [size, Dim].

    Raises:
        ValueError: Nếu kích thước coreset truyền vào không hợp lệ.
    """
    if size <= 0:
        raise ValueError("Kích thước coreset phải là một số nguyên dương > 0.")
    if len(features) <= size:
        return features

    rng = np.random.default_rng(seed)

    # Chiếu ngẫu nhiên giảm chiều vector xuống 64D nếu chiều ban đầu lớn
    projected = features
    if features.shape[1] > 64:
        projection = rng.normal(size=(features.shape[1], 64)).astype(np.float32)
        projection /= np.sqrt(64.0)
        projected = features @ projection

    # Khởi tạo điểm ngẫu nhiên đầu tiên
    selected = [int(rng.integers(len(features)))]
    min_dist = np.full(len(features), np.inf, dtype=np.float32)

    # Chọn k-1 điểm tiếp theo bằng cách lấy điểm xa nhất so với các tâm đã chọn
    for _ in range(1, size):
        center = projected[selected[-1]]
        distance = np.sum((projected - center) ** 2, axis=1)
        min_dist = np.minimum(min_dist, distance)
        selected.append(int(np.argmax(min_dist)))

    return features[np.asarray(selected)]


def apply_heatmap_smoothing(heatmap: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Áp dụng bộ lọc Gaussian Blur làm mịn bản đồ nhiệt anomaly map.

    Args:
        heatmap: Mảng 2D chứa giá trị khoảng cách bất thường [H, W].
        sigma: Độ lệch chuẩn của bộ lọc Gaussian (nếu sigma <= 0 sẽ giữ nguyên).

    Returns:
        np.ndarray: Mảng 2D bản đồ nhiệt đã được làm mịn.
    """
    if sigma <= 0:
        return heatmap
    return gaussian_filter(heatmap.astype(np.float32), sigma=sigma)


def create_heatmap_overlay_b64(
    image: Image.Image,
    heatmap: np.ndarray,
    threshold: float | None = None,
    alpha: float = 0.5,
) -> str:
    """Tạo ảnh trực quan hóa vùng lỗi (Heatmap Overlay) trên ảnh gốc và mã hóa thành chuỗi Base64 PNG.

    Args:
        image: Ảnh gốc PIL.Image.
        heatmap: Mảng 2D bản đồ nhiệt anomaly map [H, W].
        threshold: Ngưỡng phát hiện lỗi (nếu truyền vào sẽ làm nổi bật vùng vượt ngưỡng).
        alpha: Tỷ lệ hòa trộn màu sắc heatmap lên ảnh gốc (0.0 đến 1.0).

    Returns:
        str: Chuỗi mã hóa Base64 dạng 'data:image/png;base64,...'.
    """
    # Resize ảnh gốc và heatmap về 224x224
    img_resized = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    img_np = np.asarray(img_resized, dtype=np.float32) / 255.0

    # Chuẩn hóa heatmap về dải [0, 1]
    h_min, h_max = heatmap.min(), heatmap.max()
    norm_heat = (heatmap - h_min) / (h_max - h_min + 1e-8)
    norm_heat = np.clip(norm_heat, 0.0, 1.0)

    # Phóng to heatmap khớp kích thước 224x224
    heat_pil = Image.fromarray((norm_heat * 255).astype(np.uint8)).resize(
        (224, 224), Image.Resampling.BILINEAR
    )
    heat_resized = np.asarray(heat_pil, dtype=np.float32) / 255.0

    # Tạo Jet-like Color map đơn giản không phụ thuộc matplotlib (Blue -> Green -> Yellow -> Red)
    # Channel R, G, B tính toán theo cường độ heat
    r = np.clip(1.5 - np.abs(heat_resized * 4.0 - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(heat_resized * 4.0 - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(heat_resized * 4.0 - 1.0), 0.0, 1.0)
    color_map = np.stack([r, g, b], axis=-1)

    # Hòa trộn màu overlay với ảnh gốc
    overlay = (1.0 - alpha) * img_np + alpha * color_map
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)

    # Chuyển đổi thành PIL Image và xuất ra buffer PNG Base64
    result_img = Image.fromarray(overlay)
    buffer = io.BytesIO()
    result_img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"
