"""Module định vị và xử lý bản đồ nhiệt khuyết tật (Defect Localization).

Cung cấp các hàm:
- Gaussian smoothing làm mịn bản đồ nhiệt khoảng cách patch.
- Tính toán tỷ lệ diện tích bất thường (Anomalous Area Ratio).
- Tạo ảnh phủ màu bản đồ nhiệt (Heatmap Overlay) xuất ra chuỗi Base64 PNG.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


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


def compute_anomalous_area_ratio(
    heatmap: np.ndarray, pixel_threshold: float
) -> float:
    """Tính toán tỷ lệ diện tích bề mặt có dấu hiệu bất thường vượt pixel_threshold.

    Args:
        heatmap: Mảng 2D bản đồ nhiệt đã làm mịn [H, W].
        pixel_threshold: Ngưỡng phát hiện lỗi cấp pixel đã hiệu chỉnh.

    Returns:
        float: Tỷ lệ diện tích lỗi trong khoảng [0.0, 1.0].
    """
    if heatmap.size == 0 or pixel_threshold <= 0:
        return 0.0
    anomalous_pixels = np.sum(heatmap >= pixel_threshold)
    return float(anomalous_pixels / heatmap.size)


def create_heatmap_overlay_b64(
    image: Image.Image,
    heatmap: np.ndarray,
    threshold: float | None = None,
    alpha: float = 0.45,
) -> str:
    """Tạo ảnh trực quan hóa vùng lỗi (Heatmap Overlay) trên ảnh gốc và mã hóa thành Base64 PNG.

    Args:
        image: Ảnh gốc PIL.Image.
        heatmap: Mảng 2D bản đồ nhiệt anomaly map [H, W].
        threshold: Ngưỡng phát hiện lỗi tùy chọn.
        alpha: Tỷ lệ hòa trộn màu sắc heatmap lên ảnh gốc (mặc định 0.45).

    Returns:
        str: Chuỗi mã hóa Base64 dạng 'data:image/png;base64,...'.
    """
    img_resized = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    img_np = np.asarray(img_resized, dtype=np.float32) / 255.0

    # Chuẩn hóa heatmap về dải [0, 1]
    h_min, h_max = float(heatmap.min()), float(heatmap.max())
    norm_heat = (heatmap - h_min) / (h_max - h_min + 1e-8)
    norm_heat = np.clip(norm_heat, 0.0, 1.0)

    # Phóng to heatmap khớp kích thước 224x224
    heat_pil = Image.fromarray((norm_heat * 255).astype(np.uint8)).resize(
        (224, 224), Image.Resampling.BILINEAR
    )
    heat_resized = np.asarray(heat_pil, dtype=np.float32) / 255.0

    # Tạo Jet-like colormap không phụ thuộc matplotlib
    r = np.clip(1.5 - np.abs(heat_resized * 4.0 - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(heat_resized * 4.0 - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(heat_resized * 4.0 - 1.0), 0.0, 1.0)
    color_map = np.stack([r, g, b], axis=-1)

    # Hòa trộn overlay
    overlay = (1.0 - alpha) * img_np + alpha * color_map
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)

    result_img = Image.fromarray(overlay)
    buffer = io.BytesIO()
    result_img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"
