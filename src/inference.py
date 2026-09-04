"""Module thực hiện suy luận (Inference Engine) cho hệ thống kiểm tra lỗi ngoại quan.

Cung cấp lớp AnomalyDetector để tải mô hình đã huấn luyện, thực hiện trích xuất
đặc trưng ảnh đầu vào, tìm kiếm láng giềng gần nhất trong Memory Bank Coreset,
tính toán bản đồ nhiệt lỗi (Anomaly Heatmap), làm mịn bằng Gaussian Filter và
tạo ảnh trực quan hóa vùng lỗi (Visual Overlay Base64).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from PIL import Image

from .data import TFM
from .features import FeatureExtractor
from .utils import apply_heatmap_smoothing, create_heatmap_overlay_b64


class AnomalyDetector:
    """Mô hình phát hiện lỗi ngoại quan dựa trên thuật toán PatchCore.

    Attributes:
        nn: Chỉ mục k-NN đã fit trên tập coreset memory bank.
        config: Cấu hình và thông số ngưỡng (threshold) đọc từ file JSON.
        model_version: Phiên bản artifact mô hình.
        smooth_sigma: Độ mịn bộ lọc Gaussian làm mịn anomaly map.
        dev: Thiết bị tính toán PyTorch ('cuda' hoặc 'cpu').
        net: Mạng trích xuất đặc trưng FeatureExtractor ResNet18.
    """

    def __init__(self, model_dir: str | Path = "models") -> None:
        """Khởi tạo mô hình AnomalyDetector từ thư mục chứa artifacts.

        Args:
            model_dir: Đường dẫn tới thư mục lưu trữ artifacts (mặc định: 'models').

        Raises:
            FileNotFoundError: Nếu không tìm thấy tập tin mô hình patch_nn.joblib hoặc config.json.
        """
        model_path = Path(model_dir)
        nn_file = model_path / "patch_nn.joblib"
        config_file = model_path / "config.json"

        if not nn_file.exists() or not config_file.exists():
            raise FileNotFoundError(
                f"Không tìm thấy artifacts mô hình tại '{model_path}'. "
                "Vui lòng chạy 'python -m src.train' trước khi thực hiện suy luận."
            )

        self.nn = joblib.load(nn_file)
        self.config: dict[str, Any] = json.loads(
            config_file.read_text(encoding="utf-8")
        )
        self.model_version: str = self.config.get("version", "unknown")
        self.smooth_sigma: float = float(self.config.get("smooth_sigma", 1.0))
        self.threshold: float = float(self.config.get("threshold", 0.0))

        self.dev: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.net: FeatureExtractor = FeatureExtractor().to(self.dev)

    @torch.inference_mode()
    def score(self, image: Image.Image) -> tuple[float, np.ndarray]:
        """Tính toán điểm số bất thường (anomaly score) và bản đồ nhiệt lỗi (heatmap).

        Args:
            image: Ảnh PIL đầu vào.

        Returns:
            tuple[float, np.ndarray]:
                - anomaly_score (float): Điểm số bất toàn diện của bức ảnh (percentile 99 của khoảng cách patch đã làm mịn).
                - smoothed_heatmap (np.ndarray): Mảng 2D bản đồ nhiệt khoảng cách patch đã làm mịn [H_map, W_map].
        """
        x = TFM(image.convert("RGB")).unsqueeze(0).to(self.dev)
        patches, (h, w) = self.net.extract_spatial_features(x)

        # Tìm kiếm khoảng cách tới patch gần nhất trong Memory Bank
        dist, _ = self.nn.kneighbors(patches.cpu().numpy())
        raw_heat = dist.reshape(h, w)

        # Làm mịn bản đồ nhiệt bằng bộ lọc Gaussian
        smoothed_heat = apply_heatmap_smoothing(raw_heat, sigma=self.smooth_sigma)

        # Điểm số bất thường lấy theo 99th percentile khoảng cách patch
        image_score = float(np.percentile(smoothed_heat, 99))
        return image_score, smoothed_heat

    @torch.inference_mode()
    def inspect(
        self, image: Image.Image, include_overlay: bool = True
    ) -> dict[str, Any]:
        """Kiểm định bức ảnh: tính score, xác định quyết định PASS/REVIEW/FAIL và tùy chọn tạo ảnh overlay Base64.

        Args:
            image: Ảnh PIL đầu vào.
            include_overlay: Tùy chọn có tạo chuỗi Base64 overlay bản đồ nhiệt hay không (mặc định: True).

        Returns:
            dict[str, Any]: Dictionary chứa đầy đủ kết quả kiểm định trực quan.
        """
        score, smoothed_heat = self.score(image)
        ratio = score / (self.threshold + 1e-12)

        # Logic đưa ra quyết định vận hành nhà máy:
        # - score >= threshold: FAIL (Lỗi ngoại quan)
        # - 0.8 <= ratio < 1.0: REVIEW (Cần nhân viên QC xem xét thủ công)
        # - ratio < 0.8: PASS (Sản phẩm đạt chuẩn)
        if score >= self.threshold:
            decision = "FAIL"
        elif ratio >= 0.8:
            decision = "REVIEW"
        else:
            decision = "PASS"

        overlay_b64 = (
            create_heatmap_overlay_b64(
                image=image,
                heatmap=smoothed_heat,
                threshold=self.threshold,
                alpha=0.45,
            )
            if include_overlay
            else None
        )

        return {
            "anomaly_score": score,
            "threshold": self.threshold,
            "decision": decision,
            "heatmap_shape": list(smoothed_heat.shape),
            "model_version": self.model_version,
            "overlay_b64": overlay_b64,
        }

    @torch.inference_mode()
    def inspect_detailed(self, image: Image.Image) -> dict[str, Any]:
        """Alias cho inspect(image, include_overlay=True) giữ tương thích ngược."""
        return self.inspect(image, include_overlay=True)

