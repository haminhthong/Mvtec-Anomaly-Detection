"""Module thực hiện suy luận (Inference Engine) cho hệ thống kiểm tra lỗi ngoại quan.

Triển khai AnomalyDetector theo Thiết kế B:
- Nạp cấu hình PreprocessingConfig đồng bộ với quá trình huấn luyện.
- Nạp memory_bank.npy (hoặc fallback memory.npy) và dựng 1-NN index tại runtime.
- Tính toán anomaly score (99th percentile compromise), bản đồ nhiệt làm mịn Gaussian.
- Ra quyết định kép (PASS/REVIEW/FAIL), phân cấp độ nghiêm trọng (Severity) và tỷ lệ diện tích lỗi.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import uuid

import numpy as np
import torch
from PIL import Image

from ..data.transforms import PreprocessingConfig, build_transform
from ..model.memory_bank import MemoryBank
from ..model.patch_embedding import FeatureExtractor
from .decision import classify_decision_and_severity
from .localization import (
    apply_heatmap_smoothing,
    compute_anomalous_area_ratio,
    create_heatmap_overlay_b64,
)


class AnomalyDetector:
    """Mô hình phát hiện lỗi ngoại quan PatchCore-style.

    Attributes:
        model_dir: Thư mục chứa model artifacts.
        config: Dictionary cấu hình đọc từ config.json.
        category: Tên danh mục sản phẩm (ví dụ: 'bottle').
        model_version: Phiên bản model artifact.
        threshold: Ngưỡng phát hiện lỗi FAIL (P99 normal calibration).
        review_threshold: Ngưỡng cảnh báo rà soát REVIEW (P95 normal calibration).
        pixel_threshold: Ngưỡng phát hiện lỗi cấp pixel (P99 normal heatmap pixels).
        smooth_sigma: Độ lệch chuẩn Gaussian smoothing.
        memory_bank: Đối tượng MemoryBank quản lý vector coreset và runtime 1-NN index.
        net: FeatureExtractor trích xuất Layer2 + Layer3 ResNet18.
    """

    def __init__(
        self, model_dir: str | Path = "models", category: str | None = None
    ) -> None:
        """Khởi tạo mô hình AnomalyDetector.

        Args:
            model_dir: Đường dẫn thư mục lưu trữ artifacts (mặc định: 'models').
            category: Danh mục sản phẩm tùy chọn. Nếu truyền vào, sẽ tìm kiếm trong model_dir/category.

        Raises:
            FileNotFoundError: Nếu thiếu file config.json hoặc file memory_bank.npy.
            ValueError: Nếu threshold trong config không hợp lệ (<= 0).
        """
        base_path = Path(model_dir)

        # Xác định thư mục artifact (hỗ trợ cả cấu trúc models/{category}/ và models/)
        target_dir = base_path
        if category:
            candidate = base_path / category
            if candidate.exists() and (candidate / "config.json").exists():
                target_dir = candidate

        config_file = target_dir / "config.json"
        if not config_file.exists() and (base_path / "config.json").exists():
            config_file = base_path / "config.json"

        if not config_file.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file config tại '{config_file}'. "
                "Vui lòng chạy 'python -m src.train' trước khi thực hiện suy luận."
            )

        self.config: dict[str, Any] = json.loads(
            config_file.read_text(encoding="utf-8")
        )
        self.category: str = self.config.get("category", category or "bottle")
        self.model_version: str = self.config.get("version", "unknown")
        self.smooth_sigma: float = float(self.config.get("smooth_sigma", 1.0))

        # Đọc ngưỡng FAIL / Image threshold
        th_val = self.config.get("threshold")
        if th_val is None:
            th_val = self.config.get("thresholds", {}).get("fail_threshold")

        if th_val is None:
            raise ValueError(
                f"Config tại '{config_file}' thiếu thông số 'threshold'. "
                "Không thể suy luận khi chưa có ngưỡng phát hiện lỗi được căn chỉnh."
            )
        self.threshold: float = float(th_val)
        if self.threshold <= 0.0:
            raise ValueError(
                f"Ngưỡng threshold ({self.threshold}) không hợp lệ (phải > 0.0)."
            )

        # Đọc review_threshold và pixel_threshold
        thresholds_dict = self.config.get("thresholds", {})
        self.review_threshold: float = float(
            self.config.get(
                "review_threshold",
                thresholds_dict.get("review_threshold", 0.8 * self.threshold),
            )
        )
        self.pixel_threshold: float = float(
            self.config.get(
                "pixel_threshold",
                thresholds_dict.get("pixel_threshold", self.threshold),
            )
        )

        # Tìm kiếm tập tin memory_bank (Thiết kế B)
        memory_candidates = [
            target_dir / "memory_bank.npy",
            target_dir / "memory.npy",
            base_path / "memory_bank.npy",
            base_path / "memory.npy",
        ]
        memory_file: Path | None = None
        for cand in memory_candidates:
            if cand.exists():
                memory_file = cand
                break

        if memory_file is None:
            raise FileNotFoundError(
                f"Không tìm thấy file memory_bank.npy (hoặc memory.npy) tại '{target_dir}'. "
                "Vui lòng chạy 'python -m src.train' trước khi thực hiện suy luận."
            )

        # Nạp memory bank và khởi dựng runtime 1-NN index
        self.memory_bank: MemoryBank = MemoryBank.load(memory_file)

        # Khởi tạo PreprocessingConfig đồng bộ
        prep_data = self.config.get("preprocessing", {})
        self.preprocessing_config: PreprocessingConfig = (
            PreprocessingConfig.from_dict(prep_data)
            if prep_data
            else PreprocessingConfig()
        )
        self.transform = build_transform(self.preprocessing_config)

        # Khởi tạo FeatureExtractor
        self.dev: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.net: FeatureExtractor = FeatureExtractor().to(self.dev)

    @torch.inference_mode()
    def score(self, image: Image.Image) -> tuple[float, np.ndarray]:
        """Tính toán điểm số bất thường (anomaly score) và bản đồ nhiệt lỗi (heatmap).

        Args:
            image: Ảnh PIL đầu vào.

        Returns:
            tuple[float, np.ndarray]:
                - anomaly_score: 99th percentile của bản đồ nhiệt đã làm mịn.
                - smoothed_heatmap: Mảng 2D khoảng cách patch [28, 28].
        """
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.dev)
        patches, (h, w) = self.net.extract_spatial_features(x)

        # Tìm kiếm khoảng cách tới patch gần nhất trong Memory Bank
        dist, _ = self.memory_bank.kneighbors(patches.cpu().numpy())
        raw_heat = dist.reshape(h, w)

        # Làm mịn bản đồ nhiệt bằng bộ lọc Gaussian
        smoothed_heat = apply_heatmap_smoothing(raw_heat, sigma=self.smooth_sigma)

        # Anomaly score lấy theo 99th percentile khoảng cách patch
        image_score = float(np.percentile(smoothed_heat, 99))
        return image_score, smoothed_heat

    @torch.inference_mode()
    def inspect(
        self, image: Image.Image, include_overlay: bool = True
    ) -> dict[str, Any]:
        """Kiểm định bức ảnh đầy đủ: tính điểm, quyết định vận hành, tỷ lệ diện tích lỗi và overlay.

        Args:
            image: Ảnh PIL đầu vào.
            include_overlay: Có tạo chuỗi Base64 overlay bản đồ nhiệt hay không (mặc định: True).

        Returns:
            dict[str, Any]: Kết quả kiểm định giàu thông tin.
        """
        score, smoothed_heat = self.score(image)
        peak_score = float(np.max(smoothed_heat))
        area_ratio = compute_anomalous_area_ratio(
            smoothed_heat, self.pixel_threshold
        )

        decision, severity = classify_decision_and_severity(
            anomaly_score=score,
            review_threshold=self.review_threshold,
            fail_threshold=self.threshold,
            anomalous_area_ratio=area_ratio,
            peak_score=peak_score,
        )

        overlay_b64 = (
            create_heatmap_overlay_b64(
                image=image,
                heatmap=smoothed_heat,
                threshold=self.pixel_threshold,
                alpha=0.45,
            )
            if include_overlay
            else None
        )

        inspection_id = f"insp_{uuid.uuid4().hex[:12]}"

        return {
            "inspection_id": inspection_id,
            "prediction": {
                "decision": decision,
                "severity": severity,
                "anomaly_score": score,
                "review_threshold": self.review_threshold,
                "fail_threshold": self.threshold,
            },
            "localization": {
                "peak_score": peak_score,
                "anomalous_area_ratio": area_ratio,
                "pixel_threshold": self.pixel_threshold,
            },
            "model": {
                "version": self.model_version,
                "category": self.category,
            },
            # Các trường tương thích ngược với clients cũ
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
