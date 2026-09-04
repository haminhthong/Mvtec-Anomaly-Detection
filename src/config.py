"""Cấu hình và kiểm soát tham số cho hệ thống kiểm tra lỗi ngoại quan MVTec AD (PatchCore).

Module này cung cấp dataclass TrainConfig để quản lý tất cả các tham số huấn luyện,
hiệu chỉnh ngưỡng (calibration), kích thước memory bank coreset và các thiết lập
xử lý bản đồ nhiệt (heatmap smoothing).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

# Các giá trị mặc định của hệ thống
DEFAULT_CATEGORY: str = "bottle"
DEFAULT_SEED: int = 42


@dataclass(frozen=True)
class TrainConfig:
    """Dataclass chứa toàn bộ tham số cấu hình cho pipeline PatchCore.

    Attributes:
        category: Tên danh mục sản phẩm cần phát hiện lỗi (mặc định: 'bottle').
        seed: Seed cho các bộ sinh số ngẫu nhiên nhằm đảm bảo tính tái lập.
        batch_size: Kích thước batch khi trích xuất đặc trưng hình ảnh.
        calibration_fraction: Tỷ lệ ảnh normal held-out dùng để căn chỉnh threshold.
        coreset_fraction: Tỷ lệ patch trích xuất để tạo coreset đại diện.
        min_coreset_size: Kích thước tối thiểu của tập patch memory bank coreset.
        max_coreset_size: Kích thước tối đa của tập patch memory bank coreset.
        smooth_sigma: Độ lệch chuẩn Sigma cho bộ lọc Gaussian Smoothing làm mịn anomaly map.
    """

    category: str = DEFAULT_CATEGORY
    seed: int = DEFAULT_SEED
    batch_size: int = 8
    calibration_fraction: float = 0.2
    threshold_quantile: float = 0.99
    coreset_fraction: float = 0.05
    min_coreset_size: int = 100
    max_coreset_size: int = 1000
    smooth_sigma: float = 1.0

    def validate(self) -> None:
        """Kiểm tra tính hợp lệ của tất cả các tham số cấu hình.

        Raises:
            ValueError: Nếu bất kỳ tham số nào nằm ngoài dải hợp lệ.
        """
        if not self.category.strip():
            raise ValueError("Tên danh mục (category) không được để trống.")
        if self.batch_size <= 0:
            raise ValueError("Kích thước batch (batch_size) phải lớn hơn 0.")
        if not 0 < self.calibration_fraction < 0.5:
            raise ValueError("calibration_fraction phải thuộc khoảng (0, 0.5).")
        if not 0.5 <= self.threshold_quantile < 1.0:
            raise ValueError("threshold_quantile phải thuộc khoảng [0.5, 1.0).")
        if not 0 < self.coreset_fraction <= 1:
            raise ValueError("coreset_fraction phải thuộc khoảng (0, 1].")
        if not 1 <= self.min_coreset_size <= self.max_coreset_size:
            raise ValueError(
                "Kích thước coreset tối thiểu/tối đa không hợp lệ (min phải <= max và min >= 1)."
            )
        if self.smooth_sigma < 0:
            raise ValueError("smooth_sigma không được âm.")


def parse_args() -> TrainConfig:
    """Đọc tham số dòng lệnh CLI và trả về cấu hình TrainConfig đã kiểm tra hợp lệ.

    Returns:
        TrainConfig: Cấu hình huấn luyện hoàn chỉnh.
    """
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình phát hiện lỗi ngoại quan PatchCore cho MVTec AD"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=DEFAULT_CATEGORY,
        help="Tên danh mục sản phẩm trong MVTec AD",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Giá trị seed ngẫu nhiên"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Kích thước batch cho DataLoader"
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.2,
        help="Tỷ lệ ảnh normal giữ riêng cho calibration",
    )
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.99,
        help="Phân vị score normal dùng làm threshold",
    )
    parser.add_argument(
        "--coreset-fraction",
        type=float,
        default=0.05,
        help="Tỷ lệ mẫu patch giữ lại qua coreset",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=1.0,
        help="Độ mịn Gaussian smoothing cho anomaly map",
    )

    args = parser.parse_args()
    config = TrainConfig(
        category=args.category,
        seed=args.seed,
        batch_size=args.batch_size,
        calibration_fraction=args.calibration_fraction,
        threshold_quantile=args.threshold_quantile,
        coreset_fraction=args.coreset_fraction,
        smooth_sigma=args.smooth_sigma,
    )
    config.validate()
    return config
