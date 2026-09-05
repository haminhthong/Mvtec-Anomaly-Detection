"""Module phân tách dữ liệu và căn chỉnh ngưỡng kép độc lập (Held-Out Normal Calibration).

Đảm bảo:
1. Huấn luyện One-Class: Chỉ học manifold trên ảnh train/good.
2. Tách held-out bình thường: 80% Memory Set và 20% Calibration Set.
3. Căn chỉnh ngưỡng độc lập:
   - review_threshold = Quantile 95% của Normal Image Score
   - fail_threshold (image_threshold) = Quantile 99% của Normal Image Score
   - pixel_threshold = Quantile 99% của toàn bộ pixel patch heatmap Normal
Hoàn toàn không rò rỉ bất kỳ ảnh defect nào trong tập test trong quá trình căn chỉnh.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def split_normal_paths(
    paths: list[Path],
    calibration_fraction: float = 0.2,
    seed: int = 42,
    min_calibration_samples: int = 20,
) -> tuple[list[Path], list[Path]]:
    """Tách ảnh normal ngẫu nhiên, tái lập được cho memory và calibration.

    Args:
        paths: Danh sách đường dẫn ảnh bình thường (train/good).
        calibration_fraction: Tỷ lệ ảnh dành riêng cho calibration (mặc định 0.2).
        seed: Random seed cho việc xáo trộn và chia tách.
        min_calibration_samples: Số lượng mẫu calibration tối thiểu cần có (mặc định 20).

    Returns:
        tuple[list[Path], list[Path]]: (memory_paths, calibration_paths).

    Raises:
        ValueError: Nếu tổng số ảnh hoặc số ảnh calibration nhỏ hơn yêu cầu thống kê.
    """
    if len(paths) < min_calibration_samples:
        raise ValueError(
            f"Tổng số ảnh normal ({len(paths)}) nhỏ hơn số lượng calibration tối thiểu yêu cầu ({min_calibration_samples})."
        )
    memory, calibration = train_test_split(
        sorted(paths), test_size=calibration_fraction, random_state=seed, shuffle=True
    )
    if len(calibration) < min_calibration_samples:
        raise ValueError(
            f"Số lượng ảnh calibration ({len(calibration)}) nhỏ hơn ngưỡng yêu cầu ({min_calibration_samples}). "
            "Hãy tăng calibration_fraction hoặc bổ sung ảnh train/good để đảm bảo ước lượng quantile đáng tin cậy."
        )
    return sorted(memory), sorted(calibration)


def calibrate_thresholds(
    normal_scores: list[float],
    normal_heatmaps: list[np.ndarray],
    review_quantile: float = 0.95,
    fail_quantile: float = 0.99,
    pixel_quantile: float = 0.99,
) -> tuple[float, float, float]:
    """Căn chỉnh ngưỡng kép (Dual-threshold) và ngưỡng pixel dựa trên phân phối normal.

    Args:
        normal_scores: Danh sách điểm bất thường tổng thể của các ảnh normal calibration.
        normal_heatmaps: Danh sách các mảng 2D bản đồ nhiệt smoothed anomaly map của normal.
        review_quantile: Phân vị cho ngưỡng cảnh báo xem xét (mặc định: 0.95).
        fail_quantile: Phân vị cho ngưỡng lỗi nghiêm trọng (mặc định: 0.99).
        pixel_quantile: Phân vị trên tập hợp tất cả các pixel normal heatmap (mặc định: 0.99).

    Returns:
        tuple[float, float, float]: (review_threshold, fail_threshold, pixel_threshold).
    """
    if not normal_scores:
        raise ValueError("Danh sách normal_scores rỗng, không thể căn chỉnh threshold.")

    review_threshold = float(np.quantile(normal_scores, review_quantile))
    fail_threshold = float(np.quantile(normal_scores, fail_quantile))

    # Gom toàn bộ pixel của các heatmap normal để xác định ngưỡng lỗi cục bộ cấp pixel
    if normal_heatmaps:
        all_pixels = np.concatenate([h.ravel() for h in normal_heatmaps])
        pixel_threshold = float(np.quantile(all_pixels, pixel_quantile))
    else:
        pixel_threshold = fail_threshold

    return review_threshold, fail_threshold, pixel_threshold
