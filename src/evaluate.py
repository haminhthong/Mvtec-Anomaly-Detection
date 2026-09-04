"""Module đánh giá hiệu năng (Evaluation Pipeline) hệ thống phát hiện lỗi ngoại quan.

Tính toán các chỉ số đánh giá tiêu chuẩn trong nghiên cứu Anomaly Detection (PatchCore/MVTec AD):
1. Image-level AUROC & Average Precision (AP): Khả năng phân biệt ảnh lỗi / không lỗi.
2. Pixel-level AUROC & Average Precision (AP): Độ chính xác của bản đồ nhiệt định vị vị trí lỗi.
3. AUPRO@0.3 (Area Under Per-Region Overlap): Chỉ số đánh giá định vị lỗi không bị thiên vị bởi kích thước vùng lỗi lớn/nhỏ.
4. Accuracy tại ngưỡng threshold đã chọn.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import label
from sklearn.metrics import average_precision_score, roc_auc_score

from .utils import save_json

# Cấu hình UTF-8 cho stdout trên Windows console để tránh UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def compute_aupro(masks: np.ndarray, maps: np.ndarray, max_fpr: float = 0.3) -> float:
    """Tính toán chỉ số AUPRO (Area Under Per-Region Overlap) trong miền FPR [0, max_fpr].

    AUPRO đánh giá độ phủ của bản đồ nhiệt dự đoán trên từng thành phần liên thông (connected component)
    của ground-truth mask. Khác với Pixel AUROC, AUPRO coi các vùng lỗi có vai trò ngang nhau bất kể
    diện tích lớn hay nhỏ.

    Args:
        masks: Mảng 3D boolean ground-truth masks [N, H, W] (True nếu là pixel lỗi).
        maps: Mảng 3D float32 bản đồ nhiệt dư đoán [N, H, W].
        max_fpr: Giới hạn Tỷ lệ Dương tính Giả (False Positive Rate) tối đa (mặc định: 0.3).

    Returns:
        float: Giá trị chỉ số AUPRO chuẩn hóa trong khoảng [0, 1].
    """
    normal = ~masks
    thresholds = np.quantile(maps, np.linspace(0, 1, 200))
    points: list[tuple[float, float]] = []

    for threshold in thresholds:
        predicted = maps >= threshold
        fpr = float(predicted[normal].mean()) if normal.any() else 0.0
        if fpr > max_fpr:
            continue

        overlaps: list[float] = []
        for mask, pred in zip(masks, predicted, strict=True):
            components, count = label(mask)
            for component_id in range(1, count + 1):
                region = components == component_id
                overlaps.append(float(pred[region].mean()))

        points.append((fpr, float(np.mean(overlaps)) if overlaps else 0.0))

    if len(points) < 2:
        return 0.0

    # Giữ lại giá trị PRO tốt nhất cho từng FPR
    best_by_fpr: dict[float, float] = {}
    for fpr, pro in points:
        best_by_fpr[fpr] = max(pro, best_by_fpr.get(fpr, 0.0))

    x = np.array(sorted(best_by_fpr))
    y = np.array([best_by_fpr[value] for value in x])

    # Nội suy tích phân hình thang đến max_fpr
    if x[-1] < max_fpr:
        x = np.append(x, max_fpr)
        y = np.append(y, y[-1])

    return float(np.trapezoid(y, x) / max_fpr)


def main() -> None:
    """Hàm chạy đánh giá toàn bộ tập test và lưu báo cáo vào reports/test_metrics.json."""
    from .data import find_category_root
    from .inference import AnomalyDetector

    root = find_category_root()
    det = AnomalyDetector()

    ys: list[int] = []
    scores: list[float] = []
    masks: list[np.ndarray] = []
    maps: list[np.ndarray] = []

    print(
        f"[EVALUATION] Bắt đầu đánh giá mô hình PatchCore cho danh mục '{det.config['category']}'..."
    )

    test_dir = root / "test"
    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir():
            continue

        is_defective = 0 if defect_dir.name == "good" else 1

        for p in defect_dir.glob("*.png"):
            with Image.open(p) as img:
                s, heat = det.score(img)

            ys.append(is_defective)
            scores.append(s)

            # Đọc ground-truth mask nếu là ảnh có lỗi
            if is_defective:
                mask_path = (
                    root / "ground_truth" / defect_dir.name / f"{p.stem}_mask.png"
                )
                with Image.open(mask_path) as m_img:
                    mask = (
                        np.asarray(
                            m_img.convert("L").resize(
                                (224, 224), Image.Resampling.NEAREST
                            )
                        )
                        > 0
                    )
            else:
                mask = np.zeros((224, 224), dtype=bool)

            # Resize anomaly map về 224x224
            anomaly_map = np.asarray(
                Image.fromarray(heat.astype(np.float32)).resize(
                    (224, 224), Image.Resampling.BILINEAR
                )
            )

            masks.append(mask)
            maps.append(anomaly_map)

    masks_array = np.asarray(masks)
    maps_array = np.asarray(maps)
    threshold = float(det.config["threshold"])
    pred = np.asarray(scores) >= threshold

    metrics: dict[str, Any] = {
        "category": det.config["category"],
        "model_version": det.model_version,
        "image_auroc": float(roc_auc_score(ys, scores)),
        "image_average_precision": float(average_precision_score(ys, scores)),
        "pixel_auroc": float(roc_auc_score(masks_array.ravel(), maps_array.ravel())),
        "pixel_average_precision": float(
            average_precision_score(masks_array.ravel(), maps_array.ravel())
        ),
        "aupro_0.3": compute_aupro(masks_array, maps_array),
        "threshold": threshold,
        "accuracy_at_threshold": float((pred == np.asarray(ys)).mean()),
    }

    save_json("reports/test_metrics.json", metrics)

    print("\n=================== KẾT QUẢ ĐÁNH GIÁ METRICS ===================")
    print(f"  - Category                    : {metrics['category']}")
    print(f"  - Image-level AUROC           : {metrics['image_auroc']:.4f}")
    print(
        f"  - Image-level Average Precision: {metrics['image_average_precision']:.4f}"
    )
    print(f"  - Pixel-level AUROC           : {metrics['pixel_auroc']:.4f}")
    print(
        f"  - Pixel-level Average Precision: {metrics['pixel_average_precision']:.4f}"
    )
    print(f"  - AUPRO (max_fpr=0.3)         : {metrics['aupro_0.3']:.4f}")
    print(f"  - Threshold (Ngưỡng)          : {metrics['threshold']:.4f}")
    print(f"  - Accuracy tại ngưỡng         : {metrics['accuracy_at_threshold']:.4f}")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
