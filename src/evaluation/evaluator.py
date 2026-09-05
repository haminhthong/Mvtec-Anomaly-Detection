"""Module thực hiện đánh giá toàn diện mô hình trên tập test MVTec AD.

Đảm bảo:
1. P0 Fix: Đồng bộ 100% giữa Artifact Config và Dataset Category (không hardcode).
2. Quy tắc Report-Only: Tuyệt đối không chọn lại ngưỡng trên tập test; sử dụng nguyên
   vẹn threshold đã căn chỉnh từ held-out normal.
3. Xuất kết quả 3 tầng metrics ra console và lưu vào reports/test_metrics.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..data.dataset import find_category_root
from ..inference.detector import AnomalyDetector
from .metrics import calculate_3tier_metrics


def evaluate_category(
    category: str | None = None,
    model_dir: str | Path = "models",
    output_report: str | Path = "reports/test_metrics.json",
) -> dict[str, Any]:
    """Đánh giá mô hình PatchCore trên toàn bộ tập test của danh mục tương ứng.

    Args:
        category: Tên danh mục (nếu None sẽ đọc từ config của detector).
        model_dir: Đường dẫn thư mục chứa model artifacts.
        output_report: Đường dẫn lưu trữ báo cáo JSON.

    Returns:
        dict[str, Any]: Báo cáo metrics 3 tầng.
    """
    # Khởi tạo detector trước để đọc chính xác category từ model artifact (Fix P0)
    det = AnomalyDetector(model_dir=model_dir, category=category)
    resolved_category = det.category
    root = find_category_root(category=resolved_category)

    ys: list[int] = []
    scores: list[float] = []
    masks: list[np.ndarray] = []
    maps: list[np.ndarray] = []

    print(
        f"\n[EVALUATION] Bắt đầu đánh giá mô hình PatchCore cho danh mục '{resolved_category}'..."
    )
    print(f"  - Artifact version: {det.model_version}")
    print(f"  - Calibrated Image Threshold: {det.threshold:.4f}")
    print(f"  - Calibrated Review Threshold: {det.review_threshold:.4f}")
    print(f"  - Calibrated Pixel Threshold: {det.pixel_threshold:.4f}")

    test_dir = root / "test"
    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir():
            continue

        is_defective = 0 if defect_dir.name == "good" else 1

        for p in sorted(defect_dir.glob("*.png")):
            with Image.open(p) as img:
                s, heat = det.score(img)

            ys.append(is_defective)
            scores.append(s)

            # Đọc ground-truth mask nếu là ảnh có lỗi
            if is_defective:
                mask_path = (
                    root / "ground_truth" / defect_dir.name / f"{p.stem}_mask.png"
                )
                if mask_path.exists():
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

    # Tính toán 3 tầng chỉ số
    metrics_result = calculate_3tier_metrics(
        y_true=ys,
        scores=scores,
        masks=masks_array,
        maps=maps_array,
        threshold=det.threshold,
    )

    metrics_result["category"] = resolved_category
    metrics_result["model_version"] = det.model_version

    # Lưu báo cáo JSON
    rep_path = Path(output_report)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(
        json.dumps(metrics_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    op = metrics_result["operational_decision"]
    det_tier = metrics_result["detection"]
    loc_tier = metrics_result["localization"]
    cm = op["confusion_matrix"]

    print("\n" + "=" * 65)
    print(f"      KẾT QUẢ ĐÁNH GIÁ 3 TẦNG: {resolved_category.upper()}")
    print("=" * 65)
    print(" [TIER 1: DETECTION (Định danh lỗi toàn ảnh)]")
    print(f"  - Image AUROC                : {det_tier['image_auroc']:.4f}")
    print(f"  - Image Average Precision    : {det_tier['image_average_precision']:.4f}")
    print("\n [TIER 2: LOCALIZATION (Khoanh vùng khuyết tật pixel)]")
    print(f"  - Pixel AUROC                : {loc_tier['pixel_auroc']:.4f}")
    print(f"  - Pixel Average Precision    : {loc_tier['pixel_average_precision']:.4f}")
    print(f"  - AUPRO (max_fpr=0.3)        : {loc_tier['aupro_0.3']:.4f}")
    print("\n [TIER 3: OPERATIONAL QC (Quyết định vận hành tại ngưỡng Calibrated)]")
    print(f"  - Calibrated Threshold       : {op['threshold']:.4f}")
    print(f"  - Accuracy                   : {op['accuracy']:.4f}")
    print(f"  - Precision                  : {op['precision']:.4f}")
    print(f"  - Defect Recall (TPR)        : {op['defect_recall']:.4f} (Độ nhạy bắt lỗi)")
    print(f"  - Specificity (TNR)          : {op['specificity']:.4f} (Độ đặc hiệu)")
    print(f"  - F1 Score                   : {op['f1_score']:.4f}")
    print(f"  - False Reject Rate (FRR)    : {op['false_reject_rate']:.4f} (Tỷ lệ loại nhầm hàng tốt)")
    print(f"  - False Accept Rate (FAR)    : {op['false_accept_rate']:.4f} (Tỷ lệ lọt sản phẩm lỗi)")
    print("\n [CONFUSION MATRIX]")
    print(f"                 Pred PASS    Pred FAIL")
    print(f"  Normal (Good):   TN={cm['tn']:<4}      FP={cm['fp']:<4}  (Total: {cm['tn']+cm['fp']})")
    print(f"  Defect:          FN={cm['fn']:<4}      TP={cm['tp']:<4}  (Total: {cm['tp']+cm['fn']})")
    print("=" * 65 + "\n")

    return metrics_result
