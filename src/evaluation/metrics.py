"""Module tính toán bộ chỉ số đánh giá 3 tầng (3-Tier Evaluation Metrics).

Cung cấp đánh giá toàn diện chuẩn nghiên cứu và vận hành công nghiệp:
1. Tier 1: Detection (Image-level AUROC & Average Precision)
2. Tier 2: Localization (Pixel-level AUROC, Average Precision & AUPRO@0.3)
3. Tier 3: Operational Decision (Accuracy, Precision, Defect Recall, Specificity, F1, FRR, FAR & Confusion Matrix)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .aupro import compute_aupro


def calculate_3tier_metrics(
    y_true: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
    masks: np.ndarray,
    maps: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Tính toán bộ chỉ số đánh giá 3 tầng hoàn chỉnh.

    Args:
        y_true: Nhãn thực tế cấp ảnh (0: normal/good, 1: defective).
        scores: Điểm bất thường dự đoán cho từng ảnh.
        masks: Mảng 3D boolean ground-truth masks [N, H, W].
        maps: Mảng 3D float32 anomaly heatmaps dự đoán [N, H, W].
        threshold: Ngưỡng phát hiện lỗi đã được căn chỉnh trên held-out normal.

    Returns:
        dict[str, Any]: Dictionary có cấu trúc 3 tầng kèm confusion matrix.
    """
    y_arr = np.asarray(y_true, dtype=int)
    scores_arr = np.asarray(scores, dtype=np.float32)
    y_pred = (scores_arr >= threshold).astype(int)

    # Ma trận nhầm lẫn (Confusion Matrix)
    # y=1 là defect, y=0 là normal
    tp = int(np.sum((y_pred == 1) & (y_arr == 1)))
    fp = int(np.sum((y_pred == 1) & (y_arr == 0)))
    tn = int(np.sum((y_pred == 0) & (y_arr == 0)))
    fn = int(np.sum((y_pred == 0) & (y_arr == 1)))

    total_defect = tp + fn
    total_normal = tn + fp
    total_samples = len(y_arr)

    # Các chỉ số vận hành QC
    accuracy = float((tp + tn) / total_samples) if total_samples > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    defect_recall = float(tp / total_defect) if total_defect > 0 else 0.0
    specificity = float(tn / total_normal) if total_normal > 0 else 0.0
    f1_score = (
        float(2 * precision * defect_recall / (precision + defect_recall))
        if (precision + defect_recall) > 0
        else 0.0
    )

    # False Reject Rate (FRR): Sản phẩm tốt nhưng bị báo lỗi (gây lãng phí phế phẩm)
    false_reject_rate = float(fp / total_normal) if total_normal > 0 else 0.0
    # False Accept Rate (FAR): Sản phẩm lỗi nhưng bị lọt qua thành PASS (rủi ro nghiêm trọng cho khách hàng)
    false_accept_rate = float(fn / total_defect) if total_defect > 0 else 0.0

    # Tier 1: Detection
    image_auroc = float(roc_auc_score(y_arr, scores_arr))
    image_ap = float(average_precision_score(y_arr, scores_arr))

    # Tier 2: Localization
    pixel_auroc = float(roc_auc_score(masks.ravel(), maps.ravel()))
    pixel_ap = float(average_precision_score(masks.ravel(), maps.ravel()))
    aupro_val = compute_aupro(masks, maps)

    return {
        "detection": {
            "image_auroc": image_auroc,
            "image_average_precision": image_ap,
        },
        "localization": {
            "pixel_auroc": pixel_auroc,
            "pixel_average_precision": pixel_ap,
            "aupro_0.3": aupro_val,
        },
        "operational_decision": {
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "defect_recall": defect_recall,
            "specificity": specificity,
            "f1_score": f1_score,
            "false_reject_rate": false_reject_rate,
            "false_accept_rate": false_accept_rate,
            "confusion_matrix": {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            },
        },
        # Các trường cấp cao (flat) để giữ tương thích ngược
        "image_auroc": image_auroc,
        "image_average_precision": image_ap,
        "pixel_auroc": pixel_auroc,
        "pixel_average_precision": pixel_ap,
        "aupro_0.3": aupro_val,
        "threshold": threshold,
        "accuracy_at_threshold": accuracy,
        "defect_recall": defect_recall,
        "false_reject_rate": false_reject_rate,
        "false_accept_rate": false_accept_rate,
    }
