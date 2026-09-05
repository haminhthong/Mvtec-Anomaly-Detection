"""Module công cụ ra quyết định vận hành QC (Operational Decision Engine).

Thực hiện đánh giá:
- PASS: anomaly_score < review_threshold (Sản phẩm đạt chuẩn)
- REVIEW: review_threshold <= anomaly_score < fail_threshold (Cảnh báo vùng đệm rà soát QC)
- FAIL: anomaly_score >= fail_threshold (Khuyết tật vượt ngưỡng lỗi)

Xác định mức độ nghiêm trọng (Severity):
- PASS: Đạt
- REVIEW: Xem xét
- FAIL_MINOR: Lỗi nhỏ (diện tích bất thường thấp và điểm không quá cao)
- FAIL_MAJOR: Lỗi nghiêm trọng (diện tích lỗi rộng >= 5% hoặc peak_score >= 1.5 * fail_threshold)
"""

from __future__ import annotations


def classify_decision_and_severity(
    anomaly_score: float,
    review_threshold: float,
    fail_threshold: float,
    anomalous_area_ratio: float = 0.0,
    peak_score: float = 0.0,
) -> tuple[str, str]:
    """Phân loại quyết định vận hành và cấp độ nghiêm trọng của mẫu kiểm định.

    Args:
        anomaly_score: Điểm bất thường tổng thể của bức ảnh.
        review_threshold: Ngưỡng rà soát (P95 normal calibration).
        fail_threshold: Ngưỡng lỗi (P99 normal calibration).
        anomalous_area_ratio: Tỷ lệ diện tích bề mặt nghi ngờ lỗi.
        peak_score: Giá trị khoảng cách bất thường cao nhất trên bản đồ nhiệt.

    Returns:
        tuple[str, str]: (decision, severity).
            decision in {"PASS", "REVIEW", "FAIL"}
            severity in {"PASS", "REVIEW", "FAIL_MINOR", "FAIL_MAJOR"}
    """
    if anomaly_score < review_threshold:
        return "PASS", "PASS"
    elif anomaly_score < fail_threshold:
        return "REVIEW", "REVIEW"
    else:
        # Quyết định là FAIL, đánh giá mức độ nghiêm trọng
        is_major = (anomalous_area_ratio >= 0.05) or (
            peak_score >= (1.5 * fail_threshold)
        )
        severity = "FAIL_MAJOR" if is_major else "FAIL_MINOR"
        return "FAIL", severity
