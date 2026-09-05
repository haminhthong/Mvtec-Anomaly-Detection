"""Unit tests kiểm tra công cụ ra quyết định và tính toán diện tích khuyết tật."""

from __future__ import annotations

import numpy as np
from src.inference.decision import classify_decision_and_severity
from src.inference.localization import compute_anomalous_area_ratio


def test_classify_decision_and_severity_cases() -> None:
    """Kiểm tra đầy đủ các kịch bản PASS, REVIEW, FAIL_MINOR, FAIL_MAJOR."""
    review_th = 3.0
    fail_th = 4.0

    # 1. Điểm an toàn < review_th -> PASS
    dec, sev = classify_decision_and_severity(
        anomaly_score=2.5, review_threshold=review_th, fail_threshold=fail_th
    )
    assert dec == "PASS"
    assert sev == "PASS"

    # 2. Vùng đệm rà soát -> REVIEW
    dec, sev = classify_decision_and_severity(
        anomaly_score=3.5, review_threshold=review_th, fail_threshold=fail_th
    )
    assert dec == "REVIEW"
    assert sev == "REVIEW"

    # 3. Vượt ngưỡng lỗi nhưng diện tích nhỏ -> FAIL_MINOR
    dec, sev = classify_decision_and_severity(
        anomaly_score=4.2,
        review_threshold=review_th,
        fail_threshold=fail_th,
        anomalous_area_ratio=0.01,
        peak_score=4.5,
    )
    assert dec == "FAIL"
    assert sev == "FAIL_MINOR"

    # 4. Vượt ngưỡng lỗi và diện tích lớn (>= 5%) -> FAIL_MAJOR
    dec, sev = classify_decision_and_severity(
        anomaly_score=4.5,
        review_threshold=review_th,
        fail_threshold=fail_th,
        anomalous_area_ratio=0.08,
        peak_score=5.0,
    )
    assert dec == "FAIL"
    assert sev == "FAIL_MAJOR"

    # 5. Vượt ngưỡng lỗi và peak_score cực cao (>= 1.5 * fail_th) -> FAIL_MAJOR
    dec, sev = classify_decision_and_severity(
        anomaly_score=4.1,
        review_threshold=review_th,
        fail_threshold=fail_th,
        anomalous_area_ratio=0.005,
        peak_score=6.5,
    )
    assert dec == "FAIL"
    assert sev == "FAIL_MAJOR"


def test_compute_anomalous_area_ratio() -> None:
    """Kiểm tra tính tỷ lệ diện tích pixel vượt ngưỡng trên heatmap."""
    heat = np.zeros((10, 10), dtype=np.float32)
    heat[:2, :5] = 5.0  # 10 pixels trên tổng 100 pixels = 0.10 (10%)
    ratio = compute_anomalous_area_ratio(heat, pixel_threshold=4.0)
    assert abs(ratio - 0.10) < 1e-5
