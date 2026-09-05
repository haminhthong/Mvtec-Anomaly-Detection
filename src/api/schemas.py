"""Pydantic Schemas cho API REST phát hiện lỗi ngoại quan (Industrial Anomaly Detection).

Cung cấp data contracts chuẩn mực cho kiểm tra sức khỏe (/health/live, /health/ready),
quản lý mô hình (/models) và kết quả kiểm định chi tiết (/inspect).
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field, model_validator


class HealthResponse(BaseModel):
    """Pydantic Schema cho endpoint kiểm tra sức khỏe tổng thể (/health)."""

    status: str = Field(..., description="Trạng thái dịch vụ: 'ok' hoặc 'degraded'")
    model_ready: bool = Field(
        ..., description="Trạng thái mô hình đã sẵn sàng suy luận hay chưa"
    )
    model_version: str = Field(..., description="Phiên bản mô hình đang hoạt động")
    categories: list[str] = Field(
        default_factory=list, description="Danh sách các danh mục sản phẩm sẵn sàng"
    )


class ReadinessResponse(BaseModel):
    """Pydantic Schema cho endpoint kiểm tra tính sẵn sàng (/health/ready)."""

    ready: bool = Field(..., description="Mô hình và runtime index đã sẵn sàng phục vụ")
    categories: list[str] = Field(
        ..., description="Các danh mục đã tải và khởi tạo chỉ mục 1-NN"
    )
    active_device: str = Field(..., description="Thiết bị tính toán ('cuda' hoặc 'cpu')")


class PredictionResult(BaseModel):
    """Pydantic Schema cho kết quả phân loại lỗi và cấp độ nghiêm trọng."""

    decision: str = Field(
        ...,
        description="Quyết định vận hành: 'PASS' (Đạt), 'REVIEW' (Xem xét), 'FAIL' (Lỗi)",
    )
    severity: str = Field(
        ...,
        description="Mức độ nghiêm trọng: 'PASS', 'REVIEW', 'FAIL_MINOR', 'FAIL_MAJOR'",
    )
    anomaly_score: float = Field(
        ..., description="Điểm số bất thường tổng thể (99th percentile)"
    )
    review_threshold: float = Field(
        ..., description="Ngưỡng cảnh báo rà soát (P95 normal calibration)"
    )
    fail_threshold: float = Field(
        ..., description="Ngưỡng phát hiện lỗi (P99 normal calibration)"
    )


class LocalizationResult(BaseModel):
    """Pydantic Schema cho thông tin định vị khuyết tật trên ảnh."""

    peak_score: float = Field(
        ..., description="Điểm số bất thường cao nhất trên bản đồ nhiệt"
    )
    anomalous_area_ratio: float = Field(
        ..., description="Tỷ lệ diện tích bề mặt nghi ngờ có lỗi [0.0, 1.0]"
    )
    pixel_threshold: float = Field(
        ..., description="Ngưỡng phát hiện lỗi cấp pixel (P99 normal heatmap pixels)"
    )


class ModelInfo(BaseModel):
    """Pydantic Schema cho thông tin phiên bản và danh mục mô hình."""

    version: str = Field(..., description="Phiên bản mô hình")
    category: str = Field(..., description="Danh mục sản phẩm (ví dụ: 'bottle')")


class InspectionResponse(BaseModel):
    """Pydantic Schema cho phản hồi kiểm định ảnh sản phẩm (/inspect)."""

    inspection_id: str = Field(
        default_factory=lambda: f"insp_{uuid.uuid4().hex[:12]}",
        description="Mã định danh duy nhất của lượt kiểm định",
    )
    prediction: PredictionResult | None = Field(
        default=None, description="Chi tiết quyết định vận hành"
    )
    localization: LocalizationResult | None = Field(
        default=None, description="Chi tiết định vị và tỷ lệ diện tích khuyết tật"
    )
    model: ModelInfo | None = Field(
        default=None, description="Thông tin mô hình thực hiện suy luận"
    )
    overlay_b64: str | None = Field(
        None,
        description="Chuỗi mã hóa Base64 của ảnh phủ bản đồ nhiệt lỗi (Heatmap Overlay)",
    )

    # Các trường phẳng giữ tương thích ngược hoàn toàn
    anomaly_score: float = Field(
        ..., description="Điểm số bất thường tổng thể (tương thích ngược)"
    )
    threshold: float = Field(
        ..., description="Ngưỡng phát hiện lỗi (tương thích ngược)"
    )
    decision: str = Field(
        ..., description="Quyết định vận hành (tương thích ngược)"
    )
    heatmap_shape: list[int] = Field(
        default_factory=lambda: [28, 28],
        description="Kích thước bản đồ nhiệt [H, W]",
    )
    model_version: str = Field(
        default="unknown", description="Phiên bản mô hình (tương thích ngược)"
    )

    @model_validator(mode="after")
    def populate_nested_fields(self) -> InspectionResponse:
        """Tự động đồng bộ các object lồng nhau nếu chỉ truyền các trường phẳng."""
        if self.prediction is None:
            self.prediction = PredictionResult(
                decision=self.decision,
                severity=self.decision if self.decision in {"PASS", "REVIEW"} else "FAIL_MAJOR",
                anomaly_score=self.anomaly_score,
                review_threshold=0.8 * self.threshold,
                fail_threshold=self.threshold,
            )
        if self.localization is None:
            self.localization = LocalizationResult(
                peak_score=self.anomaly_score,
                anomalous_area_ratio=0.0 if self.decision == "PASS" else 0.05,
                pixel_threshold=self.threshold,
            )
        if self.model is None:
            self.model = ModelInfo(
                version=self.model_version,
                category="bottle",
            )
        return self
