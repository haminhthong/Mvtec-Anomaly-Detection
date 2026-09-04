"""HTTP REST API Server phát hiện lỗi ngoại quan công nghiệp (Industrial Visual Anomaly Detection).

Xây dựng trên nền FastAPI với các tính năng sản xuất:
1. Pydantic Models cho dữ liệu phản hồi API chuẩn hóa.
2. Endpoint /health giúp load balancer kiểm tra tình trạng dịch vụ không tốn chi phí nạp model nặng.
3. Endpoint /inspect nhận ảnh tải lên, phân tích score, đưa ra quyết định PASS/REVIEW/FAIL và trả về ảnh heatmap overlay mã hóa Base64.
4. Kiểm soát an toàn: giới hạn dung lượng tải lên (10MB) và phòng chống tấn công Decompression Bomb.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

# Thư mục gốc dự án và đường dẫn lưu trữ mô hình
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
MODEL_DIR: Path = PROJECT_ROOT / "models"

# Giới hạn kích thước tập tin tải lên và độ phân giải tối đa (chống Decompression Bomb)
MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024  # 10 Megabytes
MAX_IMAGE_PIXELS: int = 25_000_000  # 25 Megapixels


class HealthResponse(BaseModel):
    """Pydantic Schema cho dữ liệu kiểm tra sức khỏe hệ thống (/health)."""

    status: str = Field(..., description="Trạng thái dịch vụ: 'ok' hoặc 'degraded'")
    model_ready: bool = Field(
        ..., description="Trạng thái mô hình đã sẵn sàng suy luận hay chưa"
    )
    model_version: str = Field(..., description="Phiên bản mô hình đang hoạt động")


class InspectionResponse(BaseModel):
    """Pydantic Schema cho kết quả kiểm định lỗi ngoại quan (/inspect)."""

    anomaly_score: float = Field(
        ..., description="Điểm số bất thường tổng thể của bức ảnh"
    )
    threshold: float = Field(
        ...,
        description="Ngưỡng phát hiện lỗi đã được căn chỉnh (Calibration Threshold)",
    )
    decision: str = Field(
        ...,
        description="Quyết định vận hành: 'PASS' (Đạt), 'REVIEW' (Xem xét), 'FAIL' (Lỗi)",
    )
    heatmap_shape: list[int] = Field(
        ..., description="Kích thước ma trận bản đồ nhiệt [H, W]"
    )
    model_version: str = Field(..., description="Phiên bản mô hình thực hiện suy luận")
    overlay_b64: str | None = Field(
        None,
        description="Chuỗi mã hóa Base64 của ảnh phủ bản đồ nhiệt lỗi (Heatmap Overlay)",
    )


# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="Industrial Visual Anomaly Detection API",
    description="Hệ thống phát hiện lỗi ngoại quan theo hướng One-Class (PatchCore MVTec AD)",
    version="1.0.0",
)

# Biến toàn cục giữ instance AnomalyDetector (lazy loading)
_det = None


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health() -> HealthResponse:
    """Endpoint kiểm tra sức khỏe dịch vụ và tình trạng sẵn sàng của mô hình."""
    config_path = MODEL_DIR / "config.json"
    nn_path = MODEL_DIR / "patch_nn.joblib"
    ready = config_path.exists() and nn_path.exists()
    version = "not_trained"

    if config_path.exists():
        try:
            version = json.loads(config_path.read_text(encoding="utf-8")).get(
                "version", "unknown"
            )
        except (OSError, KeyError, json.JSONDecodeError):
            ready = False

    return HealthResponse(
        status="ok" if ready else "degraded",
        model_ready=ready,
        model_version=version,
    )


@app.post("/inspect", response_model=InspectionResponse, tags=["Inspection"])
async def inspect(
    file: Annotated[
        UploadFile, File(..., description="Tệp ảnh sản phẩm cần kiểm tra (PNG/JPG)")
    ],
    include_overlay: Annotated[
        bool, Form(description="Có bao gồm ảnh overlay Base64 trong kết quả hay không")
    ] = True,
) -> InspectionResponse:
    """Endpoint kiểm định ảnh sản phẩm và trả về kết quả phát hiện lỗi ngoại quan."""
    global _det

    # Kiểm tra kích thước file tải lên
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail="Tệp ảnh tải lên vượt quá giới hạn 10 MB."
        )

    # Đọc và xác thực tính hợp lệ của ảnh (phòng chống Decompression Bomb)
    try:
        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
        image = Image.open(io.BytesIO(content))
        image.verify()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=415, detail="Tệp tải lên không phải là ảnh định dạng hợp lệ."
        ) from exc

    # Lazy import và khởi tạo AnomalyDetector nếu chưa nạp
    try:
        if _det is None:
            from .inference import AnomalyDetector

            _det = AnomalyDetector(model_dir=MODEL_DIR)

        result = _det.inspect(image, include_overlay=include_overlay)
        return InspectionResponse(**result)
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Mô hình chưa sẵn sàng hoặc suy luận thất bại: {exc}",
        ) from exc
