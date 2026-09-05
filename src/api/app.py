"""HTTP REST API Server phát hiện lỗi ngoại quan công nghiệp (FastAPI Enterprise).

Hỗ trợ:
1. Endpoints kiểm tra sức khỏe phân tầng: /health, /health/live (Liveness), /health/ready (Readiness).
2. Quản lý mô hình đa danh mục qua ModelRegistry: GET /models, GET /models/{category}.
3. Endpoint suy luận /inspect trả về cấu trúc giàu thông tin (Prediction, Localization, Severity, Base64).
4. Cơ chế bảo mật và tối ưu: chống Decompression Bomb, giới hạn tải lên 10MB, lazy caching.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Annotated, Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from ..model.registry import ModelRegistry
from .schemas import HealthResponse, InspectionResponse, ReadinessResponse

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
MODEL_DIR: Path = PROJECT_ROOT / "models"

MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024  # 10 MB
MAX_IMAGE_PIXELS: int = 25_000_000  # 25 MP

app = FastAPI(
    title="Industrial Visual Anomaly Detection API",
    description="Hệ thống phát hiện lỗi ngoại quan theo hướng One-Class (PatchCore-style MVTec AD)",
    version="2.0.0",
)

registry = ModelRegistry(base_dir=MODEL_DIR)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health() -> HealthResponse:
    """Endpoint kiểm tra sức khỏe cơ bản giữ tương thích ngược."""
    categories = registry.list_categories()
    ready = len(categories) > 0
    version = registry.version(categories[0]) if categories else "not_trained"
    return HealthResponse(
        status="ok" if ready else "degraded",
        model_ready=ready,
        model_version=version,
        categories=categories,
    )


@app.get("/health/live", tags=["Monitoring"])
def health_live() -> dict[str, str]:
    """Liveness probe kiểm tra process API server đang hoạt động."""
    return {"status": "alive"}


@app.get("/health/ready", response_model=ReadinessResponse, tags=["Monitoring"])
def health_ready() -> ReadinessResponse:
    """Readiness probe kiểm tra model artifacts và runtime NN search index đã sẵn sàng."""
    categories = registry.list_categories()
    if not categories:
        raise HTTPException(
            status_code=503,
            detail="Chưa có artifact mô hình nào sẵn sàng trong hệ thống.",
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Kiểm tra nạp thử mô hình đầu tiên
    try:
        registry.get_detector(categories[0])
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Mô hình không thể khởi tạo: {exc}",
        ) from exc

    return ReadinessResponse(
        ready=True,
        categories=categories,
        active_device=device,
    )


@app.get("/models", tags=["Model Registry"])
def list_models() -> dict[str, Any]:
    """Liệt kê danh sách tất cả các danh mục sản phẩm đã được huấn luyện."""
    categories = registry.list_categories()
    return {
        "total_categories": len(categories),
        "categories": categories,
    }


@app.get("/models/{category}", tags=["Model Registry"])
def get_model_details(category: str) -> dict[str, Any]:
    """Xem thông tin chi tiết về cấu hình và ngưỡng của một danh mục sản phẩm."""
    try:
        return registry.get_metadata(category)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy cấu hình cho danh mục '{category}'.",
        ) from exc


@app.post("/inspect", response_model=InspectionResponse, tags=["Inspection"])
async def inspect(
    file: Annotated[
        UploadFile, File(..., description="Tệp ảnh sản phẩm cần kiểm tra (PNG/JPG)")
    ],
    category: Annotated[
        str | None,
        Query(description="Danh mục sản phẩm (ví dụ: 'bottle'). Mặc định tự động chọn"),
    ] = None,
    include_overlay: Annotated[
        bool, Form(description="Có bao gồm ảnh overlay Base64 trong kết quả hay không")
    ] = True,
) -> InspectionResponse:
    """Endpoint kiểm định ảnh sản phẩm, ra quyết định phân loại và khoanh vùng khuyết tật."""
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail="Tệp ảnh tải lên vượt quá giới hạn 10 MB."
        )

    try:
        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
        image = Image.open(io.BytesIO(content))
        image.verify()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=415, detail="Tệp tải lên không phải là ảnh định dạng hợp lệ."
        ) from exc

    # Xác định category
    target_category = category
    if not target_category:
        available = registry.list_categories()
        target_category = available[0] if available else "bottle"

    try:
        detector = registry.get_detector(target_category)
        result = detector.inspect(image, include_overlay=include_overlay)
        return InspectionResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Mô hình cho danh mục '{target_category}' chưa được huấn luyện: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi trong quá trình suy luận: {exc}",
        ) from exc
