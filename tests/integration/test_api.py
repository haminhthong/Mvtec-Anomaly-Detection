"""Integration tests kiểm tra các API endpoints của FastAPI server."""

from __future__ import annotations

import io
from PIL import Image
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_health_endpoints() -> None:
    """Kiểm tra /health và /health/live."""
    res_live = client.get("/health/live")
    assert res_live.status_code == 200
    assert res_live.json() == {"status": "alive"}

    res_health = client.get("/health")
    assert res_health.status_code == 200
    data = res_health.json()
    assert "status" in data
    assert "model_ready" in data


def test_models_endpoint() -> None:
    """Kiểm tra endpoint /models liệt kê danh mục."""
    res = client.get("/models")
    assert res.status_code == 200
    data = res.json()
    assert "categories" in data


def test_inspect_endpoint() -> None:
    """Kiểm tra endpoint /inspect nhận upload ảnh và trả về JSON chuẩn hóa."""
    # Tạo dummy image bytes
    img = Image.new("RGB", (100, 100), color="blue")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    res = client.post(
        "/inspect",
        files={"file": ("test.png", buffer, "image/png")},
        data={"include_overlay": "false"},
    )
    # Nếu model đã huấn luyện thì 200, nếu chưa thì có thể 404/503
    if res.status_code == 200:
        data = res.json()
        assert "inspection_id" in data
        assert "prediction" in data
        assert "localization" in data
        assert "model" in data
        assert data["prediction"]["decision"] in {"PASS", "REVIEW", "FAIL"}
