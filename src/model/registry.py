"""Module ModelRegistry quản lý mô hình đa danh mục (Multi-Category Support).

Hỗ trợ cấu trúc lưu trữ:
models/
├── bottle/
│   ├── memory_bank.npy
│   └── config.json
├── cable/
│   ├── memory_bank.npy
│   └── config.json
└── config.json (hoặc fallback đơn danh mục)

Cung cấp API tìm kiếm, nạp mô hình theo danh mục sản phẩm và lazy caching.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..inference.detector import AnomalyDetector


class ModelRegistry:
    """Registry quản lý, phát hiện và nạp mô hình AnomalyDetector cho từng danh mục.

    Attributes:
        base_dir: Thư mục gốc chứa các artifacts mô hình.
        _cached_detectors: Bộ nhớ đệm lưu các instance detector đã nạp.
    """

    def __init__(self, base_dir: str | Path = "models") -> None:
        self.base_dir: Path = Path(base_dir)
        self._cached_detectors: dict[str, AnomalyDetector] = {}

    def list_categories(self) -> list[str]:
        """Liệt kê danh sách tất cả các danh mục sản phẩm có artifact sẵn sàng."""
        if not self.base_dir.exists():
            return []

        categories: set[str] = set()

        # Kiểm tra các thư mục con models/{category}/
        for p in self.base_dir.iterdir():
            if p.is_dir():
                cfg = p / "config.json"
                mem = p / "memory_bank.npy"
                legacy_mem = p / "memory.npy"
                if cfg.exists() and (mem.exists() or legacy_mem.exists()):
                    categories.add(p.name)

        # Kiểm tra mô hình tại root models/
        root_cfg = self.base_dir / "config.json"
        root_mem = self.base_dir / "memory_bank.npy"
        legacy_root_mem = self.base_dir / "memory.npy"
        if root_cfg.exists() and (root_mem.exists() or legacy_root_mem.exists()):
            try:
                data = json.loads(root_cfg.read_text(encoding="utf-8"))
                cat = data.get("category", "bottle")
                categories.add(cat)
            except (json.JSONDecodeError, OSError):
                pass

        return sorted(categories)

    def resolve_category_dir(self, category: str = "bottle") -> Path:
        """Xác định đường dẫn thư mục chứa artifact cho danh mục tương ứng."""
        sub_dir = self.base_dir / category
        if sub_dir.exists() and (sub_dir / "config.json").exists():
            return sub_dir

        # Fallback về thư mục root models/ nếu category khớp hoặc thư mục con chưa tạo
        root_cfg = self.base_dir / "config.json"
        if root_cfg.exists():
            return self.base_dir

        raise FileNotFoundError(
            f"Không tìm thấy artifacts mô hình cho danh mục '{category}' tại '{self.base_dir}'."
        )

    def get_metadata(self, category: str = "bottle") -> dict[str, Any]:
        """Đọc và trả về metadata config.json của danh mục."""
        cat_dir = self.resolve_category_dir(category)
        cfg_path = cat_dir / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file config tại '{cfg_path}'.")
        return json.loads(cfg_path.read_text(encoding="utf-8"))

    def version(self, category: str = "bottle") -> str:
        """Lấy chuỗi phiên bản mô hình của danh mục."""
        try:
            return str(self.get_metadata(category).get("version", "unknown"))
        except FileNotFoundError:
            return "not_trained"

    def get_detector(self, category: str = "bottle") -> AnomalyDetector:
        """Lấy instance AnomalyDetector cho danh mục (sử dụng cache nếu đã nạp)."""
        if category in self._cached_detectors:
            return self._cached_detectors[category]

        from ..inference.detector import AnomalyDetector

        cat_dir = self.resolve_category_dir(category)
        detector = AnomalyDetector(model_dir=cat_dir)
        self._cached_detectors[category] = detector
        return detector
