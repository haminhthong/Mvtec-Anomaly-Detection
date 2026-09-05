"""Module đánh giá hiệu năng (Evaluation Pipeline) hệ thống phát hiện lỗi ngoại quan."""

from __future__ import annotations

import argparse
import sys

from .evaluation.aupro import compute_aupro
from .evaluation.evaluator import evaluate_category

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    """Entry point cho lệnh python -m src.evaluate."""
    parser = argparse.ArgumentParser(description="Đánh giá mô hình PatchCore trên tập test")
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Tên danh mục sản phẩm (mặc định: tự động đọc từ artifact config.json)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Đường dẫn thư mục chứa model artifacts",
    )
    args = parser.parse_args()
    evaluate_category(category=args.category, model_dir=args.model_dir)


if __name__ == "__main__":
    main()
