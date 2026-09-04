from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

DATASET = "foersben/mvtec-ad"
CATEGORY = "bottle"


def main():
    """Tải riêng category cần chạy để tránh tải toàn bộ archive gần 5 GB.

    Lưu ý: MVTec AD dùng giấy phép CC BY-NC-SA 4.0, không dùng thương mại.
    Trang chính thức hiện yêu cầu điền form tải dữ liệu, vì vậy dùng mirror
    Hugging Face có cùng giấy phép và giữ nguyên cấu trúc thư mục gốc.
    """
    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=DATASET,
        repo_type="dataset",
        allow_patterns=[f"{CATEGORY}/**"],
        local_dir=out,
    )
    print(f"Downloaded MVTec AD category '{CATEGORY}' into {out.resolve()}")


if __name__ == "__main__":
    main()
