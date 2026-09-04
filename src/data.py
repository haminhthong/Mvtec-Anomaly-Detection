"""Module quản lý dữ liệu hình ảnh và tiền xử lý cho dataset MVTec AD.

Cung cấp PyTorch Dataset để biến đổi hình ảnh về kích thước chuẩn (224x224),
chuẩn hóa RGB theo ImageNet, và hàm hỗ trợ tìm kiếm đường dẫn thư mục danh mục sản phẩm.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

# Pipeline chuẩn hóa hình ảnh theo tiêu chuẩn ImageNet
TFM: transforms.Compose = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class ImageFolderDataset(Dataset):
    """PyTorch Dataset đọc các tập tin ảnh từ danh sách đường dẫn truyền vào.

    Args:
        paths: Danh sách hoặc sequence các đường dẫn đường dẫn tệp ảnh (Path hoặc str).
    """

    def __init__(self, paths: Sequence[str | Path]) -> None:
        self.paths: list[Path] = [Path(p) for p in paths]

    def __len__(self) -> int:
        """Trả về tổng số lượng ảnh trong dataset."""
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[Tensor, str]:
        """Đọc và tiền xử lý ảnh tại chỉ số i.

        Args:
            i: Chỉ số ảnh cần lấy.

        Returns:
            tuple[Tensor, str]: Tensor ảnh đã được biến đổi [3, H, W] và đường dẫn tập tin dạng chuỗi.
        """
        p = self.paths[i]
        with Image.open(p) as img:
            tensor_img = TFM(img.convert("RGB"))
        return tensor_img, str(p)


def find_category_root(raw: str | Path = "data/raw", category: str = "bottle") -> Path:
    """Tìm kiếm thư mục gốc của danh mục sản phẩm trong thư mục dữ liệu thô.

    Args:
        raw: Đường dẫn tới thư mục data/raw.
        category: Tên danh mục sản phẩm (ví dụ: 'bottle', 'cable', 'capsule').

    Returns:
        Path: Đường dẫn tới thư mục danh mục sản phẩm tồn tại.

    Raises:
        FileNotFoundError: Nếu không tìm thấy danh mục sản phẩm ở các vị trí ứng viên.
    """
    raw_path = Path(raw)
    candidates = [
        raw_path / category,
        raw_path / "mvtec_anomaly_detection" / category,
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    raise FileNotFoundError(
        f"Không tìm thấy danh mục sản phẩm '{category}' tại '{raw_path}'. "
        "Vui lòng chạy 'python scripts/download_data.py' để tải dữ liệu."
    )
