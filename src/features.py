"""Module trích xuất đặc trưng PatchCore đa tầng (Multi-Layer Feature Extractor).

Sử dụng mạng backbone ResNet18 đã được tiền huấn luyện trên ImageNet để trích xuất
feature maps từ các tầng trung gian (Layer 2 và Layer 3), áp dụng nội suy không gian (spatial interpolation)
và kết hợp (concatenation) nhằm tạo ra các vector patch embedding có khả năng nắm bắt cả chi tiết cục bộ
lẫn ngữ cảnh đại thể.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class FeatureExtractor(nn.Module):
    """Trích xuất đặc trưng patch đa tầng từ ResNet18 không tính toán gradient.

    Theo nghiên cứu PatchCore (CVPR 2022), việc kết hợp feature maps từ Layer 2 và Layer 3
    giúp tối ưu hóa đồng thời độ chính xác phân loại ảnh (Image AUROC) và độ chính xác
    định vị vùng lỗi (Pixel AUROC / AUPRO).
    """

    def __init__(self) -> None:
        super().__init__()
        # Tải mô hình ResNet18 với trọng số mặc định (ImageNet V1)
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Tách các tầng trung gian của ResNet18
        self.stage1 = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
        )
        self.layer2 = (
            base_model.layer2
        )  # Trả về feature map kích thước [B, 128, 28, 28]
        self.layer3 = (
            base_model.layer3
        )  # Trả về feature map kích thước [B, 256, 14, 14]

        # Đóng đóng băng toàn bộ trọng số mạng (eval mode, không cập nhật gradient)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Thực hiện lan truyền tiến để lấy tensor đặc trưng dạng patch [N_patches, C_total].

        Args:
            x: Tensor ảnh đầu vào có hình dạng [Batch, 3, Height, Width] (ví dụ: [B, 3, 224, 224]).

        Returns:
            torch.Tensor: Tensor chứa các vector patch embedding [B * H_map * W_map, C_layer2 + C_layer3].
                          Ví dụ: với kích thước 224x224, trả về [B * 784, 384].
        """
        patches, _ = self.extract_spatial_features(x)
        return patches

    @torch.inference_mode()
    def extract_spatial_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """Trích xuất patch features kèm thông tin kích thước lưới không gian (H_map, W_map).

        Args:
            x: Tensor ảnh đầu vào [Batch, 3, Height, Width].

        Returns:
            tuple[torch.Tensor, tuple[int, int]]: (Tensor patch embeddings [B*H*W, C], (H_map, W_map)).
        """
        feats1 = self.stage1(x)
        feats2 = self.layer2(feats1)
        feats3 = self.layer3(feats2)

        feats3_upsampled = F.interpolate(
            feats3, size=feats2.shape[2:], mode="bilinear", align_corners=False
        )

        combined = torch.cat([feats2, feats3_upsampled], dim=1)
        _, c, h, w = combined.shape
        patches = combined.permute(0, 2, 3, 1).reshape(-1, c)
        return patches, (h, w)
