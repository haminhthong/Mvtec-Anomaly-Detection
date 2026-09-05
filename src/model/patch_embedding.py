"""Module trích xuất đặc trưng PatchCore-style đa tầng (Multi-Layer Feature Extractor).

Sử dụng mạng backbone ResNet18 đã được tiền huấn luyện trên ImageNet để trích xuất
feature maps từ các tầng trung gian (Layer 2 và Layer 3), áp dụng nội suy không gian
(bilinear interpolation upsampling) và kết hợp (concatenation) nhằm tạo ra các vector
patch embedding có khả năng nắm bắt cả chi tiết cục bộ (textures) lẫn ngữ cảnh ngữ nghĩa (semantics).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class FeatureExtractor(nn.Module):
    """Trích xuất đặc trưng patch đa tầng từ ResNet18 không tính toán gradient.

    Theo tinh thần PatchCore (CVPR 2022):
    - Layer 2: kích thước [B, 128, 28, 28] bảo toàn độ phân giải chi tiết cục bộ.
    - Layer 3: kích thước [B, 256, 14, 14] nắm bắt ngữ cảnh rộng hơn.
    Nội suy Layer 3 lên 28x28 rồi concatenate tạo vector 384 chiều cho mỗi vị trí trong 28x28 (784 patches/ảnh).
    """

    def __init__(self) -> None:
        super().__init__()
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.stage1 = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
        )
        self.layer2 = base_model.layer2  # [B, 128, 28, 28]
        self.layer3 = base_model.layer3  # [B, 256, 14, 14]

        # Đóng băng toàn bộ tham số (eval mode, không gradient)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Thực hiện lan truyền tiến để lấy tensor đặc trưng dạng patch [N_patches, C_total].

        Args:
            x: Tensor ảnh đầu vào [Batch, 3, Height, Width] (ví dụ: [B, 3, 224, 224]).

        Returns:
            torch.Tensor: Tensor chứa các vector patch embedding [B * 784, 384].
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

        # Nội suy Layer 3 lên cùng kích thước không gian với Layer 2 (28x28)
        feats3_upsampled = F.interpolate(
            feats3, size=feats2.shape[2:], mode="bilinear", align_corners=False
        )

        combined = torch.cat([feats2, feats3_upsampled], dim=1)
        _, c, h, w = combined.shape
        patches = combined.permute(0, 2, 3, 1).reshape(-1, c)
        return patches, (h, w)
