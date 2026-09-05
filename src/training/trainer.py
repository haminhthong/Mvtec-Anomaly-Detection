"""Module huấn luyện mô hình phát hiện lỗi ngoại quan PatchCore (Offline Model Building).

Thực hiện 4 giai đoạn đầu trong 5 giai đoạn canonical:
1. DATA PREPARATION: Đọc ảnh train/good, phân chia 80% Memory và 20% Held-out Calibration.
2. NORMAL REPRESENTATION LEARNING: Trích xuất đặc trưng đa tầng ResNet18 (Layer2 128D + Layer3 256D = 384D).
3. MEMORY BANK CONSTRUCTION: Chiếu ngẫu nhiên 64D -> Greedy K-Center Coreset -> Compact Memory Bank 384D.
4. CALIBRATION: Căn chỉnh ngưỡng kép (P95 review, P99 fail) và pixel threshold hoàn toàn trên Normal data.
Xuất artifact Thiết kế B: memory_bank.npy và config.json.
"""

from __future__ import annotations

import json
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
import torchvision
from torch.utils.data import DataLoader

from ..config import TrainConfig
from ..data.dataset import ImageFolderDataset, find_category_root
from ..data.transforms import build_transform
from ..inference.localization import apply_heatmap_smoothing
from ..model.coreset import greedy_coreset
from ..model.memory_bank import MemoryBank
from ..model.patch_embedding import FeatureExtractor
from .calibration import calibrate_thresholds, split_normal_paths


def set_seed(seed: int = 42) -> None:
    """Cố định seed ngẫu nhiên cho Python, NumPy và PyTorch để đảm bảo tính tái lập."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_patchcore(config: TrainConfig | None = None) -> dict[str, Any]:
    """Quy trình huấn luyện và căn chỉnh ngưỡng hoàn chỉnh cho một danh mục sản phẩm.

    Args:
        config: TrainConfig chứa các tham số huấn luyện. Nếu None sẽ dùng cấu hình mặc định.

    Returns:
        dict[str, Any]: Payload metadata và ngưỡng đã lưu vào artifact.
    """
    cfg = config or TrainConfig()
    cfg.validate()
    set_seed(cfg.seed)

    # 1. DATA PREPARATION
    root = find_category_root(category=cfg.category)
    all_normal_paths = sorted((root / "train" / "good").glob("*.png"))
    memory_paths, calibration_paths = split_normal_paths(
        all_normal_paths,
        calibration_fraction=cfg.calibration_fraction,
        seed=cfg.seed,
        min_calibration_samples=cfg.min_calibration_samples,
    )

    transform = build_transform(cfg.preprocessing)
    loader = DataLoader(
        ImageFolderDataset(memory_paths, transform=transform),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = FeatureExtractor().to(device)

    # 2. NORMAL REPRESENTATION LEARNING
    batches = [network(images.to(device)).cpu().numpy() for images, _ in loader]
    full_memory = np.concatenate(batches, axis=0)

    # 3. MEMORY BANK CONSTRUCTION (Coreset selection)
    coreset_size = min(
        cfg.max_coreset_size,
        max(cfg.min_coreset_size, int(cfg.coreset_fraction * len(full_memory))),
        len(full_memory),
    )
    compact_memory = greedy_coreset(full_memory, coreset_size, seed=cfg.seed)
    memory_bank = MemoryBank(compact_memory)

    # 4. CALIBRATION (Held-out Normal)
    calibration_scores: list[float] = []
    calibration_heatmaps: list[np.ndarray] = []

    for path in calibration_paths:
        tensor = ImageFolderDataset([path], transform=transform)[0][0].unsqueeze(0).to(device)
        patches, (height, width) = network.extract_spatial_features(tensor)
        distances, _ = memory_bank.kneighbors(patches.cpu().numpy())
        raw_heat = distances.reshape(height, width)
        smoothed = apply_heatmap_smoothing(raw_heat, sigma=cfg.smooth_sigma)
        calibration_scores.append(float(np.percentile(smoothed, 99)))
        calibration_heatmaps.append(smoothed)

    review_threshold, fail_threshold, pixel_threshold = calibrate_thresholds(
        normal_scores=calibration_scores,
        normal_heatmaps=calibration_heatmaps,
        review_quantile=cfg.review_quantile,
        fail_quantile=cfg.threshold_quantile,
        pixel_quantile=cfg.pixel_quantile,
    )

    # 5. LƯU TRỮ MODEL ARTIFACT (Thiết kế B)
    models_dir = Path("models")
    category_models_dir = models_dir / cfg.category
    category_models_dir.mkdir(parents=True, exist_ok=True)

    # Lưu mảng numpy memory bank
    memory_bank.save(category_models_dir / "memory_bank.npy")
    memory_bank.save(models_dir / "memory_bank.npy")
    # File memory.npy duy trì tương thích ngược
    np.save(models_dir / "memory.npy", compact_memory)

    payload: dict[str, Any] = {
        "schema_version": 3,
        "category": cfg.category,
        "version": "mvtec-resnet18-patchcore-v5",
        "seed": cfg.seed,
        "device_used": device,
        "backbone": "resnet18-imagenet1k-v1",
        "feature_layers": ["layer2", "layer3"],
        "preprocessing": cfg.preprocessing.to_dict(),
        "calibration": {
            "method": "held_out_normal_dual_calibration",
            "calibration_fraction": cfg.calibration_fraction,
            "min_calibration_samples": cfg.min_calibration_samples,
            "memory_images": len(memory_paths),
            "calibration_images": len(calibration_paths),
            "review_quantile": cfg.review_quantile,
            "threshold_quantile": cfg.threshold_quantile,
            "pixel_quantile": cfg.pixel_quantile,
        },
        "thresholds": {
            "review_threshold": review_threshold,
            "fail_threshold": fail_threshold,
            "image_threshold": fail_threshold,
            "pixel_threshold": pixel_threshold,
        },
        "threshold": fail_threshold,  # Tương thích ngược với v2/v4
        "review_threshold": review_threshold,
        "pixel_threshold": pixel_threshold,
        "coreset": {
            "fraction": cfg.coreset_fraction,
            "size": len(compact_memory),
            "full_memory_patches": len(full_memory),
        },
        "smooth_sigma": cfg.smooth_sigma,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    config_json = json.dumps(payload, ensure_ascii=False, indent=2)
    (category_models_dir / "config.json").write_text(config_json, encoding="utf-8")
    (models_dir / "config.json").write_text(config_json, encoding="utf-8")

    print(
        f"[SUCCESS] {cfg.category}: Coreset={compact_memory.shape} (từ {len(full_memory)} patches), "
        f"Review={review_threshold:.4f} (P95), Fail/Image={fail_threshold:.4f} (P99), "
        f"Pixel={pixel_threshold:.4f} | Calibration={len(calibration_paths)} ảnh normal"
    )
    return payload
