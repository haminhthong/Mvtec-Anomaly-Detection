"""Huấn luyện memory bank PatchCore với normal calibration tách ngẫu nhiên."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn
import torch
import torchvision
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from .config import TrainConfig, parse_args
from .data import ImageFolderDataset, find_category_root
from .features import FeatureExtractor
from .utils import apply_heatmap_smoothing, greedy_coreset, set_seed

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def split_normal_paths(
    paths: list[Path],
    calibration_fraction: float,
    seed: int,
    min_calibration_samples: int = 20,
) -> tuple[list[Path], list[Path]]:
    """Tách ảnh normal ngẫu nhiên, tái lập được cho memory và calibration.

    Args:
        paths: Danh sách đường dẫn ảnh bình thường (normal).
        calibration_fraction: Tỷ lệ ảnh dành riêng cho calibration.
        seed: Random seed cho việc xáo trộn và chia tách.
        min_calibration_samples: Số lượng mẫu calibration tối thiểu cần có (mặc định 20).

    Raises:
        ValueError: Nếu tổng số ảnh hoặc số ảnh calibration nhỏ hơn yêu cầu thống kê.
    """
    if len(paths) < min_calibration_samples:
        raise ValueError(
            f"Tổng số ảnh normal ({len(paths)}) nhỏ hơn số lượng calibration tối thiểu yêu cầu ({min_calibration_samples})."
        )
    memory, calibration = train_test_split(
        sorted(paths), test_size=calibration_fraction, random_state=seed, shuffle=True
    )
    if len(calibration) < min_calibration_samples:
        raise ValueError(
            f"Số lượng ảnh calibration ({len(calibration)}) nhỏ hơn ngưỡng yêu cầu ({min_calibration_samples}). "
            "Hãy tăng calibration_fraction hoặc bổ sung ảnh train/good để đảm bảo ước lượng quantile đáng tin cậy."
        )
    return sorted(memory), sorted(calibration)


def main(config: TrainConfig | None = None) -> None:
    """Tạo coreset, fit nearest-neighbor và chọn threshold trên held-out normal."""
    config = config or TrainConfig()
    config.validate()
    set_seed(config.seed)

    root = find_category_root(category=config.category)
    all_paths = sorted((root / "train" / "good").glob("*.png"))
    memory_paths, calibration_paths = split_normal_paths(
        all_paths,
        config.calibration_fraction,
        config.seed,
        min_calibration_samples=config.min_calibration_samples,
    )
    loader = DataLoader(
        ImageFolderDataset(memory_paths),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = FeatureExtractor().to(device)

    batches = [network(images.to(device)).cpu().numpy() for images, _ in loader]
    memory = np.concatenate(batches, axis=0)
    coreset_size = min(
        config.max_coreset_size,
        max(config.min_coreset_size, int(config.coreset_fraction * len(memory))),
        len(memory),
    )
    memory = greedy_coreset(memory, coreset_size, seed=config.seed)
    nearest_neighbors = NearestNeighbors(
        n_neighbors=1, metric="euclidean", n_jobs=-1
    ).fit(memory)

    calibration_scores: list[float] = []
    for path in calibration_paths:
        tensor = ImageFolderDataset([path])[0][0].unsqueeze(0).to(device)
        patches, (height, width) = network.extract_spatial_features(tensor)
        distances, _ = nearest_neighbors.kneighbors(patches.cpu().numpy())
        heatmap = distances.reshape(height, width)
        smoothed = apply_heatmap_smoothing(heatmap, sigma=config.smooth_sigma)
        calibration_scores.append(float(np.percentile(smoothed, 99)))

    threshold = float(np.quantile(calibration_scores, config.threshold_quantile))
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(nearest_neighbors, models_dir / "patch_nn.joblib")
    np.save(models_dir / "memory.npy", memory)

    payload = {
        "schema_version": 2,
        "category": config.category,
        "version": "mvtec-resnet18-patchcore-v4",
        "seed": config.seed,
        "device_used": device,
        "backbone": "resnet18-imagenet1k-v1",
        "feature_layers": ["layer2", "layer3"],
        "preprocessing": {
            "image_size": [224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "calibration_method": "random_held_out_normal",
        "calibration_fraction": config.calibration_fraction,
        "min_calibration_samples": config.min_calibration_samples,
        "threshold_quantile": config.threshold_quantile,
        "memory_images": len(memory_paths),
        "calibration_images": len(calibration_paths),
        "coreset_fraction": config.coreset_fraction,
        "coreset_size": len(memory),
        "smooth_sigma": config.smooth_sigma,
        "threshold": threshold,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (models_dir / "config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[SUCCESS] {config.category}: coreset={memory.shape}, "
        f"threshold={threshold:.4f}, calibration={len(calibration_paths)} ảnh"
    )


if __name__ == "__main__":
    main(parse_args())
