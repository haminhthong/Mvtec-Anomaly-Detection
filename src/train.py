"""Điểm nhập dòng lệnh huấn luyện memory bank PatchCore và căn chỉnh ngưỡng."""

from __future__ import annotations

import sys

from .config import TrainConfig, parse_args
from .training.calibration import split_normal_paths
from .training.trainer import set_seed, train_patchcore

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main(config: TrainConfig | None = None) -> None:
    """Entry point cho lệnh python -m src.train."""
    cfg = config or parse_args()
    train_patchcore(cfg)


if __name__ == "__main__":
    main()
