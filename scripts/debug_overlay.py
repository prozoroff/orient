"""Проверка свёртки channels→index и overlay-визуализация тайлов."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import load_index
from src.data.labels import (
    build_priority_indices,
    channels_to_index,
    colorize_label,
    overlay_prediction,
)
from src.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--index", default="dataset/dataset_index.csv")
    parser.add_argument("--classes-config", default="configs/dataset_classes.yaml")
    parser.add_argument("--out", default="runs/debug_overlay")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument(
        "--verify-tiles",
        type=int,
        default=32,
        help="сколько тайлов проверить на валидную свёртку channels→index",
    )
    args = parser.parse_args()

    classes_cfg = load_yaml(args.classes_config)
    classes = {int(k): v for k, v in classes_cfg["classes"].items()}
    priority = build_priority_indices(classes_cfg["label_priority"], classes)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_index(args.index, args.dataset_root, split="train", kinds=["tile"])
    n_classes = max(classes) + 1
    for rec in records[: args.verify_tiles]:
        if rec.channels is None:
            raise FileNotFoundError(rec.sample)
        channels = np.load(rec.channels)
        if channels.ndim != 3 or channels.shape[0] != n_classes:
            raise ValueError(
                f"{rec.map_name}/{rec.sample}: expected [{n_classes},H,W], got {channels.shape}"
            )
        label = channels_to_index(channels, priority)
        assert label.shape == channels.shape[1:], label.shape
        assert label.max() < n_classes, int(label.max())
        print(f"OK {rec.map_name}/{rec.sample} unique={np.unique(label).tolist()}")

    print(f"Verified channels→index on {min(args.verify_tiles, len(records))} tiles")

    for rec in records[: args.n]:
        image = np.array(Image.open(rec.image).convert("RGB"))
        channels = np.load(rec.channels)
        label = channels_to_index(channels, priority)
        overlay = overlay_prediction(image, label)
        Image.fromarray(overlay).save(out_dir / f"{rec.map_name}_{rec.sample}_overlay.png")
        Image.fromarray(colorize_label(label)).save(
            out_dir / f"{rec.map_name}_{rec.sample}_label.png"
        )
        print(f"saved {rec.map_name}/{rec.sample} unique={np.unique(label).tolist()}")

    print(f"Overlays → {out_dir}")


if __name__ == "__main__":
    main()
