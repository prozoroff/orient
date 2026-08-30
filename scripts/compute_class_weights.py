"""Печать весов классов из class_balance.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.losses.combined import compute_class_weights
from src.utils.config import load_experiment_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_unet.yaml")
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    w = compute_class_weights(
        class_balance_json=cfg["data"]["class_balance_json"],
        classes=cfg["classes"],
        empty_names=cfg.get("empty_classes", []),
        mode=cfg["loss"].get("class_weight_mode", "effective_number"),
        beta=float(cfg["loss"].get("effective_beta", 0.9999)),
        clip_min=float(cfg["loss"].get("weight_clip_min", 0.25)),
        clip_max=float(cfg["loss"].get("weight_clip_max", 10.0)),
        empty_weight=float(cfg["loss"].get("empty_class_weight", 0.0)),
        num_classes=int(cfg["model"]["classes"]),
    )
    for i, name in sorted(cfg["classes"].items()):
        print(f"{i:2d} {name:20s} {w[i].item():.4f}")


if __name__ == "__main__":
    main()
