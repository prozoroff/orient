#!/usr/bin/env python3
"""CLI: детекция контрольных пунктов на карте / фото."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.controls import detect_controls, draw_controls

Image.MAX_IMAGE_PIXELS = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect orienteering control points (OCR)")
    parser.add_argument("image", type=str, help="путь к RGB-изображению")
    parser.add_argument("--out-json", type=str, default=None, help="сохранить JSON")
    parser.add_argument("--out-preview", type=str, default=None, help="сохранить preview PNG")
    parser.add_argument("--prior", type=str, default=None, help="channels.npy или бинарный mask")
    parser.add_argument(
        "--sensitivity",
        choices=["balanced", "recall"],
        default="recall",
        help="recall = ниже пороги OCR/розового (больше КП, больше шума)",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=None,
        help="порог EasyOCR (по умолчанию от --sensitivity: balanced≈0.15, recall≈0.05)",
    )
    parser.add_argument(
        "--min-pink-overlap",
        type=float,
        default=None,
        help="доля розовых px в bbox (0 = не фильтровать по цвету; digit_first)",
    )
    parser.add_argument(
        "--mode",
        choices=["circle_first", "digit_first"],
        default="circle_first",
        help="circle_first = кольца→OCR (по умолчанию); digit_first = старый OCR→кольцо",
    )
    parser.add_argument("--cpu", action="store_true", help="форсировать CPU для EasyOCR")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"нет файла: {image_path}")

    prior = None
    if args.prior:
        p = Path(args.prior)
        arr = np.load(p) if p.suffix == ".npy" else np.array(Image.open(p))
        if arr.ndim == 3:
            # channels.npy → канал control_point (15), иначе max по каналам
            prior = arr[15] if arr.shape[0] == 16 else (arr.max(axis=0) if arr.shape[0] < 8 else arr[..., 0])
        else:
            prior = arr

    result = detect_controls(
        image_path,
        prior_mask=prior,
        sensitivity=args.sensitivity,
        min_ocr_conf=args.min_conf,
        min_pink_overlap=args.min_pink_overlap,
        mode=args.mode,
        gpu=False if args.cpu else None,
    )

    payload = result.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(
        f"# found {len(result.controls)} controls"
        + (f", start=({result.start.x:.0f},{result.start.y:.0f})" if result.start else ", start=None"),
        file=sys.stderr,
    )

    if args.out_json:
        out_j = Path(args.out_json)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        out_j.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.out_preview:
        rgb = np.array(Image.open(image_path).convert("RGB"))
        preview = draw_controls(rgb, result)
        out_p = Path(args.out_preview)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(preview).save(out_p)
        print(f"# preview → {out_p}", file=sys.stderr)


if __name__ == "__main__":
    main()
