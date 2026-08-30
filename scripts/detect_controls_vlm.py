#!/usr/bin/env python3
"""CLI: детекция КП через Vision LLM (Yandex Cloud / OpenAI)."""

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

from src.controls import detect_controls_vlm, draw_controls
from src.controls.vlm_yandex import list_yandex_chat_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect controls via Yandex/OpenAI Vision")
    parser.add_argument("image", type=str, nargs="?", default=None, help="путь к RGB-изображению")
    parser.add_argument("--list-models", action="store_true", help="показать chat-модели Yandex")
    parser.add_argument("--out-json", type=str, default=None)
    parser.add_argument("--out-preview", type=str, default=None)
    parser.add_argument(
        "--provider",
        choices=["yandex", "openai"],
        default="yandex",
        help="yandex = AI Studio VL-модель; openai = GPT Vision",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="имя модели; для yandex сначала --list-models",
    )
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--folder-id", type=str, default=None, help="YC_FOLDER_ID (yandex)")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=160)
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.35)
    args = parser.parse_args()

    if args.list_models:
        for uri in list_yandex_chat_models(api_key=args.api_key, folder_id=args.folder_id):
            print(uri)
        return

    if not args.image:
        raise SystemExit("укажите image или --list-models")

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"нет файла: {image_path}")

    result = detect_controls_vlm(
        image_path,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        folder_id=args.folder_id,
        tile_size=args.tile_size,
        overlap=args.overlap,
        refine_circles=not args.no_refine,
        min_confidence=args.min_confidence,
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
