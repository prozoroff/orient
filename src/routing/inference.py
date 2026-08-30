"""Сегментация карты с кэшированием для интерактивного роутинга."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.infer_fullmap import load_model_from_checkpoint, predict_fullmap
from src.postprocess import morphological_close_linear

Image.MAX_IMAGE_PIXELS = None


def _file_digest(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()[:16]


def _cache_key(
    image_path: Path,
    checkpoint: Path | None,
    window: int,
    step: int,
    tta: bool,
    postprocess: bool,
    source: str,
) -> str:
    parts = [
        source,
        _file_digest(image_path) if image_path.exists() else "noimg",
        _file_digest(checkpoint) if checkpoint and checkpoint.exists() else "nockpt",
        f"w{window}",
        f"s{step}",
        f"tta{int(tta)}",
        f"pp{int(postprocess)}",
    ]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:20]


def default_cache_dir(project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[2]
    return root / "runs" / "routing_cache"


def load_rgb(image_path: str | Path) -> np.ndarray:
    return np.array(Image.open(image_path).convert("RGB"))


def load_label_png(label_path: str | Path) -> np.ndarray:
    """Загрузка индексной маски из label.png (режим L / grayscale)."""
    arr = np.array(Image.open(label_path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8)


def segment_map(
    image_path: str | Path,
    model: Any | None = None,
    *,
    device: Any | None = None,
    cfg: dict[str, Any] | None = None,
    checkpoint: str | Path | None = None,
    window: int | None = None,
    step: int | None = None,
    tta: bool | None = None,
    batch_size: int | None = None,
    postprocess: bool = True,
    close_kernel: int = 3,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    label_path: str | Path | None = None,
    force_reload: bool = False,
) -> np.ndarray:
    """
    Карта → индексная маска [H,W] uint8.

    Приоритет источников:
      1. кэш (если use_cache)
      2. label_path (GT / заранее посчитанная маска) — без модели
      3. инференс model / checkpoint

    Кэш: runs/routing_cache/<key>.npy + .json
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    infer = (cfg or {}).get("infer", {}) if cfg else {}
    window = int(window if window is not None else infer.get("window", 512))
    step = int(step if step is not None else infer.get("step", 256))
    tta = bool(tta if tta is not None else infer.get("tta", False))
    batch_size = int(batch_size if batch_size is not None else infer.get("batch_size", 4))
    num_classes = int((cfg or {}).get("model", {}).get("classes", 16))

    ckpt_path = Path(checkpoint) if checkpoint else None
    source = "label" if label_path is not None else "model"
    key = _cache_key(image_path, ckpt_path, window, step, tta, postprocess, source)
    if label_path is not None:
        key = _cache_key(Path(label_path), None, 0, 0, False, postprocess, "label")

    cache_root = Path(cache_dir) if cache_dir else default_cache_dir()
    npy_path = cache_root / f"{key}.npy"
    meta_path = cache_root / f"{key}.json"

    if use_cache and not force_reload and npy_path.exists():
        return np.load(npy_path)

    if label_path is not None:
        pred = load_label_png(label_path)
    else:
        if model is None:
            if cfg is None or ckpt_path is None or device is None:
                raise ValueError(
                    "Нужны model либо (cfg, checkpoint, device), либо label_path."
                )
            model = load_model_from_checkpoint(ckpt_path, cfg, device)
        image = load_rgb(image_path)
        pred = predict_fullmap(
            model,
            image,
            device=device,
            num_classes=num_classes,
            window=window,
            step=step,
            blend=str(infer.get("blend", "gaussian")),
            tta=tta,
            batch_size=batch_size,
        )

    if postprocess and cfg is not None:
        pred = morphological_close_linear(pred, cfg, kernel_size=close_kernel)

    if use_cache:
        cache_root.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, pred)
        meta_path.write_text(
            json.dumps(
                {
                    "image": str(image_path),
                    "label": str(label_path) if label_path else None,
                    "checkpoint": str(ckpt_path) if ckpt_path else None,
                    "window": window,
                    "step": step,
                    "tta": tta,
                    "postprocess": postprocess,
                    "shape": list(pred.shape),
                    "source": source,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return pred.astype(np.uint8)
