"""Sliding-window инференс на карте произвольного размера."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.labels import colorize_label, overlay_prediction
from src.models import build_model
from src.postprocess import export_cost_map, morphological_close_linear
from src.utils.config import load_experiment_config
from src.utils.device import get_device, safe_torch_load

Image.MAX_IMAGE_PIXELS = None
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_model_from_checkpoint(ckpt_path: str | Path, cfg: dict, device: torch.device) -> torch.nn.Module:
    # Чекпоинт содержит все веса. Не загружаем исходные pretrained-веса,
    # чтобы не тратить сеть, память и время при старте production-сервиса.
    model = build_model(cfg, pretrained=False).to(device)
    ckpt = safe_torch_load(ckpt_path, map_location=device)
    state = ckpt.get("ema") or ckpt.get("model") or ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def _window_weight(window: int, blend: str = "gaussian") -> np.ndarray:
    if blend == "average":
        return np.ones((window, window), dtype=np.float32)
    # cosine / gaussian-like: выше вес в центре
    y = np.linspace(-1, 1, window, dtype=np.float32)
    x = np.linspace(-1, 1, window, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    if blend == "cosine":
        wy = 0.5 * (1 + np.cos(np.pi * yy))
        wx = 0.5 * (1 + np.cos(np.pi * xx))
        return (wy * wx).astype(np.float32)
    # gaussian
    sigma = 0.35
    return np.exp(-(xx**2 + yy**2) / (2 * sigma**2)).astype(np.float32)


def _pad_image(image: np.ndarray, window: int, step: int) -> tuple[np.ndarray, int, int]:
    h, w = image.shape[:2]
    pad_h = (step - (h - window) % step) % step if h > window else window - h
    pad_w = (step - (w - window) % step) % step if w > window else window - w
    if h < window:
        pad_h = window - h
    if w < window:
        pad_w = window - w
    # ensure at least one window
    pad_h = max(pad_h, 0)
    pad_w = max(pad_w, 0)
    if h + pad_h < window:
        pad_h = window - h
    if w + pad_w < window:
        pad_w = window - w
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return padded, pad_h, pad_w


def _normalize_batch(tiles: torch.Tensor) -> torch.Tensor:
    mean = tiles.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = tiles.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (tiles / 255.0 - mean) / std


def _tta_variants(tile: torch.Tensor) -> list[tuple[torch.Tensor, callable]]:
    """Возвращает (вариант, undo_fn) для усреднения softmax."""
    variants = [
        (tile, lambda x: x),
        (torch.flip(tile, dims=[3]), lambda x: torch.flip(x, dims=[3])),
        (torch.flip(tile, dims=[2]), lambda x: torch.flip(x, dims=[2])),
        (torch.flip(tile, dims=[2, 3]), lambda x: torch.flip(x, dims=[2, 3])),
    ]
    # rot90
    for k in (1, 2, 3):
        rot = torch.rot90(tile, k, dims=[2, 3])
        variants.append((rot, lambda x, kk=k: torch.rot90(x, -kk, dims=[2, 3])))
    return variants


@torch.no_grad()
def predict_tiles_batch(
    model: torch.nn.Module,
    tiles: torch.Tensor,
    device: torch.device,
    tta: bool = False,
) -> torch.Tensor:
    """tiles: [B,3,H,W] uint8/float 0..255 → softmax [B,C,H,W]."""
    tiles = tiles.to(device)
    x = _normalize_batch(tiles.float())
    if not tta:
        logits = model(x)
        return F.softmax(logits, dim=1)

    acc = None
    n = 0
    for var, undo in _tta_variants(x):
        logits = model(var)
        probs = F.softmax(logits, dim=1)
        probs = undo(probs)
        acc = probs if acc is None else acc + probs
        n += 1
    return acc / n


@torch.no_grad()
def predict_fullmap(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    num_classes: int = 16,
    window: int = 512,
    step: int = 256,
    blend: str = "gaussian",
    tta: bool = False,
    batch_size: int = 4,
) -> np.ndarray:
    """
    Sliding window с перекрытием и взвешенным блендингом softmax.
    Возвращает индексную маску [H,W] исходного размера.
    """
    assert image.ndim == 3 and image.shape[2] == 3
    orig_h, orig_w = image.shape[:2]
    padded, _, _ = _pad_image(image, window, step)
    ph, pw = padded.shape[:2]

    weight = _window_weight(window, blend=blend)
    # аккумулятор на CPU в fp16 для экономии памяти на огромных картах
    prob_acc = np.zeros((num_classes, ph, pw), dtype=np.float16)
    weight_acc = np.zeros((ph, pw), dtype=np.float16)

    coords: list[tuple[int, int]] = []
    for y in range(0, ph - window + 1, step):
        for x in range(0, pw - window + 1, step):
            coords.append((y, x))
    # edge coverage if step doesn't land on last
    if (ph - window) % step != 0:
        for x in range(0, pw - window + 1, step):
            coords.append((ph - window, x))
    if (pw - window) % step != 0:
        for y in range(0, ph - window + 1, step):
            coords.append((y, pw - window))
    coords.append((ph - window, pw - window))
    # unique
    coords = list(dict.fromkeys(coords))

    model.eval()
    for i in tqdm(range(0, len(coords), batch_size), desc="sliding-window", leave=False):
        batch_coords = coords[i : i + batch_size]
        tiles = []
        for y, x in batch_coords:
            tile = padded[y : y + window, x : x + window]
            tiles.append(torch.from_numpy(tile).permute(2, 0, 1))
        batch = torch.stack(tiles, dim=0)
        probs = predict_tiles_batch(model, batch, device, tta=tta).float().cpu().numpy()
        for j, (y, x) in enumerate(batch_coords):
            p = probs[j]
            prob_acc[:, y : y + window, x : x + window] += (p * weight).astype(np.float16)
            weight_acc[y : y + window, x : x + window] += weight.astype(np.float16)

    weight_acc = np.maximum(weight_acc, 1e-6)
    prob_acc = prob_acc / weight_acc[None, ...]
    pred = np.argmax(prob_acc, axis=0).astype(np.uint8)
    return pred[:orig_h, :orig_w]


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-map sliding-window inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True, help="RGB PNG карты")
    parser.add_argument("--out", required=True, help="префикс/директория выхода")
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--export-cost", action="store_true")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    device = get_device()
    model = load_model_from_checkpoint(args.checkpoint, cfg, device)

    image = np.array(Image.open(args.image).convert("RGB"))
    infer = cfg.get("infer", {})
    pred = predict_fullmap(
        model,
        image,
        device=device,
        num_classes=int(cfg["model"]["classes"]),
        window=int(args.window or infer.get("window", 512)),
        step=int(args.step or infer.get("step", 256)),
        blend=str(infer.get("blend", "gaussian")),
        tta=bool(args.tta or infer.get("tta", False)),
        batch_size=int(infer.get("batch_size", 4)),
    )

    if args.postprocess:
        pred = morphological_close_linear(pred, cfg)

    out = Path(args.out)
    if out.suffix:
        out.parent.mkdir(parents=True, exist_ok=True)
        stem = out.with_suffix("")
    else:
        out.mkdir(parents=True, exist_ok=True)
        stem = out / Path(args.image).stem

    Image.fromarray(pred).save(f"{stem}_label.png")
    Image.fromarray(colorize_label(pred)).save(f"{stem}_color.png")
    Image.fromarray(overlay_prediction(image, pred)).save(f"{stem}_overlay.png")

    if args.export_cost:
        cost = export_cost_map(pred, cfg)
        np.save(f"{stem}_cost.npy", cost)
        # визуализация стоимости
        cmax = max(float(cost[cost < 1e8].max()) if np.any(cost < 1e8) else 1.0, 1.0)
        vis = np.clip(cost / cmax * 255, 0, 255).astype(np.uint8)
        vis[cost >= 1e8] = 255
        Image.fromarray(vis).save(f"{stem}_cost.png")

    print(f"Saved predictions to {stem}_*.png ({pred.shape[1]}x{pred.shape[0]})")


if __name__ == "__main__":
    main()
