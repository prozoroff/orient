"""Оценка на тайлах и агрегация метрик по картам."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import SampleRecord, create_dataloaders, load_index
from src.data.labels import build_priority_indices, channels_to_index, overlay_prediction
from src.data.transforms import build_eval_transforms
from src.infer_fullmap import load_model_from_checkpoint, predict_fullmap
from src.losses import build_loss
from src.metrics import ConfusionMeter
from src.utils.config import load_experiment_config
from src.utils.device import autocast_ctx, get_device
from src.utils.seed import set_seed

Image.MAX_IMAGE_PIXELS = None


@torch.no_grad()
def evaluate_loader(model, loader, loss_fn, device, cfg) -> dict:
    model.eval()
    meter = ConfusionMeter(int(cfg["model"]["classes"]))
    total_loss = 0.0
    n = 0
    amp = bool(cfg.get("train", {}).get("amp", True))
    amp_dtype = str(cfg.get("train", {}).get("amp_dtype", "float16"))
    non_blocking = device.type == "cuda"
    for batch in loader:
        images = batch["image"].to(device, non_blocking=non_blocking)
        masks = batch["mask"].to(device, non_blocking=non_blocking)
        with autocast_ctx(amp, device, amp_dtype):
            logits = model(images)
            losses = loss_fn(logits, masks)
        total_loss += float(losses["loss"].item())
        n += 1
        meter.update(logits.argmax(1), masks)

    metrics = meter.compute(
        classes=cfg["classes"],
        exclude_background=bool(cfg.get("eval", {}).get("exclude_background", True)),
        exclude_empty=cfg.get("empty_classes", [])
        if cfg.get("eval", {}).get("exclude_empty_classes", True)
        else [],
        critical=cfg.get("critical_classes", []),
    )
    metrics["loss"] = total_loss / max(n, 1)
    return metrics


def evaluate_tiles(cfg: dict, ckpt_path: str, split: str = "test") -> dict:
    device = get_device()
    model = load_model_from_checkpoint(ckpt_path, cfg, device)
    loss_fn = build_loss(cfg).to(device)
    train_loader, val_loader, test_loader = create_dataloaders(cfg)
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[split]
    return evaluate_loader(model, loader, loss_fn, device, cfg)


def list_full_maps(cfg: dict, split: str = "test") -> list[str]:
    records = load_index(
        cfg["data"]["index_csv"],
        cfg["data"]["dataset_root"],
        split=split,
        kinds=["tile"],
    )
    return sorted({r.map_name for r in records})


def load_tile_gt(rec: SampleRecord, priority: list[int]) -> np.ndarray:
    if rec.label is not None and rec.label.exists():
        return np.array(Image.open(rec.label), dtype=np.uint8)
    if rec.channels is not None and rec.channels.exists():
        return channels_to_index(np.load(rec.channels), priority)
    raise FileNotFoundError(f"No label/channels for {rec.map_name}/{rec.sample}")


def load_fullmap_gt(map_dir: Path, priority: list[int]) -> np.ndarray:
    """Legacy: full-map label/channels в корне карты (если есть)."""
    label_path = map_dir / "label.png"
    if label_path.exists():
        return np.array(Image.open(label_path), dtype=np.uint8)
    channels_path = map_dir / "channels.npy"
    if channels_path.exists():
        return channels_to_index(np.load(channels_path), priority)
    raise FileNotFoundError(f"No label/channels in {map_dir}")


@torch.no_grad()
def evaluate_fullmaps(
    cfg: dict,
    ckpt_path: str,
    split: str = "test",
    out_dir: Path | None = None,
) -> dict:
    """
    Метрики по картам.

    Новый датасет хранит только тайлы — тогда агрегируем предсказания по тайлам карты.
    Если в корне карты есть image.png (legacy), делаем sliding-window на full-рендере.
    """
    device = get_device()
    model = load_model_from_checkpoint(ckpt_path, cfg, device)
    maps = list_full_maps(cfg, split=split)
    dataset_root = Path(cfg["data"]["dataset_root"])
    priority = build_priority_indices(cfg["label_priority"], cfg["classes"])
    meter = ConfusionMeter(int(cfg["model"]["classes"]))
    per_map: dict = {}

    all_tiles = load_index(
        cfg["data"]["index_csv"],
        cfg["data"]["dataset_root"],
        split=split,
        kinds=["tile"],
    )
    tiles_by_map: dict[str, list[SampleRecord]] = defaultdict(list)
    for rec in all_tiles:
        tiles_by_map[rec.map_name].append(rec)

    transform = build_eval_transforms()
    infer_cfg = cfg.get("infer", {})
    amp = bool(cfg.get("train", {}).get("amp", True))
    amp_dtype = str(cfg.get("train", {}).get("amp_dtype", "float16"))
    non_blocking = device.type == "cuda"

    for map_name in tqdm(maps, desc=f"fullmap-{split}"):
        map_dir = dataset_root / map_name
        full_image = map_dir / "image.png"
        map_meter = ConfusionMeter(int(cfg["model"]["classes"]))

        if full_image.exists():
            image = np.array(Image.open(full_image).convert("RGB"))
            gt = load_fullmap_gt(map_dir, priority)
            pred = predict_fullmap(
                model,
                image,
                device=device,
                num_classes=int(cfg["model"]["classes"]),
                window=int(infer_cfg.get("window", 512)),
                step=int(infer_cfg.get("step", 256)),
                blend=str(infer_cfg.get("blend", "gaussian")),
                tta=bool(infer_cfg.get("tta", False)),
                batch_size=int(infer_cfg.get("batch_size", 4)),
            )
            h, w = gt.shape
            pred = pred[:h, :w]
            meter.update(torch.from_numpy(pred[None]), torch.from_numpy(gt[None]))
            map_meter.update(torch.from_numpy(pred[None]), torch.from_numpy(gt[None]))
            if out_dir is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pred).save(out_dir / f"{map_name}_pred_label.png")
                Image.fromarray(overlay_prediction(image, pred)).save(
                    out_dir / f"{map_name}_overlay.png"
                )
        else:
            # tiles-only датасет
            for rec in tiles_by_map[map_name]:
                image = np.array(Image.open(rec.image).convert("RGB"), dtype=np.uint8)
                gt = load_tile_gt(rec, priority)
                out = transform(image=image, mask=gt)
                img_t = out["image"].unsqueeze(0).to(device, non_blocking=non_blocking)
                with autocast_ctx(amp, device, amp_dtype):
                    logits = model(img_t)
                pred = logits.argmax(1).cpu().numpy()[0].astype(np.uint8)
                meter.update(torch.from_numpy(pred[None]), torch.from_numpy(gt[None]))
                map_meter.update(torch.from_numpy(pred[None]), torch.from_numpy(gt[None]))
                if out_dir is not None:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    stem = f"{map_name}_{rec.sample}"
                    Image.fromarray(pred).save(out_dir / f"{stem}_pred_label.png")
                    Image.fromarray(overlay_prediction(image, pred)).save(
                        out_dir / f"{stem}_overlay.png"
                    )

        map_metrics = map_meter.compute(
            classes=cfg["classes"],
            exclude_background=True,
            exclude_empty=cfg.get("empty_classes", []),
            critical=cfg.get("critical_classes", []),
        )
        per_map[map_name] = {
            "mIoU": map_metrics["mIoU"],
            "critical_iou": map_metrics["critical_iou"],
        }

    metrics = meter.compute(
        classes=cfg["classes"],
        exclude_background=bool(cfg.get("eval", {}).get("exclude_background", True)),
        exclude_empty=cfg.get("empty_classes", [])
        if cfg.get("eval", {}).get("exclude_empty_classes", True)
        else [],
        critical=cfg.get("critical_classes", []),
    )
    metrics["per_map"] = per_map
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--mode", default="tiles", choices=["tiles", "fullmaps", "both"])
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    out: dict = {}
    out_dir = Path(args.out) if args.out else None

    if args.mode in {"tiles", "both"}:
        out["tiles"] = evaluate_tiles(cfg, args.checkpoint, split=args.split)
        print(f"Tiles {args.split} mIoU={out['tiles']['mIoU']:.4f}")
    if args.mode in {"fullmaps", "both"}:
        out["fullmaps"] = evaluate_fullmaps(
            cfg, args.checkpoint, split=args.split, out_dir=out_dir
        )
        print(f"Fullmaps {args.split} mIoU={out['fullmaps']['mIoU']:.4f}")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
