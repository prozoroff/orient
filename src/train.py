"""Обучение модели сегментации карт ориентирования."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# корень репозитория в PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import create_dataloaders
from src.data.labels import colorize_label, overlay_prediction
from src.losses import build_loss
from src.metrics import ConfusionMeter
from src.models import build_model, split_param_groups
from src.utils.config import load_experiment_config
from src.utils.device import autocast_ctx, get_device, make_grad_scaler, safe_torch_load
from src.utils.ema import ModelEMA
from src.utils.seed import set_seed


def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    tcfg = cfg["train"]
    epochs = int(tcfg["epochs"])
    warmup = int(tcfg.get("warmup_epochs", 0))
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup * steps_per_epoch
    name = tcfg.get("scheduler", "cosine")

    def lr_lambda(step: int) -> float:
        if step < warmup_steps and warmup_steps > 0:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        if name == "poly":
            power = float(tcfg.get("poly_power", 0.9))
            return (1.0 - progress) ** power
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, cfg, use_ema: bool = False) -> dict:
    model.eval()
    meter = ConfusionMeter(int(cfg["model"]["classes"]))
    total_loss = 0.0
    n = 0
    amp = bool(cfg["train"].get("amp", True))
    amp_dtype = str(cfg["train"].get("amp_dtype", "float16"))
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


@torch.no_grad()
def save_val_visualizations(model, loader, device, out_dir: Path, max_items: int = 8) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    count = 0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
        masks = batch["mask"]
        logits = model(images)
        preds = logits.argmax(1).cpu().numpy()
        denorm = (images * std + mean).clamp(0, 1)
        denorm = (denorm * 255).byte().cpu().numpy().transpose(0, 2, 3, 1)
        for i in range(images.size(0)):
            if count >= max_items:
                return
            img = denorm[i]
            gt = masks[i].numpy()
            pr = preds[i]
            overlay = overlay_prediction(img, pr)
            from PIL import Image

            Image.fromarray(overlay).save(out_dir / f"{batch['sample'][i]}_pred.png")
            Image.fromarray(colorize_label(gt)).save(out_dir / f"{batch['sample'][i]}_gt.png")
            count += 1


def train(cfg: dict) -> dict:
    set_seed(int(cfg.get("seed", 42)))
    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    exp = cfg.get("experiment_name", "exp")
    tcfg = cfg["train"]
    run_dir = Path(tcfg.get("runs_dir", "runs")) / exp
    ckpt_dir = Path(tcfg.get("checkpoint_dir", "checkpoints")) / exp
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    train_loader, val_loader, _ = create_dataloaders(cfg)
    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg).to(device)

    param_groups = split_param_groups(
        model,
        lr=float(tcfg["lr"]),
        encoder_lr_factor=float(tcfg.get("encoder_lr_factor", 0.1)),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
    )
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = make_grad_scaler(bool(tcfg.get("amp", True)), device)
    ema = ModelEMA(model, decay=float(tcfg.get("ema_decay", 0.999))) if tcfg.get("ema", True) else None

    start_epoch = 0
    best_miou = -1.0
    best_metrics: dict = {}
    patience = int(tcfg.get("early_stopping_patience", 20))
    bad_epochs = 0
    amp = bool(tcfg.get("amp", True))
    amp_dtype = str(tcfg.get("amp_dtype", "float16"))
    non_blocking = device.type == "cuda"

    resume = tcfg.get("resume")
    finetune_from = tcfg.get("finetune_from")
    if resume:
        ckpt = safe_torch_load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ema and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_miou = float(ckpt.get("best_miou", -1.0))
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        else:
            # старые чекпоинты без scheduler: прокрутить LR до start_epoch
            steps_done = start_epoch * len(train_loader)
            for _ in range(steps_done):
                scheduler.step()
        print(f"resume from {resume} → epoch {start_epoch}, best_mIoU={best_miou:.4f}")
    elif finetune_from:
        # Только веса модели/EMA; optimizer и epoch с нуля (для road-boost и т.п.).
        ckpt = safe_torch_load(finetune_from, map_location=device)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        if ema is not None:
            if "ema" in ckpt:
                ema.load_state_dict(ckpt["ema"])
            else:
                ema.ema.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        print(f"finetune from {finetune_from} (weights only)")

    accum = max(int(tcfg.get("grad_accumulation", 1)), 1)
    log_every = int(tcfg.get("log_every", 50))

    print(f"device={device} amp={amp} train_batches={len(train_loader)} val={len(val_loader.dataset)}")

    for epoch in range(start_epoch, int(tcfg["epochs"])):
        model.train()
        running = 0.0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for step, batch in enumerate(pbar):
            images = batch["image"].to(device, non_blocking=non_blocking)
            masks = batch["mask"].to(device, non_blocking=non_blocking)
            with autocast_ctx(amp, device, amp_dtype):
                logits = model(images)
                losses = loss_fn(logits, masks)
                loss = losses["loss"] / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
                scheduler.step()

            running += float(losses["loss"].item())
            if step % log_every == 0:
                lr = optimizer.param_groups[-1]["lr"]
                pbar.set_postfix(loss=f"{losses['loss'].item():.4f}", lr=f"{lr:.2e}")
                global_step = epoch * len(train_loader) + step
                writer.add_scalar("train/loss", losses["loss"].item(), global_step)
                writer.add_scalar("train/lr", lr, global_step)

        eval_model = ema.ema if ema is not None else model
        metrics = evaluate(eval_model, val_loader, loss_fn, device, cfg)
        writer.add_scalar("val/loss", metrics["loss"], epoch)
        writer.add_scalar("val/mIoU", metrics["mIoU"], epoch)
        writer.add_scalar("val/mDice", metrics["mDice"], epoch)
        for name, v in metrics.get("critical_iou", {}).items():
            writer.add_scalar(f"val/critical_iou/{name}", v, epoch)

        print(
            f"[{exp}] epoch {epoch}: "
            f"train_loss={running / max(len(train_loader), 1):.4f} "
            f"val_loss={metrics['loss']:.4f} mIoU={metrics['mIoU']:.4f} "
            f"time={time.time() - t0:.1f}s"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_miou": best_miou,
            "config": {k: v for k, v in cfg.items() if not k.startswith("_") or k in {"_config_path"}},
            "metrics": metrics,
        }
        if ema is not None:
            ckpt["ema"] = ema.state_dict()
        torch.save(ckpt, ckpt_dir / "last.pt")

        if metrics["mIoU"] > best_miou:
            best_miou = metrics["mIoU"]
            best_metrics = metrics
            ckpt["best_miou"] = best_miou
            torch.save(ckpt, ckpt_dir / "best.pt")
            (run_dir / "best_metrics.json").write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % int(tcfg.get("vis_every_epochs", 5)) == 0:
            save_val_visualizations(eval_model, val_loader, device, run_dir / "vis" / f"epoch_{epoch}")

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch}, best mIoU={best_miou:.4f}")
            break

    writer.close()
    print(f"Done. best mIoU={best_miou:.4f}. checkpoints → {ckpt_dir}")
    return {
        "best_miou": best_miou,
        "best_metrics": best_metrics,
        "checkpoint_dir": str(ckpt_dir),
        "run_dir": str(run_dir),
        "device": str(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train orienteering map segmentation")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
