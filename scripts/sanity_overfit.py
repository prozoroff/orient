"""Sanity-check: переобучение на 1–2 батчах (loss → ~0)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import create_dataloaders
from src.losses import build_loss
from src.models import build_model
from src.utils.config import load_experiment_config
from src.utils.device import autocast_ctx, get_device, make_grad_scaler
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_unet.yaml")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batches", type=int, default=2)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader, _, _ = create_dataloaders(cfg)
    batches = []
    for i, batch in enumerate(train_loader):
        batches.append(batch)
        if i + 1 >= args.batches:
            break

    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = make_grad_scaler(device.type == "cuda", device)
    amp = device.type == "cuda"

    model.train()
    for step in range(args.steps):
        batch = batches[step % len(batches)]
        images = batch["image"].to(device, non_blocking=amp)
        masks = batch["mask"].to(device, non_blocking=amp)
        opt.zero_grad(set_to_none=True)
        with autocast_ctx(amp, device):
            logits = model(images)
            losses = loss_fn(logits, masks)
            loss = losses["loss"]
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if step % 20 == 0 or step == args.steps - 1:
            print(f"step {step:4d} loss={loss.item():.6f} device={device}")

    print("Sanity overfit done (expect loss near 0 if targets/loss are correct).")


if __name__ == "__main__":
    main()
