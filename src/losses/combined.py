"""Комбинированный лосс: weighted CE + Dice (+ optional Lovász)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.data.labels import empty_class_indices


def compute_class_weights(
    class_balance_json: str | Path,
    classes: dict[int, str],
    empty_names: Sequence[str],
    mode: str = "effective_number",
    beta: float = 0.9999,
    clip_min: float = 0.25,
    clip_max: float = 10.0,
    empty_weight: float = 0.0,
    num_classes: int = 16,
    class_weight_boost: dict[str, float] | None = None,
) -> torch.Tensor:
    """Веса классов из reports/class_balance.json (Cui et al. effective number)."""
    data = json.loads(Path(class_balance_json).read_text(encoding="utf-8"))
    pixels = np.zeros(num_classes, dtype=np.float64)
    for row in data:
        idx = int(row["class_index"])
        pixels[idx] = float(row["pixels"])

    empty_idx = set(empty_class_indices(empty_names, classes))
    weights = np.ones(num_classes, dtype=np.float64)
    active_mask = np.array(
        [i not in empty_idx and i != 0 and pixels[i] > 0 for i in range(num_classes)]
    )

    if mode == "none":
        weights[:] = 1.0
    elif mode in {"effective_number", "inverse_freq", "inverse_sqrt_freq"}:
        # Доли по пикселям среди активных классов (background/empty исключены).
        total = pixels[active_mask].sum()
        freq = np.zeros(num_classes, dtype=np.float64)
        if total > 0:
            freq[active_mask] = pixels[active_mask] / total

        for i in range(num_classes):
            if i in empty_idx:
                weights[i] = empty_weight
                continue
            if i == 0 or not active_mask[i]:
                weights[i] = 1.0
                continue
            f = max(freq[i], 1e-12)
            if mode == "effective_number":
                # Effective number по «сэмплам»: масштабируем долю в условные samples,
                # иначе beta**pixels насыщается и все веса схлопываются в 1.
                n = max(f * 10_000.0, 1.0)
                eff = (1.0 - beta**n) / (1.0 - beta)
                w = 1.0 / max(eff, 1e-12)
            elif mode == "inverse_freq":
                w = 1.0 / f
            else:  # inverse_sqrt_freq
                w = 1.0 / np.sqrt(f)
            weights[i] = w

        if active_mask.any() and weights[active_mask].sum() > 0:
            weights[active_mask] = weights[active_mask] / weights[active_mask].mean()
            weights[active_mask] = np.clip(weights[active_mask], clip_min, clip_max)
        weights[0] = 1.0
    else:
        raise ValueError(f"Unknown class_weight_mode: {mode}")

    for i in empty_idx:
        weights[i] = empty_weight

    # Точечное усиление редких/критичных классов (после нормализации).
    if class_weight_boost:
        name_to_idx = {name: int(idx) for idx, name in classes.items()}
        for name, mult in class_weight_boost.items():
            if name not in name_to_idx:
                continue
            i = name_to_idx[name]
            if i in empty_idx:
                continue
            weights[i] *= float(mult)
        if active_mask.any():
            weights[active_mask] = np.clip(weights[active_mask], clip_min, clip_max)

    return torch.tensor(weights, dtype=torch.float32)


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    ignore_index: int | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    target_oh = F.one_hot(target.clamp(0, num_classes - 1), num_classes).permute(0, 3, 1, 2).float()

    if ignore_index is not None:
        valid = (target != ignore_index).unsqueeze(1).float()
        probs = probs * valid
        target_oh = target_oh * valid

    dims = (0, 2, 3)
    intersection = (probs * target_oh).sum(dims)
    cardinality = probs.sum(dims) + target_oh.sum(dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    loss_per_class = 1.0 - dice

    if class_weights is not None:
        w = class_weights.to(logits.device)
        # не учитываем классы с нулевым весом
        mask = w > 0
        if mask.any():
            return (loss_per_class[mask] * w[mask]).sum() / w[mask].sum()
    return loss_per_class.mean()


def lovasz_softmax_flat(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Упрощённый Lovász-Softmax по классам присутствующим в батче."""
    C = probs.shape[1]
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        class_pred = probs[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        grad = _lovasz_grad(fg_sorted)
        losses.append(torch.dot(F.relu(errors_sorted), grad))
    if not losses:
        return probs.new_zeros(())
    return torch.stack(losses).mean()


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1.0 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union
    if len(jaccard) > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_softmax_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    b, c, h, w = probs.shape
    probs = probs.permute(0, 2, 3, 1).reshape(-1, c)
    labels = target.view(-1)
    return lovasz_softmax_flat(probs, labels)


class CombinedSegLoss(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        lovasz_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.lovasz_weight = lovasz_weight
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(16),
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        w = self.class_weights.to(logits.device)
        ce = F.cross_entropy(logits, target, weight=w)
        dice = soft_dice_loss(logits, target, class_weights=w)
        total = self.ce_weight * ce + self.dice_weight * dice
        out = {"loss": total, "ce": ce.detach(), "dice": dice.detach()}
        if self.lovasz_weight > 0:
            lov = lovasz_softmax_loss(logits, target)
            total = total + self.lovasz_weight * lov
            out["loss"] = total
            out["lovasz"] = lov.detach()
        return out


def build_loss(cfg: dict[str, Any]) -> CombinedSegLoss:
    lcfg = cfg["loss"]
    boost = lcfg.get("class_weight_boost") or {}
    weights = compute_class_weights(
        class_balance_json=cfg["data"]["class_balance_json"],
        classes=cfg["classes"],
        empty_names=cfg.get("empty_classes", []),
        mode=lcfg.get("class_weight_mode", "effective_number"),
        beta=float(lcfg.get("effective_beta", 0.9999)),
        clip_min=float(lcfg.get("weight_clip_min", 0.25)),
        clip_max=float(lcfg.get("weight_clip_max", 10.0)),
        empty_weight=float(lcfg.get("empty_class_weight", 0.0)),
        num_classes=int(cfg["model"]["classes"]),
        class_weight_boost={str(k): float(v) for k, v in boost.items()},
    )
    return CombinedSegLoss(
        class_weights=weights,
        ce_weight=float(lcfg.get("ce_weight", 0.5)),
        dice_weight=float(lcfg.get("dice_weight", 0.5)),
        lovasz_weight=float(lcfg.get("lovasz_weight", 0.0)),
    )
