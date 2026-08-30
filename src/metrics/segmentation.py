"""Метрики сегментации: per-class IoU, mIoU, Dice, pixel accuracy."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from src.data.labels import class_name_to_index, empty_class_indices


class ConfusionMeter:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.conf = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def reset(self) -> None:
        self.conf.zero_()

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """preds/target: [B,H,W] long."""
        preds = preds.detach().cpu().reshape(-1)
        target = target.detach().cpu().reshape(-1)
        mask = (target >= 0) & (target < self.num_classes)
        preds = preds[mask]
        target = target[mask]
        idx = target * self.num_classes + preds
        bc = torch.bincount(idx, minlength=self.num_classes**2)
        self.conf += bc.reshape(self.num_classes, self.num_classes)

    def compute(
        self,
        classes: Mapping[int, str],
        exclude_background: bool = True,
        exclude_empty: Sequence[str] | None = None,
        critical: Sequence[str] | None = None,
    ) -> dict:
        conf = self.conf.float()
        tp = conf.diag()
        fp = conf.sum(0) - tp
        fn = conf.sum(1) - tp
        iou = tp / (tp + fp + fn).clamp_min(1e-6)
        dice = (2 * tp) / (2 * tp + fp + fn).clamp_min(1e-6)
        acc = tp.sum() / conf.sum().clamp_min(1)

        empty_idx = set(empty_class_indices(exclude_empty or [], classes))
        include = []
        for i in range(self.num_classes):
            if exclude_background and i == 0:
                continue
            if i in empty_idx:
                continue
            include.append(i)

        miou = iou[include].mean().item() if include else 0.0
        mdice = dice[include].mean().item() if include else 0.0

        per_class = {
            classes[i]: {"iou": float(iou[i]), "dice": float(dice[i])}
            for i in range(self.num_classes)
            if i in classes
        }

        name_to_idx = class_name_to_index(classes)
        critical_iou = {}
        if critical:
            for name in critical:
                if name in name_to_idx:
                    critical_iou[name] = float(iou[name_to_idx[name]])

        return {
            "mIoU": miou,
            "mDice": mdice,
            "pixel_acc": float(acc),
            "per_class": per_class,
            "critical_iou": critical_iou,
            "iou_vector": iou.tolist(),
        }
