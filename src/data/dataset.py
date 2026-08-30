"""Dataset и DataLoader для тайлов ориентировочных карт."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.labels import build_priority_indices, channels_to_index
from src.data.transforms import build_eval_transforms, build_train_transforms

Image.MAX_IMAGE_PIXELS = None


@dataclass
class SampleRecord:
    map_name: str
    sample: str
    split: str
    kind: str
    image: Path
    channels: Path | None
    label: Path | None
    fg_ratio: float


def resolve_dataset_path(raw: str, dataset_root: Path) -> Path:
    """Переносит абсолютные пути индекса в локальный dataset_root."""
    p = Path(raw)
    parts = list(p.parts)
    if "dataset" in parts:
        idx = parts.index("dataset")
        rel = Path(*parts[idx + 1 :])
        return dataset_root / rel
    if not p.is_absolute():
        return dataset_root / p
    return p


def load_index(
    index_csv: str | Path,
    dataset_root: str | Path,
    split: str | None = None,
    kinds: Sequence[str] | None = None,
    min_fg_ratio: float | None = None,
) -> list[SampleRecord]:
    dataset_root = Path(dataset_root)
    rows: list[SampleRecord] = []
    with Path(index_csv).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if split is not None and r["split"] != split:
                continue
            if kinds is not None and r["kind"] not in kinds:
                continue
            fg = float(r["fg_ratio"]) if r.get("fg_ratio") else 0.0
            if min_fg_ratio is not None and fg < min_fg_ratio:
                continue
            channels = (r.get("channels") or "").strip()
            label = (r.get("label") or "").strip()
            rows.append(
                SampleRecord(
                    map_name=r["map"],
                    sample=r["sample"],
                    split=r["split"],
                    kind=r["kind"],
                    image=resolve_dataset_path(r["image"], dataset_root),
                    channels=resolve_dataset_path(channels, dataset_root) if channels else None,
                    label=resolve_dataset_path(label, dataset_root) if label else None,
                    fg_ratio=fg,
                )
            )
    return rows


class OrienteeringSegDataset(Dataset):
    def __init__(
        self,
        records: list[SampleRecord],
        classes: dict[int, str],
        label_priority: Sequence[str],
        transform: Any | None = None,
    ) -> None:
        self.records = records
        self.priority = build_priority_indices(label_priority, classes)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def _load_mask(self, rec: SampleRecord) -> np.ndarray:
        # kind=aug (legacy): готовая индексная label; иначе channels.npy → index
        if rec.label is not None and rec.label.exists() and (
            rec.kind == "aug" or rec.channels is None
        ):
            return np.array(Image.open(rec.label), dtype=np.uint8)
        if rec.channels is None or not rec.channels.exists():
            raise FileNotFoundError(f"Missing channels for tile: {rec.sample}")
        channels = np.load(rec.channels)
        return channels_to_index(channels, self.priority)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        image = np.array(Image.open(rec.image).convert("RGB"), dtype=np.uint8)
        mask = self._load_mask(rec)

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image_t = out["image"]
            mask_t = out["mask"].long()
        else:
            image_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask_t = torch.from_numpy(mask.astype(np.int64))

        return {
            "image": image_t,
            "mask": mask_t,
            "map": rec.map_name,
            "sample": rec.sample,
            "kind": rec.kind,
            "fg_ratio": rec.fg_ratio,
        }


def create_dataloaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    dataset_root = Path(data_cfg["dataset_root"])
    index_csv = data_cfg["index_csv"]
    classes = cfg["classes"]
    priority = cfg["label_priority"]

    train_records = load_index(
        index_csv,
        dataset_root,
        split="train",
        kinds=data_cfg.get("train_kinds", ["tile"]),
    )
    val_records = load_index(
        index_csv,
        dataset_root,
        split="val",
        kinds=data_cfg.get("val_kinds", ["tile"]),
    )
    test_records = load_index(
        index_csv,
        dataset_root,
        split="test",
        kinds=data_cfg.get("test_kinds", ["tile"]),
    )

    train_ds = OrienteeringSegDataset(
        train_records,
        classes,
        priority,
        transform=build_train_transforms(cfg.get("augment", {}), data_cfg.get("image_size", 512)),
    )
    val_ds = OrienteeringSegDataset(
        val_records,
        classes,
        priority,
        transform=build_eval_transforms(),
    )
    test_ds = OrienteeringSegDataset(
        test_records,
        classes,
        priority,
        transform=build_eval_transforms(),
    )

    nw = int(data_cfg.get("num_workers", 4))
    bs = int(cfg["train"]["batch_size"])
    pin = bool(data_cfg.get("pin_memory", torch.cuda.is_available()))
    persistent = bool(data_cfg.get("persistent_workers", nw > 0))
    prefetch = int(data_cfg.get("prefetch_factor", 2)) if nw > 0 else None

    common: dict[str, Any] = {
        "batch_size": bs,
        "num_workers": nw,
        "pin_memory": pin,
    }
    if nw > 0:
        common["persistent_workers"] = persistent
        common["prefetch_factor"] = prefetch

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        **common,
    )
    return train_loader, val_loader, test_loader
