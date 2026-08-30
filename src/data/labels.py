"""Свёртка многоканальных масок в индексную по label_priority."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def class_name_to_index(classes: Mapping[int, str]) -> dict[str, int]:
    return {name: int(idx) for idx, name in classes.items()}


def build_priority_indices(
    label_priority: Sequence[str],
    classes: Mapping[int, str],
) -> list[int]:
    name_to_idx = class_name_to_index(classes)
    return [name_to_idx[n] for n in label_priority if n in name_to_idx]


def channels_to_index(
    channels: np.ndarray,
    priority_indices: Sequence[int],
) -> np.ndarray:
    """
    channels: [C, H, W] uint8/bool с перекрытиями.
    Возвращает индексную маску [H, W] uint8.

    Логика идентична пайплайну датасета (compose_label_map):
    рисуем от низкого приоритета к высокому; background (0) не затирает.
    """
    if channels.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {channels.shape}")
    _, h, w = channels.shape
    label = np.zeros((h, w), dtype=np.uint8)
    for cls_idx in reversed(list(priority_indices)):
        if cls_idx == 0:
            continue
        label[channels[cls_idx] > 0] = np.uint8(cls_idx)
    return label


def empty_class_indices(
    empty_names: Sequence[str],
    classes: Mapping[int, str],
) -> list[int]:
    name_to_idx = class_name_to_index(classes)
    return [name_to_idx[n] for n in empty_names if n in name_to_idx]


# Палитра для визуализации (RGB), индекс = класс.
CLASS_COLORS: list[tuple[int, int, int]] = [
    (0, 0, 0),  # background
    (255, 255, 255),  # forest_run
    (255, 228, 120),  # open_land
    (210, 230, 140),  # semi_open
    (120, 190, 90),  # green_slow
    (40, 120, 40),  # green_hard
    (60, 120, 220),  # water
    (140, 100, 60),  # marsh
    (80, 80, 80),  # road
    (140, 90, 40),  # track_path
    (190, 190, 190),  # paved_settlement
    (180, 90, 40),  # contours
    (220, 40, 200),  # impassable
    (40, 40, 40),  # footpath
    (0, 160, 160),  # cycle_path
    (255, 80, 180),  # control_point (розовый КП)
]


def colorize_label(label: np.ndarray) -> np.ndarray:
    h, w = label.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(CLASS_COLORS):
        out[label == i] = color
    return out


def overlay_prediction(
    image: np.ndarray,
    label: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    color = colorize_label(label).astype(np.float32)
    base = image.astype(np.float32)
    mask = (label > 0)[..., None]
    blended = base * (1.0 - alpha * mask) + color * (alpha * mask)
    return np.clip(blended, 0, 255).astype(np.uint8)
