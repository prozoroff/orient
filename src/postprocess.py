"""Пост-обработка масок и экспорт карты стоимости проходимости."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.data.labels import class_name_to_index

# Класс → стоимость для будущего A*/Dijkstra.
# 1e9 ≈ непроходимо.
DEFAULT_COST: dict[str, float] = {
    "background": 5.0,
    "forest_run": 1.2,
    "open_land": 1.0,
    "semi_open": 1.3,
    "green_slow": 2.5,
    "green_hard": 8.0,
    "water": 1e9,
    "marsh": 6.0,
    "road": 0.6,
    "track_path": 0.7,
    "paved_settlement": 0.8,
    "contours": 1.1,
    "impassable": 1e9,
    "footpath": 0.75,
    "cycle_path": 0.7,
    "control_point": 1.0,
}


def morphological_close_linear(
    label: np.ndarray,
    cfg: dict[str, Any],
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Аккуратное морфологическое замыкание только по линейным классам.
    Зоны (лес/поле/вода) не раздуваем.
    """
    name_to_idx = class_name_to_index(cfg["classes"])
    linear_names = cfg.get("linear_classes", [])
    out = label.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    for name in linear_names:
        if name not in name_to_idx:
            continue
        idx = name_to_idx[name]
        binary = (label == idx).astype(np.uint8) * 255
        if binary.sum() == 0:
            continue
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
        # ставим класс только там, где closing добавил пиксели и было background/конкурирующий линейный?
        # безопаснее: заполняем только бывший background
        add = (closed > 0) & (out == 0)
        out[add] = idx
    return out


def export_cost_map(
    label: np.ndarray,
    cfg: dict[str, Any],
    cost_mapping: dict[str, float] | None = None,
) -> np.ndarray:
    """Индексная маска → float32 карта стоимости [H,W]."""
    mapping = cost_mapping or DEFAULT_COST
    name_to_idx = class_name_to_index(cfg["classes"])
    cost = np.full(label.shape, 5.0, dtype=np.float32)
    for name, value in mapping.items():
        if name not in name_to_idx:
            continue
        cost[label == name_to_idx[name]] = float(value)
    return cost
