"""Конфиг маршрутизации: классы, скорости, цвета маршрутов."""

from __future__ import annotations

from copy import deepcopy

from src.data.labels import CLASS_COLORS
from src.postprocess import DEFAULT_COST

# Индексы классов (синхронно с configs/dataset_classes.yaml).
CLASS_NAMES: dict[int, str] = {
    0: "background",
    1: "forest_run",
    2: "open_land",
    3: "semi_open",
    4: "green_slow",
    5: "green_hard",
    6: "water",
    7: "marsh",
    8: "road",
    9: "track_path",
    10: "paved_settlement",
    11: "contours",
    12: "impassable",
    13: "footpath",
    14: "cycle_path",
    15: "control_point",
}

CLASS_NAME_TO_IDX: dict[str, int] = {v: k for k, v in CLASS_NAMES.items()}

CLASS_LABELS_RU: dict[str, str] = {
    "background": "фон / неразмечено",
    "forest_run": "лес (бег)",
    "open_land": "открытое",
    "semi_open": "полуоткрытое",
    "green_slow": "зелёнка (медленно)",
    "green_hard": "густая зелёнка",
    "water": "вода",
    "marsh": "болото",
    "road": "дорога",
    "track_path": "тропа / просека",
    "paved_settlement": "асфальт / постройки",
    "contours": "горизонтали",
    "impassable": "непроходимо",
    "footpath": "тропа",
    "cycle_path": "велодорожка",
    "control_point": "КП / дистанция",
}

# Относительные скорости бега, м/с (условные). 0 = непроходимо.
# Подобраны так, чтобы 1/speed ≈ DEFAULT_COST при базе ~1 м/с на open_land.
DEFAULT_SPEEDS_MPS: dict[str, float] = {
    "background": 0.20,
    "forest_run": 0.83,
    "open_land": 1.00,
    "semi_open": 0.77,
    "green_slow": 0.40,
    "green_hard": 0.125,
    "water": 0.0,
    "marsh": 0.17,
    "road": 1.67,
    "track_path": 1.43,
    "paved_settlement": 1.25,
    "contours": 0.91,  # overlay: скорость почти не используется (см. contour_penalty)
    "impassable": 0.0,
    "footpath": 1.33,
    "cycle_path": 1.43,
    "control_point": 1.00,  # overlay; обычно подменяется соседями
}

# Линейные «быстрые» классы — при даунсемплинге приоритет над зонами.
FAST_LINEAR_CLASSES: tuple[str, ...] = (
    "road",
    "track_path",
    "footpath",
    "cycle_path",
)

IMPASSABLE_CLASSES: tuple[str, ...] = ("water", "impassable")

# Классы, которые не должны сами задавать местность (линии / разметка поверх).
OVERLAY_CLASSES: tuple[str, ...] = ("contours", "control_point")

# Штраф рельефа: time *= (1 + CONTOUR_PENALTY * density),
# density — доля пикселей contours в ячейке [0, 1].
# 0 = выкл.; 2.0 → при density=0.25 время ×1.5.
DEFAULT_CONTOUR_PENALTY: float = 2.0

# Цвета линий маршрутов на визуализации (RGB 0–1).
ROUTE_COLORS: list[tuple[float, float, float]] = [
    (0.90, 0.10, 0.15),
    (0.10, 0.35, 0.95),
    (0.05, 0.70, 0.25),
    (0.95, 0.55, 0.05),
    (0.60, 0.15, 0.75),
]

# Порог стоимости: выше — непроходимо.
IMPASSABLE_COST: float = 1e8

DEFAULT_CELL_SIZE: int = 4
DEFAULT_K_ROUTES: int = 3
DEFAULT_PENALTY_FACTOR: float = 3.0
DEFAULT_PENALTY_WIDTH: int = 2
DEFAULT_SIMILARITY_THRESHOLD: float = 0.55
DEFAULT_MIN_SEGMENT_M: float = 8.0


def speeds_from_costs(cost_mapping: dict[str, float] | None = None, base_mps: float = 1.0) -> dict[str, float]:
    """Обратное преобразование DEFAULT_COST → скорости (м/с)."""
    mapping = cost_mapping or DEFAULT_COST
    out: dict[str, float] = {}
    for name, cost in mapping.items():
        if cost >= IMPASSABLE_COST / 10:
            out[name] = 0.0
        else:
            out[name] = base_mps / max(float(cost), 1e-6)
    return out


def copy_speeds(speeds: dict[str, float] | None = None) -> dict[str, float]:
    return deepcopy(speeds or DEFAULT_SPEEDS_MPS)


__all__ = [
    "CLASS_COLORS",
    "CLASS_LABELS_RU",
    "CLASS_NAMES",
    "CLASS_NAME_TO_IDX",
    "DEFAULT_CELL_SIZE",
    "DEFAULT_CONTOUR_PENALTY",
    "DEFAULT_COST",
    "DEFAULT_K_ROUTES",
    "DEFAULT_MIN_SEGMENT_M",
    "DEFAULT_PENALTY_FACTOR",
    "DEFAULT_PENALTY_WIDTH",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "DEFAULT_SPEEDS_MPS",
    "FAST_LINEAR_CLASSES",
    "IMPASSABLE_CLASSES",
    "IMPASSABLE_COST",
    "OVERLAY_CLASSES",
    "ROUTE_COLORS",
    "copy_speeds",
    "speeds_from_costs",
]
