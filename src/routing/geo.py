"""Пиксели ↔ ячейки, meta.json, метры."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

# дюйм → метр; resolution = scale * INCH_M / dpi
INCH_M: float = 0.0254
DEFAULT_RENDER_DPI: int = 300


def meters_per_pixel_from_scale(
    scale: int | float,
    dpi: int | float = DEFAULT_RENDER_DPI,
) -> float:
    """Метры на местности на 1 пиксель при масштабе 1:scale и DPI рендера/печати."""
    if scale <= 0 or dpi <= 0:
        raise ValueError(f"scale и dpi должны быть > 0, получено scale={scale}, dpi={dpi}")
    return float(scale) * INCH_M / float(dpi)


@dataclass(frozen=True)
class MapMeta:
    """Метаданные карты для перевода пикселей в метры."""

    resolution_m_per_px: float | None
    width: int | None = None
    height: int | None = None
    map_name: str | None = None
    scale: int | None = None
    dpi: int | None = None
    raw: dict[str, Any] | None = None

    @property
    def has_scale(self) -> bool:
        return self.resolution_m_per_px is not None and self.resolution_m_per_px > 0

    def meters_per_pixel(self) -> float:
        if self.has_scale:
            return float(self.resolution_m_per_px)
        return 1.0

    def with_resolution(
        self,
        resolution_m_per_px: float,
        *,
        scale: int | None = None,
        dpi: int | None = None,
    ) -> MapMeta:
        """Копия с переопределённым м/px (когда meta/OCD врёт, а на карте другой масштаб)."""
        return replace(
            self,
            resolution_m_per_px=float(resolution_m_per_px),
            scale=scale if scale is not None else self.scale,
            dpi=dpi if dpi is not None else self.dpi,
        )

    def warn_if_no_scale(self) -> None:
        if not self.has_scale:
            warnings.warn(
                "resolution_m_per_px не задан: дистанция в «пикселях», время условное.",
                UserWarning,
                stacklevel=2,
            )


def load_meta(path: str | Path | None) -> MapMeta:
    """Загрузка meta.json. Если path=None — без масштаба."""
    if path is None:
        meta = MapMeta(resolution_m_per_px=None)
        meta.warn_if_no_scale()
        return meta

    path = Path(path)
    if not path.exists():
        warnings.warn(f"meta.json не найден: {path}", UserWarning, stacklevel=2)
        meta = MapMeta(resolution_m_per_px=None)
        meta.warn_if_no_scale()
        return meta

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    res = raw.get("resolution_m_per_px")
    scale = int(raw["scale"]) if raw.get("scale") is not None else None
    dpi = int(raw["dpi"]) if raw.get("dpi") is not None else None
    if res is None and scale is not None:
        res = meters_per_pixel_from_scale(scale, dpi or DEFAULT_RENDER_DPI)
    meta = MapMeta(
        resolution_m_per_px=float(res) if res is not None else None,
        width=int(raw["width"]) if "width" in raw else None,
        height=int(raw["height"]) if "height" in raw else None,
        map_name=raw.get("map_name"),
        scale=scale,
        dpi=dpi,
        raw=raw,
    )
    meta.warn_if_no_scale()
    return meta


def pixel_to_cell(x: float, y: float, cell_size: int) -> tuple[int, int]:
    """(x,y) пиксель изображения → (col, row) ячейки."""
    return int(x) // cell_size, int(y) // cell_size


def cell_to_pixel_center(col: int, row: int, cell_size: int) -> tuple[float, float]:
    """Центр ячейки в координатах пикселей изображения."""
    return (col + 0.5) * cell_size, (row + 0.5) * cell_size


def cells_to_pixel_path(
    cells: list[tuple[int, int]],
    cell_size: int,
) -> list[tuple[float, float]]:
    """Список (col, row) → список (x, y) центров."""
    return [cell_to_pixel_center(c, r, cell_size) for c, r in cells]


def edge_length_m(
    dc: int,
    dr: int,
    cell_size: int,
    resolution_m_per_px: float,
) -> float:
    """Длина шага между соседними ячейками в метрах (или условных ед.)."""
    step_px = cell_size * (1.41421356237 if (dc != 0 and dr != 0) else 1.0)
    return step_px * resolution_m_per_px
