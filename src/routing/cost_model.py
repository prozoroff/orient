"""Модель стоимости: класс → скорость → cost-грид."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import ndimage

from src.routing.config import (
    CLASS_NAME_TO_IDX,
    CLASS_NAMES,
    DEFAULT_CELL_SIZE,
    DEFAULT_CONTOUR_PENALTY,
    DEFAULT_SPEEDS_MPS,
    FAST_LINEAR_CLASSES,
    IMPASSABLE_COST,
    OVERLAY_CLASSES,
    copy_speeds,
)
from src.routing.geo import MapMeta


@dataclass
class CostGrid:
    """Даунсемплированная сетка для поиска пути."""

    # время прохождения ячейки (секунды) для шага длины cell_m; inf ≈ стена
    time_grid: np.ndarray  # float32 [H_cells, W_cells]
    class_grid: np.ndarray  # uint8 [H_cells, W_cells] — класс ячейки
    cell_size: int
    cell_m: float  # длина стороны ячейки в метрах (или усл. ед.)
    resolution_m_per_px: float
    speeds: dict[str, float] = field(default_factory=dict)
    passable: np.ndarray | None = None  # bool, кэш
    # доля пикселей горизонталей в ячейке [0, 1]; None если штраф выключен
    contour_density: np.ndarray | None = None
    contour_penalty: float = 0.0

    def __post_init__(self) -> None:
        if self.passable is None:
            self.passable = np.isfinite(self.time_grid) & (self.time_grid < IMPASSABLE_COST)

    @property
    def shape(self) -> tuple[int, int]:
        return self.time_grid.shape  # (rows, cols)

    @property
    def max_speed(self) -> float:
        vals = [s for s in self.speeds.values() if s > 0]
        return max(vals) if vals else 1.0


def _speeds_to_index_array(speeds: dict[str, float], n_classes: int | None = None) -> np.ndarray:
    n = n_classes if n_classes is not None else (max(CLASS_NAMES) + 1)
    arr = np.full(n, 0.2, dtype=np.float32)
    for name, spd in speeds.items():
        idx = CLASS_NAME_TO_IDX.get(name)
        if idx is not None and idx < n:
            arr[idx] = float(spd)
    return arr


def _resolve_overlays(label: np.ndarray, speeds: dict[str, float]) -> np.ndarray:
    """
    Contours и подобные overlay-классы: подменить на моду соседних
    непрозрачных классов (или fallback-скорость через класс open_land).
    Возвращает «эффективную» индексную маску той же формы.
    """
    out = label.copy()
    overlay_ids = {CLASS_NAME_TO_IDX[n] for n in OVERLAY_CLASSES if n in CLASS_NAME_TO_IDX}
    if not overlay_ids:
        return out

    mask = np.isin(out, list(overlay_ids))
    if not mask.any():
        return out

    # Итеративно заполняем overlay ближайшим непрозрачным соседом.
    filled = out.copy()
    filled[mask] = 255  # sentinel
    # distance transform + nearest label among non-overlay
    valid = ~mask
    if not valid.any():
        fallback = CLASS_NAME_TO_IDX.get("open_land", 2)
        out[mask] = fallback
        return out

    # nearest valid pixel for each overlay pixel
    _, indices = ndimage.distance_transform_edt(~valid, return_distances=True, return_indices=True)
    nearest = filled[indices[0], indices[1]]
    out[mask] = nearest[mask]
    # если всё ещё sentinel — open_land
    still = out == 255
    if still.any():
        out[still] = CLASS_NAME_TO_IDX.get("open_land", 2)
    return out.astype(np.uint8)


def _block_class(block: np.ndarray, fast_ids: set[int], impassable_ids: set[int]) -> int:
    """
    Класс ячейки даунсемпла:
    - если в блоке есть быстрая линейка → лучшая (min index among fast? нет — max speed later)
      берём первый по приоритету: road > track > footpath > cycle (по порядку FAST_LINEAR)
    - иначе если весь блок непроходим → impassable
    - иначе мода среди проходимых
    """
    flat = block.ravel()
    # приоритет быстрых линеек
    for fid in fast_ids:  # уже в порядке приоритета
        if np.any(flat == fid):
            return int(fid)

    passable = flat[~np.isin(flat, list(impassable_ids))]
    if passable.size == 0:
        # любой impassable в блоке
        return int(flat[0])

    # мода
    vals, counts = np.unique(passable, return_counts=True)
    return int(vals[np.argmax(counts)])


def downsample_label(
    label: np.ndarray,
    cell_size: int,
    fast_linear: Sequence[str] = FAST_LINEAR_CLASSES,
) -> np.ndarray:
    """Индексная маска [H,W] → класс-грид [h,w] с приоритетом дорог."""
    if cell_size <= 1:
        return label.copy()

    h, w = label.shape
    rows = h // cell_size
    cols = w // cell_size
    cropped = label[: rows * cell_size, : cols * cell_size]
    blocks = cropped.reshape(rows, cell_size, cols, cell_size).transpose(0, 2, 1, 3)

    fast_ids = [CLASS_NAME_TO_IDX[n] for n in fast_linear if n in CLASS_NAME_TO_IDX]
    impassable_ids = {
        CLASS_NAME_TO_IDX[n]
        for n in ("water", "impassable")
        if n in CLASS_NAME_TO_IDX
    }

    out = np.zeros((rows, cols), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            out[r, c] = _block_class(blocks[r, c], fast_ids, impassable_ids)
    return out


def downsample_label_fast(
    label: np.ndarray,
    cell_size: int,
    fast_linear: Sequence[str] = FAST_LINEAR_CLASSES,
) -> np.ndarray:
    """Векторизованный даунсемпл (быстрее наивного двойного цикла)."""
    if cell_size <= 1:
        return label.copy()

    h, w = label.shape
    rows = h // cell_size
    cols = w // cell_size
    cropped = label[: rows * cell_size, : cols * cell_size]

    fast_ids = [CLASS_NAME_TO_IDX[n] for n in fast_linear if n in CLASS_NAME_TO_IDX]
    impassable_ids = np.array(
        [CLASS_NAME_TO_IDX[n] for n in ("water", "impassable") if n in CLASS_NAME_TO_IDX],
        dtype=np.uint8,
    )

    # приоритет: если в блоке есть fast-класс — ставим его
    out = np.zeros((rows, cols), dtype=np.uint8)
    assigned = np.zeros((rows, cols), dtype=bool)

    for fid in fast_ids:
        has = (
            (cropped == fid)
            .reshape(rows, cell_size, cols, cell_size)
            .any(axis=(1, 3))
        )
        take = has & ~assigned
        out[take] = fid
        assigned |= take

    # мода по блокам для оставшихся: через bincount на каждом блоке дорого;
    # берём центральный пиксель как приближение + min-cost позже не нужен
    # Более точно: reshape + mode через unique по axis — медленно.
    # Компромисс: пиксель с минимальным cost среди проходимых ≈ max speed.
    # Здесь: берём наиболее частый через np.bincount по вытянутым блокам батчами.
    remaining = ~assigned
    if remaining.any():
        blocks = cropped.reshape(rows, cell_size, cols, cell_size)
        for r, c in zip(*np.where(remaining), strict=False):
            block = blocks[r, :, c, :].ravel()
            if impassable_ids.size:
                passable = block[~np.isin(block, impassable_ids)]
            else:
                passable = block
            if passable.size == 0:
                out[r, c] = int(block[0])
            else:
                counts = np.bincount(passable, minlength=max(CLASS_NAMES) + 1)
                out[r, c] = int(np.argmax(counts))
    return out


def _contour_density(label: np.ndarray, cell_size: int) -> np.ndarray:
    """Доля пикселей класса contours в каждой ячейке даунсемпла, shape [h, w]."""
    contour_id = CLASS_NAME_TO_IDX.get("contours")
    if contour_id is None:
        h, w = label.shape
        if cell_size <= 1:
            return np.zeros((h, w), dtype=np.float32)
        return np.zeros((h // cell_size, w // cell_size), dtype=np.float32)

    mask = (label == contour_id).astype(np.float32)
    if cell_size <= 1:
        return mask

    h, w = label.shape
    rows = h // cell_size
    cols = w // cell_size
    if rows == 0 or cols == 0:
        return np.zeros((max(rows, 0), max(cols, 0)), dtype=np.float32)
    cropped = mask[: rows * cell_size, : cols * cell_size]
    return cropped.reshape(rows, cell_size, cols, cell_size).mean(axis=(1, 3)).astype(np.float32)


def build_cost_grid(
    label: np.ndarray,
    meta: MapMeta,
    speeds: dict[str, float] | None = None,
    cell_size: int = DEFAULT_CELL_SIZE,
    resolve_overlays: bool = True,
    contour_penalty: float = DEFAULT_CONTOUR_PENALTY,
) -> CostGrid:
    """
    Сегментация → CostGrid.

    time_grid[i,j] = cell_m / speed(class)  — время на ортогональный шаг
    (диагональ учитывается в pathfinding через √2).

    Contours (горизонтали) — overlay: класс местности берётся у соседей, а крутизна
    моделируется множителем ``1 + contour_penalty * density``, где density — доля
    пикселей горизонталей в ячейке.
    """
    speeds = copy_speeds(speeds)
    res = meta.meters_per_pixel()
    cell_m = cell_size * res

    effective = _resolve_overlays(label, speeds) if resolve_overlays else label.copy()
    class_grid = downsample_label_fast(effective, cell_size)

    speed_lut = _speeds_to_index_array(speeds)
    spd = speed_lut[class_grid]
    time_grid = np.full(class_grid.shape, np.inf, dtype=np.float32)
    passable = spd > 0
    time_grid[passable] = (cell_m / spd[passable]).astype(np.float32)

    density: np.ndarray | None = None
    penalty = float(contour_penalty)
    if penalty > 0:
        density = _contour_density(label, cell_size)
        if density.shape != time_grid.shape:
            # на всякий случай обрежем/паддим до размера class_grid
            rh, rw = time_grid.shape
            fixed = np.zeros((rh, rw), dtype=np.float32)
            h = min(rh, density.shape[0])
            w = min(rw, density.shape[1])
            fixed[:h, :w] = density[:h, :w]
            density = fixed
        mult = (1.0 + penalty * density).astype(np.float32)
        time_grid = np.where(np.isfinite(time_grid), time_grid * mult, time_grid)

    return CostGrid(
        time_grid=time_grid,
        class_grid=class_grid,
        cell_size=cell_size,
        cell_m=cell_m,
        resolution_m_per_px=res,
        speeds=speeds,
        contour_density=density,
        contour_penalty=penalty if penalty > 0 else 0.0,
    )


def class_name_at(grid: CostGrid, col: int, row: int) -> str:
    idx = int(grid.class_grid[row, col])
    return CLASS_NAMES.get(idx, f"class_{idx}")
