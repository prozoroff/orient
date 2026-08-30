"""Соседи сетки «на лету» и привязка кликов к проходимым ячейкам."""

from __future__ import annotations

import warnings
from typing import Iterator

import numpy as np
from scipy import ndimage

from src.routing.cost_model import CostGrid
from src.routing.geo import pixel_to_cell

# 8-связность: (dcol, drow, is_diagonal)
NEIGHBORS_8: tuple[tuple[int, int, bool], ...] = (
    (1, 0, False),
    (-1, 0, False),
    (0, 1, False),
    (0, -1, False),
    (1, 1, True),
    (1, -1, True),
    (-1, 1, True),
    (-1, -1, True),
)

SQRT2 = 1.41421356237


def in_bounds(col: int, row: int, shape: tuple[int, int]) -> bool:
    rows, cols = shape
    return 0 <= row < rows and 0 <= col < cols


def iter_neighbors(
    col: int,
    row: int,
    grid: CostGrid,
    *,
    prevent_corner_cut: bool = True,
) -> Iterator[tuple[int, int, float]]:
    """
    Yields (ncol, nrow, edge_time_seconds).

    Вес ребра = среднее time входной и выходной ячеек × (√2 если диагональ).
    """
    rows, cols = grid.shape
    t0 = float(grid.time_grid[row, col])
    if not np.isfinite(t0) or not grid.passable[row, col]:
        return

    for dc, dr, diag in NEIGHBORS_8:
        nc, nr = col + dc, row + dr
        if not (0 <= nr < rows and 0 <= nc < cols):
            continue
        if not grid.passable[nr, nc]:
            continue

        if prevent_corner_cut and diag:
            # запрет срезания угла между двумя стенами
            if not grid.passable[row, nc] and not grid.passable[nr, col]:
                continue

        t1 = float(grid.time_grid[nr, nc])
        edge = 0.5 * (t0 + t1)
        if diag:
            edge *= SQRT2
        yield nc, nr, edge


def snap_to_passable(
    x: float,
    y: float,
    grid: CostGrid,
    *,
    max_radius_cells: int | None = None,
) -> tuple[tuple[int, int], tuple[int, int], bool]:
    """
    Пиксель (x,y) → ближайшая проходимая ячейка.

    Returns:
        (requested_cell, snapped_cell, was_snapped)
        cell = (col, row)
    """
    col, row = pixel_to_cell(x, y, grid.cell_size)
    rows, cols = grid.shape
    col = int(np.clip(col, 0, cols - 1))
    row = int(np.clip(row, 0, rows - 1))
    requested = (col, row)

    if grid.passable[row, col]:
        return requested, requested, False

    # distance transform to nearest passable
    # EDT на ~passable даёт расстояние до ближайшего True в passable
    if not grid.passable.any():
        raise RuntimeError("Нет проходимых ячеек на карте.")

    _, indices = ndimage.distance_transform_edt(
        ~grid.passable,
        return_distances=True,
        return_indices=True,
    )
    nr = int(indices[0, row, col])
    nc = int(indices[1, row, col])

    if max_radius_cells is not None:
        dist = abs(nc - col) + abs(nr - row)  # грубо
        # евклидово в ячейках
        edist = ((nc - col) ** 2 + (nr - row) ** 2) ** 0.5
        if edist > max_radius_cells:
            warnings.warn(
                f"Ближайшая проходимая ячейка слишком далеко ({edist:.0f} cells) "
                f"от ({col},{row}).",
                UserWarning,
                stacklevel=2,
            )

    snapped = (nc, nr)
    if snapped != requested:
        warnings.warn(
            f"Точка ({x:.0f},{y:.0f}) в непроходимой зоне; "
            f"привязано к ячейке {snapped} (было {requested}).",
            UserWarning,
            stacklevel=2,
        )
    return requested, snapped, True
