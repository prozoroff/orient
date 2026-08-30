"""Матрица времён между стартом и КП через Dijkstra на CostGrid."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from src.routing.cost_model import CostGrid
from src.routing.geo import cells_to_pixel_path
from src.routing.graph import iter_neighbors, snap_to_passable
from src.routing.pathfinding import Route, SearchTrace, _path_distance_m


def dijkstra_times(
    grid: CostGrid,
    start_cell: tuple[int, int],
) -> np.ndarray:
    """
    Времена (сек) от start_cell до всех ячеек. Недостижимые = inf.

    shape = grid.shape (rows, cols).
    """
    times, _parents, _visit, _n = _dijkstra_core(grid, start_cell, record_visit=False)
    return times


def _dijkstra_core(
    grid: CostGrid,
    start_cell: tuple[int, int],
    *,
    record_visit: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
    """
    Returns:
        times (H,W), parents (H,W,2) int32 (-1=нет), visit_order|None, n_closed
    """
    rows, cols = grid.shape
    sc, sr = start_cell
    times = np.full((rows, cols), np.inf, dtype=np.float64)
    parents = np.full((rows, cols, 2), -1, dtype=np.int32)
    visit_order: np.ndarray | None = (
        np.full((rows, cols), -1, dtype=np.int32) if record_visit else None
    )
    n_closed = 0

    if not (0 <= sr < rows and 0 <= sc < cols):
        return times, parents, visit_order, 0
    if not grid.passable[sr, sc]:
        return times, parents, visit_order, 0

    times[sr, sc] = 0.0
    heap: list[tuple[float, int, int]] = [(0.0, sc, sr)]
    closed = np.zeros((rows, cols), dtype=bool)

    while heap:
        t, col, row = heapq.heappop(heap)
        if closed[row, col]:
            continue
        closed[row, col] = True
        if visit_order is not None:
            visit_order[row, col] = n_closed
            n_closed += 1
        if t > times[row, col]:
            continue

        for nc, nr, edge_t in iter_neighbors(col, row, grid):
            if closed[nr, nc]:
                continue
            tentative = t + edge_t
            if tentative < times[nr, nc]:
                times[nr, nc] = tentative
                parents[nr, nc, 0] = col
                parents[nr, nc, 1] = row
                heapq.heappush(heap, (tentative, nc, nr))

    return times, parents, visit_order, n_closed


def _reconstruct_cells(
    parents: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    gc, gr = goal
    if parents[gr, gc, 0] < 0 and goal != start:
        return []
    cells_rev: list[tuple[int, int]] = []
    col, row = gc, gr
    sc, sr = start
    guard = parents.shape[0] * parents.shape[1] + 2
    while True:
        cells_rev.append((col, row))
        if col == sc and row == sr:
            break
        pc, pr = int(parents[row, col, 0]), int(parents[row, col, 1])
        if pc < 0:
            return []
        col, row = pc, pr
        guard -= 1
        if guard <= 0:
            return []
    cells_rev.reverse()
    return cells_rev


def route_from_parents(
    grid: CostGrid,
    parents: np.ndarray,
    times: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> Route | None:
    """Собрать Route по parent-указателям Dijkstra."""
    cells = _reconstruct_cells(parents, start, goal)
    if not cells:
        return None
    gc, gr = goal
    t = float(times[gr, gc])
    if not np.isfinite(t):
        return None
    px = cells_to_pixel_path(cells, grid.cell_size)
    return Route(
        cells=cells,
        pixels=px,
        time_s=t,
        distance_m=_path_distance_m(cells, grid.cell_m),
    )


def dijkstra_traced(
    grid: CostGrid,
    start_xy: tuple[float, float],
) -> tuple[SearchTrace, np.ndarray, np.ndarray]:
    """
    Dijkstra от точки с visit_order (для анимации волны) + parents/times.

    SearchTrace.route = None (мультицель); goal_cell = start.
    """
    _, start, _ = snap_to_passable(start_xy[0], start_xy[1], grid)
    times, parents, visit, n_closed = _dijkstra_core(grid, start, record_visit=True)
    if visit is None:
        visit = np.full(grid.shape, -1, dtype=np.int32)
        n_closed = 0
    trace = SearchTrace(
        visit_order=visit,
        n_closed=int(n_closed),
        route=None,
        start_cell=start,
        goal_cell=start,
        cell_size=grid.cell_size,
    )
    return trace, times, parents


@dataclass
class EdgeGraph:
    """Полный граф времён + полилинии кратчайших путей на местности."""

    matrix: np.ndarray
    points_xy: list[tuple[float, float]]
    routes: dict[tuple[int, int], Route] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.points_xy)

    def path_pixels(self, i: int, j: int) -> list[tuple[float, float]]:
        r = self.routes.get((i, j))
        if r is not None and r.pixels:
            return list(r.pixels)
        return [self.points_xy[i], self.points_xy[j]]

    def tour_pixels(
        self,
        order: Sequence[int],
        *,
        return_to_start: bool = True,
    ) -> list[tuple[float, float]]:
        """Склеить рёбра тура start→…→start (order — индексы матрицы 1..n)."""
        if not order:
            return [self.points_xy[0]]
        seq = [0, *list(order)]
        if return_to_start:
            seq.append(0)
        out: list[tuple[float, float]] = []
        for a, b in zip(seq[:-1], seq[1:], strict=True):
            seg = self.path_pixels(a, b)
            if out and seg:
                out.extend(seg[1:])
            else:
                out.extend(seg)
        return out


def snap_points(
    grid: CostGrid,
    points_xy: Sequence[tuple[float, float]],
) -> list[tuple[int, int]]:
    """Пиксели → проходимые ячейки (col, row)."""
    cells: list[tuple[int, int]] = []
    for x, y in points_xy:
        _, snapped, _ = snap_to_passable(float(x), float(y), grid)
        cells.append(snapped)
    return cells


def build_time_matrix(
    grid: CostGrid,
    points_xy: Sequence[tuple[float, float]],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Симметричная по смыслу матрица кратчайших времён (сек) между точками.

    points_xy[0] обычно старт/финиш, далее КП.
    Returns (matrix [N,N], snapped_cells).
    """
    graph = build_edge_graph(grid, points_xy, store_routes=False)
    cells = snap_points(grid, points_xy)
    return graph.matrix, cells


def build_edge_graph(
    grid: CostGrid,
    points_xy: Sequence[tuple[float, float]],
    *,
    store_routes: bool = True,
) -> EdgeGraph:
    """Матрица времён + (опционально) полилинии кратчайших путей между всеми парами."""
    pts = [(float(x), float(y)) for x, y in points_xy]
    cells = snap_points(grid, pts)
    n = len(cells)
    mat = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(mat, 0.0)
    routes: dict[tuple[int, int], Route] = {}

    for i, src in enumerate(cells):
        times, parents, _visit, _n = _dijkstra_core(grid, src, record_visit=False)
        for j, dst in enumerate(cells):
            if i == j:
                continue
            dc, dr = dst
            mat[i, j] = float(times[dr, dc])
            if store_routes and np.isfinite(mat[i, j]):
                route = route_from_parents(grid, parents, times, src, dst)
                if route is not None:
                    routes[(i, j)] = route

    return EdgeGraph(matrix=mat, points_xy=pts, routes=routes)


def path_time(
    matrix: np.ndarray, order: Sequence[int], *, return_to_start: bool = True
) -> float:
    """
    Суммарное время тура по индексам в матрице.

    order — индексы КП в матрице (без стартового 0), старт всегда 0.
    """
    if not order:
        return 0.0
    total = float(matrix[0, order[0]])
    for a, b in zip(order[:-1], order[1:], strict=True):
        total += float(matrix[a, b])
    if return_to_start:
        total += float(matrix[order[-1], 0])
    return total
