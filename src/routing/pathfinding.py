"""A* и K разнообразных маршрутов (штраф-коридор)."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import ndimage

from src.routing.config import (
    DEFAULT_K_ROUTES,
    DEFAULT_PENALTY_FACTOR,
    DEFAULT_PENALTY_WIDTH,
    DEFAULT_SIMILARITY_THRESHOLD,
)
from src.routing.cost_model import CostGrid
from src.routing.geo import cells_to_pixel_path
from src.routing.graph import SQRT2, iter_neighbors, snap_to_passable


@dataclass
class Route:
    """Один найденный маршрут."""

    cells: list[tuple[int, int]]  # (col, row)
    pixels: list[tuple[float, float]]  # (x, y)
    time_s: float
    distance_m: float
    rank: int = 0
    segments: object | None = None  # заполняется в segments.py / api
    meta: dict = field(default_factory=dict)

    @property
    def time_min(self) -> float:
        return self.time_s / 60.0


@dataclass
class SearchTrace:
    """Трассировка A*: порядок закрытия ячеек (волна) + итоговый путь."""

    visit_order: np.ndarray  # (H, W) int32, -1 = не посещена, иначе 0..n_closed-1
    n_closed: int
    route: Route | None
    start_cell: tuple[int, int]
    goal_cell: tuple[int, int]
    cell_size: int


def _heuristic(col: int, row: int, goal: tuple[int, int], cell_m: float, max_speed: float) -> float:
    """Допустимая эвристика: евклидово расстояние / max_speed."""
    dc = col - goal[0]
    dr = row - goal[1]
    dist_m = ((dc * dc + dr * dr) ** 0.5) * cell_m
    return dist_m / max(max_speed, 1e-6)


def _run_astar(
    grid: CostGrid,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    time_override: np.ndarray | None = None,
    record_visit_order: bool = False,
) -> tuple[Route | None, np.ndarray | None, int]:
    """
    Ядро A* на 8-связной сетке.

    Returns:
        (route, visit_order or None, n_closed)
    """
    rows, cols = grid.shape
    sc, sr = start
    gc, gr = goal
    if not (0 <= sr < rows and 0 <= sc < cols and 0 <= gr < rows and 0 <= gc < cols):
        return None, None, 0
    if not grid.passable[sr, sc] or not grid.passable[gr, gc]:
        return None, None, 0
    if start == goal:
        px = cells_to_pixel_path([start], grid.cell_size)
        route = Route(cells=[start], pixels=px, time_s=0.0, distance_m=0.0)
        visit = None
        if record_visit_order:
            visit = np.full((rows, cols), -1, dtype=np.int32)
            visit[sr, sc] = 0
        return route, visit, 1 if record_visit_order else 0

    times = time_override if time_override is not None else grid.time_grid
    max_speed = grid.max_speed
    cell_m = grid.cell_m

    # локальный grid-like доступ к passable/time через обёртку
    class _G:
        shape = grid.shape
        passable = grid.passable
        cell_size = grid.cell_size
        cell_m = grid.cell_m
        speeds = grid.speeds

        @property
        def time_grid(self):
            return times

    gview = _G()

    open_heap: list[tuple[float, int, int, int]] = []  # (f, tie, col, row)
    tie = 0
    g_score = np.full((rows, cols), np.inf, dtype=np.float64)
    g_score[sr, sc] = 0.0
    came_from = np.full((rows, cols, 2), -1, dtype=np.int32)

    h0 = _heuristic(sc, sr, goal, cell_m, max_speed)
    heapq.heappush(open_heap, (h0, tie, sc, sr))
    closed = np.zeros((rows, cols), dtype=bool)
    visit_order: np.ndarray | None = (
        np.full((rows, cols), -1, dtype=np.int32) if record_visit_order else None
    )
    n_closed = 0

    while open_heap:
        f, _, col, row = heapq.heappop(open_heap)
        if closed[row, col]:
            continue
        closed[row, col] = True
        if visit_order is not None:
            visit_order[row, col] = n_closed
            n_closed += 1

        if col == gc and row == gr:
            break

        for nc, nr, edge_t in iter_neighbors(col, row, gview):  # type: ignore[arg-type]
            if closed[nr, nc]:
                continue
            # пересчитать edge с override-временами
            t0 = float(times[row, col])
            t1 = float(times[nr, nc])
            dc = abs(nc - col)
            dr = abs(nr - row)
            edge = 0.5 * (t0 + t1)
            if dc and dr:
                edge *= SQRT2
            tentative = g_score[row, col] + edge
            if tentative < g_score[nr, nc]:
                g_score[nr, nc] = tentative
                came_from[nr, nc] = (col, row)
                tie += 1
                hf = tentative + _heuristic(nc, nr, goal, cell_m, max_speed)
                heapq.heappush(open_heap, (hf, tie, nc, nr))

    if not np.isfinite(g_score[gr, gc]):
        return None, visit_order, n_closed

    # reconstruct
    path: list[tuple[int, int]] = []
    cur = (gc, gr)
    while cur != start:
        path.append(cur)
        pc, pr = came_from[cur[1], cur[0]]
        if pc < 0:
            return None, visit_order, n_closed
        cur = (int(pc), int(pr))
    path.append(start)
    path.reverse()

    dist = _path_distance_m(path, cell_m)
    route = Route(
        cells=path,
        pixels=cells_to_pixel_path(path, grid.cell_size),
        time_s=float(g_score[gr, gc]),
        distance_m=dist,
    )
    return route, visit_order, n_closed


def astar(
    grid: CostGrid,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    time_override: np.ndarray | None = None,
) -> Route | None:
    """
    A* на 8-связной сетке.
    time_override — альтернативная карта времён (для penalty-поиска).
    """
    route, _, _ = _run_astar(grid, start, goal, time_override=time_override)
    return route


def astar_traced(
    grid: CostGrid,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    *,
    time_override: np.ndarray | None = None,
) -> SearchTrace:
    """A* с записью порядка закрытия ячеек (для анимации волны)."""
    _, start, _ = snap_to_passable(start_xy[0], start_xy[1], grid)
    _, goal, _ = snap_to_passable(goal_xy[0], goal_xy[1], grid)
    route, visit_order, n_closed = _run_astar(
        grid,
        start,
        goal,
        time_override=time_override,
        record_visit_order=True,
    )
    if visit_order is None:
        visit_order = np.full(grid.shape, -1, dtype=np.int32)
        n_closed = 0
    return SearchTrace(
        visit_order=visit_order,
        n_closed=n_closed,
        route=route,
        start_cell=start,
        goal_cell=goal,
        cell_size=grid.cell_size,
    )


def _path_distance_m(cells: Sequence[tuple[int, int]], cell_m: float) -> float:
    if len(cells) < 2:
        return 0.0
    total = 0.0
    for (c0, r0), (c1, r1) in zip(cells[:-1], cells[1:], strict=True):
        dc = abs(c1 - c0)
        dr = abs(r1 - r0)
        step = cell_m * (SQRT2 if dc and dr else 1.0)
        total += step
    return total


def _inflate_corridor(
    base_times: np.ndarray,
    path_cells: Sequence[tuple[int, int]],
    factor: float,
    width: int,
    passable: np.ndarray,
) -> np.ndarray:
    """Повысить стоимость вдоль коридора предыдущего пути."""
    mask = np.zeros(base_times.shape, dtype=bool)
    for c, r in path_cells:
        mask[r, c] = True
    if width > 0:
        struct = ndimage.generate_binary_structure(2, 2)
        mask = ndimage.binary_dilation(mask, structure=struct, iterations=width)

    out = base_times.copy()
    apply = mask & passable & np.isfinite(out)
    out[apply] = out[apply] * float(factor)
    return out


def path_similarity(a: Sequence[tuple[int, int]], b: Sequence[tuple[int, int]]) -> float:
    """Доля общих ячеек относительно более короткого пути (IoU-подобно)."""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    return inter / min(len(sa), len(sb))


def find_k_routes(
    grid: CostGrid,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    k: int = DEFAULT_K_ROUTES,
    *,
    penalty_factor: float = DEFAULT_PENALTY_FACTOR,
    penalty_width: int = DEFAULT_PENALTY_WIDTH,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    max_attempts: int | None = None,
) -> list[Route]:
    """
    K визуально различных быстрых маршрутов (penalty corridor).

    1) A* → лучший путь
    2) штраф вдоль коридора → новый A*
    3) принять, если непохож на уже найденные
    """
    _, start, _ = snap_to_passable(start_xy[0], start_xy[1], grid)
    _, goal, _ = snap_to_passable(goal_xy[0], goal_xy[1], grid)

    if start == goal:
        px = cells_to_pixel_path([start], grid.cell_size)
        return [
            Route(cells=[start], pixels=px, time_s=0.0, distance_m=0.0, rank=1)
        ]

    attempts = max_attempts or max(k * 4, k + 2)
    routes: list[Route] = []
    times = grid.time_grid.copy()

    for attempt in range(attempts):
        if len(routes) >= k:
            break
        route = astar(grid, start, goal, time_override=times)
        if route is None:
            if not routes:
                return []
            break

        # фильтр похожести относительно уже принятых
        too_similar = any(
            path_similarity(route.cells, prev.cells) >= similarity_threshold
            for prev in routes
        )
        if too_similar and routes:
            # всё равно штрафуем, чтобы уйти дальше
            times = _inflate_corridor(
                times, route.cells, penalty_factor, penalty_width, grid.passable
            )
            continue

        # время всегда по исходной стоимости (не по штрафованной сетке)
        true_route = route if attempt == 0 else (_retime_path(grid, route.cells) or route)
        true_route.rank = len(routes) + 1
        true_route.meta = {
            "start_cell": start,
            "goal_cell": goal,
            "start_xy": start_xy,
            "goal_xy": goal_xy,
            "attempt": attempt,
        }
        routes.append(true_route)

        times = _inflate_corridor(
            times, route.cells, penalty_factor, penalty_width, grid.passable
        )

    return routes


def _retime_path(grid: CostGrid, cells: Sequence[tuple[int, int]]) -> Route | None:
    """Пересчитать время/дистанцию пути по исходной сетке."""
    if not cells:
        return None
    if len(cells) == 1:
        px = cells_to_pixel_path(list(cells), grid.cell_size)
        return Route(cells=list(cells), pixels=px, time_s=0.0, distance_m=0.0)

    total_t = 0.0
    for (c0, r0), (c1, r1) in zip(cells[:-1], cells[1:], strict=True):
        t0 = float(grid.time_grid[r0, c0])
        t1 = float(grid.time_grid[r1, c1])
        if not (np.isfinite(t0) and np.isfinite(t1)):
            return None
        edge = 0.5 * (t0 + t1)
        if (c0 != c1) and (r0 != r1):
            edge *= SQRT2
        total_t += edge

    return Route(
        cells=list(cells),
        pixels=cells_to_pixel_path(list(cells), grid.cell_size),
        time_s=total_t,
        distance_m=_path_distance_m(cells, grid.cell_m),
    )
