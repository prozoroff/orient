"""Orienteering Problem: max сумма баллов за лимит времени (старт = финиш)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from src.orienteering.matrix import path_time

# Порог для точного DP (число КП, без старта)
EXACT_DP_MAX_N = 18


@dataclass(frozen=True)
class SolverResult:
    """Порядок индексов КП в матрице (1..n), без стартового 0."""

    order: list[int]  # индексы в матрице
    total_points: int
    total_time_s: float
    method: str


@dataclass(frozen=True)
class DpTourSnapshot:
    """Полный тур start→…→start (order — индексы матрицы 1..n)."""

    order: tuple[int, ...]
    points: int
    time_s: float
    mask: int
    size: int  # число КП в туре


@dataclass
class DpTrace:
    """Промежуточные шаги DP / эвристики для анимации."""

    n: int
    budget_s: float
    points: list[int]
    method: str
    # лучший полный тур среди подмножеств размера ровно k (1..n); None если нет
    best_by_size: list[DpTourSnapshot | None] = field(default_factory=list)
    # каждое улучшение глобального best (по баллам, затем по времени)
    improvements: list[DpTourSnapshot] = field(default_factory=list)
    # для greedy: порядок вставки по шагам
    greedy_steps: list[DpTourSnapshot] = field(default_factory=list)
    result: SolverResult = field(
        default_factory=lambda: SolverResult([], 0, 0.0, "empty")
    )


def solve_orienteering(
    time_matrix: np.ndarray,
    points: Sequence[int],
    time_budget_s: float,
    *,
    exact_max_n: int = EXACT_DP_MAX_N,
) -> SolverResult:
    """
    Максимизировать сумму points при туре start→…→start за ≤ time_budget_s.

    time_matrix[0,0] — старт; points[i] соответствует узлу i+1 матрицы.
    """
    return solve_orienteering_traced(
        time_matrix, points, time_budget_s, exact_max_n=exact_max_n
    ).result


def solve_orienteering_traced(
    time_matrix: np.ndarray,
    points: Sequence[int],
    time_budget_s: float,
    *,
    exact_max_n: int = EXACT_DP_MAX_N,
) -> DpTrace:
    """То же, что ``solve_orienteering``, плюс снимок шагов для анимации."""
    mat = np.asarray(time_matrix, dtype=np.float64)
    n = len(points)
    if mat.shape != (n + 1, n + 1):
        raise ValueError(
            f"Ожидалась матрица {(n + 1, n + 1)}, получено {mat.shape}"
        )
    budget = float(time_budget_s)
    pts = [int(p) for p in points]
    if n == 0 or budget <= 0:
        empty = SolverResult(order=[], total_points=0, total_time_s=0.0, method="empty")
        return DpTrace(n=0, budget_s=budget, points=[], method="empty", result=empty)

    if n <= exact_max_n:
        return _solve_dp_traced(mat, pts, budget)
    return _solve_heuristic_traced(mat, pts, budget)


def _reconstruct_order(
    parent: list[list[tuple[int, int] | None]],
    mask: int,
    last: int,
) -> list[int]:
    order_rev: list[int] = []
    while last >= 0:
        order_rev.append(last + 1)
        prev = parent[mask][last]
        if prev is None or prev[0] < 0:
            break
        mask, last = prev
    return list(reversed(order_rev))


def _solve_dp(
    mat: np.ndarray,
    points: list[int],
    budget: float,
) -> SolverResult:
    return _solve_dp_traced(mat, points, budget).result


def _solve_dp_traced(
    mat: np.ndarray,
    points: list[int],
    budget: float,
) -> DpTrace:
    """
    DP: best_time[mask][last] = мин. время пути start→…→last, посетив ровно mask.

    last — индекс КП 0..n-1 (узел матрицы last+1).
    """
    n = len(points)
    INF = np.inf
    best_time = [[INF] * n for _ in range(1 << n)]
    parent: list[list[tuple[int, int] | None]] = [
        [None] * n for _ in range(1 << n)
    ]

    for j in range(n):
        t = float(mat[0, j + 1])
        if np.isfinite(t) and t <= budget:
            best_time[1 << j][j] = t
            parent[1 << j][j] = (-1, -1)

    for mask in range(1 << n):
        for last in range(n):
            if not (mask & (1 << last)):
                continue
            cur_t = best_time[mask][last]
            if not np.isfinite(cur_t):
                continue
            for nxt in range(n):
                if mask & (1 << nxt):
                    continue
                edge = float(mat[last + 1, nxt + 1])
                if not np.isfinite(edge):
                    continue
                nt = cur_t + edge
                if nt > budget:
                    continue
                nmask = mask | (1 << nxt)
                if nt < best_time[nmask][nxt]:
                    best_time[nmask][nxt] = nt
                    parent[nmask][nxt] = (mask, last)

    best_by_size: list[DpTourSnapshot | None] = [None] * (n + 1)
    improvements: list[DpTourSnapshot] = []
    best_score = -1
    best_time_full = INF
    best_mask = 0
    best_last = -1

    for mask in range(1 << n):
        size = mask.bit_count()
        score = sum(points[i] for i in range(n) if mask & (1 << i))
        for last in range(n):
            if not (mask & (1 << last)):
                continue
            t_to = best_time[mask][last]
            if not np.isfinite(t_to):
                continue
            ret = float(mat[last + 1, 0])
            if not np.isfinite(ret):
                continue
            total = t_to + ret
            if total > budget:
                continue

            prev_size = best_by_size[size]
            better_size = (
                prev_size is None
                or score > prev_size.points
                or (score == prev_size.points and total < prev_size.time_s)
            )
            better_global = score > best_score or (
                score == best_score and total < best_time_full
            )
            if not better_size and not better_global:
                continue

            order = tuple(_reconstruct_order(parent, mask, last))
            snap = DpTourSnapshot(
                order=order,
                points=int(score),
                time_s=float(total),
                mask=int(mask),
                size=int(size),
            )
            if better_size:
                best_by_size[size] = snap
            if better_global:
                best_score = score
                best_time_full = total
                best_mask = mask
                best_last = last
                improvements.append(snap)

    if best_last < 0:
        result = SolverResult(order=[], total_points=0, total_time_s=0.0, method="dp")
    else:
        order = _reconstruct_order(parent, best_mask, best_last)
        result = SolverResult(
            order=order,
            total_points=int(best_score),
            total_time_s=float(best_time_full),
            method="dp",
        )

    return DpTrace(
        n=n,
        budget_s=budget,
        points=list(points),
        method="dp",
        best_by_size=best_by_size,
        improvements=improvements,
        result=result,
    )


def _tour_points(order: Sequence[int], points: list[int]) -> int:
    """order — индексы матрицы (1..n)."""
    return sum(points[i - 1] for i in order)


def _feasible(mat: np.ndarray, order: Sequence[int], budget: float) -> bool:
    if not order:
        return True
    t = path_time(mat, order, return_to_start=True)
    return np.isfinite(t) and t <= budget + 1e-6


def _snap_from_order(
    mat: np.ndarray,
    points: list[int],
    order: Sequence[int],
) -> DpTourSnapshot:
    t = path_time(mat, order, return_to_start=True) if order else 0.0
    if not np.isfinite(t):
        t = float("inf")
    mask = 0
    for idx in order:
        mask |= 1 << (idx - 1)
    return DpTourSnapshot(
        order=tuple(order),
        points=_tour_points(order, points),
        time_s=float(t) if np.isfinite(t) else 0.0,
        mask=mask,
        size=len(order),
    )


def _solve_heuristic(
    mat: np.ndarray,
    points: list[int],
    budget: float,
) -> SolverResult:
    return _solve_heuristic_traced(mat, points, budget).result


def _solve_heuristic_traced(
    mat: np.ndarray,
    points: list[int],
    budget: float,
) -> DpTrace:
    """Greedy insertion (max points/Δtime) + локальный поиск."""
    n = len(points)
    order, greedy_steps = _greedy_insert_traced(mat, points, budget)
    order = _local_search(mat, points, budget, order)
    result_snap = _snap_from_order(mat, points, order)
    result = SolverResult(
        order=list(order),
        total_points=result_snap.points,
        total_time_s=result_snap.time_s,
        method="greedy_local",
    )
    return DpTrace(
        n=n,
        budget_s=budget,
        points=list(points),
        method="greedy_local",
        best_by_size=[],
        improvements=list(greedy_steps),
        greedy_steps=greedy_steps,
        result=result,
    )


def _greedy_insert(
    mat: np.ndarray,
    points: list[int],
    budget: float,
) -> list[int]:
    order, _ = _greedy_insert_traced(mat, points, budget)
    return order


def _greedy_insert_traced(
    mat: np.ndarray,
    points: list[int],
    budget: float,
) -> tuple[list[int], list[DpTourSnapshot]]:
    n = len(points)
    remaining = set(range(1, n + 1))
    order: list[int] = []
    steps: list[DpTourSnapshot] = []

    while remaining:
        best_cand: int | None = None
        best_pos = 0
        best_ratio = -1.0
        best_delta = np.inf

        for cand in list(remaining):
            for pos in range(len(order) + 1):
                trial = order[:pos] + [cand] + order[pos:]
                if not _feasible(mat, trial, budget):
                    continue
                t_new = path_time(mat, trial, return_to_start=True)
                t_old = path_time(mat, order, return_to_start=True) if order else 0.0
                delta = t_new - t_old
                if delta <= 1e-9:
                    ratio = float(points[cand - 1]) * 1e6
                else:
                    ratio = float(points[cand - 1]) / delta
                if ratio > best_ratio or (abs(ratio - best_ratio) < 1e-12 and delta < best_delta):
                    best_ratio = ratio
                    best_delta = delta
                    best_cand = cand
                    best_pos = pos

        if best_cand is None:
            break
        order = order[:best_pos] + [best_cand] + order[best_pos:]
        remaining.remove(best_cand)
        steps.append(_snap_from_order(mat, points, order))

    return order, steps


def _local_search(
    mat: np.ndarray,
    points: list[int],
    budget: float,
    order: list[int],
    *,
    max_rounds: int = 50,
) -> list[int]:
    """Улучшение: insert unused, remove, relocate, 2-opt, swap."""
    n = len(points)
    current = list(order)
    best_score = _tour_points(current, points)
    best_time = path_time(mat, current, return_to_start=True) if current else 0.0

    def accept(trial: list[int]) -> bool:
        nonlocal current, best_score, best_time
        if not _feasible(mat, trial, budget):
            return False
        sc = _tour_points(trial, points)
        tm = path_time(mat, trial, return_to_start=True)
        if sc > best_score or (sc == best_score and tm < best_time - 1e-9):
            current = trial
            best_score = sc
            best_time = tm
            return True
        return False

    for _ in range(max_rounds):
        improved = False
        used = set(current)
        unused = [i for i in range(1, n + 1) if i not in used]

        # insert
        for cand in unused:
            for pos in range(len(current) + 1):
                trial = current[:pos] + [cand] + current[pos:]
                if accept(trial):
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue

        # remove (если можно взять что-то ценнее — простая remove+reinsert позже)
        for i in range(len(current)):
            trial = current[:i] + current[i + 1 :]
            # remove только если не ухудшает score без компенсации — пропускаем
            # чистый remove полезен только ради времени для insert; делаем remove+insert
            for cand in unused:
                for pos in range(len(trial) + 1):
                    trial2 = trial[:pos] + [cand] + trial[pos:]
                    if accept(trial2):
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # relocate
        for i in range(len(current)):
            node = current[i]
            base = current[:i] + current[i + 1 :]
            for pos in range(len(base) + 1):
                if pos == i:
                    continue
                trial = base[:pos] + [node] + base[pos:]
                if accept(trial):
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue

        # swap
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                trial = list(current)
                trial[i], trial[j] = trial[j], trial[i]
                if accept(trial):
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue

        # 2-opt
        for i in range(len(current) - 1):
            for j in range(i + 1, len(current)):
                trial = current[:i] + list(reversed(current[i : j + 1])) + current[j + 1 :]
                if accept(trial):
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    return current
