"""Высокоуровневый фасад: карта → оптимальный score-O тур."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.controls import detect_controls, detect_controls_vlm
from src.controls.types import ControlPoint, CourseDetection
from src.orienteering.matrix import build_time_matrix
from src.orienteering.scoring import points_from_number
from src.orienteering.solver import solve_orienteering
from src.orienteering.types import CoursePlan, ScoredControl, scale_speeds_to_open_land
from src.routing.api import prepare_segmentation
from src.routing.config import DEFAULT_CONTOUR_PENALTY
from src.routing.cost_model import CostGrid, build_cost_grid
from src.routing.graph import snap_to_passable
from src.routing.pathfinding import Route, astar


def _to_scored(controls: list[ControlPoint]) -> list[ScoredControl]:
    return [
        ScoredControl(
            number=int(c.number),
            x=float(c.x),
            y=float(c.y),
            points=points_from_number(c.number),
        )
        for c in controls
    ]


def _resolve_start(
    detection: CourseDetection,
    start_xy: tuple[float, float] | None,
) -> tuple[float, float]:
    if start_xy is not None:
        return (float(start_xy[0]), float(start_xy[1]))
    if detection.start is None:
        raise ValueError(
            "Треугольник старта/финиша не найден. "
            "Передайте start_xy=(x, y) вручную."
        )
    return (float(detection.start.x), float(detection.start.y))


def _leg_route(
    grid: CostGrid,
    a_xy: tuple[float, float],
    b_xy: tuple[float, float],
) -> Route:
    _, a_cell, _ = snap_to_passable(a_xy[0], a_xy[1], grid)
    _, b_cell, _ = snap_to_passable(b_xy[0], b_xy[1], grid)
    route = astar(grid, a_cell, b_cell)
    if route is None:
        return Route(cells=[], pixels=[a_xy, b_xy], time_s=float("inf"), distance_m=0.0)
    return route


def build_legs(
    grid: CostGrid,
    start_xy: tuple[float, float],
    selected: list[ScoredControl],
) -> list[Route]:
    """Полилинии start → КП… → start."""
    if not selected:
        return []
    waypoints: list[tuple[float, float]] = [start_xy]
    waypoints.extend((c.x, c.y) for c in selected)
    waypoints.append(start_xy)
    legs: list[Route] = []
    for a, b in zip(waypoints[:-1], waypoints[1:], strict=True):
        legs.append(_leg_route(grid, a, b))
    return legs


def plan_course_from_parts(
    *,
    cost: CostGrid,
    controls: list[ScoredControl],
    start_xy: tuple[float, float],
    time_budget_s: float,
    speeds: dict[str, float] | None = None,
    reconstruct_legs: bool = True,
) -> CoursePlan:
    """Планирование при уже готовом CostGrid и списке КП."""
    points_xy: list[tuple[float, float]] = [start_xy]
    points_xy.extend((c.x, c.y) for c in controls)
    matrix, _cells = build_time_matrix(cost, points_xy)

    pts = [c.points for c in controls]
    result = solve_orienteering(matrix, pts, time_budget_s)

    # order — индексы матрицы 1..n → controls[i-1]
    selected = [controls[i - 1] for i in result.order]
    legs: list[Route] = []
    if reconstruct_legs and selected:
        legs = build_legs(cost, start_xy, selected)

    total_time = float(result.total_time_s)
    if legs and all(np.isfinite(lg.time_s) for lg in legs):
        total_time = float(sum(lg.time_s for lg in legs))

    return CoursePlan(
        start_xy=start_xy,
        controls=list(controls),
        selected=selected,
        total_points=int(result.total_points),
        total_time_s=total_time,
        time_budget_s=float(time_budget_s),
        legs=legs,
        time_matrix=matrix,
        method=result.method,
        speeds=dict(speeds or cost.speeds),
        meta={"n_controls": len(controls), "n_selected": len(selected)},
    )


def plan_course(
    image_path: str | Path,
    *,
    open_land_speed_mps: float,
    time_budget_s: float,
    model: Any | None = None,
    device: Any | None = None,
    cfg: dict[str, Any] | None = None,
    checkpoint: str | Path | None = None,
    meta_path: str | Path | None = None,
    label_path: str | Path | None = None,
    use_gt: bool = False,
    start_xy: tuple[float, float] | None = None,
    cell_size: int = 4,
    detector: str = "classical",
    detect_kwargs: dict[str, Any] | None = None,
    reconstruct_legs: bool = True,
    postprocess: bool = True,
    use_cache: bool = True,
    contour_penalty: float = DEFAULT_CONTOUR_PENALTY,
    resolution_m_per_px: float | None = None,
) -> CoursePlan:
    """
    Полный пайплайн: изображение → КП + сегментация → оптимальный тур.

    Баллы КП = первая цифра номера. Старт = финиш (треугольник или start_xy).

    Parameters
    ----------
    detector
        ``classical`` — ``detect_controls`` (circle-first / OCR).
        ``vlm`` — ``detect_controls_vlm`` (Yandex/OpenAI + snap к кольцам).
    detect_kwargs
        Аргументы выбранного детектора (для VLM: ``provider``, ``model``, …).
    contour_penalty
        Множитель рельефа: ``time *= (1 + penalty * density)`` горизонталей.
    resolution_m_per_px
        Явные метры/пиксель. Если ``None`` — из ``meta.json``. Нужен, когда
        в OCD/meta масштаб неверный (на карте написано иначе).
    """
    image_path = Path(image_path)
    speeds = scale_speeds_to_open_land(open_land_speed_mps)

    kwargs = dict(detect_kwargs or {})
    if detector == "classical":
        detection = detect_controls(image_path, **kwargs)
    elif detector == "vlm":
        detection = detect_controls_vlm(image_path, **kwargs)
    else:
        raise ValueError(f"Unknown detector={detector!r}, use classical|vlm")
    start = _resolve_start(detection, start_xy)
    scored = _to_scored(list(detection.controls))

    _image, label, meta = prepare_segmentation(
        image_path,
        model=model,
        device=device,
        cfg=cfg,
        checkpoint=checkpoint,
        meta_path=meta_path,
        label_path=label_path,
        use_gt=use_gt,
        postprocess=postprocess,
        use_cache=use_cache,
        resolution_m_per_px=resolution_m_per_px,
    )
    cost = build_cost_grid(
        label,
        meta,
        speeds=speeds,
        cell_size=cell_size,
        contour_penalty=contour_penalty,
    )

    plan = plan_course_from_parts(
        cost=cost,
        controls=scored,
        start_xy=start,
        time_budget_s=time_budget_s,
        speeds=speeds,
        reconstruct_legs=reconstruct_legs,
    )
    plan.label = label
    plan.meta.update(
        {
            "image_path": str(image_path),
            "detector": detector,
            "detection_start": None
            if detection.start is None
            else detection.start.to_dict(),
            "open_land_speed_mps": float(open_land_speed_mps),
            "cell_size": int(cell_size),
            "contour_penalty": float(contour_penalty),
            "resolution_m_per_px": float(meta.meters_per_pixel()),
        }
    )
    return plan


__all__ = [
    "build_legs",
    "plan_course",
    "plan_course_from_parts",
]
