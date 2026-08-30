"""Разбор маршрута на участки по типу местности."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from src.routing.config import CLASS_LABELS_RU, CLASS_NAMES, DEFAULT_MIN_SEGMENT_M
from src.routing.cost_model import CostGrid
from src.routing.graph import SQRT2
from src.routing.pathfinding import Route


@dataclass
class RouteSegment:
    class_id: int
    class_name: str
    label_ru: str
    distance_m: float
    time_s: float
    share: float  # доля дистанции
    n_cells: int


def _cell_step_m(c0: int, r0: int, c1: int, r1: int, cell_m: float) -> float:
    diag = (c0 != c1) and (r0 != r1)
    return cell_m * (SQRT2 if diag else 1.0)


def describe_segments(
    route: Route,
    grid: CostGrid,
    *,
    min_segment_m: float = DEFAULT_MIN_SEGMENT_M,
) -> list[RouteSegment]:
    """
    Run-length сегменты по class_grid + сглаживание коротких дребезгов.
    """
    cells = route.cells
    if len(cells) == 0:
        return []
    if len(cells) == 1:
        c, r = cells[0]
        cid = int(grid.class_grid[r, c])
        name = CLASS_NAMES.get(cid, f"class_{cid}")
        return [
            RouteSegment(
                class_id=cid,
                class_name=name,
                label_ru=CLASS_LABELS_RU.get(name, name),
                distance_m=0.0,
                time_s=0.0,
                share=0.0,
                n_cells=1,
            )
        ]

    # сырые шаги: класс целевой ячейки
    raw: list[dict] = []
    for (c0, r0), (c1, r1) in zip(cells[:-1], cells[1:], strict=True):
        cid = int(grid.class_grid[r1, c1])
        dist = _cell_step_m(c0, r0, c1, r1, grid.cell_m)
        t0 = float(grid.time_grid[r0, c0])
        t1 = float(grid.time_grid[r1, c1])
        edge = 0.5 * (t0 + t1)
        if (c0 != c1) and (r0 != r1):
            edge *= SQRT2
        raw.append({"class_id": cid, "distance_m": dist, "time_s": edge, "n_cells": 1})

    # run-length merge
    merged: list[dict] = []
    for step in raw:
        if merged and merged[-1]["class_id"] == step["class_id"]:
            merged[-1]["distance_m"] += step["distance_m"]
            merged[-1]["time_s"] += step["time_s"]
            merged[-1]["n_cells"] += step["n_cells"]
        else:
            merged.append(dict(step))

    # сгладить короткие сегменты: влить в соседа с большей длиной
    merged = _smooth_short(merged, min_segment_m)

    total_d = sum(s["distance_m"] for s in merged) or 1.0
    segments: list[RouteSegment] = []
    for s in merged:
        cid = int(s["class_id"])
        name = CLASS_NAMES.get(cid, f"class_{cid}")
        segments.append(
            RouteSegment(
                class_id=cid,
                class_name=name,
                label_ru=CLASS_LABELS_RU.get(name, name),
                distance_m=float(s["distance_m"]),
                time_s=float(s["time_s"]),
                share=float(s["distance_m"]) / total_d,
                n_cells=int(s["n_cells"]),
            )
        )
    return segments


def _smooth_short(segments: list[dict], min_m: float) -> list[dict]:
    if min_m <= 0 or len(segments) <= 1:
        return segments

    changed = True
    out = [dict(s) for s in segments]
    while changed and len(out) > 1:
        changed = False
        i = 0
        while i < len(out):
            if out[i]["distance_m"] >= min_m or len(out) == 1:
                i += 1
                continue
            # влить в более длинного соседа
            left = out[i - 1] if i > 0 else None
            right = out[i + 1] if i + 1 < len(out) else None
            if left is None and right is None:
                break
            if left is None:
                target = i + 1
            elif right is None:
                target = i - 1
            else:
                target = i - 1 if left["distance_m"] >= right["distance_m"] else i + 1

            out[target]["distance_m"] += out[i]["distance_m"]
            out[target]["time_s"] += out[i]["time_s"]
            out[target]["n_cells"] += out[i]["n_cells"]
            # класс остаётся у target
            del out[i]
            if target > i:
                # удалили перед target → индекс target сдвинулся
                pass
            changed = True
            # после удаления пересобрать соседние одинаковые классы
            out = _merge_adjacent(out)
            break
    return out


def _merge_adjacent(segments: list[dict]) -> list[dict]:
    if not segments:
        return segments
    out = [dict(segments[0])]
    for s in segments[1:]:
        if out[-1]["class_id"] == s["class_id"]:
            out[-1]["distance_m"] += s["distance_m"]
            out[-1]["time_s"] += s["time_s"]
            out[-1]["n_cells"] += s["n_cells"]
        else:
            out.append(dict(s))
    return out


def segments_to_records(segments: Sequence[RouteSegment]) -> list[dict[str, Any]]:
    return [
        {
            "местность": s.label_ru,
            "класс": s.class_name,
            "длина_м": round(s.distance_m, 1),
            "время_с": round(s.time_s, 1),
            "время_мин": round(s.time_s / 60.0, 2),
            "доля_%": round(s.share * 100, 1),
            "ячеек": s.n_cells,
        }
        for s in segments
    ]


def segments_to_dataframe(segments: Sequence[RouteSegment]):
    """DataFrame, если установлен pandas; иначе list[dict]."""
    rows = segments_to_records(segments)
    try:
        import pandas as pd

        return pd.DataFrame(rows)
    except ImportError:
        return rows


def summarize_route(route: Route, segments: Sequence[RouteSegment] | None = None) -> dict:
    segs = segments if segments is not None else (route.segments or [])
    by_class: dict[str, float] = {}
    for s in segs:
        by_class[s.label_ru] = by_class.get(s.label_ru, 0.0) + s.distance_m
    return {
        "rank": route.rank,
        "distance_m": route.distance_m,
        "time_s": route.time_s,
        "time_min": route.time_min,
        "n_segments": len(segs),
        "terrain_m": by_class,
    }
