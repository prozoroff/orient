"""Типы результата планирования score-O дистанции."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.routing.pathfinding import Route


@dataclass(frozen=True)
class ScoredControl:
    """КП с баллами (первая цифра номера) и координатами в пикселях."""

    number: int
    x: float
    y: float
    points: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CoursePlan:
    """Оптимальный (или эвристический) тур: старт → КП… → старт."""

    start_xy: tuple[float, float]
    controls: list[ScoredControl]  # все найденные КП
    selected: list[ScoredControl]  # выбранные, в порядке посещения
    total_points: int
    total_time_s: float
    time_budget_s: float
    legs: list[Route] = field(default_factory=list)  # N+1 сегментов (включая возврат)
    time_matrix: Any | None = None  # np.ndarray | None
    method: str = ""
    speeds: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    label: Any | None = None  # np.ndarray[H,W] сегментация, если сохранена

    @property
    def total_time_min(self) -> float:
        return self.total_time_s / 60.0

    @property
    def total_distance_m(self) -> float:
        """Суммарная длина ног маршрута (м); 0 если ноги не реконструированы."""
        if not self.legs:
            return 0.0
        return float(sum(float(leg.distance_m) for leg in self.legs))

    @property
    def order_numbers(self) -> list[int]:
        return [c.number for c in self.selected]

    def cumulative_times(self) -> list[float]:
        """Кумулятивное время после каждой ноги (сек), длина = len(legs)."""
        out: list[float] = []
        acc = 0.0
        for leg in self.legs:
            acc += float(leg.time_s)
            out.append(acc)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_xy": list(self.start_xy),
            "controls": [c.to_dict() for c in self.controls],
            "selected": [c.to_dict() for c in self.selected],
            "total_points": self.total_points,
            "total_time_s": self.total_time_s,
            "time_budget_s": self.time_budget_s,
            "order_numbers": self.order_numbers,
            "method": self.method,
            "n_legs": len(self.legs),
            "total_distance_m": self.total_distance_m,
        }


def controls_table_rows(plan: CoursePlan) -> list[dict[str, Any]]:
    """Строки таблицы: номер, баллы, кумулятивное время до КП (не включая возврат)."""
    rows: list[dict[str, Any]] = []
    cum = 0.0
    for i, cp in enumerate(plan.selected):
        if i < len(plan.legs):
            cum += float(plan.legs[i].time_s)
        rows.append(
            {
                "order": i + 1,
                "number": cp.number,
                "points": cp.points,
                "time_s": round(cum, 1),
                "time_min": round(cum / 60.0, 2),
            }
        )
    return rows


def scale_speeds_to_open_land(
    open_land_speed_mps: float,
    base: dict[str, float] | None = None,
) -> dict[str, float]:
    """Масштабировать все скорости так, чтобы open_land == open_land_speed_mps."""
    from src.routing.config import DEFAULT_SPEEDS_MPS, copy_speeds

    speeds = copy_speeds(base)
    base_open = float(speeds.get("open_land", DEFAULT_SPEEDS_MPS["open_land"]))
    if base_open <= 0:
        raise ValueError("Базовая скорость open_land должна быть > 0")
    factor = float(open_land_speed_mps) / base_open
    return {k: (0.0 if v <= 0 else float(v) * factor) for k, v in speeds.items()}
