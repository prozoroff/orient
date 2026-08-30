"""Типы результата детекции КП и старта."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Literal, Sequence


@dataclass(frozen=True)
class ControlPoint:
    """Один контрольный пункт на карте."""

    number: int
    x: float  # пиксель, ось X вправо
    y: float  # пиксель, ось Y вниз
    score: float
    source: Literal["circle", "digit", "vlm"]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StartFinish:
    """Треугольник старта/финиша (один на карте, розовый, ≈ размер кружка КП)."""

    x: float
    y: float
    score: float
    side: float = 0.0  # средняя длина стороны (px)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CourseDetection:
    """КП + опциональный старт/финиш.

    Итерируется как список КП (обратная совместимость с ``for c in detect_controls(...)``).
    """

    controls: list[ControlPoint] = field(default_factory=list)
    start: StartFinish | None = None

    def __iter__(self) -> Iterator[ControlPoint]:
        return iter(self.controls)

    def __len__(self) -> int:
        return len(self.controls)

    def __getitem__(self, idx: int) -> ControlPoint:
        return self.controls[idx]

    def to_dict(self) -> dict[str, Any]:
        return {
            "controls": [c.to_dict() for c in self.controls],
            "start": None if self.start is None else self.start.to_dict(),
        }


@dataclass(frozen=True)
class OcrWindow:
    """Прямоугольник OCR-окна (x0,y0,x1,y1)."""

    x0: int
    y0: int
    x1: int
    y1: int
    kind: Literal["directional", "around"] = "directional"


@dataclass(frozen=True)
class RingOcrEvent:
    """Один проход: кольцо → окна OCR → выбранный номер (или отказ)."""

    cx: float
    cy: float
    radius: float
    hollow: float
    windows: tuple[OcrWindow, ...]
    hit_number: int | None
    hit_conf: float
    hit_bbox: tuple[int, int, int, int] | None  # x0,y0,x1,y1
    control: ControlPoint | None


@dataclass(frozen=True)
class DigitAssistEvent:
    """Digit-assist: сильный номер → поиск кольца рядом."""

    hit_number: int
    hit_conf: float
    hit_bbox: tuple[int, int, int, int]
    control: ControlPoint


@dataclass
class CvDetectTrace:
    """Промежуточные шаги circle-first для анимации."""

    pink: Any  # np.ndarray uint8 mask
    rings: Sequence[tuple[float, float, float, float]]  # cx, cy, radius, hollow
    ring_events: list[RingOcrEvent] = field(default_factory=list)
    digit_assist: list[DigitAssistEvent] = field(default_factory=list)
    result: CourseDetection = field(default_factory=CourseDetection)


@dataclass(frozen=True)
class VlmTileHit:
    """Один hit VLM в координатах всего кадра (px)."""

    number: int
    x: float
    y: float
    confidence: float
    kept: bool  # прошёл min_confidence и попал в финальный by_number


@dataclass(frozen=True)
class VlmTileEvent:
    """Результат одного тайла Vision LLM."""

    x0: int
    y0: int
    x1: int
    y1: int
    hits: tuple[VlmTileHit, ...]
    start_xy: tuple[float, float, float] | None  # x, y, conf в px кадра


@dataclass
class VlmDetectTrace:
    """Промежуточные шаги VLM-детекции для анимации."""

    tile_size: int
    overlap: int
    refine_circles: bool
    tile_events: list[VlmTileEvent] = field(default_factory=list)
    result: CourseDetection = field(default_factory=CourseDetection)
