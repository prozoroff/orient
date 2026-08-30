"""Высокоуровневый фасад для notebook."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.routing.config import (
    DEFAULT_CELL_SIZE,
    DEFAULT_CONTOUR_PENALTY,
    DEFAULT_K_ROUTES,
    DEFAULT_MIN_SEGMENT_M,
    DEFAULT_PENALTY_FACTOR,
    DEFAULT_PENALTY_WIDTH,
    DEFAULT_SIMILARITY_THRESHOLD,
    copy_speeds,
)
from src.routing.cost_model import CostGrid, build_cost_grid
from src.routing.geo import MapMeta, load_meta
from src.routing.inference import load_rgb, segment_map
from src.routing.pathfinding import Route, find_k_routes
from src.routing.segments import describe_segments


@dataclass
class RoutingResult:
    image: np.ndarray
    label: np.ndarray
    meta: MapMeta
    cost: CostGrid
    routes: list[Route]
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    speeds: dict[str, float] = field(default_factory=dict)


def prepare_segmentation(
    image_path: str | Path,
    *,
    model: Any | None = None,
    device: Any | None = None,
    cfg: dict[str, Any] | None = None,
    checkpoint: str | Path | None = None,
    meta_path: str | Path | None = None,
    label_path: str | Path | None = None,
    use_gt: bool = False,
    postprocess: bool = True,
    use_cache: bool = True,
    resolution_m_per_px: float | None = None,
    **infer_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, MapMeta]:
    """
    Загрузить изображение и получить индексную маску сегментации.

    Ожидаемый вход — только ``image_path`` (+ модель/чекпоинт).
    ``meta.json`` подхватывается рядом с картинкой автоматически (для м/px).
    ``resolution_m_per_px`` — явный override (если в meta/OCD неверный масштаб).
    ``label_path`` / ``use_gt`` — только для отладки роутинга без модели
    (``label.png`` или ``channels.npy`` рядом с картинкой / тайлом).
    """
    image_path = Path(image_path)
    image = load_rgb(image_path)

    if meta_path is None:
        cand = image_path.parent / "meta.json"
        meta_path = cand if cand.exists() else None
    meta = load_meta(meta_path)
    if resolution_m_per_px is not None:
        meta = meta.with_resolution(float(resolution_m_per_px))


    gt_path: Path | None = None
    channels_path: Path | None = None
    if label_path is not None:
        p = Path(label_path)
        if p.suffix == ".npy":
            channels_path = p
        else:
            gt_path = p
    elif use_gt:
        cand = image_path.parent / "label.png"
        ch_cand = image_path.parent / "channels.npy"
        if cand.exists():
            gt_path = cand
        elif ch_cand.exists():
            channels_path = ch_cand
        else:
            raise FileNotFoundError(
                f"use_gt=True, но нет {cand} и {ch_cand}. "
                "Передайте label_path явно."
            )

    if gt_path is None and channels_path is None and model is None and checkpoint is None:
        raise ValueError(
            "Нужна модель или checkpoint для сегментации изображения. "
            "Для отладки на GT передайте use_gt=True или label_path=..."
        )

    if channels_path is not None:
        from src.data.labels import build_priority_indices, channels_to_index

        if cfg is None:
            raise ValueError("Для GT из channels.npy нужен cfg (label_priority).")
        priority = build_priority_indices(cfg["label_priority"], cfg["classes"])
        label = channels_to_index(np.load(channels_path), priority)
        if postprocess:
            from src.postprocess import morphological_close_linear

            label = morphological_close_linear(label, cfg)
        return image, label.astype(np.uint8), meta

    label = segment_map(
        image_path,
        model=model,
        device=device,
        cfg=cfg,
        checkpoint=checkpoint,
        label_path=gt_path,
        postprocess=postprocess,
        use_cache=use_cache,
        **infer_kwargs,
    )
    return image, label, meta


def compute_routes(
    label: np.ndarray,
    meta: MapMeta,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    *,
    speeds: dict[str, float] | None = None,
    cell_size: int = DEFAULT_CELL_SIZE,
    k: int = DEFAULT_K_ROUTES,
    penalty_factor: float = DEFAULT_PENALTY_FACTOR,
    penalty_width: int = DEFAULT_PENALTY_WIDTH,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    min_segment_m: float = DEFAULT_MIN_SEGMENT_M,
    contour_penalty: float = DEFAULT_CONTOUR_PENALTY,
    describe: bool = True,
) -> tuple[CostGrid, list[Route]]:
    """Cost-грид + K маршрутов (+ сегменты местности)."""
    speeds = copy_speeds(speeds)
    cost = build_cost_grid(
        label,
        meta,
        speeds=speeds,
        cell_size=cell_size,
        contour_penalty=contour_penalty,
    )
    routes = find_k_routes(
        cost,
        start_xy,
        goal_xy,
        k=k,
        penalty_factor=penalty_factor,
        penalty_width=penalty_width,
        similarity_threshold=similarity_threshold,
    )
    if describe:
        for r in routes:
            r.segments = describe_segments(r, cost, min_segment_m=min_segment_m)
    return cost, routes


def find_routes_on_map(
    image_path: str | Path,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    *,
    model: Any | None = None,
    device: Any | None = None,
    cfg: dict[str, Any] | None = None,
    checkpoint: str | Path | None = None,
    meta_path: str | Path | None = None,
    label_path: str | Path | None = None,
    use_gt: bool = False,
    speeds: dict[str, float] | None = None,
    cell_size: int = DEFAULT_CELL_SIZE,
    k: int = DEFAULT_K_ROUTES,
    **kwargs: Any,
) -> RoutingResult:
    """Полный пайплайн: изображение → сегментация → маршруты."""
    image, label, meta = prepare_segmentation(
        image_path,
        model=model,
        device=device,
        cfg=cfg,
        checkpoint=checkpoint,
        meta_path=meta_path,
        label_path=label_path,
        use_gt=use_gt,
        postprocess=kwargs.pop("postprocess", True),
        use_cache=kwargs.pop("use_cache", True),
    )
    cost, routes = compute_routes(
        label,
        meta,
        start_xy,
        goal_xy,
        speeds=speeds,
        cell_size=cell_size,
        k=k,
        **kwargs,
    )
    return RoutingResult(
        image=image,
        label=label,
        meta=meta,
        cost=cost,
        routes=routes,
        start_xy=start_xy,
        goal_xy=goal_xy,
        speeds=copy_speeds(speeds),
    )
