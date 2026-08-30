"""Детекция КП через OpenAI Vision + уточнение центров по magenta-кольцам."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

from src.controls.geometry import Blob, detect_circles_hough, find_start_triangle, ring_quality
from src.controls.pink_mask import merge_prior, pink_mask_hsv
from src.controls.types import (
    ControlPoint,
    CourseDetection,
    StartFinish,
    VlmDetectTrace,
    VlmTileEvent,
    VlmTileHit,
)
from src.controls.vlm_openai import call_openai_controls
from src.controls.vlm_yandex import call_yandex_controls, list_yandex_chat_models

Image.MAX_IMAGE_PIXELS = None


def _load_rgb(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr.astype(np.uint8)
    return np.array(Image.open(image).convert("RGB"), dtype=np.uint8)


def _iter_tiles(
    h: int,
    w: int,
    *,
    tile_size: int,
    overlap: int,
) -> Iterator[tuple[int, int, int, int]]:
    """Ббоксы тайлов (x0,y0,x1,y1) с overlap; один тайл = весь кадр, если влезает."""
    if max(h, w) <= tile_size:
        yield 0, 0, w, h
        return
    step = max(1, tile_size - overlap)
    ys = list(range(0, max(1, h - tile_size + 1), step))
    xs = list(range(0, max(1, w - tile_size + 1), step))
    if ys[-1] + tile_size < h:
        ys.append(h - tile_size)
    if xs[-1] + tile_size < w:
        xs.append(w - tile_size)
    for y0 in ys:
        for x0 in xs:
            x1 = min(w, x0 + tile_size)
            y1 = min(h, y0 + tile_size)
            yield x0, y0, x1, y1


def estimate_cp_radius_px(
    pink: np.ndarray,
    *,
    hint_diameter: float | None = None,
) -> float:
    """
    Оценить радиус кружка КП (px).

    Приоритет: ``hint_diameter`` (если задан) → Hough по розовой маске →
    эвристика от размера кадра.
    """
    h, w = pink.shape[:2]
    if hint_diameter is not None and hint_diameter > 0:
        return float(hint_diameter) * 0.5

    # широкий проход: на печати ISOM ~6 мм; на 300 DPI ≈ 70 px Ø, на скринах меньше
    lo = max(8, int(round(0.008 * max(h, w))))
    hi = max(lo + 8, int(round(0.045 * max(h, w))))
    circs = detect_circles_hough(
        pink,
        min_radius=lo,
        max_radius=hi,
        min_dist=max(14, lo),
        param2=11,
        min_hollow=0.22,
        max_interior=0.50,
        min_peri=0.30,
    )
    if circs:
        rs = np.array([c.radius for c in circs], dtype=np.float64)
        # отсечь выбросы, взять медиану
        q1, q3 = np.percentile(rs, [25, 75])
        iqr = max(q3 - q1, 1.0)
        kept = rs[(rs >= q1 - 1.5 * iqr) & (rs <= q3 + 1.5 * iqr)]
        if kept.size:
            return float(np.median(kept))

    # fallback: ~55 px диаметр на карте ~2–3k px
    return float(np.clip(0.018 * max(h, w), 12.0, 40.0))


def refine_ring_center(
    pink: np.ndarray,
    x: float,
    y: float,
    radius: float,
    *,
    search: float = 8.0,
    step: float = 1.0,
) -> tuple[float, float, float]:
    """Уточнить центр кольца: сетка hollow + центроид розового кольца."""
    h, w = pink.shape[:2]
    best_x, best_y = float(x), float(y)
    _, _, best_h = ring_quality(pink, best_x, best_y, radius)
    rad = int(np.ceil(search))
    for dy in np.arange(-rad, rad + 1e-6, step):
        for dx in np.arange(-rad, rad + 1e-6, step):
            gx, gy = x + float(dx), y + float(dy)
            peri, _interior, hollow = ring_quality(pink, gx, gy, radius)
            if peri < 0.25:
                continue
            if hollow > best_h:
                best_h = hollow
                best_x, best_y = gx, gy

    # центроид пикселей в кольцевой зоне вокруг текущего центра
    rr = max(radius, 1.0)
    pad = int(np.ceil(rr + 4))
    x0 = max(0, int(best_x) - pad)
    y0 = max(0, int(best_y) - pad)
    x1 = min(w, int(best_x) + pad + 1)
    y1 = min(h, int(best_y) + pad + 1)
    crop = pink[y0:y1, x0:x1] > 0
    if crop.any():
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.sqrt((xx - best_x) ** 2 + (yy - best_y) ** 2)
        annulus = crop & (d >= 0.70 * rr) & (d <= 1.30 * rr)
        if int(annulus.sum()) >= 12:
            cy = float(yy[annulus].mean())
            cx = float(xx[annulus].mean())
            peri, _interior, hollow = ring_quality(pink, cx, cy, radius)
            if hollow + 0.02 >= best_h:
                best_x, best_y, best_h = cx, cy, hollow

    for dy in (-0.5, 0.0, 0.5):
        for dx in (-0.5, 0.0, 0.5):
            gx, gy = best_x + dx, best_y + dy
            peri, _interior, hollow = ring_quality(pink, gx, gy, radius)
            if hollow > best_h:
                best_h = hollow
                best_x, best_y = gx, gy
    return best_x, best_y, float(best_h)


def _snap_to_ring(
    pink: np.ndarray,
    x: float,
    y: float,
    *,
    expected_radius: float | None = None,
    search_radius: float | None = None,
    min_r: int | None = None,
    max_r: int | None = None,
) -> tuple[float, float, float, bool]:
    """
    Притянуть VLM-точку к ближайшему полому magenta-кольцу.
    Returns (x, y, hollow, snapped).
    """
    h, w = pink.shape[:2]
    r0 = float(expected_radius) if expected_radius and expected_radius > 0 else estimate_cp_radius_px(pink)
    if min_r is None:
        min_r = max(8, int(round(0.70 * r0)))
    if max_r is None:
        max_r = max(min_r + 2, int(round(1.35 * r0)))
    if search_radius is None:
        # VLM часто попадает в цифру рядом с кружком — запас ~1.5 диаметра
        search_radius = max(2.2 * r0, 1.5 * (max_r))

    rad = int(np.ceil(search_radius))
    x0 = max(0, int(x) - rad)
    y0 = max(0, int(y) - rad)
    x1 = min(w, int(x) + rad + 1)
    y1 = min(h, int(y) + rad + 1)
    crop = pink[y0:y1, x0:x1]
    if crop.size < 25:
        return x, y, 0.0, False

    cands: list[Blob] = []
    for c in detect_circles_hough(
        crop,
        min_radius=min_r,
        max_radius=max_r,
        min_dist=max(12, min_r),
        param2=10,
        min_hollow=0.25,
        max_interior=0.45,
        min_peri=0.35,
    ):
        cands.append(
            Blob(
                label=c.label,
                area=c.area,
                cx=c.cx + x0,
                cy=c.cy + y0,
                x0=c.x0 + x0,
                y0=c.y0 + y0,
                x1=c.x1 + x0,
                y1=c.y1 + y0,
                circularity=c.circularity,
                fill_ratio=c.fill_ratio,
                has_hole=True,
                kind="circle",
                radius=c.radius,
            )
        )

    # сетка по радиусам вокруг ожидаемого r0
    radii = sorted(
        {
            int(round(r0 * s))
            for s in (0.85, 0.95, 1.0, 1.05, 1.15)
            if min_r <= int(round(r0 * s)) <= max_r
        }
        | {min_r, max_r, int(round(r0))}
    )
    step = max(2, int(round(0.25 * r0)))
    for rr in radii:
        for dy in range(-rad, rad + 1, step):
            for dx in range(-rad, rad + 1, step):
                gx, gy = x + dx, y + dy
                if not (0 <= gx < w and 0 <= gy < h):
                    continue
                peri, interior, hollow = ring_quality(pink, gx, gy, float(rr))
                if peri >= 0.35 and interior <= 0.40 and hollow >= 0.25:
                    cands.append(
                        Blob(
                            label=50_000 + len(cands),
                            area=float(np.pi * rr * rr),
                            cx=float(gx),
                            cy=float(gy),
                            x0=int(gx - rr),
                            y0=int(gy - rr),
                            x1=int(gx + rr),
                            y1=int(gy + rr),
                            circularity=float(hollow),
                            fill_ratio=float(interior),
                            has_hole=True,
                            kind="circle",
                            radius=float(rr),
                        )
                    )

    if not cands:
        return x, y, 0.0, False

    best: Blob | None = None
    best_score = -1e9
    for c in cands:
        d = float(np.hypot(c.cx - x, c.cy - y))
        if d > search_radius:
            continue
        # Качество кольца важнее точности черновой VLM-точки. На карте 1200 px
        # VLM нередко указывает на подпись КП в 40–60 px от окружности.
        # Расстояние уже ограничено search_radius, поэтому оставляем его только
        # как tie-breaker между сопоставимыми кольцами.
        r_pen = abs(c.radius - r0) / max(r0, 1.0)
        score = (
            float(c.circularity) * 2.0
            - 0.35 * (d / max(search_radius, 1.0))
            - 0.35 * r_pen
        )
        if score > best_score:
            best_score = score
            best = c
    if best is None:
        return x, y, 0.0, False

    # дожим центра под найденный радиус
    fx, fy, hollow = refine_ring_center(
        pink,
        best.cx,
        best.cy,
        max(best.radius, r0 * 0.9),
        search=max(4.0, 0.35 * r0),
        step=1.0,
    )
    return float(fx), float(fy), float(hollow), True


def detect_controls_vlm(
    image: str | Path | np.ndarray,
    *,
    provider: str = "yandex",
    model: str | None = None,
    api_key: str | None = None,
    folder_id: str | None = None,
    prior_mask: np.ndarray | None = None,
    tile_size: int = 1024,
    overlap: int = 160,
    max_side: int | None = None,
    refine_circles: bool = True,
    find_start: bool = True,
    min_confidence: float = 0.35,
    circle_diameter_px: float | None = 55.0,
    circle_search_radius_px: float | None = None,
    start_search_radius_px: float | None = None,
    client: Any | None = None,
) -> CourseDetection:
    """
    Найти КП через Vision LLM (Yandex Cloud или OpenAI).

    Parameters
    ----------
    provider
        ``yandex`` (по умолчанию) — AI Studio, модель ``qwen3.6-35b-a3b``.
        ``openai`` — OpenAI Vision (из РФ часто 403).
    model
        Имя/URI модели. По умолчанию: qwen3.6-35b-a3b (yandex) / gpt-4.1-mini (openai).
    api_key
        Yandex: ``YC_API_KEY`` / ``YANDEX_API_KEY``. OpenAI: ``OPENAI_API_KEY``.
    folder_id
        Только для Yandex: ``YC_FOLDER_ID``.
    circle_diameter_px
        Диаметр розового кружка КП в пикселях карты (для snap к центру).
        ``None`` — оценить автоматически по маске. Типично ~55 на картах ~2–3k px.
    """
    return detect_controls_vlm_trace(
        image,
        provider=provider,
        model=model,
        api_key=api_key,
        folder_id=folder_id,
        prior_mask=prior_mask,
        tile_size=tile_size,
        overlap=overlap,
        max_side=max_side,
        refine_circles=refine_circles,
        find_start=find_start,
        min_confidence=min_confidence,
        circle_diameter_px=circle_diameter_px,
        circle_search_radius_px=circle_search_radius_px,
        start_search_radius_px=start_search_radius_px,
        client=client,
    ).result


def detect_controls_vlm_trace(
    image: str | Path | np.ndarray,
    *,
    provider: str = "yandex",
    model: str | None = None,
    api_key: str | None = None,
    folder_id: str | None = None,
    prior_mask: np.ndarray | None = None,
    tile_size: int = 1024,
    overlap: int = 160,
    max_side: int | None = None,
    refine_circles: bool = True,
    find_start: bool = True,
    min_confidence: float = 0.35,
    circle_diameter_px: float | None = 55.0,
    circle_search_radius_px: float | None = None,
    start_search_radius_px: float | None = None,
    client: Any | None = None,
) -> VlmDetectTrace:
    """VLM-детекция с промежуточными шагами по тайлам (для анимации)."""
    if provider not in {"yandex", "openai"}:
        raise ValueError(f"Unknown provider={provider!r}, use yandex|openai")

    if model is None:
        model = "qwen3.6-35b-a3b" if provider == "yandex" else "gpt-4.1-mini"
    if max_side is None:
        max_side = 1536 if provider == "yandex" else 2048

    rgb = _load_rgb(image)
    h, w = rgb.shape[:2]

    pink = None
    expected_r: float | None = None
    if refine_circles or find_start:
        pink = pink_mask_hsv(
            rgb,
            use_clahe=True,
            min_saturation=25,
            min_value=30,
            preserve_thin=True,
        )
        pink = merge_prior(pink, prior_mask)
        if refine_circles:
            expected_r = estimate_cp_radius_px(pink, hint_diameter=circle_diameter_px)

    by_number: dict[int, ControlPoint] = {}
    start_cands: list[tuple[float, float, float]] = []
    tile_events: list[VlmTileEvent] = []

    for x0, y0, x1, y1 in _iter_tiles(h, w, tile_size=tile_size, overlap=overlap):
        crop = rgb[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]
        if ch < 32 or cw < 32:
            continue
        if provider == "yandex":
            tile = call_yandex_controls(
                crop,
                model=model,
                api_key=api_key,
                folder_id=folder_id,
                max_side=max_side,
                client=client,
            )
        else:
            tile = call_openai_controls(
                crop,
                model=model,
                api_key=api_key,
                max_side=max_side,
                client=client,
            )

        tile_hits: list[VlmTileHit] = []
        for hit in tile.controls:
            px = x0 + hit.x_norm * cw
            py = y0 + hit.y_norm * ch
            if hit.confidence < min_confidence:
                tile_hits.append(
                    VlmTileHit(
                        number=int(hit.number),
                        x=float(px),
                        y=float(py),
                        confidence=float(hit.confidence),
                        kept=False,
                    )
                )
                continue

            source: str = "vlm"
            score = float(hit.confidence)
            if refine_circles and pink is not None:
                sx, sy, hollow, snapped = _snap_to_ring(
                    pink,
                    px,
                    py,
                    expected_radius=expected_r,
                    search_radius=circle_search_radius_px,
                )
                if snapped:
                    px, py = sx, sy
                    source = "circle"
                    score = float(hit.confidence) * (0.55 + 0.45 * min(1.0, hollow))
            cp = ControlPoint(
                number=hit.number,
                x=float(px),
                y=float(py),
                score=float(score),
                source=source,  # type: ignore[arg-type]
            )
            prev = by_number.get(hit.number)
            kept = prev is None or cp.score > prev.score
            if kept:
                by_number[hit.number] = cp
            tile_hits.append(
                VlmTileHit(
                    number=int(hit.number),
                    x=float(px),
                    y=float(py),
                    confidence=float(hit.confidence),
                    kept=kept,
                )
            )

        start_xy: tuple[float, float, float] | None = None
        if find_start and tile.start is not None and tile.start.confidence >= min_confidence:
            sx = x0 + tile.start.x_norm * cw
            sy = y0 + tile.start.y_norm * ch
            start_xy = (float(sx), float(sy), float(tile.start.confidence))
            start_cands.append(start_xy)

        tile_events.append(
            VlmTileEvent(
                x0=int(x0),
                y0=int(y0),
                x1=int(x1),
                y1=int(y1),
                hits=tuple(tile_hits),
                start_xy=start_xy,
            )
        )

    controls = sorted(by_number.values(), key=lambda c: (c.number, -c.score))

    start: StartFinish | None = None
    ref_size = float(expected_r * 2) if expected_r else 28.0
    exclude_r = max(22.0, ref_size * 0.55)
    if find_start:
        if start_cands:
            sx, sy, sc = max(start_cands, key=lambda t: t[2])
            side = 0.0
            if pink is not None:
                tri = find_start_triangle(
                    pink,
                    ref_size=ref_size,
                    exclude_xy=[(c.x, c.y) for c in controls],
                    exclude_radius=exclude_r,
                )
                if tri is not None:
                    tx, ty, tscore, tside = tri
                    start_search_radius = (
                        float(start_search_radius_px)
                        if start_search_radius_px is not None
                        else max(40.0, ref_size)
                    )
                    if np.hypot(tx - sx, ty - sy) < start_search_radius:
                        sx, sy, sc, side = tx, ty, max(sc, tscore), tside
            start = StartFinish(x=float(sx), y=float(sy), score=float(sc), side=float(side))
        elif pink is not None:
            tri = find_start_triangle(
                pink,
                ref_size=ref_size,
                exclude_xy=[(c.x, c.y) for c in controls],
                exclude_radius=exclude_r,
            )
            if tri is not None:
                sx, sy, sc, side = tri
                start = StartFinish(x=sx, y=sy, score=sc, side=side)

    result = CourseDetection(controls=controls, start=start)
    return VlmDetectTrace(
        tile_size=int(tile_size),
        overlap=int(overlap),
        refine_circles=bool(refine_circles),
        tile_events=tile_events,
        result=result,
    )
