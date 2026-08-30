"""Визуализация score-O тура."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont

from src.controls.types import CourseDetection, StartFinish, VlmDetectTrace
from src.orienteering.matrix import EdgeGraph, build_edge_graph, dijkstra_traced
from src.orienteering.solver import DpTourSnapshot, DpTrace, solve_orienteering_traced
from src.orienteering.types import CoursePlan, ScoredControl, controls_table_rows
from src.routing.cost_model import CostGrid
from src.routing.pathfinding import Route
from src.routing.visualize import (
    render_class_reveal_frames,
    render_search_frames,
    render_speed_map,
    render_speed_morph_frames,
    render_window_scan_frames,
    save_animation_gif,
    save_animation_mp4,
    speed_legend_handles,
)


def _drawing_scale(image: np.ndarray) -> float:
    """Масштаб маркеров относительно эталона ~1000 px по длинной стороне."""
    h, w = image.shape[:2]
    return max(float(max(h, w)) / 1000.0, 0.75)


def draw_course_plan(
    image_rgb: np.ndarray,
    plan: CoursePlan,
    *,
    radius: int | None = None,
    thickness: int | None = None,
    route_thickness: int | None = None,
    font_scale: float | None = None,
    route_color: tuple[int, int, int] = (30, 90, 220),
    selected_color: tuple[int, int, int] = (220, 60, 40),
    other_color: tuple[int, int, int] = (140, 140, 140),
) -> np.ndarray:
    """RGB с полилинией тура, всеми КП и S/F (размеры зависят от разрешения карты)."""
    out = image_rgb.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)

    scale = _drawing_scale(out)
    if radius is None:
        radius = max(12, int(round(14 * scale)))
    if thickness is None:
        thickness = max(2, int(round(2.5 * scale)))
    if route_thickness is None:
        route_thickness = max(5, int(round(7 * scale)))
    if font_scale is None:
        font_scale = 0.55 * scale

    selected_nums = {c.number for c in plan.selected}

    for leg in plan.legs:
        pts = _route_polyline(leg)
        if len(pts) >= 2:
            cv2.polylines(
                out,
                [pts],
                False,
                route_color,
                route_thickness,
                lineType=cv2.LINE_AA,
            )

    for cp in plan.controls:
        if cp.number in selected_nums:
            continue
        _draw_cp(out, cp, other_color, radius, thickness, font_scale=font_scale)

    for i, cp in enumerate(plan.selected):
        _draw_cp(
            out,
            cp,
            selected_color,
            radius,
            thickness,
            order=i + 1,
            font_scale=font_scale,
        )

    sx, sy = plan.start_xy
    start = StartFinish(x=sx, y=sy, score=1.0, side=float(radius * 3))
    _draw_start(out, start, radius, thickness, font_scale=font_scale)

    return out


def _route_polyline(leg: Route) -> np.ndarray:
    if not leg.pixels:
        return np.zeros((0, 2), dtype=np.int32)
    pts = np.array(
        [[int(round(x)), int(round(y))] for x, y in leg.pixels],
        dtype=np.int32,
    )
    return pts


def _draw_cp(
    out: np.ndarray,
    cp: ScoredControl,
    color: tuple[int, int, int],
    radius: int,
    thickness: int,
    *,
    order: int | None = None,
    font_scale: float = 0.55,
) -> None:
    center = (int(round(cp.x)), int(round(cp.y)))
    cv2.circle(out, center, radius, color, thickness, lineType=cv2.LINE_AA)
    dot_r = max(2, radius // 4)
    cv2.circle(out, center, dot_r, color, -1, lineType=cv2.LINE_AA)
    label = f"{cp.number}({cp.points})"
    if order is not None:
        label = f"{order}:{label}"
    _outlined_text(
        out,
        label,
        (center[0] + radius + 4, center[1] - 2),
        color,
        font_scale=font_scale,
    )


def _draw_start(
    out: np.ndarray,
    start: StartFinish,
    radius: int,
    thickness: int,
    *,
    font_scale: float = 0.55,
) -> None:
    cx, cy = int(round(start.x)), int(round(start.y))
    side = int(round(start.side)) if start.side > 0 else radius * 3
    h_tri = int(round(side * np.sqrt(3) / 2))
    pts = np.array(
        [
            [cx, cy - 2 * h_tri // 3],
            [cx - side // 2, cy + h_tri // 3],
            [cx + side // 2, cy + h_tri // 3],
        ],
        dtype=np.int32,
    )
    color_s = (220, 60, 40)
    cv2.polylines(out, [pts], True, color_s, thickness, lineType=cv2.LINE_AA)
    cv2.circle(out, (cx, cy), max(3, radius // 3), color_s, -1, lineType=cv2.LINE_AA)
    _outlined_text(
        out,
        "S/F",
        (cx + radius + 4, cy - 2),
        color_s,
        font_scale=font_scale,
    )


def _outlined_text(
    out: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    *,
    font_scale: float = 0.55,
) -> None:
    outline = max(2, int(round(font_scale * 3.5)))
    fill = max(1, int(round(font_scale * 1.4)))
    cv2.putText(
        out,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (20, 20, 20),
        outline,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        fill,
        cv2.LINE_AA,
    )


def show_course_plan(
    image_rgb: np.ndarray,
    plan: CoursePlan,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 10),
    **draw_kwargs: object,
) -> tuple[Figure, Axes]:
    """matplotlib-превью тура."""
    overlay = draw_course_plan(image_rgb, plan, **draw_kwargs)  # type: ignore[arg-type]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.imshow(overlay)
    if title is None:
        dist = plan.total_distance_m
        dist_txt = f"{dist / 1000:.2f} км" if dist >= 1000 else f"{dist:.0f} м"
        title = (
            f"Баллы: {plan.total_points} | "
            f"длина: {dist_txt} | "
            f"время: {plan.total_time_min:.1f} / {plan.time_budget_s / 60:.0f} мин | "
            f"КП: {len(plan.selected)}/{len(plan.controls)} ({plan.method})"
        )
    ax.set_title(title)
    ax.axis("off")
    return fig, ax


def show_course_with_speed(
    image_rgb: np.ndarray,
    plan: CoursePlan,
    *,
    label: np.ndarray | None = None,
    figsize: tuple[float, float] = (18, 9),
) -> tuple[Figure, Axes, Axes]:
    """Слева — тур на карте, справа — сегментация, окрашенная по скорости."""
    label = label if label is not None else plan.label
    if label is None:
        raise ValueError("Нужен label (сегментация): передайте label=… или plan.label")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)
    show_course_plan(image_rgb, plan, ax=ax0)
    speed_rgb = render_speed_map(label, plan.speeds or {})
    # маршрут поверх карты скоростей
    ax1.imshow(draw_course_plan(speed_rgb, plan))
    ax1.set_title("Сегментация: цвет = скорость передвижения")
    ax1.axis("off")
    handles = speed_legend_handles(plan.speeds or {})
    if handles:
        ax1.legend(
            handles=handles,
            loc="lower left",
            fontsize=7,
            framealpha=0.9,
            ncol=2,
        )
    return fig, ax0, ax1


def print_plan_summary(plan: CoursePlan) -> None:
    """Краткий текстовый отчёт + таблица."""
    dist = plan.total_distance_m
    dist_txt = f"{dist / 1000:.2f} км" if dist >= 1000 else f"{dist:.0f} м"
    print(
        f"Метод: {plan.method} | "
        f"баллы: {plan.total_points} | "
        f"длина: {dist_txt} | "
        f"время: {plan.total_time_s:.0f} с "
        f"({plan.total_time_min:.1f} мин) / бюджет {plan.time_budget_s / 60:.0f} мин"
    )
    print(f"Порядок: {plan.order_numbers}")
    rows = controls_table_rows(plan)
    if not rows:
        print("(ни одного КП не взято)")
        return
    print(f"{'#':>3} {'номер':>6} {'баллы':>6} {'время_мин':>10}")
    for r in rows:
        print(
            f"{r['order']:>3} {r['number']:>6} {r['points']:>6} {r['time_min']:>10.2f}"
        )
    if plan.legs:
        print(f"Возврат на S/F: +{plan.legs[-1].time_s / 60:.2f} мин")
        print(f"Итоговая длина маршрута: {dist_txt} ({dist:.1f} м)")


# ---------------------------------------------------------------------------
# Полная анимация пайплайна (seg → VLM+refine → A* по ногам тура)
# ---------------------------------------------------------------------------


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0) * 255.0
    return np.ascontiguousarray(arr.astype(np.uint8))


def _load_ui_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _target_hw(image: np.ndarray, max_side: int | None) -> tuple[int, int]:
    h, w = _as_uint8_rgb(image).shape[:2]
    if max_side is None or max(h, w) <= max_side:
        return h, w
    scale = max_side / float(max(h, w))
    return max(1, int(round(h * scale))), max(1, int(round(w * scale)))


def _resize_to(frame: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    rgb = _as_uint8_rgb(frame)
    th, tw = hw
    if rgb.shape[0] == th and rgb.shape[1] == tw:
        return rgb
    return np.asarray(
        Image.fromarray(rgb).resize((tw, th), Image.Resampling.BILINEAR)
    )


def _stamp_caption(canvas: np.ndarray, text: str) -> np.ndarray:
    h, w = canvas.shape[:2]
    scale = max(w, h) / 1000.0
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img, "RGBA")
    font = _load_ui_font(max(14, int(round(22 * scale))))
    pad = max(8, int(round(10 * scale)))
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x0, y0 = pad, pad
    x1, y1 = x0 + tw + 2 * pad, y0 + th + 2 * pad
    draw.rectangle((x0, y0, x1, y1), fill=(15, 15, 20, 200))
    draw.text((x0 + pad, y0 + pad // 2), text, fill=(245, 245, 245, 255), font=font)
    return np.asarray(img.convert("RGB"))


def _stage_card(
    hw: tuple[int, int],
    title: str,
    subtitle: str = "",
    *,
    n: int = 18,
    bg: tuple[int, int, int] = (18, 22, 28),
) -> list[np.ndarray]:
    th, tw = hw
    img = Image.new("RGB", (tw, th), bg)
    draw = ImageDraw.Draw(img)
    scale = max(tw, th) / 1000.0
    font_t = _load_ui_font(max(22, int(round(36 * scale))))
    font_s = _load_ui_font(max(14, int(round(20 * scale))))
    tb = draw.textbbox((0, 0), title, font=font_t)
    tw_t = tb[2] - tb[0]
    th_t = tb[3] - tb[1]
    y = th // 2 - th_t
    draw.text(((tw - tw_t) // 2, y), title, fill=(245, 245, 245), font=font_t)
    if subtitle:
        sb = draw.textbbox((0, 0), subtitle, font=font_s)
        tw_s = sb[2] - sb[0]
        draw.text(
            ((tw - tw_s) // 2, y + th_t + int(16 * scale)),
            subtitle,
            fill=(180, 190, 200),
            font=font_s,
        )
    frame = np.asarray(img)
    return [frame] * max(n, 1)


def _display_saved(path: Path) -> None:
    try:
        from IPython.display import Image as IPyImage
        from IPython.display import Video, display

        if path.suffix.lower() == ".gif":
            display(IPyImage(filename=str(path)))
        else:
            display(Video(str(path), embed=True, html_attributes="controls loop autoplay"))
    except Exception:
        pass


def _partial_plan(
    plan: CoursePlan,
    *,
    n_visited: int,
    legs: Sequence[Route],
) -> CoursePlan:
    """Курс с уже посещёнными КП и пройденными ногами (для фона волны)."""
    return CoursePlan(
        start_xy=plan.start_xy,
        controls=list(plan.controls),
        selected=list(plan.selected[: max(n_visited, 0)]),
        total_points=sum(c.points for c in plan.selected[: max(n_visited, 0)]),
        total_time_s=float(sum(lg.time_s for lg in legs)),
        time_budget_s=plan.time_budget_s,
        legs=list(legs),
        method=plan.method,
        speeds=dict(plan.speeds),
        meta=dict(plan.meta),
        label=plan.label,
    )


def _matrix_xy(plan: CoursePlan, idx: int) -> tuple[float, float]:
    """Индекс матрицы (0 = старт, 1..n = controls[i-1]) → пиксели."""
    if idx <= 0:
        return plan.start_xy
    cp = plan.controls[idx - 1]
    return (float(cp.x), float(cp.y))


def _plan_points_xy(plan: CoursePlan) -> list[tuple[float, float]]:
    return [plan.start_xy, *((c.x, c.y) for c in plan.controls)]


def _tour_pixels(
    plan: CoursePlan,
    order: Sequence[int],
    *,
    edge_graph: EdgeGraph | None = None,
    return_to_start: bool = True,
    edge_limit: int | None = None,
) -> list[tuple[float, float]]:
    """Полилиния тура: по рёбрам на местности, иначе прямые между КП."""
    seq = [0, *list(order)]
    if return_to_start and order:
        seq.append(0)
    if edge_limit is not None:
        seq = seq[: max(edge_limit + 1, 1)]

    if edge_graph is not None:
        out: list[tuple[float, float]] = []
        for a, b in zip(seq[:-1], seq[1:], strict=True):
            seg = edge_graph.path_pixels(a, b)
            if out and seg:
                out.extend(seg[1:])
            else:
                out.extend(seg)
        return out if out else [plan.start_xy]

    pts = [_matrix_xy(plan, i) for i in seq]
    return pts


def _draw_cp_markers(
    draw: ImageDraw.ImageDraw,
    plan: CoursePlan,
    *,
    scale: float,
    selected_idx: set[int] | None = None,
    selected_only: bool = False,
) -> None:
    font = _load_ui_font(max(13, int(round(18 * scale))))
    r = max(7.0, 9.0 * scale)
    line_w = max(2, int(round(3.5 * scale)))
    other = (150, 150, 150)
    pick = (220, 60, 40)
    selected_idx = selected_idx or set()

    for i, cp in enumerate(plan.controls):
        idx = i + 1
        cx, cy = cp.x * scale, cp.y * scale
        in_tour = idx in selected_idx
        if selected_only and not in_tour:
            color = other
        else:
            color = pick if in_tour else other
        draw.ellipse(
            (cx - r, cy - r, cx + r, cy + r),
            outline=color,
            width=max(2, line_w - 1),
        )
        draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=color)
        label = f"{cp.number}({cp.points})"
        tx, ty = cx + r + 3, cy - r - 2
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            draw.text((tx + dx, ty + dy), label, fill=(20, 20, 20), font=font)
        draw.text((tx, ty), label, fill=color, font=font)

    sx, sy = plan.start_xy[0] * scale, plan.start_xy[1] * scale
    side = r * 2.4
    h_tri = side * 0.866
    tri = [
        (sx, sy - 2 * h_tri / 3),
        (sx - side / 2, sy + h_tri / 3),
        (sx + side / 2, sy + h_tri / 3),
    ]
    draw.polygon(tri, outline=pick, width=max(2, line_w - 1))
    draw.text((sx + r + 3, sy - r), "S/F", fill=pick, font=font)


def _draw_polyline(
    draw: ImageDraw.ImageDraw,
    pixels: Sequence[tuple[float, float]],
    *,
    scale: float,
    color: tuple[int, int, int],
    width: int,
) -> None:
    if len(pixels) < 2:
        return
    # прореживание для скорости отрисовки длинных путей
    step = max(1, len(pixels) // 400)
    pts = pixels[::step]
    if pts[-1] != pixels[-1]:
        pts = list(pts) + [pixels[-1]]
    flat: list[float] = []
    for x, y in pts:
        flat.extend((x * scale, y * scale))
    draw.line(flat, fill=color, width=width, joint="curve")


def _draw_solver_tour_frame(
    image: np.ndarray,
    plan: CoursePlan,
    snap: DpTourSnapshot | None,
    *,
    max_side: int | None,
    caption: str,
    edge_graph: EdgeGraph | None = None,
    selected_only: bool = False,
    edge_limit: int | None = None,
) -> np.ndarray:
    """Кадр: КП + тур по рёбрам на местности (если есть EdgeGraph)."""
    base, scale = _resize_rgb_local(image, max_side)
    img = Image.fromarray(base.copy())
    draw = ImageDraw.Draw(img)
    line_w = max(2, int(round(3.5 * scale)))
    route_c = (40, 110, 230)
    selected_idx = set(snap.order) if snap is not None else set()

    if snap is not None and snap.order:
        poly = _tour_pixels(
            plan,
            snap.order,
            edge_graph=edge_graph,
            edge_limit=edge_limit,
        )
        _draw_polyline(draw, poly, scale=scale, color=route_c, width=line_w)

    _draw_cp_markers(
        draw,
        plan,
        scale=scale,
        selected_idx=selected_idx,
        selected_only=selected_only,
    )
    return _stamp_caption(np.asarray(img), caption)


def _resize_rgb_local(image: np.ndarray, max_side: int | None) -> tuple[np.ndarray, float]:
    base = _as_uint8_rgb(image)
    h0, w0 = base.shape[:2]
    if max_side is None or max(h0, w0) <= max_side:
        return base, 1.0
    scale = max_side / float(max(h0, w0))
    new_w = max(1, int(round(w0 * scale)))
    new_h = max(1, int(round(h0 * scale)))
    out = np.asarray(
        Image.fromarray(base).resize((new_w, new_h), Image.Resampling.BILINEAR)
    )
    return out, scale


def _draw_overlay_routes(
    image: np.ndarray,
    plan: CoursePlan,
    routes: Sequence[Route],
    *,
    max_side: int | None,
    caption: str,
    route_color: tuple[int, int, int] = (40, 160, 255),
    dim: float = 0.55,
) -> np.ndarray:
    base, scale = _resize_rgb_local(image, max_side)
    canvas = (base.astype(np.float32) * dim).astype(np.uint8)
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    line_w = max(2, int(round(2.5 * scale)))
    for route in routes:
        if route.pixels:
            _draw_polyline(
                draw, route.pixels, scale=scale, color=route_color, width=line_w
            )
    _draw_cp_markers(draw, plan, scale=scale)
    return _stamp_caption(np.asarray(img), caption)


def render_edge_matrix_frames(
    image: np.ndarray,
    cost: CostGrid,
    plan: CoursePlan,
    *,
    edge_graph: EdgeGraph | None = None,
    max_side: int | None = 768,
    n_wave_frames: int = 24,
    hold_path: int = 4,
    hold_end: int = 20,
    animate_extra_sources: int = 2,
    extra_wave_frames: int = 10,
) -> tuple[list[np.ndarray], EdgeGraph]:
    """
    Анимация построения рёбер графа: Dijkstra по местности → матрица времён.

    Сначала полная (ускоренная) волна от S/F, затем коротко ещё несколько источников.
    """
    points = _plan_points_xy(plan)
    if edge_graph is None:
        edge_graph = build_edge_graph(cost, points, store_routes=True)

    rgb = _as_uint8_rgb(image)
    frames: list[np.ndarray] = []
    n = len(points)

    intro = _draw_overlay_routes(
        rgb,
        plan,
        [],
        max_side=max_side,
        caption=f"Рёбра графа  {n} узлов · Dijkstra на cost-grid",
        dim=0.7,
    )
    frames.extend([intro] * max(8, hold_path))

    # --- источник 0: старт ---
    trace, _times, _parents = dijkstra_traced(cost, points[0])
    # волна без path (route=None)
    wave = render_search_frames(
        rgb,
        trace,
        start_xy=points[0],
        goal_xy=points[0],
        n_wave_frames=n_wave_frames,
        n_path_frames=0,
        hold_frames=0,
        max_side=max_side,
    )
    _, w0 = rgb.shape[:2]
    for fr in wave:
        img = Image.fromarray(fr.copy())
        draw = ImageDraw.Draw(img)
        sc = fr.shape[1] / float(w0)
        _draw_cp_markers(draw, plan, scale=sc)
        frames.append(
            _stamp_caption(np.asarray(img), "Dijkstra от S/F  → времена до всех КП")
        )

    # рёбра S→КП появляются по одному
    from_start: list[Route] = []
    for j in range(1, n):
        r = edge_graph.routes.get((0, j))
        if r is not None:
            from_start.append(r)
        fr = _draw_overlay_routes(
            rgb,
            plan,
            from_start,
            max_side=max_side,
            caption=f"Рёбра S/F → КП  {len(from_start)}/{n - 1}  (время по местности)",
        )
        frames.extend([fr] * max(hold_path, 2))

    # --- дополнительные источники (ускоренно) ---
    extra = min(max(animate_extra_sources, 0), max(n - 1, 0))
    shown: list[Route] = list(from_start)
    for k in range(1, extra + 1):
        src_xy = points[k]
        label = (
            f"КП {plan.controls[k - 1].number}"
            if k - 1 < len(plan.controls)
            else f"узел {k}"
        )
        trace_k, _, _ = dijkstra_traced(cost, src_xy)
        wave_k = render_search_frames(
            rgb,
            trace_k,
            start_xy=src_xy,
            goal_xy=src_xy,
            n_wave_frames=extra_wave_frames,
            n_path_frames=0,
            hold_frames=0,
            max_side=max_side,
        )
        for fr in wave_k:
            frames.append(
                _stamp_caption(fr, f"Dijkstra от {label}  (узел {k + 1}/{n})")
            )
        # добавить исходящие рёбра этого источника
        for j in range(n):
            if j == k:
                continue
            r = edge_graph.routes.get((k, j))
            if r is not None:
                shown.append(r)
        fr = _draw_overlay_routes(
            rgb,
            plan,
            shown[-min(len(shown), 40) :],  # не рисовать всю кашу — последние
            max_side=max_side,
            caption=f"Матрица: источник {k + 1}/{n} · рёбер {len(edge_graph.routes)}",
            route_color=(80, 180, 220),
        )
        frames.extend([fr] * max(hold_path, 2))

    if n > extra + 1:
        # остальные источники без анимации волны
        hold = _draw_overlay_routes(
            rgb,
            plan,
            from_start,
            max_side=max_side,
            caption=(
                f"Остальные {n - extra - 1} источников — так же (без волны) · "
                f"матрица {n}×{n} готова"
            ),
        )
        frames.extend([hold] * max(hold_end, 12))
    else:
        hold = _draw_overlay_routes(
            rgb,
            plan,
            from_start,
            max_side=max_side,
            caption=f"Матрица времён {n}×{n} готова · рёбра = кратчайшие пути",
        )
        frames.extend([hold] * max(hold_end, 12))

    return frames, edge_graph


def _solver_keyframes(trace: DpTrace, *, max_improve: int = 36) -> list[tuple[str, DpTourSnapshot]]:
    """Подписи + снимки тура для покадровой анимации солвера."""
    keys: list[tuple[str, DpTourSnapshot]] = []
    budget_m = trace.budget_s / 60.0

    if trace.method == "dp":
        running: DpTourSnapshot | None = None
        for size in range(1, trace.n + 1):
            snap = (
                trace.best_by_size[size]
                if size < len(trace.best_by_size)
                else None
            )
            if snap is None:
                continue
            if (
                running is None
                or snap.points > running.points
                or (snap.points == running.points and snap.time_s < running.time_s)
            ):
                running = snap
            assert running is not None
            keys.append(
                (
                    f"DP  |S|={size}  лучший тур: {running.points} б · "
                    f"{running.time_s / 60:.1f}/{budget_m:.0f} мин  "
                    f"(по матрице времён)",
                    running,
                )
            )

        score_ups = []
        prev_pts = -1
        for snap in trace.improvements:
            if snap.points > prev_pts:
                score_ups.append(snap)
                prev_pts = snap.points
        if len(keys) < 6 and score_ups:
            for snap in score_ups[-max_improve:]:
                keys.append(
                    (
                        f"DP  новый рекорд: {snap.points} б · "
                        f"{snap.time_s / 60:.1f}/{budget_m:.0f} мин",
                        snap,
                    )
                )
    else:
        for i, snap in enumerate(trace.greedy_steps[:max_improve]):
            keys.append(
                (
                    f"Greedy  вставка {i + 1}/{len(trace.greedy_steps)}: "
                    f"{snap.points} б · {snap.time_s / 60:.1f}/{budget_m:.0f} мин",
                    snap,
                )
            )

    if trace.result.order:
        final = DpTourSnapshot(
            order=tuple(trace.result.order),
            points=int(trace.result.total_points),
            time_s=float(trace.result.total_time_s),
            mask=0,
            size=len(trace.result.order),
        )
        keys.append(
            (
                f"{trace.method.upper()}  итог: {final.points} б · "
                f"{final.time_s / 60:.1f}/{budget_m:.0f} мин · {final.size} КП",
                final,
            )
        )
    return keys


def render_solver_frames(
    image: np.ndarray,
    plan: CoursePlan,
    *,
    trace: DpTrace | None = None,
    edge_graph: EdgeGraph | None = None,
    max_side: int | None = 960,
    frames_per_step: int = 10,
    hold_end: int = 28,
    reveal_edges: bool = True,
) -> tuple[list[np.ndarray], DpTrace]:
    """
    Анимация выбора тура (DP / greedy) по уже готовой матрице времён.

    Пути рисуются по рёбрам ``edge_graph`` (реальная геометрия на местности).
    """
    if trace is None:
        if plan.time_matrix is None and edge_graph is None:
            raise ValueError("Нужен plan.time_matrix, edge_graph или DpTrace")
        mat = edge_graph.matrix if edge_graph is not None else plan.time_matrix
        pts = [c.points for c in plan.controls]
        trace = solve_orienteering_traced(mat, pts, plan.time_budget_s)

    rgb = _as_uint8_rgb(image)
    frames: list[np.ndarray] = []
    budget_m = plan.time_budget_s / 60.0
    path_note = "рёбра с местности" if edge_graph is not None else "схема"

    intro = _draw_solver_tour_frame(
        rgb,
        plan,
        None,
        max_side=max_side,
        edge_graph=edge_graph,
        caption=(
            f"Solver  {len(plan.controls)} КП · бюджет {budget_m:.0f} мин · "
            f"{trace.method} · {path_note}"
        ),
    )
    frames.extend([intro] * max(frames_per_step, 8))

    for cap, snap in _solver_keyframes(trace):
        fr = _draw_solver_tour_frame(
            rgb,
            plan,
            snap,
            max_side=max_side,
            edge_graph=edge_graph,
            caption=cap,
            selected_only=True,
        )
        frames.extend([fr] * max(frames_per_step, 1))

    if reveal_edges and trace.result.order:
        final = DpTourSnapshot(
            order=tuple(trace.result.order),
            points=int(trace.result.total_points),
            time_s=float(trace.result.total_time_s),
            mask=0,
            size=len(trace.result.order),
        )
        n_edges = len(final.order) + 1
        for e in range(1, n_edges + 1):
            fr = _draw_solver_tour_frame(
                rgb,
                plan,
                final,
                max_side=max_side,
                edge_graph=edge_graph,
                caption=f"Порядок тура  ребро {e}/{n_edges}",
                selected_only=True,
                edge_limit=e,
            )
            frames.extend([fr] * max(frames_per_step // 2, 3))

        hold = _draw_solver_tour_frame(
            rgb,
            plan,
            final,
            max_side=max_side,
            edge_graph=edge_graph,
            caption=(
                f"Выбран тур: {final.points} б · {final.size} КП · "
                f"{final.time_s / 60:.1f}/{budget_m:.0f} мин"
            ),
            selected_only=True,
        )
        frames.extend([hold] * max(hold_end, 1))

    return frames, trace


def render_final_tour_frames(
    image: np.ndarray,
    plan: CoursePlan,
    *,
    max_side: int | None = 960,
    frames_per_leg: int = 16,
    hold_end: int = 36,
) -> list[np.ndarray]:
    """
    Финальный тур: плавное проявление уже выбранных ног (без повторного поиска).
    """
    rgb = _as_uint8_rgb(image)
    if not plan.selected or not plan.legs:
        hw = _target_hw(rgb, max_side)
        empty = draw_course_plan(rgb, plan)
        empty = _resize_to(empty, hw)
        return [_stamp_caption(empty, "Финальный тур (пусто)")] * max(hold_end, 1)

    frames: list[np.ndarray] = []
    n_legs = len(plan.legs)
    labels = ["S/F"] + [str(c.number) for c in plan.selected] + ["S/F"]

    for i in range(n_legs):
        partial = _partial_plan(plan, n_visited=min(i + 1, len(plan.selected)), legs=plan.legs[: i + 1])
        # на промежуточных шагах не помечаем ещё не достигнутые как selected полностью:
        # после ноги i посещены selected[:i] если i < n_legs-1 return; for return leg i==n_legs-1 all selected
        if i < n_legs - 1:
            partial = _partial_plan(plan, n_visited=i + 1, legs=plan.legs[: i + 1])
        else:
            partial = _partial_plan(
                plan, n_visited=len(plan.selected), legs=plan.legs[: i + 1]
            )

        overlay = draw_course_plan(rgb, partial)
        if max_side is not None:
            overlay = _resize_to(overlay, _target_hw(rgb, max_side))
        cap = f"Финальный маршрут  {labels[i]} → {labels[i + 1]}  ({i + 1}/{n_legs})"
        frames.extend([_stamp_caption(overlay, cap)] * max(frames_per_leg, 1))

    final = draw_course_plan(rgb, plan)
    if max_side is not None:
        final = _resize_to(final, _target_hw(rgb, max_side))
    dist = plan.total_distance_m
    dist_txt = f"{dist / 1000:.2f} км" if dist >= 1000 else f"{dist:.0f} м"
    final_cap = (
        f"Итог: {plan.total_points} б · {len(plan.selected)} КП · "
        f"{dist_txt} · {plan.total_time_min:.0f}/{plan.time_budget_s / 60:.0f} мин"
    )
    frames.extend([_stamp_caption(final, final_cap)] * max(hold_end, 1))
    return frames


def render_course_legs_frames(
    image: np.ndarray,
    cost: CostGrid,
    plan: CoursePlan,
    *,
    n_wave_frames: int = 48,
    n_path_frames: int = 24,
    hold_leg: int = 6,
    hold_end: int = 36,
    max_side: int | None = 960,
) -> list[np.ndarray]:
    """
    Устарело для пайплайна: волны A* по ногам.

    Оставлено для точечной отладки; в общей анимации используйте
    ``render_edge_matrix_frames`` + ``render_final_tour_frames``.
    """
    from src.routing.pathfinding import astar_traced

    if not plan.selected:
        hw = _target_hw(image, max_side)
        return _stage_card(hw, "Нет выбранных КП", "бюджет слишком мал?", n=hold_end)

    waypoints: list[tuple[float, float]] = [plan.start_xy]
    waypoints.extend((c.x, c.y) for c in plan.selected)
    waypoints.append(plan.start_xy)
    labels = ["S/F"] + [str(c.number) for c in plan.selected] + ["S/F"]
    n_legs = len(waypoints) - 1
    scale_n = max(1.0, n_legs / 4.0)
    wave_n = max(16, int(round(n_wave_frames / scale_n)))
    path_n = max(10, int(round(n_path_frames / scale_n)))

    frames: list[np.ndarray] = []
    completed: list[Route] = []
    rgb = _as_uint8_rgb(image)

    for i in range(n_legs):
        a, b = waypoints[i], waypoints[i + 1]
        caption = f"A*  {labels[i]} → {labels[i + 1]}  ({i + 1}/{n_legs})"
        bg_plan = _partial_plan(plan, n_visited=i, legs=completed)
        bg = draw_course_plan(rgb, bg_plan)
        trace = astar_traced(cost, a, b)
        leg_frames = render_search_frames(
            bg,
            trace,
            start_xy=a,
            goal_xy=b,
            n_wave_frames=wave_n,
            n_path_frames=path_n,
            hold_frames=hold_leg,
            max_side=max_side,
        )
        for fr in leg_frames:
            frames.append(_stamp_caption(fr, caption))
        route = trace.route or Route(
            cells=[], pixels=[a, b], time_s=float("inf"), distance_m=0.0
        )
        completed.append(route)

    final = draw_course_plan(rgb, plan)
    if max_side is not None:
        final = _resize_to(final, _target_hw(rgb, max_side))
    dist = plan.total_distance_m
    dist_txt = f"{dist / 1000:.2f} км" if dist >= 1000 else f"{dist:.0f} м"
    final_cap = (
        f"Тур: {plan.total_points} б · {len(plan.selected)} КП · "
        f"{dist_txt} · {plan.total_time_min:.0f}/{plan.time_budget_s / 60:.0f} мин"
    )
    frames.extend([_stamp_caption(final, final_cap)] * max(hold_end, 1))
    return frames


def render_course_pipeline_frames(
    image: np.ndarray,
    plan: CoursePlan,
    *,
    cost: CostGrid | None = None,
    vlm_trace: VlmDetectTrace | None = None,
    vlm_raw: CourseDetection | None = None,
    dp_trace: DpTrace | None = None,
    edge_graph: EdgeGraph | None = None,
    seg_mode: str = "scan",
    max_side: int | None = 960,
    include_seg: bool = True,
    include_vlm: bool = True,
    include_refine: bool = True,
    include_matrix: bool = True,
    include_solver: bool = True,
    include_path: bool = True,
    n_wave_frames: int = 24,
    n_path_frames: int = 16,
    hold_stage: int = 16,
    animate_extra_sources: int = 2,
    **seg_kwargs: object,
) -> list[np.ndarray]:
    """
    Кадры полного пайплайна без записи на диск.

    Этапы: SegFormer → VLM → refine → рёбра (Dijkstra) → DP → финальный тур.
    """
    from src.controls.visualize import render_vlm_detect_frames, render_vlm_refine_frames

    rgb = _as_uint8_rgb(image)
    hw = _target_hw(rgb, max_side)
    frames: list[np.ndarray] = []
    stage = 0
    graph = edge_graph
    can_matrix = include_matrix and cost is not None
    can_solver = include_solver and (
        dp_trace is not None
        or plan.time_matrix is not None
        or graph is not None
        or cost is not None
    )
    can_final = include_path and bool(plan.legs or plan.selected)

    n_stages = sum(
        [
            bool(include_seg and plan.label is not None),
            bool(include_vlm and vlm_trace is not None),
            bool(include_refine and vlm_raw is not None),
            bool(can_matrix),
            bool(can_solver),
            bool(can_final),
        ]
    )

    def add_stage(title: str, subtitle: str, chunk: Sequence[np.ndarray]) -> None:
        nonlocal stage
        stage += 1
        frames.extend(
            _stage_card(
                hw,
                f"{stage}/{n_stages}  {title}",
                subtitle,
                n=hold_stage,
            )
        )
        for fr in chunk:
            frames.append(_resize_to(fr, hw))

    if include_seg and plan.label is not None:
        speeds = plan.speeds or None
        if seg_mode == "morph":
            seg = render_speed_morph_frames(
                rgb, plan.label, speeds=speeds, max_side=max_side, **seg_kwargs  # type: ignore[arg-type]
            )
            sub = "карта → скорости"
        elif seg_mode == "reveal":
            seg = render_class_reveal_frames(
                rgb, plan.label, speeds=speeds, max_side=max_side, **seg_kwargs  # type: ignore[arg-type]
            )
            sub = "классы по скорости"
        else:
            seg = render_window_scan_frames(
                rgb, plan.label, speeds=speeds, max_side=max_side, **seg_kwargs  # type: ignore[arg-type]
            )
            sub = "sliding-window SegFormer"
        add_stage("SegFormer", sub, [_stamp_caption(f, f"SegFormer · {sub}") for f in seg])

    if include_vlm and vlm_trace is not None:
        vlm_frames = render_vlm_detect_frames(
            rgb, vlm_trace, max_side=max_side, hold_frames=max(12, hold_stage)
        )
        add_stage("VLM", f"{len(vlm_trace.tile_events)} тайлов", vlm_frames)

    if include_refine and vlm_raw is not None:
        refine_frames, _ = render_vlm_refine_frames(
            rgb, vlm_raw, max_side=max_side, hold_frames=max(12, hold_stage)
        )
        add_stage("Refine", "snap к magenta-кольцам", refine_frames)

    if can_matrix:
        assert cost is not None
        matrix_frames, graph = render_edge_matrix_frames(
            rgb,
            cost,
            plan,
            edge_graph=graph,
            max_side=max_side,
            n_wave_frames=n_wave_frames,
            hold_end=max(16, hold_stage),
            animate_extra_sources=animate_extra_sources,
            extra_wave_frames=max(6, n_wave_frames // 3),
        )
        add_stage(
            "Рёбра графа",
            f"Dijkstra · матрица {graph.n}×{graph.n}",
            matrix_frames,
        )
    elif graph is None and cost is not None and can_solver:
        # тихо построить рёбра для отрисовки DP
        graph = build_edge_graph(cost, _plan_points_xy(plan), store_routes=True)

    if can_solver:
        solver_frames, used_trace = render_solver_frames(
            rgb,
            plan,
            trace=dp_trace,
            edge_graph=graph,
            max_side=max_side,
            frames_per_step=max(6, hold_stage // 2),
            hold_end=max(20, hold_stage),
        )
        method = used_trace.method
        add_stage(
            "DP" if method == "dp" else "Solver",
            f"{method} · {used_trace.result.total_points} б · "
            f"{len(used_trace.result.order)} КП",
            solver_frames,
        )

    if can_final:
        path_frames = render_final_tour_frames(
            rgb,
            plan,
            max_side=max_side,
            frames_per_leg=max(8, n_path_frames // 2),
            hold_end=max(24, hold_stage * 2),
        )
        add_stage(
            "Финальный маршрут",
            f"{len(plan.selected)} КП · {plan.method}",
            path_frames,
        )

    if not frames:
        raise ValueError(
            "Нет кадров: нужен plan.label / vlm / cost / time_matrix / legs "
            "и соответствующие include_* флаги"
        )
    return frames


def animate_course_pipeline(
    image: np.ndarray,
    plan: CoursePlan,
    *,
    cost: CostGrid | None = None,
    vlm_trace: VlmDetectTrace | None = None,
    vlm_raw: CourseDetection | None = None,
    dp_trace: DpTrace | None = None,
    edge_graph: EdgeGraph | None = None,
    out_path: str | Path | None = None,
    format: str = "gif",
    fps: int = 14,
    max_side: int | None = 768,
    seg_mode: str = "scan",
    include_seg: bool = True,
    include_vlm: bool = True,
    include_refine: bool = True,
    include_matrix: bool = True,
    include_solver: bool = True,
    include_path: bool = True,
    display: bool = True,
    **kwargs: object,
) -> Path:
    """
    Полная анимация score-O пайплайна → GIF/MP4.

    Порядок: seg → VLM → refine → **рёбра (Dijkstra)** → **DP** → **финальный тур**.
    """
    frames = render_course_pipeline_frames(
        image,
        plan,
        cost=cost,
        vlm_trace=vlm_trace,
        vlm_raw=vlm_raw,
        dp_trace=dp_trace,
        edge_graph=edge_graph,
        seg_mode=seg_mode,
        max_side=max_side,
        include_seg=include_seg,
        include_vlm=include_vlm,
        include_refine=include_refine,
        include_matrix=include_matrix,
        include_solver=include_solver,
        include_path=include_path,
        **kwargs,
    )

    if out_path is None:
        out_path = Path("runs") / "course_vis" / f"pipeline.{format}"
    else:
        out_path = Path(out_path)

    if format == "mp4":
        saved = save_animation_mp4(frames, out_path.with_suffix(".mp4"), fps=fps)
    else:
        saved = save_animation_gif(frames, out_path.with_suffix(".gif"), fps=fps)

    if display:
        _display_saved(saved)
    return saved
