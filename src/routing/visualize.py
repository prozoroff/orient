"""Визуализация маршрутов для Jupyter."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image, ImageDraw

from src.data.labels import CLASS_COLORS, colorize_label, overlay_prediction
from src.routing.config import (
    CLASS_LABELS_RU,
    CLASS_NAME_TO_IDX,
    CLASS_NAMES,
    ROUTE_COLORS,
    copy_speeds,
)
from src.routing.cost_model import CostGrid, _resolve_overlays
from src.routing.pathfinding import Route, SearchTrace, astar_traced
from src.routing.segments import RouteSegment, segments_to_dataframe, summarize_route


def render_segmentation_map(label: np.ndarray) -> np.ndarray:
    """Индексная маска → RGB-карта, отрисованная только по классам (без исходного изображения)."""
    return colorize_label(label)


def _speed_field(
    label: np.ndarray,
    speeds: dict[str, float],
    *,
    resolve_overlays: bool = True,
) -> np.ndarray:
    """Пиксельная карта скоростей (м/с). Overlay-классы → соседняя местность."""
    speeds = copy_speeds(speeds)
    effective = _resolve_overlays(label, speeds) if resolve_overlays else label
    n = max(int(effective.max()) + 1, max(CLASS_NAMES) + 1, 16)
    speed_lut = np.zeros(n, dtype=np.float32)
    for name, spd in speeds.items():
        idx = CLASS_NAME_TO_IDX.get(name)
        if idx is not None and idx < n:
            speed_lut[idx] = float(spd)
    return speed_lut[np.clip(effective.astype(np.int32), 0, n - 1)]


def _speed_to_rgb(
    spd: np.ndarray,
    *,
    cmap_name: str = "RdYlGn",
    impassable_rgb: tuple[int, int, int] = (40, 40, 45),
) -> np.ndarray:
    """Скорость → RGB: быстро = зелёный, медленно = красный (RdYlGn)."""
    vmax = float(spd[spd > 0].max()) if np.any(spd > 0) else 1.0
    t = np.clip(spd / vmax, 0.0, 1.0)
    cmap = _get_cmap(cmap_name)
    rgba = cmap(t)
    out = (rgba[..., :3] * 255.0).astype(np.uint8)
    out[spd <= 0] = impassable_rgb
    return out


def render_speed_map(
    label: np.ndarray,
    speeds: dict[str, float] | None = None,
    *,
    resolve_overlays: bool = True,
    cmap_name: str = "RdYlGn",
    impassable_rgb: tuple[int, int, int] = (40, 40, 45),
) -> np.ndarray:
    """
    Сегментация → RGB, где цвет кодирует скорость передвижения.

    Быстро → зелёный, медленно → красный, непроходимо → тёмно-серый.
    """
    spd = _speed_field(label, copy_speeds(speeds), resolve_overlays=resolve_overlays)
    return _speed_to_rgb(spd, cmap_name=cmap_name, impassable_rgb=impassable_rgb)


def speed_legend_handles(
    speeds: dict[str, float] | None = None,
    *,
    cmap_name: str = "RdYlGn",
    impassable_rgb: tuple[int, int, int] = (40, 40, 45),
) -> list[Patch]:
    """Легенда классов, отсортированная по скорости (убыв.)."""
    speeds = copy_speeds(speeds)
    items = sorted(speeds.items(), key=lambda kv: (-kv[1], kv[0]))
    vmax = max((v for _, v in items if v > 0), default=1.0)
    cmap = _get_cmap(cmap_name)
    handles: list[Patch] = []
    for name, spd in items:
        ru = CLASS_LABELS_RU.get(name, name)
        if spd <= 0:
            color = tuple(c / 255.0 for c in impassable_rgb)
            text = f"{ru}: непроходимо"
        else:
            color = tuple(cmap(spd / vmax)[:3])
            text = f"{ru}: {spd:.2f} м/с"
        handles.append(Patch(facecolor=color, edgecolor="k", label=text))
    return handles


def show_speed_map(
    label: np.ndarray,
    *,
    image: np.ndarray | None = None,
    speeds: dict[str, float] | None = None,
    mode: str = "speed",
    alpha: float = 0.55,
    figsize: tuple[float, float] = (12, 10),
    show_legend: bool = True,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Карта, раскрашенная по скорости бега.

    mode:
      - \"speed\" — чистая карта скоростей (зелёный→красный)
      - \"overlay\" — полупрозрачно поверх image
    """
    speeds = copy_speeds(speeds)
    speed_rgb = render_speed_map(label, speeds)
    if mode == "overlay":
        if image is None:
            raise ValueError("mode='overlay' требует image")
        base = _as_uint8_rgb(image).astype(np.float32)
        canvas = (base * (1.0 - alpha) + speed_rgb.astype(np.float32) * alpha).astype(np.uint8)
        default_title = "Скорость (оверлей)"
    else:
        canvas = speed_rgb
        default_title = "Скорость передвижения (зелёный → красный)"

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.imshow(canvas)
    ax.set_axis_off()
    ax.set_title(title or default_title)
    if show_legend:
        handles = speed_legend_handles(speeds)
        if handles:
            ax.legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.9, ncol=2)
    return ax


def _downscale_rgb(rgb: np.ndarray, max_side: int | None) -> np.ndarray:
    arr = _as_uint8_rgb(rgb)
    if max_side is None:
        return arr
    h, w = arr.shape[:2]
    if max(h, w) <= max_side:
        return arr
    scale = max_side / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return np.asarray(Image.fromarray(arr).resize((new_w, new_h), Image.Resampling.BILINEAR))


def _display_saved_animation(path: Path) -> None:
    try:
        from IPython.display import Image as IPyImage
        from IPython.display import Video, display

        if path.suffix.lower() == ".gif":
            display(IPyImage(filename=str(path)))
        else:
            display(Video(str(path), embed=True, html_attributes="controls loop autoplay"))
    except Exception:
        pass


def render_speed_morph_frames(
    image: np.ndarray,
    label: np.ndarray,
    *,
    speeds: dict[str, float] | None = None,
    n_fade: int = 40,
    hold_start: int = 12,
    hold_end: int = 20,
    max_side: int | None = 960,
) -> list[np.ndarray]:
    """Кадры: исходная карта → плавный переход к карте скоростей."""
    speeds = copy_speeds(speeds)
    base = _downscale_rgb(image, max_side)
    speed = _downscale_rgb(render_speed_map(label, speeds), max_side)
    if speed.shape[:2] != base.shape[:2]:
        speed = np.asarray(
            Image.fromarray(speed).resize((base.shape[1], base.shape[0]), Image.Resampling.NEAREST)
        )
    frames: list[np.ndarray] = [base] * max(hold_start, 0)
    for i in range(max(n_fade, 1)):
        t = (i + 1) / max(n_fade, 1)
        # ease-in-out
        te = t * t * (3.0 - 2.0 * t)
        blend = (base.astype(np.float32) * (1.0 - te) + speed.astype(np.float32) * te).astype(np.uint8)
        frames.append(blend)
    frames.extend([speed] * max(hold_end, 0))
    return frames


def render_class_reveal_frames(
    image: np.ndarray,
    label: np.ndarray,
    *,
    speeds: dict[str, float] | None = None,
    frames_per_class: int = 6,
    hold_end: int = 18,
    dim: float = 0.35,
    max_side: int | None = 960,
    slow_first: bool = True,
) -> list[np.ndarray]:
    """
    Покадровая «раскраска» карты: классы появляются по возрастанию/убыванию скорости.

    На фоне — приглушённая исходная карта; сверху накапливается speed-раскраска.
    """
    speeds = copy_speeds(speeds)
    base = _downscale_rgb(image, max_side)
    h, w = base.shape[:2]
    lab = np.asarray(
        Image.fromarray(label.astype(np.uint8)).resize((w, h), Image.Resampling.NEAREST)
    )
    speed_full = render_speed_map(lab, speeds)
    dimmed = (base.astype(np.float32) * dim).astype(np.uint8)

    present = [int(x) for x in np.unique(lab)]
    order = sorted(
        present,
        key=lambda idx: (
            speeds.get(CLASS_NAMES.get(idx, ""), 0.0),
            idx,
        ),
        reverse=not slow_first,
    )

    revealed = np.zeros((h, w), dtype=bool)
    frames: list[np.ndarray] = []
    canvas = dimmed.copy()
    frames.append(canvas.copy())

    for cls in order:
        mask = lab == cls
        if not mask.any():
            continue
        ys = np.where(mask.any(axis=1))[0]
        for k in range(max(frames_per_class, 1)):
            t = (k + 1) / max(frames_per_class, 1)
            if k < frames_per_class - 1 and len(ys):
                # wipe сверху вниз внутри класса
                y_cut = ys[0] + int(np.ceil(t * (ys[-1] - ys[0] + 1)))
                step_mask = mask & (np.arange(h)[:, None] < y_cut)
            else:
                step_mask = mask
            show = revealed | step_mask
            out = dimmed.copy()
            out[show] = speed_full[show]
            frames.append(out)
        revealed |= mask

    frames.extend([speed_full] * max(hold_end, 0))
    return frames


def render_window_scan_frames(
    image: np.ndarray,
    label: np.ndarray,
    *,
    speeds: dict[str, float] | None = None,
    window: int = 128,
    step: int | None = None,
    hold_end: int = 16,
    max_side: int | None = 960,
    cursor_color: tuple[int, int, int] = (255, 255, 255),
) -> list[np.ndarray]:
    """
    Анимация «как модель сканирует карту»: speed-раскраска проявляется окнами sliding-window.
    """
    speeds = copy_speeds(speeds)
    step = int(step if step is not None else max(window // 2, 1))
    base = _downscale_rgb(image, max_side)
    h, w = base.shape[:2]
    lab = np.asarray(
        Image.fromarray(label.astype(np.uint8)).resize((w, h), Image.Resampling.NEAREST)
    )
    speed_full = render_speed_map(lab, speeds)
    dimmed = (base.astype(np.float32) * 0.45).astype(np.uint8)

    win = min(window, h, w)
    coords: list[tuple[int, int]] = []
    for y in range(0, max(h - win, 0) + 1, step):
        for x in range(0, max(w - win, 0) + 1, step):
            coords.append((y, x))
    if (h, w) != (win, win):
        coords.append((max(h - win, 0), max(w - win, 0)))
    coords = list(dict.fromkeys(coords))

    revealed = np.zeros((h, w), dtype=bool)
    frames: list[np.ndarray] = []
    for y, x in coords:
        revealed[y : y + win, x : x + win] = True
        out = dimmed.copy()
        out[revealed] = speed_full[revealed]
        img = Image.fromarray(out)
        draw = ImageDraw.Draw(img)
        draw.rectangle(
            (x, y, x + win - 1, y + win - 1),
            outline=cursor_color,
            width=max(2, win // 64),
        )
        frames.append(np.asarray(img))

    frames.extend([speed_full] * max(hold_end, 0))
    return frames


def animate_segmentation(
    image: np.ndarray,
    label: np.ndarray,
    *,
    mode: str = "morph",
    speeds: dict[str, float] | None = None,
    out_path: str | Path | None = None,
    fps: int = 16,
    max_side: int | None = 960,
    display: bool = True,
    **kwargs: object,
) -> Path:
    """
    Сохранить GIF-анимацию сегментации.

    mode:
      - \"morph\" — кроссфейд карта → скорости
      - \"reveal\" — классы проявляются по скорости
      - \"scan\" — sliding-window заливка
    """
    speeds = copy_speeds(speeds)
    if mode == "morph":
        frames = render_speed_morph_frames(image, label, speeds=speeds, max_side=max_side, **kwargs)  # type: ignore[arg-type]
        stem = "seg_speed_morph"
    elif mode == "reveal":
        frames = render_class_reveal_frames(image, label, speeds=speeds, max_side=max_side, **kwargs)  # type: ignore[arg-type]
        stem = "seg_class_reveal"
    elif mode == "scan":
        frames = render_window_scan_frames(image, label, speeds=speeds, max_side=max_side, **kwargs)  # type: ignore[arg-type]
        stem = "seg_window_scan"
    else:
        raise ValueError(f"Неизвестный mode={mode!r}; ожидается morph|reveal|scan")

    if out_path is None:
        out_path = Path("runs") / "seg_vis" / f"{stem}.gif"
    else:
        out_path = Path(out_path)

    saved = save_animation_gif(frames, out_path.with_suffix(".gif"), fps=fps)
    if display:
        _display_saved_animation(saved)
    return saved


def _class_legend_handles(label: np.ndarray) -> list[Patch]:
    present = sorted(int(x) for x in np.unique(label))
    handles: list[Patch] = []
    for idx in present:
        name = CLASS_NAMES.get(idx, str(idx))
        ru = CLASS_LABELS_RU.get(name, name)
        color = tuple(c / 255.0 for c in CLASS_COLORS[idx])
        handles.append(Patch(facecolor=color, edgecolor="k", label=ru))
    return handles


def show_routes(
    image: np.ndarray,
    routes: Sequence[Route],
    *,
    start_xy: tuple[float, float] | None = None,
    goal_xy: tuple[float, float] | None = None,
    label: np.ndarray | None = None,
    background: str = "image",
    seg_overlay: bool | None = None,
    overlay_alpha: float = 0.35,
    figsize: tuple[float, float] = (12, 10),
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Фон + линии маршрутов + маркеры старт/финиш + легенда.

    background:
      - \"image\" — исходная карта
      - \"segmentation\" — карта, отрендеренная с нуля по классам
      - \"overlay\" — полупрозрачная сегментация поверх исходной
    seg_overlay=True — устаревший алиас для background=\"overlay\".
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if seg_overlay is True:
        background = "overlay"

    if background == "segmentation":
        if label is None:
            raise ValueError("background='segmentation' требует label")
        bg = render_segmentation_map(label)
    elif background == "overlay":
        if label is None:
            raise ValueError("background='overlay' требует label")
        bg = overlay_prediction(image, label, alpha=overlay_alpha)
    else:
        bg = image
    ax.imshow(bg)

    for i, route in enumerate(routes):
        color = ROUTE_COLORS[i % len(ROUTE_COLORS)]
        if len(route.pixels) >= 2:
            xs = [p[0] for p in route.pixels]
            ys = [p[1] for p in route.pixels]
            ax.plot(xs, ys, color=color, linewidth=2.4, solid_capstyle="round", zorder=3)
        elif route.pixels:
            ax.scatter(route.pixels[0][0], route.pixels[0][1], c=[color], s=40, zorder=3)

    if start_xy is not None:
        ax.scatter(
            [start_xy[0]],
            [start_xy[1]],
            c="lime",
            s=120,
            marker="o",
            edgecolors="black",
            linewidths=1.2,
            zorder=5,
            label="старт",
        )
    if goal_xy is not None:
        ax.scatter(
            [goal_xy[0]],
            [goal_xy[1]],
            c="red",
            s=140,
            marker="*",
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
            label="финиш",
        )

    legend_handles: list = []
    for i, route in enumerate(routes):
        color = ROUTE_COLORS[i % len(ROUTE_COLORS)]
        label_txt = (
            f"#{route.rank}: {route.distance_m:.0f} м, {route.time_min:.1f} мин"
        )
        legend_handles.append(Line2D([0], [0], color=color, lw=3, label=label_txt))
    if start_xy is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="lime",
                markeredgecolor="k",
                markersize=10,
                label="старт",
            )
        )
    if goal_xy is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="red",
                markeredgecolor="k",
                markersize=14,
                label="финиш",
            )
        )
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return ax


def show_segmentation(
    label: np.ndarray,
    *,
    image: np.ndarray | None = None,
    mode: str = "rendered",
    alpha: float = 0.45,
    figsize: tuple[float, float] = (12, 10),
    show_legend: bool = True,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Показать сегментацию.

    mode:
      - \"rendered\" — карта с нуля по классам (по умолчанию)
      - \"overlay\" — полупрозрачно поверх image (нужен image)
    """
    if mode == "overlay":
        if image is None:
            raise ValueError("mode='overlay' требует image")
        canvas = overlay_prediction(image, label, alpha=alpha)
        default_title = "Сегментация (оверлей)"
    else:
        canvas = render_segmentation_map(label)
        default_title = "Сегментация (рендер)"

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.imshow(canvas)
    ax.set_axis_off()
    ax.set_title(title or default_title)
    if show_legend:
        handles = _class_legend_handles(label)
        if handles:
            ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)
    return ax


def show_route_table(route: Route, segments: Sequence[RouteSegment] | None = None):
    """Печать сводки + DataFrame сегментов (для отображения в notebook)."""
    segs = list(segments if segments is not None else (route.segments or []))
    summary = summarize_route(route, segs)
    print(
        f"Маршрут #{summary['rank']}: "
        f"{summary['distance_m']:.0f} м, "
        f"{summary['time_min']:.1f} мин "
        f"({summary['n_segments']} участков)"
    )
    if summary["terrain_m"]:
        parts = [f"{k}: {v:.0f} м" for k, v in summary["terrain_m"].items()]
        print("Состав: " + "; ".join(parts))
    df = segments_to_dataframe(segs)
    return df


def format_route_description(route: Route, segments: Sequence[RouteSegment] | None = None) -> str:
    segs = list(segments if segments is not None else (route.segments or []))
    lines = [
        f"Маршрут #{route.rank}: {route.distance_m:.0f} м, {route.time_min:.1f} мин",
        "",
    ]
    for i, s in enumerate(segs, 1):
        lines.append(
            f"{i}. {s.label_ru}: {s.distance_m:.0f} м "
            f"({s.time_s:.0f} с, {s.share * 100:.0f}%)"
        )
    return "\n".join(lines)


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0) * 255.0
    return np.ascontiguousarray(arr.astype(np.uint8))


def _cell_center_xy(col: int, row: int, cell_size: int) -> tuple[float, float]:
    return (col + 0.5) * cell_size, (row + 0.5) * cell_size


def _get_cmap(name: str):
    try:
        return plt.colormaps[name]
    except (AttributeError, KeyError):
        return cm.get_cmap(name)


def _wave_rgba(
    visit_order: np.ndarray,
    threshold: int,
    n_closed: int,
    cmap,
    alpha: float,
) -> np.ndarray:
    """RGBA-оверлей волны в разрешении cost-грида (uint8)."""
    gh, gw = visit_order.shape
    out = np.zeros((gh, gw, 4), dtype=np.uint8)
    visible = (visit_order >= 0) & (visit_order < threshold)
    if not visible.any():
        return out
    denom = max(n_closed - 1, 1)
    norm = visit_order.astype(np.float64) / denom
    rgba = cmap(np.clip(norm, 0.0, 1.0))
    out[visible, :3] = (rgba[visible, :3] * 255.0).astype(np.uint8)
    out[visible, 3] = int(round(255 * alpha))
    return out


def render_search_frames(
    image: np.ndarray,
    trace: SearchTrace,
    *,
    start_xy: tuple[float, float] | None = None,
    goal_xy: tuple[float, float] | None = None,
    n_wave_frames: int = 90,
    n_path_frames: int = 45,
    hold_frames: int = 24,
    wave_alpha: float = 0.55,
    path_color: tuple[int, int, int] = (0, 220, 255),
    path_width: int = 4,
    cmap_name: str = "plasma",
    max_side: int | None = 1280,
) -> list[np.ndarray]:
    """
    Кадры анимации: распространение closed-множества A* → отрисовка пути.

    max_side — если задан, кадры уменьшаются так, чтобы длинная сторона ≤ max_side
    (ускоряет GIF и уменьшает размер файла).

    Returns:
        список RGB uint8 кадров.
    """
    base = _as_uint8_rgb(image)
    h0, w0 = base.shape[:2]
    scale = 1.0
    if max_side is not None and max(h0, w0) > max_side:
        scale = max_side / float(max(h0, w0))
        new_w = max(1, int(round(w0 * scale)))
        new_h = max(1, int(round(h0 * scale)))
        base = np.asarray(
            Image.fromarray(base).resize((new_w, new_h), Image.Resampling.BILINEAR)
        )

    def _xy(x: float, y: float) -> tuple[float, float]:
        return x * scale, y * scale

    h, w = base.shape[:2]
    cell_px = max(trace.cell_size * scale, 1.0)
    dim = (base.astype(np.float32) * 0.55).astype(np.uint8)
    cmap = _get_cmap(cmap_name)

    if start_xy is None:
        sx, sy = _xy(*_cell_center_xy(*trace.start_cell, trace.cell_size))
    else:
        sx, sy = _xy(*start_xy)
    if goal_xy is None:
        gx, gy = _xy(*_cell_center_xy(*trace.goal_cell, trace.cell_size))
    else:
        gx, gy = _xy(*goal_xy)

    n_closed = max(int(trace.n_closed), 1)
    wave_steps = max(int(n_wave_frames), 1)
    frames: list[np.ndarray] = []
    marker_r = max(6, int(round(cell_px)))
    line_w = max(2, int(round(path_width * scale)))

    def _draw_markers(canvas: np.ndarray) -> np.ndarray:
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        r = marker_r
        draw.ellipse((sx - r, sy - r, sx + r, sy + r), fill=(50, 255, 80), outline=(0, 0, 0), width=2)
        star_r = r + 2
        draw.polygon(
            [
                (gx, gy - star_r),
                (gx + star_r * 0.7, gy),
                (gx, gy + star_r),
                (gx - star_r * 0.7, gy),
            ],
            fill=(255, 40, 40),
            outline=(0, 0, 0),
        )
        return np.asarray(img)

    def _blend_wave(threshold: int) -> np.ndarray:
        overlay = _wave_rgba(trace.visit_order, threshold, n_closed, cmap, wave_alpha)
        overlay_img = Image.fromarray(overlay, mode="RGBA").resize(
            (w, h), Image.Resampling.NEAREST
        )
        base_img = Image.fromarray(dim).convert("RGBA")
        composed = Image.alpha_composite(base_img, overlay_img).convert("RGB")
        return _draw_markers(np.asarray(composed))

    for i in range(wave_steps):
        t = (i + 1) / wave_steps
        threshold = int(np.ceil((t ** 0.85) * n_closed))
        frames.append(_blend_wave(threshold))

    full = _blend_wave(n_closed)

    path_pixels = list(trace.route.pixels) if trace.route is not None else []
    if len(path_pixels) >= 2 and n_path_frames > 0:
        for i in range(n_path_frames):
            t = (i + 1) / n_path_frames
            n_pts = max(2, int(np.ceil(t * len(path_pixels))))
            canvas = Image.fromarray(full.copy())
            draw = ImageDraw.Draw(canvas)
            pts = [_xy(p[0], p[1]) for p in path_pixels[:n_pts]]
            draw.line(pts, fill=path_color, width=line_w, joint="curve")
            frames.append(_draw_markers(np.asarray(canvas)))
        final = frames[-1]
    else:
        final = full
        frames.append(final)

    for _ in range(max(hold_frames, 0)):
        frames.append(final)

    return frames


def save_animation_gif(frames: Sequence[np.ndarray], path: str | Path, *, fps: int = 20) -> Path:
    """Сохранить кадры в GIF (удобно смотреть прямо в Jupyter)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("Нет кадров для сохранения")
    imgs = [Image.fromarray(_as_uint8_rgb(f)) for f in frames]
    duration_ms = max(int(round(1000 / max(fps, 1))), 20)
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return path


def save_animation_mp4(frames: Sequence[np.ndarray], path: str | Path, *, fps: int = 20) -> Path:
    """Сохранить кадры в MP4 через OpenCV (если кодек доступен)."""
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("Нет кадров для сохранения")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Не удалось открыть VideoWriter для {path}")
    try:
        for frame in frames:
            bgr = cv2.cvtColor(_as_uint8_rgb(frame), cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()
    return path


def animate_path_search(
    image: np.ndarray,
    cost: CostGrid,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    *,
    out_path: str | Path | None = None,
    format: str = "gif",
    fps: int = 20,
    n_wave_frames: int = 90,
    n_path_frames: int = 45,
    hold_frames: int = 24,
    max_side: int | None = 1280,
    display: bool = True,
) -> tuple[SearchTrace, Path]:
    """
    Прогнать A* с трассировкой и сохранить анимацию волны + пути.

    format: \"gif\" | \"mp4\"
    display=True — показать результат в Jupyter (IPython.display).
    """
    trace = astar_traced(cost, start_xy, goal_xy)
    frames = render_search_frames(
        image,
        trace,
        start_xy=start_xy,
        goal_xy=goal_xy,
        n_wave_frames=n_wave_frames,
        n_path_frames=n_path_frames,
        hold_frames=hold_frames,
        max_side=max_side,
    )

    if out_path is None:
        out_dir = Path("runs") / "routing_vis"
        stem = "astar_wave"
        out_path = out_dir / f"{stem}.{format}"
    else:
        out_path = Path(out_path)

    if format == "mp4":
        saved = save_animation_mp4(frames, out_path, fps=fps)
    else:
        saved = save_animation_gif(frames, out_path.with_suffix(".gif"), fps=fps)

    if display:
        _display_saved_animation(saved)

    return trace, saved
