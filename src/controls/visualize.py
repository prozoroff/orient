"""Визуализация найденных КП и старта (+ анимация circle-first)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.controls.types import (
    ControlPoint,
    CourseDetection,
    CvDetectTrace,
    StartFinish,
    VlmDetectTrace,
)

Image.MAX_IMAGE_PIXELS = None


def draw_controls(
    image_rgb: np.ndarray,
    controls: Sequence[ControlPoint] | CourseDetection,
    *,
    start: StartFinish | None = None,
    radius: int = 10,
    thickness: int = 2,
) -> np.ndarray:
    """Копия RGB с маркерами КП (+ старт, если есть)."""
    if isinstance(controls, CourseDetection):
        start = start if start is not None else controls.start
        controls = controls.controls

    out = image_rgb.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)

    if start is not None:
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
        cv2.circle(out, (cx, cy), 3, color_s, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            out,
            "S/F",
            (cx + radius + 2, cy - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            "S/F",
            (cx + radius + 2, cy - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color_s,
            1,
            cv2.LINE_AA,
        )

    for cp in controls:
        if cp.source == "circle":
            color = (0, 200, 80)
        elif cp.source == "vlm":
            color = (220, 140, 40)  # оранжевый — VLM без snap к кольцу
        else:
            color = (40, 120, 255)
        center = (int(round(cp.x)), int(round(cp.y)))
        cv2.circle(out, center, radius, color, thickness, lineType=cv2.LINE_AA)
        cv2.circle(out, center, 2, color, -1, lineType=cv2.LINE_AA)
        label = f"{cp.number}"
        cv2.putText(
            out,
            label,
            (center[0] + radius + 2, center[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            label,
            (center[0] + radius + 2, center[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


# ---------------------------------------------------------------------------
# Анимация circle-first (как animate_path_search в routing)
# ---------------------------------------------------------------------------


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


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


def _resize_rgb(image: np.ndarray, max_side: int | None) -> tuple[np.ndarray, float]:
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


def _draw_caption(canvas: np.ndarray, text: str, *, scale: float = 1.0) -> np.ndarray:
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


def _blend_pink_mask(
    base: np.ndarray,
    pink: np.ndarray,
    *,
    alpha: float,
    color: tuple[int, int, int] = (255, 40, 180),
) -> np.ndarray:
    out = base.astype(np.float32)
    mask = pink > 0
    if mask.any():
        overlay = np.zeros_like(out)
        overlay[mask] = color
        a = float(np.clip(alpha, 0.0, 1.0))
        out[mask] = (1.0 - a) * out[mask] + a * overlay[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_ring(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    radius: float,
    *,
    color: tuple[int, int, int],
    width: int = 2,
    fill_center: bool = True,
) -> None:
    r = max(radius, 2.0)
    box = (cx - r, cy - r, cx + r, cy + r)
    draw.ellipse(box, outline=color, width=width)
    if fill_center:
        cr = max(2.0, min(4.0, r * 0.2))
        draw.ellipse((cx - cr, cy - cr, cx + cr, cy + cr), fill=color)


def _draw_cp_marker(
    draw: ImageDraw.ImageDraw,
    cp: ControlPoint,
    *,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    color: tuple[int, int, int] = (0, 210, 90),
    radius: float = 10.0,
) -> None:
    _draw_ring(draw, cp.x, cp.y, radius, color=color, width=3)
    label = str(cp.number)
    tx, ty = cp.x + radius + 3, cp.y - radius - 2
    # обводка для читаемости
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((tx + dx, ty + dy), label, fill=(20, 20, 20), font=font)
    draw.text((tx, ty), label, fill=color, font=font)


def _draw_start(
    draw: ImageDraw.ImageDraw,
    start: StartFinish,
    *,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> None:
    side = start.side if start.side > 0 else 28.0
    h_tri = side * np.sqrt(3) / 2
    pts = [
        (start.x, start.y - 2 * h_tri / 3),
        (start.x - side / 2, start.y + h_tri / 3),
        (start.x + side / 2, start.y + h_tri / 3),
    ]
    color = (220, 60, 40)
    draw.polygon(pts, outline=color)
    draw.ellipse((start.x - 3, start.y - 3, start.x + 3, start.y + 3), fill=color)
    draw.text((start.x + 12, start.y - 10), "S/F", fill=color, font=font)


def render_cv_detect_frames(
    image: np.ndarray,
    trace: CvDetectTrace,
    *,
    n_mask_frames: int = 24,
    n_ring_frames: int = 40,
    frames_per_ocr: int = 6,
    hold_frames: int = 28,
    max_side: int | None = 1280,
    pink_alpha: float = 0.55,
) -> list[np.ndarray]:
    """
    Кадры анимации circle-first:
    маска → кольца → OCR у колец → digit-assist → финал.
    """
    base, scale = _resize_rgb(image, max_side)
    h, w = base.shape[:2]
    dim = (base.astype(np.float32) * 0.62).astype(np.uint8)

    pink = np.asarray(trace.pink)
    if pink.shape[:2] != (image.shape[0], image.shape[1]):
        raise ValueError("pink mask size != image size")
    if scale != 1.0:
        pink = np.asarray(
            Image.fromarray(pink).resize((w, h), Image.Resampling.NEAREST)
        )

    def S(v: float) -> float:
        return float(v) * scale

    font = _load_ui_font(max(13, int(round(18 * scale))))
    frames: list[np.ndarray] = []

    def push(canvas: np.ndarray, caption: str, n: int = 1) -> None:
        framed = _draw_caption(canvas, caption, scale=scale)
        for _ in range(max(n, 1)):
            frames.append(framed)

    # 1) вход
    push(base.copy(), "1/5  Входная карта", n=max(8, hold_frames // 3))

    # 2) magenta-маска
    steps = max(int(n_mask_frames), 1)
    for i in range(steps):
        a = pink_alpha * ((i + 1) / steps)
        canvas = _blend_pink_mask(dim, pink, alpha=a)
        push(canvas, "2/5  Magenta-маска (HSV)", n=1)
    mask_full = _blend_pink_mask(dim, pink, alpha=pink_alpha)
    push(mask_full, "2/5  Magenta-маска (HSV)", n=max(6, hold_frames // 4))

    # 3) кольца появляются
    rings = list(trace.rings)
    n_rings = max(len(rings), 1)
    ring_steps = max(int(n_ring_frames), n_rings)
    accepted: list[ControlPoint] = []

    for i in range(ring_steps):
        n_show = int(np.ceil((i + 1) / ring_steps * n_rings))
        img = Image.fromarray(mask_full.copy())
        draw = ImageDraw.Draw(img)
        for cx, cy, radius, _hollow in rings[:n_show]:
            _draw_ring(
                draw,
                S(cx),
                S(cy),
                S(radius),
                color=(80, 220, 255),
                width=max(2, int(round(2 * scale))),
            )
        push(
            np.asarray(img),
            f"3/5  Hough-кольца  ({min(n_show, n_rings)}/{n_rings})",
            n=1,
        )

    rings_full = Image.fromarray(mask_full.copy())
    draw_r = ImageDraw.Draw(rings_full)
    for cx, cy, radius, _hollow in rings:
        _draw_ring(
            draw_r,
            S(cx),
            S(cy),
            S(radius),
            color=(80, 220, 255),
            width=max(2, int(round(2 * scale))),
        )
    rings_only = np.asarray(rings_full)
    push(rings_only, f"3/5  Hough-кольца  ({n_rings})", n=max(6, hold_frames // 4))

    marker_r = max(8.0, 10.0 * scale)

    def _scaled_cp(cp: ControlPoint) -> ControlPoint:
        return ControlPoint(
            number=cp.number, x=S(cp.x), y=S(cp.y), score=cp.score, source=cp.source
        )

    def _canvas_with_accepted() -> tuple[Image.Image, ImageDraw.ImageDraw]:
        img = Image.fromarray(rings_only.copy())
        draw = ImageDraw.Draw(img)
        for cp in accepted:
            _draw_cp_marker(draw, _scaled_cp(cp), font=font, radius=marker_r)
        return img, draw

    # 4) OCR у каждого кольца (принятые первыми, затем часть отказов)
    ocr_events = list(trace.ring_events)
    accepted_events = [e for e in ocr_events if e.control is not None]
    other_events = [e for e in ocr_events if e.control is None][:12]
    show_events = accepted_events + other_events

    for idx, ev in enumerate(show_events):
        img, draw = _canvas_with_accepted()

        # текущее кольцо
        _draw_ring(
            draw,
            S(ev.cx),
            S(ev.cy),
            S(ev.radius),
            color=(255, 230, 40),
            width=max(3, int(round(3 * scale))),
        )
        # OCR-окна
        for win in ev.windows:
            color = (255, 180, 40) if win.kind == "directional" else (180, 120, 255)
            box = (S(win.x0), S(win.y0), S(win.x1), S(win.y1))
            draw.rectangle(box, outline=color, width=max(1, int(round(2 * scale))))

        caption = f"4/5  OCR у кольца  ({idx + 1}/{len(show_events)})"
        hold = max(2, frames_per_ocr // 2)
        push(np.asarray(img), caption, n=hold)

        # hit bbox + номер
        if ev.hit_bbox is not None and ev.hit_number is not None:
            x0, y0, x1, y1 = ev.hit_bbox
            draw.rectangle(
                (S(x0), S(y0), S(x1), S(y1)),
                outline=(40, 220, 120),
                width=max(2, int(round(2 * scale))),
            )
            draw.text(
                (S(x0), S(y0) - 16 * scale),
                f"{ev.hit_number} ({ev.hit_conf:.2f})",
                fill=(40, 220, 120),
                font=font,
            )
            push(np.asarray(img), caption + f"  → {ev.hit_number}", n=hold)

        if ev.control is not None:
            accepted.append(ev.control)
            img, draw = _canvas_with_accepted()
            push(
                np.asarray(img),
                f"4/5  Принят КП {ev.control.number}",
                n=max(3, frames_per_ocr),
            )

    # 5) digit-assist
    for da in trace.digit_assist:
        img, draw = _canvas_with_accepted()
        x0, y0, x1, y1 = da.hit_bbox
        draw.rectangle(
            (S(x0), S(y0), S(x1), S(y1)),
            outline=(255, 140, 40),
            width=max(2, int(round(2 * scale))),
        )
        accepted.append(da.control)
        _draw_cp_marker(
            draw,
            _scaled_cp(da.control),
            font=font,
            radius=marker_r,
            color=(255, 160, 40),
        )
        push(
            np.asarray(img),
            f"4/5  Digit-assist → КП {da.control.number}",
            n=max(4, frames_per_ocr),
        )

    # 6) финал на исходной карте
    final = Image.fromarray(base.copy())
    draw = ImageDraw.Draw(final)
    for cp in trace.result.controls:
        _draw_cp_marker(
            draw,
            ControlPoint(
                number=cp.number, x=S(cp.x), y=S(cp.y), score=cp.score, source=cp.source
            ),
            font=font,
            radius=max(8.0, 10.0 * scale),
        )
    if trace.result.start is not None:
        st = trace.result.start
        _draw_start(
            draw,
            StartFinish(x=S(st.x), y=S(st.y), score=st.score, side=S(st.side)),
            font=font,
        )
    n_cp = len(trace.result.controls)
    push(
        np.asarray(final),
        f"5/5  Итог: {n_cp} КП"
        + (" + старт" if trace.result.start is not None else ""),
        n=max(hold_frames, 20),
    )
    return frames


def save_animation_gif(frames: Sequence[np.ndarray], path: str | Path, *, fps: int = 16) -> Path:
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


def save_animation_mp4(frames: Sequence[np.ndarray], path: str | Path, *, fps: int = 16) -> Path:
    """Сохранить кадры в MP4 через OpenCV."""
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


def animate_detect_controls_cv(
    image: str | Path | np.ndarray,
    *,
    prior_mask: np.ndarray | None = None,
    sensitivity: str = "recall",
    min_ocr_conf: float | None = None,
    find_start: bool = True,
    circle_diameter_px: float | None = None,
    out_path: str | Path | None = None,
    format: str = "gif",
    fps: int = 16,
    n_mask_frames: int = 24,
    n_ring_frames: int = 40,
    frames_per_ocr: int = 6,
    hold_frames: int = 28,
    max_side: int | None = 1280,
    display: bool = True,
    reader: Any | None = None,
    gpu: bool | None = None,
) -> tuple[CourseDetection, Path]:
    """
    Прогнать circle-first детектор с трассировкой и сохранить анимацию.

    format: \"gif\" | \"mp4\"
    display=True — показать результат в Jupyter.
    """
    from src.controls.detect import _load_rgb, detect_controls_cv_trace

    rgb = _load_rgb(image)
    trace = detect_controls_cv_trace(
        rgb,
        prior_mask=prior_mask,
        sensitivity=sensitivity,
        min_ocr_conf=min_ocr_conf,
        find_start=find_start,
        circle_diameter_px=circle_diameter_px,
        reader=reader,
        gpu=gpu,
    )
    frames = render_cv_detect_frames(
        rgb,
        trace,
        n_mask_frames=n_mask_frames,
        n_ring_frames=n_ring_frames,
        frames_per_ocr=frames_per_ocr,
        hold_frames=hold_frames,
        max_side=max_side,
    )

    if out_path is None:
        out_path = Path("runs") / "controls_vis" / f"cv_detect.{format}"
    else:
        out_path = Path(out_path)

    if format == "mp4":
        saved = save_animation_mp4(frames, out_path, fps=fps)
    else:
        saved = save_animation_gif(frames, out_path.with_suffix(".gif"), fps=fps)

    if display:
        _display_animation(saved)

    return trace.result, saved


def render_vlm_detect_frames(
    image: np.ndarray,
    trace: VlmDetectTrace,
    *,
    frames_per_tile: int = 10,
    frames_per_hit: int = 3,
    hold_frames: int = 28,
    max_side: int | None = 1280,
    dim_factor: float = 0.45,
) -> list[np.ndarray]:
    """
    Кадры анимации VLM: сетка тайлов → обход тайлов → сырые hits → итог.

    Без refine: маркеры оранжевые (source=vlm).
    """
    base, scale = _resize_rgb(image, max_side)
    dim = (base.astype(np.float32) * float(np.clip(dim_factor, 0.2, 1.0))).astype(np.uint8)

    def S(v: float) -> float:
        return float(v) * scale

    font = _load_ui_font(max(13, int(round(18 * scale))))
    frames: list[np.ndarray] = []
    marker_r = max(8.0, 10.0 * scale)
    line_w = max(2, int(round(3 * scale)))
    vlm_color = (220, 140, 40)
    reject_color = (160, 160, 160)
    tile_color = (80, 200, 255)

    def push(canvas: np.ndarray, caption: str, n: int = 1) -> None:
        framed = _draw_caption(canvas, caption, scale=scale)
        for _ in range(max(n, 1)):
            frames.append(framed)

    def _draw_hit(
        draw: ImageDraw.ImageDraw,
        x: float,
        y: float,
        number: int,
        *,
        color: tuple[int, int, int],
        square: bool = True,
    ) -> None:
        cx, cy = S(x), S(y)
        r = marker_r
        if square:
            draw.rectangle((cx - r, cy - r, cx + r, cy + r), outline=color, width=line_w)
        else:
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=line_w)
        draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=color)
        label = str(number)
        tx, ty = cx + r + 3, cy - r - 2
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            draw.text((tx + dx, ty + dy), label, fill=(20, 20, 20), font=font)
        draw.text((tx, ty), label, fill=color, font=font)

    # 1) вход
    mode = "без refine" if not trace.refine_circles else "с refine"
    push(base.copy(), f"1/4  VLM ({mode}): входная карта", n=max(8, hold_frames // 3))

    # 2) сетка тайлов
    grid = Image.fromarray(base.copy())
    draw = ImageDraw.Draw(grid)
    for ev in trace.tile_events:
        draw.rectangle(
            (S(ev.x0), S(ev.y0), S(ev.x1), S(ev.y1)),
            outline=tile_color,
            width=max(1, line_w - 1),
        )
    n_tiles = len(trace.tile_events)
    push(
        np.asarray(grid),
        f"2/4  Тайлы  {n_tiles} шт  (size={trace.tile_size}, overlap={trace.overlap})",
        n=max(10, hold_frames // 2),
    )

    # накопленные kept-хиты (по номеру — лучший)
    kept_by_number: dict[int, tuple[float, float, float]] = {}  # num -> x,y,conf
    start_best: tuple[float, float, float] | None = None

    # 3) обход тайлов
    for ti, ev in enumerate(trace.tile_events):
        # фон: затемнённый + яркий текущий тайл
        canvas = dim.copy()
        x0, y0, x1, y1 = (
            int(round(S(ev.x0))),
            int(round(S(ev.y0))),
            int(round(S(ev.x1))),
            int(round(S(ev.y1))),
        )
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(base.shape[1], x1), min(base.shape[0], y1)
        canvas[y0:y1, x0:x1] = base[y0:y1, x0:x1]

        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        draw.rectangle((x0, y0, x1, y1), outline=tile_color, width=line_w)

        # уже накопленные КП (полупрозрачно на dim-зоне тоже видны)
        for num, (hx, hy, _hc) in kept_by_number.items():
            _draw_hit(draw, hx, hy, num, color=vlm_color)

        caption = f"3/4  VLM тайл {ti + 1}/{n_tiles}"
        push(np.asarray(img), caption, n=max(2, frames_per_tile // 2))

        # hits появляются по одному
        for hit in ev.hits:
            color = vlm_color if hit.kept else reject_color
            _draw_hit(draw, hit.x, hit.y, hit.number, color=color, square=True)
            if hit.kept:
                prev = kept_by_number.get(hit.number)
                if prev is None or hit.confidence >= prev[2]:
                    kept_by_number[hit.number] = (hit.x, hit.y, hit.confidence)
            tag = "✓" if hit.kept else "×"
            push(
                np.asarray(img),
                f"{caption}  → #{hit.number} {tag} ({hit.confidence:.2f})",
                n=max(2, frames_per_hit),
            )

        if ev.start_xy is not None:
            sx, sy, sc = ev.start_xy
            if start_best is None or sc >= start_best[2]:
                start_best = (sx, sy, sc)
            # крестик старта
            cx, cy = S(sx), S(sy)
            r = marker_r + 2
            draw.line((cx - r, cy, cx + r, cy), fill=(220, 60, 40), width=line_w)
            draw.line((cx, cy - r, cx, cy + r), fill=(220, 60, 40), width=line_w)
            draw.text((cx + r + 2, cy - r), "S?", fill=(220, 60, 40), font=font)
            push(np.asarray(img), f"{caption}  → старт?", n=max(2, frames_per_hit))

        push(
            np.asarray(img),
            f"3/4  Тайл {ti + 1}/{n_tiles} готов  (КП: {len(kept_by_number)})",
            n=max(2, frames_per_tile // 2),
        )

    # 4) финал по result (dedup)
    final = Image.fromarray(base.copy())
    draw = ImageDraw.Draw(final)
    for cp in trace.result.controls:
        _draw_hit(draw, cp.x, cp.y, cp.number, color=vlm_color, square=True)
    if trace.result.start is not None:
        st = trace.result.start
        _draw_start(
            draw,
            StartFinish(x=S(st.x), y=S(st.y), score=st.score, side=S(st.side) if st.side else 0.0),
            font=font,
        )
    n_cp = len(trace.result.controls)
    push(
        np.asarray(final),
        f"4/4  Итог VLM ({mode}): {n_cp} КП"
        + (" + старт" if trace.result.start is not None else ""),
        n=max(hold_frames, 20),
    )
    return frames


def _display_animation(saved: Path) -> None:
    try:
        from IPython.display import Image as IPyImage
        from IPython.display import Video, display as ipy_display

        if saved.suffix.lower() == ".gif":
            ipy_display(IPyImage(filename=str(saved)))
        else:
            ipy_display(
                Video(str(saved), embed=True, html_attributes="controls loop autoplay")
            )
    except Exception:
        pass


def animate_detect_controls_vlm(
    image: str | Path | np.ndarray | None = None,
    *,
    trace: VlmDetectTrace | None = None,
    prior_mask: np.ndarray | None = None,
    provider: str = "yandex",
    model: str | None = None,
    refine_circles: bool = False,
    find_start: bool = True,
    min_confidence: float = 0.35,
    circle_diameter_px: float | None = 55.0,
    tile_size: int = 1024,
    overlap: int = 160,
    out_path: str | Path | None = None,
    format: str = "gif",
    fps: int = 16,
    frames_per_tile: int = 10,
    frames_per_hit: int = 3,
    hold_frames: int = 28,
    max_side: int | None = 1280,
    display: bool = True,
    api_key: str | None = None,
    folder_id: str | None = None,
    client: Any | None = None,
) -> tuple[CourseDetection, Path]:
    """
    Анимация VLM-детекции (по умолчанию без refine).

    Передайте готовый ``trace`` из ``detect_controls_vlm_trace``, чтобы не
    дергать API повторно. Иначе нужен ``image`` — будет новый вызов модели.
    """
    from src.controls.detect_vlm import _load_rgb, detect_controls_vlm_trace

    if trace is None:
        if image is None:
            raise ValueError("Нужен image или готовый trace")
        rgb = _load_rgb(image)
        trace = detect_controls_vlm_trace(
            rgb,
            provider=provider,
            model=model,
            api_key=api_key,
            folder_id=folder_id,
            prior_mask=prior_mask,
            tile_size=tile_size,
            overlap=overlap,
            refine_circles=refine_circles,
            find_start=find_start,
            min_confidence=min_confidence,
            circle_diameter_px=circle_diameter_px,
            client=client,
        )
    else:
        if image is None:
            raise ValueError("Для рендера кадров нужен image вместе с trace")
        rgb = _load_rgb(image)

    frames = render_vlm_detect_frames(
        rgb,
        trace,
        frames_per_tile=frames_per_tile,
        frames_per_hit=frames_per_hit,
        hold_frames=hold_frames,
        max_side=max_side,
    )

    stem = "vlm_raw" if not trace.refine_circles else "vlm_refine"
    if out_path is None:
        out_path = Path("runs") / "controls_vis" / f"{stem}.{format}"
    else:
        out_path = Path(out_path)

    if format == "mp4":
        saved = save_animation_mp4(frames, out_path, fps=fps)
    else:
        saved = save_animation_gif(frames, out_path.with_suffix(".gif"), fps=fps)

    if display:
        _display_animation(saved)

    return trace.result, saved


def render_vlm_refine_frames(
    image: np.ndarray,
    raw: CourseDetection,
    *,
    prior_mask: np.ndarray | None = None,
    circle_diameter_px: float | None = 55.0,
    n_mask_frames: int = 20,
    frames_per_cp: int = 8,
    hold_frames: int = 28,
    max_side: int | None = 1280,
    pink_alpha: float = 0.45,
) -> tuple[list[np.ndarray], CourseDetection]:
    """
    Кадры: сырые VLM-точки → magenta-маска → snap/refine по каждому КП → итог.

    Без API: только ``_snap_to_ring`` / ``refine_ring_center``.
    """
    from src.controls.detect_vlm import _snap_to_ring, estimate_cp_radius_px
    from src.controls.pink_mask import merge_prior, pink_mask_hsv

    base, scale = _resize_rgb(image, max_side)
    rgb_full = _as_uint8_rgb(image)

    pink = pink_mask_hsv(
        rgb_full,
        use_clahe=True,
        min_saturation=25,
        min_value=30,
        preserve_thin=True,
    )
    pink = merge_prior(pink, prior_mask)
    expected_r = estimate_cp_radius_px(pink, hint_diameter=circle_diameter_px)
    search_r = max(2.2 * expected_r, 1.5 * max(8, int(round(1.35 * expected_r))))

    if scale != 1.0:
        h, w = base.shape[:2]
        pink_vis = np.asarray(
            Image.fromarray(pink).resize((w, h), Image.Resampling.NEAREST)
        )
    else:
        pink_vis = pink

    def S(v: float) -> float:
        return float(v) * scale

    font = _load_ui_font(max(13, int(round(18 * scale))))
    frames: list[np.ndarray] = []
    marker_r = max(8.0, 10.0 * scale)
    line_w = max(2, int(round(3 * scale)))
    raw_color = (220, 140, 40)
    ref_color = (0, 210, 90)
    search_color = (80, 200, 255)

    def push(canvas: np.ndarray, caption: str, n: int = 1) -> None:
        framed = _draw_caption(canvas, caption, scale=scale)
        for _ in range(max(n, 1)):
            frames.append(framed)

    def _marker(
        draw: ImageDraw.ImageDraw,
        x: float,
        y: float,
        number: int,
        *,
        color: tuple[int, int, int],
        square: bool,
    ) -> None:
        cx, cy = S(x), S(y)
        r = marker_r
        if square:
            draw.rectangle((cx - r, cy - r, cx + r, cy + r), outline=color, width=line_w)
        else:
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=line_w)
            _draw_ring(
                draw, cx, cy, S(expected_r), color=color, width=line_w, fill_center=False
            )
        draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=color)
        label = str(number)
        tx, ty = cx + r + 3, cy - r - 2
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            draw.text((tx + dx, ty + dy), label, fill=(20, 20, 20), font=font)
        draw.text((tx, ty), label, fill=color, font=font)

    # 1) сырые точки
    img0 = Image.fromarray(base.copy())
    draw0 = ImageDraw.Draw(img0)
    for cp in raw.controls:
        _marker(draw0, cp.x, cp.y, cp.number, color=raw_color, square=True)
    push(
        np.asarray(img0),
        f"1/4  VLM raw  ({len(raw.controls)} КП)",
        n=max(10, hold_frames // 2),
    )

    # 2) маска
    steps = max(int(n_mask_frames), 1)
    dim = (base.astype(np.float32) * 0.65).astype(np.uint8)
    for i in range(steps):
        a = pink_alpha * ((i + 1) / steps)
        canvas = _blend_pink_mask(dim, pink_vis, alpha=a)
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        for cp in raw.controls:
            _marker(draw, cp.x, cp.y, cp.number, color=raw_color, square=True)
        push(np.asarray(img), "2/4  Magenta-маска для snap", n=1)
    mask_base = _blend_pink_mask(dim, pink_vis, alpha=pink_alpha)
    push(mask_base, "2/4  Magenta-маска для snap", n=max(6, hold_frames // 4))

    # 3) refine по каждому КП
    refined_list: list[ControlPoint] = []
    snaps: list[tuple[ControlPoint, ControlPoint, bool, float]] = []

    for cp in raw.controls:
        sx, sy, hollow, snapped = _snap_to_ring(
            pink, cp.x, cp.y, expected_radius=expected_r
        )
        if snapped:
            score = float(cp.score) * (0.55 + 0.45 * min(1.0, hollow))
            ref_cp = ControlPoint(
                number=cp.number,
                x=float(sx),
                y=float(sy),
                score=score,
                source="circle",
            )
        else:
            ref_cp = cp
        dist = float(np.hypot(ref_cp.x - cp.x, ref_cp.y - cp.y))
        snaps.append((cp, ref_cp, snapped, dist))

    for idx, (cp, ref_cp, snapped, dist) in enumerate(snaps):
        img = Image.fromarray(mask_base.copy())
        draw = ImageDraw.Draw(img)

        for _prev_raw, prev_ref, prev_ok, _ in snaps[:idx]:
            color = ref_color if prev_ok else raw_color
            _marker(
                draw,
                prev_ref.x,
                prev_ref.y,
                prev_ref.number,
                color=color,
                square=not prev_ok,
            )

        cx, cy = S(cp.x), S(cp.y)
        sr = S(search_r)
        draw.ellipse(
            (cx - sr, cy - sr, cx + sr, cy + sr),
            outline=search_color,
            width=max(1, line_w - 1),
        )
        _marker(draw, cp.x, cp.y, cp.number, color=raw_color, square=True)
        push(
            np.asarray(img),
            f"3/4  Snap КП {cp.number}  ({idx + 1}/{len(snaps)})",
            n=max(3, frames_per_cp // 2),
        )

        if snapped and dist >= 0.5:
            rx, ry = S(ref_cp.x), S(ref_cp.y)
            draw.line((cx, cy, rx, ry), fill=ref_color, width=line_w)
            draw.ellipse((rx - 3, ry - 3, rx + 3, ry + 3), fill=ref_color)
            _marker(draw, ref_cp.x, ref_cp.y, ref_cp.number, color=ref_color, square=False)
            push(
                np.asarray(img),
                f"3/4  КП {cp.number}: refine  Δ={dist:.1f}px",
                n=max(4, frames_per_cp),
            )
        elif snapped:
            _marker(draw, ref_cp.x, ref_cp.y, ref_cp.number, color=ref_color, square=False)
            push(
                np.asarray(img),
                f"3/4  КП {cp.number}: уже в центре",
                n=max(3, frames_per_cp // 2),
            )
        else:
            push(
                np.asarray(img),
                f"3/4  КП {cp.number}: snap не найден",
                n=max(3, frames_per_cp // 2),
            )

        refined_list.append(ref_cp)

    result = CourseDetection(controls=refined_list, start=raw.start)

    # 4) финал
    final = Image.fromarray(base.copy())
    draw = ImageDraw.Draw(final)
    for cp, ref_cp, snapped, dist in snaps:
        if snapped and dist >= 0.5:
            draw.line(
                (S(cp.x), S(cp.y), S(ref_cp.x), S(ref_cp.y)),
                fill=(180, 180, 180),
                width=max(1, line_w - 1),
            )
            _marker(draw, cp.x, cp.y, cp.number, color=raw_color, square=True)
        _marker(
            draw,
            ref_cp.x,
            ref_cp.y,
            ref_cp.number,
            color=ref_color if snapped else raw_color,
            square=not snapped,
        )
    if result.start is not None:
        st = result.start
        _draw_start(
            draw,
            StartFinish(
                x=S(st.x), y=S(st.y), score=st.score, side=S(st.side) if st.side else 0.0
            ),
            font=font,
        )
    n_snap = sum(1 for _, _, ok, _ in snaps if ok)
    push(
        np.asarray(final),
        f"4/4  Итог refine: {n_snap}/{len(snaps)} snap",
        n=max(hold_frames, 20),
    )
    return frames, result


def animate_detect_controls_vlm_refine(
    image: str | Path | np.ndarray,
    raw: CourseDetection,
    *,
    prior_mask: np.ndarray | None = None,
    circle_diameter_px: float | None = 55.0,
    out_path: str | Path | None = None,
    format: str = "gif",
    fps: int = 16,
    n_mask_frames: int = 20,
    frames_per_cp: int = 8,
    hold_frames: int = 28,
    max_side: int | None = 1280,
    display: bool = True,
) -> tuple[CourseDetection, Path]:
    """
    Анимация VLM + refine_ring_center поверх уже найденных сырых точек.

    Не вызывает Vision API — только snap к magenta-кольцам.
    """
    from src.controls.detect_vlm import _load_rgb

    rgb = _load_rgb(image)
    frames, result = render_vlm_refine_frames(
        rgb,
        raw,
        prior_mask=prior_mask,
        circle_diameter_px=circle_diameter_px,
        n_mask_frames=n_mask_frames,
        frames_per_cp=frames_per_cp,
        hold_frames=hold_frames,
        max_side=max_side,
    )

    if out_path is None:
        out_path = Path("runs") / "controls_vis" / f"vlm_refine.{format}"
    else:
        out_path = Path(out_path)

    if format == "mp4":
        saved = save_animation_mp4(frames, out_path, fps=fps)
    else:
        saved = save_animation_gif(frames, out_path.with_suffix(".gif"), fps=fps)

    if display:
        _display_animation(saved)

    return result, saved
