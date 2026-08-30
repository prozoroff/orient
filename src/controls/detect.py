"""Оркестрация: розовая маска + OCR → список ControlPoint.

Режимы:
- ``circle_first`` (по умолчанию): кольца КП → OCR в окрестности → только пары.
- ``digit_first``: старый пайплайн EasyOCR → поиск кружка рядом.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from PIL import Image

from src.controls.geometry import (
    Blob,
    classify_blobs,
    cluster_digit_blobs,
    detect_cp_rings,
    expand_bbox,
    find_circle_near,
    find_start_triangle,
    ring_quality,
)
from src.controls.ocr import OcrHit, get_reader, read_digits_crop, read_digits_full
from src.controls.pink_mask import merge_prior, pink_mask_hsv
from src.controls.types import (
    ControlPoint,
    CourseDetection,
    CvDetectTrace,
    DigitAssistEvent,
    OcrWindow,
    RingOcrEvent,
    StartFinish,
)

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


def _pink_overlap_ratio(hit: OcrHit, pink: np.ndarray) -> float:
    x0, y0 = max(0, hit.x0), max(0, hit.y0)
    x1, y1 = min(pink.shape[1], hit.x1), min(pink.shape[0], hit.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    region = pink[y0:y1, x0:x1] > 0
    if region.size == 0:
        return 0.0
    return float(region.mean())


def _nearest_circle(
    cx: float,
    cy: float,
    circles: Sequence[Blob],
    max_dist: float,
    *,
    digit_bbox: tuple[int, int, int, int] | None = None,
    digit_height: float = 20.0,
    pink: np.ndarray | None = None,
) -> Blob | None:
    """Ближайший полый кружок; центры внутри bbox цифры отбрасываем."""
    best: Blob | None = None
    best_score = -1e9
    pad = max(2, int(0.15 * digit_height))
    for c in circles:
        d = float(np.hypot(c.cx - cx, c.cy - cy))
        if d >= max_dist:
            continue
        if digit_bbox is not None:
            x0, y0, x1, y1 = digit_bbox
            if (x0 - pad) <= c.cx <= (x1 + pad) and (y0 - pad) <= c.cy <= (y1 + pad):
                continue
        elif d < 0.45 * digit_height:
            continue
        hollow = float(c.circularity) if c.circularity <= 1.5 else 0.5
        if pink is not None and c.radius > 0:
            _, _, hollow = ring_quality(pink, c.cx, c.cy, max(c.radius, 1.0))
            if hollow < 0.22:
                continue
            if c.radius < 0.45 * digit_height or c.radius > 0.95 * digit_height:
                continue
        score = hollow * 2.0 - 0.2 * (d / max(max_dist, 1.0))
        if score > best_score:
            best_score = score
            best = c
    return best


def _bbox_iou(a: OcrHit, b: OcrHit) -> float:
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    if inter <= 0:
        return 0.0
    area_a = max(1, (a.x1 - a.x0) * (a.y1 - a.y0))
    area_b = max(1, (b.x1 - b.x0) * (b.y1 - b.y0))
    return inter / float(area_a + area_b - inter)


def _merge_hits(hits: list[OcrHit]) -> list[OcrHit]:
    """Слить дубли; при конфликте — conf и типичный двузначный номер КП."""
    if not hits:
        return []

    def _rank(h: OcrHit) -> tuple:
        two_digit = 1 if 10 <= h.number <= 99 else 0
        return (h.conf + 0.15 * two_digit, two_digit, h.conf)

    hits = sorted(hits, key=_rank, reverse=True)
    kept: list[OcrHit] = []
    for h in hits:
        drop = False
        for i, k in enumerate(kept):
            iou = _bbox_iou(h, k)
            near = abs(h.cx - k.cx) < 28 and abs(h.cy - k.cy) < 28
            if not (iou > 0.15 or near):
                continue
            hs, ks = str(h.number), str(k.number)
            if hs != ks and hs in ks:
                drop = True
                break
            if hs != ks and ks in hs and _rank(h) >= _rank(k):
                kept[i] = h
                drop = True
                break
            if h.number == k.number or _rank(h) <= _rank(k):
                drop = True
                break
            kept[i] = h
            drop = True
            break
        if not drop:
            kept.append(h)
    return kept


def _plausible_digit_size(
    hit: OcrHit,
    h: int,
    w: int,
    *,
    max_height_frac: float = 0.28,
    max_width_frac: float = 0.28,
    max_height_px: int = 140,
    min_height_px: int = 6,
    min_width_px: int = 4,
) -> bool:
    """Отсечь OCR по огромным заголовкам карты и микрошум."""
    bh = hit.y1 - hit.y0
    bw = hit.x1 - hit.x0
    if bh < min_height_px or bw < min_width_px:
        return False
    if bw > max_width_frac * w:
        return False
    if bh > max_height_frac * h or bh > max_height_px:
        return False
    return True


def _ocr_windows_around_ring(
    cx: float, cy: float, r: float
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """Окна OCR: (направленные R/L/T/B, широкая окрестность).

    Широкое окно отдельно — оно иногда галлюцинирует («81»→«31»).
    """
    directional = [
        (int(cx + r * 0.35), int(cy - r * 1.4), int(cx + r * 3.6), int(cy + r * 1.4)),
        (int(cx - r * 3.6), int(cy - r * 1.4), int(cx - r * 0.35), int(cy + r * 1.4)),
        (int(cx - r * 1.7), int(cy - r * 3.3), int(cx + r * 1.7), int(cy - r * 0.35)),
        (int(cx - r * 1.7), int(cy + r * 0.35), int(cx + r * 1.7), int(cy + r * 3.3)),
    ]
    around = [
        (int(cx - 3.2 * r), int(cy - 2.8 * r), int(cx + 3.2 * r), int(cy + 2.8 * r)),
    ]
    return directional, around


def _best_digit_for_ring(
    hits: Sequence[OcrHit],
    cx: float,
    cy: float,
    r: float,
    *,
    min_conf: float,
) -> OcrHit | None:
    """Выбрать номер у кольца: голосование по окнам + отсев склеек вроде «818»."""
    candidates: list[tuple[float, OcrHit]] = []
    for hit in hits:
        conf_floor = max(min_conf, 0.25)
        n_digits = len(str(hit.number))
        if n_digits < 2:
            continue  # однозначные — частый шум дуг/линий
        if n_digits >= 3:
            conf_floor = max(conf_floor, 0.85)
            if hit.number > 199:
                continue  # коды КП обычно ≤199; «818» и т.п. — склейки
        if hit.conf < conf_floor:
            continue
        d = float(np.hypot(hit.cx - cx, hit.cy - cy))
        if d < 0.75 * r or d > 3.5 * r:
            continue
        if (hit.x0 - 3) <= cx <= (hit.x1 + 3) and (hit.y0 - 3) <= cy <= (hit.y1 + 3):
            continue
        bh = max(1, hit.y1 - hit.y0)
        bw = hit.x1 - hit.x0
        if bh < 0.45 * r or bh > 2.2 * r or bw > 4.5 * r:
            continue
        if bw / float(bh) > 2.8:
            continue  # слишком широкая «цифра» — часто склейка двух номеров
        score = float(hit.conf) + 0.15 * (n_digits - 1)
        if hit.cx > cx:
            score += 0.05  # номер чаще справа от кружка
        candidates.append((score, hit))

    if not candidates:
        return None

    # голосование: сумма score по number, затем лучший hit этого number
    votes: dict[int, float] = {}
    best_hit: dict[int, OcrHit] = {}
    best_sc: dict[int, float] = {}
    for score, hit in candidates:
        votes[hit.number] = votes.get(hit.number, 0.0) + score
        if hit.number not in best_sc or score > best_sc[hit.number]:
            best_sc[hit.number] = score
            best_hit[hit.number] = hit
    winner = max(votes.keys(), key=lambda n: (votes[n], best_sc[n]))
    return best_hit[winner]


def _center_taken(
    x: float, y: float, used: Sequence[tuple[float, float]], tol: float = 14.0
) -> bool:
    return any(np.hypot(x - ux, y - uy) < tol for ux, uy in used)


def _detect_circle_first(
    rgb: np.ndarray,
    pink: np.ndarray,
    *,
    reader: Any,
    min_ocr_conf: float,
    find_start: bool,
    circle_diameter_px: float | None = None,
) -> CourseDetection:
    """Кольца → OCR рядом → digit-assist для пропущенных номеров."""
    return _detect_circle_first_traced(
        rgb,
        pink,
        reader=reader,
        min_ocr_conf=min_ocr_conf,
        find_start=find_start,
        circle_diameter_px=circle_diameter_px,
    ).result


def _ring_radius_bounds(
    circle_diameter_px: float | None,
) -> tuple[int, int, float]:
    """(min_radius, max_radius, expected_radius) для Hough колец КП."""
    if circle_diameter_px is not None and circle_diameter_px > 0:
        r0 = float(circle_diameter_px) * 0.5
        min_r = max(6, int(round(0.70 * r0)))
        max_r = max(min_r + 2, int(round(1.35 * r0)))
        return min_r, max_r, r0
    # дефолт под тайлы/скрины ~Ø 22–36 px
    return 11, 18, 14.5


def _detect_circle_first_traced(
    rgb: np.ndarray,
    pink: np.ndarray,
    *,
    reader: Any,
    min_ocr_conf: float,
    find_start: bool,
    circle_diameter_px: float | None = None,
) -> CvDetectTrace:
    """То же, что circle-first, плюс промежуточные шаги для анимации."""
    h, w = rgb.shape[:2]
    min_r, max_r, expected_r = _ring_radius_bounds(circle_diameter_px)
    rings = detect_cp_rings(
        pink,
        min_radius=min_r,
        max_radius=max_r,
        min_dist=max(16.0, 1.1 * expected_r),
    )
    by_number: dict[int, ControlPoint] = {}
    used_centers: list[tuple[float, float]] = []
    crop_conf = max(0.05, min_ocr_conf * 0.5)
    ring_events: list[RingOcrEvent] = []
    digit_assist: list[DigitAssistEvent] = []
    # крупные кольца без сильного hollow — чаще ложные; порог масштабируем от Ø
    large_r_cut = max(17.5, 1.15 * expected_r)

    for ring in rings:
        directional, around = _ocr_windows_around_ring(ring.cx, ring.cy, ring.radius)
        windows: list[OcrWindow] = [
            OcrWindow(x0, y0, x1, y1, kind="directional")
            for x0, y0, x1, y1 in directional
        ]
        dir_hits: list[OcrHit] = []
        for x0, y0, x1, y1 in directional:
            dir_hits.extend(
                read_digits_crop(rgb, x0, y0, x1, y1, min_conf=crop_conf, reader=reader)
            )
        dir_hit = _best_digit_for_ring(
            dir_hits, ring.cx, ring.cy, ring.radius, min_conf=min_ocr_conf
        )

        around_hits: list[OcrHit] = []
        # around нужен, если нет directional или он неуверенный (склейки/обломки цифр)
        if dir_hit is None or dir_hit.conf < 0.90:
            for x0, y0, x1, y1 in around:
                windows.append(OcrWindow(x0, y0, x1, y1, kind="around"))
                around_hits.extend(
                    read_digits_crop(
                        rgb, x0, y0, x1, y1, min_conf=crop_conf, reader=reader
                    )
                )
        around_hit = _best_digit_for_ring(
            around_hits, ring.cx, ring.cy, ring.radius, min_conf=min_ocr_conf
        )

        hit: OcrHit | None
        if dir_hit is None:
            hit = around_hit
        elif around_hit is None:
            hit = dir_hit
        elif dir_hit.number == around_hit.number:
            hit = dir_hit if dir_hit.conf >= around_hit.conf else around_hit
        elif dir_hit.conf >= 0.90:
            # уверенный directional важнее around («81» не перебивать «31»)
            hit = dir_hit
        elif around_hit.conf >= 0.95 and around_hit.conf >= dir_hit.conf + 0.15:
            # around собрал целый номер из обломков («5»+«2»→«52» vs «10»)
            hit = around_hit
        else:
            hit = dir_hit if dir_hit.conf >= around_hit.conf else around_hit

        cp: ControlPoint | None = None
        if hit is not None and not _center_taken(ring.cx, ring.cy, used_centers):
            hollow = float(ring.circularity)
            # слабые / крупные ложные кольца
            reject = False
            if hollow < 0.55 and hit.conf < 0.9:
                reject = True
            if ring.radius > large_r_cut and hollow < 0.7:
                reject = True
            geo = min(1.0, hollow)
            score = float(hit.conf) * (0.55 + 0.45 * geo)
            if score < 0.45:
                reject = True
            if not reject:
                cand = ControlPoint(
                    number=hit.number,
                    x=float(ring.cx),
                    y=float(ring.cy),
                    score=float(score),
                    source="circle",
                )
                prev = by_number.get(hit.number)
                if prev is None or cand.score > prev.score:
                    if prev is not None:
                        used_centers = [
                            (ux, uy)
                            for ux, uy in used_centers
                            if np.hypot(ux - prev.x, uy - prev.y) >= 1.0
                        ]
                    by_number[hit.number] = cand
                    used_centers.append((ring.cx, ring.cy))
                    cp = cand

        ring_events.append(
            RingOcrEvent(
                cx=float(ring.cx),
                cy=float(ring.cy),
                radius=float(ring.radius),
                hollow=float(ring.circularity),
                windows=tuple(windows),
                hit_number=None if hit is None else int(hit.number),
                hit_conf=0.0 if hit is None else float(hit.conf),
                hit_bbox=None if hit is None else (hit.x0, hit.y0, hit.x1, hit.y1),
                control=cp,
            )
        )

    # Digit-assist: сильные номера без пары (напр. «100» слева от кольца)
    full_hits = [
        hit
        for hit in read_digits_full(rgb, min_conf=max(0.35, min_ocr_conf), reader=reader)
        if _plausible_digit_size(hit, h, w, max_height_frac=0.35, max_width_frac=0.35)
        and 10 <= hit.number <= 199
        and hit.number not in by_number
    ]
    for hit in sorted(full_hits, key=lambda hh: hh.conf, reverse=True):
        if hit.number in by_number:
            continue
        pink_ov = _pink_overlap_ratio(hit, pink)
        if pink_ov < 0.02 and hit.conf < 0.7:
            continue
        digit_h = float(max(1, hit.y1 - hit.y0))
        local = find_circle_near(
            pink,
            hit.cx,
            hit.cy,
            search_radius=max(3.2 * digit_h, 36.0, 2.2 * expected_r),
            digit_height=digit_h,
            digit_bbox=(hit.x0, hit.y0, hit.x1, hit.y1),
        )
        if local is None or _center_taken(local.cx, local.cy, used_centers):
            continue
        if float(local.circularity) < 0.35:
            continue
        geo = float(local.circularity)
        score = float(hit.conf) * (0.55 + 0.45 * geo) * 0.95
        cp = ControlPoint(
            number=hit.number,
            x=float(local.cx),
            y=float(local.cy),
            score=float(score),
            source="circle",
        )
        by_number[hit.number] = cp
        used_centers.append((local.cx, local.cy))
        digit_assist.append(
            DigitAssistEvent(
                hit_number=int(hit.number),
                hit_conf=float(hit.conf),
                hit_bbox=(hit.x0, hit.y0, hit.x1, hit.y1),
                control=cp,
            )
        )

    controls = sorted(by_number.values(), key=lambda c: (c.number, -c.score))

    start: StartFinish | None = None
    if find_start:
        radii = [r.radius for r in rings] or [expected_r]
        ref_size = float(np.median(radii)) * 2.0
        tri = find_start_triangle(
            pink,
            ref_size=ref_size,
            exclude_xy=[(c.x, c.y) for c in controls],
            exclude_radius=max(20.0, 0.8 * ref_size),
        )
        if tri is not None:
            sx, sy, sscore, side = tri
            start = StartFinish(x=sx, y=sy, score=sscore, side=side)

    result = CourseDetection(controls=controls, start=start)
    return CvDetectTrace(
        pink=pink,
        rings=[
            (float(r.cx), float(r.cy), float(r.radius), float(r.circularity))
            for r in rings
        ],
        ring_events=ring_events,
        digit_assist=digit_assist,
        result=result,
    )


def _detect_digit_first(
    rgb: np.ndarray,
    pink: np.ndarray,
    *,
    reader: Any,
    min_ocr_conf: float,
    min_pink_overlap: float,
    circle_search_scale: float,
    sensitivity: str,
    high_conf_bypass: float,
    size_kw: dict,
    find_start: bool,
) -> CourseDetection:
    """Старый пайплайн: OCR → кружок рядом."""
    h, w = rgb.shape[:2]
    circles, digit_blobs = classify_blobs(pink)
    hits: list[OcrHit] = []

    for hit in read_digits_full(rgb, min_conf=min_ocr_conf, reader=reader):
        if not _plausible_digit_size(hit, h, w, **size_kw):
            continue
        ov = _pink_overlap_ratio(hit, pink)
        if ov >= min_pink_overlap or (ov >= 0.005 and hit.conf >= high_conf_bypass):
            hits.append(hit)

    crop_conf = min_ocr_conf * (0.5 if sensitivity == "recall" else 0.75)
    pad = 0.0
    for x0, y0, x1, y1 in cluster_digit_blobs(digit_blobs):
        x0, y0, x1, y1 = expand_bbox(x0, y0, x1, y1, h, w, pad=pad)
        for hit in read_digits_crop(
            rgb, x0, y0, x1, y1, min_conf=crop_conf, reader=reader
        ):
            if _plausible_digit_size(hit, h, w, **size_kw):
                hits.append(hit)

    hits = _merge_hits(hits)

    heights = [max(1, hh.y1 - hh.y0) for hh in hits]
    med_h = float(np.median(heights)) if heights else 20.0
    max_dist = min(
        max(36.0, 2.8 * med_h),
        max(36.0, circle_search_scale * med_h * 0.75),
    )

    by_number: dict[int, ControlPoint] = {}
    used_circle_labels: set[int] = set()
    used_centers: list[tuple[float, float]] = []

    for hit in sorted(hits, key=lambda hh: hh.conf, reverse=True):
        digit_h = float(max(1, hit.y1 - hit.y0))
        digit_bbox = (hit.x0, hit.y0, hit.x1, hit.y1)
        free_circles = [c for c in circles if c.label not in used_circle_labels]
        circle = _nearest_circle(
            hit.cx,
            hit.cy,
            free_circles,
            max_dist=max_dist,
            digit_bbox=digit_bbox,
            digit_height=digit_h,
            pink=pink,
        )
        local = find_circle_near(
            pink,
            hit.cx,
            hit.cy,
            search_radius=max_dist,
            digit_height=digit_h,
            digit_bbox=digit_bbox,
        )
        if local is not None and not _center_taken(local.cx, local.cy, used_centers):
            if circle is None or float(local.circularity) >= float(circle.circularity) - 0.02:
                circle = local
        if circle is not None and _center_taken(circle.cx, circle.cy, used_centers):
            circle = None

        if circle is not None:
            x, y = circle.cx, circle.cy
            source = "circle"
            geo = 1.0 - min(
                1.0,
                float(np.hypot(circle.cx - hit.cx, circle.cy - hit.cy)) / max_dist,
            )
            used_circle_labels.add(circle.label)
            used_centers.append((circle.cx, circle.cy))
        else:
            x, y = hit.cx, hit.cy
            source = "digit"
            geo = 0.7
        pink_ov = _pink_overlap_ratio(hit, pink)
        score = float(hit.conf) * (0.55 + 0.45 * geo) * (0.7 + 0.3 * min(1.0, pink_ov * 3))
        cp = ControlPoint(
            number=hit.number,
            x=float(x),
            y=float(y),
            score=float(score),
            source=source,  # type: ignore[arg-type]
        )
        prev = by_number.get(hit.number)
        if prev is None or cp.score > prev.score:
            by_number[hit.number] = cp

    controls = sorted(by_number.values(), key=lambda c: (c.number, -c.score))

    start: StartFinish | None = None
    if find_start:
        ref_size = med_h
        tri = find_start_triangle(
            pink,
            ref_size=ref_size,
            exclude_xy=[(c.x, c.y) for c in controls],
            exclude_radius=max(20.0, 0.8 * ref_size),
        )
        if tri is not None:
            sx, sy, sscore, side = tri
            start = StartFinish(x=sx, y=sy, score=sscore, side=side)

    return CourseDetection(controls=controls, start=start)


def detect_controls(
    image: str | Path | np.ndarray,
    *,
    prior_mask: np.ndarray | None = None,
    min_ocr_conf: float | None = None,
    min_pink_overlap: float | None = None,
    circle_search_scale: float = 4.0,
    use_clahe: bool = True,
    sensitivity: str = "balanced",
    min_saturation: int | None = None,
    min_value: int | None = None,
    gpu: bool | None = None,
    reader: Any | None = None,
    find_start: bool = True,
    mode: Literal["circle_first", "digit_first"] = "circle_first",
    circle_diameter_px: float | None = None,
) -> CourseDetection:
    """
    Найти контрольные пункты (и треугольник старта/финиша) на карте / фото.

    Parameters
    ----------
    mode
        ``circle_first`` — кольца пурпурной краски, затем OCR рядом (рекомендуется).
        ``digit_first`` — старый OCR-first пайплайн.
    circle_diameter_px
        Ожидаемый диаметр кружка КП в пикселях (для Hough в ``circle_first``).
        ``None`` — диапазон по умолчанию (~Ø 22–36 px, как на тайлах датасета).
        Для скринов/карт с крупными кружками задайте явно (напр. 60).

    Returns
    -------
    CourseDetection
        ``.controls`` — список КП; ``.start`` — старт/финиш или None.
    """
    if sensitivity not in {"balanced", "recall"}:
        raise ValueError(f"Unknown sensitivity={sensitivity!r}, use balanced|recall")
    if mode not in {"circle_first", "digit_first"}:
        raise ValueError(f"Unknown mode={mode!r}, use circle_first|digit_first")

    if sensitivity == "recall":
        conf_default = 0.05
        pink_default = 0.0
        sat_default = 25
        val_default = 30
        max_h_frac, max_w_frac, max_h_px = 0.40, 0.40, 200
        high_conf_bypass = 0.35
    else:
        conf_default = 0.15
        pink_default = 0.01
        sat_default = 35
        val_default = 35
        max_h_frac, max_w_frac, max_h_px = 0.32, 0.32, 160
        high_conf_bypass = 0.45

    min_ocr_conf = conf_default if min_ocr_conf is None else float(min_ocr_conf)
    min_pink_overlap = pink_default if min_pink_overlap is None else float(min_pink_overlap)
    min_saturation = sat_default if min_saturation is None else int(min_saturation)
    min_value = val_default if min_value is None else int(min_value)

    rgb = _load_rgb(image)
    reader = reader or get_reader(gpu=gpu)

    preserve_thin = mode == "circle_first"
    pink = pink_mask_hsv(
        rgb,
        use_clahe=use_clahe,
        min_saturation=min_saturation,
        min_value=min_value,
        preserve_thin=preserve_thin,
    )
    pink = merge_prior(pink, prior_mask)

    if mode == "circle_first":
        return _detect_circle_first(
            rgb,
            pink,
            reader=reader,
            min_ocr_conf=min_ocr_conf,
            find_start=find_start,
            circle_diameter_px=circle_diameter_px,
        )

    size_kw = dict(
        max_height_frac=max_h_frac,
        max_width_frac=max_w_frac,
        max_height_px=max_h_px,
        min_height_px=6 if sensitivity == "recall" else 8,
        min_width_px=4 if sensitivity == "recall" else 6,
    )
    return _detect_digit_first(
        rgb,
        pink,
        reader=reader,
        min_ocr_conf=min_ocr_conf,
        min_pink_overlap=min_pink_overlap,
        circle_search_scale=circle_search_scale,
        sensitivity=sensitivity,
        high_conf_bypass=high_conf_bypass,
        size_kw=size_kw,
        find_start=find_start,
    )


def detect_controls_cv_trace(
    image: str | Path | np.ndarray,
    *,
    prior_mask: np.ndarray | None = None,
    min_ocr_conf: float | None = None,
    use_clahe: bool = True,
    sensitivity: str = "balanced",
    min_saturation: int | None = None,
    min_value: int | None = None,
    gpu: bool | None = None,
    reader: Any | None = None,
    find_start: bool = True,
    circle_diameter_px: float | None = None,
) -> CvDetectTrace:
    """Circle-first детекция с промежуточными шагами (для анимации)."""
    if sensitivity not in {"balanced", "recall"}:
        raise ValueError(f"Unknown sensitivity={sensitivity!r}, use balanced|recall")

    if sensitivity == "recall":
        conf_default, sat_default, val_default = 0.05, 25, 30
    else:
        conf_default, sat_default, val_default = 0.15, 35, 35

    min_ocr_conf = conf_default if min_ocr_conf is None else float(min_ocr_conf)
    min_saturation = sat_default if min_saturation is None else int(min_saturation)
    min_value = val_default if min_value is None else int(min_value)

    rgb = _load_rgb(image)
    reader = reader or get_reader(gpu=gpu)
    pink = pink_mask_hsv(
        rgb,
        use_clahe=use_clahe,
        min_saturation=min_saturation,
        min_value=min_value,
        preserve_thin=True,
    )
    pink = merge_prior(pink, prior_mask)
    return _detect_circle_first_traced(
        rgb,
        pink,
        reader=reader,
        min_ocr_conf=min_ocr_conf,
        find_start=find_start,
        circle_diameter_px=circle_diameter_px,
    )
