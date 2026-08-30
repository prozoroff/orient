"""Геометрия розовой маски: отфильтровать линии, найти кружки и digit-blobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass
class Blob:
    label: int
    area: float
    cx: float
    cy: float
    x0: int
    y0: int
    x1: int
    y1: int
    circularity: float
    fill_ratio: float  # area / bbox_area
    has_hole: bool
    kind: str  # "circle" | "digit" | "line" | "noise"
    radius: float = 0.0  # для Hough-кружков


def _component_stats(mask: np.ndarray) -> list[Blob]:
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    h, w = mask.shape
    blobs: list[Blob] = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < 8:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = float(centroids[i, 0]), float(centroids[i, 1])
        bbox_area = max(bw * bh, 1)
        fill = area / bbox_area

        comp = (labels == i).astype(np.uint8) * 255
        # закрыть мелкие разрывы кольца перед поиском дырки
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        comp_closed = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, k_close, iterations=2)
        cnts, hier = cv2.findContours(comp_closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        peri = 0.0
        has_hole = False
        if cnts:
            peri = float(cv2.arcLength(cnts[0], True))
            if hier is not None and len(hier) > 0:
                for j in range(len(cnts)):
                    if hier[0][j][3] >= 0:
                        has_hole = True
                        break
        circularity = (4.0 * np.pi * area / (peri * peri)) if peri > 1e-3 else 0.0

        aspect = max(bw, bh) / max(min(bw, bh), 1)
        is_line = aspect >= 4.0 and fill < 0.35 and area < 0.02 * h * w
        if not is_line and aspect >= 3.0 and bh <= 6 and bw >= 40:
            is_line = True
        if not is_line and aspect >= 3.0 and bw <= 6 and bh >= 40:
            is_line = True

        # почти квадратный bbox + умеренная заполненность ≈ кольцо/кружок
        # важно: одиночные цифры («7») тоже почти квадратные — требуем размер как у КП
        square = abs(bw - bh) <= max(bw, bh) * 0.40
        big_enough_for_cp = min(bw, bh) >= 26 and area >= 220
        ring_like = (
            square
            and big_enough_for_cp
            and (
                has_hole
                or (0.15 <= fill <= 0.65 and circularity > 0.35)
                or (0.20 <= fill <= 0.55 and circularity > 0.55)
            )
        )

        if is_line:
            kind = "line"
        elif ring_like:
            kind = "circle"
        elif 12 <= area <= 8000 and aspect <= 3.5 and fill >= 0.15 and max(bw, bh) <= 120:
            kind = "digit"
        else:
            kind = "noise"

        radius = 0.5 * float(min(bw, bh))
        blobs.append(
            Blob(
                label=i,
                area=area,
                cx=cx,
                cy=cy,
                x0=x,
                y0=y,
                x1=x + bw,
                y1=y + bh,
                circularity=float(circularity),
                fill_ratio=float(fill),
                has_hole=has_hole,
                kind=kind,
                radius=radius,
            )
        )
    return blobs


def remove_line_components(mask: np.ndarray) -> np.ndarray:
    """Убрать длинные тонкие компоненты (линии дистанции) из маски."""
    blobs = _component_stats(mask)
    n, labels, _, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    keep = np.zeros_like(mask)
    line_ids = {b.label for b in blobs if b.kind == "line"}
    for i in range(1, n):
        if i not in line_ids:
            keep[labels == i] = 255
    return keep


def _circle_pink_support(pink: np.ndarray, cx: float, cy: float, r: float, n: int = 32) -> float:
    """Доля точек окружности, попадающих в розовую маску (с небольшим допуском)."""
    h, w = pink.shape[:2]
    hits = 0
    rr = max(r, 1.0)
    for i in range(n):
        ang = 2.0 * np.pi * i / n
        for scale in (0.85, 1.0, 1.15):
            x = int(round(cx + scale * rr * np.cos(ang)))
            y = int(round(cy + scale * rr * np.sin(ang)))
            if 0 <= x < w and 0 <= y < h and pink[y, x] > 0:
                hits += 1
                break
    return hits / float(n)


def _circle_interior_fill(pink: np.ndarray, cx: float, cy: float, r: float, frac: float = 0.5) -> float:
    """Доля розовых пикселей внутри диска радиуса frac*r (у цифры высокая, у кольца КП низкая)."""
    h, w = pink.shape[:2]
    rr = max(1, int(round(r * frac)))
    y0, y1 = max(0, int(cy) - rr), min(h, int(cy) + rr + 1)
    x0, x1 = max(0, int(cx) - rr), min(w, int(cx) + rr + 1)
    if y1 <= y0 or x1 <= x0:
        return 1.0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= float(rr * rr)
    region = pink[y0:y1, x0:x1][disk]
    if region.size == 0:
        return 1.0
    return float((region > 0).mean())


def ring_quality(pink: np.ndarray, cx: float, cy: float, r: float) -> tuple[float, float, float]:
    """
    Метрики кольца КП: (perimeter_support, interior_fill, hollow_score).
    Настоящий кружок: peri высокий, interior низкий → hollow большой.
    Ложный «кружок» на цифре: оба высокие → hollow маленький.
    """
    peri = _circle_pink_support(pink, cx, cy, r, n=48)
    interior = _circle_interior_fill(pink, cx, cy, r, frac=0.5)
    hollow = peri - interior
    return peri, interior, hollow


def detect_circles_hough(
    pink: np.ndarray,
    *,
    min_radius: int = 8,
    max_radius: int | None = None,
    min_dist: int = 22,
    param2: int = 14,
    min_hollow: float = 0.28,
    max_interior: float = 0.35,
    min_peri: float = 0.35,
) -> list[Blob]:
    """
    Кружки КП через Hough — работает даже если кольцо слито с линией дистанции.
    Фильтр hollow отсекает ложные кружки на сплошных цифрах.
    """
    h, w = pink.shape[:2]
    if max_radius is None:
        max_radius = max(40, min(h, w) // 6)

    blur = cv2.GaussianBlur(pink, (5, 5), 1.2)
    found = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=float(min_dist),
        param1=50,
        param2=float(param2),
        minRadius=int(min_radius),
        maxRadius=int(max_radius),
    )
    if found is None:
        return []

    out: list[Blob] = []
    for idx, (cx, cy, r) in enumerate(found[0]):
        cx_f, cy_f, r_f = float(cx), float(cy), float(r)
        peri, interior, hollow = ring_quality(pink, cx_f, cy_f, r_f)
        if peri < min_peri or interior > max_interior or hollow < min_hollow:
            continue
        x0 = max(0, int(cx_f - r_f))
        y0 = max(0, int(cy_f - r_f))
        x1 = min(w, int(cx_f + r_f) + 1)
        y1 = min(h, int(cy_f + r_f) + 1)
        out.append(
            Blob(
                label=10_000 + idx,
                area=float(np.pi * r_f * r_f),
                cx=cx_f,
                cy=cy_f,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                circularity=float(hollow),
                fill_ratio=float(interior),
                has_hole=True,
                kind="circle",
                radius=r_f,
            )
        )
    return out


def detect_cp_rings(
    pink: np.ndarray,
    *,
    min_radius: int = 11,
    max_radius: int = 18,
    min_dist: float = 16.0,
    min_hollow: float = 0.40,
    max_interior: float = 0.35,
    min_peri: float = 0.45,
    param2_values: Sequence[int] = (8, 10, 12, 14),
) -> list[Blob]:
    """
    Кандидаты колец КП (circle-first): несколько проходов Hough + NMS.

    ``circularity`` хранит hollow_score (peri − interior).
    Типичный радиус кружка на скрине/тайле ≈ 12–16 px.
    """
    h, w = pink.shape[:2]
    blur = cv2.GaussianBlur(pink, (5, 5), 1.0)
    raw: list[tuple[float, Blob]] = []
    idx = 0
    for param2 in param2_values:
        found = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.1,
            minDist=float(max(14, min_radius)),
            param1=40,
            param2=float(param2),
            minRadius=int(min_radius),
            maxRadius=int(max_radius),
        )
        if found is None:
            continue
        for cx, cy, r in found[0]:
            cx_f, cy_f, r_f = float(cx), float(cy), float(r)
            peri, interior, hollow = ring_quality(pink, cx_f, cy_f, r_f)
            if peri < min_peri or interior > max_interior or hollow < min_hollow:
                continue
            if not (min_radius <= r_f <= max_radius):
                continue
            x0 = max(0, int(cx_f - r_f))
            y0 = max(0, int(cy_f - r_f))
            x1 = min(w, int(cx_f + r_f) + 1)
            y1 = min(h, int(cy_f + r_f) + 1)
            blob = Blob(
                label=30_000 + idx,
                area=float(np.pi * r_f * r_f),
                cx=cx_f,
                cy=cy_f,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                circularity=float(hollow),
                fill_ratio=float(interior),
                has_hole=True,
                kind="circle",
                radius=r_f,
            )
            raw.append((hollow * peri, blob))
            idx += 1

    raw.sort(key=lambda t: t[0], reverse=True)
    kept: list[Blob] = []
    for _, blob in raw:
        if any(np.hypot(blob.cx - k.cx, blob.cy - k.cy) < min_dist for k in kept):
            continue
        kept.append(blob)
    return kept


def find_circle_near(
    pink: np.ndarray,
    cx: float,
    cy: float,
    *,
    search_radius: float,
    digit_height: float,
    digit_bbox: tuple[int, int, int, int] | None = None,
) -> Blob | None:
    """
    Лучшее полое кольцо рядом с цифрой (не на самой цифре).
    """
    h, w = pink.shape[:2]
    # не раздувать поиск: иначе цепляем соседние КП
    rad = int(min(max(float(search_radius), 2.2 * digit_height), 3.5 * digit_height, 90.0))
    x0 = max(0, int(cx - rad))
    y0 = max(0, int(cy - rad))
    x1 = min(w, int(cx + rad) + 1)
    y1 = min(h, int(cy + rad) + 1)
    crop = pink[y0:y1, x0:x1]
    if crop.size < 25:
        return None

    min_r = max(10, int(round(0.50 * digit_height)))
    # кружок КП ≈ размер цифры, не больше; дуги внутри глифов меньше
    max_r = max(min_r + 3, int(round(0.85 * digit_height)))
    local = detect_circles_hough(
        crop,
        min_radius=min_r,
        max_radius=max_r,
        min_dist=max(12, min_r),
        param2=11,
        min_hollow=0.25,
        max_interior=0.40,
        min_peri=0.35,
    )

    def _hough_ok() -> bool:
        for c in local:
            gx, gy = c.cx + x0, c.cy + y0
            d = float(np.hypot(gx - cx, gy - cy))
            if (
                0.7 * digit_height <= d <= rad
                and c.circularity >= 0.25
                and c.radius >= 0.45 * digit_height
            ):
                return True
        return False

    # сетка, если Hough не нашёл кольцо рядом с цифрой
    if not _hough_ok():
        step = max(3, int(digit_height * 0.22))
        radii = sorted(
            {
                min_r,
                int(round(0.55 * digit_height)),
                int(round(0.65 * digit_height)),
                max_r,
            }
        )
        radii = [r for r in radii if min_r <= r <= max_r]
        for dy in range(-rad, rad + 1, step):
            for dx in range(-rad, rad + 1, step):
                if abs(dx) + abs(dy) < max(10, int(0.7 * digit_height)):
                    continue
                gx, gy = cx + dx, cy + dy
                if not (x0 <= gx < x1 and y0 <= gy < y1):
                    continue
                for r in radii:
                    peri, interior, hollow = ring_quality(pink, gx, gy, float(r))
                    if peri >= 0.45 and interior <= 0.35 and hollow >= 0.30:
                        local.append(
                            Blob(
                                label=20_000 + len(local),
                                area=float(np.pi * r * r),
                                cx=gx - x0,
                                cy=gy - y0,
                                x0=int(gx - r) - x0,
                                y0=int(gy - r) - y0,
                                x1=int(gx + r) - x0,
                                y1=int(gy + r) - y0,
                                circularity=float(hollow),
                                fill_ratio=float(interior),
                                has_hole=True,
                                kind="circle",
                                radius=float(r),
                            )
                        )

    if not local:
        return None

    def _inside_digit(gx: float, gy: float) -> bool:
        if digit_bbox is None:
            return float(np.hypot(gx - cx, gy - cy)) < 0.45 * digit_height
        dx0, dy0, dx1, dy1 = digit_bbox
        pad = max(2, int(0.15 * digit_height))
        return (dx0 - pad) <= gx <= (dx1 + pad) and (dy0 - pad) <= gy <= (dy1 + pad)

    best: Blob | None = None
    best_score = -1e9
    max_pair = float(rad)
    min_ok_r = 0.45 * digit_height
    max_ok_r = 0.95 * digit_height
    for c in local:
        if c.radius < min_ok_r or c.radius > max_ok_r:
            continue
        gx, gy = c.cx + x0, c.cy + y0
        if _inside_digit(gx, gy):
            continue
        dist = float(np.hypot(gx - cx, gy - cy))
        if dist > max_pair:
            continue
        # близость важнее «идеальности» — иначе чужой КП перетягивает
        score = float(c.circularity) * 1.0 - 1.25 * (dist / max(max_pair, 1.0))
        if score > best_score:
            best_score = score
            best = Blob(
                label=c.label,
                area=c.area,
                cx=gx,
                cy=gy,
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
    return best


def find_start_triangle(
    pink: np.ndarray,
    *,
    ref_size: float | None = None,
    exclude_xy: Sequence[tuple[float, float]] | None = None,
    exclude_radius: float = 25.0,
) -> tuple[float, float, float, float] | None:
    """
    Розовый треугольник старта/финиша (обычно один, ≈ размер кружка КП).

    Returns
    -------
    (cx, cy, score, mean_side) или None
    """
    h, w = pink.shape[:2]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(pink, cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # типичный размер кружка КП → окно площади треугольника
    if ref_size is not None and ref_size > 0:
        side_lo = 0.55 * ref_size
        side_hi = 2.2 * ref_size
    else:
        side_lo = 18.0
        side_hi = min(h, w) * 0.12

    area_lo = 0.3 * (side_lo**2) * np.sqrt(3) / 4
    area_hi = 1.4 * (side_hi**2) * np.sqrt(3) / 4

    best: tuple[float, float, float, float] | None = None
    best_score = -1e9

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < area_lo or area > area_hi:
            continue
        peri = float(cv2.arcLength(cnt, True))
        if peri < 1e-3:
            continue
        # несколько eps — тонкий контур иногда даёт 4 точки
        approx = None
        for eps in (0.03, 0.04, 0.05, 0.06):
            ap = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(ap) == 3:
                approx = ap
                break
        if approx is None:
            continue

        pts = approx.reshape(3, 2).astype(np.float64)
        sides = [
            float(np.linalg.norm(pts[i] - pts[(i + 1) % 3])) for i in range(3)
        ]
        mean_side = float(np.mean(sides))
        if mean_side < side_lo or mean_side > side_hi:
            continue
        if max(sides) / max(min(sides), 1e-6) > 1.55:
            continue  # не почти равносторонний

        # центр масс контура
        m = cv2.moments(cnt)
        if m["m00"] <= 1e-6:
            continue
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])

        if exclude_xy:
            if any(np.hypot(cx - ex, cy - ey) < exclude_radius for ex, ey in exclude_xy):
                continue

        # внутри треугольника мало розового (контур), снаружи по сторонам — много
        # упрощённо: fill_ratio контура умеренный
        x, y, bw, bh = cv2.boundingRect(cnt)
        bbox_area = max(bw * bh, 1)
        fill = area / bbox_area
        if fill < 0.15 or fill > 0.75:
            continue

        # предпочитаем остриём вверх (старт ISOM): вершина с min y
        tip = pts[int(np.argmin(pts[:, 1]))]
        tip_up = tip[1] <= min(pts[:, 1]) + 1e-6
        equil = 1.0 - (max(sides) - min(sides)) / max(mean_side, 1e-6)
        score = equil * 2.0 + (0.25 if tip_up else 0.0) + min(1.0, area / max(area_hi, 1.0))

        if score > best_score:
            best_score = score
            best = (cx, cy, float(score), mean_side)

    return best


def classify_blobs(mask: np.ndarray) -> tuple[list[Blob], list[Blob]]:
    """Вернуть (circles, digit_candidates) после фильтрации линий."""
    cleaned = remove_line_components(mask)
    blobs = _component_stats(cleaned)
    circles = [b for b in blobs if b.kind == "circle"]
    # Hough ловит кружки, слитые с линиями (CC их часто теряет)
    hough = detect_circles_hough(mask)
    circles = _merge_circle_lists(circles, hough)

    digits = [b for b in blobs if b.kind == "digit"]
    for b in blobs:
        if b.kind != "noise":
            continue
        bw, bh = b.x1 - b.x0, b.y1 - b.y0
        if 15 <= b.area <= 6000 and max(bw, bh) <= 100 and min(bw, bh) >= 8:
            digits.append(b)
    return circles, digits


def _merge_circle_lists(a: list[Blob], b: list[Blob], min_dist: float = 12.0) -> list[Blob]:
    out = list(a)
    for c in b:
        if any(np.hypot(c.cx - o.cx, c.cy - o.cy) < min_dist for o in out):
            continue
        out.append(c)
    return out


def expand_bbox(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    h: int,
    w: int,
    pad: float = 0.35,
) -> tuple[int, int, int, int]:
    bw, bh = x1 - x0, y1 - y0
    px = int(round(bw * pad)) + 2
    py = int(round(bh * pad)) + 2
    return (
        max(0, x0 - px),
        max(0, y0 - py),
        min(w, x1 + px),
        min(h, y1 + py),
    )


def cluster_digit_blobs(
    blobs: Sequence[Blob],
    *,
    max_gap_frac: float = 0.55,
    max_dy_frac: float = 0.45,
) -> list[tuple[int, int, int, int]]:
    """Склеить соседние глифы номера («7»+«7»→один bbox) для OCR."""
    if not blobs:
        return []
    items = sorted(blobs, key=lambda b: (b.cy, b.cx))
    used = [False] * len(items)
    clusters: list[tuple[int, int, int, int]] = []
    for i, a in enumerate(items):
        if used[i]:
            continue
        used[i] = True
        group = [a]
        changed = True
        while changed:
            changed = False
            for j, b in enumerate(items):
                if used[j]:
                    continue
                for g in group:
                    gh = max(1.0, float(g.y1 - g.y0))
                    avg_h = 0.5 * (gh + max(1.0, float(b.y1 - b.y0)))
                    dy = abs(g.cy - b.cy)
                    if dy > max_dy_frac * avg_h:
                        continue
                    gap = max(0, b.x0 - g.x1, g.x0 - b.x1)
                    if gap <= max_gap_frac * avg_h:
                        used[j] = True
                        group.append(b)
                        changed = True
                        break
        x0 = min(g.x0 for g in group)
        y0 = min(g.y0 for g in group)
        x1 = max(g.x1 for g in group)
        y1 = max(g.y1 for g in group)
        clusters.append((x0, y0, x1, y1))
    return clusters
