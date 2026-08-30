"""Выделение розовой/пурпурной краски дистанции (КП) на карте или фото."""

from __future__ import annotations

import cv2
import numpy as np


def _clahe_v(bgr: np.ndarray) -> np.ndarray:
    """CLAHE по каналу V — стабилизирует порог на фото с бликами."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v2 = clahe.apply(v)
    return cv2.cvtColor(cv2.merge([h, s, v2]), cv2.COLOR_HSV2BGR)


def pink_mask_hsv(
    image_rgb: np.ndarray,
    *,
    use_clahe: bool = True,
    min_saturation: int = 40,
    min_value: int = 40,
    preserve_thin: bool = False,
) -> np.ndarray:
    """
    Бинарная маска розовой/красной краски курса (uint8 0/255).

    Покрывает magenta (типичный ISOM purple) и красноватый pink на старых картах.

    ``preserve_thin=True`` — без MORPH_OPEN (не стирает тонкие кольца КП),
    плюс RGB-эвристика magenta; для circle-first.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB, got {image_rgb.shape}")

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if use_clahe:
        bgr = _clahe_v(bgr)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # OpenCV H: 0..179. Magenta ~140-175, red wraps 0-15.
    h_lo = 130 if preserve_thin else 135
    s_lo = max(15, min_saturation - (10 if preserve_thin else 0))
    lower_mag = np.array([h_lo, s_lo, min_value], dtype=np.uint8)
    upper_mag = np.array([179, 255, 255], dtype=np.uint8)
    lower_red = np.array([0, min_saturation, min_value], dtype=np.uint8)
    upper_red = np.array([15, 255, 255], dtype=np.uint8)

    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_mag, upper_mag),
        cv2.inRange(hsv, lower_red, upper_red),
    )

    # Чуть мягче порог S для выцветших фото (но не захватываем серое).
    soft = cv2.inRange(
        hsv,
        np.array([140, max(20, min_saturation // 2), min_value], dtype=np.uint8),
        np.array([175, 255, 255], dtype=np.uint8),
    )
    mask = cv2.bitwise_or(mask, soft)

    if preserve_thin:
        # R+B высоки относительно G — типичный ISOM purple на печати/фото
        rgb = image_rgb.astype(np.float32)
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        mag = (
            (r > 90)
            & (b > 70)
            & (g < 0.85 * np.minimum(r, b))
            & (((r + b) * 0.5 - g) > 25)
        )
        mask = cv2.bitwise_or(mask, mag.astype(np.uint8) * 255)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if not preserve_thin:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    else:
        # только CLOSE — склеить разрывы тонкого кольца, не стирая штрих
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def merge_prior(pink: np.ndarray, prior_mask: np.ndarray | None) -> np.ndarray:
    """OR с опциональной маской класса control_point (сегментация / GT)."""
    if prior_mask is None:
        return pink
    if prior_mask.shape[:2] != pink.shape[:2]:
        raise ValueError(
            f"prior_mask shape {prior_mask.shape[:2]} != pink {pink.shape[:2]}"
        )
    prior_u8 = (prior_mask > 0).astype(np.uint8) * 255
    return cv2.bitwise_or(pink, prior_u8)
