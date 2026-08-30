"""EasyOCR-обёртка для цифр номеров КП."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

_READER: Any | None = None


def get_reader(gpu: bool | None = None) -> Any:
    """Lazy singleton EasyOCR Reader (digits)."""
    global _READER
    if _READER is not None:
        return _READER
    try:
        import easyocr
    except ImportError as e:
        raise ImportError(
            "Для детекции КП нужен easyocr. Установите: pip install easyocr"
        ) from e

    _ensure_easyocr_models()

    use_gpu = gpu
    if use_gpu is None:
        try:
            import torch

            use_gpu = bool(torch.cuda.is_available())
        except Exception:
            use_gpu = False

    _READER = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
    return _READER


def _ensure_easyocr_models() -> None:
    """Скачать CRAFT + english recognizer, если их ещё нет (с обходом SSL на macOS)."""
    import ssl
    import urllib.request
    import zipfile
    from pathlib import Path

    home = Path.home() / ".EasyOCR" / "model"
    home.mkdir(parents=True, exist_ok=True)
    needed = {
        "craft_mlt_25k.pth": (
            "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip"
        ),
        "english_g2.pth": (
            "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip"
        ),
    }
    missing = [n for n, _ in needed.items() if not (home / n).exists()]
    if not missing:
        return

    ctx = ssl.create_default_context()
    try:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    for name in missing:
        url = needed[name]
        zip_path = home / f"{name}.download.zip"
        with urllib.request.urlopen(url, context=ctx) as resp:
            zip_path.write_bytes(resp.read())
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(home)
        zip_path.unlink(missing_ok=True)


@dataclass
class OcrHit:
    text: str
    number: int
    conf: float
    x0: int
    y0: int
    x1: int
    y1: int
    cx: float
    cy: float


_NUM_RE = re.compile(r"^\d{1,3}$")


def _parse_number(text: str) -> int | None:
    t = re.sub(r"[^\d]", "", text.strip())
    if not _NUM_RE.match(t):
        return None
    # отбрасываем ведущие нули как «000», но «01»→1 ок для КП редко; оставляем int
    n = int(t)
    if n < 0 or n > 999:
        return None
    return n


def _quad_to_bbox(box: list) -> tuple[int, int, int, int]:
    xs = [float(p[0]) for p in box]
    ys = [float(p[1]) for p in box]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def read_digits_full(
    image_rgb: np.ndarray,
    *,
    min_conf: float = 0.15,
    reader: Any | None = None,
) -> list[OcrHit]:
    """OCR по всему изображению, только цифры 1–3 знака."""
    reader = reader or get_reader()
    raw = reader.readtext(
        image_rgb,
        allowlist="0123456789",
        paragraph=False,
        detail=1,
    )
    hits: list[OcrHit] = []
    for box, text, conf in raw:
        conf_f = float(conf)
        if conf_f < min_conf:
            continue
        number = _parse_number(str(text))
        if number is None:
            continue
        x0, y0, x1, y1 = _quad_to_bbox(box)
        hits.append(
            OcrHit(
                text=str(text),
                number=number,
                conf=conf_f,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                cx=0.5 * (x0 + x1),
                cy=0.5 * (y0 + y1),
            )
        )
    return hits


def read_digits_crop(
    image_rgb: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    min_conf: float = 0.1,
    reader: Any | None = None,
) -> list[OcrHit]:
    """OCR на кропе; координаты переводятся в систему полного изображения."""
    import cv2

    from src.controls.pink_mask import pink_mask_hsv

    h, w = image_rgb.shape[:2]
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w, x1)
    y1 = min(h, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return []
    crop0 = image_rgb[y0:y1, x0:x1]
    ch, cw = crop0.shape[:2]
    # мелкие розовые цифры: EasyOCR стабильно читает при ~3×
    scale = max(3.0, 96.0 / max(ch, cw))
    out_w, out_h = max(1, int(cw * scale)), max(1, int(ch * scale))
    crop = cv2.resize(crop0, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

    # розовый на белом — легче EasyOCR, чем цифры поверх рельефа карты
    pink = pink_mask_hsv(crop0, use_clahe=False, min_saturation=20, min_value=25)
    if int((pink > 0).sum()) >= 20:
        clean0 = np.full_like(crop0, 255)
        clean0[pink > 0] = (220, 20, 140)
        crop_for_ocr = cv2.resize(clean0, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    else:
        crop_for_ocr = crop

    reader = reader or get_reader()
    raw_boxes: list = []
    # сначала цветной апскейл — на части номеров («71») розовая изоляция галлюцинирует
    raw_boxes.extend(
        reader.readtext(crop, allowlist="0123456789", paragraph=False, detail=1)
    )
    if crop_for_ocr is not crop:
        raw_boxes.extend(
            reader.readtext(
                crop_for_ocr, allowlist="0123456789", paragraph=False, detail=1
            )
        )
    hits: list[OcrHit] = []
    for box, text, conf in raw_boxes:
        conf_f = float(conf)
        if conf_f < min_conf:
            continue
        number = _parse_number(str(text))
        if number is None:
            continue
        bx0, by0, bx1, by1 = _quad_to_bbox(box)
        bx0, by0, bx1, by1 = (
            int(bx0 / scale),
            int(by0 / scale),
            int(bx1 / scale),
            int(by1 / scale),
        )
        gx0, gy0, gx1, gy1 = x0 + bx0, y0 + by0, x0 + bx1, y0 + by1
        hits.append(
            OcrHit(
                text=str(text),
                number=number,
                conf=conf_f,
                x0=gx0,
                y0=gy0,
                x1=gx1,
                y1=gy1,
                cx=0.5 * (gx0 + gx1),
                cy=0.5 * (gy0 + gy1),
            )
        )
    return hits
