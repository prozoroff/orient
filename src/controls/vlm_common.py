"""Общие промпты и парсинг для VLM-детекции КП."""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_SYSTEM = """\
You are an expert at reading orienteering (ISOM) race maps.
Your job is to find course overprint control points and the start triangle.

Control points: purple/magenta/pink hollow CIRCLES of similar size, each with a
nearby purple number (usually 2 digits, sometimes 3 like 100).
Start/finish: one purple equilateral TRIANGLE (often tip up), no number.

Ignore: map legend, north lines, contour numbers, brown/black map symbols,
course connection lines between controls (thin purple lines without a circle).

Return ONLY valid JSON matching the schema. Coordinates are NORMALIZED:
x = 0 left … 1 right, y = 0 top … 1 bottom, pointing at the CIRCLE CENTER
(not the digit). If unsure about a candidate, omit it.
"""

_USER_TEMPLATE = """\
List every control point (purple circle + number) visible in this image crop,
and the start triangle if present.

JSON schema:
{{
  "controls": [{{"number": <int>, "x": <float 0-1>, "y": <float 0-1>, "confidence": <float 0-1>}}],
  "start": {{"x": <float>, "y": <float>, "confidence": <float>}} | null
}}

Image size: {width}x{height} px.
"""

# OpenAI / Yandex Chat Completions: принудительный JSON в content.
RESPONSE_FORMAT_JSON = {"type": "json_object"}


@dataclass(frozen=True)
class VlmControlHit:
    number: int
    x_norm: float
    y_norm: float
    confidence: float


@dataclass(frozen=True)
class VlmStartHit:
    x_norm: float
    y_norm: float
    confidence: float


@dataclass
class VlmTileResult:
    controls: list[VlmControlHit]
    start: VlmStartHit | None
    raw_text: str


def read_dotenv_value(*keys: str) -> str | None:
    """Прочитать ключ из окружения или `.env` (cwd / корень репо)."""
    for key in keys:
        env = os.environ.get(key, "").strip()
        if env:
            return env
    root = Path(__file__).resolve().parents[2]
    for path in (Path.cwd() / ".env", root / ".env"):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            if k not in keys:
                continue
            val = v.strip().strip('"').strip("'")
            if val:
                return val
    return None


def encode_jpeg_b64(
    rgb: np.ndarray, *, quality: int = 90, max_side: int = 2048
) -> tuple[str, int, int, float]:
    """JPEG base64 + W/H после даунскейла + scale (sent/original)."""
    h, w = rgb.shape[:2]
    scale = 1.0
    out = rgb
    long_side = max(h, w)
    if long_side > max_side:
        scale = max_side / float(long_side)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        out = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    b64 = base64.standard_b64encode(buf.tobytes()).decode("ascii")
    sh, sw = out.shape[:2]
    return b64, sw, sh, scale


def message_text(message: Any) -> str:
    """Текст из ``message.content`` (structured JSON)."""
    if message is None:
        return ""
    for attr in ("content", "text", "reasoning_content", "reasoning"):
        val = getattr(message, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    if isinstance(message, dict):
        for key in ("content", "text", "reasoning_content", "reasoning"):
            val = message.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""


def extract_json(text: str) -> dict[str, Any]:
    """Разобрать JSON из content при response_format=json_object."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Пустой content у VLM (ожидался JSON)")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Ожидался JSON-объект, получено {type(data).__name__}")
    return data


def parse_vlm_response_text(text: str) -> VlmTileResult:
    return parse_controls_json(extract_json(text))


def parse_controls_json(data: dict[str, Any]) -> VlmTileResult:
    controls: list[VlmControlHit] = []
    for item in data.get("controls") or []:
        if not isinstance(item, dict):
            continue
        try:
            number = int(item["number"])
            x = float(item["x"])
            y = float(item["y"])
            conf = float(item.get("confidence", 0.7))
        except (KeyError, TypeError, ValueError):
            continue
        if not (10 <= number <= 199):
            continue
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            continue
        controls.append(
            VlmControlHit(
                number=number,
                x_norm=x,
                y_norm=y,
                confidence=float(np.clip(conf, 0.0, 1.0)),
            )
        )

    start: VlmStartHit | None = None
    raw_start = data.get("start")
    if isinstance(raw_start, dict):
        try:
            sx, sy = float(raw_start["x"]), float(raw_start["y"])
            sc = float(raw_start.get("confidence", 0.7))
            if 0.0 <= sx <= 1.0 and 0.0 <= sy <= 1.0:
                start = VlmStartHit(sx, sy, float(np.clip(sc, 0.0, 1.0)))
        except (KeyError, TypeError, ValueError):
            start = None

    return VlmTileResult(controls=controls, start=start, raw_text="")


def build_user_prompt(width: int, height: int) -> str:
    return _USER_TEMPLATE.format(width=width, height=height)


def system_prompt() -> str:
    return _SYSTEM
