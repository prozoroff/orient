"""OpenAI Vision: детекция КП по изображению карты (JSON)."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.controls.vlm_common import (
    RESPONSE_FORMAT_JSON,
    VlmTileResult,
    build_user_prompt,
    encode_jpeg_b64,
    message_text,
    parse_vlm_response_text,
    read_dotenv_value,
    system_prompt,
)


def _load_api_key(api_key: str | None) -> str:
    if api_key:
        return api_key
    found = read_dotenv_value("OPENAI_API_KEY")
    if found:
        return found
    raise RuntimeError(
        "Нет OPENAI_API_KEY. Экспортируйте в окружение или положите в .env:\n"
        "  OPENAI_API_KEY=sk-...\n"
        "Из РФ OpenAI часто отвечает 403 — используйте provider='yandex'."
    )


def call_openai_controls(
    rgb: np.ndarray,
    *,
    model: str = "gpt-4.1-mini",
    api_key: str | None = None,
    max_side: int = 2048,
    detail: str = "high",
    temperature: float = 0.0,
    client: Any | None = None,
) -> VlmTileResult:
    """Один вызов OpenAI Vision → нормализованные КП (+ старт)."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "Для VLM-детекции нужен пакет openai: pip install openai"
        ) from e

    key = _load_api_key(api_key)
    client = client or OpenAI(api_key=key)

    b64, sw, sh, _scale = encode_jpeg_b64(rgb, max_side=max_side)
    user = build_user_prompt(sw, sh)

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format=RESPONSE_FORMAT_JSON,
        messages=[
            {"role": "system", "content": system_prompt()},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": detail,
                        },
                    },
                ],
            },
        ],
    )
    text = message_text(resp.choices[0].message)
    if not text:
        raise ValueError("Пустой content у VLM при response_format=json_object")
    parsed = parse_vlm_response_text(text)
    parsed.raw_text = text
    return parsed