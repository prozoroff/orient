"""Yandex Cloud AI Studio: детекция КП через multimodal LLM.

Default: ``qwen3.6-35b-a3b`` (нативно multimodal; gemma часто снята с каталога).
У Qwen3.6 reasoning в Yandex включён по умолчанию — запросы шлют
``reasoningOptions.mode=DISABLED``, иначе ``content`` пустой при JSON.
Текстовые модели (yandexgpt-lite, gpt-oss и т.п.) картинку не видят.
"""

from __future__ import annotations

from typing import Any

import httpx
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

_YANDEX_BASE_URL = "https://llm.api.cloud.yandex.net/v1"
# Qwen3.5/3.6 в AI Studio — нативно multimodal; gemma часто снята с каталога.
_DEFAULT_MODEL = "qwen3.6-35b-a3b"
# Имена без «-vl»: Qwen3.5/3.6 и Alice AI — multimodal с vision encoder.
_VISION_MODEL_HINTS = (
    "gemma",
    "vl-",
    "-vl",
    "vision",
    "multimodal",
    "qwen2.5-vl",
    "qwen3-vl",
    "qwen3.5",
    "qwen3.6",
    "aliceai",
)
# Reasoning по умолчанию у Qwen3.6 в Yandex → пустой content при json_object.
# Пробуем несколько имён поля: OpenAI-compat / Foundation Models / Qwen.
_REASONING_DISABLE_BODIES: tuple[dict[str, Any], ...] = (
    {"reasoningOptions": {"mode": "DISABLED"}},
    {"reasoning_mode": "DISABLED"},
    {"enable_thinking": False},
    {"chat_template_kwargs": {"enable_thinking": False}},
    {},  # последний шанс без флага
)


def _load_yandex_creds(
    api_key: str | None,
    folder_id: str | None,
) -> tuple[str, str]:
    key = api_key or read_dotenv_value(
        "YC_API_KEY", "YANDEX_API_KEY", "YC_IAM_TOKEN", "YC_TOKEN"
    )
    folder = folder_id or read_dotenv_value("YC_FOLDER_ID", "YANDEX_FOLDER_ID")
    if not key:
        raise RuntimeError(
            "Нет ключа Yandex Cloud. Положите в .env:\n"
            "  YC_API_KEY=...\n"
            "  YC_FOLDER_ID=...\n"
            "API-ключ сервисного аккаунта (не OAuth). "
            "Роль на каталоге: ai.languageModels.user"
        )
    if not folder:
        raise RuntimeError(
            "Нет YC_FOLDER_ID. ID каталога — в консоли Yandex Cloud."
        )
    return key.strip(), folder.strip()


def _auth_header_value(key: str) -> str:
    """Yandex API-ключ → ``Api-Key …``; IAM-токен → ``Bearer …``."""
    k = key.strip()
    low = k.lower()
    if low.startswith("api-key ") or low.startswith("bearer "):
        return k
    if k.startswith("t1.") or len(k) > 200:
        return f"Bearer {k}"
    return f"Api-Key {k}"


def _resolve_model(model: str, folder_id: str) -> str:
    m = model.strip()
    if m.startswith("gpt://") or m.startswith("ds://"):
        return m
    return f"gpt://{folder_id}/{m}/latest"


def _model_short_name(model_uri: str) -> str:
    if not model_uri.startswith("gpt://"):
        return model_uri
    parts = model_uri.split("/")
    # gpt://folder/name/latest → name
    if len(parts) >= 4:
        return parts[3]
    return parts[-1]


def list_yandex_chat_models(
    *,
    api_key: str | None = None,
    folder_id: str | None = None,
) -> list[str]:
    """Список URI chat-моделей, доступных в каталоге."""
    key, folder = _load_yandex_creds(api_key, folder_id)
    auth = _auth_header_value(key)
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(
            f"{_YANDEX_BASE_URL}/models",
            headers={"Authorization": auth, "x-folder-id": folder},
        )
        resp.raise_for_status()
        data = resp.json()
    items = data.get("data") or data.get("models") or []
    out: list[str] = []
    for item in items:
        mid = item.get("id") or item.get("uri") or ""
        if isinstance(mid, str) and mid.startswith("gpt://"):
            out.append(mid)
    return out


def _looks_vision_capable(model_uri: str) -> bool:
    s = model_uri.lower()
    return any(h in s for h in _VISION_MODEL_HINTS)


def ensure_vision_model_available(
    model: str,
    *,
    api_key: str | None = None,
    folder_id: str | None = None,
) -> str:
    """
    Проверить, что model есть в каталоге. Вернуть полный URI.
    Если multimodal-моделей нет — понятная ошибка (не молчаливый 403).
    """
    key, folder = _load_yandex_creds(api_key, folder_id)
    model_uri = _resolve_model(model, folder)
    short = _model_short_name(model_uri).lower()

    try:
        available = list_yandex_chat_models(api_key=key, folder_id=folder)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Не удалось получить список моделей Yandex Cloud: {exc}"
        ) from exc

    shorts = {_model_short_name(u).lower(): u for u in available}
    visionish = [u for u in available if _looks_vision_capable(u)]

    if short in shorts:
        return shorts[short]

    # точное совпадение полного URI
    if model_uri in available:
        return model_uri

    lines = [
        f"Модель «{model}» недоступна в каталоге {folder}.",
        "",
        "Проверка API: ключ и роль ai.languageModels.user в порядке "
        "(yandexgpt-lite отвечает 200), но multimodal-модели нет.",
        "Ранее используемая gemma-3-27b-it в AI Studio часто снимают "
        "(ответ: «Model … is not available anymore» → в chat это 403 Forbidden).",
        "",
        "Доступные chat-модели сейчас:",
    ]
    for u in available:
        mark = "  ← возможно vision" if _looks_vision_capable(u) else ""
        lines.append(f"  - {_model_short_name(u)}{mark}")
    if not visionish:
        lines += [
            "",
            "Среди них нет multimodal (с картинками). Текстовые модели "
            "image_url игнорируют — для детекции КП по фото они не подходят.",
            "",
            "Что можно сделать:",
            "1) В консоли AI Studio проверить, появилась ли новая VL-модель, "
            "и передать model=\"…\".",
            "2) Поднять Gemma/Qwen2.5-VL на GPU VM в Yandex Cloud (vLLM) "
            "и указать свой base_url.",
            "3) Пока использовать классический detect_controls(mode='circle_first').",
        ]
    raise RuntimeError("\n".join(lines))


def _make_openai_client(api_key: str, folder_id: str) -> Any:
    """OpenAI SDK с заголовком ``Authorization: Api-Key`` (не Bearer)."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "Для VLM-детекции нужен пакет openai: pip install openai"
        ) from e

    auth = _auth_header_value(api_key)

    def _force_auth(request: httpx.Request) -> None:
        request.headers["Authorization"] = auth
        request.headers["x-folder-id"] = folder_id

    http_client = httpx.Client(
        event_hooks={"request": [_force_auth]},
        timeout=httpx.Timeout(120.0, connect=30.0),
    )
    return OpenAI(
        api_key=api_key,
        base_url=_YANDEX_BASE_URL,
        http_client=http_client,
        default_headers={"x-folder-id": folder_id},
    )


def _call_via_sdk(
    rgb: np.ndarray,
    *,
    model: str,
    api_key: str,
    folder_id: str,
    max_side: int,
    temperature: float,
) -> VlmTileResult | None:
    try:
        from yandex_ai_studio_sdk import AIStudio
    except ImportError:
        return None

    b64, sw, sh, _ = encode_jpeg_b64(rgb, max_side=max_side)
    prompt = system_prompt() + "\n\n" + build_user_prompt(sw, sh)
    short_model = _model_short_name(model)

    sdk = AIStudio(folder_id=folder_id, auth=api_key)
    chat_model = sdk.chat.completions(short_model)
    # reasoning_mode=DISABLED: иначе Qwen3.6 тратит токены на thinking → content пустой.
    for kwargs in (
        {
            "temperature": temperature,
            "response_format": RESPONSE_FORMAT_JSON,
            "reasoning_mode": "DISABLED",
        },
        {"temperature": temperature, "response_format": RESPONSE_FORMAT_JSON},
        {"temperature": temperature, "reasoning_mode": "DISABLED"},
        {"temperature": temperature},
    ):
        try:
            chat_model = chat_model.configure(**kwargs)
            break
        except TypeError:
            continue
    request = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
            ],
        }
    ]
    result = chat_model.run(request)
    text = message_text(result)
    if not text:
        return None  # fallback на OpenAI-path с reasoningOptions
    parsed = parse_vlm_response_text(text)
    parsed.raw_text = text
    return parsed


def call_yandex_controls(
    rgb: np.ndarray,
    *,
    model: str = _DEFAULT_MODEL,
    api_key: str | None = None,
    folder_id: str | None = None,
    max_side: int = 1536,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    client: Any | None = None,
    skip_preflight: bool = False,
) -> VlmTileResult:
    """Один вызов Yandex AI Studio (мультимодальный chat) → JSON с КП."""
    key, folder = _load_yandex_creds(api_key, folder_id)
    if skip_preflight:
        model_uri = _resolve_model(model, folder)
    else:
        model_uri = ensure_vision_model_available(
            model, api_key=key, folder_id=folder
        )

    if client is None:
        try:
            via_sdk = _call_via_sdk(
                rgb,
                model=model_uri,
                api_key=key,
                folder_id=folder,
                max_side=max_side,
                temperature=temperature,
            )
            if via_sdk is not None:
                return via_sdk
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "403" in msg or "forbidden" in msg or "not available" in msg:
                raise RuntimeError(_forbidden_hint(exc, folder, model_uri)) from exc

    b64, sw, sh, _scale = encode_jpeg_b64(rgb, max_side=max_side)
    prompt = system_prompt() + "\n\n" + build_user_prompt(sw, sh)
    openai_client = client or _make_openai_client(key, folder)

    message = None
    finish = None
    last_exc: BaseException | None = None
    for extra in _REASONING_DISABLE_BODIES:
        create_kwargs: dict[str, Any] = {
            "model": model_uri,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": RESPONSE_FORMAT_JSON,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                            },
                        },
                    ],
                },
            ],
        }
        if extra:
            create_kwargs["extra_body"] = extra
        try:
            resp = openai_client.chat.completions.create(**create_kwargs)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "403" in msg or "forbidden" in msg or "permission" in msg:
                raise RuntimeError(_forbidden_hint(exc, folder, model_uri)) from exc
            last_exc = exc
            continue
        message = resp.choices[0].message
        finish = getattr(resp.choices[0], "finish_reason", None)
        text = message_text(message)
        if text:
            parsed = parse_vlm_response_text(text)
            parsed.raw_text = text
            return parsed

    if last_exc is not None and message is None:
        raise last_exc
    raise ValueError(
        f"Пустой content у VLM ({_model_short_name(model_uri)}, "
        f"finish_reason={finish!r}) при response_format=json_object.\n"
        "У Qwen3.6 в Yandex reasoning включён по умолчанию — код пробует "
        "отключить его (reasoningOptions / enable_thinking). "
        "Если всё ещё пусто — finish_reason=length (мало max_tokens) "
        "или модель не пишет JSON в content. Попробуйте classical detector."
    )
def _forbidden_hint(exc: BaseException, folder_id: str, model_uri: str) -> str:
    return (
        f"Yandex Cloud отклонил модель {model_uri} (каталог {folder_id}).\n"
        f"raw: {exc}\n\n"
        "Частая причина: multimodal-модель снята с AI Studio "
        "(gemma-3-27b-it → «not available anymore»).\n"
        "Проверка: list_yandex_chat_models() / ensure_vision_model_available().\n"
        "Пока multimodal нет — используйте detect_controls(mode='circle_first')."
    )
