"""Утилиты устройства / AMP / безопасная загрузка чекпоинтов (Linux+CUDA)."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cuda_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update(
            {
                "device_name": props.name,
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
                "capability": f"{props.major}.{props.minor}",
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )
    return info


def make_grad_scaler(enabled: bool, device: torch.device) -> torch.amp.GradScaler:
    use = bool(enabled) and device.type == "cuda"
    # torch.amp.GradScaler(device_type) — API 2.x; fallback на старый путь
    try:
        return torch.amp.GradScaler("cuda", enabled=use)
    except (TypeError, AttributeError):
        from torch.cuda.amp import GradScaler as CudaGradScaler

        return CudaGradScaler(enabled=use)  # type: ignore[return-value]


@contextmanager
def autocast_ctx(enabled: bool, device: torch.device, dtype: str = "float16") -> Iterator[None]:
    use = bool(enabled) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if dtype in {"bf16", "bfloat16"} else torch.float16
    try:
        with torch.amp.autocast("cuda", enabled=use, dtype=amp_dtype):
            yield
    except (TypeError, AttributeError):
        from torch.cuda.amp import autocast as cuda_autocast

        with cuda_autocast(enabled=use):
            yield


def safe_torch_load(path: str | Path, map_location: Any = None) -> Any:
    """torch.load с weights_only=False (нужно для полных training-чекпоинтов в torch>=2.6)."""
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
