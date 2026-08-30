"""Опциональный sampler с упором на редкие классы / высокий fg_ratio."""

from __future__ import annotations

import numpy as np
from torch.utils.data import WeightedRandomSampler

from src.data.dataset import SampleRecord


def build_fg_weighted_sampler(
    records: list[SampleRecord],
    power: float = 1.0,
) -> WeightedRandomSampler:
    """Чаще сэмплирует тайлы с большей долей foreground."""
    weights = np.array([max(r.fg_ratio, 1e-3) ** power for r in records], dtype=np.float64)
    weights = weights / weights.sum()
    return WeightedRandomSampler(
        weights=torch_weights(weights),
        num_samples=len(records),
        replacement=True,
    )


def torch_weights(weights: np.ndarray):
    import torch

    return torch.as_tensor(weights, dtype=torch.double)
