"""Exponential Moving Average весов модели."""

from __future__ import annotations

import copy
from typing import Iterator

import torch
from torch import nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if not v.dtype.is_floating_point:
                v.copy_(msd[k])
                continue
            v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)

    def state_dict(self) -> dict:
        return self.ema.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.ema.load_state_dict(state)

    def parameters(self) -> Iterator[nn.Parameter]:
        return self.ema.parameters()
