from __future__ import annotations

import torch
from torch import nn

from model.base import ModelContext


class PolicyValueDecoder(nn.Module):
    def forward(
        self, features: torch.Tensor, context: ModelContext | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover - interface
        raise NotImplementedError
