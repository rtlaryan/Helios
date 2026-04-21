from __future__ import annotations

import torch
from torch import nn


class TargetEncoder(nn.Module):
    output_dim: int

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError
