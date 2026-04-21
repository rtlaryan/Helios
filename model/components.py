from __future__ import annotations

from torch import nn


def make_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"unsupported activation: {name}")


def maybe_norm(name: str, width: int) -> nn.Module | None:
    if name == "none":
        return None
    if name == "layernorm":
        return nn.LayerNorm(width)
    raise ValueError(f"unsupported norm: {name}")
