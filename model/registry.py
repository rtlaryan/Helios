from __future__ import annotations

from collections.abc import Callable

from model.base import ActorCriticModel
from model.builders.cnn_mlp import build_cnn_mlp
from model.builders.transformer import build_transformer
from model.config import ModelConfig

ModelBuilder = Callable[[ModelConfig, int], ActorCriticModel]

_REGISTRY: dict[str, ModelBuilder] = {
    "cnn_mlp": build_cnn_mlp,
    "transformer": build_transformer,
}


def build_model(config: ModelConfig, action_dim: int) -> ActorCriticModel:
    try:
        builder = _REGISTRY[config.architecture]
    except KeyError as error:  # pragma: no cover - validation should catch this first
        raise ValueError(f"unknown model architecture: {config.architecture}") from error
    return builder(config, action_dim)


def registered_architectures() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
