from __future__ import annotations

from model.base import ActorCriticModel
from model.config import ModelConfig


def build_transformer_encoder(config: ModelConfig, action_dim: int) -> ActorCriticModel:
    del config, action_dim
    raise NotImplementedError(
        "model architecture 'transformer' is scaffolded but not implemented yet"
    )
