from __future__ import annotations

from model.base import ActorCriticModel
from model.config import ModelConfig
from model.encoders.transformer import build_transformer_encoder


def build_transformer(config: ModelConfig, action_dim: int) -> ActorCriticModel:
    return build_transformer_encoder(config=config, action_dim=action_dim)
