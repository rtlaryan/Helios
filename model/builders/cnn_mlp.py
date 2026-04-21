from __future__ import annotations

from model.assemblies.actor_critic import ComposedActorCritic
from model.base import ActorCriticModel
from model.config import ModelConfig, elementContextFeatureDim, globalContextFeatureDim
from model.decoders.coordinate_conditioned import CoordinateConditionedDecoder
from model.decoders.flat_action import FlatActionDecoder
from model.encoders.cnn import CNNEncoder


def build_cnn_mlp(config: ModelConfig, action_dim: int) -> ActorCriticModel:
    encoder = CNNEncoder(common=config.common, config=config.encoder.cnn)
    if config.decoder.type == "flat_action":
        decoder = FlatActionDecoder(
            input_dim=encoder.output_dim,
            action_dim=action_dim,
            global_context_dim=globalContextFeatureDim(config.context),
            common=config.common,
            config=config.decoder.flat_action,
        )
    elif config.decoder.type == "coordinate_conditioned":
        decoder = CoordinateConditionedDecoder(
            input_dim=encoder.output_dim,
            action_dim=action_dim,
            global_context_dim=globalContextFeatureDim(config.context),
            element_context_dim=elementContextFeatureDim(config.context),
            common=config.common,
            config=config.decoder.coordinate_conditioned,
        )
    else:  # pragma: no cover - validated in config
        raise ValueError(f"unsupported decoder type: {config.decoder.type}")
    return ComposedActorCritic(
        encoder=encoder,
        decoder=decoder,
        action_dim=action_dim,
        log_std_mode=config.common.logStdMode,
        log_std_init=config.common.logStdInit,
    )
