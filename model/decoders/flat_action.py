from __future__ import annotations

import torch
from torch import nn

from model.base import ModelContext
from model.components import make_activation, maybe_norm
from model.config import CommonModelConfig, FlatActionDecoderConfig
from model.decoders.base import PolicyValueDecoder


class FlatActionDecoder(PolicyValueDecoder):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        global_context_dim: int,
        common: CommonModelConfig,
        config: FlatActionDecoderConfig,
    ) -> None:
        super().__init__()
        trunkLayers: list[nn.Module] = []
        mlpLayers = tuple(config.mlpLayers) or (common.hiddenDim,)
        previousWidth = input_dim + global_context_dim
        for width in mlpLayers:
            trunkLayers.append(nn.Linear(previousWidth, width))
            normalization = maybe_norm(common.norm, width)
            if normalization is not None:
                trunkLayers.append(normalization)
            trunkLayers.append(make_activation(common.activation))
            if config.dropout > 0:
                trunkLayers.append(nn.Dropout(config.dropout))
            previousWidth = width

        self.trunk = nn.Sequential(*trunkLayers) if trunkLayers else nn.Identity()
        self.policyHead = nn.Linear(previousWidth, action_dim)
        self.valueHead = nn.Linear(previousWidth, 1)
        self.global_context_dim = global_context_dim

    def forward(
        self, features: torch.Tensor, context: ModelContext | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fusedParts = [features]
        if self.global_context_dim > 0:
            if context is None or context.globalFeatures is None:
                raise ValueError("FlatActionDecoder expected context.globalFeatures to be provided")
            fusedParts.append(context.globalFeatures)

        decoded = self.trunk(torch.cat(fusedParts, dim=-1))
        policyMean = self.policyHead(decoded)
        value = self.valueHead(decoded).squeeze(-1)
        return policyMean, value
