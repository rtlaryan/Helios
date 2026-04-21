from __future__ import annotations

import torch
from torch import nn

from model.base import ModelContext
from model.components import make_activation, maybe_norm
from model.config import CommonModelConfig, CoordinateConditionedDecoderConfig
from model.decoders.base import PolicyValueDecoder


class CoordinateConditionedDecoder(PolicyValueDecoder):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        global_context_dim: int,
        element_context_dim: int,
        common: CommonModelConfig,
        config: CoordinateConditionedDecoderConfig,
    ) -> None:
        super().__init__()
        if element_context_dim <= 0:
            raise ValueError("CoordinateConditionedDecoder requires element_context_dim > 0")

        fusedInputDim = input_dim + global_context_dim + element_context_dim
        valueInputDim = input_dim + global_context_dim

        trunkLayers: list[nn.Module] = []
        mlpLayers = tuple(config.mlpLayers) or (common.hiddenDim,)
        previousWidth = fusedInputDim
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
        self.policyHead = nn.Linear(previousWidth, 2)
        self.valueHead = nn.Linear(valueInputDim, 1)
        self.action_dim = action_dim
        self.global_context_dim = global_context_dim

    def forward(
        self, features: torch.Tensor, context: ModelContext | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context is None or context.elementFeatures is None:
            raise ValueError(
                "CoordinateConditionedDecoder requires context.elementFeatures to be provided"
            )

        batchSize, elementCount, _ = context.elementFeatures.shape
        if features.shape[0] != batchSize:
            raise ValueError("global feature batch size must match element context batch size")

        repeatedFeatures = features.unsqueeze(1).expand(-1, elementCount, -1)
        fusedParts = [repeatedFeatures]
        valueParts = [features]

        if context.globalFeatures is not None:
            repeatedGlobal = context.globalFeatures.unsqueeze(1).expand(-1, elementCount, -1)
            fusedParts.append(repeatedGlobal)
            valueParts.append(context.globalFeatures)

        fusedParts.append(context.elementFeatures)
        fused = torch.cat(fusedParts, dim=-1)
        decoded = self.trunk(fused)
        policyMean = self.policyHead(decoded).reshape(batchSize, -1)
        if policyMean.shape[-1] != self.action_dim:
            raise ValueError(
                "coordinate-conditioned decoder produced unexpected action width "
                f"(expected {self.action_dim}, got {policyMean.shape[-1]})"
            )
        value = self.valueHead(torch.cat(valueParts, dim=-1)).squeeze(-1)
        return policyMean, value
