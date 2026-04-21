from __future__ import annotations

import torch
from torch import nn

from model.base import ActorCriticModel, ActorCriticOutput, ModelInput
from model.decoders.base import PolicyValueDecoder
from model.encoders.base import TargetEncoder


class ComposedActorCritic(ActorCriticModel):
    def __init__(
        self,
        encoder: TargetEncoder,
        decoder: PolicyValueDecoder,
        action_dim: int,
        log_std_mode: str,
        log_std_init: float,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.encoder = encoder
        self.decoder = decoder
        if log_std_mode == "global":
            self.logStd = nn.Parameter(torch.tensor([log_std_init], dtype=torch.float32))
        else:
            self.logStd = nn.Parameter(torch.full((action_dim,), log_std_init))
        self.logStdMode = log_std_mode

    def forward(self, inputs: ModelInput | torch.Tensor) -> ActorCriticOutput:
        modelInput = inputs if isinstance(inputs, ModelInput) else ModelInput(targetTensor=inputs)
        features = self.encoder(modelInput.targetTensor)
        policyMean, value = self.decoder(features, modelInput.context)
        if self.logStdMode == "global":
            logStd = self.logStd.view(1, 1).expand_as(policyMean)
        else:
            logStd = self.logStd.view(1, -1).expand_as(policyMean)
        return ActorCriticOutput(policyMean=policyMean, logStd=logStd, value=value)
