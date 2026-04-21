from __future__ import annotations

import torch
from torch import nn

from model.components import make_activation
from model.config import CNNEncoderConfig, CommonModelConfig
from model.encoders.base import TargetEncoder


class CNNEncoder(TargetEncoder):
    def __init__(self, common: CommonModelConfig, config: CNNEncoderConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        inChannels = config.inChannels
        for outChannels, kernelSize, stride, padding in zip(
            config.convChannels,
            config.kernelSizes,
            config.strides,
            config.paddings,
            strict=True,
        ):
            layers.append(
                nn.Conv2d(
                    in_channels=inChannels,
                    out_channels=outChannels,
                    kernel_size=kernelSize,
                    stride=stride,
                    padding=padding,
                )
            )
            layers.append(make_activation(common.activation))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            inChannels = outChannels
        self.encoder = nn.Sequential(*layers)

        if config.pooling == "adaptive_avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))

        self.output_dim = config.convChannels[-1]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(inputs)
        return self.pool(encoded).flatten(1)
