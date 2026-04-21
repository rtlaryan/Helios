from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from scripts.arrayBatch import ArrayBatch


@dataclass
class ActorCriticOutput:
    policyMean: torch.Tensor
    logStd: torch.Tensor
    value: torch.Tensor


@dataclass
class ModelContext:
    globalFeatures: torch.Tensor | None = None
    elementFeatures: torch.Tensor | None = None
    elementMask: torch.Tensor | None = None
    arrayBatch: "ArrayBatch | None" = None
    arraySpec: dict[str, Any] | None = None

    def index(self, index: int | slice | torch.Tensor) -> "ModelContext":
        globalFeatures = None if self.globalFeatures is None else self.globalFeatures[index]
        elementFeatures = None if self.elementFeatures is None else self.elementFeatures[index]
        elementMask = None if self.elementMask is None else self.elementMask[index]
        arrayBatch = None if self.arrayBatch is None else self.arrayBatch.fetch(index)
        return ModelContext(
            globalFeatures=globalFeatures,
            elementFeatures=elementFeatures,
            elementMask=elementMask,
            arrayBatch=arrayBatch,
            arraySpec=self.arraySpec,
        )


@dataclass
class ModelInput:
    targetTensor: torch.Tensor
    context: ModelContext | None = None

    def index(self, index: int | slice | torch.Tensor) -> "ModelInput":
        context = None if self.context is None else self.context.index(index)
        return ModelInput(targetTensor=self.targetTensor[index], context=context)


class ActorCriticModel(nn.Module):
    action_dim: int

    def forward(
        self, inputs: ModelInput | torch.Tensor
    ) -> ActorCriticOutput:  # pragma: no cover - interface
        raise NotImplementedError
