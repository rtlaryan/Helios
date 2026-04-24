from __future__ import annotations

from typing import Literal

import torch

from scripts.arrayBatch import ArrayBatch
from simulation.arraySim import arrayResponseBatch, arrayResponseBatchSharedGrid

SimulationBackend = Literal["v1", "v2"]


def _validate_backend(backend: str) -> SimulationBackend:
    if backend not in {"v1", "v2"}:
        raise ValueError("simulation.backend must be either 'v1' or 'v2'")
    return backend  # type: ignore[return-value]


def responseBatch(
    batch: ArrayBatch,
    relativeTargetAZEL: tuple[torch.Tensor, torch.Tensor],
    *,
    backend: SimulationBackend = "v1",
    chunkSize: int | None = None,
    dB: bool = False,
    normalize: bool = False,
) -> torch.Tensor:
    backend = _validate_backend(backend)
    if backend == "v1":
        return arrayResponseBatch(
            batch,
            relativeTargetAZEL,
            chunkSize=chunkSize,
            dB=dB,
            normalize=normalize,
        )
    raise ValueError("simulation.backend='v2' does not support batched/per-sample az/el grids")


def responseBatchSharedGrid(
    batch: ArrayBatch,
    relativeTargetAZEL: tuple[torch.Tensor, torch.Tensor],
    *,
    backend: SimulationBackend = "v1",
    chunkSize: int | None = None,
    dB: bool = False,
    normalize: bool = False,
) -> torch.Tensor:
    backend = _validate_backend(backend)
    if backend == "v1":
        return arrayResponseBatchSharedGrid(
            batch,
            relativeTargetAZEL,
            chunkSize=chunkSize,
            dB=dB,
            normalize=normalize,
        )

    del chunkSize
    from simulation.arraySimV2 import arrayResponseBatchSharedGridV2

    return arrayResponseBatchSharedGridV2(
        batch,
        relativeTargetAZEL,
        dB=dB,
        normalize=normalize,
    )
