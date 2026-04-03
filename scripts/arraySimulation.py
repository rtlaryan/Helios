from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

import torch

from .arrayBatch import ArrayBatch

ChunkShapeStrategy = Literal["balanced", "cap_reduction", "grid_first"]

DEFAULT_CHUNK_SHAPE_STRATEGY: ChunkShapeStrategy = "balanced"
DEFAULT_REDUCTION_TILE_CAP = 256

_ACTIVE_CHUNK_SHAPE_STRATEGY: ChunkShapeStrategy = DEFAULT_CHUNK_SHAPE_STRATEGY
_ACTIVE_REDUCTION_TILE_CAP = DEFAULT_REDUCTION_TILE_CAP


def _validateReductionTileCap(reductionTileCap: int) -> int:
    reductionTileCap = int(reductionTileCap)
    if reductionTileCap <= 0:
        raise ValueError("reductionTileCap must be positive")
    return reductionTileCap


def getChunkShapeStrategy() -> tuple[ChunkShapeStrategy, int]:
    return _ACTIVE_CHUNK_SHAPE_STRATEGY, _ACTIVE_REDUCTION_TILE_CAP


def setChunkShapeStrategy(
    strategy: ChunkShapeStrategy,
    *,
    reductionTileCap: int = DEFAULT_REDUCTION_TILE_CAP,
) -> None:
    global _ACTIVE_CHUNK_SHAPE_STRATEGY, _ACTIVE_REDUCTION_TILE_CAP
    _ACTIVE_CHUNK_SHAPE_STRATEGY = strategy
    _ACTIVE_REDUCTION_TILE_CAP = _validateReductionTileCap(reductionTileCap)


@contextmanager
def useChunkShapeStrategy(
    strategy: ChunkShapeStrategy,
    *,
    reductionTileCap: int = DEFAULT_REDUCTION_TILE_CAP,
) -> Iterator[None]:
    previousStrategy, previousReductionTileCap = getChunkShapeStrategy()
    setChunkShapeStrategy(strategy, reductionTileCap=reductionTileCap)
    try:
        yield
    finally:
        setChunkShapeStrategy(
            previousStrategy,
            reductionTileCap=previousReductionTileCap,
        )


def precomputeGridWaveVector(
    wavelength: float, azimuth: torch.Tensor, elevation: torch.Tensor
) -> tuple[torch.Tensor, torch.Size]:
    azimuth, elevation = torch.broadcast_tensors(azimuth, elevation)
    gridShape = azimuth.shape

    propagationVectors = torch.stack(
        [
            torch.cos(elevation) * torch.cos(azimuth),
            torch.cos(elevation) * torch.sin(azimuth),
            torch.sin(elevation),
        ],
        dim=-1,
    )

    waveVector = (2 * torch.pi / wavelength) * propagationVectors
    return waveVector, gridShape


def normalizePower(
    power: torch.Tensor, sumNorm: bool = False, threshold: float = 1e-10
) -> torch.Tensor:
    if sumNorm:
        powerFlat = power.flatten(1)
        return powerFlat / powerFlat.sum(dim=1, keepdim=True).clamp_min(threshold)
    else:
        powerMax = power.flatten(1).amax(dim=1).clamp_min(threshold)
        return power / powerMax.view(-1, *([1] * (power.ndim - 1)))


def todB(power: torch.Tensor, threshold: float = 1e-10) -> torch.Tensor:
    responsedB = 10.0 * torch.log10(power.clamp_min(threshold))
    return responsedB


def toLinear(powerdB: torch.Tensor) -> torch.Tensor:
    responseLinear = torch.pow(10, powerdB / 10)
    return responseLinear


def ensureBatchedGrid(
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    batchSize: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Accepts either:
      shared grid: [...]
      batched grid: [B, ...]

    Returns:
      azimuth, elevation: [B, ...]
    """
    azimuth, elevation = torch.broadcast_tensors(azimuth, elevation)

    if azimuth.ndim > 0 and azimuth.shape[0] == batchSize:
        return azimuth, elevation

    azimuth = azimuth.unsqueeze(0).expand(batchSize, *azimuth.shape)
    elevation = elevation.unsqueeze(0).expand(batchSize, *elevation.shape)
    return azimuth, elevation


def resolveResponseChunkSize(
    batchSize: int,
    elementCount: int,
    gridSize: int,
    realDtype: torch.dtype,
    device: torch.device,
    requestedChunkSize: int | None = None,
) -> int:
    batchSize = int(batchSize)
    elementCount = int(elementCount)
    gridSize = int(gridSize)
    fullProduct = max(1, batchSize * elementCount * max(1, gridSize))
    if requestedChunkSize is not None:
        requestedChunkSize = int(requestedChunkSize)
        if requestedChunkSize <= 0:
            raise ValueError("chunkSize must be positive")
        return int(min(fullProduct, requestedChunkSize))

    if device.type != "cuda":
        defaultProduct = max(
            1,
            batchSize * max(1, min(elementCount, 512)) * max(1, min(gridSize, 512)),
        )
        return int(min(fullProduct, defaultProduct))

    try:
        freeBytes, _totalBytes = torch.cuda.mem_get_info(device)
    except RuntimeError:
        freeBytes = 512 * 1024 * 1024

    budgetFloorBytes = 128 * 1024 * 1024
    budgetCapBytes = 1536 * 1024 * 1024
    budgetBytes = int(min(budgetCapBytes, max(budgetFloorBytes, freeBytes * 0.10)))

    realBytes = torch.finfo(realDtype).bits // 8
    bytesPerTriplet = max(1, realBytes * 8)
    autoChunkSize = max(1, budgetBytes // bytesPerTriplet)
    return int(min(fullProduct, autoChunkSize))


def chooseChunkShape(
    batchRemaining: int,
    elementCount: int,
    gridRemaining: int,
    chunkSize: int,
    strategy: ChunkShapeStrategy | None = None,
    reductionTileCap: int | None = None,
) -> tuple[int, int, int]:
    """
    Choose (Bc, Nc, Pc) such that Bc * Nc * Pc <= chunkSize.
    """
    batchRemaining = int(batchRemaining)
    elementCount = int(elementCount)
    gridRemaining = int(gridRemaining)
    chunkSize = int(chunkSize)
    if chunkSize <= 0:
        raise ValueError("chunkSize must be positive")
    strategy = _ACTIVE_CHUNK_SHAPE_STRATEGY if strategy is None else strategy
    reductionTileCap = (
        _ACTIVE_REDUCTION_TILE_CAP if reductionTileCap is None else reductionTileCap
    )
    reductionTileCap = _validateReductionTileCap(reductionTileCap)

    def _balanced_shape() -> tuple[int, int, int]:
        bc = min(batchRemaining, max(1, int(round(chunkSize ** (1 / 3)))))
        pc = min(gridRemaining, max(1, int(round((chunkSize / bc) ** 0.5))))
        nc = min(elementCount, max(1, chunkSize // (bc * pc)))
        pc = min(gridRemaining, max(1, chunkSize // (bc * nc)))
        return bc, nc, pc

    if strategy == "balanced":
        return _balanced_shape()

    if strategy == "cap_reduction":
        bc, nc, pc = _balanced_shape()
        nc = min(nc, elementCount, reductionTileCap)
        pc = min(gridRemaining, max(1, chunkSize // (bc * nc)))
        return bc, nc, pc

    if strategy == "grid_first":
        nc = min(elementCount, reductionTileCap, chunkSize)
        pc = min(gridRemaining, max(1, int(round((chunkSize / nc) ** 0.5))))
        bc = min(batchRemaining, max(1, chunkSize // (nc * pc)))
        pc = min(gridRemaining, max(1, chunkSize // (bc * nc)))
        return bc, nc, pc

    raise ValueError(f"unknown chunk shape strategy: {strategy}")


def arrayResponseCoreOLD(
    elementLocalPosition: torch.Tensor,
    weights: torch.Tensor,
    wavelength: float,
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    gain: torch.Tensor,
    chunkSize: int | None = 8192,
    dB: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    batchSize = elementLocalPosition.shape[0]

    waveVector, gridShape = precomputeGridWaveVector(wavelength, azimuth, elevation)

    # -----------------------------------------
    # Handle shared grid [...] vs batched grid [B, ...]
    # -----------------------------------------
    if waveVector.ndim >= 2 and waveVector.shape[0] == batchSize:
        # batched grid: [B, ..., 3]
        spatialShape = gridShape[1:]
        waveVectorFlat = waveVector.reshape(batchSize, -1, 3).transpose(1, 2)  # [B, 3, P]
        gridSize = waveVectorFlat.shape[-1]

        if chunkSize is None:
            chunkSize = gridSize

        weightsConj = weights.conj().clone()

        fullResponse = []
        for chunk in range(0, gridSize, chunkSize):
            waveVectorChunk = waveVectorFlat[:, :, chunk : chunk + chunkSize]  # [B, 3, Pc]
            phaseChunk = torch.einsum("bin,bip->bnp", elementLocalPosition, waveVectorChunk)
            arrayManifoldChunk = torch.exp(1j * phaseChunk)
            chunkResponse = torch.einsum("bn,bnp->bp", weightsConj, arrayManifoldChunk)
            fullResponse.append(chunkResponse)

        fullResponse = torch.cat(fullResponse, dim=1).reshape(batchSize, *spatialShape)

    else:
        # shared grid: [..., 3]
        spatialShape = gridShape
        waveVectorFlat = waveVector.reshape(-1, 3).transpose(0, 1)  # [3, P]
        gridSize = waveVectorFlat.shape[-1]

        if chunkSize is None:
            chunkSize = gridSize

        weightsConj = weights.conj().clone()

        fullResponse = []
        for chunk in range(0, gridSize, chunkSize):
            waveVectorChunk = waveVectorFlat[:, chunk : chunk + chunkSize]  # [3, Pc]
            phaseChunk = torch.einsum("bin,ip->bnp", elementLocalPosition, waveVectorChunk)
            arrayManifoldChunk = torch.exp(1j * phaseChunk)
            chunkResponse = torch.einsum("bn,bnp->bp", weightsConj, arrayManifoldChunk)
            fullResponse.append(chunkResponse)

        fullResponse = torch.cat(fullResponse, dim=1).reshape(batchSize, *spatialShape)

    fullResponse = fullResponse.abs().square()

    if normalize:
        fullResponse = normalizePower(fullResponse)

    if dB:
        gainView = gain.view(-1, *([1] * (fullResponse.ndim - 1)))
        fullResponse = 10.0 * torch.log10(fullResponse.clamp_min(1e-10)) + gainView

    return fullResponse


def _arrayResponseCoreReference(
    elementLocalPosition: torch.Tensor,
    weights: torch.Tensor,
    wavelength: float,
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    gain: torch.Tensor,
    chunkSize: int | None = 2_000_000,
    dB: bool = True,
    normalize: bool = True,
    clearCache: bool = False,
) -> torch.Tensor:
    """
    Frozen reference implementation used as the ground truth for parity tests.
    elementLocalPosition: [B, 3, N]
    weights: [B, N]
    gain: [B]
    azimuth/elevation: [B, ...] or shared [...]

    chunkSize controls the dominant temporary:
      [Bc, Nc, Pc]
    """
    batchSize = elementLocalPosition.shape[0]
    elementCount = elementLocalPosition.shape[-1]
    device = weights.device
    realDtype = weights.real.dtype
    complexDtype = weights.dtype

    waveVector, gridShape = precomputeGridWaveVector(wavelength, azimuth, elevation)
    isBatchedGrid = waveVector.ndim >= 2 and waveVector.shape[0] == batchSize
    spatialShape = gridShape[1:] if isBatchedGrid else gridShape
    waveVectorFlat = (
        waveVector.reshape(batchSize, -1, 3).transpose(1, 2)
        if isBatchedGrid
        else waveVector.reshape(-1, 3).transpose(0, 1)
    )
    gridSize = waveVectorFlat.shape[-1]

    weightsConj = weights.conj()
    chunkSize = resolveResponseChunkSize(
        batchSize=batchSize,
        elementCount=elementCount,
        gridSize=gridSize,
        realDtype=realDtype,
        device=device,
        requestedChunkSize=chunkSize,
    )

    fullResponse = torch.empty(
        (batchSize, *spatialShape),
        device=device,
        dtype=realDtype,
    )

    bStart = 0
    while bStart < batchSize:
        batchRemaining = batchSize - bStart
        bc, _, _ = chooseChunkShape(
            batchRemaining=batchRemaining,
            elementCount=elementCount,
            gridRemaining=gridSize,
            chunkSize=chunkSize,
        )
        bEnd = min(batchSize, bStart + bc)
        bc = bEnd - bStart

        posB = elementLocalPosition[bStart:bEnd]  # [Bc, 3, N]
        wB = weightsConj[bStart:bEnd]  # [Bc, N]
        waveB = waveVectorFlat[bStart:bEnd] if isBatchedGrid else waveVectorFlat

        responseFlatB = torch.empty(
            (bc, gridSize),
            device=device,
            dtype=complexDtype,
        )

        pStart = 0
        while pStart < gridSize:
            gridRemaining = gridSize - pStart
            _, nc, pc = chooseChunkShape(
                batchRemaining=bc,
                elementCount=elementCount,
                gridRemaining=gridRemaining,
                chunkSize=chunkSize,
            )
            pEnd = min(gridSize, pStart + pc)
            pc = pEnd - pStart

            waveChunk = waveB[:, :, pStart:pEnd] if isBatchedGrid else waveB[:, pStart:pEnd]
            responseChunk = torch.zeros(
                (bc, pc),
                device=device,
                dtype=complexDtype,
            )

            nStart = 0
            while nStart < elementCount:
                nEnd = min(elementCount, nStart + nc)

                posChunk = posB[:, :, nStart:nEnd]  # [Bc, 3, Nc]
                wChunk = wB[:, nStart:nEnd]  # [Bc, Nc]

                if isBatchedGrid:
                    phaseChunk = torch.einsum("bin,bip->bnp", posChunk, waveChunk)
                else:
                    phaseChunk = torch.einsum("bin,ip->bnp", posChunk, waveChunk)
                manifoldChunk = torch.exp(1j * phaseChunk)
                responseChunk += torch.einsum("bn,bnp->bp", wChunk, manifoldChunk)

                del posChunk, wChunk, phaseChunk, manifoldChunk
                nStart = nEnd

            responseFlatB[:, pStart:pEnd] = responseChunk
            del waveChunk, responseChunk
            pStart = pEnd

        fullResponse[bStart:bEnd] = responseFlatB.abs().square().reshape(bc, *spatialShape)
        del posB, wB, waveB, responseFlatB
        bStart = bEnd

    del waveVector, waveVectorFlat, weightsConj

    if normalize:
        fullResponse = normalizePower(fullResponse)

    if dB:
        gainView = gain.view(-1, *([1] * (fullResponse.ndim - 1)))
        fullResponse = todB(fullResponse) + gainView
        del gainView

    if clearCache and device.type == "cuda":
        torch.cuda.empty_cache()

    return fullResponse


def arrayResponseCore(
    elementLocalPosition: torch.Tensor,
    weights: torch.Tensor,
    wavelength: float,
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    gain: torch.Tensor,
    chunkSize: int | None = 2_000_000,
    dB: bool = True,
    normalize: bool = True,
    clearCache: bool = False,
) -> torch.Tensor:
    """
    elementLocalPosition: [B, 3, N]
    weights: [B, N]
    gain: [B]
    azimuth/elevation: [B, ...] or shared [...]

    chunkSize controls the dominant temporary:
      [Bc, Nc, Pc]
    """
    batchSize = elementLocalPosition.shape[0]
    elementCount = elementLocalPosition.shape[-1]
    device = weights.device
    realDtype = weights.real.dtype
    complexDtype = weights.dtype

    waveVector, gridShape = precomputeGridWaveVector(wavelength, azimuth, elevation)
    isBatchedGrid = waveVector.ndim >= 2 and waveVector.shape[0] == batchSize
    spatialShape = gridShape[1:] if isBatchedGrid else gridShape
    waveVectorFlat = (
        waveVector.reshape(batchSize, -1, 3).transpose(1, 2)
        if isBatchedGrid
        else waveVector.reshape(-1, 3).transpose(0, 1)
    )
    gridSize = waveVectorFlat.shape[-1]

    weightsConj = weights.conj()
    chunkSize = resolveResponseChunkSize(
        batchSize=batchSize,
        elementCount=elementCount,
        gridSize=gridSize,
        realDtype=realDtype,
        device=device,
        requestedChunkSize=chunkSize,
    )

    fullResponse = torch.empty(
        (batchSize, *spatialShape),
        device=device,
        dtype=realDtype,
    )
    fullResponseFlat = fullResponse.reshape(batchSize, -1)

    bStart = 0
    while bStart < batchSize:
        batchRemaining = batchSize - bStart
        bc, _, pcMax = chooseChunkShape(
            batchRemaining=batchRemaining,
            elementCount=elementCount,
            gridRemaining=gridSize,
            chunkSize=chunkSize,
        )
        bEnd = min(batchSize, bStart + bc)
        bc = bEnd - bStart

        posB = elementLocalPosition[bStart:bEnd]  # [Bc, 3, N]
        wB = weightsConj[bStart:bEnd]  # [Bc, N]
        waveB = waveVectorFlat[bStart:bEnd] if isBatchedGrid else waveVectorFlat
        fullResponseFlatB = fullResponseFlat[bStart:bEnd]

        # Reuse one complex scratch tile and write real power directly into the output.
        responseChunkBuffer = torch.empty(
            (bc, pcMax),
            device=device,
            dtype=complexDtype,
        )

        pStart = 0
        while pStart < gridSize:
            gridRemaining = gridSize - pStart
            _, nc, pc = chooseChunkShape(
                batchRemaining=bc,
                elementCount=elementCount,
                gridRemaining=gridRemaining,
                chunkSize=chunkSize,
            )
            pEnd = min(gridSize, pStart + pc)
            pc = pEnd - pStart

            waveChunk = waveB[:, :, pStart:pEnd] if isBatchedGrid else waveB[:, pStart:pEnd]
            responseChunk = responseChunkBuffer[:, :pc]
            responseChunk.zero_()

            nStart = 0
            while nStart < elementCount:
                nEnd = min(elementCount, nStart + nc)

                posChunk = posB[:, :, nStart:nEnd]  # [Bc, 3, Nc]
                wChunk = wB[:, nStart:nEnd]  # [Bc, Nc]

                if isBatchedGrid:
                    phaseChunk = torch.einsum("bin,bip->bnp", posChunk, waveChunk)
                else:
                    phaseChunk = torch.einsum("bin,ip->bnp", posChunk, waveChunk)
                manifoldChunk = torch.exp(1j * phaseChunk)
                responseChunk += torch.einsum("bn,bnp->bp", wChunk, manifoldChunk)

                del posChunk, wChunk, phaseChunk, manifoldChunk
                nStart = nEnd

            fullResponseFlatB[:, pStart:pEnd] = responseChunk.abs().square()
            del waveChunk, responseChunk
            pStart = pEnd

        del posB, wB, waveB, fullResponseFlatB, responseChunkBuffer
        bStart = bEnd

    del waveVector, waveVectorFlat, weightsConj, fullResponseFlat

    if normalize:
        fullResponse = normalizePower(fullResponse)

    if dB:
        gainView = gain.view(-1, *([1] * (fullResponse.ndim - 1)))
        fullResponse = todB(fullResponse) + gainView
        del gainView

    if clearCache and device.type == "cuda":
        torch.cuda.empty_cache()

    return fullResponse


def arrayResponseSample(
    batch: ArrayBatch,
    sampleID: int,
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    chunkSize: int | None = 3_000_000,
    dB: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    azimuth, elevation = torch.broadcast_tensors(azimuth, elevation)
    azimuth = azimuth.unsqueeze(0)  # [1, ...]
    elevation = elevation.unsqueeze(0)  # [1, ...]

    return arrayResponseCore(
        elementLocalPosition=batch.elementLocalPosition[sampleID : sampleID + 1],
        weights=batch.weights[sampleID : sampleID + 1],
        wavelength=batch.wavelength,
        azimuth=azimuth,
        elevation=elevation,
        gain=batch.gain[sampleID : sampleID + 1],
        chunkSize=chunkSize,
        dB=dB,
        normalize=normalize,
    )[0]


def arrayResponseBatch(
    batch: ArrayBatch,
    relativeTargetAZEL: tuple[torch.Tensor, torch.Tensor],
    chunkSize: int | None = None,
    dB: bool = False,
    normalize: bool = False,
) -> torch.Tensor:
    return arrayResponseCore(
        elementLocalPosition=batch.elementLocalPosition,
        weights=batch.weights,
        wavelength=batch.wavelength,
        azimuth=relativeTargetAZEL[0],
        elevation=relativeTargetAZEL[1],
        gain=batch.gain,
        chunkSize=chunkSize,
        dB=dB,
        normalize=normalize,
    )
