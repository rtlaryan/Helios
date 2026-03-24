import torch

from .arrayBatch import ArrayBatch


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


def chooseChunkShape(
    batchRemaining: int,
    elementCount: int,
    gridRemaining: int,
    chunkSize: int,
) -> tuple[int, int, int]:
    """
    Choose (Bc, Nc, Pc) such that Bc * Nc * Pc <= chunkSize.
    """
    if chunkSize <= 0:
        raise ValueError("chunkSize must be positive")

    bc = min(batchRemaining, max(1, int(round(chunkSize ** (1 / 3)))))
    pc = min(gridRemaining, max(1, int(round((chunkSize / bc) ** 0.5))))
    nc = min(elementCount, max(1, chunkSize // (bc * pc)))
    pc = min(gridRemaining, max(1, chunkSize // (bc * nc)))

    return bc, nc, pc


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
    azimuth/elevation: [B, ...]

    chunkSize controls the dominant temporary:
      [Bc, Nc, Pc]
    """
    batchSize = elementLocalPosition.shape[0]
    elementCount = elementLocalPosition.shape[-1]
    device = weights.device
    real_dtype = weights.real.dtype
    complex_dtype = weights.dtype

    waveVector, gridShape = precomputeGridWaveVector(wavelength, azimuth, elevation)
    spatialShape = gridShape[1:]
    waveVectorFlat = waveVector.reshape(batchSize, -1, 3).transpose(1, 2)  # [B, 3, P]
    gridSize = waveVectorFlat.shape[-1]

    weightsConj = weights.conj()

    if chunkSize is None:
        chunkSize = max(1, batchSize * elementCount * min(gridSize, 1024))

    fullResponse = torch.empty(
        (batchSize, *spatialShape),
        device=device,
        dtype=real_dtype,
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
        waveB = waveVectorFlat[bStart:bEnd]  # [Bc, 3, P]

        responseFlatB = torch.empty(
            (bc, gridSize),
            device=device,
            dtype=complex_dtype,
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

            waveChunk = waveB[:, :, pStart:pEnd]  # [Bc, 3, Pc]
            responseChunk = torch.zeros(
                (bc, pc),
                device=device,
                dtype=complex_dtype,
            )

            nStart = 0
            while nStart < elementCount:
                nEnd = min(elementCount, nStart + nc)

                posChunk = posB[:, :, nStart:nEnd]  # [Bc, 3, Nc]
                wChunk = wB[:, nStart:nEnd]  # [Bc, Nc]

                phaseChunk = torch.einsum("bin,bip->bnp", posChunk, waveChunk)
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


def arrayResponseSample(
    batch: ArrayBatch,
    sampleID: int,
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    chunkSize: int | None = 2_000_000,
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
    chunkSize: int | None = 2_000_000_000,
    dB: bool = False,
    normalize: bool = False,
) -> torch.Tensor:
    azimuth, elevation = ensureBatchedGrid(
        azimuth=relativeTargetAZEL[0],
        elevation=relativeTargetAZEL[1],
        batchSize=batch.batchSize,
    )

    return arrayResponseCore(
        elementLocalPosition=batch.elementLocalPosition,
        weights=batch.weights,
        wavelength=batch.wavelength,
        azimuth=azimuth,
        elevation=elevation,
        gain=batch.gain,
        chunkSize=chunkSize,
        dB=dB,
        normalize=normalize,
    )
