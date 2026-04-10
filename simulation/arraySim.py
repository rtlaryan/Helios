import psutil
import torch
from scripts.arrayBatch import ArrayBatch


def precomputeGridWaveVector(
    wavelength: float, azimuth: torch.Tensor, elevation: torch.Tensor
) -> tuple[torch.Tensor, torch.Size]:
    azimuth, elevation = torch.broadcast_tensors(azimuth, elevation)
    gridShape = azimuth.shape

    waveVector = (2 * torch.pi / wavelength) * torch.stack(
        [
            torch.cos(elevation) * torch.cos(azimuth),
            torch.cos(elevation) * torch.sin(azimuth),
            torch.sin(elevation),
        ],
        dim=1,
    ).flatten(start_dim=2)  # [B, 3, P]
    return waveVector, gridShape


def precomputeSharedGridWaveVector(
    wavelength: float, azimuth: torch.Tensor, elevation: torch.Tensor
) -> tuple[torch.Tensor, torch.Size]:
    azimuth, elevation = torch.broadcast_tensors(azimuth, elevation)
    gridShape = azimuth.shape

    waveVector = (2 * torch.pi / wavelength) * torch.stack(
        [
            torch.cos(elevation) * torch.cos(azimuth),
            torch.cos(elevation) * torch.sin(azimuth),
            torch.sin(elevation),
        ],
        dim=0,
    ).flatten(start_dim=1)  # [3, P]
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


def resolveChunkSize(
    batchSize: int,
    elementCount: int,
    gridSize: int,
    real_dtype: torch.dtype,
    device: torch.device,
    requestedChunkSize: int | None = None,
    safetyFactor: float = 1.25,
    memSafe: bool = True,
) -> int:
    fullProduct = max(1, batchSize * elementCount * gridSize)
    if requestedChunkSize is not None:
        if not memSafe:
            return int(min(fullProduct, requestedChunkSize))

        else:
            if device.type == "cuda":
                freeMem, _ = torch.cuda.mem_get_info(device)
            elif device.type == "mps":
                maxMem = torch.mps.recommended_max_memory()
                allocatedMem = torch.mps.driver_allocated_memory()
                freeMem = maxMem - allocatedMem
            else:
                freeMem = psutil.virtual_memory().available

            bytesPerReal = torch.finfo(real_dtype).bits // 8
            chunkBudget = max(1, freeMem // (bytesPerReal * 8))

            return int(min(chunkBudget, fullProduct, requestedChunkSize))

    else:
        if device.type == "cuda":
            freeMem, _ = torch.cuda.mem_get_info(device)
        elif device.type == "mps":
            maxMem = torch.mps.recommended_max_memory()
            allocatedMem = torch.mps.driver_allocated_memory()
            freeMem = maxMem - allocatedMem
        else:
            freeMem = psutil.virtual_memory().available

        bytesPerReal = torch.finfo(real_dtype).bits // 8
        chunkBudget = max(1, freeMem // (bytesPerReal * 8 * safetyFactor))

        return int(min(chunkBudget, fullProduct))


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
    baisN: bool = True,
) -> tuple[int, int, int]:
    """
    Choose (Bc, Nc, Pc) such that Bc * Nc * Pc <= chunkSize.
    """
    if baisN:
        nc = min(elementCount, chunkSize)
        if batchRemaining * nc <= chunkSize:
            bc = batchRemaining
            pc = min(gridRemaining, max(1, chunkSize // (bc * nc)))
            return bc, nc, pc

        bc = min(batchRemaining, max(1, chunkSize // nc))
        return bc, nc, 1

    pc = min(gridRemaining, chunkSize)
    if batchRemaining * pc <= chunkSize:
        bc = batchRemaining
        nc = min(elementCount, max(1, chunkSize // (bc * pc)))
        return bc, nc, pc

    bc = min(batchRemaining, max(1, chunkSize // pc))
    return bc, 1, pc


@torch.no_grad()
def arrayResponseCore(
    elementLocalPosition: torch.Tensor,
    weights: torch.Tensor,
    wavelength: float,
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    gain: torch.Tensor,
    chunkSize: int | None = 2_000_000,
    dB: bool = False,
    normalize: bool = False,
    memSafe: bool = True,
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

    waveVector, gridShape = precomputeGridWaveVector(wavelength, azimuth, elevation)
    spatialShape = gridShape[1:]
    gridSize = waveVector.shape[-1]
    elementLocalPositionT = elementLocalPosition.transpose(1, 2)  # [B, N, 3]

    chunkSize = resolveChunkSize(
        batchSize, elementCount, gridSize, real_dtype, device, chunkSize, memSafe
    )

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

        posB = elementLocalPositionT[bStart:bEnd]  # [Bc, N, 3]
        wRealB = weights[bStart:bEnd].real  # [Bc, N]
        wImagB = weights[bStart:bEnd].imag  # [Bc, N]
        waveB = waveVector[bStart:bEnd]  # [Bc, 3, P]

        responseFlatB = torch.empty(
            (bc, gridSize),
            device=device,
            dtype=real_dtype,
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
            responseReal = torch.zeros(
                (bc, pc),
                device=device,
                dtype=real_dtype,
            )
            responseImag = torch.zeros(
                (bc, pc),
                device=device,
                dtype=real_dtype,
            )

            nStart = 0
            while nStart < elementCount:
                nEnd = min(elementCount, nStart + nc)

                posChunk = posB[:, nStart:nEnd, :]  # [Bc, Nc, 3]
                wRealChunk = wRealB[:, nStart:nEnd]  # [Bc, Nc]
                wImagChunk = wImagB[:, nStart:nEnd]  # [Bc, Nc]

                phaseChunk = torch.bmm(posChunk, waveChunk)  # [Bc, Nc, Pc]
                cosPhase = torch.cos(phaseChunk)
                sinPhase = torch.sin(phaseChunk)

                # conj(w) * exp(j * phase) with w = wr + j wi:
                # real = wr * cos + wi * sin
                # imag = wr * sin - wi * cos
                responseReal += torch.bmm(wRealChunk.unsqueeze(1), cosPhase).squeeze(1)
                responseReal += torch.bmm(wImagChunk.unsqueeze(1), sinPhase).squeeze(1)
                responseImag += torch.bmm(wRealChunk.unsqueeze(1), sinPhase).squeeze(1)
                responseImag -= torch.bmm(wImagChunk.unsqueeze(1), cosPhase).squeeze(1)

                del posChunk, wRealChunk, wImagChunk, phaseChunk, cosPhase, sinPhase
                nStart = nEnd

            responseFlatB[:, pStart:pEnd] = responseReal.square() + responseImag.square()
            del waveChunk, responseReal, responseImag
            pStart = pEnd

        fullResponse[bStart:bEnd] = responseFlatB.reshape(bc, *spatialShape)
        del posB, wRealB, wImagB, waveB, responseFlatB
        bStart = bEnd

    del waveVector, elementLocalPositionT

    if normalize:
        fullResponse = normalizePower(fullResponse)

    if dB:
        gainView = gain.view(-1, *([1] * (fullResponse.ndim - 1)))
        fullResponse = todB(fullResponse) + gainView
        del gainView

    return fullResponse


@torch.no_grad()
def arrayResponseCoreSharedGrid(
    elementLocalPosition: torch.Tensor,
    weights: torch.Tensor,
    wavelength: float,
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    gain: torch.Tensor,
    chunkSize: int | None = 2_000_000,
    dB: bool = False,
    normalize: bool = False,
    memSafe: bool = True,
) -> torch.Tensor:
    """
    elementLocalPosition: [B, 3, N]
    weights: [B, N]
    gain: [B]
    azimuth/elevation: [...]

    chunkSize controls the dominant temporary:
      [Bc, Nc, Pc]
    """
    batchSize = elementLocalPosition.shape[0]
    elementCount = elementLocalPosition.shape[-1]
    device = weights.device
    real_dtype = weights.real.dtype

    azimuth, elevation = torch.broadcast_tensors(azimuth, elevation)
    if azimuth.ndim > 1 and azimuth.shape[0] == batchSize:
        raise ValueError(
            "arrayResponseCoreSharedGrid expects a shared grid [...], not a batched grid [B, ...]; "
            "use arrayResponseCore for batched grids."
        )

    waveVector, spatialShape = precomputeSharedGridWaveVector(wavelength, azimuth, elevation)
    gridSize = waveVector.shape[-1]
    elementLocalPositionT = elementLocalPosition.transpose(1, 2)  # [B, N, 3]

    chunkSize = resolveChunkSize(
        batchSize, elementCount, gridSize, real_dtype, device, chunkSize, memSafe
    )

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

        posB = elementLocalPositionT[bStart:bEnd]  # [Bc, N, 3]
        wRealB = weights[bStart:bEnd].real  # [Bc, N]
        wImagB = weights[bStart:bEnd].imag  # [Bc, N]

        responseFlatB = torch.empty(
            (bc, gridSize),
            device=device,
            dtype=real_dtype,
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

            waveChunk = waveVector[:, pStart:pEnd]  # [3, Pc]
            responseReal = torch.zeros(
                (bc, pc),
                device=device,
                dtype=real_dtype,
            )
            responseImag = torch.zeros(
                (bc, pc),
                device=device,
                dtype=real_dtype,
            )

            nStart = 0
            while nStart < elementCount:
                nEnd = min(elementCount, nStart + nc)

                posChunk = posB[:, nStart:nEnd, :]  # [Bc, Nc, 3]
                wRealChunk = wRealB[:, nStart:nEnd]  # [Bc, Nc]
                wImagChunk = wImagB[:, nStart:nEnd]  # [Bc, Nc]

                phaseChunk = torch.matmul(posChunk, waveChunk)  # [Bc, Nc, Pc]
                cosPhase = torch.cos(phaseChunk)
                sinPhase = torch.sin(phaseChunk)

                responseReal += torch.bmm(wRealChunk.unsqueeze(1), cosPhase).squeeze(1)
                responseReal += torch.bmm(wImagChunk.unsqueeze(1), sinPhase).squeeze(1)
                responseImag += torch.bmm(wRealChunk.unsqueeze(1), sinPhase).squeeze(1)
                responseImag -= torch.bmm(wImagChunk.unsqueeze(1), cosPhase).squeeze(1)

                del posChunk, wRealChunk, wImagChunk, phaseChunk, cosPhase, sinPhase
                nStart = nEnd

            responseFlatB[:, pStart:pEnd] = responseReal.square() + responseImag.square()
            del waveChunk, responseReal, responseImag
            pStart = pEnd

        fullResponse[bStart:bEnd] = responseFlatB.reshape(bc, *spatialShape)
        del posB, wRealB, wImagB, responseFlatB
        bStart = bEnd

    del waveVector, elementLocalPositionT

    if normalize:
        fullResponse = normalizePower(fullResponse)

    if dB:
        gainView = gain.view(-1, *([1] * (fullResponse.ndim - 1)))
        fullResponse = todB(fullResponse) + gainView
        del gainView

    return fullResponse


def arrayResponseSample(
    batch: ArrayBatch,
    sampleID: int,
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    chunkSize: int | None = 2_000_000,
    dB: bool = False,
    normalize: bool = False,
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


def arrayResponseBatchSharedGrid(
    batch: ArrayBatch,
    relativeTargetAZEL: tuple[torch.Tensor, torch.Tensor],
    chunkSize: int | None = 2_000_000_000,
    dB: bool = False,
    normalize: bool = False,
) -> torch.Tensor:
    azimuth, elevation = torch.broadcast_tensors(relativeTargetAZEL[0], relativeTargetAZEL[1])

    return arrayResponseCoreSharedGrid(
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
