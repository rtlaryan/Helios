import torch

from .arrayBatch import ArrayBatch

DEFAULT_REDUCTION_TILE_CAP = 256
MAX_CUDA_BATCH_STREAMS = 4
DEFAULT_TORCH_COMPILE_BACKEND = "inductor"

_COMPILED_ARRAY_RESPONSE_CORE = None
_COMPILED_ARRAY_RESPONSE_CORE_CONFIG: tuple[str, str | None, bool] | None = None


def _validateReductionTileCap(reductionTileCap: int) -> int:
    reductionTileCap = int(reductionTileCap)
    if reductionTileCap <= 0:
        raise ValueError("reductionTileCap must be positive")
    return reductionTileCap


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
    reductionTileCap: int = DEFAULT_REDUCTION_TILE_CAP,
) -> tuple[int, int, int]:
    batchRemaining = int(batchRemaining)
    elementCount = int(elementCount)
    gridRemaining = int(gridRemaining)
    chunkSize = int(chunkSize)
    if batchRemaining <= 0:
        raise ValueError("batchRemaining must be positive")
    if elementCount <= 0:
        raise ValueError("elementCount must be positive")
    if gridRemaining <= 0:
        raise ValueError("gridRemaining must be positive")
    if chunkSize <= 0:
        raise ValueError("chunkSize must be positive")
    reductionTileCap = _validateReductionTileCap(reductionTileCap)

    nc = min(elementCount, reductionTileCap, chunkSize)
    if batchRemaining * nc <= chunkSize:
        bc = batchRemaining
        pc = min(gridRemaining, max(1, chunkSize // (bc * nc)))
        return bc, nc, pc

    bc = min(batchRemaining, max(1, chunkSize // nc))
    return bc, nc, 1


def _accumulateResponseChunk(
    posB: torch.Tensor,
    wB: torch.Tensor,
    waveChunk: torch.Tensor,
    *,
    elementCount: int,
    nc: int,
    isBatchedGrid: bool,
    responseChunk: torch.Tensor,
) -> None:
    responseChunk.zero_()

    nStart = 0
    while nStart < elementCount:
        nEnd = min(elementCount, nStart + nc)

        posChunk = posB[:, :, nStart:nEnd]
        wChunk = wB[:, nStart:nEnd]

        if isBatchedGrid:
            phaseChunk = torch.einsum("bin,bip->bnp", posChunk, waveChunk)
        else:
            phaseChunk = torch.einsum("bin,ip->bnp", posChunk, waveChunk)
        manifoldChunk = torch.exp(1j * phaseChunk)
        responseChunk += torch.einsum("bn,bnp->bp", wChunk, manifoldChunk)

        nStart = nEnd


def _writeResponseTile(
    *,
    posB: torch.Tensor,
    wB: torch.Tensor,
    waveB: torch.Tensor,
    fullResponseFlatB: torch.Tensor,
    responseChunkBuffer: torch.Tensor,
    pStart: int,
    pEnd: int,
    elementCount: int,
    nc: int,
    isBatchedGrid: bool,
) -> None:
    waveChunk = waveB[:, :, pStart:pEnd] if isBatchedGrid else waveB[:, pStart:pEnd]
    responseChunk = responseChunkBuffer[:, : pEnd - pStart]
    _accumulateResponseChunk(
        posB,
        wB,
        waveChunk,
        elementCount=elementCount,
        nc=nc,
        isBatchedGrid=isBatchedGrid,
        responseChunk=responseChunk,
    )
    fullResponseFlatB[:, pStart:pEnd] = responseChunk.abs().square()


def _buildBatchStripes(
    *,
    bStart: int,
    batchSize: int,
    batchStride: int,
    waveVectorFlat: torch.Tensor,
    elementLocalPosition: torch.Tensor,
    weightsConj: torch.Tensor,
    fullResponseFlat: torch.Tensor,
    isBatchedGrid: bool,
    device: torch.device,
    complexDtype: torch.dtype,
    pcMax: int,
) -> list[dict[str, torch.Tensor | int]]:
    batchStripes: list[dict[str, torch.Tensor | int]] = []
    stripeStart = bStart
    while stripeStart < batchSize and len(batchStripes) < MAX_CUDA_BATCH_STREAMS:
        stripeEnd = min(batchSize, stripeStart + batchStride)
        stripeBc = stripeEnd - stripeStart
        batchStripes.append(
            {
                "start": stripeStart,
                "end": stripeEnd,
                "batchSize": stripeBc,
                "pos": elementLocalPosition[stripeStart:stripeEnd],
                "weights": weightsConj[stripeStart:stripeEnd],
                "wave": waveVectorFlat[stripeStart:stripeEnd] if isBatchedGrid else waveVectorFlat,
                "output": fullResponseFlat[stripeStart:stripeEnd],
                "buffer": torch.empty(
                    (stripeBc, pcMax),
                    device=device,
                    dtype=complexDtype,
                ),
            }
        )
        stripeStart = stripeEnd
    return batchStripes


def _arrayResponseCoreReference(
    elementLocalPosition: torch.Tensor,
    weights: torch.Tensor,
    wavelength: float,
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    gain: torch.Tensor,
    chunkSize: int | None = 2_000_000,
    reductionTileCap: int = DEFAULT_REDUCTION_TILE_CAP,
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
    reductionTileCap = _validateReductionTileCap(reductionTileCap)

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
            reductionTileCap=reductionTileCap,
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
                reductionTileCap=reductionTileCap,
            )
            pEnd = min(gridSize, pStart + pc)
            pc = pEnd - pStart

            responseChunk = torch.zeros(
                (bc, pc),
                device=device,
                dtype=complexDtype,
            )

            waveChunk = waveB[:, :, pStart:pEnd] if isBatchedGrid else waveB[:, pStart:pEnd]
            _accumulateResponseChunk(
                posB,
                wB,
                waveChunk,
                elementCount=elementCount,
                nc=nc,
                isBatchedGrid=isBatchedGrid,
                responseChunk=responseChunk,
            )

            responseFlatB[:, pStart:pEnd] = responseChunk
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


def _arrayResponseCoreEager(
    elementLocalPosition: torch.Tensor,
    weights: torch.Tensor,
    wavelength: float,
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    gain: torch.Tensor,
    chunkSize: int | None = 2_000_000,
    reductionTileCap: int = DEFAULT_REDUCTION_TILE_CAP,
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
    reductionTileCap = _validateReductionTileCap(reductionTileCap)

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
            reductionTileCap=reductionTileCap,
        )
        batchStripes = _buildBatchStripes(
            bStart=bStart,
            batchSize=batchSize,
            batchStride=bc,
            waveVectorFlat=waveVectorFlat,
            elementLocalPosition=elementLocalPosition,
            weightsConj=weightsConj,
            fullResponseFlat=fullResponseFlat,
            isBatchedGrid=isBatchedGrid,
            device=device,
            complexDtype=complexDtype,
            pcMax=pcMax,
        )
        bc = int(batchStripes[0]["batchSize"])
        streams = []
        if device.type == "cuda" and len(batchStripes) > 1:
            streams = [torch.cuda.Stream(device=device) for _ in batchStripes]

        pStart = 0
        while pStart < gridSize:
            gridRemaining = gridSize - pStart
            _, nc, pc = chooseChunkShape(
                batchRemaining=bc,
                elementCount=elementCount,
                gridRemaining=gridRemaining,
                chunkSize=chunkSize,
                reductionTileCap=reductionTileCap,
            )
            pEnd = min(gridSize, pStart + pc)
            pc = pEnd - pStart

            for stripeIndex, stripe in enumerate(batchStripes):
                stream = streams[stripeIndex] if stripeIndex < len(streams) else None
                if stream is None:
                    _writeResponseTile(
                        posB=stripe["pos"],
                        wB=stripe["weights"],
                        waveB=stripe["wave"],
                        fullResponseFlatB=stripe["output"],
                        responseChunkBuffer=stripe["buffer"],
                        pStart=pStart,
                        pEnd=pEnd,
                        elementCount=elementCount,
                        nc=nc,
                        isBatchedGrid=isBatchedGrid,
                    )
                    continue

                with torch.cuda.stream(stream):
                    _writeResponseTile(
                        posB=stripe["pos"],
                        wB=stripe["weights"],
                        waveB=stripe["wave"],
                        fullResponseFlatB=stripe["output"],
                        responseChunkBuffer=stripe["buffer"],
                        pStart=pStart,
                        pEnd=pEnd,
                        elementCount=elementCount,
                        nc=nc,
                        isBatchedGrid=isBatchedGrid,
                    )
            if streams:
                currentStream = torch.cuda.current_stream(device=device)
                for stream in streams:
                    currentStream.wait_stream(stream)
            pStart = pEnd

        bStart = int(batchStripes[-1]["end"])

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


def isTorchCompileAvailable() -> bool:
    return callable(getattr(torch, "compile", None))


def resetArrayResponseCoreCompiledCache() -> None:
    global _COMPILED_ARRAY_RESPONSE_CORE, _COMPILED_ARRAY_RESPONSE_CORE_CONFIG
    _COMPILED_ARRAY_RESPONSE_CORE = None
    _COMPILED_ARRAY_RESPONSE_CORE_CONFIG = None


def _getCompiledArrayResponseCore(
    *,
    backend: str = DEFAULT_TORCH_COMPILE_BACKEND,
    mode: str | None = None,
    dynamic: bool = False,
):
    global _COMPILED_ARRAY_RESPONSE_CORE, _COMPILED_ARRAY_RESPONSE_CORE_CONFIG
    if not isTorchCompileAvailable():
        raise RuntimeError("torch.compile is not available in this PyTorch build")

    config = (backend, mode, dynamic)
    if _COMPILED_ARRAY_RESPONSE_CORE is None or _COMPILED_ARRAY_RESPONSE_CORE_CONFIG != config:
        _COMPILED_ARRAY_RESPONSE_CORE = torch.compile(
            _arrayResponseCoreEager,
            backend=backend,
            mode=mode,
            dynamic=dynamic,
        )
        _COMPILED_ARRAY_RESPONSE_CORE_CONFIG = config
    return _COMPILED_ARRAY_RESPONSE_CORE


def arrayResponseCore(
    elementLocalPosition: torch.Tensor,
    weights: torch.Tensor,
    wavelength: float,
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    gain: torch.Tensor,
    chunkSize: int | None = 2_000_000,
    reductionTileCap: int = DEFAULT_REDUCTION_TILE_CAP,
    dB: bool = True,
    normalize: bool = True,
    clearCache: bool = False,
) -> torch.Tensor:
    return _arrayResponseCoreEager(
        elementLocalPosition=elementLocalPosition,
        weights=weights,
        wavelength=wavelength,
        azimuth=azimuth,
        elevation=elevation,
        gain=gain,
        chunkSize=chunkSize,
        reductionTileCap=reductionTileCap,
        dB=dB,
        normalize=normalize,
        clearCache=clearCache,
    )


def arrayResponseCoreCompiled(
    elementLocalPosition: torch.Tensor,
    weights: torch.Tensor,
    wavelength: float,
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    gain: torch.Tensor,
    chunkSize: int | None = 2_000_000,
    reductionTileCap: int = DEFAULT_REDUCTION_TILE_CAP,
    dB: bool = True,
    normalize: bool = True,
    clearCache: bool = False,
    *,
    backend: str = DEFAULT_TORCH_COMPILE_BACKEND,
    mode: str | None = None,
    dynamic: bool = False,
) -> torch.Tensor:
    compiledCore = _getCompiledArrayResponseCore(
        backend=backend,
        mode=mode,
        dynamic=dynamic,
    )
    return compiledCore(
        elementLocalPosition=elementLocalPosition,
        weights=weights,
        wavelength=wavelength,
        azimuth=azimuth,
        elevation=elevation,
        gain=gain,
        chunkSize=chunkSize,
        reductionTileCap=reductionTileCap,
        dB=dB,
        normalize=normalize,
        clearCache=clearCache,
    )


def arrayResponseSample(
    batch: ArrayBatch,
    sampleID: int,
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    chunkSize: int | None = 3_000_000,
    reductionTileCap: int = DEFAULT_REDUCTION_TILE_CAP,
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
        reductionTileCap=reductionTileCap,
        dB=dB,
        normalize=normalize,
    )[0]


def arrayResponseSampleCompiled(
    batch: ArrayBatch,
    sampleID: int,
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    chunkSize: int | None = 3_000_000,
    reductionTileCap: int = DEFAULT_REDUCTION_TILE_CAP,
    dB: bool = True,
    normalize: bool = True,
    *,
    backend: str = DEFAULT_TORCH_COMPILE_BACKEND,
    mode: str | None = None,
    dynamic: bool = False,
) -> torch.Tensor:
    azimuth, elevation = torch.broadcast_tensors(azimuth, elevation)
    azimuth = azimuth.unsqueeze(0)
    elevation = elevation.unsqueeze(0)

    return arrayResponseCoreCompiled(
        elementLocalPosition=batch.elementLocalPosition[sampleID : sampleID + 1],
        weights=batch.weights[sampleID : sampleID + 1],
        wavelength=batch.wavelength,
        azimuth=azimuth,
        elevation=elevation,
        gain=batch.gain[sampleID : sampleID + 1],
        chunkSize=chunkSize,
        reductionTileCap=reductionTileCap,
        dB=dB,
        normalize=normalize,
        backend=backend,
        mode=mode,
        dynamic=dynamic,
    )[0]


def arrayResponseBatch(
    batch: ArrayBatch,
    relativeTargetAZEL: tuple[torch.Tensor, torch.Tensor],
    chunkSize: int | None = None,
    reductionTileCap: int = DEFAULT_REDUCTION_TILE_CAP,
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
        reductionTileCap=reductionTileCap,
        dB=dB,
        normalize=normalize,
    )


def arrayResponseBatchCompiled(
    batch: ArrayBatch,
    relativeTargetAZEL: tuple[torch.Tensor, torch.Tensor],
    chunkSize: int | None = None,
    reductionTileCap: int = DEFAULT_REDUCTION_TILE_CAP,
    dB: bool = False,
    normalize: bool = False,
    *,
    backend: str = DEFAULT_TORCH_COMPILE_BACKEND,
    mode: str | None = None,
    dynamic: bool = False,
) -> torch.Tensor:
    return arrayResponseCoreCompiled(
        elementLocalPosition=batch.elementLocalPosition,
        weights=batch.weights,
        wavelength=batch.wavelength,
        azimuth=relativeTargetAZEL[0],
        elevation=relativeTargetAZEL[1],
        gain=batch.gain,
        chunkSize=chunkSize,
        reductionTileCap=reductionTileCap,
        dB=dB,
        normalize=normalize,
        backend=backend,
        mode=mode,
        dynamic=dynamic,
    )
