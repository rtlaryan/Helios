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


def normalizePower(power: torch.Tensor, threshold: float = 1e-12) -> torch.Tensor:
    powerMax = power.flatten(1).amax(dim=1).clamp_min(threshold)
    return power / powerMax.view(-1, *([1] * (power.ndim - 1)))


def todB(power: torch.Tensor, threshold: float = 1e-12) -> torch.Tensor:
    responsedB = 10.0 * torch.log10(power.clamp_min(threshold))
    return responsedB


def arrayResponseCore(
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
        fullResponse = 10.0 * torch.log10(fullResponse.clamp_min(1e-12)) + gainView

    return fullResponse


def arrayResponseSample(
    batch: ArrayBatch,
    sampleID: int,
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    chunkSize: int | None = 8192,
    dB: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    wavelength = batch.wavelength
    elementLocalPosition = batch.elementLocalPosition[sampleID : sampleID + 1]
    weights = batch.weights[sampleID : sampleID + 1]
    gain = batch.gain[sampleID : sampleID + 1]
    return arrayResponseCore(
        elementLocalPosition,
        weights,
        wavelength,
        azimuth,
        elevation,
        gain,
        chunkSize,
        dB,
        normalize,
    )[0]


def arrayResponseBatch(
    batch: ArrayBatch,
    relativeTargetAZEL: tuple[torch.Tensor, torch.Tensor],
    chunkSize: int | None = 8192,
    dB: bool = False,
    normalize: bool = False,
) -> torch.Tensor:
    azimuth, elevation = relativeTargetAZEL
    return arrayResponseCore(
        batch.elementLocalPosition,
        batch.weights,
        batch.wavelength,
        azimuth,
        elevation,
        batch.gain,
        chunkSize,
        dB,
        normalize,
    )
