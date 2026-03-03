import torch

def precomputeGridWaveVector(wavelength: float, azimuth: torch.Tensor, elevation: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    azimuth, elevation = torch.broadcast_tensors(azimuth, elevation)
    gridShape = azimuth.shape
    azimuthFlat, elevationFlat = azimuth.reshape(-1), elevation.reshape(-1)

    propagationVectors = torch.stack([
        torch.cos(elevationFlat) * torch.cos(azimuthFlat),
        torch.cos(elevationFlat) * torch.sin(azimuthFlat),
        torch.sin(elevationFlat)
    ], dim = 0)

    waveVector = (2 * torch.pi / wavelength) * propagationVectors
    return waveVector, gridShape


def arrayResponse(elementPosition: torch.Tensor, weights: torch.Tensor, wavelength: float, azimuth: torch.Tensor, elevation: torch.Tensor, chunkSize: int = 8192) -> torch.Tensor:
    batchSize = elementPosition.shape[0]
    gridWaveVector, gridShape = precomputeGridWaveVector(wavelength, azimuth, elevation)
    gridSize = gridWaveVector.shape[1]
    fullResponse = []

    for chunk in range(0, gridSize, chunkSize):
        waveVector = gridWaveVector[:, chunk:chunk+chunkSize]
        phaseChunk = torch.einsum("b3n, 3p->bnp", elementPosition, waveVector)
        arrayManifoldChunk = torch.exp(1j * phaseChunk)
        chunkReponse = torch.einsum("bn, bnp -> bp", weights.conj(), arrayManifoldChunk)
        fullResponse.append(chunkReponse)

    fullResponse = torch.cat(fullResponse, dim=1)
    return fullResponse.reshape(batchSize, *gridShape)
