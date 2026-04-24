# scripts/batchFactory.py
import math

import torch

from .arrayBatch import ArrayBatch, _normalize_amplitudes, infer_geometry_cache_key
from .arraySpec import ArraySpec
from .coordinateTransforms import LLAtoECEF, getECEFtoENUMapping


def sampleElementPositions(
    spec: ArraySpec,
    batchSize: int,
    elementCount: int,
    aspectRatio: float,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    yElements, zElements = (
        int(math.ceil(math.sqrt(elementCount * aspectRatio))),
        int(math.ceil(math.sqrt(elementCount / aspectRatio))),
    )
    ySpan, zSpan = (yElements - 1) * spec.elementSpacing, (zElements - 1) * spec.elementSpacing
    yPositions = torch.linspace(
        (-0.5 * ySpan), (0.5 * ySpan), yElements, device=device, dtype=dtype
    )
    zPositions = torch.linspace(
        (-0.5 * zSpan), (0.5 * zSpan), zElements, device=device, dtype=dtype
    )
    yGrid, zGrid = torch.meshgrid(yPositions, zPositions, indexing="ij")
    yGrid, zGrid = yGrid.flatten(), zGrid.flatten()

    if spec.geometry == "UCA":
        yStep = (yPositions[1] - yPositions[0]).abs() if yElements > 1 else yPositions.new_zeros(())
        zStep = (zPositions[1] - zPositions[0]).abs() if zElements > 1 else zPositions.new_zeros(())
        radius = 0.5 * min(ySpan, zSpan) + 0.5 * min(yStep, zStep)

        keep = (yGrid.square() + zGrid.square()) <= (radius**2)
        yGrid, zGrid = yGrid[keep], zGrid[keep]

    xGrid = torch.zeros_like(yGrid)

    localPosition = torch.stack([xGrid, yGrid, zGrid], dim=0)
    localPositionBatch = localPosition.unsqueeze(0).expand(batchSize, -1, -1).clone()

    if spec.positionJitterSTD > 0:
        localPositionBatch = localPositionBatch + spec.positionJitterSTD * torch.randn_like(
            localPositionBatch,
            generator=generator,
        )

    return localPositionBatch


def sampleElementMask(
    spec: ArraySpec,
    batchSize: int,
    elementCount,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor | None:
    if spec.failRate == 0.0 and spec.sparsityRange == (1.0, 1.0):
        return None

    mask = torch.ones((batchSize, elementCount), device=device, dtype=torch.bool)

    # Array Failures
    if spec.failRate > 0:
        failMask = torch.rand(
            (batchSize, elementCount), device=device, generator=generator
        ) < float(spec.failRate)
        mask = mask & (~failMask)

    # Sparsity (active fraction)
    if spec.sparsityRange != (1.0, 1.0):
        pattern = torch.empty((batchSize, 1), device=device).uniform_(
            spec.sparsityRange[0],
            spec.sparsityRange[1],
            generator=generator,
        )
        activeFraction = (
            torch.rand((batchSize, elementCount), device=device, generator=generator) < pattern
        )
        mask = mask & activeFraction

    return mask


def sampleRandomWeights(
    spec: ArraySpec,
    batchSize: int,
    elementCount: int,
    device: torch.device,
    dtype: torch.dtype,
    elementMask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    # Amplitude
    amplitude = torch.rand(
        (batchSize, elementCount), device=device, dtype=dtype, generator=generator
    )
    amplitude = _normalize_amplitudes(amplitude, elementMask)

    # Phase
    phase = torch.empty((batchSize, elementCount), device=device, dtype=dtype).uniform_(
        -torch.pi,
        torch.pi,
        generator=generator,
    )
    if spec.phaseJitterSTD > 0:
        phase = phase + spec.phaseJitterSTD * torch.randn_like(phase, generator=generator)

    complexWeights = torch.polar(amplitude, phase)
    return complexWeights


def sampleUniformWeights(
    spec: ArraySpec,
    batchSize: int,
    elementCount: int,
    device: torch.device,
    dtype: torch.dtype,
    elementMask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    # Amplitude
    amplitude = torch.ones((batchSize, elementCount), device=device, dtype=dtype)
    amplitude = _normalize_amplitudes(amplitude, elementMask)

    # Phase
    phase = torch.zeros((batchSize, elementCount), device=device, dtype=dtype)
    if spec.phaseJitterSTD > 0:
        phase = phase + spec.phaseJitterSTD * torch.randn_like(phase, generator=generator)

    complexWeights = torch.polar(amplitude, phase)
    return complexWeights


def sampleDirectedWeights(
    spec: ArraySpec,
    batchSize: int,
    localPositions: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    targetLLA: torch.Tensor,
    arrayLLA: torch.Tensor,
    elementMask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    elementCount = localPositions.shape[-1]

    # Pad targetLLA if altitude is omitted
    if targetLLA.shape[-1] == 2:
        zeros = torch.zeros_like(targetLLA[..., :1])
        targetLLA = torch.cat([targetLLA, zeros], dim=-1)

    targetECEF = LLAtoECEF(targetLLA)
    arrayECEF = LLAtoECEF(arrayLLA)
    directionECEF = targetECEF - arrayECEF
    rotationMatrix = getECEFtoENUMapping(arrayLLA)

    if directionECEF.dim() == 1:
        directionECEF = directionECEF.unsqueeze(0).expand(batchSize, -1)
    if rotationMatrix.dim() == 2:
        rotationMatrix = rotationMatrix.unsqueeze(0).expand(batchSize, -1, -1)

    enu = torch.einsum("bij,bj->bi", rotationMatrix, directionECEF)
    east, north, up = enu.unbind(dim=-1)

    x, y, z = -up, east, north
    directionLocal = torch.stack([x, y, z], dim=-1)
    directionLocal = directionLocal / directionLocal.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    waveVector = (2 * torch.pi / spec.wavelength) * directionLocal

    phase = torch.einsum("bi,bin->bn", waveVector, localPositions)

    if spec.phaseJitterSTD > 0:
        phase = phase + spec.phaseJitterSTD * torch.randn_like(phase, generator=generator)

    amplitude = torch.ones((batchSize, elementCount), device=device, dtype=dtype)
    amplitude = _normalize_amplitudes(amplitude, elementMask)

    complexWeights = torch.polar(amplitude, phase)
    return complexWeights


def sampleLLA(
    spec: ArraySpec,
    batchSize: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    latitudes = torch.empty((batchSize,), device=device, dtype=dtype).uniform_(
        spec.latitudeRange[0],
        spec.latitudeRange[1],
        generator=generator,
    )
    longitudes = torch.empty((batchSize,), device=device, dtype=dtype).uniform_(
        spec.longitudeRange[0],
        spec.longitudeRange[1],
        generator=generator,
    )
    altitudes = torch.empty((batchSize,), device=device, dtype=dtype).uniform_(
        spec.altitudeRange[0],
        spec.altitudeRange[1],
        generator=generator,
    )

    return torch.stack([latitudes, longitudes, altitudes], dim=1)  # [B, 3]


def sampleGain(
    spec: ArraySpec,
    batchSize: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if spec.gainRange != (1.0, 1.0):
        gain = torch.empty((batchSize), device=device, dtype=dtype).uniform_(
            spec.gainRange[0],
            spec.gainRange[1],
            generator=generator,
        )
    else:
        gain = torch.ones(batchSize, device=device, dtype=dtype)
    return gain


def chooseOne(
    options,
    device: torch.device = "cpu",
    generator: torch.Generator | None = None,
):
    index = torch.randint(0, len(options), (1,), device=device, generator=generator)
    return options[index.item()]


def generateBatch(
    spec: ArraySpec,
    batchSize: int,
    device: torch.device,
    dtype: torch.dtype,
    elementCount: int | None = None,
    aspectRatio: float | None = None,
    weightsType: str = "uniform",
    targetLLA: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> ArrayBatch:
    if elementCount is None:
        elementCount = int(chooseOne(spec.allowedElementCount, device=device, generator=generator))
    if aspectRatio is None:
        aspectRatio = float(chooseOne(spec.allowedAspectRatio, device=device, generator=generator))

    localPositions = sampleElementPositions(
        spec,
        batchSize,
        elementCount,
        aspectRatio,
        device,
        dtype,
        generator=generator,
    )
    trueElementCount = localPositions.shape[-1]
    elementMask = sampleElementMask(
        spec,
        batchSize,
        trueElementCount,
        device,
        dtype,
        generator=generator,
    )

    LLAPosition = sampleLLA(spec, batchSize, device, dtype, generator=generator)
    ECEFPosition = LLAtoECEF(LLAPosition)
    gain = sampleGain(spec, batchSize, device, dtype, generator=generator)

    if weightsType == "random":
        weights = sampleRandomWeights(
            spec,
            batchSize,
            trueElementCount,
            device,
            dtype,
            elementMask,
            generator=generator,
        )
    elif weightsType == "uniform":
        weights = sampleUniformWeights(
            spec,
            batchSize,
            trueElementCount,
            device,
            dtype,
            elementMask,
            generator=generator,
        )
    elif weightsType == "directed":
        if targetLLA is None:
            raise ValueError("Must provide targetLLA to generate directed weights.")

        weights = sampleDirectedWeights(
            spec,
            batchSize,
            localPositions,
            device,
            dtype,
            targetLLA=targetLLA,
            arrayLLA=LLAPosition,
            elementMask=elementMask,
            generator=generator,
        )
    else:
        raise ValueError(f"Unknown weightsType: {weightsType}")

    return ArrayBatch(
        elementLocalPosition=localPositions,
        weights=weights,
        wavelength=spec.wavelength,
        gain=gain,
        LLAPosition=LLAPosition,
        ECEFPosition=ECEFPosition,
        elementMask=elementMask,
        geometryCacheKey=infer_geometry_cache_key(
            elementLocalPosition=localPositions,
            LLAPosition=LLAPosition,
            ECEFPosition=ECEFPosition,
            gain=gain,
            wavelength=spec.wavelength,
            elementMask=elementMask,
        ),
    )
