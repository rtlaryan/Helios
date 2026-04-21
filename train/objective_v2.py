from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from scripts.arrayBatch import ArrayBatch
from scripts.coordinateTransforms import LLAtoECEF, getECEFtoENUMapping, mapLLAtoArrayAZEL
from scripts.targetSpec import TargetBatch, TargetLike, TargetSpec
from simulation.arraySim import arrayResponseBatch, arrayResponseBatchSharedGrid

TargetMode = Literal["auto", "shared", "per_sample"]


@dataclass
class LossConfigV2:
    w_shape: float = 1.0
    w_coverage: float = 2.0
    w_shell: float = 0.5
    w_global: float = 4.0
    w_peak: float = 8.0
    importance_cutoff: float = 0.05
    boundary_cells: int = 1
    concern_dilation_cells: int = 1
    peak_topk: int = 8
    samples_per_hpbw: float = 4.0
    min_front_grid_size: int = 32
    max_front_grid_size: int = 256


@dataclass
class TargetPrepV2:
    targetCoordinates: torch.Tensor
    targetNormFlat: torch.Tensor
    serviceMaskFlat: torch.Tensor
    concernMaskFlat: torch.Tensor
    shellMaskFlat: torch.Tensor
    boundaryWeightFlat: torch.Tensor
    thresholdLinear: torch.Tensor
    targetShape: torch.Size
    targetECEF: torch.Tensor | None = None


@dataclass
class BatchEvaluationV2:
    totalLoss: torch.Tensor
    shapeLoss: torch.Tensor
    coverageLoss: torch.Tensor
    shellLoss: torch.Tensor
    globalLoss: torch.Tensor
    peakLoss: torch.Tensor
    linearResponse: torch.Tensor
    targetAZEL: tuple[torch.Tensor, torch.Tensor]
    targetMode: TargetMode
    stats: dict[str, float] | None = None
    metadata: dict[str, float | int] | None = None


@dataclass
class _FineSharedCache:
    azimuth: torch.Tensor
    elevation: torch.Tensor
    steering: torch.Tensor
    qService: torch.Tensor
    qShell: torch.Tensor
    serviceWeightSum: torch.Tensor
    shellWeightSum: torch.Tensor


@dataclass
class _CoarseFrontSharedCache:
    azimuthGrid: torch.Tensor
    elevationGrid: torch.Tensor
    steering: torch.Tensor
    omegaGrid: torch.Tensor
    concernMask: torch.Tensor
    qFrontOutside: torch.Tensor
    outsideWeightSum: torch.Tensor
    angleStep: float
    hpbw: float


_SHARED_PREP_CACHE_V2: dict[tuple[int | float | str, ...], TargetPrepV2] = {}
_FINE_SHARED_CACHE_V2: dict[tuple[int | float | str, ...], _FineSharedCache] = {}
_COARSE_GRID_CACHE_V2: dict[
    tuple[torch.device, torch.dtype, int, int], tuple[torch.Tensor, torch.Tensor]
] = {}
_COARSE_SHARED_CACHE_V2: dict[tuple[int | float | str, ...], _CoarseFrontSharedCache] = {}


def clearCachesV2() -> None:
    _SHARED_PREP_CACHE_V2.clear()
    _FINE_SHARED_CACHE_V2.clear()
    _COARSE_GRID_CACHE_V2.clear()
    _COARSE_SHARED_CACHE_V2.clear()


def _to_linear(powerMap: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    if torch.any(powerMap < 0):
        return torch.pow(10.0, powerMap / 10.0).clamp_min(eps)
    return powerMap.clamp_min(0.0)


def _resolveTargetMode(target: TargetLike, targetMode: TargetMode) -> TargetMode:
    if targetMode == "auto":
        return "per_sample" if isinstance(target, TargetBatch) else "shared"
    if targetMode == "shared" and isinstance(target, TargetBatch):
        raise ValueError("targetMode='shared' is incompatible with TargetBatch")
    if targetMode == "per_sample" and isinstance(target, TargetSpec):
        raise ValueError("targetMode='per_sample' is incompatible with TargetSpec")
    return targetMode


def _sharedPrepCacheKey(
    target: TargetSpec,
    dtype: torch.dtype,
    params: LossConfigV2,
) -> tuple[int | float | str, ...]:
    return (
        target.searchLatitudes.data_ptr(),
        target.searchLatitudes._version,
        target.searchLongitudes.data_ptr(),
        target.searchLongitudes._version,
        target.importanceMap.data_ptr(),
        target.importanceMap._version,
        target.powerMap.data_ptr(),
        target.powerMap._version,
        target.hotspotCoordinates.data_ptr(),
        target.hotspotCoordinates._version,
        hash(dtype),
        float(target.thresholdDB),
        float(params.importance_cutoff),
        int(params.boundary_cells),
    )


def _erodeMask(mask: torch.Tensor, cells: int) -> torch.Tensor:
    if cells <= 0:
        return mask
    squeezed = False
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
        squeezed = True
    inverse = (~mask).to(dtype=torch.float32)
    dilatedInverse = F.max_pool2d(
        inverse.unsqueeze(1),
        kernel_size=2 * cells + 1,
        stride=1,
        padding=cells,
    ).squeeze(1)
    eroded = dilatedInverse == 0
    if squeezed:
        return eroded[0]
    return eroded


def _buildTargetPrep(
    targetCoordinates: torch.Tensor,
    powerMap: torch.Tensor,
    importanceMap: torch.Tensor,
    targetShape: torch.Size,
    thresholdDB: torch.Tensor,
    dtype: torch.dtype,
    params: LossConfigV2,
    targetECEF: torch.Tensor | None = None,
) -> TargetPrepV2:
    eps = torch.finfo(dtype).eps
    powerLinear = _to_linear(powerMap)
    if powerLinear.ndim == 2:
        powerLinear = powerLinear.unsqueeze(0)
        importanceMap = importanceMap.unsqueeze(0)
        thresholdDB = thresholdDB.view(1)
    peak = powerLinear.flatten(1).amax(dim=1).clamp_min(eps)
    if bool(torch.any(peak <= eps)):
        raise ValueError("objective_v2 requires powerMap to contain positive support")
    targetNorm = powerLinear / peak.view(-1, 1, 1)
    thresholdLinear = torch.pow(10.0, -thresholdDB.to(dtype=dtype) / 10.0).view(-1, 1, 1)
    serviceMask = targetNorm >= thresholdLinear
    if bool(torch.any(serviceMask.flatten(1).sum(dim=1) == 0)):
        raise ValueError("objective_v2 derived an empty serviceMask from targetNorm and thresholdDB")
    concernMask = serviceMask | (importanceMap.clamp(0.0, 1.0) >= float(params.importance_cutoff))
    shellMask = concernMask & (~serviceMask)
    if params.boundary_cells > 0:
        erodedService = _erodeMask(serviceMask, params.boundary_cells)
        boundaryMask = serviceMask & (~erodedService)
    else:
        boundaryMask = torch.zeros_like(serviceMask)
    boundaryWeight = 1.0 + boundaryMask.to(dtype=dtype)
    return TargetPrepV2(
        targetCoordinates=targetCoordinates,
        targetNormFlat=targetNorm.flatten(1),
        serviceMaskFlat=serviceMask.flatten(1),
        concernMaskFlat=concernMask.flatten(1),
        shellMaskFlat=shellMask.flatten(1),
        boundaryWeightFlat=boundaryWeight.flatten(1),
        thresholdLinear=thresholdLinear.flatten(1),
        targetShape=targetShape,
        targetECEF=targetECEF,
    )


def _prepareSharedTargetV2(target: TargetSpec, params: LossConfigV2, dtype: torch.dtype) -> TargetPrepV2:
    key = _sharedPrepCacheKey(target, dtype, params)
    cached = _SHARED_PREP_CACHE_V2.get(key)
    if cached is not None:
        return cached
    targetCoordinates = target.targetCoordinates
    targetLLA = torch.cat([targetCoordinates, torch.zeros_like(targetCoordinates[:, :1])], dim=-1)
    prep = _buildTargetPrep(
        targetCoordinates=targetCoordinates,
        powerMap=target.powerMap,
        importanceMap=target.importanceMap,
        targetShape=target.targetShape,
        thresholdDB=torch.tensor([float(target.thresholdDB)], device=target.powerMap.device, dtype=dtype),
        dtype=dtype,
        params=params,
        targetECEF=LLAtoECEF(targetLLA),
    )
    _SHARED_PREP_CACHE_V2[key] = prep
    return prep


def _prepareBatchedTargetV2(target: TargetBatch, params: LossConfigV2) -> TargetPrepV2:
    threshold = target.thresholdDB
    if not isinstance(threshold, torch.Tensor):
        threshold = torch.full(
            (target.batchSize,),
            float(threshold),
            device=target.powerMap.device,
            dtype=target.powerMap.dtype,
        )
    elif threshold.ndim == 0:
        threshold = threshold.expand(target.batchSize)
    return _buildTargetPrep(
        targetCoordinates=target.targetCoordinates,
        powerMap=target.powerMap,
        importanceMap=target.importanceMap,
        targetShape=target.targetShape,
        thresholdDB=threshold,
        dtype=target.powerMap.dtype,
        params=params,
    )


def _prepareTargetV2(target: TargetLike, params: LossConfigV2, targetMode: TargetMode) -> TargetPrepV2:
    if targetMode == "shared":
        assert isinstance(target, TargetSpec)
        return _prepareSharedTargetV2(target, params, target.searchLatitudes.dtype)
    assert isinstance(target, TargetBatch)
    return _prepareBatchedTargetV2(target, params)


def _rowsAreIdentical(tensor: torch.Tensor) -> bool:
    if tensor.shape[0] <= 1:
        return True
    return bool(torch.equal(tensor, tensor[:1].expand_as(tensor)))


def _canUseSharedTargetFastPath(batch: ArrayBatch, prep: TargetPrepV2) -> bool:
    if prep.targetECEF is None:
        return False
    return (
        _rowsAreIdentical(batch.LLAPosition)
        and _rowsAreIdentical(batch.ECEFPosition)
        and _rowsAreIdentical(batch.elementLocalPosition)
        and _rowsAreIdentical(batch.gain)
    )


def _sharedTargetAZELForHomogeneousBatch(
    batch: ArrayBatch,
    prep: TargetPrepV2,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert prep.targetECEF is not None
    rotationMatrix = getECEFtoENUMapping(batch.LLAPosition[:1])
    directionVector = prep.targetECEF.unsqueeze(0) - batch.ECEFPosition[:1, None, :]
    east, north, up = torch.einsum("bij,bpj->bpi", rotationMatrix, directionVector).unbind(dim=-1)
    x, y, z = -up, east, north
    azimuth = torch.atan2(y, x).reshape(-1)
    rho = torch.hypot(x, y).clamp_min(1e-12)
    elevation = torch.atan2(z, rho).reshape(-1)
    return azimuth, elevation


def _batchGeometryCacheKey(batch: ArrayBatch) -> tuple[int | float | str, ...]:
    return (
        batch.elementLocalPosition.data_ptr(),
        batch.elementLocalPosition._version,
        batch.LLAPosition.data_ptr(),
        batch.LLAPosition._version,
        batch.ECEFPosition.data_ptr(),
        batch.ECEFPosition._version,
        float(batch.wavelength),
        str(batch.device),
        hash(batch.dtype),
    )


def _complex_dtype(realDType: torch.dtype) -> torch.dtype:
    if realDType == torch.float64:
        return torch.complex128
    return torch.complex64


def _steeringMatrix(
    positions: torch.Tensor,
    wavelength: float,
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
) -> torch.Tensor:
    pos = positions.to(dtype=azimuth.dtype)
    wave = (2.0 * torch.pi / wavelength) * torch.stack(
        [
            torch.cos(elevation) * torch.cos(azimuth),
            torch.cos(elevation) * torch.sin(azimuth),
            torch.sin(elevation),
        ],
        dim=0,
    ).reshape(3, -1)
    phase = pos.transpose(0, 1) @ wave
    complexDType = _complex_dtype(phase.dtype)
    return torch.exp(1j * phase.to(dtype=complexDType))


def _responseFromSteering(weights: torch.Tensor, steering: torch.Tensor) -> torch.Tensor:
    response = weights.conj() @ steering
    return response.real.square() + response.imag.square()


def _buildOperator(steering: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weightedSteering = steering * weights.to(dtype=steering.real.dtype).unsqueeze(0)
    return weightedSteering @ steering.conj().transpose(0, 1)


def _quadraticForm(weights: torch.Tensor, operator: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bi,ij,bj->b", weights.conj(), operator, weights).real


def _frontMaskFromAZEL(azimuth: torch.Tensor, elevation: torch.Tensor) -> torch.Tensor:
    x = torch.cos(elevation) * torch.cos(azimuth)
    return x >= 0


def _estimateFrontGridShape(
    batch: ArrayBatch,
    params: LossConfigV2,
) -> tuple[int, int, float, float]:
    ySpan = (batch.elementLocalPosition[:, 1, :].amax(dim=1) - batch.elementLocalPosition[:, 1, :].amin(dim=1)).amax()
    zSpan = (batch.elementLocalPosition[:, 2, :].amax(dim=1) - batch.elementLocalPosition[:, 2, :].amin(dim=1)).amax()
    dy = max(float(ySpan.item()), float(batch.wavelength))
    dz = max(float(zSpan.item()), float(batch.wavelength))
    hpbwY = 0.886 * float(batch.wavelength) / dy
    hpbwZ = 0.886 * float(batch.wavelength) / dz
    hpbw = max(1e-4, min(hpbwY, hpbwZ))
    angleStep = max(hpbw / max(float(params.samples_per_hpbw), 1.0), 1e-4)
    azSize = int(max(params.min_front_grid_size, min(params.max_front_grid_size, math.ceil(math.pi / angleStep) + 1)))
    elSize = int(max(params.min_front_grid_size, min(params.max_front_grid_size, math.ceil(math.pi / angleStep) + 1)))
    return azSize, elSize, angleStep, hpbw


def _getFrontGrid(
    device: torch.device,
    dtype: torch.dtype,
    azSize: int,
    elSize: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (device, dtype, azSize, elSize)
    cached = _COARSE_GRID_CACHE_V2.get(key)
    if cached is not None:
        return cached
    azimuth = torch.linspace(-torch.pi / 2, torch.pi / 2, azSize, device=device, dtype=dtype)
    elevation = torch.linspace(-torch.pi / 2, torch.pi / 2, elSize, device=device, dtype=dtype)
    cached = torch.meshgrid(azimuth, elevation, indexing="ij")
    _COARSE_GRID_CACHE_V2[key] = cached
    return cached


def _rasterizeProjectedMask(
    targetAZEL: tuple[torch.Tensor, torch.Tensor],
    flatMask: torch.Tensor,
    azimuthGrid: torch.Tensor,
    elevationGrid: torch.Tensor,
    dilationCells: int,
) -> torch.Tensor:
    flatAZ, flatEL = targetAZEL
    batchSize = flatAZ.shape[0]
    gridAZ, gridEL = azimuthGrid.shape
    raster = torch.zeros((batchSize, gridAZ, gridEL), device=flatAZ.device, dtype=torch.bool)
    azMin = azimuthGrid[0, 0]
    azMax = azimuthGrid[-1, 0]
    elMin = elevationGrid[0, 0]
    elMax = elevationGrid[0, -1]
    azScale = (gridAZ - 1) / (azMax - azMin).clamp_min(torch.finfo(flatAZ.dtype).eps)
    elScale = (gridEL - 1) / (elMax - elMin).clamp_min(torch.finfo(flatEL.dtype).eps)
    azIdx = ((flatAZ - azMin) * azScale).round().long().clamp(0, gridAZ - 1)
    elIdx = ((flatEL - elMin) * elScale).round().long().clamp(0, gridEL - 1)
    valid = flatMask.expand_as(flatAZ) & torch.isfinite(flatAZ) & torch.isfinite(flatEL) & _frontMaskFromAZEL(flatAZ, flatEL)
    if bool(valid.any()):
        sampleIdx = torch.arange(batchSize, device=flatAZ.device).unsqueeze(1).expand_as(valid)
        raster[sampleIdx[valid], azIdx[valid], elIdx[valid]] = True
    if dilationCells > 0:
        raster = (
            F.max_pool2d(
                raster.unsqueeze(1).to(dtype=azimuthGrid.dtype),
                kernel_size=2 * dilationCells + 1,
                stride=1,
                padding=dilationCells,
            ).squeeze(1)
            > 0
        )
    return raster


def _maskedWeightedMean(
    values: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    numerator = (values * weights).sum(dim=-1)
    denominator = weights.sum(dim=-1).clamp_min(eps)
    return numerator / denominator


def _topKMaskedMean(values: torch.Tensor, mask: torch.Tensor, topk: int) -> torch.Tensor:
    flatValues = values.flatten(1)
    flatMask = mask.flatten(1)
    maskedValues = flatValues.masked_fill(~flatMask, float("-inf"))
    k = max(1, min(int(topk), maskedValues.shape[1]))
    topValues = maskedValues.topk(k, dim=1).values
    finiteMask = torch.isfinite(topValues)
    counts = finiteMask.sum(dim=1).clamp_min(1)
    safeValues = torch.where(finiteMask, topValues, torch.zeros_like(topValues))
    return safeValues.sum(dim=1) / counts


def _buildFineSharedCache(
    batch: ArrayBatch,
    prep: TargetPrepV2,
    params: LossConfigV2,
) -> _FineSharedCache:
    cacheKey = (_batchGeometryCacheKey(batch), prep.targetNormFlat.data_ptr(), prep.serviceMaskFlat.data_ptr(), prep.shellMaskFlat.data_ptr())
    cached = _FINE_SHARED_CACHE_V2.get(cacheKey)
    if cached is not None:
        return cached
    azimuth, elevation = _sharedTargetAZELForHomogeneousBatch(batch, prep)
    steering = _steeringMatrix(batch.elementLocalPosition[0], batch.wavelength, azimuth, elevation)
    frontMask = _frontMaskFromAZEL(azimuth, elevation).to(dtype=batch.dtype)
    omega = torch.cos(elevation).clamp_min(0.0)
    serviceWeights = omega * prep.serviceMaskFlat[0].to(dtype=batch.dtype) * frontMask
    shellWeights = omega * prep.shellMaskFlat[0].to(dtype=batch.dtype) * frontMask
    cached = _FineSharedCache(
        azimuth=azimuth,
        elevation=elevation,
        steering=steering,
        qService=_buildOperator(steering, serviceWeights),
        qShell=_buildOperator(steering, shellWeights),
        serviceWeightSum=serviceWeights.sum().clamp_min(torch.finfo(batch.dtype).eps),
        shellWeightSum=shellWeights.sum().clamp_min(torch.finfo(batch.dtype).eps),
    )
    _FINE_SHARED_CACHE_V2[cacheKey] = cached
    return cached


def _buildCoarseSharedCache(
    batch: ArrayBatch,
    prep: TargetPrepV2,
    params: LossConfigV2,
    targetAZEL: tuple[torch.Tensor, torch.Tensor],
    azimuthGrid: torch.Tensor,
    elevationGrid: torch.Tensor,
    angleStep: float,
    hpbw: float,
) -> _CoarseFrontSharedCache:
    cacheKey = (
        _batchGeometryCacheKey(batch),
        prep.concernMaskFlat.data_ptr(),
        azimuthGrid.shape[0],
        elevationGrid.shape[1],
        int(params.concern_dilation_cells),
    )
    cached = _COARSE_SHARED_CACHE_V2.get(cacheKey)
    if cached is not None:
        return cached
    steering = _steeringMatrix(
        batch.elementLocalPosition[0],
        batch.wavelength,
        azimuthGrid.reshape(-1),
        elevationGrid.reshape(-1),
    )
    concernMask = _rasterizeProjectedMask(
        targetAZEL=(targetAZEL[0][:1], targetAZEL[1][:1]),
        flatMask=prep.concernMaskFlat[:1],
        azimuthGrid=azimuthGrid,
        elevationGrid=elevationGrid,
        dilationCells=params.concern_dilation_cells,
    )[0]
    omegaGrid = torch.cos(elevationGrid).clamp_min(0.0)
    outsideWeights = (omegaGrid * (~concernMask).to(dtype=batch.dtype)).reshape(-1)
    cached = _CoarseFrontSharedCache(
        azimuthGrid=azimuthGrid,
        elevationGrid=elevationGrid,
        steering=steering,
        omegaGrid=omegaGrid,
        concernMask=concernMask,
        qFrontOutside=_buildOperator(steering, outsideWeights),
        outsideWeightSum=outsideWeights.sum().clamp_min(torch.finfo(batch.dtype).eps),
        angleStep=angleStep,
        hpbw=hpbw,
    )
    _COARSE_SHARED_CACHE_V2[cacheKey] = cached
    return cached


def evaluateBatchV2(
    batch: ArrayBatch,
    target: TargetLike,
    params: LossConfigV2,
    targetMode: TargetMode = "auto",
    logTerms: bool = False,
    linearResponseChunkSize: int | None = None,
    wideResponseChunkSize: int | None = None,
    allowSharedTargetFastPath: bool = True,
    collectStats: bool = False,
) -> BatchEvaluationV2:
    startedAt = time.perf_counter()
    resolvedMode = _resolveTargetMode(target, targetMode)
    target = target.to(batch.device, batch.dtype)
    prepStarted = time.perf_counter()
    prep = _prepareTargetV2(target, params, resolvedMode)
    prepSeconds = time.perf_counter() - prepStarted
    useSharedFastPath = (
        allowSharedTargetFastPath
        and resolvedMode == "shared"
        and _canUseSharedTargetFastPath(batch, prep)
    )

    fineOperatorBuildSeconds = 0.0
    fineReuseSeconds = 0.0
    fineStarted = time.perf_counter()
    if useSharedFastPath:
        cacheLookupStarted = time.perf_counter()
        fineCache = _buildFineSharedCache(batch, prep, params)
        fineReuseSeconds = time.perf_counter() - cacheLookupStarted
        linearResponseFlat = _responseFromSteering(batch.weights, fineCache.steering)
        targetAZEL = (
            fineCache.azimuth.unsqueeze(0).expand(batch.batchSize, -1),
            fineCache.elevation.unsqueeze(0).expand(batch.batchSize, -1),
        )
        linearResponse = linearResponseFlat.reshape(batch.batchSize, *prep.targetShape)
    else:
        targetAZEL = mapLLAtoArrayAZEL(batch, prep.targetCoordinates)
        linearResponse = arrayResponseBatch(
            batch=batch,
            relativeTargetAZEL=targetAZEL,
            chunkSize=linearResponseChunkSize,
        ).reshape(batch.batchSize, *prep.targetShape)
    fineSeconds = time.perf_counter() - fineStarted

    azSize, elSize, angleStep, hpbw = _estimateFrontGridShape(batch, params)
    coarseGridStarted = time.perf_counter()
    frontAZGrid, frontELGrid = _getFrontGrid(batch.device, batch.dtype, azSize, elSize)
    coarseResponse = arrayResponseBatchSharedGrid(
        batch=batch,
        relativeTargetAZEL=(frontAZGrid, frontELGrid),
        chunkSize=wideResponseChunkSize,
    )
    coarseSeconds = time.perf_counter() - coarseGridStarted

    finePeak = linearResponse.flatten(1).amax(dim=1)
    coarsePeak = coarseResponse.flatten(1).amax(dim=1)
    peakRef = torch.maximum(finePeak, coarsePeak).clamp_min(torch.finfo(batch.dtype).eps)
    responseNormFlat = linearResponse.flatten(1) / peakRef.unsqueeze(1)
    coarseNorm = coarseResponse / peakRef.view(-1, 1, 1)

    serviceWeights = prep.serviceMaskFlat.to(dtype=batch.dtype) * prep.boundaryWeightFlat
    shapePerPixel = F.smooth_l1_loss(
        responseNormFlat,
        prep.targetNormFlat.expand_as(responseNormFlat),
        reduction="none",
    )
    shapeLoss = _maskedWeightedMean(shapePerPixel, serviceWeights)

    coverageShortfall = torch.relu(prep.thresholdLinear.expand_as(responseNormFlat) - responseNormFlat)
    coverageLoss = _maskedWeightedMean(coverageShortfall.square(), prep.serviceMaskFlat.to(dtype=batch.dtype))

    if useSharedFastPath:
        operatorStarted = time.perf_counter()
        shellEnergy = _quadraticForm(batch.weights, fineCache.qShell)
        shellLoss = shellEnergy / peakRef / fineCache.shellWeightSum
        coarseCache = _buildCoarseSharedCache(
            batch=batch,
            prep=prep,
            params=params,
            targetAZEL=targetAZEL,
            azimuthGrid=frontAZGrid,
            elevationGrid=frontELGrid,
            angleStep=angleStep,
            hpbw=hpbw,
        )
        fineOperatorBuildSeconds += time.perf_counter() - operatorStarted
        globalEnergy = _quadraticForm(batch.weights, coarseCache.qFrontOutside)
        globalLoss = globalEnergy / peakRef / coarseCache.outsideWeightSum
        projectedConcernMask = coarseCache.concernMask.unsqueeze(0).expand(batch.batchSize, -1, -1)
    else:
        localOmega = torch.cos(targetAZEL[1]).clamp_min(0.0)
        localShellWeights = localOmega * prep.shellMaskFlat.to(dtype=batch.dtype)
        shellLoss = _maskedWeightedMean(responseNormFlat, localShellWeights)
        projectedConcernMask = _rasterizeProjectedMask(
            targetAZEL=targetAZEL,
            flatMask=prep.concernMaskFlat,
            azimuthGrid=frontAZGrid,
            elevationGrid=frontELGrid,
            dilationCells=params.concern_dilation_cells,
        )
        coarseOmega = torch.cos(frontELGrid).clamp_min(0.0).unsqueeze(0).expand(batch.batchSize, -1, -1)
        globalWeights = coarseOmega * (~projectedConcernMask).to(dtype=batch.dtype)
        globalLoss = _maskedWeightedMean(coarseNorm.flatten(1), globalWeights.flatten(1))

    offTargetMask = ~projectedConcernMask
    peakLoss = _topKMaskedMean(coarseNorm, offTargetMask, params.peak_topk)

    totalLoss = (
        params.w_shape * shapeLoss
        + params.w_coverage * coverageLoss
        + params.w_shell * shellLoss
        + params.w_global * globalLoss
        + params.w_peak * peakLoss
    )
    if logTerms:
        print(
            f"shape={params.w_shape * shapeLoss.mean().item():.4f} | "
            f"coverage={params.w_coverage * coverageLoss.mean().item():.4f} | "
            f"shell={params.w_shell * shellLoss.mean().item():.4f} | "
            f"global={params.w_global * globalLoss.mean().item():.4f} | "
            f"peak={params.w_peak * peakLoss.mean().item():.4f}"
        )

    totalSeconds = time.perf_counter() - startedAt
    stats = None
    metadata = None
    if collectStats:
        stats = {
            "prepSeconds": float(prepSeconds),
            "fineResponseSeconds": float(fineSeconds),
            "coarseResponseSeconds": float(coarseSeconds),
            "operatorSeconds": float(fineOperatorBuildSeconds),
            "cacheLookupSeconds": float(fineReuseSeconds),
            "totalSeconds": float(totalSeconds),
        }
        metadata = {
            "frontAzSize": int(azSize),
            "frontElSize": int(elSize),
            "angleStepRad": float(angleStep),
            "estimatedHPBWRad": float(hpbw),
            "frontSampleCount": int(azSize * elSize),
        }

    return BatchEvaluationV2(
        totalLoss=totalLoss,
        shapeLoss=shapeLoss,
        coverageLoss=coverageLoss,
        shellLoss=shellLoss,
        globalLoss=globalLoss,
        peakLoss=peakLoss,
        linearResponse=linearResponse,
        targetAZEL=targetAZEL,
        targetMode=resolvedMode,
        stats=stats,
        metadata=metadata,
    )


def batchLossV2(
    batch: ArrayBatch,
    target: TargetLike,
    params: LossConfigV2,
    logTerms: bool = False,
    targetMode: TargetMode = "auto",
    linearResponseChunkSize: int | None = None,
    wideResponseChunkSize: int | None = None,
    allowSharedTargetFastPath: bool = True,
) -> torch.Tensor:
    return evaluateBatchV2(
        batch=batch,
        target=target,
        params=params,
        targetMode=targetMode,
        logTerms=logTerms,
        linearResponseChunkSize=linearResponseChunkSize,
        wideResponseChunkSize=wideResponseChunkSize,
        allowSharedTargetFastPath=allowSharedTargetFastPath,
    ).totalLoss
