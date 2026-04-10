from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from scripts.arrayBatch import ArrayBatch
from scripts.coordinateTransforms import LLAtoECEF, getECEFtoENUMapping, mapLLAtoArrayAZEL
from scripts.targetSpec import TargetBatch, TargetLike, TargetSpec
from simulation.arraySim import arrayResponseBatch, arrayResponseBatchSharedGrid

LossType = Literal["MSE", "HUBER"]  # kept for backward compat, unused internally
TargetMode = Literal["auto", "shared", "per_sample"]


@dataclass
class LossConfig:
    w_shape: float = 1.0
    w_eff: float = 0.1
    w_wide: float = 5
    wide_grid_size: int = 256
    wide_support_dilation_cells: int = 1


LossParameters = LossConfig


@dataclass
class TargetPrep:
    targetCoordinates: torch.Tensor
    powerFlat: torch.Tensor
    importanceFlat: torch.Tensor
    targetShape: torch.Size
    targetECEF: torch.Tensor | None = None


@dataclass
class BatchEvaluation:
    totalLoss: torch.Tensor
    shapeLoss: torch.Tensor
    efficiencyLoss: torch.Tensor
    wideSupportLoss: torch.Tensor
    linearResponse: torch.Tensor
    targetAZEL: tuple[torch.Tensor, torch.Tensor]
    targetMode: TargetMode


_SHARED_TARGET_CACHE: dict[tuple[int | float, ...], TargetPrep] = {}
_WIDE_GRID_CACHE: dict[
    tuple[torch.device, torch.dtype, int], tuple[torch.Tensor, torch.Tensor]
] = {}


def _sharedTargetCacheKey(target: TargetSpec, dtype: torch.dtype) -> tuple[int | float, ...]:
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
    )


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


def _prepareSharedTarget(target: TargetSpec, dtype: torch.dtype) -> TargetPrep:
    key = _sharedTargetCacheKey(target, dtype)
    cached = _SHARED_TARGET_CACHE.get(key)
    if cached is not None:
        return cached

    targetCoordinates = target.targetCoordinates
    targetLLA = torch.cat(
        [targetCoordinates, torch.zeros_like(targetCoordinates[:, :1])],
        dim=-1,
    )
    prep = TargetPrep(
        targetCoordinates=targetCoordinates,
        powerFlat=_to_linear(target.powerMap).flatten().unsqueeze(0),
        importanceFlat=target.importanceMap.flatten().clamp(0.0, 1.0).unsqueeze(0),
        targetShape=target.targetShape,
        targetECEF=LLAtoECEF(targetLLA),
    )
    _SHARED_TARGET_CACHE[key] = prep
    return prep


def _prepareBatchedTarget(target: TargetBatch) -> TargetPrep:
    return TargetPrep(
        targetCoordinates=target.targetCoordinates,
        powerFlat=_to_linear(target.powerMap).flatten(1),
        importanceFlat=target.importanceMap.flatten(1).clamp(0.0, 1.0),
        targetShape=target.targetShape,
    )


def _prepareTarget(target: TargetLike, params: LossConfig, targetMode: TargetMode) -> TargetPrep:
    del params
    if targetMode == "shared":
        assert isinstance(target, TargetSpec)
        return _prepareSharedTarget(target, target.searchLatitudes.dtype)
    assert isinstance(target, TargetBatch)
    return _prepareBatchedTarget(target)


def _shapeFidelityLoss(
    linearResponse: torch.Tensor,
    prep: TargetPrep,
    eps: float = 1e-10,
) -> torch.Tensor:
    responseFlat = linearResponse.flatten(1)
    targetFlat = prep.powerFlat.expand_as(responseFlat)
    weights = prep.importanceFlat.expand_as(responseFlat)
    perPixel = F.smooth_l1_loss(responseFlat, targetFlat, reduction="none")
    weightedLoss = (perPixel * weights).sum(dim=-1)
    normalizer = weights.sum(dim=-1).clamp_min(eps)
    return weightedLoss / normalizer


def _powerEfficiencyLoss(
    linearResponse: torch.Tensor,
    prep: TargetPrep,
    eps: float = 1e-10,
) -> torch.Tensor:
    responseFlat = linearResponse.flatten(1)
    insidePower = (responseFlat * prep.importanceFlat).sum(dim=-1)
    totalPower = responseFlat.sum(dim=-1).clamp_min(eps)
    efficiency = insidePower / totalPower
    return 1.0 - efficiency


def _getWideGrid(
    device: torch.device,
    dtype: torch.dtype,
    gridSize: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (device, dtype, gridSize)
    cached = _WIDE_GRID_CACHE.get(key)
    if cached is not None:
        return cached

    wideAZ = torch.linspace(-torch.pi, torch.pi, gridSize, device=device, dtype=dtype)
    wideEL = torch.linspace(-torch.pi / 2, torch.pi / 2, gridSize, device=device, dtype=dtype)
    cached = torch.meshgrid(wideAZ, wideEL, indexing="ij")
    _WIDE_GRID_CACHE[key] = cached
    return cached


def _wideAreaResponse(
    batch: ArrayBatch,
    gridSize: int,
    chunkSize: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    wideAZGrid, wideELGrid = _getWideGrid(batch.device, batch.dtype, gridSize)
    wideResponse = arrayResponseBatchSharedGrid(
        batch,
        (wideAZGrid, wideELGrid),
        chunkSize=chunkSize,
    )
    return wideResponse, wideAZGrid, wideELGrid


def _rowsAreIdentical(tensor: torch.Tensor) -> bool:
    if tensor.shape[0] <= 1:
        return True
    return bool(torch.equal(tensor, tensor[:1].expand_as(tensor)))


def _canUseSharedTargetFastPath(batch: ArrayBatch, prep: TargetPrep) -> bool:
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
    prep: TargetPrep,
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


def _rasterizeWideSupportMask(
    targetAZEL: tuple[torch.Tensor, torch.Tensor],
    prep: TargetPrep,
    wideAZGrid: torch.Tensor,
    wideELGrid: torch.Tensor,
    dilationCells: int,
) -> torch.Tensor:
    flatAZ, flatEL = targetAZEL
    batchSize = flatAZ.shape[0]
    gridSizeAZ, gridSizeEL = wideAZGrid.shape
    supportMask = torch.zeros(
        (batchSize, gridSizeAZ, gridSizeEL),
        device=flatAZ.device,
        dtype=torch.bool,
    )

    azMin = wideAZGrid[0, 0]
    azMax = wideAZGrid[-1, 0]
    elMin = wideELGrid[0, 0]
    elMax = wideELGrid[0, -1]
    azScale = (gridSizeAZ - 1) / (azMax - azMin).clamp_min(torch.finfo(flatAZ.dtype).eps)
    elScale = (gridSizeEL - 1) / (elMax - elMin).clamp_min(torch.finfo(flatEL.dtype).eps)

    azIdx = ((flatAZ - azMin) * azScale).round().long().clamp(0, gridSizeAZ - 1)
    elIdx = ((flatEL - elMin) * elScale).round().long().clamp(0, gridSizeEL - 1)

    validMask = (
        (prep.importanceFlat.expand_as(flatAZ) > 0)
        & torch.isfinite(flatAZ)
        & torch.isfinite(flatEL)
    )

    if validMask.any():
        sampleIdx = torch.arange(batchSize, device=flatAZ.device).unsqueeze(1).expand_as(validMask)
        supportMask[
            sampleIdx[validMask],
            azIdx[validMask],
            elIdx[validMask],
        ] = True

    if dilationCells > 0:
        kernelSize = 2 * dilationCells + 1
        supportMask = (
            F.max_pool2d(
                supportMask.unsqueeze(1).to(dtype=wideAZGrid.dtype),
                kernel_size=kernelSize,
                stride=1,
                padding=dilationCells,
            ).squeeze(1)
            > 0
        )

    return supportMask


def _wideSupportLossFromResponse(
    wideResponse: torch.Tensor,
    globalPeak: torch.Tensor,
    supportMask: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    normalizedResponse = wideResponse / globalPeak.view(-1, 1, 1).clamp_min(eps)
    outsideMask = (~supportMask).to(dtype=wideResponse.dtype)
    outsideCount = outsideMask.sum(dim=(1, 2)).clamp_min(1.0)
    outsideEnergy = (normalizedResponse.square() * outsideMask).sum(dim=(1, 2))
    return outsideEnergy / outsideCount


def evaluateBatch(
    batch: ArrayBatch,
    target: TargetLike,
    params: LossConfig,
    targetMode: TargetMode = "auto",
    logTerms: bool = False,
    linearResponseChunkSize: int | None = None,
    wideResponseChunkSize: int | None = None,
    allowSharedTargetFastPath: bool = True,
) -> BatchEvaluation:
    resolvedMode = _resolveTargetMode(target, targetMode)
    target = target.to(batch.device, batch.dtype)
    prep = _prepareTarget(target, params, resolvedMode)
    useSharedTargetFastPath = (
        allowSharedTargetFastPath
        and resolvedMode == "shared"
        and _canUseSharedTargetFastPath(batch, prep)
    )

    if useSharedTargetFastPath:
        sharedAZ, sharedEL = _sharedTargetAZELForHomogeneousBatch(batch, prep)
        targetAZEL = (
            sharedAZ.unsqueeze(0).expand(batch.batchSize, -1),
            sharedEL.unsqueeze(0).expand(batch.batchSize, -1),
        )
        linearResponse = arrayResponseBatchSharedGrid(
            batch,
            (sharedAZ, sharedEL),
            chunkSize=linearResponseChunkSize,
        ).reshape(batch.batchSize, *prep.targetShape)
    else:
        targetAZEL = mapLLAtoArrayAZEL(batch, prep.targetCoordinates)
        linearResponse = arrayResponseBatch(
            batch,
            targetAZEL,
            chunkSize=linearResponseChunkSize,
        ).reshape(batch.batchSize, *prep.targetShape)

    wideResponse, wideAZGrid, wideELGrid = _wideAreaResponse(
        batch,
        gridSize=params.wide_grid_size,
        chunkSize=wideResponseChunkSize,
    )

    if useSharedTargetFastPath:
        sharedSupportMask = _rasterizeWideSupportMask(
            targetAZEL=(targetAZEL[0][:1], targetAZEL[1][:1]),
            prep=prep,
            wideAZGrid=wideAZGrid,
            wideELGrid=wideELGrid,
            dilationCells=params.wide_support_dilation_cells,
        )
        wideSupportMask = sharedSupportMask.expand(batch.batchSize, -1, -1)
    else:
        wideSupportMask = _rasterizeWideSupportMask(
            targetAZEL=targetAZEL,
            prep=prep,
            wideAZGrid=wideAZGrid,
            wideELGrid=wideELGrid,
            dilationCells=params.wide_support_dilation_cells,
        )

    globalPeak = wideResponse.flatten(1).amax(dim=-1).clamp_min(1e-10)
    normalizedShapeResponse = linearResponse / globalPeak.view(
        batch.batchSize, *([1] * len(prep.targetShape))
    )

    shapeLoss = _shapeFidelityLoss(normalizedShapeResponse, prep)
    efficiencyLoss = _powerEfficiencyLoss(linearResponse, prep)
    wideSupportLoss = _wideSupportLossFromResponse(
        wideResponse=wideResponse,
        globalPeak=globalPeak,
        supportMask=wideSupportMask,
    )
    totalLoss = (
        params.w_shape * shapeLoss + params.w_eff * efficiencyLoss + params.w_wide * wideSupportLoss
    )

    if logTerms:
        print(
            f"shape={params.w_shape * shapeLoss.mean().item():.4f} | "
            f"efficiency={params.w_eff * efficiencyLoss.mean().item():.4f} | "
            f"wide={params.w_wide * wideSupportLoss.mean().item():.4f}"
        )

    return BatchEvaluation(
        totalLoss=totalLoss,
        shapeLoss=shapeLoss,
        efficiencyLoss=efficiencyLoss,
        wideSupportLoss=wideSupportLoss,
        linearResponse=linearResponse,
        targetAZEL=targetAZEL,
        targetMode=resolvedMode,
    )


def batchLoss(
    batch: ArrayBatch,
    target: TargetLike,
    params: LossConfig,
    lossType: LossType = "MSE",
    logTerms: bool = False,
    targetMode: TargetMode = "auto",
    linearResponseChunkSize: int | None = None,
    wideResponseChunkSize: int | None = None,
    allowSharedTargetFastPath: bool = True,
) -> torch.Tensor:
    del lossType
    return evaluateBatch(
        batch=batch,
        target=target,
        params=params,
        targetMode=targetMode,
        logTerms=logTerms,
        linearResponseChunkSize=linearResponseChunkSize,
        wideResponseChunkSize=wideResponseChunkSize,
        allowSharedTargetFastPath=allowSharedTargetFastPath,
    ).totalLoss
