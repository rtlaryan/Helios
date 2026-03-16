from typing import Literal

import torch
from scripts.arrayBatch import ArrayBatch
from scripts.arraySimulation import arrayResponseBatch, normalizePower, todB
from scripts.coordinateTransforms import mapLLAtoArrayAZEL
from scripts.targetSpec import TargetSpec

LossType = Literal["MSE", "HUBER"]


def patternMatchLoss(
    batchResponse: torch.Tensor,
    target: TargetSpec,
    lossType: LossType,
    delta: float = 0.1,
) -> torch.Tensor:
    """
    batchResponse: [B, ...] normalized linear power
    returns: [B]
    """

    responseFlat = batchResponse.flatten(1)
    targetImportance = target.importanceMap.flatten()
    targetPower = target.powerMap.flatten()
    importanceSum = targetImportance.sum().clamp_min(1e-8)

    error = responseFlat - targetPower

    if lossType == "MSE":
        weightedError = targetImportance * error.square()
        loss = weightedError.sum(dim=1) / importanceSum

    elif lossType == "HUBER":
        absError = error.abs()
        huber = torch.where(
            absError <= delta,
            0.5 * error.square(),
            delta * (absError - 0.5 * delta),
        )
        weightedError = targetImportance * huber
        loss = weightedError.sum(dim=1) / importanceSum

    else:
        raise ValueError(f"Unsupported Loss Type: {lossType}")

    return loss


def powerConcentrationLoss(
    batchResponse: torch.Tensor,
    target: TargetSpec,
    sigmaDeg: float = 2.0,
) -> torch.Tensor:
    """
    batchResponse: [B, ...] normalized linear power
    returns: [B]
    """

    searchLat = target.searchLatitudes
    searchLon = target.searchLongitudes
    hotspotCoords = target.hotspotCoordinates

    hotspotMap = torch.zeros_like(searchLat)

    for hotspot in hotspotCoords:
        hotspotLat = hotspot[0]
        hotspotLon = hotspot[1]

        distanceSquared = (searchLat - hotspotLat).square() + (searchLon - hotspotLon).square()
        gaussianWeight = torch.exp(-0.5 * distanceSquared / (sigmaDeg**2))
        hotspotMap = torch.maximum(hotspotMap, gaussianWeight)

    hotspotMapFlat = hotspotMap.flatten()

    batchResponseFlat = batchResponse.flatten(1)
    weightedHotspotPower = (batchResponseFlat * hotspotMapFlat).sum(dim=1)
    totalPower = batchResponseFlat.sum(dim=1).clamp_min(1e-8)

    concentration = weightedHotspotPower / totalPower

    return 1.0 - concentration


def batchScore(
    batch: ArrayBatch,
    target: TargetSpec,
    lossType: LossType = "MSE",
) -> torch.Tensor:
    target = target.to(batch.device, batch.dtype)
    targetAZEL = mapLLAtoArrayAZEL(batch, target.targetCoordinates)

    batchResponse = arrayResponseBatch(batch, targetAZEL)
    normalizedResponse = normalizePower(batchResponse)
    dBResponse = todB(normalizedResponse)
    patternLoss = patternMatchLoss(dBResponse, target, lossType)
    concentrationLoss = powerConcentrationLoss(normalizedResponse, target)

    loss = patternLoss + 0.1 * concentrationLoss
    return loss
