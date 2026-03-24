from dataclasses import dataclass
from typing import Literal

import torch
from scripts.arrayBatch import ArrayBatch
from scripts.arraySimulation import arrayResponseBatch
from scripts.coordinateTransforms import mapLLAtoArrayAZEL
from scripts.targetSpec import TargetSpec

LossType = Literal["MSE", "HUBER"]


@dataclass
class LossParameters:
    shapeK: float = 5e-4
    concentrationK: float = 0.5
    deliveredK: float = 1e-3
    fractionK: float = 0.5
    spillK: float = 0.7


def _target_linear_map(target: TargetSpec, eps: float = 1e-10) -> torch.Tensor:
    powerMap = target.powerMap
    if torch.any(powerMap < 0):
        return torch.pow(10.0, powerMap / 10.0).clamp_min(eps)
    return powerMap.clamp_min(0.0)


def _target_distribution(target: TargetSpec, eps: float = 1e-10) -> torch.Tensor:
    powerMap = _target_linear_map(target, eps=eps)
    return powerMap / powerMap.sum().clamp_min(eps)


def _target_support_map(target: TargetSpec, eps: float = 1e-10) -> torch.Tensor:
    powerMap = _target_linear_map(target, eps=eps)
    return powerMap / powerMap.max().clamp_min(eps)


def shapeMatchLoss(
    linearResponse: torch.Tensor,
    target: TargetSpec,
    lossType: LossType = "HUBER",
    delta: float = 1e-4,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    linearResponse: [B, ...] raw linear power
    returns: [B]
    """

    # --- flatten ---
    responseFlat = linearResponse.flatten(1)  # [B, N]
    targetFlat = _target_distribution(target, eps=eps).flatten().unsqueeze(0)  # [1, N]

    responseDist = responseFlat / responseFlat.sum(dim=1, keepdim=True).clamp_min(eps)
    cellCount = responseFlat.shape[1]

    weight = target.importanceMap.flatten().unsqueeze(0)  # [1, N]

    # --- error ---
    error = cellCount * (responseDist - targetFlat)

    # --- loss per element ---
    if lossType == "MSE":
        perCell = error.square()

    elif lossType == "HUBER":
        absError = error.abs()
        perCell = torch.where(
            absError <= delta,
            0.5 * error.square(),
            delta * (absError - 0.5 * delta),
        )
    else:
        raise ValueError(f"Unsupported Loss Type: {lossType}")

    # --- weighted average ---
    loss = (perCell * weight).sum(dim=1) / weight.sum().clamp_min(eps)

    return loss


def powerConcentration(
    linearResponse: torch.Tensor, target: TargetSpec, sigmaDeg: float = 2.0, eps: float = 1e-10
) -> torch.Tensor:
    """
    linearResponse: [B, ...] raw linear power
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

    hotspotMapFlat = hotspotMap.flatten().unsqueeze(0)  # [1, N]

    responseFlat = linearResponse.flatten(1)  # [B, N]
    weightedHotspotPower = (responseFlat * hotspotMapFlat).sum(dim=1)
    totalPower = responseFlat.sum(dim=1).clamp_min(eps)

    concentration = weightedHotspotPower / totalPower
    return concentration


def deliveredPower(linearResponse: torch.Tensor, target: TargetSpec) -> torch.Tensor:
    responseFlat = linearResponse.flatten(1)
    targetFlat = _target_distribution(target).flatten().unsqueeze(0)
    return (responseFlat * targetFlat).sum(dim=1)


def targetFraction(linearResponse: torch.Tensor, target: TargetSpec) -> torch.Tensor:
    responseFlat = linearResponse.flatten(1)
    supportFlat = torch.maximum(
        _target_support_map(target).flatten(),
        target.importanceMap.flatten().clamp_min(0.0),
    ).unsqueeze(0)
    weightedPower = (responseFlat * supportFlat).sum(dim=1)
    totalPower = linearResponse.flatten(1).sum(dim=1).clamp_min(1e-10)
    return weightedPower / totalPower


def spill(
    linearResponse: torch.Tensor,
    target: TargetSpec,
    thresholdRatio: float = 0.01,
    eps: float = 1e-10,
):
    responseFlat = linearResponse.flatten(1)
    importanceFlat = target.importanceMap.flatten()

    threshold = thresholdRatio * importanceFlat.max()
    mask = (importanceFlat >= threshold).to(responseFlat.dtype).unsqueeze(0)

    insidePower = (responseFlat * mask).sum(dim=1)
    totalPower = responseFlat.sum(dim=1).clamp_min(eps)
    outsideFraction = insidePower / totalPower
    return outsideFraction


def batchLoss(
    batch: ArrayBatch,
    target: TargetSpec,
    params: LossParameters,
    lossType: LossType = "MSE",
    logTerms: bool = False,
) -> torch.Tensor:
    target = target.to(batch.device, batch.dtype)
    targetAZEL = mapLLAtoArrayAZEL(batch, target.targetCoordinates)

    linearResponse = arrayResponseBatch(batch, targetAZEL)  # raw linear power

    shapeTerm = shapeMatchLoss(linearResponse, target, lossType)
    concentrationTerm = powerConcentration(linearResponse, target)
    deliveredPowerTerm = deliveredPower(linearResponse, target)
    targetFractionTerm = targetFraction(linearResponse, target)
    spillTerm = spill(linearResponse, target)
    if logTerms:
        print(
            f"shape={params.shapeK * shapeTerm.mean().item():.4f} | "
            f"concentration={params.concentrationK * -concentrationTerm.mean().item():.4f} | "
            f"delivered={params.deliveredK * -deliveredPowerTerm.mean().item():.4f} | "
            f"fraction={params.fractionK * -targetFractionTerm.mean().item():.4f} | "
            f"spill={params.spillK * -spillTerm.mean().item():.4f}"
        )
    loss = (
        shapeTerm * params.shapeK
        - params.concentrationK * concentrationTerm
        - params.deliveredK * deliveredPowerTerm
        - params.fractionK * targetFractionTerm
        - spillTerm * params.spillK
    )
    return loss
