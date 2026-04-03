from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TargetConstraints:
    latitudeRange: tuple[float, float] = (20, 50)  # deg
    longitudesRange: tuple[float, float] = (70, 120)  # deg
    zoneCount: int = 1


@dataclass
class TargetSpec:
    searchLatitudes: torch.Tensor  # [H, W] degrees
    searchLongitudes: torch.Tensor  # [H, W] degrees
    importanceMap: torch.Tensor  # [H, W] float
    powerMap: torch.Tensor  # [H, W] floats
    hotspotCoordinates: torch.Tensor  # [num hotspots, 2] lat/lon coordinates of each hotspot
    thresholdDB: float = 10

    @property
    def centerLatitude(self) -> float:
        return self.searchLatitudes.mean().item()

    @property
    def centerLongitude(self) -> float:
        return self.searchLongitudes.mean().item()

    @property
    def targetCoordinates(self) -> torch.Tensor:
        return torch.stack(
            [self.searchLatitudes.reshape(-1), self.searchLongitudes.reshape(-1)], dim=-1
        )

    @property
    def targetShape(self) -> torch.Size:
        return self.searchLatitudes.shape

    def to(self, device: torch.device, dtype: torch.dtype) -> TargetSpec:
        self.searchLatitudes = self.searchLatitudes.to(device=device, dtype=dtype)
        self.searchLongitudes = self.searchLongitudes.to(device=device, dtype=dtype)
        self.importanceMap = self.importanceMap.to(device=device, dtype=dtype)
        self.powerMap = self.powerMap.to(device=device, dtype=dtype)
        self.hotspotCoordinates = self.hotspotCoordinates.to(device=device, dtype=dtype)
        return self

    def clone(self) -> TargetSpec:
        return TargetSpec(
            searchLatitudes=self.searchLatitudes.clone(),
            searchLongitudes=self.searchLongitudes.clone(),
            importanceMap=self.importanceMap.clone(),
            powerMap=self.powerMap.clone(),
            hotspotCoordinates=self.hotspotCoordinates.clone(),
            thresholdDB=self.thresholdDB,
        )

    def decimate(self, decimationFactor: int = 4) -> TargetSpec:
        height, width = self.searchLatitudes.shape
        outHeight = max(1, (height + decimationFactor - 1) // decimationFactor)
        outWidth = max(1, (width + decimationFactor - 1) // decimationFactor)

        decimatedLatitudes = (
            F.adaptive_avg_pool2d(
                self.searchLatitudes.unsqueeze(0).unsqueeze(0), (outHeight, outWidth)
            )
            .squeeze(0)
            .squeeze(0)
        )
        decimatedLongitudes = (
            F.adaptive_avg_pool2d(
                self.searchLongitudes.unsqueeze(0).unsqueeze(0), (outHeight, outWidth)
            )
            .squeeze(0)
            .squeeze(0)
        )
        decimatedImportance = (
            F.adaptive_max_pool2d(
                self.importanceMap.unsqueeze(0).unsqueeze(0), (outHeight, outWidth)
            )
            .squeeze(0)
            .squeeze(0)
        )

        weightedPower = self.powerMap * self.importanceMap
        averagePower = (
            F.adaptive_avg_pool2d(weightedPower.unsqueeze(0).unsqueeze(0), (outHeight, outWidth))
            .squeeze(0)
            .squeeze(0)
        )
        averageImportance = (
            F.adaptive_avg_pool2d(
                self.importanceMap.unsqueeze(0).unsqueeze(0), (outHeight, outWidth)
            )
            .squeeze(0)
            .squeeze(0)
        )
        decimatedPower = averagePower / averageImportance.clamp_min(1e-12)

        return TargetSpec(
            decimatedLatitudes,
            decimatedLongitudes,
            decimatedImportance,
            decimatedPower,
            self.hotspotCoordinates.clone(),
            self.thresholdDB,
        )

    def serializeTargetSpec(self) -> dict:
        return {
            "searchLatitudes": self.searchLatitudes.detach().cpu(),
            "searchLongitudes": self.searchLongitudes.detach().cpu(),
            "importanceMap": self.importanceMap.detach().cpu(),
            "powerMap": self.powerMap.detach().cpu(),
            "hotspotCoordinates": self.hotspotCoordinates.detach().cpu(),
            "thresholdDB": self.thresholdDB,
        }

    @classmethod
    def fromSerializedTargetSpec(cls, payload: dict) -> TargetSpec:
        return cls(
            searchLatitudes=torch.as_tensor(payload["searchLatitudes"]),
            searchLongitudes=torch.as_tensor(payload["searchLongitudes"]),
            importanceMap=torch.as_tensor(payload["importanceMap"]),
            powerMap=torch.as_tensor(payload["powerMap"]),
            hotspotCoordinates=torch.as_tensor(payload["hotspotCoordinates"]),
            thresholdDB=float(payload.get("thresholdDB", 10.0)),
        )

    @classmethod
    def fromMapping(cls, payload: dict) -> TargetSpec:
        return cls.fromSerializedTargetSpec(payload)


@dataclass
class TargetBatch:
    searchLatitudes: torch.Tensor  # [B, H, W]
    searchLongitudes: torch.Tensor  # [B, H, W]
    importanceMap: torch.Tensor  # [B, H, W]
    powerMap: torch.Tensor  # [B, H, W]
    hotspotCoordinates: torch.Tensor  # [B, K, 2]
    thresholdDB: torch.Tensor | float = 10

    @property
    def batchSize(self) -> int:
        return int(self.searchLatitudes.shape[0])

    @property
    def targetShape(self) -> torch.Size:
        return self.searchLatitudes.shape[1:]

    @property
    def targetCoordinates(self) -> torch.Tensor:
        return torch.stack(
            [self.searchLatitudes.flatten(1), self.searchLongitudes.flatten(1)],
            dim=-1,
        )

    def to(self, device: torch.device, dtype: torch.dtype) -> TargetBatch:
        self.searchLatitudes = self.searchLatitudes.to(device=device, dtype=dtype)
        self.searchLongitudes = self.searchLongitudes.to(device=device, dtype=dtype)
        self.importanceMap = self.importanceMap.to(device=device, dtype=dtype)
        self.powerMap = self.powerMap.to(device=device, dtype=dtype)
        self.hotspotCoordinates = self.hotspotCoordinates.to(device=device, dtype=dtype)
        if isinstance(self.thresholdDB, torch.Tensor):
            self.thresholdDB = self.thresholdDB.to(device=device, dtype=dtype)
        else:
            self.thresholdDB = torch.tensor(self.thresholdDB, device=device, dtype=dtype)
        return self

    def fetch(self, idx: int | slice | torch.Tensor) -> TargetSpec | TargetBatch:
        if isinstance(idx, int):
            return TargetSpec(
                searchLatitudes=self.searchLatitudes[idx],
                searchLongitudes=self.searchLongitudes[idx],
                importanceMap=self.importanceMap[idx],
                powerMap=self.powerMap[idx],
                hotspotCoordinates=self.hotspotCoordinates[idx],
                thresholdDB=float(
                    self.thresholdDB[idx].item()
                    if isinstance(self.thresholdDB, torch.Tensor) and self.thresholdDB.ndim > 0
                    else self.thresholdDB.item()
                    if isinstance(self.thresholdDB, torch.Tensor)
                    else self.thresholdDB
                ),
            )

        threshold = self.thresholdDB
        if isinstance(threshold, torch.Tensor) and threshold.ndim > 0:
            threshold = threshold[idx]
        return TargetBatch(
            searchLatitudes=self.searchLatitudes[idx],
            searchLongitudes=self.searchLongitudes[idx],
            importanceMap=self.importanceMap[idx],
            powerMap=self.powerMap[idx],
            hotspotCoordinates=self.hotspotCoordinates[idx],
            thresholdDB=threshold,
        )

    def decimate(self, decimationFactor: int = 4) -> TargetBatch:
        height, width = self.targetShape
        outHeight = max(1, (height + decimationFactor - 1) // decimationFactor)
        outWidth = max(1, (width + decimationFactor - 1) // decimationFactor)
        return TargetBatch(
            searchLatitudes=F.adaptive_avg_pool2d(
                self.searchLatitudes.unsqueeze(1), (outHeight, outWidth)
            ).squeeze(1),
            searchLongitudes=F.adaptive_avg_pool2d(
                self.searchLongitudes.unsqueeze(1), (outHeight, outWidth)
            ).squeeze(1),
            importanceMap=F.adaptive_max_pool2d(
                self.importanceMap.unsqueeze(1), (outHeight, outWidth)
            ).squeeze(1),
            powerMap=F.adaptive_avg_pool2d(
                self.powerMap.unsqueeze(1), (outHeight, outWidth)
            ).squeeze(1),
            hotspotCoordinates=self.hotspotCoordinates.clone(),
            thresholdDB=self.thresholdDB.clone()
            if isinstance(self.thresholdDB, torch.Tensor)
            else self.thresholdDB,
        )

    def serializeTargetBatch(self) -> dict:
        return {
            "searchLatitudes": self.searchLatitudes.detach().cpu(),
            "searchLongitudes": self.searchLongitudes.detach().cpu(),
            "importanceMap": self.importanceMap.detach().cpu(),
            "powerMap": self.powerMap.detach().cpu(),
            "hotspotCoordinates": self.hotspotCoordinates.detach().cpu(),
            "thresholdDB": self.thresholdDB.detach().cpu()
            if isinstance(self.thresholdDB, torch.Tensor)
            else self.thresholdDB,
        }

    def serializeTargetSample(self, idx: int) -> dict:
        target = self.fetch(idx)
        assert isinstance(target, TargetSpec)
        return target.serializeTargetSpec()

    @classmethod
    def fromSerializedTargetBatch(cls, payload: dict) -> TargetBatch:
        threshold = payload.get("thresholdDB", 10.0)
        if isinstance(threshold, list):
            threshold = torch.as_tensor(threshold)
        elif isinstance(threshold, torch.Tensor):
            threshold = threshold
        return cls(
            searchLatitudes=torch.as_tensor(payload["searchLatitudes"]),
            searchLongitudes=torch.as_tensor(payload["searchLongitudes"]),
            importanceMap=torch.as_tensor(payload["importanceMap"]),
            powerMap=torch.as_tensor(payload["powerMap"]),
            hotspotCoordinates=torch.as_tensor(payload["hotspotCoordinates"]),
            thresholdDB=threshold,
        )

    @classmethod
    def fromTargetSpecs(cls, targets: list[TargetSpec]) -> TargetBatch:
        if not targets:
            raise ValueError("targets must not be empty")

        shapes = {tuple(target.targetShape) for target in targets}
        if len(shapes) != 1:
            raise ValueError("all targets in TargetBatch must share the same spatial shape")

        hotspotCounts = {int(target.hotspotCoordinates.shape[0]) for target in targets}
        if len(hotspotCounts) != 1:
            raise ValueError("all targets in TargetBatch must share the same hotspot count")

        return cls(
            searchLatitudes=torch.stack([target.searchLatitudes for target in targets], dim=0),
            searchLongitudes=torch.stack([target.searchLongitudes for target in targets], dim=0),
            importanceMap=torch.stack([target.importanceMap for target in targets], dim=0),
            powerMap=torch.stack([target.powerMap for target in targets], dim=0),
            hotspotCoordinates=torch.stack([target.hotspotCoordinates for target in targets], dim=0),
            thresholdDB=torch.tensor([target.thresholdDB for target in targets]),
        )


TargetLike: TypeAlias = TargetSpec | TargetBatch


def _sanitizeCenterWeights(weightMap: torch.Tensor) -> torch.Tensor:
    finiteWeights = torch.where(torch.isfinite(weightMap), weightMap, torch.zeros_like(weightMap))
    return finiteWeights.clamp_min(0.0)


def _powerMapToCenterWeights(powerMap: torch.Tensor) -> torch.Tensor:
    finitePower = torch.where(torch.isfinite(powerMap), powerMap, torch.full_like(powerMap, -120.0))
    if torch.any(finitePower < 0):
        return torch.pow(10.0, finitePower / 10.0).clamp_min(0.0)
    return finitePower.clamp_min(0.0)


def inferTargetCenter(target: TargetLike) -> torch.Tensor:
    primaryWeights = _sanitizeCenterWeights(target.importanceMap)
    fallbackWeights = _powerMapToCenterWeights(target.powerMap)

    if isinstance(target, TargetBatch):
        primarySum = primaryWeights.flatten(1).sum(dim=1)
        fallbackSum = fallbackWeights.flatten(1).sum(dim=1)
        usePrimary = primarySum > 0
        useFallback = (~usePrimary) & (fallbackSum > 0)
        centerWeights = torch.where(
            usePrimary.view(-1, 1, 1),
            primaryWeights,
            torch.where(useFallback.view(-1, 1, 1), fallbackWeights, torch.ones_like(primaryWeights)),
        )
        normalizer = centerWeights.flatten(1).sum(dim=1).clamp_min(1e-12)
        centerLatitude = (target.searchLatitudes * centerWeights).flatten(1).sum(dim=1) / normalizer
        centerLongitude = (target.searchLongitudes * centerWeights).flatten(1).sum(dim=1) / normalizer
        return torch.stack([centerLatitude, centerLongitude], dim=-1)

    if primaryWeights.sum() > 0:
        centerWeights = primaryWeights
    elif fallbackWeights.sum() > 0:
        centerWeights = fallbackWeights
    else:
        centerWeights = torch.ones_like(primaryWeights)

    normalizer = centerWeights.sum().clamp_min(1e-12)
    centerLatitude = (target.searchLatitudes * centerWeights).sum() / normalizer
    centerLongitude = (target.searchLongitudes * centerWeights).sum() / normalizer
    return torch.stack([centerLatitude, centerLongitude], dim=-1)


def serializeTarget(target: TargetLike) -> dict:
    if isinstance(target, TargetBatch):
        return target.serializeTargetBatch()
    return target.serializeTargetSpec()


def fetchTargetSample(target: TargetLike, idx: int) -> TargetSpec:
    if isinstance(target, TargetBatch):
        sample = target.fetch(idx)
        assert isinstance(sample, TargetSpec)
        return sample
    return target
