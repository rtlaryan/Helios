from __future__ import annotations

from dataclasses import dataclass

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

    def decimate(self, decimationFactor: int = 4) -> TargetSpec:
        height, width = self.searchLatitudes.shape
        outHeight = max(1, (height + decimationFactor - 1) // decimationFactor)
        outWidth = max(1, (width + decimationFactor - 1) // decimationFactor)

        # Average Pooling
        decimatedLatitudes = (
            F.adaptive_avg_pool2d(
                self.searchLatitudes.unsqueeze(0).unsqueeze(0), (outHeight, outWidth)
            )
            .squeeze(0)
            .squeeze(0)
        )
        decimatedLongtitudes = (
            F.adaptive_avg_pool2d(
                self.searchLongitudes.unsqueeze(0).unsqueeze(0), (outHeight, outWidth)
            )
            .squeeze(0)
            .squeeze(0)
        )

        # max pooling
        decimatedImportance = (
            F.adaptive_max_pool2d(
                self.importanceMap.unsqueeze(0).unsqueeze(0), (outHeight, outWidth)
            )
            .squeeze(0)
            .squeeze(0)
        )

        # importance weighted pooling
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
        decimatedPower = averagePower / averageImportance

        return TargetSpec(
            decimatedLatitudes,
            decimatedLongtitudes,
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
