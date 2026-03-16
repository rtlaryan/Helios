from __future__ import annotations

from dataclasses import dataclass

import torch


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
    powerMap: torch.Tensor  # [H, W] bool/int
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
