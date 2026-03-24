# scripts/antennaSpec.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

ArrayType = Literal["URA", "UCA"]


@dataclass(frozen=True)
class ArraySpec:
    # Fixed Constants
    centerFrequency: float = 30e9
    elementSpacingRatio: float = 0.5  # element spacing ratio in wavelengths

    # Noise Parameters:
    positionJitterSTD: float = 0.0  # position noise
    phaseJitterSTD: float = 0.0
    failRate: float = 0.0

    # Array Constraints
    allowedElementCount: Sequence[int] = (
        64,
        144,
        256,
        400,
        576,
    )
    allowedAspectRatio: Sequence[float] = (1,)  # , 1 / 2, 1 / 4, 1 / 8)
    geometry: ArrayType = "URA"
    gainRange: tuple[float, float] = (1.0, 1.0)  # fixed gain
    sparsityRange: tuple[float, float] = (1.0, 1.0)  # active fraction

    # Position Constraints
    latitudeRange: tuple[float, float] = (0.0, 0.0)  # degrees
    longitudeRange: tuple[float, float] = (-83.0, -83.0)  # degrees
    altitudeRange: tuple[float, float] = (3.6e7, 3.6e7)  # meters

    @property
    def wavelength(self) -> float:
        c = 299_792_458.0
        return c / self.centerFrequency

    @property
    def elementSpacing(self) -> float:
        return self.wavelength * self.elementSpacingRatio

    def serializeArraySpec(self) -> dict:
        return {
            "centerFrequency": self.centerFrequency,
            "elementSpacingRatio": self.elementSpacingRatio,
            "positionJitterSTD": self.positionJitterSTD,
            "phaseJitterSTD": self.phaseJitterSTD,
            "failRate": self.failRate,
            "allowedElementCount": tuple(self.allowedElementCount),
            "allowedAspectRatio": tuple(self.allowedAspectRatio),
            "geometry": self.geometry,
            "gainRange": tuple(self.gainRange),
            "sparsityRange": tuple(self.sparsityRange),
            "latitudeRange": tuple(self.latitudeRange),
            "longitudeRange": tuple(self.longitudeRange),
            "altitudeRange": tuple(self.altitudeRange),
            "wavelength": self.wavelength,
            "elementSpacing": self.elementSpacing,
        }
