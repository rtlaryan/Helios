from dataclasses import dataclass
import hashlib

import torch


def _normalize_amplitudes(
    amplitude: torch.Tensor,
    elementMask: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    if elementMask is not None:
        amplitude = amplitude * elementMask.to(dtype=amplitude.dtype)
    norm = amplitude.norm(dim=1, keepdim=True).clamp_min(eps)
    return amplitude / norm


def _rows_are_identical(tensor: torch.Tensor) -> bool:
    if tensor.shape[0] <= 1:
        return True
    return bool(torch.equal(tensor, tensor[:1].expand_as(tensor)))


def _stable_tensor_digest(tensor: torch.Tensor) -> str:
    cpuTensor = tensor.detach().cpu().contiguous()
    return hashlib.blake2b(cpuTensor.numpy().tobytes(), digest_size=16).hexdigest()


def infer_geometry_cache_key(
    elementLocalPosition: torch.Tensor,
    LLAPosition: torch.Tensor,
    ECEFPosition: torch.Tensor,
    gain: torch.Tensor,
    wavelength: float,
    elementMask: torch.Tensor | None = None,
) -> tuple[int | float | str, ...] | None:
    if not (
        _rows_are_identical(elementLocalPosition)
        and _rows_are_identical(LLAPosition)
        and _rows_are_identical(ECEFPosition)
        and _rows_are_identical(gain)
        and (elementMask is None or _rows_are_identical(elementMask))
    ):
        return None

    key: list[int | float | str] = [
        "geometry_v1",
        tuple(elementLocalPosition.shape[1:]),
        str(elementLocalPosition.dtype),
        str(LLAPosition.dtype),
        float(wavelength),
        _stable_tensor_digest(elementLocalPosition[:1]),
        _stable_tensor_digest(LLAPosition[:1]),
        _stable_tensor_digest(ECEFPosition[:1]),
        _stable_tensor_digest(gain[:1]),
    ]
    if elementMask is None:
        key.append("mask:none")
    else:
        key.append(_stable_tensor_digest(elementMask[:1].to(dtype=torch.uint8)))
    return tuple(key)


@dataclass
class ArrayBatch:
    elementLocalPosition: torch.Tensor  # [B, 3, N] (Real)
    weights: torch.Tensor  # [B, N] (Complex)
    wavelength: float
    gain: torch.Tensor  # [B] real
    LLAPosition: torch.Tensor  # [B, 3] = [latitude degrees, longitude degrees, altitude meters]
    ECEFPosition: torch.Tensor  # [B, 3] = [X, Y, Z] meters
    elementMask: torch.Tensor | None = None  # [B, N] bool (if failRate != 0 or sparsity is desired)
    geometryCacheKey: tuple[int | float | str, ...] | None = None

    @property
    def N(self) -> int:
        return int(self.elementLocalPosition.shape[2])

    @property
    def K(self) -> float:
        return 2.0 * torch.pi / self.wavelength

    @property
    def batchSize(self) -> int:
        return int(self.elementLocalPosition.shape[0])

    @property
    def device(self) -> torch.device:
        return self.elementLocalPosition.device

    @property
    def dtype(self) -> torch.dtype:
        return self.elementLocalPosition.dtype

    def to(self, device: torch.device) -> "ArrayBatch":
        self.elementLocalPosition = self.elementLocalPosition.to(device)
        self.weights = self.weights.to(device)
        self.LLAPosition = self.LLAPosition.to(device)
        self.ECEFPosition = self.ECEFPosition.to(device)
        self.gain = self.gain.to(device)
        if self.elementMask is not None:
            self.elementMask = self.elementMask.to(device)
        return self

    def effective_weights(self) -> torch.Tensor:
        w = self.weights
        if self.elementMask is not None:
            w = w * self.elementMask.to(w.dtype)
        return w

    def fetch(self, i: int | slice | torch.Tensor = 1) -> "ArrayBatch":
        if isinstance(i, int):
            i = slice(i, i + 1)

        elementMask = None if self.elementMask is None else self.elementMask[i]

        return ArrayBatch(
            elementLocalPosition=self.elementLocalPosition[i],
            weights=self.weights[i],
            wavelength=self.wavelength,
            gain=self.gain[i],
            LLAPosition=self.LLAPosition[i],
            ECEFPosition=self.ECEFPosition[i],
            elementMask=elementMask,
            geometryCacheKey=self.geometryCacheKey,
        )

    def mutateWeights(
        self, phaseSigma: float, amplitudeSigma: float, generator: torch.Generator | None = None
    ) -> "ArrayBatch":
        phase = self.weights.angle()
        amplitude = self.weights.abs()

        phaseNoise = torch.randn_like(phase, generator=generator) * phaseSigma
        amplitudeNoise = torch.randn_like(amplitude, generator=generator) * amplitudeSigma

        mutatedPhase = phase + phaseNoise
        mutatedPhase = torch.remainder(mutatedPhase + torch.pi, 2 * torch.pi) - torch.pi

        mutatedAmplitude = amplitude * (1.0 + amplitudeNoise)
        mutatedAmplitude = mutatedAmplitude.clamp_min(0.0)
        mutatedAmplitude = _normalize_amplitudes(mutatedAmplitude, self.elementMask)

        weights = mutatedAmplitude * torch.exp(1j * mutatedPhase)

        return ArrayBatch(
            elementLocalPosition=self.elementLocalPosition,
            weights=weights,
            wavelength=self.wavelength,
            gain=self.gain,
            LLAPosition=self.LLAPosition,
            ECEFPosition=self.ECEFPosition,
            elementMask=self.elementMask,
            geometryCacheKey=self.geometryCacheKey,
        )

    def crossoverWeights(
        self,
        partner: "ArrayBatch",
        generator: torch.Generator | None = None,
    ) -> "ArrayBatch":
        if self.batchSize != partner.batchSize:
            raise ValueError("crossover batches must have the same batch size")
        if self.N != partner.N:
            raise ValueError("crossover batches must have the same element count")

        selectionMask = torch.rand(
            self.weights.shape,
            device=self.weights.device,
            generator=generator,
        ) < 0.5
        weights = torch.where(selectionMask, self.weights, partner.weights)
        normalizedAmplitude = _normalize_amplitudes(weights.abs(), self.elementMask)
        weights = normalizedAmplitude * torch.exp(1j * weights.angle())

        return ArrayBatch(
            elementLocalPosition=self.elementLocalPosition,
            weights=weights,
            wavelength=self.wavelength,
            gain=self.gain,
            LLAPosition=self.LLAPosition,
            ECEFPosition=self.ECEFPosition,
            elementMask=self.elementMask,
            geometryCacheKey=(
                self.geometryCacheKey
                if self.geometryCacheKey == partner.geometryCacheKey
                else None
            ),
        )

    def serializeBatch(self) -> dict:
        return {
            "elementLocalPosition": self.elementLocalPosition.detach().cpu(),
            "weights": self.weights.detach().cpu(),
            "wavelength": self.wavelength,
            "gain": self.gain.detach().cpu(),
            "LLAPosition": self.LLAPosition.detach().cpu(),
            "ECEFPosition": self.ECEFPosition.detach().cpu(),
            "elementMask": None if self.elementMask is None else self.elementMask.detach().cpu(),
            "geometryCacheKey": self.geometryCacheKey,
        }

    def serializeBatchSample(self, idx: int) -> dict:
        return {
            "elementLocalPosition": self.elementLocalPosition[idx].detach().cpu(),
            "weights": self.weights[idx].detach().cpu(),
            "wavelength": self.wavelength,
            "gain": self.gain[idx].detach().cpu(),
            "LLAPosition": self.LLAPosition[idx].detach().cpu(),
            "ECEFPosition": self.ECEFPosition[idx].detach().cpu(),
            "elementMask": None
            if self.elementMask is None
            else self.elementMask[idx].detach().cpu(),
            "geometryCacheKey": self.geometryCacheKey,
        }

    @classmethod
    def fromSerializedBatch(cls, payload: dict) -> "ArrayBatch":
        return cls(
            elementLocalPosition=payload["elementLocalPosition"],
            weights=payload["weights"],
            wavelength=payload["wavelength"],
            gain=payload["gain"],
            LLAPosition=payload["LLAPosition"],
            ECEFPosition=payload["ECEFPosition"],
            elementMask=payload["elementMask"],
            geometryCacheKey=payload.get("geometryCacheKey"),
        )


def merge(batches: list[ArrayBatch]) -> "ArrayBatch":
    referenceBatch = batches[0]

    def concatenateTensors(attr):
        tensors = [getattr(batch, attr) for batch in batches if getattr(batch, attr) is not None]
        return torch.cat(tensors, dim=0)

    for batch in batches:
        assert batch.device == referenceBatch.device
        assert batch.dtype == referenceBatch.dtype
        assert batch.N == referenceBatch.N
        assert batch.wavelength == referenceBatch.wavelength
        assert type(batch.elementMask) is type(referenceBatch.elementMask)

    if referenceBatch.elementMask is None:
        elementMasks = None
    else:
        elementMasks = concatenateTensors("elementMask")

    geometryCacheKey = referenceBatch.geometryCacheKey
    if geometryCacheKey is None or any(batch.geometryCacheKey != geometryCacheKey for batch in batches[1:]):
        geometryCacheKey = None

    return ArrayBatch(
        elementLocalPosition=concatenateTensors("elementLocalPosition"),
        weights=concatenateTensors("weights"),
        wavelength=referenceBatch.wavelength,
        gain=concatenateTensors("gain"),
        LLAPosition=concatenateTensors("LLAPosition"),
        ECEFPosition=concatenateTensors("ECEFPosition"),
        elementMask=elementMasks,
        geometryCacheKey=geometryCacheKey,
    )
