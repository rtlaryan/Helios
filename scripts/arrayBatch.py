from dataclasses import dataclass

import torch


@dataclass
class ArrayBatch:
    elementLocalPosition: torch.Tensor  # [B, 3, N] (Real)
    weights: torch.Tensor  # [B, N] (Complex)
    wavelength: float
    gain: torch.Tensor  # [B] real
    LLAPosition: torch.Tensor  # [B, 3] = [latitude degrees, longitude degrees, altitude meters]
    ECEFPosition: torch.Tensor  # [B, 3] = [X, Y, Z] meters
    elementMask: torch.Tensor | None = None  # [B, N] bool (if failRate != 0 or sparsity is desired)

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
        mutatedAmplitude = mutatedAmplitude / mutatedAmplitude.norm(dim=1, keepdim=True)

        weights = mutatedAmplitude * torch.exp(1j * mutatedPhase)

        return ArrayBatch(
            elementLocalPosition=self.elementLocalPosition,
            weights=weights,
            wavelength=self.wavelength,
            gain=self.gain,
            LLAPosition=self.LLAPosition,
            ECEFPosition=self.ECEFPosition,
            elementMask=self.elementMask,
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
        }


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

    return ArrayBatch(
        elementLocalPosition=concatenateTensors("elementLocalPosition"),
        weights=concatenateTensors("weights"),
        wavelength=referenceBatch.wavelength,
        gain=concatenateTensors("gain"),
        LLAPosition=concatenateTensors("LLAPosition"),
        ECEFPosition=concatenateTensors("ECEFPosition"),
        elementMask=elementMasks,
    )
