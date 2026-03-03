from dataclasses import dataclass
import torch

@dataclass
class ArrayBatch:
    elementLocalPosition: torch.Tensor #[B, 3, N] (Real)
    weights: torch.Tensor   #[B, N] (Complex)
    wavelength: float
    gain: torch.Tensor #[B] real
    LLAPosition: torch.Tensor #[B, 3] = [latitude degrees, longitude degrees, altitude meters]
    ECEFPosition: torch.Tensor #[B, 3] = [X, Y, Z] meters
    elementMask: torch.Tensor | None = None # [B, N] bool (if failRate != 0 or sparsity is desired)
    
    @property
    def N(self) -> int:
        return int(self.elementLocalPosition.shape[2])
    
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