from __future__ import annotations

import torch
import triton
import triton.language as tl

from scripts.arrayBatch import ArrayBatch
from simulation.arraySim import normalizePower, todB


def _as_angle_tensor(
    value: float | torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, device=device, dtype=dtype)


def _validate_v2_inputs(
    batch: ArrayBatch,
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
) -> None:
    if batch.device.type != "cuda":
        raise RuntimeError("simulation.backend='v2' requires a CUDA ArrayBatch")
    if batch.dtype != torch.float32:
        raise RuntimeError("simulation.backend='v2' requires float32 array tensors")
    if batch.weights.dtype != torch.complex64:
        raise RuntimeError("simulation.backend='v2' requires complex64 weights")
    if azimuth.device.type != "cuda" or elevation.device.type != "cuda":
        raise RuntimeError("simulation.backend='v2' requires CUDA azimuth/elevation tensors")
    if azimuth.dtype != torch.float32 or elevation.dtype != torch.float32:
        raise RuntimeError("simulation.backend='v2' requires float32 azimuth/elevation tensors")
    if azimuth.ndim > 0 and azimuth.shape[0] == batch.batchSize and batch.batchSize > 1:
        raise ValueError(
            "simulation.backend='v2' supports shared az/el grids only, not batched grids"
        )


@triton.jit
def _array_response_trig_kernel(
    position_ptr,
    weights_ri_ptr,
    azimuth_ptr,
    elevation_ptr,
    output_ptr,
    wave_number: tl.constexpr,
    element_count: tl.constexpr,
    point_count: tl.constexpr,
    block_points: tl.constexpr,
    block_elements: tl.constexpr,
) -> None:
    point_offsets = tl.program_id(0) * block_points + tl.arange(0, block_points)
    point_mask = point_offsets < point_count

    azimuth = tl.load(azimuth_ptr + point_offsets, mask=point_mask, other=0.0)
    elevation = tl.load(elevation_ptr + point_offsets, mask=point_mask, other=0.0)

    cos_elevation = tl.cos(elevation)
    wave_x = wave_number * cos_elevation * tl.cos(azimuth)
    wave_y = wave_number * cos_elevation * tl.sin(azimuth)
    wave_z = wave_number * tl.sin(elevation)

    response_real = tl.zeros((block_points,), dtype=tl.float32)
    response_imag = tl.zeros((block_points,), dtype=tl.float32)

    for element_start in range(0, element_count, block_elements):
        element_offsets = element_start + tl.arange(0, block_elements)
        element_mask = element_offsets < element_count

        x = tl.load(position_ptr + element_offsets, mask=element_mask, other=0.0)
        y = tl.load(
            position_ptr + element_count + element_offsets,
            mask=element_mask,
            other=0.0,
        )
        z = tl.load(
            position_ptr + 2 * element_count + element_offsets,
            mask=element_mask,
            other=0.0,
        )
        weight_real = tl.load(
            weights_ri_ptr + 2 * element_offsets,
            mask=element_mask,
            other=0.0,
        )
        weight_imag = tl.load(
            weights_ri_ptr + 2 * element_offsets + 1,
            mask=element_mask,
            other=0.0,
        )

        phase = (
            x[:, None] * wave_x[None, :]
            + y[:, None] * wave_y[None, :]
            + z[:, None] * wave_z[None, :]
        )
        cos_phase = tl.cos(phase)
        sin_phase = tl.sin(phase)

        response_real += tl.sum(
            weight_real[:, None] * cos_phase + weight_imag[:, None] * sin_phase,
            axis=0,
        )
        response_imag += tl.sum(
            weight_real[:, None] * sin_phase - weight_imag[:, None] * cos_phase,
            axis=0,
        )

    response = response_real * response_real + response_imag * response_imag
    tl.store(output_ptr + point_offsets, response, mask=point_mask)


@torch.no_grad()
def arrayResponseSampleV2(
    batch: ArrayBatch,
    sampleID: int,
    azimuth: float | torch.Tensor,
    elevation: float | torch.Tensor,
    *,
    dB: bool = False,
    normalize: bool = False,
) -> torch.Tensor:
    azimuth = _as_angle_tensor(azimuth, device=batch.device, dtype=batch.dtype)
    elevation = _as_angle_tensor(elevation, device=batch.device, dtype=batch.dtype)
    azimuth, elevation = torch.broadcast_tensors(azimuth, elevation)
    _validate_v2_inputs(batch, azimuth, elevation)

    grid_shape = azimuth.shape
    azimuth_flat = azimuth.contiguous().flatten()
    elevation_flat = elevation.contiguous().flatten()
    point_count = azimuth_flat.numel()

    position = batch.elementLocalPosition[sampleID].contiguous()
    weights_ri = torch.view_as_real(batch.weights[sampleID].contiguous()).contiguous()
    output_flat = torch.empty(point_count, device=batch.device, dtype=batch.dtype)

    block_points = 128
    block_elements = 64
    _array_response_trig_kernel[(triton.cdiv(point_count, block_points),)](
        position,
        weights_ri,
        azimuth_flat,
        elevation_flat,
        output_flat,
        float(2.0 * torch.pi / batch.wavelength),
        position.shape[1],
        point_count,
        block_points,
        block_elements,
        num_warps=4,
    )

    response = output_flat.reshape(grid_shape)
    if normalize:
        response = normalizePower(response.unsqueeze(0))[0]

    if dB:
        response = todB(response) + batch.gain[sampleID]

    return response


@torch.no_grad()
def arrayResponseBatchSharedGridV2(
    batch: ArrayBatch,
    relativeTargetAZEL: tuple[torch.Tensor, torch.Tensor],
    *,
    dB: bool = False,
    normalize: bool = False,
) -> torch.Tensor:
    azimuth, elevation = torch.broadcast_tensors(relativeTargetAZEL[0], relativeTargetAZEL[1])
    _validate_v2_inputs(batch, azimuth, elevation)
    responses = [
        arrayResponseSampleV2(
            batch,
            sampleID,
            azimuth,
            elevation,
            dB=dB,
            normalize=normalize,
        )
        for sampleID in range(batch.batchSize)
    ]
    return torch.stack(responses, dim=0)
