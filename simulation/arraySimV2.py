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
    if azimuth.ndim > 1 and azimuth.shape[0] == batch.batchSize and batch.batchSize > 1:
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


@triton.jit
def _array_response_trig_forward_autograd_kernel(
    position_ptr,
    weights_ri_ptr,
    azimuth_ptr,
    elevation_ptr,
    output_ptr,
    response_real_ptr,
    response_imag_ptr,
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
    tl.store(response_real_ptr + point_offsets, response_real, mask=point_mask)
    tl.store(response_imag_ptr + point_offsets, response_imag, mask=point_mask)


@triton.jit
def _array_response_trig_backward_element_kernel(
    position_ptr,
    weights_ri_ptr,
    azimuth_ptr,
    elevation_ptr,
    grad_output_ptr,
    response_real_ptr,
    response_imag_ptr,
    grad_position_ptr,
    grad_weights_ri_ptr,
    wave_number: tl.constexpr,
    element_count: tl.constexpr,
    point_count: tl.constexpr,
    block_points: tl.constexpr,
    block_elements: tl.constexpr,
) -> None:
    element_offsets = tl.program_id(0) * block_elements + tl.arange(0, block_elements)
    element_mask = element_offsets < element_count

    x = tl.load(position_ptr + element_offsets, mask=element_mask, other=0.0)
    y = tl.load(position_ptr + element_count + element_offsets, mask=element_mask, other=0.0)
    z = tl.load(
        position_ptr + 2 * element_count + element_offsets,
        mask=element_mask,
        other=0.0,
    )
    weight_real = tl.load(weights_ri_ptr + 2 * element_offsets, mask=element_mask, other=0.0)
    weight_imag = tl.load(
        weights_ri_ptr + 2 * element_offsets + 1,
        mask=element_mask,
        other=0.0,
    )

    grad_x = tl.zeros((block_elements,), dtype=tl.float32)
    grad_y = tl.zeros((block_elements,), dtype=tl.float32)
    grad_z = tl.zeros((block_elements,), dtype=tl.float32)
    grad_weight_real = tl.zeros((block_elements,), dtype=tl.float32)
    grad_weight_imag = tl.zeros((block_elements,), dtype=tl.float32)

    for point_start in range(0, point_count, block_points):
        point_offsets = point_start + tl.arange(0, block_points)
        point_mask = point_offsets < point_count

        azimuth = tl.load(azimuth_ptr + point_offsets, mask=point_mask, other=0.0)
        elevation = tl.load(elevation_ptr + point_offsets, mask=point_mask, other=0.0)
        upstream = tl.load(grad_output_ptr + point_offsets, mask=point_mask, other=0.0)
        response_real = tl.load(
            response_real_ptr + point_offsets,
            mask=point_mask,
            other=0.0,
        )
        response_imag = tl.load(
            response_imag_ptr + point_offsets,
            mask=point_mask,
            other=0.0,
        )

        cos_elevation = tl.cos(elevation)
        wave_x = wave_number * cos_elevation * tl.cos(azimuth)
        wave_y = wave_number * cos_elevation * tl.sin(azimuth)
        wave_z = wave_number * tl.sin(elevation)

        phase = (
            x[:, None] * wave_x[None, :]
            + y[:, None] * wave_y[None, :]
            + z[:, None] * wave_z[None, :]
        )
        cos_phase = tl.cos(phase)
        sin_phase = tl.sin(phase)

        alpha = 2.0 * upstream * response_real
        beta = 2.0 * upstream * response_imag

        grad_weight_real += tl.sum(
            alpha[None, :] * cos_phase + beta[None, :] * sin_phase,
            axis=1,
        )
        grad_weight_imag += tl.sum(
            alpha[None, :] * sin_phase - beta[None, :] * cos_phase,
            axis=1,
        )

        phase_grad = (
            alpha[None, :] * (-weight_real[:, None] * sin_phase + weight_imag[:, None] * cos_phase)
            + beta[None, :] * (weight_real[:, None] * cos_phase + weight_imag[:, None] * sin_phase)
        )
        grad_x += tl.sum(phase_grad * wave_x[None, :], axis=1)
        grad_y += tl.sum(phase_grad * wave_y[None, :], axis=1)
        grad_z += tl.sum(phase_grad * wave_z[None, :], axis=1)

    tl.store(grad_position_ptr + element_offsets, grad_x, mask=element_mask)
    tl.store(grad_position_ptr + element_count + element_offsets, grad_y, mask=element_mask)
    tl.store(
        grad_position_ptr + 2 * element_count + element_offsets,
        grad_z,
        mask=element_mask,
    )
    tl.store(grad_weights_ri_ptr + 2 * element_offsets, grad_weight_real, mask=element_mask)
    tl.store(
        grad_weights_ri_ptr + 2 * element_offsets + 1,
        grad_weight_imag,
        mask=element_mask,
    )


@triton.jit
def _array_response_trig_backward_angle_kernel(
    position_ptr,
    weights_ri_ptr,
    azimuth_ptr,
    elevation_ptr,
    grad_output_ptr,
    response_real_ptr,
    response_imag_ptr,
    grad_azimuth_ptr,
    grad_elevation_ptr,
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
    upstream = tl.load(grad_output_ptr + point_offsets, mask=point_mask, other=0.0)
    response_real = tl.load(response_real_ptr + point_offsets, mask=point_mask, other=0.0)
    response_imag = tl.load(response_imag_ptr + point_offsets, mask=point_mask, other=0.0)

    sin_azimuth = tl.sin(azimuth)
    cos_azimuth = tl.cos(azimuth)
    sin_elevation = tl.sin(elevation)
    cos_elevation = tl.cos(elevation)

    wave_x = wave_number * cos_elevation * cos_azimuth
    wave_y = wave_number * cos_elevation * sin_azimuth
    wave_z = wave_number * sin_elevation

    grad_azimuth = tl.zeros((block_points,), dtype=tl.float32)
    grad_elevation = tl.zeros((block_points,), dtype=tl.float32)
    alpha = 2.0 * upstream * response_real
    beta = 2.0 * upstream * response_imag

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
        phase_grad = (
            alpha[None, :] * (-weight_real[:, None] * sin_phase + weight_imag[:, None] * cos_phase)
            + beta[None, :] * (weight_real[:, None] * cos_phase + weight_imag[:, None] * sin_phase)
        )

        dphase_dazimuth = wave_number * cos_elevation[None, :] * (
            -x[:, None] * sin_azimuth[None, :] + y[:, None] * cos_azimuth[None, :]
        )
        dphase_delevation = wave_number * (
            -x[:, None] * sin_elevation[None, :] * cos_azimuth[None, :]
            - y[:, None] * sin_elevation[None, :] * sin_azimuth[None, :]
            + z[:, None] * cos_elevation[None, :]
        )
        grad_azimuth += tl.sum(phase_grad * dphase_dazimuth, axis=0)
        grad_elevation += tl.sum(phase_grad * dphase_delevation, axis=0)

    tl.store(grad_azimuth_ptr + point_offsets, grad_azimuth, mask=point_mask)
    tl.store(grad_elevation_ptr + point_offsets, grad_elevation, mask=point_mask)


@triton.jit
def _array_response_trig_batch_kernel(
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
    batch_id = tl.program_id(1)
    point_offsets = tl.program_id(0) * block_points + tl.arange(0, block_points)
    point_mask = point_offsets < point_count

    position_base = batch_id * 3 * element_count
    weights_base = batch_id * 2 * element_count
    output_base = batch_id * point_count

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

        x = tl.load(position_ptr + position_base + element_offsets, mask=element_mask, other=0.0)
        y = tl.load(
            position_ptr + position_base + element_count + element_offsets,
            mask=element_mask,
            other=0.0,
        )
        z = tl.load(
            position_ptr + position_base + 2 * element_count + element_offsets,
            mask=element_mask,
            other=0.0,
        )
        weight_real = tl.load(
            weights_ri_ptr + weights_base + 2 * element_offsets,
            mask=element_mask,
            other=0.0,
        )
        weight_imag = tl.load(
            weights_ri_ptr + weights_base + 2 * element_offsets + 1,
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
    tl.store(output_ptr + output_base + point_offsets, response, mask=point_mask)


@triton.jit
def _array_response_trig_batch_forward_autograd_kernel(
    position_ptr,
    weights_ri_ptr,
    azimuth_ptr,
    elevation_ptr,
    output_ptr,
    response_real_ptr,
    response_imag_ptr,
    wave_number: tl.constexpr,
    element_count: tl.constexpr,
    point_count: tl.constexpr,
    block_points: tl.constexpr,
    block_elements: tl.constexpr,
) -> None:
    batch_id = tl.program_id(1)
    point_offsets = tl.program_id(0) * block_points + tl.arange(0, block_points)
    point_mask = point_offsets < point_count

    position_base = batch_id * 3 * element_count
    weights_base = batch_id * 2 * element_count
    output_base = batch_id * point_count

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

        x = tl.load(position_ptr + position_base + element_offsets, mask=element_mask, other=0.0)
        y = tl.load(
            position_ptr + position_base + element_count + element_offsets,
            mask=element_mask,
            other=0.0,
        )
        z = tl.load(
            position_ptr + position_base + 2 * element_count + element_offsets,
            mask=element_mask,
            other=0.0,
        )
        weight_real = tl.load(
            weights_ri_ptr + weights_base + 2 * element_offsets,
            mask=element_mask,
            other=0.0,
        )
        weight_imag = tl.load(
            weights_ri_ptr + weights_base + 2 * element_offsets + 1,
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
    tl.store(output_ptr + output_base + point_offsets, response, mask=point_mask)
    tl.store(response_real_ptr + output_base + point_offsets, response_real, mask=point_mask)
    tl.store(response_imag_ptr + output_base + point_offsets, response_imag, mask=point_mask)


@triton.jit
def _array_response_trig_batch_backward_element_kernel(
    position_ptr,
    weights_ri_ptr,
    azimuth_ptr,
    elevation_ptr,
    grad_output_ptr,
    response_real_ptr,
    response_imag_ptr,
    grad_position_ptr,
    grad_weights_ri_ptr,
    wave_number: tl.constexpr,
    element_count: tl.constexpr,
    point_count: tl.constexpr,
    block_points: tl.constexpr,
    block_elements: tl.constexpr,
) -> None:
    batch_id = tl.program_id(1)
    element_offsets = tl.program_id(0) * block_elements + tl.arange(0, block_elements)
    element_mask = element_offsets < element_count

    position_base = batch_id * 3 * element_count
    weights_base = batch_id * 2 * element_count
    output_base = batch_id * point_count

    x = tl.load(position_ptr + position_base + element_offsets, mask=element_mask, other=0.0)
    y = tl.load(
        position_ptr + position_base + element_count + element_offsets,
        mask=element_mask,
        other=0.0,
    )
    z = tl.load(
        position_ptr + position_base + 2 * element_count + element_offsets,
        mask=element_mask,
        other=0.0,
    )
    weight_real = tl.load(
        weights_ri_ptr + weights_base + 2 * element_offsets,
        mask=element_mask,
        other=0.0,
    )
    weight_imag = tl.load(
        weights_ri_ptr + weights_base + 2 * element_offsets + 1,
        mask=element_mask,
        other=0.0,
    )

    grad_x = tl.zeros((block_elements,), dtype=tl.float32)
    grad_y = tl.zeros((block_elements,), dtype=tl.float32)
    grad_z = tl.zeros((block_elements,), dtype=tl.float32)
    grad_weight_real = tl.zeros((block_elements,), dtype=tl.float32)
    grad_weight_imag = tl.zeros((block_elements,), dtype=tl.float32)

    for point_start in range(0, point_count, block_points):
        point_offsets = point_start + tl.arange(0, block_points)
        point_mask = point_offsets < point_count

        azimuth = tl.load(azimuth_ptr + point_offsets, mask=point_mask, other=0.0)
        elevation = tl.load(elevation_ptr + point_offsets, mask=point_mask, other=0.0)
        upstream = tl.load(
            grad_output_ptr + output_base + point_offsets,
            mask=point_mask,
            other=0.0,
        )
        response_real = tl.load(
            response_real_ptr + output_base + point_offsets,
            mask=point_mask,
            other=0.0,
        )
        response_imag = tl.load(
            response_imag_ptr + output_base + point_offsets,
            mask=point_mask,
            other=0.0,
        )

        cos_elevation = tl.cos(elevation)
        wave_x = wave_number * cos_elevation * tl.cos(azimuth)
        wave_y = wave_number * cos_elevation * tl.sin(azimuth)
        wave_z = wave_number * tl.sin(elevation)

        phase = (
            x[:, None] * wave_x[None, :]
            + y[:, None] * wave_y[None, :]
            + z[:, None] * wave_z[None, :]
        )
        cos_phase = tl.cos(phase)
        sin_phase = tl.sin(phase)

        alpha = 2.0 * upstream * response_real
        beta = 2.0 * upstream * response_imag

        grad_weight_real += tl.sum(
            alpha[None, :] * cos_phase + beta[None, :] * sin_phase,
            axis=1,
        )
        grad_weight_imag += tl.sum(
            alpha[None, :] * sin_phase - beta[None, :] * cos_phase,
            axis=1,
        )

        phase_grad = (
            alpha[None, :] * (-weight_real[:, None] * sin_phase + weight_imag[:, None] * cos_phase)
            + beta[None, :] * (weight_real[:, None] * cos_phase + weight_imag[:, None] * sin_phase)
        )
        grad_x += tl.sum(phase_grad * wave_x[None, :], axis=1)
        grad_y += tl.sum(phase_grad * wave_y[None, :], axis=1)
        grad_z += tl.sum(phase_grad * wave_z[None, :], axis=1)

    tl.store(grad_position_ptr + position_base + element_offsets, grad_x, mask=element_mask)
    tl.store(
        grad_position_ptr + position_base + element_count + element_offsets,
        grad_y,
        mask=element_mask,
    )
    tl.store(
        grad_position_ptr + position_base + 2 * element_count + element_offsets,
        grad_z,
        mask=element_mask,
    )
    tl.store(
        grad_weights_ri_ptr + weights_base + 2 * element_offsets,
        grad_weight_real,
        mask=element_mask,
    )
    tl.store(
        grad_weights_ri_ptr + weights_base + 2 * element_offsets + 1,
        grad_weight_imag,
        mask=element_mask,
    )


@triton.jit
def _array_response_trig_batch_backward_shared_angle_kernel(
    position_ptr,
    weights_ri_ptr,
    azimuth_ptr,
    elevation_ptr,
    grad_output_ptr,
    response_real_ptr,
    response_imag_ptr,
    grad_azimuth_ptr,
    grad_elevation_ptr,
    wave_number: tl.constexpr,
    element_count: tl.constexpr,
    point_count: tl.constexpr,
    block_points: tl.constexpr,
    block_elements: tl.constexpr,
) -> None:
    batch_id = tl.program_id(1)
    point_offsets = tl.program_id(0) * block_points + tl.arange(0, block_points)
    point_mask = point_offsets < point_count

    position_base = batch_id * 3 * element_count
    weights_base = batch_id * 2 * element_count
    output_base = batch_id * point_count

    azimuth = tl.load(azimuth_ptr + point_offsets, mask=point_mask, other=0.0)
    elevation = tl.load(elevation_ptr + point_offsets, mask=point_mask, other=0.0)
    upstream = tl.load(
        grad_output_ptr + output_base + point_offsets,
        mask=point_mask,
        other=0.0,
    )
    response_real = tl.load(
        response_real_ptr + output_base + point_offsets,
        mask=point_mask,
        other=0.0,
    )
    response_imag = tl.load(
        response_imag_ptr + output_base + point_offsets,
        mask=point_mask,
        other=0.0,
    )

    sin_azimuth = tl.sin(azimuth)
    cos_azimuth = tl.cos(azimuth)
    sin_elevation = tl.sin(elevation)
    cos_elevation = tl.cos(elevation)

    wave_x = wave_number * cos_elevation * cos_azimuth
    wave_y = wave_number * cos_elevation * sin_azimuth
    wave_z = wave_number * sin_elevation

    grad_azimuth = tl.zeros((block_points,), dtype=tl.float32)
    grad_elevation = tl.zeros((block_points,), dtype=tl.float32)
    alpha = 2.0 * upstream * response_real
    beta = 2.0 * upstream * response_imag

    for element_start in range(0, element_count, block_elements):
        element_offsets = element_start + tl.arange(0, block_elements)
        element_mask = element_offsets < element_count

        x = tl.load(position_ptr + position_base + element_offsets, mask=element_mask, other=0.0)
        y = tl.load(
            position_ptr + position_base + element_count + element_offsets,
            mask=element_mask,
            other=0.0,
        )
        z = tl.load(
            position_ptr + position_base + 2 * element_count + element_offsets,
            mask=element_mask,
            other=0.0,
        )
        weight_real = tl.load(
            weights_ri_ptr + weights_base + 2 * element_offsets,
            mask=element_mask,
            other=0.0,
        )
        weight_imag = tl.load(
            weights_ri_ptr + weights_base + 2 * element_offsets + 1,
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
        phase_grad = (
            alpha[None, :] * (-weight_real[:, None] * sin_phase + weight_imag[:, None] * cos_phase)
            + beta[None, :] * (weight_real[:, None] * cos_phase + weight_imag[:, None] * sin_phase)
        )

        dphase_dazimuth = wave_number * cos_elevation[None, :] * (
            -x[:, None] * sin_azimuth[None, :] + y[:, None] * cos_azimuth[None, :]
        )
        dphase_delevation = wave_number * (
            -x[:, None] * sin_elevation[None, :] * cos_azimuth[None, :]
            - y[:, None] * sin_elevation[None, :] * sin_azimuth[None, :]
            + z[:, None] * cos_elevation[None, :]
        )
        grad_azimuth += tl.sum(phase_grad * dphase_dazimuth, axis=0)
        grad_elevation += tl.sum(phase_grad * dphase_delevation, axis=0)

    tl.atomic_add(grad_azimuth_ptr + point_offsets, grad_azimuth, mask=point_mask)
    tl.atomic_add(
        grad_elevation_ptr + point_offsets,
        grad_elevation,
        mask=point_mask,
    )


class _ArrayResponseSampleFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        position: torch.Tensor,
        weights: torch.Tensor,
        azimuth_flat: torch.Tensor,
        elevation_flat: torch.Tensor,
        wavelength: float,
    ) -> torch.Tensor:
        position = position.contiguous()
        weights = weights.contiguous()
        azimuth_flat = azimuth_flat.contiguous()
        elevation_flat = elevation_flat.contiguous()
        weights_ri = torch.view_as_real(weights).contiguous()
        point_count = azimuth_flat.numel()
        element_count = position.shape[1]
        block_points = 128
        block_elements = 64

        output_flat = torch.empty(point_count, device=position.device, dtype=position.dtype)
        response_real = torch.empty_like(output_flat)
        response_imag = torch.empty_like(output_flat)
        _array_response_trig_forward_autograd_kernel[(triton.cdiv(point_count, block_points),)](
            position,
            weights_ri,
            azimuth_flat,
            elevation_flat,
            output_flat,
            response_real,
            response_imag,
            float(2.0 * torch.pi / wavelength),
            element_count,
            point_count,
            block_points,
            block_elements,
            num_warps=4,
        )

        ctx.save_for_backward(
            position,
            weights,
            azimuth_flat,
            elevation_flat,
            response_real,
            response_imag,
        )
        ctx.wavelength = float(wavelength)
        ctx.element_count = element_count
        ctx.point_count = point_count
        ctx.block_points = block_points
        ctx.block_elements = block_elements
        return output_flat

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        position, weights, azimuth_flat, elevation_flat, response_real, response_imag = (
            ctx.saved_tensors
        )
        grad_output = grad_output.contiguous()
        weights_ri = torch.view_as_real(weights).contiguous()
        wave_number = float(2.0 * torch.pi / ctx.wavelength)
        needs_position, needs_weights, needs_azimuth, needs_elevation, _ = ctx.needs_input_grad

        grad_position = None
        grad_weights = None
        if needs_position or needs_weights:
            grad_position_buffer = torch.empty_like(position)
            grad_weights_ri = torch.empty_like(weights_ri)
            _array_response_trig_backward_element_kernel[
                (triton.cdiv(ctx.element_count, ctx.block_elements),)
            ](
                position,
                weights_ri,
                azimuth_flat,
                elevation_flat,
                grad_output,
                response_real,
                response_imag,
                grad_position_buffer,
                grad_weights_ri,
                wave_number,
                ctx.element_count,
                ctx.point_count,
                ctx.block_points,
                ctx.block_elements,
                num_warps=4,
            )
            if needs_position:
                grad_position = grad_position_buffer
            if needs_weights:
                grad_weights = torch.view_as_complex(grad_weights_ri)

        grad_azimuth = None
        grad_elevation = None
        if needs_azimuth or needs_elevation:
            grad_azimuth_buffer = torch.empty_like(azimuth_flat)
            grad_elevation_buffer = torch.empty_like(elevation_flat)
            _array_response_trig_backward_angle_kernel[
                (triton.cdiv(ctx.point_count, ctx.block_points),)
            ](
                position,
                weights_ri,
                azimuth_flat,
                elevation_flat,
                grad_output,
                response_real,
                response_imag,
                grad_azimuth_buffer,
                grad_elevation_buffer,
                wave_number,
                ctx.element_count,
                ctx.point_count,
                ctx.block_points,
                ctx.block_elements,
                num_warps=4,
            )
            if needs_azimuth:
                grad_azimuth = grad_azimuth_buffer
            if needs_elevation:
                grad_elevation = grad_elevation_buffer

        return grad_position, grad_weights, grad_azimuth, grad_elevation, None


def _array_response_sample_flat_no_grad(
    position: torch.Tensor,
    weights: torch.Tensor,
    azimuth_flat: torch.Tensor,
    elevation_flat: torch.Tensor,
    wavelength: float,
) -> torch.Tensor:
    position = position.contiguous()
    weights_ri = torch.view_as_real(weights.contiguous()).contiguous()
    point_count = azimuth_flat.numel()
    output_flat = torch.empty(point_count, device=position.device, dtype=position.dtype)

    block_points = 128
    block_elements = 64
    _array_response_trig_kernel[(triton.cdiv(point_count, block_points),)](
        position,
        weights_ri,
        azimuth_flat,
        elevation_flat,
        output_flat,
        float(2.0 * torch.pi / wavelength),
        position.shape[1],
        point_count,
        block_points,
        block_elements,
        num_warps=4,
    )
    return output_flat


class _ArrayResponseBatchSharedGridFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        position: torch.Tensor,
        weights: torch.Tensor,
        azimuth_flat: torch.Tensor,
        elevation_flat: torch.Tensor,
        wavelength: float,
    ) -> torch.Tensor:
        position = position.contiguous()
        weights = weights.contiguous()
        azimuth_flat = azimuth_flat.contiguous()
        elevation_flat = elevation_flat.contiguous()
        weights_ri = torch.view_as_real(weights).contiguous()
        batch_size = position.shape[0]
        point_count = azimuth_flat.numel()
        element_count = position.shape[2]
        block_points = 128
        block_elements = 64

        output_flat = torch.empty(
            (batch_size, point_count),
            device=position.device,
            dtype=position.dtype,
        )
        response_real = torch.empty_like(output_flat)
        response_imag = torch.empty_like(output_flat)
        grid = (triton.cdiv(point_count, block_points), batch_size)
        _array_response_trig_batch_forward_autograd_kernel[grid](
            position,
            weights_ri,
            azimuth_flat,
            elevation_flat,
            output_flat,
            response_real,
            response_imag,
            float(2.0 * torch.pi / wavelength),
            element_count,
            point_count,
            block_points,
            block_elements,
            num_warps=4,
        )

        ctx.save_for_backward(
            position,
            weights,
            azimuth_flat,
            elevation_flat,
            response_real,
            response_imag,
        )
        ctx.wavelength = float(wavelength)
        ctx.batch_size = batch_size
        ctx.element_count = element_count
        ctx.point_count = point_count
        ctx.block_points = block_points
        ctx.block_elements = block_elements
        return output_flat

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        position, weights, azimuth_flat, elevation_flat, response_real, response_imag = (
            ctx.saved_tensors
        )
        grad_output = grad_output.contiguous()
        weights_ri = torch.view_as_real(weights).contiguous()
        wave_number = float(2.0 * torch.pi / ctx.wavelength)
        needs_position, needs_weights, needs_azimuth, needs_elevation, _ = ctx.needs_input_grad

        grad_position = None
        grad_weights = None
        if needs_position or needs_weights:
            grad_position_buffer = torch.empty_like(position)
            grad_weights_ri = torch.empty_like(weights_ri)
            _array_response_trig_batch_backward_element_kernel[
                (triton.cdiv(ctx.element_count, ctx.block_elements), ctx.batch_size)
            ](
                position,
                weights_ri,
                azimuth_flat,
                elevation_flat,
                grad_output,
                response_real,
                response_imag,
                grad_position_buffer,
                grad_weights_ri,
                wave_number,
                ctx.element_count,
                ctx.point_count,
                ctx.block_points,
                ctx.block_elements,
                num_warps=4,
            )
            if needs_position:
                grad_position = grad_position_buffer
            if needs_weights:
                grad_weights = torch.view_as_complex(grad_weights_ri)

        grad_azimuth = None
        grad_elevation = None
        if needs_azimuth or needs_elevation:
            grad_azimuth_buffer = torch.zeros_like(azimuth_flat)
            grad_elevation_buffer = torch.zeros_like(elevation_flat)
            _array_response_trig_batch_backward_shared_angle_kernel[
                (triton.cdiv(ctx.point_count, ctx.block_points), ctx.batch_size)
            ](
                position,
                weights_ri,
                azimuth_flat,
                elevation_flat,
                grad_output,
                response_real,
                response_imag,
                grad_azimuth_buffer,
                grad_elevation_buffer,
                wave_number,
                ctx.element_count,
                ctx.point_count,
                ctx.block_points,
                ctx.block_elements,
                num_warps=4,
            )
            if needs_azimuth:
                grad_azimuth = grad_azimuth_buffer
            if needs_elevation:
                grad_elevation = grad_elevation_buffer

        return grad_position, grad_weights, grad_azimuth, grad_elevation, None


def _array_response_batch_shared_grid_flat_no_grad(
    position: torch.Tensor,
    weights: torch.Tensor,
    azimuth_flat: torch.Tensor,
    elevation_flat: torch.Tensor,
    wavelength: float,
) -> torch.Tensor:
    position = position.contiguous()
    weights_ri = torch.view_as_real(weights.contiguous()).contiguous()
    batch_size = position.shape[0]
    point_count = azimuth_flat.numel()
    output_flat = torch.empty(
        (batch_size, point_count),
        device=position.device,
        dtype=position.dtype,
    )

    block_points = 128
    block_elements = 64
    _array_response_trig_batch_kernel[(triton.cdiv(point_count, block_points), batch_size)](
        position,
        weights_ri,
        azimuth_flat,
        elevation_flat,
        output_flat,
        float(2.0 * torch.pi / wavelength),
        position.shape[2],
        point_count,
        block_points,
        block_elements,
        num_warps=4,
    )
    return output_flat


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

    position = batch.elementLocalPosition[sampleID]
    weights = batch.weights[sampleID]
    if torch.is_grad_enabled() and (
        position.requires_grad
        or weights.requires_grad
        or azimuth_flat.requires_grad
        or elevation_flat.requires_grad
    ):
        output_flat = _ArrayResponseSampleFunction.apply(
            position,
            weights,
            azimuth_flat,
            elevation_flat,
            batch.wavelength,
        )
    else:
        output_flat = _array_response_sample_flat_no_grad(
            position,
            weights,
            azimuth_flat,
            elevation_flat,
            batch.wavelength,
        )

    response = output_flat.reshape(grid_shape)
    if normalize:
        response = normalizePower(response.unsqueeze(0))[0]

    if dB:
        response = todB(response) + batch.gain[sampleID]

    return response


def arrayResponseBatchSharedGridV2(
    batch: ArrayBatch,
    relativeTargetAZEL: tuple[torch.Tensor, torch.Tensor],
    *,
    dB: bool = False,
    normalize: bool = False,
) -> torch.Tensor:
    azimuth, elevation = torch.broadcast_tensors(relativeTargetAZEL[0], relativeTargetAZEL[1])
    _validate_v2_inputs(batch, azimuth, elevation)

    grid_shape = azimuth.shape
    azimuth_flat = azimuth.contiguous().flatten()
    elevation_flat = elevation.contiguous().flatten()
    position = batch.elementLocalPosition
    weights = batch.weights

    if torch.is_grad_enabled() and (
        position.requires_grad
        or weights.requires_grad
        or azimuth_flat.requires_grad
        or elevation_flat.requires_grad
    ):
        output_flat = _ArrayResponseBatchSharedGridFunction.apply(
            position,
            weights,
            azimuth_flat,
            elevation_flat,
            batch.wavelength,
        )
    else:
        output_flat = _array_response_batch_shared_grid_flat_no_grad(
            position,
            weights,
            azimuth_flat,
            elevation_flat,
            batch.wavelength,
        )

    response = output_flat.reshape(batch.batchSize, *grid_shape)
    if normalize:
        response = normalizePower(response)

    if dB:
        gain_view = batch.gain.view(-1, *([1] * (response.ndim - 1)))
        response = todB(response) + gain_view

    return response
