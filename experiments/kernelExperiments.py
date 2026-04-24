from __future__ import annotations

import gc
from collections.abc import Callable, Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import torch
import yaml

from scripts.arrayBatch import ArrayBatch
from simulation.arraySim import arrayResponseBatchSharedGrid, normalizePower, todB
from simulation.arraySimV2 import (
    arrayResponseBatchSharedGridV2,
    arrayResponseSampleV2,
)

ArrayResponseFn = Callable[..., torch.Tensor]


def _synchronize_for_timing(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def clear_torch_memory(device: torch.device | str | None = None) -> None:
    gc.collect()
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            return
    else:
        device = torch.device(device)

    _synchronize_for_timing(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif device.type == "mps":
        torch.mps.empty_cache()
    _synchronize_for_timing(device)


def _as_angle_tensor(
    value: float | torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, device=device, dtype=dtype)


@torch.no_grad()
def dense_array_response(
    batch: ArrayBatch,
    azimuth: float | torch.Tensor,
    elevation: float | torch.Tensor,
    *,
    sample_id: int = 0,
    dB: bool = False,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Dense single-sample array response using the same trig steering math as the
    chunked implementation in simulation.arraySim.arrayResponseCore.

    Assumes the array sample of interest is batch index 0 by default.
    azimuth/elevation are radians and may be scalars, vectors, or broadcastable
    grids. The returned tensor has the broadcast az/el shape.
    """
    device = batch.device
    dtype = batch.dtype
    azimuth = _as_angle_tensor(azimuth, device=device, dtype=dtype)
    elevation = _as_angle_tensor(elevation, device=device, dtype=dtype)
    azimuth, elevation = torch.broadcast_tensors(azimuth, elevation)

    grid_shape = azimuth.shape
    wave_vector = (2.0 * torch.pi / batch.wavelength) * torch.stack(
        [
            torch.cos(elevation) * torch.cos(azimuth),
            torch.cos(elevation) * torch.sin(azimuth),
            torch.sin(elevation),
        ],
        dim=0,
    ).reshape(3, -1)  # [3, P]

    position = batch.elementLocalPosition[sample_id].transpose(0, 1)  # [N, 3]
    weights = batch.weights[sample_id]
    phase = position @ wave_vector  # [N, P]

    cos_phase = torch.cos(phase)
    sin_phase = torch.sin(phase)
    weight_real = weights.real.unsqueeze(1)
    weight_imag = weights.imag.unsqueeze(1)

    response_real = (weight_real * cos_phase + weight_imag * sin_phase).sum(dim=0)
    response_imag = (weight_real * sin_phase - weight_imag * cos_phase).sum(dim=0)
    response = (response_real.square() + response_imag.square()).reshape(grid_shape)

    if normalize:
        response = normalizePower(response.unsqueeze(0))[0]

    if dB:
        response = todB(response) + batch.gain[sample_id]

    return response


@torch.no_grad()
def array_response_core_v2(
    batch: ArrayBatch,
    azimuth: float | torch.Tensor,
    elevation: float | torch.Tensor,
    *,
    sample_id: int = 0,
    dB: bool = False,
    normalize: bool = False,
) -> torch.Tensor:
    return arrayResponseSampleV2(
        batch,
        sample_id,
        azimuth,
        elevation,
        dB=dB,
        normalize=normalize,
    )


def compare_response_tensors(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> dict[str, float | bool | tuple[int, ...]]:
    reference = reference.detach()
    candidate = candidate.detach()

    if reference.shape != candidate.shape:
        return {
            "shape_match": False,
            "reference_shape": tuple(reference.shape),
            "candidate_shape": tuple(candidate.shape),
            "allclose": False,
        }

    difference = candidate - reference
    abs_difference = difference.abs()
    abs_reference = reference.abs().clamp_min(atol)
    rel_difference = abs_difference / abs_reference

    return {
        "shape_match": True,
        "reference_shape": tuple(reference.shape),
        "candidate_shape": tuple(candidate.shape),
        "allclose": bool(torch.allclose(candidate, reference, atol=atol, rtol=rtol)),
        "atol": float(atol),
        "rtol": float(rtol),
        "max_abs": float(abs_difference.max().item()),
        "mean_abs": float(abs_difference.mean().item()),
        "rmse": float(torch.sqrt(abs_difference.square().mean()).item()),
        "max_rel": float(rel_difference.max().item()),
        "mean_rel": float(rel_difference.mean().item()),
    }


def compare_response_cores(
    batch: ArrayBatch,
    azimuth: float | torch.Tensor,
    elevation: float | torch.Tensor,
    *,
    candidate_fn: ArrayResponseFn = array_response_core_v2,
    reference_fn: ArrayResponseFn = dense_array_response,
    sample_id: int = 0,
    dB: bool = False,
    normalize: bool = False,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> dict[str, Any]:
    reference = reference_fn(
        batch,
        azimuth,
        elevation,
        sample_id=sample_id,
        dB=dB,
        normalize=normalize,
    )
    candidate = candidate_fn(
        batch,
        azimuth,
        elevation,
        sample_id=sample_id,
        dB=dB,
        normalize=normalize,
    )
    return {
        "reference": reference,
        "candidate": candidate,
        "metrics": compare_response_tensors(reference, candidate, atol=atol, rtol=rtol),
    }


def time_response_core(
    response_fn: ArrayResponseFn,
    batch: ArrayBatch,
    azimuth: float | torch.Tensor,
    elevation: float | torch.Tensor,
    *,
    sample_id: int = 0,
    dB: bool = False,
    normalize: bool = False,
    warmup: int = 3,
    repeat: int = 10,
) -> dict[str, Any]:
    device = batch.device
    output: torch.Tensor | None = None

    for _ in range(max(0, warmup)):
        output = response_fn(
            batch,
            azimuth,
            elevation,
            sample_id=sample_id,
            dB=dB,
            normalize=normalize,
        )
        _synchronize_for_timing(device)

    timings_ms: list[float] = []
    for _ in range(max(1, repeat)):
        _synchronize_for_timing(device)
        start = perf_counter()
        output = response_fn(
            batch,
            azimuth,
            elevation,
            sample_id=sample_id,
            dB=dB,
            normalize=normalize,
        )
        _synchronize_for_timing(device)
        timings_ms.append((perf_counter() - start) * 1000.0)

    timings = torch.tensor(timings_ms, dtype=torch.float64)
    return {
        "mean_ms": float(timings.mean().item()),
        "std_ms": float(timings.std(unbiased=False).item()),
        "min_ms": float(timings.min().item()),
        "max_ms": float(timings.max().item()),
        "timings_ms": timings_ms,
        "output": output,
    }


def compare_response_core_timings(
    batch: ArrayBatch,
    azimuth: float | torch.Tensor,
    elevation: float | torch.Tensor,
    *,
    reference_fn: ArrayResponseFn = dense_array_response,
    candidate_fn: ArrayResponseFn = array_response_core_v2,
    sample_id: int = 0,
    dB: bool = False,
    normalize: bool = False,
    warmup: int = 3,
    repeat: int = 10,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> dict[str, Any]:
    reference_timing = time_response_core(
        reference_fn,
        batch,
        azimuth,
        elevation,
        sample_id=sample_id,
        dB=dB,
        normalize=normalize,
        warmup=warmup,
        repeat=repeat,
    )
    candidate_timing = time_response_core(
        candidate_fn,
        batch,
        azimuth,
        elevation,
        sample_id=sample_id,
        dB=dB,
        normalize=normalize,
        warmup=warmup,
        repeat=repeat,
    )
    reference_output = reference_timing["output"]
    candidate_output = candidate_timing["output"]
    speedup = reference_timing["mean_ms"] / max(candidate_timing["mean_ms"], 1e-12)

    return {
        "reference": reference_timing,
        "candidate": candidate_timing,
        "speedup": float(speedup),
        "metrics": compare_response_tensors(
            reference_output,
            candidate_output,
            atol=atol,
            rtol=rtol,
        ),
    }


def make_benchmark_batch(
    *,
    batch_size: int = 50,
    element_count: int = 25_000,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    wavelength: float = 299_792_458.0 / 30e9,
    element_spacing_ratio: float = 0.5,
    random_weights: bool = True,
    seed: int = 123,
) -> ArrayBatch:
    device = torch.device(device)
    side_y = int(torch.ceil(torch.sqrt(torch.tensor(float(element_count)))).item())
    side_z = int(torch.ceil(torch.tensor(float(element_count / side_y))).item())
    spacing = wavelength * element_spacing_ratio

    y_axis = torch.linspace(
        -0.5 * spacing * (side_y - 1),
        0.5 * spacing * (side_y - 1),
        side_y,
        device=device,
        dtype=dtype,
    )
    z_axis = torch.linspace(
        -0.5 * spacing * (side_z - 1),
        0.5 * spacing * (side_z - 1),
        side_z,
        device=device,
        dtype=dtype,
    )
    y_grid, z_grid = torch.meshgrid(y_axis, z_axis, indexing="ij")
    y_flat = y_grid.flatten()[:element_count]
    z_flat = z_grid.flatten()[:element_count]
    x_flat = torch.zeros_like(y_flat)
    local_position = torch.stack([x_flat, y_flat, z_flat], dim=0)
    element_local_position = local_position.unsqueeze(0).expand(batch_size, -1, -1).clone()

    generator = torch.Generator(device=device).manual_seed(seed)
    if random_weights:
        amplitude = torch.rand(
            (batch_size, element_count),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        amplitude = amplitude / amplitude.norm(dim=1, keepdim=True).clamp_min(1e-12)
        phase = torch.empty((batch_size, element_count), device=device, dtype=dtype).uniform_(
            -torch.pi,
            torch.pi,
            generator=generator,
        )
    else:
        amplitude = torch.full(
            (batch_size, element_count),
            1.0 / element_count**0.5,
            device=device,
            dtype=dtype,
        )
        phase = torch.zeros((batch_size, element_count), device=device, dtype=dtype)

    weights = torch.polar(amplitude, phase)
    gain = torch.ones(batch_size, device=device, dtype=dtype)
    lla_position = torch.zeros((batch_size, 3), device=device, dtype=dtype)
    ecef_position = torch.zeros((batch_size, 3), device=device, dtype=dtype)
    return ArrayBatch(
        elementLocalPosition=element_local_position,
        weights=weights,
        wavelength=wavelength,
        gain=gain,
        LLAPosition=lla_position,
        ECEFPosition=ecef_position,
    )


def build_shared_az_el_grid(
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    azimuth_points: int = 256,
    elevation_points: int = 256,
    azimuth_range: tuple[float, float] = (-torch.pi, torch.pi),
    elevation_range: tuple[float, float] = (-torch.pi / 2, torch.pi / 2),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device(device)
    azimuth_axis = torch.linspace(
        float(azimuth_range[0]),
        float(azimuth_range[1]),
        azimuth_points,
        device=device,
        dtype=dtype,
    )
    elevation_axis = torch.linspace(
        float(elevation_range[0]),
        float(elevation_range[1]),
        elevation_points,
        device=device,
        dtype=dtype,
    )
    azimuth_grid, elevation_grid = torch.meshgrid(azimuth_axis, elevation_axis, indexing="ij")
    return azimuth_axis, elevation_axis, azimuth_grid, elevation_grid


def load_evolution_chunk_size(
    config_path: str | Path,
    *,
    key: str = "wideResponseChunkSize",
) -> int:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    try:
        value = config["evolution"][key]
    except KeyError as exc:
        raise KeyError(f"Missing evolution.{key} in {config_path}") from exc

    return int(value)


@torch.no_grad()
def triton_response_batch_shared_grid(
    batch: ArrayBatch,
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    *,
    dB: bool = False,
    normalize: bool = False,
) -> torch.Tensor:
    return arrayResponseBatchSharedGridV2(
        batch,
        (azimuth, elevation),
        dB=dB,
        normalize=normalize,
    )


def _time_callable(
    fn: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    output: torch.Tensor | None = None
    for _ in range(max(0, warmup)):
        output = fn()
        _synchronize_for_timing(device)

    timings_ms: list[float] = []
    for _ in range(max(1, repeat)):
        _synchronize_for_timing(device)
        start = perf_counter()
        output = fn()
        _synchronize_for_timing(device)
        timings_ms.append((perf_counter() - start) * 1000.0)

    timings = torch.tensor(timings_ms, dtype=torch.float64)
    return {
        "mean_ms": float(timings.mean().item()),
        "std_ms": float(timings.std(unbiased=False).item()),
        "min_ms": float(timings.min().item()),
        "max_ms": float(timings.max().item()),
        "timings_ms": timings_ms,
        "output": output,
    }


def compare_chunked_triton_batch_timings(
    batch: ArrayBatch,
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    *,
    chunk_size: int | None = 2_000_000,
    dB: bool = False,
    normalize: bool = False,
    warmup: int = 1,
    repeat: int = 3,
    compare_outputs: bool = True,
    clear_memory_between: bool = True,
    atol: float = 5e-3,
    rtol: float = 1e-4,
) -> dict[str, Any]:
    device = batch.device

    if clear_memory_between:
        clear_torch_memory(device)

    chunked_timing = _time_callable(
        lambda: arrayResponseBatchSharedGrid(
            batch,
            (azimuth, elevation),
            chunkSize=chunk_size,
            dB=dB,
            normalize=normalize,
        ),
        device=device,
        warmup=warmup,
        repeat=repeat,
    )

    if clear_memory_between:
        clear_torch_memory(device)

    triton_timing = _time_callable(
        lambda: triton_response_batch_shared_grid(
            batch,
            azimuth,
            elevation,
            dB=dB,
            normalize=normalize,
        ),
        device=device,
        warmup=warmup,
        repeat=repeat,
    )

    if clear_memory_between:
        clear_torch_memory(device)

    payload: dict[str, Any] = {
        "chunked": chunked_timing,
        "triton": triton_timing,
        "speedup": float(chunked_timing["mean_ms"] / max(triton_timing["mean_ms"], 1e-12)),
        "config": {
            "batch_size": batch.batchSize,
            "element_count": batch.N,
            "grid_shape": tuple(azimuth.shape),
            "chunk_size": chunk_size,
            "dB": dB,
            "normalize": normalize,
            "warmup": warmup,
            "repeat": repeat,
            "clear_memory_between": clear_memory_between,
        },
    }

    if compare_outputs:
        payload["metrics"] = compare_response_tensors(
            chunked_timing["output"],
            triton_timing["output"],
            atol=atol,
            rtol=rtol,
        )

    return payload


def build_cut_axes(
    batch: ArrayBatch,
    *,
    points: int = 721,
    azimuth_range: tuple[float, float] = (-torch.pi, torch.pi),
    elevation_range: tuple[float, float] = (-torch.pi / 2, torch.pi / 2),
) -> tuple[torch.Tensor, torch.Tensor]:
    azimuth_axis = torch.linspace(
        float(azimuth_range[0]),
        float(azimuth_range[1]),
        points,
        device=batch.device,
        dtype=batch.dtype,
    )
    elevation_axis = torch.linspace(
        float(elevation_range[0]),
        float(elevation_range[1]),
        points,
        device=batch.device,
        dtype=batch.dtype,
    )
    return azimuth_axis, elevation_axis


def az_el_cuts(
    batch: ArrayBatch,
    *,
    sample_id: int = 0,
    points: int = 721,
    fixed_azimuth: float = 0.0,
    fixed_elevation: float = 0.0,
    dB: bool = True,
    normalize: bool = True,
) -> dict[str, torch.Tensor]:
    azimuth_axis, elevation_axis = build_cut_axes(batch, points=points)
    azimuth_response = dense_array_response(
        batch,
        azimuth_axis,
        fixed_elevation,
        sample_id=sample_id,
        dB=dB,
        normalize=normalize,
    )
    elevation_response = dense_array_response(
        batch,
        fixed_azimuth,
        elevation_axis,
        sample_id=sample_id,
        dB=dB,
        normalize=normalize,
    )
    return {
        "azimuth_axis": azimuth_axis,
        "azimuth_response": azimuth_response,
        "elevation_axis": elevation_axis,
        "elevation_response": elevation_response,
    }


def dense_response_grid(
    batch: ArrayBatch,
    *,
    sample_id: int = 0,
    azimuth_points: int = 361,
    elevation_points: int = 181,
    dB: bool = True,
    normalize: bool = True,
) -> dict[str, torch.Tensor]:
    azimuth_axis = torch.linspace(
        -torch.pi,
        torch.pi,
        azimuth_points,
        device=batch.device,
        dtype=batch.dtype,
    )
    elevation_axis = torch.linspace(
        -torch.pi / 2,
        torch.pi / 2,
        elevation_points,
        device=batch.device,
        dtype=batch.dtype,
    )
    azimuth_grid, elevation_grid = torch.meshgrid(azimuth_axis, elevation_axis, indexing="ij")
    response = dense_array_response(
        batch,
        azimuth_grid,
        elevation_grid,
        sample_id=sample_id,
        dB=dB,
        normalize=normalize,
    )
    return {
        "azimuth_axis": azimuth_axis,
        "elevation_axis": elevation_axis,
        "response": response,
    }


def plot_az_el_cuts(
    cuts: Mapping[str, torch.Tensor],
    *,
    title: str = "Dense Array Response Cuts",
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)

    azimuth_deg = torch.rad2deg(cuts["azimuth_axis"]).detach().cpu()
    azimuth_response = cuts["azimuth_response"].detach().cpu()
    elevation_deg = torch.rad2deg(cuts["elevation_axis"]).detach().cpu()
    elevation_response = cuts["elevation_response"].detach().cpu()

    axes[0].plot(azimuth_deg, azimuth_response)
    axes[0].set_title("Azimuth Cut")
    axes[0].set_xlabel("Azimuth (deg)")
    axes[0].set_ylabel("Power (dB)")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(elevation_deg, elevation_response)
    axes[1].set_title("Elevation Cut")
    axes[1].set_xlabel("Elevation (deg)")
    axes[1].set_ylabel("Power (dB)")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(title)
    return fig


def plot_response_grid(
    grid: Mapping[str, torch.Tensor],
    *,
    title: str = "Dense Array Response",
    vmin: float = -40.0,
    vmax: float = 0.0,
) -> plt.Figure:
    azimuth_deg = torch.rad2deg(grid["azimuth_axis"]).detach().cpu()
    elevation_deg = torch.rad2deg(grid["elevation_axis"]).detach().cpu()
    response = grid["response"].detach().cpu()

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    image = ax.imshow(
        response.T,
        origin="lower",
        aspect="auto",
        extent=[
            float(azimuth_deg.min()),
            float(azimuth_deg.max()),
            float(elevation_deg.min()),
            float(elevation_deg.max()),
        ],
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    ax.set_title(title)
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")
    fig.colorbar(image, ax=ax, label="Power (dB)")
    return fig


def plot_response_comparison(
    comparison: Mapping[str, Any],
    grid: Mapping[str, torch.Tensor],
    *,
    title: str = "Response Core Comparison",
) -> plt.Figure:
    reference = comparison["reference"].detach().cpu()
    candidate = comparison["candidate"].detach().cpu()
    error = candidate - reference
    azimuth_deg = torch.rad2deg(grid["azimuth_axis"]).detach().cpu()
    elevation_deg = torch.rad2deg(grid["elevation_axis"]).detach().cpu()
    extent = [
        float(azimuth_deg.min()),
        float(azimuth_deg.max()),
        float(elevation_deg.min()),
        float(elevation_deg.max()),
    ]
    error_limit = max(float(error.abs().max().item()), 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for ax, data, panel_title in (
        (axes[0], reference, "Reference"),
        (axes[1], candidate, "Candidate"),
    ):
        image = ax.imshow(
            data.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
        )
        ax.set_title(panel_title)
        ax.set_xlabel("Azimuth (deg)")
        ax.set_ylabel("Elevation (deg)")
        fig.colorbar(image, ax=ax, pad=0.02)

    error_image = axes[2].imshow(
        error.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=-error_limit,
        vmax=error_limit,
        cmap="coolwarm",
    )
    axes[2].set_title("Candidate - Reference")
    axes[2].set_xlabel("Azimuth (deg)")
    axes[2].set_ylabel("Elevation (deg)")
    fig.colorbar(error_image, ax=axes[2], pad=0.02)
    fig.suptitle(title)
    return fig


def plot_outputs(
    batch: ArrayBatch,
    *,
    sample_id: int = 0,
    cut_points: int = 721,
    azimuth_points: int = 361,
    elevation_points: int = 181,
) -> dict[str, Any]:
    cuts = az_el_cuts(batch, sample_id=sample_id, points=cut_points)
    grid = dense_response_grid(
        batch,
        sample_id=sample_id,
        azimuth_points=azimuth_points,
        elevation_points=elevation_points,
    )
    return {
        "cuts": cuts,
        "grid": grid,
        "cut_figure": plot_az_el_cuts(cuts),
        "grid_figure": plot_response_grid(grid),
    }
