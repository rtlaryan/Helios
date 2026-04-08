from __future__ import annotations

import pytest
import torch

from scripts.arraySimulation import (
    _arrayResponseCoreReference,
    arrayResponseCore,
    chooseChunkShape,
)


def _make_response_inputs(
    device: torch.device,
    *,
    batched_grid: bool,
) -> dict[str, torch.Tensor | float]:
    generator = torch.Generator().manual_seed(1234)

    batch_size = 3
    element_count = 17
    grid_shape = (5, 7)

    element_local_position = torch.randn(
        (batch_size, 3, element_count),
        generator=generator,
        dtype=torch.float32,
    )
    weight_real = torch.randn(
        (batch_size, element_count),
        generator=generator,
        dtype=torch.float32,
    )
    weight_imag = torch.randn(
        (batch_size, element_count),
        generator=generator,
        dtype=torch.float32,
    )
    weights = torch.complex(weight_real, weight_imag)
    gain = torch.randn(
        (batch_size,),
        generator=generator,
        dtype=torch.float32,
    )
    element_local_position = element_local_position.to(device)
    weights = weights.to(device)
    gain = gain.to(device)

    azimuth_axis = torch.linspace(-0.4, 0.6, steps=grid_shape[0], device=device)
    elevation_axis = torch.linspace(-0.3, 0.5, steps=grid_shape[1], device=device)
    azimuth_shared, elevation_shared = torch.meshgrid(
        azimuth_axis,
        elevation_axis,
        indexing="ij",
    )

    if batched_grid:
        azimuth_offsets = torch.linspace(0.0, 0.08, steps=batch_size, device=device).view(
            batch_size, 1, 1
        )
        elevation_offsets = torch.linspace(
            0.02, 0.10, steps=batch_size, device=device
        ).view(batch_size, 1, 1)
        azimuth = azimuth_shared.unsqueeze(0) + azimuth_offsets
        elevation = elevation_shared.unsqueeze(0) - elevation_offsets
    else:
        azimuth = azimuth_shared
        elevation = elevation_shared

    return {
        "elementLocalPosition": element_local_position,
        "weights": weights,
        "wavelength": 0.31,
        "azimuth": azimuth,
        "elevation": elevation,
        "gain": gain,
    }


@pytest.mark.parametrize("batched_grid", [False, True], ids=["shared_grid", "batched_grid"])
@pytest.mark.parametrize("normalize,dB", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("chunk_size", [None, 37, 211, 1024])
def test_array_response_core_matches_reference(
    batched_grid: bool,
    normalize: bool,
    dB: bool,
    chunk_size: int | None,
) -> None:
    inputs = _make_response_inputs(torch.device("cpu"), batched_grid=batched_grid)

    expected = _arrayResponseCoreReference(
        **inputs,
        chunkSize=chunk_size,
        normalize=normalize,
        dB=dB,
    )
    actual = arrayResponseCore(
        **inputs,
        chunkSize=chunk_size,
        normalize=normalize,
        dB=dB,
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for parity check")
@pytest.mark.parametrize("chunk_size", [37, 211], ids=["batch_split", "full_batch"])
def test_array_response_core_matches_reference_on_cuda(chunk_size: int) -> None:
    inputs = _make_response_inputs(torch.device("cuda"), batched_grid=True)

    expected = _arrayResponseCoreReference(
        **inputs,
        chunkSize=chunk_size,
        normalize=True,
        dB=True,
    )
    actual = arrayResponseCore(
        **inputs,
        chunkSize=chunk_size,
        normalize=True,
        dB=True,
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_choose_chunk_shape_caps_reduction_tile() -> None:
    bc, nc, pc = chooseChunkShape(
        50,
        50_176,
        50_176,
        10_000_000,
        reductionTileCap=128,
    )

    assert 1 <= bc <= 50
    assert 1 <= nc <= 128
    assert 1 <= pc <= 50_176
    assert bc * nc * pc <= 10_000_000


def test_choose_chunk_shape_prefers_full_remaining_batch_when_it_fits() -> None:
    bc, nc, pc = chooseChunkShape(
        4,
        16,
        1_000,
        512,
        reductionTileCap=64,
    )

    assert nc == 16
    assert bc == 4
    assert pc == 8
    assert bc * nc * pc <= 512


def test_choose_chunk_shape_falls_back_to_batch_split_when_needed() -> None:
    bc, nc, pc = chooseChunkShape(
        50,
        1_000,
        10_000,
        4_096,
        reductionTileCap=128,
    )

    assert nc == 128
    assert bc < 50
    assert pc == 1
    assert bc * nc * pc <= 4_096
