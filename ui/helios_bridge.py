"""
helios_bridge.py
Converts between REST JSON payloads and native Helios dataclasses.
All imports are from the existing Helios modules; nothing is copied.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
from scripts.arrayBatch import ArrayBatch
from scripts.arraySpec import ArraySpec
from scripts.coordinateTransforms import LLAtoECEF, mapLLAtoArrayAZEL
from simulation.arraySim import arrayResponseSample

# Make sure project root is on the path so `scripts` is importable
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ─────────────────────────────────────────────
# Antenna batch helpers
# ─────────────────────────────────────────────


def build_array_spec(params: dict) -> ArraySpec:
    """Build an ArraySpec from a flat params dict (from the REST request body)."""
    return ArraySpec(
        centerFrequency=float(params.get("frequency", 30e9)),
        elementSpacingRatio=float(params.get("spacing_ratio", 0.5)),
        allowedElementCount=(int(params.get("element_count", 64)),),
        allowedAspectRatio=(float(params.get("aspect_ratio", 1.0)),),
        geometry=params.get("geometry", "URA"),
        latitudeRange=(float(params.get("lat", 0.0)), float(params.get("lat", 0.0))),
        longitudeRange=(float(params.get("lon", 0.0)), float(params.get("lon", 0.0))),
        altitudeRange=(float(params.get("alt", 3.6e7)), float(params.get("alt", 3.6e7))),
    )


def batch_to_json(batch: ArrayBatch, sample_id: int = 0) -> dict:
    """Serialize one sample from an ArrayBatch to a JSON-friendly dict."""
    pos = batch.elementLocalPosition[sample_id] * 1000  # → mm
    x, y, z = pos.cpu().tolist()  # each is a list of N floats

    weights = batch.weights[sample_id].cpu()
    amp = weights.abs().tolist()
    phase = torch.angle(weights).tolist()

    lla = batch.LLAPosition[sample_id].cpu().tolist()

    return {
        "sample_id": sample_id,
        "element_count": batch.N,
        "positions_mm": {"x": x, "y": y, "z": z},
        "weights": {"amplitude": amp, "phase": phase},
        "lla": {"lat": lla[0], "lon": lla[1], "alt": lla[2]},
        "wavelength_m": batch.wavelength,
        "frequency_ghz": round(3e8 / batch.wavelength / 1e9, 3),
    }


# ─────────────────────────────────────────────
# Radiation pattern helpers
# ─────────────────────────────────────────────


def compute_pattern_2d(batch: ArrayBatch, sample_id: int = 0, resolution: int = 200) -> dict:
    """
    Compute az/el cuts and a downsampled 3D pattern for one batch sample.
    Returns a JSON-serialisable dict.
    """
    dev = batch.device
    dtype = batch.dtype

    az_axis = torch.linspace(-math.pi, math.pi, resolution * 2, device=dev, dtype=dtype)
    el_zero = torch.zeros(1, device=dev, dtype=dtype)
    az_response = (
        arrayResponseSample(
            batch,
            sample_id,
            az_axis,
            el_zero,
            dB=True,
            normalize=True,
        )
        .cpu()
        .tolist()
    )

    el_axis = torch.linspace(-math.pi / 2, math.pi / 2, resolution, device=dev, dtype=dtype)
    az_zero = torch.zeros(1, device=dev, dtype=dtype)
    el_response = (
        arrayResponseSample(
            batch,
            sample_id,
            az_zero,
            el_axis,
            dB=True,
            normalize=True,
        )
        .cpu()
        .tolist()
    )

    el_axis_display = el_axis

    # 3D surface (coarser grid for performance)
    res3d = min(resolution, 100)
    az_v = torch.linspace(-math.pi, math.pi, res3d * 2, device=dev, dtype=dtype)
    el_v = torch.linspace(-math.pi / 2, math.pi / 2, res3d, device=dev, dtype=dtype)
    az_g, el_g = torch.meshgrid(az_v, el_v, indexing="ij")

    raw = arrayResponseSample(batch, sample_id, az_g, el_g, dB=False)
    power = raw
    pmax = power.max().clamp_min(1e-12)
    norm = (power / pmax).clamp_min(1e-12)
    db_grid = (10.0 * torch.log10(norm)).cpu()
    floor_db = -40.0
    R = norm.sqrt().cpu()
    R[db_grid < floor_db] = 0.0

    az_np = az_g.cpu()
    el_np = el_g.cpu()
    X = (R * torch.cos(el_np) * torch.cos(az_np)).tolist()
    Y = (R * torch.cos(el_np) * torch.sin(az_np)).tolist()
    Z = (R * torch.sin(el_np)).tolist()
    C = db_grid.tolist()

    return {
        "az_axis_deg": torch.rad2deg(az_axis).cpu().tolist(),
        "az_response_db": az_response,
        "el_axis_deg": torch.rad2deg(el_axis_display).cpu().tolist(),
        "el_response_db": el_response,
        "pattern3d": {"X": X, "Y": Y, "Z": Z, "C": C},
    }


def compute_ground_projection(
    batch: ArrayBatch,
    sample_id: int,
    target_maps: dict,
    resolution_deg: float = 1.0,
) -> dict:
    """
    Project the radiation pattern onto the Earth surface for a given antenna position.
    Returns a lat/lon grid with dB values.
    """
    lat_vec = np.arange(-90.0, 90.0 + resolution_deg, resolution_deg, dtype=np.float32)
    lon_vec = np.arange(-180.0, 180.0 + resolution_deg, resolution_deg, dtype=np.float32)

    lat_t = torch.tensor(lat_vec, dtype=batch.dtype)
    lon_t = torch.tensor(lon_vec, dtype=batch.dtype)
    lon_g, lat_g = torch.meshgrid(lon_t, lat_t, indexing="xy")

    coords = torch.stack([lat_g.reshape(-1), lon_g.reshape(-1)], dim=-1)

    sub_batch = ArrayBatch(
        elementLocalPosition=batch.elementLocalPosition[sample_id : sample_id + 1],
        weights=batch.weights[sample_id : sample_id + 1],
        wavelength=batch.wavelength,
        gain=batch.gain[sample_id : sample_id + 1],
        LLAPosition=batch.LLAPosition[sample_id : sample_id + 1],
        ECEFPosition=batch.ECEFPosition[sample_id : sample_id + 1],
        elementMask=None,
    )

    az, el = mapLLAtoArrayAZEL(sub_batch, coords)
    response = arrayResponseSample(
        sub_batch,
        0,
        az[0],
        el[0],
        dB=True,
    )

    # Geometric horizon masking:
    # A point on Earth's surface is visible from the satellite iff the dot product
    # of the satellite's ECEF position and the target's ECEF position is positive.
    # Both vectors point outward from Earth's centre; a positive dot product means
    # they are in the same hemisphere and the target is not over the horizon.
    target_lla = torch.cat(
        [coords, torch.zeros(coords.shape[0], 1, dtype=batch.dtype)], dim=-1
    )  # [N, 3] with altitude = 0
    target_ecef = LLAtoECEF(target_lla)  # [N, 3]
    sat_ecef = sub_batch.ECEFPosition[0]  # [3]
    visibility = (target_ecef * sat_ecef).sum(dim=-1)
    below_horizon = visibility < 0
    response = response.clone()
    response[below_horizon] = -100.0

    H = len(lat_vec)
    W = len(lon_vec)
    # Flat index k → lat_t[k//W], lon_t[k%W], so natural reshape is [H, W].
    # Flip rows so row 0 = lat +90° (north), matching equirectangular UV
    # (Three.js sphere: row 0 of texture = top = north pole).
    db_map = response.reshape(H, W).cpu().tolist()

    return {
        "lat_vec": lat_vec.tolist(),
        "lon_vec": lon_vec.tolist(),
        "db_map": db_map,
        "shape": [H, W],
    }
