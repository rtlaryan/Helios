"""
helios_bridge.py
Converts between REST JSON payloads and native Helios dataclasses.
All imports are from the existing scripts/ package — nothing is copied.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scripts.arrayBatch import ArrayBatch
from scripts.arraySimulation import arrayResponseCore, arrayResponseSample
from scripts.arraySpec import ArraySpec
from scripts.coordinateTransforms import LLAtoECEF, mapLLAtoArrayAZEL
from scripts.targetSpec import TargetSpec

# Make sure project root is on the path so `scripts` is importable
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ─────────────────────────────────────────────
# Target map generation
# ─────────────────────────────────────────────


def _gaussian_blob(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    center_lat: float,
    center_lon: float,
    radius_deg: float,
    sigma_deg: float,
    peak_linear: float,
) -> np.ndarray:
    """Return a flat-top blob: uniform inside radius_deg, Gaussian rolloff outside."""
    dlat = lat_grid - center_lat
    dlon = (lon_grid - center_lon) * np.cos(np.radians(center_lat))
    dist = np.sqrt(dlat**2 + dlon**2)
    effective_dist = np.maximum(0.0, dist - radius_deg)
    return peak_linear * np.exp(-(effective_dist**2) / (2 * max(sigma_deg, 0.01) ** 2))


def _point_in_polygon_vectorized(
    lat_flat: np.ndarray,
    lon_flat: np.ndarray,
    verts_lat: np.ndarray,
    verts_lon: np.ndarray,
) -> np.ndarray:
    """Vectorized even-odd ray-casting point-in-polygon test."""
    n = len(verts_lat)
    inside = np.zeros(len(lat_flat), dtype=bool)
    j = n - 1
    for i in range(n):
        yi, xi = verts_lat[i], verts_lon[i]
        yj, xj = verts_lat[j], verts_lon[j]
        cond1 = (yi > lat_flat) != (yj > lat_flat)
        cond2 = lon_flat < (xj - xi) * (lat_flat - yi) / (yj - yi + 1e-12) + xi
        inside ^= cond1 & cond2
        j = i
    return inside


def _min_dist_to_polygon_edges(
    lat_flat: np.ndarray,
    lon_flat: np.ndarray,
    verts_lat: np.ndarray,
    verts_lon: np.ndarray,
) -> np.ndarray:
    """Minimum distance (approx degrees) from each point to the polygon boundary."""
    n = len(verts_lat)
    min_d = np.full(len(lat_flat), np.inf, dtype=np.float32)
    for i in range(n):
        j = (i + 1) % n
        # Use mid-edge latitude for cosine correction
        clat = (verts_lat[i] + verts_lat[j]) / 2.0
        cos_c = np.cos(np.radians(clat))
        ax = verts_lon[i] * cos_c
        ay = float(verts_lat[i])
        bx = verts_lon[j] * cos_c
        by = float(verts_lat[j])
        px = lon_flat * cos_c
        py = lat_flat
        abx, aby = bx - ax, by - ay
        ab2 = abx**2 + aby**2 + 1e-20
        t = np.clip(((px - ax) * abx + (py - ay) * aby) / ab2, 0.0, 1.0)
        dx = px - (ax + t * abx)
        dy = py - (ay + t * aby)
        d = np.sqrt(dx**2 + dy**2)
        min_d = np.minimum(min_d, d)
    return min_d


def _polygon_blob(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    verts: list[dict],
    sigma_deg: float,
    peak_linear: float,
) -> np.ndarray:
    """
    Rasterize a polygon zone:
    - Points INSIDE the polygon  → peak_linear
    - Points OUTSIDE             → peak_linear * exp(-d² / 2σ²)
      where d = distance to the nearest polygon edge
    """
    verts_lat = np.array([v["lat"] for v in verts], dtype=np.float32)
    verts_lon = np.array([v["lon"] for v in verts], dtype=np.float32)

    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()

    inside = _point_in_polygon_vectorized(lat_flat, lon_flat, verts_lat, verts_lon)
    dist = _min_dist_to_polygon_edges(lat_flat, lon_flat, verts_lat, verts_lon)

    # Inside → dist=0 (full peak), outside → Gaussian rolloff
    effective_dist = np.where(inside, 0.0, dist)
    blob = peak_linear * np.exp(-(effective_dist**2) / (2 * sigma_deg**2))
    return blob.reshape(lat_grid.shape).astype(np.float32)


def build_target_maps(
    zones: list[dict],
    resolution_deg: float = 0.5,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
    normalize: bool = False,
) -> dict[str, Any]:
    """
    Build power and importance maps from a list of zone dicts.

    Zone shapes:
      circle  — {"shape":"circle", "lat", "lon", "radius_deg", "peak_db", "rolloff", "type"}
      polygon — {"shape":"polygon", "verts":[{"lat","lon"},...], "peak_db", "rolloff", "type"}

    lat_range / lon_range constrain the output grid (Focus Map).
    """
    lat_min, lat_max = lat_range if lat_range else (-90.0, 90.0)
    lon_min, lon_max = lon_range if lon_range else (-180.0, 180.0)

    lat_vec = np.arange(lat_min, lat_max + resolution_deg, resolution_deg, dtype=np.float32)
    lon_vec = np.arange(lon_min, lon_max + resolution_deg, resolution_deg, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lon_vec, lat_vec)  # [H, W]

    H, W = lat_grid.shape
    power_lin = np.zeros((H, W), dtype=np.float32)
    importance = np.zeros((H, W), dtype=np.float32)

    for zone in zones:
        shape = zone.get("shape", "circle")
        sigma = max(float(zone.get("rolloff", 1.0)), 0.01)

        if shape == "polygon":
            verts = zone.get("verts", [])
            if len(verts) < 3:
                continue
            if zone.get("type") == "power":
                peak_db = float(zone.get("peak_db", 0.0))
                peak_lin = 10 ** (peak_db / 10.0)
                blob = _polygon_blob(lat_grid, lon_grid, verts, sigma, peak_lin)
                power_lin = np.maximum(power_lin, blob)
            else:
                blob = _polygon_blob(lat_grid, lon_grid, verts, sigma, 1.0)
                importance = np.maximum(importance, blob)

        else:  # circle (default)
            clat = float(zone["lat"])
            clon = float(zone["lon"])
            r_deg = float(zone.get("radius_deg", 5.0))
            if zone.get("type") == "power":
                peak_db = float(zone.get("peak_db", 0.0))
                peak_lin = 10 ** (peak_db / 10.0)
                blob = _gaussian_blob(lat_grid, lon_grid, clat, clon, r_deg, sigma, peak_lin)
                power_lin = np.maximum(power_lin, blob)
            else:
                blob = _gaussian_blob(lat_grid, lon_grid, clat, clon, r_deg, sigma, 1.0)
                importance = np.maximum(importance, blob)

    # Normalise importance to [0, 1]
    imp_max = importance.max()
    if imp_max > 0:
        importance = importance / imp_max

    if normalize:
        # Output linear [0, 1] power map — no dB conversion
        pwr_max = power_lin.max()
        if pwr_max > 0:
            power_out = (power_lin / pwr_max).astype(np.float32)
        else:
            power_out = power_lin.astype(np.float32)
        return {
            "lat_vec": lat_vec.tolist(),
            "lon_vec": lon_vec.tolist(),
            "power_map": power_out.tolist(),
            "importance_map": importance.tolist(),
            "shape": [H, W],
            "power_normalized": True,
        }

    # Convert power to dB (floor at -100 dB)
    floor_lin = 10 ** (-100 / 10.0)
    power_lin_safe = np.maximum(power_lin, floor_lin)
    power_db = 10.0 * np.log10(power_lin_safe)
    no_signal = power_lin < floor_lin
    power_db[no_signal] = -100.0

    return {
        "lat_vec": lat_vec.tolist(),
        "lon_vec": lon_vec.tolist(),
        "power_map": power_db.tolist(),
        "importance_map": importance.tolist(),
        "shape": [H, W],
        "power_normalized": False,
    }


def build_target_spec(maps: dict, zones: list[dict], device: str = "cpu") -> TargetSpec:
    """Convert the output of build_target_maps into a TargetSpec."""
    lat_vec = torch.tensor(maps["lat_vec"])
    lon_vec = torch.tensor(maps["lon_vec"])
    lon_grid, lat_grid = torch.meshgrid(lon_vec, lat_vec, indexing="xy")  # [H, W]

    power_map = torch.tensor(maps["power_map"], dtype=torch.float32)
    importance_map = torch.tensor(maps["importance_map"], dtype=torch.float32)

    # Compute hotspot coordinates from power zones
    hotspots = []
    for z in zones:
        if z.get("type") == "power":
            if z.get("shape") == "polygon" and "verts" in z:
                verts = z["verts"]
                if not verts:
                    continue
                clat = sum(v["lat"] for v in verts) / len(verts)
                clon = sum(v["lon"] for v in verts) / len(verts)
                hotspots.append([clat, clon])
            else:
                hotspots.append([float(z["lat"]), float(z["lon"])])

    if not hotspots:
        # Fallback: place a single hotspot at the origin if no power zones defined
        hotspots = [[0.0, 0.0]]

    hotspot_tensor = torch.tensor(hotspots, dtype=torch.float32)

    return TargetSpec(
        searchLatitudes=lat_grid,
        searchLongitudes=lon_grid,
        importanceMap=importance_map,
        powerMap=power_map,
        hotspotCoordinates=hotspot_tensor,
    )


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
    az_response = arrayResponseSample(batch, sample_id, az_axis, el_zero).cpu().tolist()

    el_axis = torch.linspace(-math.pi / 2, math.pi / 2, resolution, device=dev, dtype=dtype)
    az_zero = torch.zeros(1, device=dev, dtype=dtype)
    el_response = arrayResponseSample(batch, sample_id, az_zero, el_axis).cpu().tolist()

    el_axis_display = el_axis

    # 3D surface (coarser grid for performance)
    res3d = min(resolution, 100)
    az_v = torch.linspace(-math.pi, math.pi, res3d * 2, device=dev, dtype=dtype)
    el_v = torch.linspace(-math.pi / 2, math.pi / 2, res3d, device=dev, dtype=dtype)
    az_g, el_g = torch.meshgrid(az_v, el_v, indexing="ij")

    raw = arrayResponseSample(batch, sample_id, az_g, el_g, dB=False)
    power = raw.abs().square()
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
    response = arrayResponseCore(
        sub_batch.elementLocalPosition,
        sub_batch.weights,
        sub_batch.wavelength,
        az[0],
        el[0],
        sub_batch.gain,
        dB=True,
    )[0]

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
