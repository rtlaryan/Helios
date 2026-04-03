from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .targetSpec import TargetSpec


def _gaussian_blob(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    center_lat: float,
    center_lon: float,
    radius_deg: float,
    sigma_deg: float,
    peak_linear: float,
) -> np.ndarray:
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
    n = len(verts_lat)
    min_d = np.full(len(lat_flat), np.inf, dtype=np.float32)
    for i in range(n):
        j = (i + 1) % n
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
    verts: list[dict[str, float]],
    sigma_deg: float,
    peak_linear: float,
) -> np.ndarray:
    verts_lat = np.array([v["lat"] for v in verts], dtype=np.float32)
    verts_lon = np.array([v["lon"] for v in verts], dtype=np.float32)

    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()

    inside = _point_in_polygon_vectorized(lat_flat, lon_flat, verts_lat, verts_lon)
    dist = _min_dist_to_polygon_edges(lat_flat, lon_flat, verts_lat, verts_lon)
    effective_dist = np.where(inside, 0.0, dist)
    blob = peak_linear * np.exp(-(effective_dist**2) / (2 * max(sigma_deg, 0.01) ** 2))
    return blob.reshape(lat_grid.shape).astype(np.float32)


def _zone_center(zone: dict[str, Any]) -> list[float]:
    if zone.get("shape") == "polygon":
        verts = zone.get("verts", [])
        if verts:
            return [
                float(sum(vertex["lat"] for vertex in verts) / len(verts)),
                float(sum(vertex["lon"] for vertex in verts) / len(verts)),
            ]
    return [float(zone.get("lat", 0.0)), float(zone.get("lon", 0.0))]


def _build_hotspot_tensor(zones: list[dict[str, Any]], hotspot_count: int | None) -> torch.Tensor:
    hotspots = [_zone_center(zone) for zone in zones if zone.get("type") == "power"]
    if not hotspots:
        hotspots = [[0.0, 0.0]]

    if hotspot_count is not None:
        hotspot_count = int(hotspot_count)
        if hotspot_count <= 0:
            raise ValueError("hotspot_count must be positive when provided")
        hotspots = hotspots[:hotspot_count]
        primary = hotspots[0]
        while len(hotspots) < hotspot_count:
            hotspots.append([primary[0], primary[1]])

    return torch.tensor(hotspots, dtype=torch.float32)


def build_target_maps(
    zones: list[dict[str, Any]],
    resolution_deg: float = 0.5,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
    normalize: bool = False,
) -> dict[str, Any]:
    lat_min, lat_max = lat_range if lat_range else (-90.0, 90.0)
    lon_min, lon_max = lon_range if lon_range else (-180.0, 180.0)

    lat_vec = np.arange(lat_min, lat_max + resolution_deg, resolution_deg, dtype=np.float32)
    lon_vec = np.arange(lon_min, lon_max + resolution_deg, resolution_deg, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lon_vec, lat_vec)

    height, width = lat_grid.shape
    power_lin = np.zeros((height, width), dtype=np.float32)
    importance = np.zeros((height, width), dtype=np.float32)

    for zone in zones:
        shape = zone.get("shape", "circle")
        sigma = max(float(zone.get("rolloff", 1.0)), 0.01)
        is_power = zone.get("type") == "power"

        if shape == "polygon":
            verts = zone.get("verts", [])
            if len(verts) < 3:
                continue
            peak_linear = 10 ** (float(zone.get("peak_db", 0.0)) / 10.0) if is_power else 1.0
            blob = _polygon_blob(lat_grid, lon_grid, verts, sigma, peak_linear)
        else:
            center_lat = float(zone["lat"])
            center_lon = float(zone["lon"])
            radius_deg = float(zone.get("radius_deg", 5.0))
            peak_linear = 10 ** (float(zone.get("peak_db", 0.0)) / 10.0) if is_power else 1.0
            blob = _gaussian_blob(
                lat_grid,
                lon_grid,
                center_lat,
                center_lon,
                radius_deg,
                sigma,
                peak_linear,
            )

        if is_power:
            power_lin = np.maximum(power_lin, blob)
        else:
            importance = np.maximum(importance, blob)

    imp_max = importance.max()
    if imp_max > 0:
        importance = importance / imp_max

    if normalize:
        pwr_max = power_lin.max()
        power_out = power_lin / pwr_max if pwr_max > 0 else power_lin
        return {
            "lat_vec": lat_vec.tolist(),
            "lon_vec": lon_vec.tolist(),
            "power_map": power_out.astype(np.float32).tolist(),
            "importance_map": importance.astype(np.float32).tolist(),
            "shape": [height, width],
            "power_normalized": True,
        }

    floor_lin = 10 ** (-100 / 10.0)
    power_lin_safe = np.maximum(power_lin, floor_lin)
    power_db = 10.0 * np.log10(power_lin_safe)
    power_db[power_lin < floor_lin] = -100.0

    return {
        "lat_vec": lat_vec.tolist(),
        "lon_vec": lon_vec.tolist(),
        "power_map": power_db.astype(np.float32).tolist(),
        "importance_map": importance.astype(np.float32).tolist(),
        "shape": [height, width],
        "power_normalized": False,
    }


def build_target_spec(
    maps: dict[str, Any],
    zones: list[dict[str, Any]],
    hotspot_count: int | None = None,
) -> TargetSpec:
    lat_vec = torch.tensor(maps["lat_vec"], dtype=torch.float32)
    lon_vec = torch.tensor(maps["lon_vec"], dtype=torch.float32)
    lon_grid, lat_grid = torch.meshgrid(lon_vec, lat_vec, indexing="xy")

    power_map = torch.tensor(maps["power_map"], dtype=torch.float32)
    importance_map = torch.tensor(maps["importance_map"], dtype=torch.float32)
    hotspots = _build_hotspot_tensor(zones, hotspot_count)

    return TargetSpec(
        searchLatitudes=lat_grid,
        searchLongitudes=lon_grid,
        importanceMap=importance_map,
        powerMap=power_map,
        hotspotCoordinates=hotspots,
    )


def load_target_from_zones_json(
    json_path: str | Path,
    resolution_deg: float = 0.5,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
    hotspot_count: int | None = None,
) -> TargetSpec:
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as handle:
        zones = json.load(handle)
    maps = build_target_maps(
        zones,
        resolution_deg=resolution_deg,
        lat_range=lat_range,
        lon_range=lon_range,
    )
    return build_target_spec(maps, zones, hotspot_count=hotspot_count)
