from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import math
import os
from pathlib import Path
import random
from typing import Any
import uuid

import torch
import yaml

from scripts.target_generation import build_target_maps, build_target_spec
from scripts.targetSpec import TargetSpec

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


CATEGORY_NAMES = ("single_circle", "irregular_shape", "multibeam")
POWER_MODES = {"normalized", "db"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class CorpusOutputConfig:
    root: str | Path
    manifestPath: str | Path | None = None
    prefix: str = "target"


@dataclass
class CorpusGridConfig:
    latRange: tuple[float, float]
    lonRange: tuple[float, float]
    resolutionDeg: float = 0.5
    decimate: int = 1
    powerMode: str = "normalized"


@dataclass
class CurriculumConfig:
    weights: dict[str, float]


@dataclass
class SingleCircleConfig:
    radiusRange: tuple[float, float] = (1.0, 4.0)
    rolloffRange: tuple[float, float] = (0.5, 1.5)
    peakDBRange: tuple[float, float] = (-3.0, 0.0)


@dataclass
class IrregularShapeConfig:
    componentCountRange: tuple[int, int] = (2, 4)
    circleComponentProbability: float = 0.5
    radiusRange: tuple[float, float] = (1.0, 3.0)
    polygonRadiusRange: tuple[float, float] = (1.5, 4.0)
    polygonVertexCountRange: tuple[int, int] = (3, 6)
    componentJitterRange: tuple[float, float] = (1.0, 5.0)
    rolloffRange: tuple[float, float] = (0.5, 1.5)
    peakDBRange: tuple[float, float] = (-6.0, 0.0)


@dataclass
class MultibeamConfig:
    beamCountRange: tuple[int, int] = (2, 4)
    radiusRange: tuple[float, float] = (1.0, 3.0)
    rolloffRange: tuple[float, float] = (0.5, 1.5)
    peakDBRange: tuple[float, float] = (-3.0, 0.0)
    minSeparationDeg: float = 5.0
    placementAttempts: int = 64


@dataclass
class CategoryConfig:
    single_circle: SingleCircleConfig
    irregular_shape: IrregularShapeConfig
    multibeam: MultibeamConfig


@dataclass
class TargetCorpusConfig:
    count: int
    seed: int
    output: CorpusOutputConfig
    grid: CorpusGridConfig
    curriculum: CurriculumConfig
    categories: CategoryConfig
    workers: int | str = 1

    @property
    def hotspotCount(self) -> int:
        return int(self.categories.multibeam.beamCountRange[1])


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("YAML root must be a mapping")
    return payload


def _require_mapping(payload: Any, section: str) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"{section} must be a mapping")
    return payload


def _float_range(payload: Any, field_name: str) -> tuple[float, float]:
    if not isinstance(payload, (list, tuple)) or len(payload) != 2:
        raise ValueError(f"{field_name} must be a 2-item list or tuple")
    low = float(payload[0])
    high = float(payload[1])
    if high < low:
        raise ValueError(f"{field_name} must be ordered low <= high")
    return low, high


def _int_range(payload: Any, field_name: str) -> tuple[int, int]:
    low, high = _float_range(payload, field_name)
    return int(low), int(high)


def loadTargetCorpusConfig(path: str | Path) -> TargetCorpusConfig:
    payload = _load_yaml(Path(path))

    output_payload = _require_mapping(payload.get("output"), "output")
    grid_payload = _require_mapping(payload.get("grid"), "grid")
    curriculum_payload = _require_mapping(payload.get("curriculum"), "curriculum")
    category_payload = _require_mapping(payload.get("categories"), "categories")
    curriculum_weights = _require_mapping(curriculum_payload.get("weights"), "curriculum.weights")

    output = CorpusOutputConfig(
        root=output_payload.get("root", "data/targets/generated"),
        manifestPath=output_payload.get("manifestPath"),
        prefix=str(output_payload.get("prefix", "target")),
    )
    grid = CorpusGridConfig(
        latRange=_float_range(grid_payload.get("latRange", [-90.0, 90.0]), "grid.latRange"),
        lonRange=_float_range(grid_payload.get("lonRange", [-180.0, 180.0]), "grid.lonRange"),
        resolutionDeg=float(grid_payload.get("resolutionDeg", 0.5)),
        decimate=int(grid_payload.get("decimate", 1)),
        powerMode=str(grid_payload.get("powerMode", "normalized")),
    )
    curriculum = CurriculumConfig(
        weights={name: float(curriculum_weights.get(name, 0.0)) for name in CATEGORY_NAMES}
    )

    single_circle_payload = _require_mapping(category_payload.get("single_circle"), "categories.single_circle")
    irregular_payload = _require_mapping(category_payload.get("irregular_shape"), "categories.irregular_shape")
    multibeam_payload = _require_mapping(category_payload.get("multibeam"), "categories.multibeam")

    categories = CategoryConfig(
        single_circle=SingleCircleConfig(
            radiusRange=_float_range(single_circle_payload.get("radiusRange", [1.0, 4.0]), "categories.single_circle.radiusRange"),
            rolloffRange=_float_range(single_circle_payload.get("rolloffRange", [0.5, 1.5]), "categories.single_circle.rolloffRange"),
            peakDBRange=_float_range(single_circle_payload.get("peakDBRange", [-3.0, 0.0]), "categories.single_circle.peakDBRange"),
        ),
        irregular_shape=IrregularShapeConfig(
            componentCountRange=_int_range(irregular_payload.get("componentCountRange", [2, 4]), "categories.irregular_shape.componentCountRange"),
            circleComponentProbability=float(irregular_payload.get("circleComponentProbability", 0.5)),
            radiusRange=_float_range(irregular_payload.get("radiusRange", [1.0, 3.0]), "categories.irregular_shape.radiusRange"),
            polygonRadiusRange=_float_range(irregular_payload.get("polygonRadiusRange", [1.5, 4.0]), "categories.irregular_shape.polygonRadiusRange"),
            polygonVertexCountRange=_int_range(irregular_payload.get("polygonVertexCountRange", [3, 6]), "categories.irregular_shape.polygonVertexCountRange"),
            componentJitterRange=_float_range(irregular_payload.get("componentJitterRange", [1.0, 5.0]), "categories.irregular_shape.componentJitterRange"),
            rolloffRange=_float_range(irregular_payload.get("rolloffRange", [0.5, 1.5]), "categories.irregular_shape.rolloffRange"),
            peakDBRange=_float_range(irregular_payload.get("peakDBRange", [-6.0, 0.0]), "categories.irregular_shape.peakDBRange"),
        ),
        multibeam=MultibeamConfig(
            beamCountRange=_int_range(multibeam_payload.get("beamCountRange", [2, 4]), "categories.multibeam.beamCountRange"),
            radiusRange=_float_range(multibeam_payload.get("radiusRange", [1.0, 3.0]), "categories.multibeam.radiusRange"),
            rolloffRange=_float_range(multibeam_payload.get("rolloffRange", [0.5, 1.5]), "categories.multibeam.rolloffRange"),
            peakDBRange=_float_range(multibeam_payload.get("peakDBRange", [-3.0, 0.0]), "categories.multibeam.peakDBRange"),
            minSeparationDeg=float(multibeam_payload.get("minSeparationDeg", 5.0)),
            placementAttempts=int(multibeam_payload.get("placementAttempts", 64)),
        ),
    )

    config = TargetCorpusConfig(
        count=int(payload.get("count", 0)),
        seed=int(payload.get("seed", 0)),
        output=output,
        grid=grid,
        curriculum=curriculum,
        categories=categories,
        workers=payload.get("workers", 1),
    )
    validateTargetCorpusConfig(config)
    return config


def validateTargetCorpusConfig(config: TargetCorpusConfig) -> None:
    if config.count <= 0:
        raise ValueError("count must be positive")
    if isinstance(config.workers, str):
        if config.workers != "auto":
            raise ValueError("workers must be a positive integer or 'auto'")
    elif int(config.workers) <= 0:
        raise ValueError("workers must be positive")
    if config.grid.resolutionDeg <= 0:
        raise ValueError("grid.resolutionDeg must be positive")
    if config.grid.decimate <= 0:
        raise ValueError("grid.decimate must be positive")
    if config.grid.powerMode not in POWER_MODES:
        raise ValueError(f"grid.powerMode must be one of {sorted(POWER_MODES)}")
    if config.hotspotCount <= 0:
        raise ValueError("multibeam beamCountRange upper bound must be positive")
    total_weight = sum(max(0.0, value) for value in config.curriculum.weights.values())
    if total_weight <= 0:
        raise ValueError("curriculum.weights must contain at least one positive value")


def _resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base_dir / path


def _resolve_worker_count(workers: int | str) -> int:
    if workers == "auto":
        cpu_count = os.cpu_count() or 1
        return max(1, cpu_count - 1)
    return max(1, int(workers))


class _NullProgressBar:
    def update(self, _n: int = 1) -> None:
        return None

    def close(self) -> None:
        return None


def _create_progress_bar(*, total: int, enabled: bool, description: str):
    if not enabled or tqdm is None:
        return _NullProgressBar()
    return tqdm(total=total, desc=description, unit="target", dynamic_ncols=True)


def _sample_float(value_range: tuple[float, float], rng: random.Random) -> float:
    low, high = value_range
    return low if math.isclose(low, high) else rng.uniform(low, high)


def _sample_int(value_range: tuple[int, int], rng: random.Random) -> int:
    low, high = value_range
    return low if low == high else rng.randint(low, high)


def _sample_center(value_range: tuple[float, float], margin: float, rng: random.Random) -> float:
    low, high = value_range
    if high - low <= 2 * margin:
        return rng.uniform(low, high)
    return rng.uniform(low + margin, high - margin)


def _clamp(value: float, value_range: tuple[float, float]) -> float:
    return max(value_range[0], min(value_range[1], value))


def _sample_circle_zone(
    rng: random.Random,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    radius_range: tuple[float, float],
    rolloff_range: tuple[float, float],
    peak_db_range: tuple[float, float],
) -> list[dict[str, Any]]:
    radius = _sample_float(radius_range, rng)
    rolloff = _sample_float(rolloff_range, rng)
    center_lat = _sample_center(lat_range, radius, rng)
    center_lon = _sample_center(lon_range, radius, rng)
    peak_db = _sample_float(peak_db_range, rng)
    return [
        {
            "shape": "circle",
            "type": "power",
            "lat": center_lat,
            "lon": center_lon,
            "radius_deg": radius,
            "rolloff": rolloff,
            "peak_db": peak_db,
        },
        {
            "shape": "circle",
            "type": "importance",
            "lat": center_lat,
            "lon": center_lon,
            "radius_deg": radius,
            "rolloff": rolloff,
        },
    ]


def _sample_polygon_vertices(
    center_lat: float,
    center_lon: float,
    vertex_count: int,
    radius_deg: float,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    rng: random.Random,
) -> list[dict[str, float]]:
    angles = sorted(rng.uniform(0.0, 2.0 * math.pi) for _ in range(vertex_count))
    cos_center = max(math.cos(math.radians(center_lat)), 1e-3)
    vertices: list[dict[str, float]] = []
    for angle in angles:
        radial_scale = rng.uniform(0.6, 1.0)
        lat = center_lat + math.sin(angle) * radius_deg * radial_scale
        lon = center_lon + (math.cos(angle) * radius_deg * radial_scale) / cos_center
        vertices.append({"lat": _clamp(lat, lat_range), "lon": _clamp(lon, lon_range)})
    return vertices


def _sample_irregular_zones(
    rng: random.Random,
    grid: CorpusGridConfig,
    config: IrregularShapeConfig,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    component_count = _sample_int(config.componentCountRange, rng)
    base_radius = max(config.radiusRange[1], config.polygonRadiusRange[1], config.componentJitterRange[1])
    anchor_lat = _sample_center(grid.latRange, base_radius, rng)
    anchor_lon = _sample_center(grid.lonRange, base_radius, rng)

    zones: list[dict[str, Any]] = []
    polygon_count = 0
    circle_count = 0
    for _ in range(component_count):
        jitter = _sample_float(config.componentJitterRange, rng)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        cos_anchor = max(math.cos(math.radians(anchor_lat)), 1e-3)
        center_lat = _clamp(anchor_lat + math.sin(angle) * jitter, grid.latRange)
        center_lon = _clamp(anchor_lon + (math.cos(angle) * jitter) / cos_anchor, grid.lonRange)
        rolloff = _sample_float(config.rolloffRange, rng)
        peak_db = _sample_float(config.peakDBRange, rng)

        if rng.random() < config.circleComponentProbability:
            radius = _sample_float(config.radiusRange, rng)
            circle_count += 1
            shape_payload: dict[str, Any] = {
                "shape": "circle",
                "lat": center_lat,
                "lon": center_lon,
                "radius_deg": radius,
            }
        else:
            polygon_count += 1
            vertex_count = _sample_int(config.polygonVertexCountRange, rng)
            radius = _sample_float(config.polygonRadiusRange, rng)
            shape_payload = {
                "shape": "polygon",
                "verts": _sample_polygon_vertices(
                    center_lat,
                    center_lon,
                    vertex_count,
                    radius,
                    grid.latRange,
                    grid.lonRange,
                    rng,
                ),
            }

        zones.append({**shape_payload, "type": "power", "rolloff": rolloff, "peak_db": peak_db})
        zones.append({**shape_payload, "type": "importance", "rolloff": rolloff})

    return zones, {
        "componentCount": component_count,
        "polygonCount": polygon_count,
        "circleCount": circle_count,
    }


def _great_circle_like_distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    mean_lat = math.radians((point_a[0] + point_b[0]) / 2.0)
    dlat = point_a[0] - point_b[0]
    dlon = (point_a[1] - point_b[1]) * math.cos(mean_lat)
    return math.sqrt(dlat**2 + dlon**2)


def _sample_multibeam_zones(
    rng: random.Random,
    grid: CorpusGridConfig,
    config: MultibeamConfig,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    beam_count = _sample_int(config.beamCountRange, rng)
    max_radius = config.radiusRange[1]
    centers: list[tuple[float, float]] = []
    zones: list[dict[str, Any]] = []

    for _ in range(beam_count):
        chosen_center: tuple[float, float] | None = None
        for _attempt in range(config.placementAttempts):
            candidate = (
                _sample_center(grid.latRange, max_radius, rng),
                _sample_center(grid.lonRange, max_radius, rng),
            )
            if all(
                _great_circle_like_distance(candidate, existing) >= config.minSeparationDeg
                for existing in centers
            ):
                chosen_center = candidate
                break
        if chosen_center is None:
            chosen_center = (
                _sample_center(grid.latRange, max_radius, rng),
                _sample_center(grid.lonRange, max_radius, rng),
            )

        centers.append(chosen_center)
        radius = _sample_float(config.radiusRange, rng)
        rolloff = _sample_float(config.rolloffRange, rng)
        peak_db = _sample_float(config.peakDBRange, rng)
        zones.append(
            {
                "shape": "circle",
                "type": "power",
                "lat": chosen_center[0],
                "lon": chosen_center[1],
                "radius_deg": radius,
                "rolloff": rolloff,
                "peak_db": peak_db,
            }
        )
        zones.append(
            {
                "shape": "circle",
                "type": "importance",
                "lat": chosen_center[0],
                "lon": chosen_center[1],
                "radius_deg": radius,
                "rolloff": rolloff,
            }
        )

    return zones, {"beamCount": beam_count}


def _allocate_category_counts(total: int, weights: dict[str, float]) -> dict[str, int]:
    normalized = {name: max(0.0, weights.get(name, 0.0)) for name in CATEGORY_NAMES}
    total_weight = sum(normalized.values())
    if total_weight <= 0:
        raise ValueError("curriculum.weights must contain at least one positive value")
    raw = {name: total * normalized[name] / total_weight for name in CATEGORY_NAMES}
    counts = {name: int(math.floor(raw[name])) for name in CATEGORY_NAMES}
    assigned = sum(counts.values())
    remainders = sorted(
        CATEGORY_NAMES,
        key=lambda name: (raw[name] - counts[name], normalized[name], name),
        reverse=True,
    )
    for name in remainders[: total - assigned]:
        counts[name] += 1
    return counts


def _build_category_schedule(counts: dict[str, int], rng: random.Random) -> list[str]:
    schedule = [name for name in CATEGORY_NAMES for _ in range(counts[name])]
    rng.shuffle(schedule)
    return schedule


def _generate_target_for_category(
    category: str,
    config: TargetCorpusConfig,
    rng: random.Random,
) -> tuple[TargetSpec, list[dict[str, Any]], dict[str, Any]]:
    if category == "single_circle":
        zones = _sample_circle_zone(
            rng,
            config.grid.latRange,
            config.grid.lonRange,
            config.categories.single_circle.radiusRange,
            config.categories.single_circle.rolloffRange,
            config.categories.single_circle.peakDBRange,
        )
        metadata: dict[str, Any] = {"beamCount": 1}
    elif category == "irregular_shape":
        zones, metadata = _sample_irregular_zones(rng, config.grid, config.categories.irregular_shape)
    elif category == "multibeam":
        zones, metadata = _sample_multibeam_zones(rng, config.grid, config.categories.multibeam)
    else:
        raise ValueError(f"unsupported category: {category}")

    maps = build_target_maps(
        zones,
        resolution_deg=config.grid.resolutionDeg,
        lat_range=config.grid.latRange,
        lon_range=config.grid.lonRange,
        normalize=config.grid.powerMode == "normalized",
    )
    target = build_target_spec(maps, zones, hotspot_count=config.hotspotCount)
    if config.grid.decimate > 1:
        target = target.decimate(config.grid.decimate)
    metadata.update({"powerMode": config.grid.powerMode, "zoneCount": len(zones)})
    return target, zones, metadata


def _record_relative_path(path: Path, manifest_path: Path) -> str:
    try:
        return str(path.relative_to(manifest_path.parent))
    except ValueError:
        return str(path)


def _save_target_atomic(target: TargetSpec, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f".{target_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(target, temp_path)
        os.replace(temp_path, target_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _generate_target_payload(
    *,
    index: int,
    category: str,
    config: TargetCorpusConfig,
) -> tuple[int, str, TargetSpec, dict[str, Any]]:
    rng = random.Random(config.seed + index)
    target, _zones, metadata = _generate_target_for_category(category, config, rng)
    return index, category, target, metadata


def _generate_target_payload_star(
    args: tuple[int, str, TargetCorpusConfig],
) -> tuple[int, str, TargetSpec, dict[str, Any]]:
    index, category, config = args
    return _generate_target_payload(index=index, category=category, config=config)


def _save_generated_target(
    *,
    index: int,
    category: str,
    target: TargetSpec,
    metadata: dict[str, Any],
    output_root: Path,
    manifest_path: Path,
    prefix: str,
) -> dict[str, Any]:
    target_path = output_root / f"{prefix}_{index:05d}.pt"
    _save_target_atomic(target, target_path)
    return {
        "index": index,
        "category": category,
        "path": _record_relative_path(target_path, manifest_path),
        **metadata,
    }


def generateTargetCorpus(
    config_path: str | Path,
    *,
    workers: int | str | None = None,
    showProgress: bool = True,
    progressDescription: str = "Generating targets",
) -> dict[str, Any]:
    source_path = Path(config_path)
    config = loadTargetCorpusConfig(source_path)
    if workers is not None:
        config.workers = workers
        validateTargetCorpusConfig(config)

    output_root = _resolve_path(config.output.root, PROJECT_ROOT)
    manifest_path = _resolve_path(
        config.output.manifestPath or output_root / "manifest.yaml",
        PROJECT_ROOT,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    schedule_rng = random.Random(config.seed)
    curriculum_counts = _allocate_category_counts(config.count, config.curriculum.weights)
    schedule = _build_category_schedule(curriculum_counts, schedule_rng)

    worker_count = _resolve_worker_count(config.workers)
    job_args = [(index, category, config) for index, category in enumerate(schedule)]
    records_by_index: dict[int, dict[str, Any]] = {}
    progress_bar = _create_progress_bar(
        total=len(job_args),
        enabled=showProgress,
        description=progressDescription,
    )
    try:
        if worker_count == 1:
            for args in job_args:
                index, category, target, metadata = _generate_target_payload_star(args)
                records_by_index[index] = _save_generated_target(
                    index=index,
                    category=category,
                    target=target,
                    metadata=metadata,
                    output_root=output_root,
                    manifest_path=manifest_path,
                    prefix=config.output.prefix,
                )
                progress_bar.update(1)
        else:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(_generate_target_payload_star, args) for args in job_args]
                for future in as_completed(futures):
                    index, category, target, metadata = future.result()
                    records_by_index[index] = _save_generated_target(
                        index=index,
                        category=category,
                        target=target,
                        metadata=metadata,
                        output_root=output_root,
                        manifest_path=manifest_path,
                        prefix=config.output.prefix,
                    )
                    progress_bar.update(1)
    finally:
        progress_bar.close()

    records = [records_by_index[index] for index in range(len(job_args))]
    manifest = {
        "version": 1,
        "seed": config.seed,
        "count": config.count,
        "workers": worker_count,
        "powerMode": config.grid.powerMode,
        "maxHotspots": config.hotspotCount,
        "grid": {
            "latRange": list(config.grid.latRange),
            "lonRange": list(config.grid.lonRange),
            "resolutionDeg": config.grid.resolutionDeg,
            "decimate": config.grid.decimate,
            "powerMode": config.grid.powerMode,
        },
        "curriculum": {
            "weights": config.curriculum.weights,
            "counts": curriculum_counts,
        },
        "records": records,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)
    return manifest
