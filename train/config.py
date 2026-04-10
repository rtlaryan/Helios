from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import torch
import yaml
from scripts.arraySpec import ArraySpec
from scripts.target_generation import load_target_from_zones_json
from scripts.targetSpec import TargetBatch, TargetLike, TargetSpec

from train.evolve import (
    CheckpointConfig,
    DeviceConfig,
    EvolutionConfig,
    ExperimentConfig,
    LoggingConfig,
    WorkerConfig,
)
from train.objective import LossConfig

T = TypeVar("T")
_INT_PATTERN = re.compile(r"^[+-]?\d+$")
_FLOAT_PATTERN = re.compile(r"^[+-]?(?:\d+\.\d*|\d+|\.\d+)(?:[eE][+-]?\d+)?$")


@dataclass
class TargetConfig:
    reference: str | None = None
    references: list[str] | None = None
    manifest: str | None = None
    inline: dict[str, Any] | None = None
    inlineBatch: list[dict[str, Any]] | None = None
    format: str | None = None
    selection: str = "first"
    selectionCount: int | None = None
    selectionSeed: int | None = None
    resolutionDeg: float = 0.5
    decimate: int = 1


@dataclass
class RunConfig:
    experiment: ExperimentConfig
    device: DeviceConfig
    array: ArraySpec
    evolution: EvolutionConfig
    loss: LossConfig
    logging: LoggingConfig
    checkpoint: CheckpointConfig
    workers: WorkerConfig
    target: TargetConfig
    sourcePath: Path | None = None


def _coerce_dataclass(cls: type[T], payload: dict[str, Any]) -> T:
    allowed = {field.name for field in fields(cls)}
    filtered = {key: value for key, value in payload.items() if key in allowed}
    return cls(**filtered)


def _ensure_dict(payload: Any, section: str) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"{section} must be a mapping")
    return payload


def _normalize_yaml_scalars(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: _normalize_yaml_scalars(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_normalize_yaml_scalars(value) for value in payload]
    if not isinstance(payload, str):
        return payload

    stripped = payload.strip()
    if not stripped:
        return payload
    if _INT_PATTERN.fullmatch(stripped):
        try:
            return int(stripped)
        except ValueError:
            return payload
    if _FLOAT_PATTERN.fullmatch(stripped):
        try:
            return float(stripped)
        except ValueError:
            return payload
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("YAML root must be a mapping")
    return _normalize_yaml_scalars(payload)


def loadRunConfig(path: str | Path) -> RunConfig:
    configPath = Path(path)
    payload = _load_yaml(configPath)
    config = buildRunConfig(payload)
    config.sourcePath = configPath
    return config


def buildRunConfig(payload: dict[str, Any]) -> RunConfig:
    evolutionPayload = _ensure_dict(payload.get("evolution"), "evolution")
    if "responseChunkShapeStrategy" in evolutionPayload:
        raise ValueError("evolution.responseChunkShapeStrategy has been removed")
    if "responseReductionTileCap" in evolutionPayload:
        raise ValueError("evolution.responseReductionTileCap has been removed")

    experiment = _coerce_dataclass(
        ExperimentConfig, _ensure_dict(payload.get("experiment"), "experiment")
    )
    device = _coerce_dataclass(DeviceConfig, _ensure_dict(payload.get("device"), "device"))
    array = _coerce_dataclass(ArraySpec, _ensure_dict(payload.get("array"), "array"))
    evolution = _coerce_dataclass(
        EvolutionConfig,
        evolutionPayload,
    )
    loss = _coerce_dataclass(LossConfig, _ensure_dict(payload.get("loss"), "loss"))
    logging = _coerce_dataclass(
        LoggingConfig,
        _ensure_dict(payload.get("logging"), "logging"),
    )
    checkpoint = _coerce_dataclass(
        CheckpointConfig,
        _ensure_dict(payload.get("checkpoint"), "checkpoint"),
    )
    workers = _coerce_dataclass(
        WorkerConfig,
        _ensure_dict(payload.get("workers"), "workers"),
    )
    target = _coerce_dataclass(TargetConfig, _ensure_dict(payload.get("target"), "target"))

    config = RunConfig(
        experiment=experiment,
        device=device,
        array=array,
        evolution=evolution,
        loss=loss,
        logging=logging,
        checkpoint=checkpoint,
        workers=workers,
        target=target,
    )
    validateRunConfig(config)
    return config


def validateRunConfig(config: RunConfig) -> None:
    if config.evolution.initialWeightsType not in {"uniform", "random", "directed"}:
        raise ValueError("evolution.initialWeightsType must be 'uniform', 'random', or 'directed'")

    totalFraction = (
        config.evolution.cloneFraction
        + config.evolution.crossoverFraction
        + config.evolution.mutateFraction
        + config.evolution.randomFraction
    )
    if abs(totalFraction - 1.0) > 1e-6:
        raise ValueError("evolution clone/crossover/mutate/random fractions must sum to 1.0")
    if config.evolution.stagnationWindow < 0:
        raise ValueError("evolution.stagnationWindow must be non-negative")
    if config.evolution.sigmaBoostDuration < 0:
        raise ValueError("evolution.sigmaBoostDuration must be non-negative")
    if config.evolution.sigmaBoostMultiplier < 1.0:
        raise ValueError("evolution.sigmaBoostMultiplier must be at least 1.0")
    if not (0.0 <= config.evolution.sigmaBoostRandomFraction <= 1.0):
        raise ValueError("evolution.sigmaBoostRandomFraction must be in [0, 1]")
    if config.evolution.phaseMinSigmaScale < 0 or config.evolution.amplitudeMinSigmaScale < 0:
        raise ValueError("adaptive min-sigma scales must be non-negative")
    if config.evolution.wideGridSizeStart is not None and config.evolution.wideGridSizeStart <= 0:
        raise ValueError("evolution.wideGridSizeStart must be positive when set")
    if config.evolution.wideGridRampSteps is not None and config.evolution.wideGridRampSteps < 0:
        raise ValueError("evolution.wideGridRampSteps must be non-negative when set")
    if (
        config.evolution.linearResponseChunkSize is not None
        and config.evolution.linearResponseChunkSize <= 0
    ):
        raise ValueError("evolution.linearResponseChunkSize must be positive when set")
    if (
        config.evolution.wideResponseChunkSize is not None
        and config.evolution.wideResponseChunkSize <= 0
    ):
        raise ValueError("evolution.wideResponseChunkSize must be positive when set")
    if config.logging.datasetFlushEverySteps <= 0:
        raise ValueError("logging.datasetFlushEverySteps must be positive")
    if config.logging.responseCompactSize <= 0:
        raise ValueError("logging.responseCompactSize must be positive")
    if config.checkpoint.checkpointEverySteps <= 0:
        raise ValueError("checkpoint.checkpointEverySteps must be positive")
    if config.workers.datasetWriterWorkers < 0 or config.workers.checkpointWriterWorkers < 0:
        raise ValueError("worker counts must be non-negative")
    if config.workers.ioQueueSize <= 0:
        raise ValueError("workers.ioQueueSize must be positive")
    targetSources = [
        config.target.reference is not None,
        config.target.references is not None,
        config.target.manifest is not None,
        config.target.inline is not None,
        config.target.inlineBatch is not None,
    ]
    if sum(targetSources) != 1:
        raise ValueError("target must define exactly one source mode")
    if config.target.selection not in {"first", "random_without_replacement"}:
        raise ValueError("target.selection must be either 'first' or 'random_without_replacement'")
    if config.target.selectionCount is not None and config.target.selectionCount <= 0:
        raise ValueError("target.selectionCount must be positive when set")
    if config.target.decimate <= 0:
        raise ValueError("target.decimate must be positive")


def resolveDevice(config: RunConfig) -> tuple[torch.device, torch.dtype]:
    dtypeMap = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if config.device.dtype not in dtypeMap:
        raise ValueError(f"unsupported dtype: {config.device.dtype}")
    return torch.device(config.device.device), dtypeMap[config.device.dtype]


def _detectTargetFormat(path: Path, explicitFormat: str | None) -> str:
    if explicitFormat is not None:
        return explicitFormat
    if path.suffix == ".pt":
        return "pt"
    if path.suffix == ".json":
        return "zones_json"
    raise ValueError(f"unable to infer target format from path: {path}")


def _loadTargetReference(path: Path, fmt: str, resolutionDeg: float) -> TargetSpec:
    if fmt == "pt":
        payload = torch.load(path, weights_only=False)
        if isinstance(payload, TargetSpec):
            return payload
        if isinstance(payload, dict):
            return TargetSpec.fromSerializedTargetSpec(payload)
        raise ValueError(f"unsupported target payload in {path}")
    if fmt == "zones_json":
        return load_target_from_zones_json(path, resolution_deg=resolutionDeg)
    raise ValueError(f"unsupported target format: {fmt}")


def _loadInlineTarget(payload: dict[str, Any]) -> TargetSpec:
    return TargetSpec.fromMapping(payload)


def _resolveManifestRecordPath(manifestPath: Path, recordPath: str) -> Path:
    path = Path(recordPath)
    return path if path.is_absolute() else manifestPath.parent / path


def _loadTargetManifest(config: RunConfig) -> TargetBatch:
    targetConfig = config.target
    assert targetConfig.manifest is not None

    manifestPath = Path(targetConfig.manifest)
    manifestPayload = _load_yaml(manifestPath)
    records = manifestPayload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"target manifest {manifestPath} must contain a non-empty records list")

    selectionCount = (
        config.evolution.batchSize
        if targetConfig.selectionCount is None
        else int(targetConfig.selectionCount)
    )
    if selectionCount > len(records):
        raise ValueError(
            f"target manifest {manifestPath} only has {len(records)} records "
            f"(requested {selectionCount})"
        )

    if targetConfig.selection == "first":
        selectedRecords = records[:selectionCount]
    else:
        rng = random.Random(targetConfig.selectionSeed)
        selectedRecords = rng.sample(records, selectionCount)

    targets = []
    for record in selectedRecords:
        if not isinstance(record, dict) or "path" not in record:
            raise ValueError(
                f"target manifest {manifestPath} contains an invalid record: {record!r}"
            )
        recordPath = _resolveManifestRecordPath(manifestPath, str(record["path"]))
        targets.append(
            _loadTargetReference(
                recordPath,
                _detectTargetFormat(recordPath, targetConfig.format),
                targetConfig.resolutionDeg,
            )
        )
    return TargetBatch.fromTargetSpecs(targets)


def resolveTarget(config: RunConfig) -> TargetLike:
    targetConfig = config.target
    if targetConfig.reference is not None:
        path = Path(targetConfig.reference)
        fmt = _detectTargetFormat(path, targetConfig.format)
        target = _loadTargetReference(path, fmt, targetConfig.resolutionDeg)
    elif targetConfig.references is not None:
        targets = [
            _loadTargetReference(
                Path(path),
                _detectTargetFormat(Path(path), targetConfig.format),
                targetConfig.resolutionDeg,
            )
            for path in targetConfig.references
        ]
        target = TargetBatch.fromTargetSpecs(targets)
    elif targetConfig.manifest is not None:
        target = _loadTargetManifest(config)
    elif targetConfig.inline is not None:
        target = _loadInlineTarget(targetConfig.inline)
    elif targetConfig.inlineBatch is not None:
        target = TargetBatch.fromTargetSpecs(
            [_loadInlineTarget(payload) for payload in targetConfig.inlineBatch]
        )
    else:
        raise ValueError("target configuration did not resolve to a target source")

    if isinstance(target, TargetBatch) and target.batchSize != config.evolution.batchSize:
        raise ValueError(
            "target batch size must match evolution.batchSize "
            f"({target.batchSize} != {config.evolution.batchSize})"
        )
    if targetConfig.decimate > 1:
        target = target.decimate(targetConfig.decimate)
    return target


def runConfigToDict(config: RunConfig) -> dict[str, Any]:
    return {
        "experiment": asdict(config.experiment),
        "device": asdict(config.device),
        "array": asdict(config.array),
        "evolution": {
            key: value for key, value in asdict(config.evolution).items() if key != "generator"
        },
        "loss": asdict(config.loss),
        "logging": asdict(config.logging),
        "checkpoint": asdict(config.checkpoint),
        "workers": asdict(config.workers),
        "target": asdict(config.target),
    }


def dumpRunConfig(config: RunConfig, path: str | Path) -> None:
    outputPath = Path(path)
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    with outputPath.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(runConfigToDict(config), handle, sort_keys=False)


def configsMatch(path: str | Path, config: RunConfig) -> bool:
    existing = _load_yaml(Path(path))
    return existing == runConfigToDict(config)
