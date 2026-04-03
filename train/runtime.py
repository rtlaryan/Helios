from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


LogMode = Literal["off", "metrics_only", "dataset_compact", "dataset_full"]
CheckpointMode = Literal["off", "periodic"]
TargetMode = Literal["auto", "shared", "per_sample"]


@dataclass
class ExperimentConfig:
    name: str = "evo_1"
    logDir: str | Path | None = "runs"
    archiveDir: str | Path = "data/archive"
    resume: bool = True
    plotProjection: bool = False
    tensorboard: bool = True
    targetMode: TargetMode = "auto"


@dataclass
class DeviceConfig:
    device: str = "cpu"
    dtype: str = "float32"


@dataclass
class LoggingConfig:
    logMode: LogMode = "metrics_only"
    datasetFlushEverySteps: int = 10
    responseCompactSize: int = 64


@dataclass
class CheckpointConfig:
    checkpointMode: CheckpointMode = "periodic"
    checkpointEverySteps: int = 100


@dataclass
class WorkerConfig:
    asyncIO: bool = True
    datasetWriterWorkers: int = 1
    checkpointWriterWorkers: int = 1
    ioQueueSize: int = 2
