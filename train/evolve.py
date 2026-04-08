from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import re
import shutil
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from scripts.arrayBatch import ArrayBatch, merge
from scripts.arraySpec import ArraySpec
from scripts.batchFactory import generateBatch
from scripts.plots import projectResponseOnTarget
from scripts.targetSpec import (
    TargetBatch,
    TargetLike,
    TargetSpec,
    fetchTargetSample,
    inferTargetCenter,
    serializeTarget,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from train.objective import (
    BatchEvaluation,
    EvaluationDiagnostics,
    LossConfig,
    LossType,
    TargetMode,
    evaluateBatch,
)
from train.persistence import AsyncWriterPool
from train.runtime import CheckpointConfig, LoggingConfig, WorkerConfig


_INT_PATTERN = re.compile(r"^[+-]?\d+$")
_FLOAT_PATTERN = re.compile(
    r"^[+-]?(?:\d+\.\d*|\d+|\.\d+)(?:[eE][+-]?\d+)?$"
)
_RESUME_MUTABLE_EVOLUTION_KEYS = {
    "initialWeightsType",
    "phaseSigma",
    "amplitudeSigma",
    "gainSigma",
    "phaseSigmaDecay",
    "amplitudeSigmaDecay",
    "randomFractionDecay",
    "phaseMinSigma",
    "amplitudeMinSigma",
    "minRandomFraction",
    "stagnationWindow",
    "sigmaBoostDuration",
    "sigmaBoostMultiplier",
    "sigmaBoostRandomFraction",
    "phaseAdaptiveSigmaFloor",
    "amplitudeAdaptiveSigmaFloor",
    "phaseMinSigmaScale",
    "amplitudeMinSigmaScale",
    "wideGridSizeStart",
    "wideGridRampSteps",
    "linearResponseChunkSize",
    "wideResponseChunkSize",
    "responseReductionTileCap",
}
_RESUME_MUTABLE_EXPERIMENT_KEYS = {
    "resume",
    "plotProjection",
}


def _normalize_config_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        normalized = {
            key: _normalize_config_payload(value)
            for key, value in payload.items()
        }
        evolutionConfig = normalized.get("evolution")
        if isinstance(evolutionConfig, dict):
            evolutionConfig.pop("generator", None)
        return normalized

    if isinstance(payload, list):
        return [_normalize_config_payload(value) for value in payload]

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


def _comparison_config_payload(payload: Any, allowResumeMutation: bool = False) -> Any:
    normalized = _normalize_config_payload(payload)
    if not allowResumeMutation or not isinstance(normalized, dict):
        return normalized

    experimentConfig = normalized.get("experiment")
    if isinstance(experimentConfig, dict):
        for key in _RESUME_MUTABLE_EXPERIMENT_KEYS:
            experimentConfig.pop(key, None)

    evolutionConfig = normalized.get("evolution")
    if isinstance(evolutionConfig, dict):
        for key in _RESUME_MUTABLE_EVOLUTION_KEYS:
            evolutionConfig.pop(key, None)
    return normalized


@dataclass
class PopulationCounts:
    cloneCount: int
    crossoverCount: int
    mutateCount: int
    randomCount: int


@dataclass
class SchedulerState:
    currentPhaseSigma: float
    currentAmplitudeSigma: float
    currentRandomFraction: float
    stagnationCount: int = 0
    boostStepsRemaining: int = 0

    def serialize(self) -> dict[str, Any]:
        return {
            "currentPhaseSigma": self.currentPhaseSigma,
            "currentAmplitudeSigma": self.currentAmplitudeSigma,
            "currentRandomFraction": self.currentRandomFraction,
            "stagnationCount": self.stagnationCount,
            "boostStepsRemaining": self.boostStepsRemaining,
        }

    @classmethod
    def fromPayload(cls, payload: dict[str, Any] | None) -> "SchedulerState | None":
        if payload is None:
            return None
        return cls(
            currentPhaseSigma=float(payload["currentPhaseSigma"]),
            currentAmplitudeSigma=float(payload["currentAmplitudeSigma"]),
            currentRandomFraction=float(payload["currentRandomFraction"]),
            stagnationCount=int(payload.get("stagnationCount", 0)),
            boostStepsRemaining=int(payload.get("boostStepsRemaining", 0)),
        )


@dataclass
class PopulationState:
    batch: ArrayBatch
    evaluation: BatchEvaluation
    lossParams: LossConfig

    @property
    def scores(self) -> torch.Tensor:
        return self.evaluation.totalLoss


@dataclass
class EvolutionConfig:
    generator: torch.Generator | None = None
    batchSize: int = 100
    evolutionSteps: int = 100
    initialWeightsType: str = "random"

    cloneFraction: float = 0.2
    crossoverFraction: float = 0.0
    mutateFraction: float = 0.5
    randomFraction: float = 0.3
    parentPoolFraction: float = 0.2

    phaseSigma: float = 0.2
    amplitudeSigma: float = 0.02
    gainSigma: float = 0

    phaseSigmaDecay: float = 0.999
    amplitudeSigmaDecay: float = 0.998
    randomFractionDecay: float = 0.999

    phaseMinSigma: float = 1e-3
    amplitudeMinSigma: float = 1e-4
    minRandomFraction: float = 0.05
    lossType: LossType = "HUBER"

    stagnationWindow: int = 250
    sigmaBoostDuration: int = 25
    sigmaBoostMultiplier: float = 2.0
    sigmaBoostRandomFraction: float = 0.15
    phaseAdaptiveSigmaFloor: bool = True
    amplitudeAdaptiveSigmaFloor: bool = True
    phaseMinSigmaScale: float = 0.1
    amplitudeMinSigmaScale: float = 0.1
    wideGridSizeStart: int | None = None
    wideGridRampSteps: int | None = None
    linearResponseChunkSize: int | None = None
    wideResponseChunkSize: int | None = None
    responseReductionTileCap: int = 256

    @property
    def cloneCount(self) -> int:
        return int(self.cloneFraction * self.batchSize)

    @property
    def crossoverCount(self) -> int:
        return int(self.crossoverFraction * self.batchSize)

    @property
    def parentPoolCount(self) -> int:
        return max(1, int(self.parentPoolFraction * self.batchSize))

    def baseSigmaAt(self, step: int) -> tuple[float, float]:
        phaseSigma = self.phaseSigma * (self.phaseSigmaDecay**step)
        amplitudeSigma = self.amplitudeSigma * (self.amplitudeSigmaDecay**step)
        return phaseSigma, amplitudeSigma

    def scheduledRandomFraction(self, step: int) -> float:
        decayed = self.randomFraction * (self.randomFractionDecay**step)
        return max(self.minRandomFraction, decayed)

    def countsForRandomFraction(self, randomFraction: float) -> PopulationCounts:
        cloneCount = self.cloneCount
        crossoverCount = self.crossoverCount
        randomCount = int(randomFraction * self.batchSize)
        randomCount = min(randomCount, self.batchSize - cloneCount - crossoverCount)
        mutateCount = self.batchSize - cloneCount - crossoverCount - randomCount
        return PopulationCounts(
            cloneCount=cloneCount,
            crossoverCount=crossoverCount,
            mutateCount=mutateCount,
            randomCount=randomCount,
        )

    def serializeEvolutionConfig(self) -> dict:
        return {
            "batchSize": self.batchSize,
            "cloneFraction": self.cloneFraction,
            "crossoverFraction": self.crossoverFraction,
            "mutateFraction": self.mutateFraction,
            "randomFraction": self.randomFraction,
            "parentPoolFraction": self.parentPoolFraction,
            "phaseSigma": self.phaseSigma,
            "amplitudeSigma": self.amplitudeSigma,
            "gainSigma": self.gainSigma,
            "phaseSigmaDecay": self.phaseSigmaDecay,
            "amplitudeSigmaDecay": self.amplitudeSigmaDecay,
            "randomFractionDecay": self.randomFractionDecay,
            "phaseMinSigma": self.phaseMinSigma,
            "amplitudeMinSigma": self.amplitudeMinSigma,
            "minRandomFraction": self.minRandomFraction,
            "lossType": self.lossType,
            "stagnationWindow": self.stagnationWindow,
            "sigmaBoostDuration": self.sigmaBoostDuration,
            "sigmaBoostMultiplier": self.sigmaBoostMultiplier,
            "sigmaBoostRandomFraction": self.sigmaBoostRandomFraction,
            "phaseAdaptiveSigmaFloor": self.phaseAdaptiveSigmaFloor,
            "amplitudeAdaptiveSigmaFloor": self.amplitudeAdaptiveSigmaFloor,
            "phaseMinSigmaScale": self.phaseMinSigmaScale,
            "amplitudeMinSigmaScale": self.amplitudeMinSigmaScale,
            "wideGridSizeStart": self.wideGridSizeStart,
            "wideGridRampSteps": self.wideGridRampSteps,
            "linearResponseChunkSize": self.linearResponseChunkSize,
            "wideResponseChunkSize": self.wideResponseChunkSize,
            "responseReductionTileCap": self.responseReductionTileCap,
        }


def _select_evaluation(evaluation: BatchEvaluation, ids: torch.Tensor) -> BatchEvaluation:
    return BatchEvaluation(
        totalLoss=evaluation.totalLoss[ids],
        shapeLoss=evaluation.shapeLoss[ids],
        efficiencyLoss=evaluation.efficiencyLoss[ids],
        wideSupportLoss=evaluation.wideSupportLoss[ids],
        linearResponse=evaluation.linearResponse[ids],
        targetAZEL=(evaluation.targetAZEL[0][ids], evaluation.targetAZEL[1][ids]),
        targetMode=evaluation.targetMode,
        diagnostics=EvaluationDiagnostics.zeroedFrom(evaluation.diagnostics),
    )


def _merge_evaluations(evaluations: list[BatchEvaluation]) -> BatchEvaluation:
    if not evaluations:
        raise ValueError("evaluations must not be empty")
    reference = evaluations[0]
    return BatchEvaluation(
        totalLoss=torch.cat([evaluation.totalLoss for evaluation in evaluations], dim=0),
        shapeLoss=torch.cat([evaluation.shapeLoss for evaluation in evaluations], dim=0),
        efficiencyLoss=torch.cat([evaluation.efficiencyLoss for evaluation in evaluations], dim=0),
        wideSupportLoss=torch.cat([evaluation.wideSupportLoss for evaluation in evaluations], dim=0),
        linearResponse=torch.cat([evaluation.linearResponse for evaluation in evaluations], dim=0),
        targetAZEL=(
            torch.cat([evaluation.targetAZEL[0] for evaluation in evaluations], dim=0),
            torch.cat([evaluation.targetAZEL[1] for evaluation in evaluations], dim=0),
        ),
        targetMode=reference.targetMode,
        diagnostics=EvaluationDiagnostics.merge(
            [evaluation.diagnostics for evaluation in evaluations]
        ),
    )


@dataclass
class EvolutionController:
    config: EvolutionConfig
    targetSpec: TargetLike
    arraySpec: ArraySpec
    lossParams: LossConfig
    experimentName: str = "evo_1"
    archiveRoot: str | Path = "data/archive"
    loggingConfig: LoggingConfig = field(default_factory=LoggingConfig)
    checkpointConfig: CheckpointConfig = field(default_factory=CheckpointConfig)
    workerConfig: WorkerConfig = field(default_factory=WorkerConfig)
    targetMode: TargetMode = "auto"
    sourceConfigPath: str | Path | None = None
    resolvedConfig: dict[str, Any] | None = None
    writer: SummaryWriter | None = None
    datasetWriter: AsyncWriterPool | None = None
    checkpointWriter: AsyncWriterPool | None = None
    datasetBuffer: list[dict[str, Any]] = field(default_factory=list)
    datasetShardIndex: int = 0
    writerLogDir: str | Path | None = "runs"

    @property
    def archiveLocation(self) -> Path:
        return Path(self.archiveRoot) / self.experimentName

    @property
    def checkpointsLocation(self) -> Path:
        return self.archiveLocation / "checkpoints"

    @property
    def datasetLocation(self) -> Path:
        return self.archiveLocation / "dataset"

    @property
    def checkpointPath(self) -> Path:
        return self.checkpointsLocation / "latest.pt"

    @property
    def bestPath(self) -> Path:
        return self.archiveLocation / "best.pt"

    def runLocation(self, logDir: str | Path | None = None) -> Path | None:
        activeLogDir = self.writerLogDir if logDir is None else logDir
        if activeLogDir is None:
            return None
        return Path(activeLogDir) / self.experimentName

    def resetExperiment(self, logDir: str | Path | None = None) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

        if self.archiveLocation.exists():
            shutil.rmtree(self.archiveLocation)

        runLocation = self.runLocation(logDir)
        if runLocation is not None and runLocation.exists():
            shutil.rmtree(runLocation)

    def listStepPaths(self) -> list[Path]:
        return sorted(self.archiveLocation.glob("step_*.pt"))

    def _startWriters(self) -> None:
        asyncEnabled = self.workerConfig.asyncIO
        self.datasetWriter = AsyncWriterPool(
            workerCount=self.workerConfig.datasetWriterWorkers,
            queueSize=self.workerConfig.ioQueueSize,
            enabled=asyncEnabled,
        )
        self.checkpointWriter = AsyncWriterPool(
            workerCount=self.workerConfig.checkpointWriterWorkers,
            queueSize=self.workerConfig.ioQueueSize,
            enabled=asyncEnabled,
        )

    def _closeWriters(self) -> None:
        if self.datasetWriter is not None:
            self.datasetWriter.close()
            self.datasetWriter = None
        if self.checkpointWriter is not None:
            self.checkpointWriter.close()
            self.checkpointWriter = None

    def _writeRunConfig(self, runLocation: Path, resume: bool = False) -> None:
        configPath = runLocation / "config.yaml"
        runLocation.mkdir(parents=True, exist_ok=True)

        if configPath.exists():
            with configPath.open("r", encoding="utf-8") as handle:
                existing = _normalize_config_payload(yaml.safe_load(handle) or {})
            if self.resolvedConfig is not None:
                active = _normalize_config_payload(self.resolvedConfig)
                existingComparable = _comparison_config_payload(
                    existing,
                    allowResumeMutation=resume,
                )
                activeComparable = _comparison_config_payload(
                    active,
                    allowResumeMutation=resume,
                )
                if existingComparable != activeComparable:
                    raise ValueError(f"config mismatch for resumed experiment at {configPath}")
                if resume and existing != active:
                    with configPath.open("w", encoding="utf-8") as handle:
                        yaml.safe_dump(active, handle, sort_keys=False)
            return

        if self.sourceConfigPath is not None:
            shutil.copy2(self.sourceConfigPath, configPath)
            return

        if self.resolvedConfig is not None:
            with configPath.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(self.resolvedConfig, handle, sort_keys=False)

    def _serializeMetrics(
        self,
        step: int,
        scores: torch.Tensor,
        evaluation: BatchEvaluation,
        schedulerState: SchedulerState,
        lossParams: LossConfig,
        parentPhaseStd: float,
        parentAmplitudeStd: float,
    ) -> dict:
        bestIdx = int(torch.argmin(scores).item())
        worstIdx = int(torch.argmax(scores).item())
        summary = {
            "step": step,
            "batchSize": int(scores.shape[0]),
            "bestIdx": bestIdx,
            "worstIdx": worstIdx,
            "bestScore": float(scores[bestIdx].item()),
            "meanScore": float(scores.mean().item()),
            "medianScore": float(scores.median().item()),
            "worstScore": float(scores[worstIdx].item()),
            "stdScore": float(scores.std(unbiased=False).item()),
            "shapeLoss": float(evaluation.shapeLoss.mean().item()),
            "efficiencyLoss": float(evaluation.efficiencyLoss.mean().item()),
            "wideSupportLoss": float(evaluation.wideSupportLoss.mean().item()),
            "phaseSigma": schedulerState.currentPhaseSigma,
            "amplitudeSigma": schedulerState.currentAmplitudeSigma,
            "randomFraction": schedulerState.currentRandomFraction,
            "wideGridSize": int(lossParams.wide_grid_size),
            "parentPhaseStd": parentPhaseStd,
            "parentAmplitudeStd": parentAmplitudeStd,
        }
        diagnostics = evaluation.diagnostics
        summary.update(
            {
                "evalTargetProjectionMS": diagnostics.targetProjectionMS,
                "evalLinearResponseMS": diagnostics.linearResponseMS,
                "evalWideResponseMS": diagnostics.wideResponseMS,
                "evalTotalMS": diagnostics.totalMS,
                "usedSharedTargetFastPath": float(diagnostics.usedSharedTargetFastPath),
            }
        )
        if diagnostics.gpuMaxMemoryMB is not None:
            summary["gpuMaxMemoryMB"] = diagnostics.gpuMaxMemoryMB
        return summary

    def _logMetrics(self, step: int, summary: dict[str, Any]) -> None:
        if self.writer is None:
            return

        self.writer.add_scalar("Score/Best", summary["bestScore"], step)
        self.writer.add_scalar("Score/Mean", summary["meanScore"], step)
        self.writer.add_scalar("Score/Worst", summary["worstScore"], step)
        self.writer.add_scalar("Score/Std", summary["stdScore"], step)
        self.writer.add_scalar("Loss/Shape", summary["shapeLoss"], step)
        self.writer.add_scalar("Loss/Efficiency", summary["efficiencyLoss"], step)
        self.writer.add_scalar("Loss/WideSupport", summary["wideSupportLoss"], step)
        self.writer.add_scalar("Param/PhaseSigma", summary["phaseSigma"], step)
        self.writer.add_scalar("Param/AmplitudeSigma", summary["amplitudeSigma"], step)
        self.writer.add_scalar("Param/RandomFraction", summary["randomFraction"], step)
        self.writer.add_scalar("Param/WideGridSize", summary["wideGridSize"], step)
        self.writer.add_scalar("Parent/PhaseStd", summary["parentPhaseStd"], step)
        self.writer.add_scalar("Parent/AmplitudeStd", summary["parentAmplitudeStd"], step)

        if step % 10 == 0:
            self.writer.add_scalar("Eval/TargetProjectionMS", summary["evalTargetProjectionMS"], step)
            self.writer.add_scalar("Eval/LinearResponseMS", summary["evalLinearResponseMS"], step)
            self.writer.add_scalar("Eval/WideResponseMS", summary["evalWideResponseMS"], step)
            self.writer.add_scalar("Eval/TotalMS", summary["evalTotalMS"], step)
            self.writer.add_scalar("Eval/UsedSharedTargetFastPath", summary["usedSharedTargetFastPath"], step)
            if "gpuMaxMemoryMB" in summary:
                self.writer.add_scalar("GPU/MaxMemoryMB", summary["gpuMaxMemoryMB"], step)

    def _compactResponse(self, response: torch.Tensor) -> torch.Tensor:
        size = self.loggingConfig.responseCompactSize
        return (
            F.adaptive_avg_pool2d(response.unsqueeze(0).unsqueeze(0), (size, size))
            .squeeze(0)
            .squeeze(0)
            .detach()
            .cpu()
        )

    def _serializeResponsePayload(self, response: torch.Tensor) -> dict[str, Any]:
        responseCPU = response.detach().cpu()
        payload = {
            "stats": {
                "max": float(responseCPU.max().item()),
                "mean": float(responseCPU.mean().item()),
                "sum": float(responseCPU.sum().item()),
            }
        }
        if self.loggingConfig.logMode == "dataset_full":
            payload["map"] = responseCPU
        else:
            payload["map"] = self._compactResponse(response)
        return payload

    def _bufferDatasetRecords(self, step: int, batch: ArrayBatch, evaluation: BatchEvaluation) -> None:
        if self.loggingConfig.logMode not in {"dataset_compact", "dataset_full"}:
            return

        sharedTarget = (
            serializeTarget(self.targetSpec) if isinstance(self.targetSpec, TargetSpec) else None
        )

        for sampleIdx in range(batch.batchSize):
            record = {
                "step": step,
                "sampleID": sampleIdx,
                "array": batch.serializeBatchSample(sampleIdx),
                "response": self._serializeResponsePayload(evaluation.linearResponse[sampleIdx]),
                "loss": {
                    "total": float(evaluation.totalLoss[sampleIdx].item()),
                    "shape": float(evaluation.shapeLoss[sampleIdx].item()),
                    "efficiency": float(evaluation.efficiencyLoss[sampleIdx].item()),
                    "wideSupport": float(evaluation.wideSupportLoss[sampleIdx].item()),
                },
            }

            if sharedTarget is None:
                record["target"] = fetchTargetSample(self.targetSpec, sampleIdx).serializeTargetSpec()
            self.datasetBuffer.append(record)

        shouldFlush = (
            (step + 1) % self.loggingConfig.datasetFlushEverySteps == 0
            and len(self.datasetBuffer) > 0
        )
        if shouldFlush:
            payload = {
                "experimentName": self.experimentName,
                "sharedTarget": sharedTarget,
                "records": self.datasetBuffer,
            }
            path = self.datasetLocation / f"shard_{self.datasetShardIndex:05d}.pt"
            assert self.datasetWriter is not None
            self.datasetWriter.submit(path, payload)
            self.datasetShardIndex += 1
            self.datasetBuffer = []

    def _flushDatasetBuffer(self) -> None:
        if not self.datasetBuffer:
            return

        sharedTarget = serializeTarget(self.targetSpec) if isinstance(self.targetSpec, TargetSpec) else None
        payload = {
            "experimentName": self.experimentName,
            "sharedTarget": sharedTarget,
            "records": self.datasetBuffer,
        }
        path = self.datasetLocation / f"shard_{self.datasetShardIndex:05d}.pt"
        assert self.datasetWriter is not None
        self.datasetWriter.submit(path, payload)
        self.datasetShardIndex += 1
        self.datasetBuffer = []

    def _queueCheckpoint(
        self,
        step: int,
        batch: ArrayBatch,
        schedulerState: SchedulerState,
        scores: torch.Tensor,
        bestScoreOverall: float | None,
        bestStepOverall: int | None,
        bestSampleOverall: dict[str, Any] | None,
        historyBest: list[float],
        historyMean: list[float],
        historyWorst: list[float],
    ) -> None:
        if self.checkpointConfig.checkpointMode == "off":
            return
        if (step + 1) % self.checkpointConfig.checkpointEverySteps != 0:
            return

        payload = {
            "step": step,
            "batch": batch.serializeBatch(),
            "scores": scores.detach().cpu(),
            "schedulerState": schedulerState.serialize(),
            "bestScoreOverall": bestScoreOverall,
            "bestStepOverall": bestStepOverall,
            "bestSampleOverall": bestSampleOverall,
            "history": {
                "bestScore": list(historyBest),
                "meanScore": list(historyMean),
                "worstScore": list(historyWorst),
            },
        }
        assert self.checkpointWriter is not None
        self.checkpointWriter.submit(self.checkpointPath, payload)

    def _queueBestSample(self, samplePayload: dict[str, Any]) -> None:
        assert self.checkpointWriter is not None
        self.checkpointWriter.submit(self.bestPath, samplePayload)

    def _wideGridSizeAt(self, step: int) -> int:
        finalSize = int(self.lossParams.wide_grid_size)
        startSize = self.config.wideGridSizeStart
        if startSize is None or startSize == finalSize:
            return finalSize

        rampSteps = self.config.wideGridRampSteps
        if rampSteps is None:
            rampSteps = max(1, int(0.25 * self.config.evolutionSteps))
        if rampSteps <= 0 or step >= rampSteps:
            return finalSize

        progress = step / rampSteps
        interpolated = startSize + (finalSize - startSize) * progress
        return int(round(interpolated))

    def _lossParamsForStep(self, step: int) -> LossConfig:
        activeWideGridSize = self._wideGridSizeAt(step)
        if activeWideGridSize == self.lossParams.wide_grid_size:
            return self.lossParams
        return replace(self.lossParams, wide_grid_size=activeWideGridSize)

    def loadResumeState(self, device: torch.device) -> dict | None:
        if self.checkpointPath.exists():
            payload = torch.load(self.checkpointPath, weights_only=False)
            batch = ArrayBatch.fromSerializedBatch(payload["batch"]).to(device)
            return {
                "batch": batch,
                "startStep": int(payload["step"]) + 1,
                "schedulerState": SchedulerState.fromPayload(payload.get("schedulerState")),
                "bestScoreOverall": payload.get("bestScoreOverall"),
                "bestStepOverall": payload.get("bestStepOverall"),
                "bestSampleOverall": payload.get("bestSampleOverall"),
                "historyBest": list(payload.get("history", {}).get("bestScore", [])),
                "historyMean": list(payload.get("history", {}).get("meanScore", [])),
                "historyWorst": list(payload.get("history", {}).get("worstScore", [])),
            }

        stepPaths = self.listStepPaths()
        if not stepPaths:
            return None

        historyBest: list[float] = []
        historyMean: list[float] = []
        historyWorst: list[float] = []
        bestScoreOverall: float | None = None
        bestStepOverall: int | None = None
        bestSampleOverall: dict | None = None

        for stepPath in stepPaths:
            payload = torch.load(stepPath, weights_only=False)
            summary = payload["summary"]
            historyBest.append(float(summary["bestScore"]))
            historyMean.append(float(summary["meanScore"]))
            historyWorst.append(float(summary["worstScore"]))

            bestScore = float(summary["bestScore"])
            if bestScoreOverall is None or bestScore < bestScoreOverall:
                bestScoreOverall = bestScore
                bestStepOverall = int(payload["step"])
                bestSampleOverall = {
                    **payload["bestSample"],
                    "step": int(payload["step"]),
                }

        latestPayload = torch.load(stepPaths[-1], weights_only=False)
        batch = ArrayBatch.fromSerializedBatch(latestPayload["batch"]).to(device)
        return {
            "batch": batch,
            "startStep": int(latestPayload["step"]) + 1,
            "schedulerState": SchedulerState.fromPayload(latestPayload.get("schedulerState")),
            "bestScoreOverall": bestScoreOverall,
            "bestStepOverall": bestStepOverall,
            "bestSampleOverall": bestSampleOverall,
            "historyBest": historyBest,
            "historyMean": historyMean,
            "historyWorst": historyWorst,
        }

    def evaluate(self, batch: ArrayBatch, lossParams: LossConfig | None = None) -> BatchEvaluation:
        activeLossParams = self.lossParams if lossParams is None else lossParams
        return evaluateBatch(
            batch=batch,
            target=self.targetSpec,
            params=activeLossParams,
            targetMode=self.targetMode,
            linearResponseChunkSize=self.config.linearResponseChunkSize,
            wideResponseChunkSize=self.config.wideResponseChunkSize,
            responseReductionTileCap=self.config.responseReductionTileCap,
        )

    def _populationForStep(self, batch: ArrayBatch, step: int) -> PopulationState:
        lossParams = self._lossParamsForStep(step)
        return PopulationState(batch=batch, evaluation=self.evaluate(batch, lossParams), lossParams=lossParams)

    def _initialSchedulerState(self) -> SchedulerState:
        phaseSigma, amplitudeSigma = self.config.baseSigmaAt(0)
        return SchedulerState(
            currentPhaseSigma=max(self.config.phaseMinSigma, phaseSigma),
            currentAmplitudeSigma=max(self.config.amplitudeMinSigma, amplitudeSigma),
            currentRandomFraction=self.config.scheduledRandomFraction(0),
        )

    def _selectionWeights(self, count: int, device: torch.device) -> torch.Tensor:
        return torch.arange(count, 0, -1, device=device, dtype=torch.float32)

    def sample(
        self,
        parentIDs: torch.Tensor,
        childCount: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if childCount <= 0:
            return parentIDs[:0]
        weights = self._selectionWeights(parentIDs.shape[0], device)
        samples = torch.multinomial(weights, childCount, replacement=True, generator=generator)
        return parentIDs[samples]

    def sampleParentPairs(
        self,
        parentIDs: torch.Tensor,
        childCount: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if childCount <= 0:
            empty = parentIDs[:0]
            return empty, empty

        weights = self._selectionWeights(parentIDs.shape[0], device)
        primaryPositions = torch.multinomial(
            weights,
            childCount,
            replacement=True,
            generator=generator,
        )
        secondaryPositions = torch.multinomial(
            weights,
            childCount,
            replacement=True,
            generator=generator,
        )
        if parentIDs.shape[0] > 1:
            duplicateMask = primaryPositions == secondaryPositions
            secondaryPositions = secondaryPositions.clone()
            secondaryPositions[duplicateMask] = (
                secondaryPositions[duplicateMask] + 1
            ) % parentIDs.shape[0]
        return parentIDs[primaryPositions], parentIDs[secondaryPositions]

    def initEvolution(self, dtype: torch.dtype, device: torch.device) -> ArrayBatch:
        targetLLA = None
        if self.config.initialWeightsType == "directed":
            targetLLA = inferTargetCenter(self.targetSpec).to(device=device, dtype=dtype)
        return generateBatch(
            self.arraySpec,
            batchSize=self.config.batchSize,
            device=device,
            dtype=dtype,
            weightsType=self.config.initialWeightsType,
            targetLLA=targetLLA,
            generator=self.config.generator,
        )

    def _parentDiversity(self, parentBatch: ArrayBatch) -> tuple[float, float]:
        if parentBatch.batchSize == 0:
            return 0.0, 0.0

        phase = parentBatch.weights.angle()
        phaseReference = phase[:1]
        phaseDelta = torch.remainder(phase - phaseReference + torch.pi, 2 * torch.pi) - torch.pi
        amplitude = parentBatch.weights.abs()
        return (
            float(phaseDelta.std(unbiased=False).item()),
            float(amplitude.std(unbiased=False).item()),
        )

    def _improvementEpsilon(self, bestScoreOverall: float | None) -> float:
        if bestScoreOverall is None:
            return 0.0
        return max(1e-8, 1e-6 * abs(bestScoreOverall))

    def _advanceScheduler(
        self,
        step: int,
        currentState: SchedulerState,
        parentBatch: ArrayBatch,
        improved: bool,
    ) -> SchedulerState:
        nextStep = step + 1
        basePhaseSigma, baseAmplitudeSigma = self.config.baseSigmaAt(nextStep)
        parentPhaseStd, parentAmplitudeStd = self._parentDiversity(parentBatch)

        phaseFloor = self.config.phaseMinSigma
        if self.config.phaseAdaptiveSigmaFloor:
            phaseFloor = max(phaseFloor, self.config.phaseMinSigmaScale * parentPhaseStd)

        amplitudeFloor = self.config.amplitudeMinSigma
        if self.config.amplitudeAdaptiveSigmaFloor:
            amplitudeFloor = max(
                amplitudeFloor,
                self.config.amplitudeMinSigmaScale * parentAmplitudeStd,
            )

        nextPhaseSigma = max(basePhaseSigma, phaseFloor)
        nextAmplitudeSigma = max(baseAmplitudeSigma, amplitudeFloor)
        nextRandomFraction = self.config.scheduledRandomFraction(nextStep)

        boostStepsRemaining = max(currentState.boostStepsRemaining - 1, 0)
        stagnationCount = 0 if improved else currentState.stagnationCount + 1
        if improved:
            boostStepsRemaining = 0

        if (
            not improved
            and boostStepsRemaining == 0
            and self.config.sigmaBoostDuration > 0
            and stagnationCount >= self.config.stagnationWindow
        ):
            boostStepsRemaining = self.config.sigmaBoostDuration
            stagnationCount = 0

        if boostStepsRemaining > 0:
            if self.config.phaseSigma > 0:
                nextPhaseSigma = min(
                    self.config.phaseSigma,
                    nextPhaseSigma * self.config.sigmaBoostMultiplier,
                )
            if self.config.amplitudeSigma > 0:
                nextAmplitudeSigma = min(
                    self.config.amplitudeSigma,
                    nextAmplitudeSigma * self.config.sigmaBoostMultiplier,
                )
            nextRandomFraction = max(nextRandomFraction, self.config.sigmaBoostRandomFraction)

        return SchedulerState(
            currentPhaseSigma=nextPhaseSigma,
            currentAmplitudeSigma=nextAmplitudeSigma,
            currentRandomFraction=nextRandomFraction,
            stagnationCount=stagnationCount,
            boostStepsRemaining=boostStepsRemaining,
        )

    def evolutionStep(
        self,
        step: int,
        population: PopulationState,
        schedulerState: SchedulerState,
        sortedScoresIDs: torch.Tensor | None = None,
    ) -> PopulationState:
        batch = population.batch
        device, dtype = batch.device, batch.dtype
        counts = self.config.countsForRandomFraction(schedulerState.currentRandomFraction)
        if sortedScoresIDs is None:
            sortedScoresIDs = torch.argsort(population.scores, dim=0, descending=False)
        parentIDs = sortedScoresIDs[: self.config.parentPoolCount]

        clonedEvaluations: list[BatchEvaluation] = []
        nextBatches: list[ArrayBatch] = []
        offspringBatches: list[ArrayBatch] = []

        if counts.cloneCount > 0:
            cloneIDs = sortedScoresIDs[: counts.cloneCount]
            clones = batch.fetch(cloneIDs)
            nextBatches.append(clones)
            if self._wideGridSizeAt(step) == self._wideGridSizeAt(step + 1):
                clonedEvaluations.append(_select_evaluation(population.evaluation, cloneIDs))

        if counts.crossoverCount > 0:
            primaryIDs, secondaryIDs = self.sampleParentPairs(
                parentIDs,
                counts.crossoverCount,
                device,
                generator=self.config.generator,
            )
            crossoverParents = batch.fetch(primaryIDs)
            crossoverPartners = batch.fetch(secondaryIDs)
            crossoverChildren = crossoverParents.crossoverWeights(
                crossoverPartners,
                generator=self.config.generator,
            )
            nextBatches.append(crossoverChildren)
            offspringBatches.append(crossoverChildren)

        if counts.mutateCount > 0:
            mutationIDs = self.sample(
                parentIDs,
                counts.mutateCount,
                device,
                generator=self.config.generator,
            )
            mutationParents = batch.fetch(mutationIDs)
            mutationChildren = mutationParents.mutateWeights(
                schedulerState.currentPhaseSigma,
                schedulerState.currentAmplitudeSigma,
                generator=self.config.generator,
            )
            nextBatches.append(mutationChildren)
            offspringBatches.append(mutationChildren)

        if counts.randomCount > 0:
            randomChildren = generateBatch(
                self.arraySpec,
                counts.randomCount,
                device,
                dtype,
                weightsType="random",
                generator=self.config.generator,
            )
            nextBatches.append(randomChildren)
            offspringBatches.append(randomChildren)

        nextBatch = merge(nextBatches)
        nextLossParams = self._lossParamsForStep(step + 1)
        if (
            counts.cloneCount > 0
            and clonedEvaluations
            and nextLossParams.wide_grid_size == population.lossParams.wide_grid_size
        ):
            if offspringBatches:
                offspringBatch = merge(offspringBatches)
                offspringEvaluation = self.evaluate(offspringBatch, nextLossParams)
                nextEvaluation = _merge_evaluations(clonedEvaluations + [offspringEvaluation])
            else:
                nextEvaluation = _merge_evaluations(clonedEvaluations)
        else:
            nextEvaluation = self.evaluate(nextBatch, nextLossParams)

        return PopulationState(
            batch=nextBatch,
            evaluation=nextEvaluation,
            lossParams=nextLossParams,
        )

    def train(
        self,
        dtype: torch.dtype,
        device: torch.device,
        logDir: str | Path | None = None,
        plotProjection: bool | None = None,
        resume: bool | None = None,
    ) -> dict:
        activeLogDir = self.writerLogDir if logDir is None else logDir
        activePlotProjection = False if plotProjection is None else plotProjection
        activeResume = True if resume is None else resume

        if not activeResume:
            self.resetExperiment(activeLogDir)

        self.writerLogDir = activeLogDir
        self.datasetShardIndex = 0
        self.datasetBuffer = []

        if self.writer is not None:
            self.writer.close()
            self.writer = None

        runPath = self.runLocation(activeLogDir)
        if runPath is not None:
            self._writeRunConfig(runPath, resume=activeResume)

        self._startWriters()

        try:
            resumeState = self.loadResumeState(device=device) if activeResume else None
            if resumeState is None:
                batch = self.initEvolution(dtype=dtype, device=device)
                startStep = 0
                schedulerState = self._initialSchedulerState()
                bestScoreOverall: float | None = None
                bestStepOverall: int | None = None
                bestSampleOverall: dict | None = None
                historyBest: list[float] = []
                historyMean: list[float] = []
                historyWorst: list[float] = []
            else:
                batch = resumeState["batch"]
                startStep = int(resumeState["startStep"])
                schedulerState = resumeState["schedulerState"] or self._initialSchedulerState()
                bestScoreOverall = resumeState["bestScoreOverall"]
                bestStepOverall = resumeState["bestStepOverall"]
                bestSampleOverall = resumeState["bestSampleOverall"]
                historyBest = resumeState["historyBest"]
                historyMean = resumeState["historyMean"]
                historyWorst = resumeState["historyWorst"]

            if startStep >= self.config.evolutionSteps:
                return {
                    "experimentName": self.experimentName,
                    "bestScoreOverall": bestScoreOverall,
                    "bestStepOverall": bestStepOverall,
                    "bestSampleOverall": bestSampleOverall,
                    "history": {
                        "bestScore": torch.tensor(historyBest),
                        "meanScore": torch.tensor(historyMean),
                        "worstScore": torch.tensor(historyWorst),
                    },
                }

            if (
                activeLogDir is not None
                and self.loggingConfig.logMode != "off"
                and runPath is not None
            ):
                self.writer = SummaryWriter(log_dir=str(runPath), purge_step=startStep)

            population = self._populationForStep(batch, startStep)

            pbar = tqdm(
                range(startStep, self.config.evolutionSteps),
                desc=f"Evolving {self.experimentName}",
            )
            for step in pbar:
                evaluation = population.evaluation
                scores = population.scores
                sortedScoresIDs = torch.argsort(scores, dim=0, descending=False)
                parentIDs = sortedScoresIDs[: self.config.parentPoolCount]
                parentBatch = population.batch.fetch(parentIDs)
                parentPhaseStd, parentAmplitudeStd = self._parentDiversity(parentBatch)
                summary = self._serializeMetrics(
                    step,
                    scores,
                    evaluation,
                    schedulerState,
                    population.lossParams,
                    parentPhaseStd=parentPhaseStd,
                    parentAmplitudeStd=parentAmplitudeStd,
                )
                self._logMetrics(step, summary)

                bestIdx = int(summary["bestIdx"])
                bestScore = float(summary["bestScore"])
                meanScore = float(summary["meanScore"])
                historyBest.append(bestScore)
                historyMean.append(meanScore)
                historyWorst.append(float(summary["worstScore"]))
                pbar.set_postfix({"best": f"{bestScore:.4f}", "mean": f"{meanScore:.4f}"})

                if activePlotProjection:
                    projectResponseOnTarget(
                        batch=population.batch,
                        target=self.targetSpec,
                        sampleID=bestIdx,
                        normalizedInputs=True,
                    )

                improvementEpsilon = self._improvementEpsilon(bestScoreOverall)
                improved = bestScoreOverall is None or bestScore < bestScoreOverall - improvementEpsilon
                if improved:
                    bestScoreOverall = bestScore
                    bestStepOverall = step
                    bestSampleOverall = {
                        **population.batch.serializeBatchSample(bestIdx),
                        "score": bestScore,
                        "step": step,
                    }
                    self._queueBestSample(bestSampleOverall)

                self._bufferDatasetRecords(step, population.batch, evaluation)

                if step < self.config.evolutionSteps - 1:
                    nextSchedulerState = self._advanceScheduler(
                        step,
                        schedulerState,
                        parentBatch,
                        improved,
                    )
                    nextPopulation = self.evolutionStep(
                        step,
                        population,
                        nextSchedulerState,
                        sortedScoresIDs=sortedScoresIDs,
                    )
                    checkpointBatch = nextPopulation.batch
                    checkpointScores = nextPopulation.scores
                    checkpointSchedulerState = nextSchedulerState
                else:
                    nextPopulation = None
                    checkpointBatch = population.batch
                    checkpointScores = scores
                    checkpointSchedulerState = schedulerState

                self._queueCheckpoint(
                    step=step,
                    batch=checkpointBatch,
                    schedulerState=checkpointSchedulerState,
                    scores=checkpointScores,
                    bestScoreOverall=bestScoreOverall,
                    bestStepOverall=bestStepOverall,
                    bestSampleOverall=bestSampleOverall,
                    historyBest=historyBest,
                    historyMean=historyMean,
                    historyWorst=historyWorst,
                )

                if nextPopulation is not None:
                    population = nextPopulation
                    schedulerState = nextSchedulerState

            self._flushDatasetBuffer()

            finalPayload = {
                "experimentName": self.experimentName,
                "bestScoreOverall": bestScoreOverall,
                "bestStepOverall": bestStepOverall,
                "bestSampleOverall": bestSampleOverall,
                "history": {
                    "bestScore": torch.tensor(historyBest),
                    "meanScore": torch.tensor(historyMean),
                    "worstScore": torch.tensor(historyWorst),
                },
            }
            assert self.checkpointWriter is not None
            self.checkpointWriter.submit(self.archiveLocation / "final.pt", finalPayload)
            if self.datasetWriter is not None:
                self.datasetWriter.flush()
            if self.checkpointWriter is not None:
                self.checkpointWriter.flush()

            return finalPayload
        finally:
            if self.writer:
                self.writer.close()
                self.writer = None
            self._closeWriters()
