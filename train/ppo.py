from __future__ import annotations

import argparse
import os
import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from model.base import ActorCriticModel, ModelContext, ModelInput
from model.config import ModelConfig, modelConfigToDict
from model.registry import build_model
from scripts.arrayBatch import ArrayBatch, _normalize_amplitudes
from scripts.arraySpec import ArraySpec
from scripts.batchFactory import generateBatch
from scripts.coordinateTransforms import getECEFtoENUMapping, semiMajorAxis
from scripts.targetSpec import TargetBatch, TargetSpec, fetchTargetSample
from train.config import (
    PPOConfig,
    PPORunConfig,
    TargetConfig,
    loadPPORunConfig,
    loadTargetReferenceForConfig,
    ppoRunConfigToDict,
    resolvePPOTargetRecordPaths,
)
from train.evolve import (
    AsyncWriterPool,
    CheckpointConfig,
    DeviceConfig,
    ExperimentConfig,
    LoggingConfig,
    WorkerConfig,
)
from train.objective import BatchEvaluation, LossConfig, evaluateBatch
from train.objective_v2 import BatchEvaluationV2, LossConfigV2, evaluateBatchV2
from train.evaluation_utils import (
    ObjectiveVersion,
    evaluation_diagnostic_means,
    evaluation_weighted_loss_means,
    evaluation_loss_record,
    loss_term_keys_for_objective,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:

    class SummaryWriter:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            pass

        def add_scalar(self, *args, **kwargs) -> None:
            pass

        def close(self) -> None:
            pass


try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:

    class _TqdmFallback:
        def __init__(self, iterable=None, *args, **kwargs) -> None:
            self._iterable = iterable

        def __iter__(self):
            if self._iterable is None:
                return iter(())
            return iter(self._iterable)

        def set_postfix(self, *args, **kwargs) -> None:
            pass

        def set_postfix_str(self, *args, **kwargs) -> None:
            pass

        def refresh(self) -> None:
            pass

        def update(self, n: int = 1) -> None:
            pass

        def close(self) -> None:
            pass

    def tqdm(iterable=None, *args, **kwargs):  # type: ignore[override]
        return _TqdmFallback(iterable, *args, **kwargs)


_LOG_STD_MIN = -5.0
_LOG_STD_MAX = 2.0
_LOG_RATIO_MIN = -20.0
_LOG_RATIO_MAX = 20.0


@dataclass
class PPOStepBatch:
    modelInputs: ModelInput
    targetBatch: TargetBatch
    actions: torch.Tensor
    oldLogProb: torch.Tensor
    oldValue: torch.Tensor
    reward: torch.Tensor
    evaluation: BatchEvaluation | BatchEvaluationV2
    arrayBatch: ArrayBatch


@dataclass
class CorpusBlock:
    epochIndex: int
    blockStart: int
    orderIndices: list[int]
    targets: list[TargetSpec]
    cursor: int = 0
    loadSeconds: float = 0.0

    @property
    def size(self) -> int:
        return len(self.orderIndices)


@dataclass
class PPOCorpusLoader:
    recordPaths: list[Path]
    targetConfig: TargetConfig
    rolloutBatchSize: int
    seed: int
    progressDesc: str = "Loading PPO corpus"
    prefetchExecutor: ThreadPoolExecutor | None = None
    epochIndex: int = 0
    epochOrder: list[int] = field(default_factory=list)
    activeBlock: CorpusBlock | None = None
    nextBlock: CorpusBlock | None = None
    nextBlockFuture: Future[CorpusBlock] | None = None
    nextBlockRequest: tuple[int, int] | None = None
    targetsSeen: int = 0

    def __post_init__(self) -> None:
        if not self.recordPaths:
            raise ValueError("recordPaths must not be empty")
        if self.targetConfig.loaderWorkers == "auto":
            cpuCount = os.cpu_count() or 1
            self.resolvedLoaderWorkers = min(8, max(1, cpuCount // 2))
        else:
            self.resolvedLoaderWorkers = int(self.targetConfig.loaderWorkers)
        if self.targetConfig.prefetchNextBlock:
            self.prefetchExecutor = ThreadPoolExecutor(max_workers=1)
        self.epochOrder = self._make_epoch_order(self.epochIndex)

    @property
    def blockSize(self) -> int:
        return int(self.targetConfig.blockSize)

    def close(self) -> None:
        if self.prefetchExecutor is not None:
            self.prefetchExecutor.shutdown(wait=True)
            self.prefetchExecutor = None
        self.nextBlockFuture = None
        self.nextBlock = None

    def _make_epoch_order(self, epochIndex: int) -> list[int]:
        order = list(range(len(self.recordPaths)))
        if self.targetConfig.shuffleEachEpoch:
            rng = torch.Generator(device="cpu")
            rng.manual_seed(int(self.seed) + epochIndex)
            permutation = torch.randperm(len(order), generator=rng).tolist()
            order = [order[idx] for idx in permutation]
        return order

    def _total_blocks_for_epoch(self, epochOrder: list[int] | None = None) -> int:
        activeOrder = self.epochOrder if epochOrder is None else epochOrder
        return max(1, ceil(len(activeOrder) / self.blockSize))

    def _validate_block_shapes(self, targets: list[TargetSpec], request: tuple[int, int]) -> None:
        targetShapes = {tuple(target.targetShape) for target in targets}
        if len(targetShapes) != 1:
            raise ValueError(
                "all PPO targets in a block must share the same spatial shape "
                f"(request={request}, shapes={sorted(targetShapes)})"
            )

    def _load_block(
        self,
        epochIndex: int,
        blockStart: int,
        showProgress: bool = False,
    ) -> CorpusBlock:
        epochOrder = self.epochOrder if epochIndex == self.epochIndex else self._make_epoch_order(epochIndex)
        orderIndices = epochOrder[blockStart : blockStart + self.blockSize]
        if not orderIndices:
            raise ValueError(f"attempted to load an empty PPO corpus block at epoch {epochIndex}")

        def load_one(recordIndex: int) -> TargetSpec:
            return loadTargetReferenceForConfig(self.recordPaths[recordIndex], self.targetConfig)

        startedAt = time.perf_counter()
        if len(orderIndices) == 1:
            targets = [load_one(orderIndices[0])]
        else:
            workerCount = min(self.resolvedLoaderWorkers, len(orderIndices))
            with ThreadPoolExecutor(max_workers=workerCount) as executor:
                iterator = executor.map(load_one, orderIndices)
                if showProgress:
                    iterator = tqdm(
                        iterator,
                        total=len(orderIndices),
                        desc=self.progressDesc,
                        dynamic_ncols=True,
                    )
                targets = list(iterator)
        loadSeconds = time.perf_counter() - startedAt
        request = (epochIndex, blockStart)
        self._validate_block_shapes(targets, request)
        return CorpusBlock(
            epochIndex=epochIndex,
            blockStart=blockStart,
            orderIndices=list(orderIndices),
            targets=targets,
            cursor=0,
            loadSeconds=loadSeconds,
        )

    def _next_block_request_after(self, block: CorpusBlock) -> tuple[int, int]:
        nextStart = block.blockStart + block.size
        nextEpoch = block.epochIndex
        if nextStart >= len(self.epochOrder) and nextEpoch == self.epochIndex:
            nextEpoch += 1
            nextStart = 0
        elif nextStart >= len(self._make_epoch_order(nextEpoch)):
            nextEpoch += 1
            nextStart = 0
        return nextEpoch, nextStart

    def _start_prefetch(self) -> None:
        if not self.targetConfig.prefetchNextBlock or self.prefetchExecutor is None:
            return
        if self.activeBlock is None or self.nextBlockFuture is not None or self.nextBlock is not None:
            return
        request = self._next_block_request_after(self.activeBlock)
        self.nextBlockRequest = request
        self.nextBlockFuture = self.prefetchExecutor.submit(self._load_block, request[0], request[1], False)

    def _ensure_initialized(self, showProgress: bool = False) -> None:
        if self.activeBlock is not None:
            return
        self.activeBlock = self._load_block(self.epochIndex, 0, showProgress=showProgress)
        self._start_prefetch()

    def _activate_next_block(self, showProgress: bool = False) -> None:
        if self.activeBlock is None:
            self._ensure_initialized(showProgress=showProgress)
            return
        request = self._next_block_request_after(self.activeBlock)
        prefetched = self.nextBlock is not None and self.nextBlockRequest == request
        if prefetched:
            nextBlock = self.nextBlock
            assert nextBlock is not None
            self.nextBlock = None
        elif self.nextBlockFuture is not None and self.nextBlockRequest == request:
            nextBlock = self.nextBlockFuture.result()
        else:
            nextBlock = self._load_block(request[0], request[1], showProgress=showProgress)

        if self.nextBlockFuture is not None:
            self.nextBlockFuture = None
        self.nextBlockRequest = None

        self.epochIndex = nextBlock.epochIndex
        self.epochOrder = self._make_epoch_order(self.epochIndex)
        self.activeBlock = nextBlock
        print(
            "Swapped PPO corpus block "
            f"-> epoch={self.activeBlock.epochIndex + 1} "
            f"block={self.activeBlock.blockStart // self.blockSize + 1}/{self._total_blocks_for_epoch()} "
            f"size={self.activeBlock.size} "
            f"prefetched={prefetched}",
            flush=True,
        )
        self._start_prefetch()

    def maybe_promote_prefetch(self) -> None:
        if self.nextBlockFuture is None or not self.nextBlockFuture.done():
            return
        self.nextBlock = self.nextBlockFuture.result()
        self.nextBlockFuture = None

    def next_target_batch(
        self,
        device: torch.device,
        dtype: torch.dtype,
        showProgress: bool = False,
    ) -> TargetBatch:
        self._ensure_initialized(showProgress=showProgress)
        self.maybe_promote_prefetch()

        selected: list[TargetSpec] = []
        remaining = self.rolloutBatchSize
        while remaining > 0:
            assert self.activeBlock is not None
            available = self.activeBlock.size - self.activeBlock.cursor
            if available == 0:
                self._activate_next_block(showProgress=False)
                self.maybe_promote_prefetch()
                continue

            take = min(remaining, available)
            sliceStart = self.activeBlock.cursor
            sliceEnd = sliceStart + take
            selected.extend(target.clone() for target in self.activeBlock.targets[sliceStart:sliceEnd])
            self.activeBlock.cursor = sliceEnd
            self.targetsSeen += take
            remaining -= take

            if self.activeBlock.cursor == self.activeBlock.size:
                self._activate_next_block(showProgress=False)
                self.maybe_promote_prefetch()

        return TargetBatch.fromTargetSpecs(selected).to(device, dtype)

    def status(self) -> dict[str, float | int | bool]:
        self._ensure_initialized(showProgress=False)
        self.maybe_promote_prefetch()
        assert self.activeBlock is not None
        prefetchReady = self.nextBlock is not None
        prefetchLoading = self.nextBlockFuture is not None
        return {
            "epoch": self.activeBlock.epochIndex + 1,
            "blockIndex": self.activeBlock.blockStart // self.blockSize + 1,
            "blockCount": self._total_blocks_for_epoch(),
            "targetsSeen": self.targetsSeen,
            "currentBlockSize": self.activeBlock.size,
            "blockLoadSeconds": self.activeBlock.loadSeconds,
            "prefetchReady": prefetchReady,
            "prefetchLoading": prefetchLoading,
        }

    def serializeState(self) -> dict[str, Any]:
        self._ensure_initialized(showProgress=False)
        self.maybe_promote_prefetch()
        assert self.activeBlock is not None
        nextBlock = self.nextBlock
        return {
            "epochIndex": self.epochIndex,
            "epochOrder": list(self.epochOrder),
            "targetsSeen": self.targetsSeen,
            "activeBlock": {
                "epochIndex": self.activeBlock.epochIndex,
                "blockStart": self.activeBlock.blockStart,
                "cursor": self.activeBlock.cursor,
                "orderIndices": list(self.activeBlock.orderIndices),
            },
            "nextBlock": None
            if nextBlock is None
            else {
                "epochIndex": nextBlock.epochIndex,
                "blockStart": nextBlock.blockStart,
                "orderIndices": list(nextBlock.orderIndices),
            },
            "nextBlockRequest": self.nextBlockRequest,
        }

    def restoreState(self, payload: dict[str, Any]) -> None:
        self.close()
        if self.targetConfig.prefetchNextBlock:
            self.prefetchExecutor = ThreadPoolExecutor(max_workers=1)

        self.epochIndex = int(payload["epochIndex"])
        self.epochOrder = [int(index) for index in payload["epochOrder"]]
        self.targetsSeen = int(payload.get("targetsSeen", 0))

        activePayload = payload["activeBlock"]
        self.activeBlock = self._load_block(
            int(activePayload["epochIndex"]),
            int(activePayload["blockStart"]),
            showProgress=False,
        )
        self.activeBlock.cursor = int(activePayload["cursor"])

        self.nextBlock = None
        self.nextBlockFuture = None
        self.nextBlockRequest = None
        nextPayload = payload.get("nextBlock")
        if nextPayload is not None:
            self.nextBlock = self._load_block(
                int(nextPayload["epochIndex"]),
                int(nextPayload["blockStart"]),
                showProgress=False,
            )
            self.nextBlockRequest = (
                int(nextPayload["epochIndex"]),
                int(nextPayload["blockStart"]),
            )
        elif payload.get("nextBlockRequest") is not None and self.targetConfig.prefetchNextBlock:
            request = payload["nextBlockRequest"]
            self.nextBlockRequest = (int(request[0]), int(request[1]))
            if self.prefetchExecutor is not None:
                self.nextBlockFuture = self.prefetchExecutor.submit(
                    self._load_block,
                    self.nextBlockRequest[0],
                    self.nextBlockRequest[1],
                    False,
                )


@dataclass
class PPOController:
    config: PPOConfig
    modelConfig: ModelConfig
    corpusLoader: PPOCorpusLoader
    arraySpec: ArraySpec
    lossParams: LossConfig | LossConfigV2
    objectiveVersion: ObjectiveVersion = "v1"
    experimentName: str = "ppo_1"
    archiveRoot: str | Path = "data/archive"
    loggingConfig: LoggingConfig = field(default_factory=LoggingConfig)
    checkpointConfig: CheckpointConfig = field(default_factory=CheckpointConfig)
    workerConfig: WorkerConfig = field(default_factory=WorkerConfig)
    sourceConfigPath: str | Path | None = None
    modelSourcePath: str | Path | None = None
    resolvedConfig: dict[str, Any] | None = None
    writerLogDir: str | Path | None = "runs"
    writer: SummaryWriter | None = None
    checkpointWriter: AsyncWriterPool | None = None
    model: ActorCriticModel | None = None
    optimizer: torch.optim.Optimizer | None = None
    templateBatch: ArrayBatch | None = None
    rng: torch.Generator = field(default_factory=lambda: torch.Generator(device="cpu"))

    def __post_init__(self) -> None:
        self.rng.manual_seed(int(self.config.seed))

    @property
    def archiveLocation(self) -> Path:
        return Path(self.archiveRoot) / self.experimentName

    @property
    def checkpointsLocation(self) -> Path:
        return self.archiveLocation / "checkpoints"

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

    def _startWriters(self) -> None:
        asyncEnabled = self.workerConfig.asyncIO
        self.checkpointWriter = AsyncWriterPool(
            workerCount=self.workerConfig.checkpointWriterWorkers,
            queueSize=self.workerConfig.ioQueueSize,
            enabled=asyncEnabled,
        )

    def _closeWriters(self) -> None:
        if self.checkpointWriter is not None:
            self.checkpointWriter.close()
            self.checkpointWriter = None

    def _writeRunArtifacts(self, runLocation: Path) -> None:
        runLocation.mkdir(parents=True, exist_ok=True)
        configPath = runLocation / "config.yaml"
        modelPath = runLocation / "model.yaml"
        activeConfig = (
            ppoRunConfigToDict(self._resolvedRunConfig())
            if self.resolvedConfig is None
            else self.resolvedConfig
        )

        if configPath.exists():
            with configPath.open("r", encoding="utf-8") as handle:
                existing = yaml.safe_load(handle) or {}
            if existing != activeConfig:
                raise ValueError(f"config mismatch for resumed experiment at {configPath}")

        with configPath.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(activeConfig, handle, sort_keys=False)

        if self.modelSourcePath is not None:
            shutil.copy2(self.modelSourcePath, modelPath)
        else:
            with modelPath.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(modelConfigToDict(self.modelConfig), handle, sort_keys=False)

    def _resolvedRunConfig(self) -> PPORunConfig:
        raise RuntimeError("resolved run config is not available on the controller")

    def _logMetrics(self, step: int, summary: dict[str, float]) -> None:
        if self.writer is None:
            return

        self.writer.add_scalar("Reward/Mean", summary["meanReward"], step)
        self.writer.add_scalar("Reward/Best", summary["bestReward"], step)
        self.writer.add_scalar("Loss/Total", summary["meanTotalLoss"], step)
        for name in loss_term_keys_for_objective(self.objectiveVersion):
            metricKey = f"mean{name[0].upper()}{name[1:]}Loss"
            if metricKey in summary:
                self.writer.add_scalar(f"Loss/{name[0].upper()}{name[1:]}", summary[metricKey], step)
        for key, value in summary.items():
            if key.startswith("mean") and key.endswith("Metric"):
                diagnosticName = key[len("mean") : -len("Metric")]
                self.writer.add_scalar(
                    f"Diagnostics/{diagnosticName[0].lower()}{diagnosticName[1:]}",
                    value,
                    step,
                )
        self.writer.add_scalar("PPO/PolicyLoss", summary["policyLoss"], step)
        self.writer.add_scalar("PPO/ValueLoss", summary["valueLoss"], step)
        self.writer.add_scalar("PPO/Entropy", summary["entropy"], step)
        self.writer.add_scalar("PPO/ClipFrac", summary["clipFrac"], step)
        self.writer.add_scalar("PPO/ApproxKL", summary["approxKL"], step)
        self.writer.add_scalar("Action/RealStd", summary["realStd"], step)
        self.writer.add_scalar("Action/ImagStd", summary["imagStd"], step)
        self.writer.add_scalar("Corpus/Epoch", summary["corpusEpoch"], step)
        self.writer.add_scalar("Corpus/BlockIndex", summary["corpusBlockIndex"], step)
        self.writer.add_scalar("Corpus/BlockCount", summary["corpusBlockCount"], step)
        self.writer.add_scalar("Corpus/TargetsSeen", summary["corpusTargetsSeen"], step)
        self.writer.add_scalar("Corpus/BlockLoadSeconds", summary["corpusBlockLoadSeconds"], step)

    def _init_model(self, dtype: torch.dtype, device: torch.device) -> None:
        template = generateBatch(
            self.arraySpec,
            batchSize=1,
            device=torch.device("cpu"),
            dtype=dtype,
            weightsType="uniform",
            generator=self.rng,
        )
        self.templateBatch = template.to(device)
        actionDim = 2 * self.templateBatch.N
        self.model = build_model(self.modelConfig, actionDim).to(device=device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learningRate)

    def _sample_rollout_targets(
        self, device: torch.device, dtype: torch.dtype, showProgress: bool = False
    ) -> TargetBatch:
        return self.corpusLoader.next_target_batch(device, dtype, showProgress=showProgress)

    def _template_context_batch(self, batchSize: int) -> ArrayBatch:
        assert self.templateBatch is not None
        template = self.templateBatch
        elementMask = (
            None
            if template.elementMask is None
            else template.elementMask.expand(batchSize, -1).clone()
        )
        return ArrayBatch(
            elementLocalPosition=template.elementLocalPosition.expand(batchSize, -1, -1).clone(),
            weights=template.weights.expand(batchSize, -1).clone(),
            wavelength=template.wavelength,
            gain=template.gain.expand(batchSize).clone(),
            LLAPosition=template.LLAPosition.expand(batchSize, -1).clone(),
            ECEFPosition=template.ECEFPosition.expand(batchSize, -1).clone(),
            elementMask=elementMask,
        )

    def _power_to_linear(self, powerMap: torch.Tensor) -> torch.Tensor:
        return torch.where(powerMap < 0, torch.pow(10.0, powerMap / 10.0), powerMap.clamp_min(0.0))

    def _targets_to_inputs(self, targetBatch: TargetBatch) -> torch.Tensor:
        power = self._power_to_linear(targetBatch.powerMap)
        powerScale = power.flatten(1).amax(dim=1, keepdim=True).clamp_min(1e-10)
        normalizedPower = power / powerScale.view(-1, 1, 1)
        importance = targetBatch.importanceMap.clamp(0.0, 1.0)
        return torch.stack([normalizedPower, importance], dim=1)

    def _normalize_array_lla(self, lla: torch.Tensor) -> torch.Tensor:
        altitudeScale = max(
            1.0,
            abs(float(self.arraySpec.altitudeRange[0])),
            abs(float(self.arraySpec.altitudeRange[1])),
        )
        return torch.stack(
            [
                lla[:, 0] / 90.0,
                lla[:, 1] / 180.0,
                lla[:, 2] / altitudeScale,
            ],
            dim=-1,
        )

    def _normalize_array_ecef(self, ecef: torch.Tensor) -> torch.Tensor:
        altitudeExtent = max(
            abs(float(self.arraySpec.altitudeRange[0])),
            abs(float(self.arraySpec.altitudeRange[1])),
        )
        scale = max(semiMajorAxis + altitudeExtent, 1.0)
        return ecef / scale

    def _normalized_aperture_coordinates(self, localPosition: torch.Tensor) -> torch.Tensor:
        centered = localPosition - localPosition.mean(dim=-1, keepdim=True)
        scale = centered.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        return (centered / scale).transpose(1, 2)

    def _element_global_coordinates(self, arrayBatch: ArrayBatch) -> torch.Tensor:
        local = arrayBatch.elementLocalPosition.transpose(1, 2)
        rotation = getECEFtoENUMapping(arrayBatch.LLAPosition)
        east = rotation[:, 0, :]
        north = rotation[:, 1, :]
        up = rotation[:, 2, :]
        xLocal = local[..., 0].unsqueeze(-1)
        yLocal = local[..., 1].unsqueeze(-1)
        zLocal = local[..., 2].unsqueeze(-1)
        offsets = yLocal * east.unsqueeze(1) + zLocal * north.unsqueeze(1) - xLocal * up.unsqueeze(1)
        return arrayBatch.ECEFPosition.unsqueeze(1) + offsets

    def _geometry_one_hot(self, batchSize: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        geometry = str(self.arraySpec.geometry)
        values = {
            "URA": (1.0, 0.0),
            "UCA": (0.0, 1.0),
        }.get(geometry, (0.0, 0.0))
        return torch.tensor(values, device=device, dtype=dtype).view(1, -1).expand(batchSize, -1)

    def _build_model_context(
        self,
        batchSize: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> ModelContext:
        arrayBatch = self._template_context_batch(batchSize)
        globalParts: list[torch.Tensor] = []
        for feature in self.modelConfig.context.globalFeatures:
            if feature == "array_lla":
                globalParts.append(self._normalize_array_lla(arrayBatch.LLAPosition))
            elif feature == "array_ecef":
                globalParts.append(self._normalize_array_ecef(arrayBatch.ECEFPosition))
            elif feature == "gain":
                globalParts.append(arrayBatch.gain.unsqueeze(-1))
            elif feature == "wavelength":
                globalParts.append(
                    torch.full((batchSize, 1), arrayBatch.wavelength, device=device, dtype=dtype)
                )
            elif feature == "element_spacing":
                globalParts.append(
                    torch.full(
                        (batchSize, 1),
                        float(self.arraySpec.elementSpacing),
                        device=device,
                        dtype=dtype,
                    )
                )
            elif feature == "element_count":
                globalParts.append(
                    torch.full((batchSize, 1), float(arrayBatch.N), device=device, dtype=dtype)
                )
            elif feature == "geometry_one_hot":
                globalParts.append(self._geometry_one_hot(batchSize, device, dtype))

        elementParts: list[torch.Tensor] = []
        if self.modelConfig.context.elementFeatures:
            localPosition = arrayBatch.elementLocalPosition.transpose(1, 2)
            for feature in self.modelConfig.context.elementFeatures:
                if feature == "element_local_xyz":
                    elementParts.append(localPosition / max(float(self.arraySpec.elementSpacing), 1e-6))
                elif feature == "element_global_xyz":
                    elementParts.append(self._normalize_array_ecef(self._element_global_coordinates(arrayBatch)))
                elif feature == "normalized_aperture_xyz":
                    elementParts.append(self._normalized_aperture_coordinates(arrayBatch.elementLocalPosition))
                elif feature == "element_mask":
                    mask = (
                        torch.ones((batchSize, arrayBatch.N), device=device, dtype=dtype)
                        if arrayBatch.elementMask is None
                        else arrayBatch.elementMask.to(device=device, dtype=dtype)
                    )
                    elementParts.append(mask.unsqueeze(-1))

        return ModelContext(
            globalFeatures=None if not globalParts else torch.cat(globalParts, dim=-1),
            elementFeatures=None if not elementParts else torch.cat(elementParts, dim=-1),
            elementMask=arrayBatch.elementMask,
            arrayBatch=arrayBatch,
            arraySpec=self.arraySpec.serializeArraySpec(),
        )

    def _build_model_input(self, targetBatch: TargetBatch) -> ModelInput:
        inputs = self._targets_to_inputs(targetBatch)
        context = self._build_model_context(
            batchSize=targetBatch.batchSize,
            device=targetBatch.searchLatitudes.device,
            dtype=targetBatch.searchLatitudes.dtype,
        )
        return ModelInput(targetTensor=inputs, context=context)

    def _sample_actions(self, mean: torch.Tensor, logStd: torch.Tensor) -> torch.Tensor:
        std = self._bounded_log_std(logStd).exp()
        noise = torch.randn(mean.shape, generator=self.rng, dtype=mean.dtype, device="cpu").to(
            mean.device
        )
        return mean + std * noise

    def _log_prob(self, mean: torch.Tensor, logStd: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        boundedLogStd = self._bounded_log_std(logStd)
        std = boundedLogStd.exp().clamp_min(1e-8)
        variance = std.square()
        logProb = -0.5 * (
            ((actions - mean) ** 2) / variance
            + 2 * boundedLogStd
            + torch.log(torch.tensor(2 * torch.pi, device=mean.device, dtype=mean.dtype))
        )
        return logProb.sum(dim=-1)

    def _entropy(self, logStd: torch.Tensor) -> torch.Tensor:
        boundedLogStd = self._bounded_log_std(logStd)
        entropy = boundedLogStd + 0.5 * (
            1.0
            + torch.log(torch.tensor(2 * torch.pi, device=logStd.device, dtype=logStd.dtype))
        )
        return entropy.sum(dim=-1)

    def _bounded_log_std(self, logStd: torch.Tensor) -> torch.Tensor:
        return logStd.clamp(min=_LOG_STD_MIN, max=_LOG_STD_MAX)

    def _clamp_model_parameters(self) -> None:
        if self.model is None:
            return
        logStd = getattr(self.model, "logStd", None)
        if isinstance(logStd, torch.nn.Parameter):
            with torch.no_grad():
                logStd.clamp_(min=_LOG_STD_MIN, max=_LOG_STD_MAX)

    def _ensure_finite(
        self,
        name: str,
        tensor: torch.Tensor,
        step: int,
        corpusStatus: dict[str, float | int | bool] | None = None,
    ) -> None:
        if torch.isfinite(tensor).all():
            return

        details = [f"step={step}"]
        if corpusStatus is not None:
            details.append(f"epoch={int(corpusStatus['epoch'])}")
            details.append(
                f"block={int(corpusStatus['blockIndex'])}/{int(corpusStatus['blockCount'])}"
            )
            details.append(f"targetsSeen={int(corpusStatus['targetsSeen'])}")
        raise FloatingPointError(
            f"non-finite tensor detected for {name} ({', '.join(details)})"
        )

    def _weights_from_actions(self, actions: torch.Tensor) -> torch.Tensor:
        assert self.templateBatch is not None
        batchSize = actions.shape[0]
        elementCount = self.templateBatch.N
        real, imag = actions[:, :elementCount], actions[:, elementCount:]
        complexWeights = torch.complex(real, imag)
        amplitude = complexWeights.abs()
        phase = complexWeights.angle()
        elementMask = (
            None
            if self.templateBatch.elementMask is None
            else self.templateBatch.elementMask.expand(batchSize, -1)
        )
        normalizedAmplitude = _normalize_amplitudes(amplitude, elementMask)
        return torch.polar(normalizedAmplitude, phase)

    def _batch_from_weights(self, weights: torch.Tensor) -> ArrayBatch:
        assert self.templateBatch is not None
        batchSize = weights.shape[0]
        template = self.templateBatch
        elementMask = (
            None
            if template.elementMask is None
            else template.elementMask.expand(batchSize, -1).clone()
        )
        return ArrayBatch(
            elementLocalPosition=template.elementLocalPosition.expand(batchSize, -1, -1).clone(),
            weights=weights,
            wavelength=template.wavelength,
            gain=template.gain.expand(batchSize).clone(),
            LLAPosition=template.LLAPosition.expand(batchSize, -1).clone(),
            ECEFPosition=template.ECEFPosition.expand(batchSize, -1).clone(),
            elementMask=elementMask,
        )

    def _collect_step_batch(self, dtype: torch.dtype, device: torch.device) -> PPOStepBatch:
        assert self.model is not None
        targetBatch = self._sample_rollout_targets(device=device, dtype=dtype)
        modelInputs = self._build_model_input(targetBatch)

        with torch.no_grad():
            outputs = self.model(modelInputs)
            self._ensure_finite("policyMean", outputs.policyMean, step=-1)
            self._ensure_finite("logStd", self._bounded_log_std(outputs.logStd), step=-1)
            self._ensure_finite("value", outputs.value, step=-1)
            actions = self._sample_actions(outputs.policyMean, outputs.logStd)
            oldLogProb = self._log_prob(outputs.policyMean, outputs.logStd, actions)
            self._ensure_finite("actions", actions, step=-1)
            self._ensure_finite("oldLogProb", oldLogProb, step=-1)
            weights = self._weights_from_actions(actions)
            arrayBatch = self._batch_from_weights(weights)
            if self.objectiveVersion == "v1":
                assert isinstance(self.lossParams, LossConfig)
                evaluation = evaluateBatch(
                    batch=arrayBatch,
                    target=targetBatch,
                    params=self.lossParams,
                    targetMode="per_sample",
                )
            else:
                assert isinstance(self.lossParams, LossConfigV2)
                evaluation = evaluateBatchV2(
                    batch=arrayBatch,
                    target=targetBatch,
                    params=self.lossParams,
                    targetMode="per_sample",
                )
            reward = -evaluation.totalLoss.detach()
            self._ensure_finite("evaluation.totalLoss", evaluation.totalLoss, step=-1)
            self._ensure_finite("reward", reward, step=-1)

        return PPOStepBatch(
            modelInputs=modelInputs,
            targetBatch=targetBatch,
            actions=actions.detach(),
            oldLogProb=oldLogProb.detach(),
            oldValue=outputs.value.detach(),
            reward=reward,
            evaluation=evaluation,
            arrayBatch=arrayBatch,
        )

    def _queueCheckpoint(
        self,
        step: int,
        bestRewardOverall: float | None,
        bestStepOverall: int | None,
        bestSampleOverall: dict[str, Any] | None,
        historyBest: list[float],
        historyMean: list[float],
    ) -> None:
        if self.checkpointConfig.checkpointMode == "off":
            return
        if (step + 1) % self.checkpointConfig.checkpointEverySteps != 0:
            return
        assert self.model is not None
        assert self.optimizer is not None
        assert self.checkpointWriter is not None
        payload = {
            "step": step,
            "modelState": self.model.state_dict(),
            "optimizerState": self.optimizer.state_dict(),
            "rngState": self.rng.get_state(),
            "corpusLoaderState": self.corpusLoader.serializeState(),
            "bestRewardOverall": bestRewardOverall,
            "bestStepOverall": bestStepOverall,
            "bestSampleOverall": bestSampleOverall,
            "history": {
                "bestReward": list(historyBest),
                "meanReward": list(historyMean),
            },
        }
        self.checkpointWriter.submit(self.checkpointPath, payload)

    def _queueBestSample(self, samplePayload: dict[str, Any]) -> None:
        assert self.checkpointWriter is not None
        self.checkpointWriter.submit(self.bestPath, samplePayload)

    def loadResumeState(self, device: torch.device) -> dict[str, Any] | None:
        if not self.checkpointPath.exists():
            return None
        payload = torch.load(self.checkpointPath, weights_only=False, map_location=device)
        return {
            "step": int(payload["step"]) + 1,
            "modelState": payload["modelState"],
            "optimizerState": payload["optimizerState"],
            "rngState": payload.get("rngState"),
            "corpusLoaderState": payload.get("corpusLoaderState"),
            "bestRewardOverall": payload.get("bestRewardOverall"),
            "bestStepOverall": payload.get("bestStepOverall"),
            "bestSampleOverall": payload.get("bestSampleOverall"),
            "historyBest": list(payload.get("history", {}).get("bestReward", [])),
            "historyMean": list(payload.get("history", {}).get("meanReward", [])),
        }

    def train(
        self,
        dtype: torch.dtype,
        device: torch.device,
        logDir: str | Path | None = None,
        resume: bool | None = None,
    ) -> dict[str, Any]:
        activeLogDir = self.writerLogDir if logDir is None else logDir
        activeResume = True if resume is None else resume

        if not activeResume:
            self.resetExperiment(activeLogDir)

        self.writerLogDir = activeLogDir
        if self.writer is not None:
            self.writer.close()
            self.writer = None

        runPath = self.runLocation(activeLogDir)
        if runPath is not None:
            self._writeRunArtifacts(runPath)

        self._init_model(dtype=dtype, device=device)
        assert self.model is not None
        assert self.optimizer is not None
        self._startWriters()
        pbar = None

        try:
            resumeState = self.loadResumeState(device=device) if activeResume else None
            if resumeState is None:
                startStep = 0
                bestRewardOverall: float | None = None
                bestStepOverall: int | None = None
                bestSampleOverall: dict[str, Any] | None = None
                historyBest: list[float] = []
                historyMean: list[float] = []
                self.corpusLoader._ensure_initialized(showProgress=True)
                loaderStatus = self.corpusLoader.status()
                print(
                    "Initialized PPO corpus block "
                    f"epoch={int(loaderStatus['epoch'])} "
                    f"block={int(loaderStatus['blockIndex'])}/{int(loaderStatus['blockCount'])} "
                    f"size={int(loaderStatus['currentBlockSize'])}",
                    flush=True,
                )
            else:
                startStep = int(resumeState["step"])
                self.model.load_state_dict(resumeState["modelState"])
                self.optimizer.load_state_dict(resumeState["optimizerState"])
                if resumeState.get("rngState") is not None:
                    self.rng.set_state(resumeState["rngState"])
                if resumeState.get("corpusLoaderState") is not None:
                    self.corpusLoader.restoreState(resumeState["corpusLoaderState"])
                else:
                    self.corpusLoader._ensure_initialized(showProgress=True)
                bestRewardOverall = resumeState["bestRewardOverall"]
                bestStepOverall = resumeState["bestStepOverall"]
                bestSampleOverall = resumeState["bestSampleOverall"]
                historyBest = resumeState["historyBest"]
                historyMean = resumeState["historyMean"]

            if startStep >= self.config.updateSteps:
                return {
                    "experimentName": self.experimentName,
                    "bestRewardOverall": bestRewardOverall,
                    "bestStepOverall": bestStepOverall,
                    "bestSampleOverall": bestSampleOverall,
                    "history": {
                        "bestReward": torch.tensor(historyBest),
                        "meanReward": torch.tensor(historyMean),
                    },
                }

            if (
                activeLogDir is not None
                and self.loggingConfig.logMode != "off"
                and runPath is not None
            ):
                self.writer = SummaryWriter(log_dir=str(runPath), purge_step=startStep)

            pbar = tqdm(
                total=self.config.updateSteps,
                initial=startStep,
                desc=f"PPO {self.experimentName}",
                dynamic_ncols=True,
            )
            pbar.set_postfix_str("collecting")
            pbar.refresh()

            for step in range(startStep, self.config.updateSteps):
                self.corpusLoader.maybe_promote_prefetch()
                corpusStatus = self.corpusLoader.status()
                rollout = self._collect_step_batch(dtype=dtype, device=device)
                returns = rollout.reward
                advantages = returns - rollout.oldValue
                self._ensure_finite("returns", returns, step=step, corpusStatus=corpusStatus)
                self._ensure_finite("advantages_pre_norm", advantages, step=step, corpusStatus=corpusStatus)
                if advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / advantages.std(
                        unbiased=False
                    ).clamp_min(1e-8)
                self._ensure_finite("advantages", advantages, step=step, corpusStatus=corpusStatus)

                policyLossTotal = 0.0
                valueLossTotal = 0.0
                entropyTotal = 0.0
                clipFracTotal = 0.0
                approxKLTotal = 0.0
                optimizationSteps = 0

                for _ in range(self.config.ppoEpochs):
                    permutation = torch.randperm(
                        rollout.modelInputs.targetTensor.shape[0], generator=self.rng, device="cpu"
                    )
                    for start in range(0, rollout.modelInputs.targetTensor.shape[0], self.config.minibatchSize):
                        ids = permutation[start : start + self.config.minibatchSize].to(
                            device=rollout.modelInputs.targetTensor.device
                        )
                        minibatchInputs = rollout.modelInputs.index(ids)
                        minibatchActions = rollout.actions[ids]
                        minibatchOldLogProb = rollout.oldLogProb[ids]
                        minibatchAdvantages = advantages[ids]
                        minibatchReturns = returns[ids]

                        outputs = self.model(minibatchInputs)
                        self._ensure_finite(
                            "policyMean",
                            outputs.policyMean,
                            step=step,
                            corpusStatus=corpusStatus,
                        )
                        self._ensure_finite(
                            "logStd",
                            self._bounded_log_std(outputs.logStd),
                            step=step,
                            corpusStatus=corpusStatus,
                        )
                        self._ensure_finite(
                            "value",
                            outputs.value,
                            step=step,
                            corpusStatus=corpusStatus,
                        )
                        newLogProb = self._log_prob(
                            outputs.policyMean,
                            outputs.logStd,
                            minibatchActions,
                        )
                        entropy = self._entropy(outputs.logStd).mean()
                        self._ensure_finite(
                            "newLogProb",
                            newLogProb,
                            step=step,
                            corpusStatus=corpusStatus,
                        )
                        self._ensure_finite(
                            "entropy",
                            entropy.unsqueeze(0),
                            step=step,
                            corpusStatus=corpusStatus,
                        )
                        logRatio = (newLogProb - minibatchOldLogProb).clamp(
                            min=_LOG_RATIO_MIN,
                            max=_LOG_RATIO_MAX,
                        )
                        ratio = torch.exp(logRatio)
                        unclipped = ratio * minibatchAdvantages
                        clipped = torch.clamp(
                            ratio,
                            1.0 - self.config.clipEpsilon,
                            1.0 + self.config.clipEpsilon,
                        ) * minibatchAdvantages
                        policyLoss = -torch.min(unclipped, clipped).mean()
                        valueLoss = F.mse_loss(outputs.value, minibatchReturns)
                        loss = (
                            policyLoss
                            + self.config.valueLossCoef * valueLoss
                            - self.config.entropyCoef * entropy
                        )
                        self._ensure_finite("ratio", ratio, step=step, corpusStatus=corpusStatus)
                        self._ensure_finite("policyLoss", policyLoss.unsqueeze(0), step=step, corpusStatus=corpusStatus)
                        self._ensure_finite("valueLoss", valueLoss.unsqueeze(0), step=step, corpusStatus=corpusStatus)
                        self._ensure_finite("loss", loss.unsqueeze(0), step=step, corpusStatus=corpusStatus)

                        self.optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.maxGradNorm
                        )
                        self.optimizer.step()
                        self._clamp_model_parameters()

                        clipFrac = (
                            (torch.abs(ratio - 1.0) > self.config.clipEpsilon)
                            .to(dtype=outputs.value.dtype)
                            .mean()
                        )
                        approxKL = (minibatchOldLogProb - newLogProb).mean()
                        policyLossTotal += float(policyLoss.item())
                        valueLossTotal += float(valueLoss.item())
                        entropyTotal += float(entropy.item())
                        clipFracTotal += float(clipFrac.item())
                        approxKLTotal += float(approxKL.item())
                        optimizationSteps += 1

                meanReward = float(rollout.reward.mean().item())
                bestIdx = int(torch.argmax(rollout.reward).item())
                bestReward = float(rollout.reward[bestIdx].item())
                historyBest.append(bestReward)
                historyMean.append(meanReward)

                if bestRewardOverall is None or bestReward > bestRewardOverall:
                    bestRewardOverall = bestReward
                    bestStepOverall = step
                    bestSampleOverall = {
                        **rollout.arrayBatch.serializeBatchSample(bestIdx),
                        "target": fetchTargetSample(rollout.targetBatch, bestIdx).serializeTargetSpec(),
                        "reward": bestReward,
                        "loss": evaluation_loss_record(rollout.evaluation, bestIdx),
                        "step": step,
                    }
                    self._queueBestSample(bestSampleOverall)

                logStd = self.model(rollout.modelInputs.index(slice(0, 1))).logStd.detach()
                actionDim = logStd.shape[-1] // 2
                corpusStatus = self.corpusLoader.status()
                summary = {
                    "meanReward": meanReward,
                    "bestReward": bestReward,
                    "meanTotalLoss": float(rollout.evaluation.totalLoss.mean().item()),
                    "policyLoss": policyLossTotal / max(1, optimizationSteps),
                    "valueLoss": valueLossTotal / max(1, optimizationSteps),
                    "entropy": entropyTotal / max(1, optimizationSteps),
                    "clipFrac": clipFracTotal / max(1, optimizationSteps),
                    "approxKL": approxKLTotal / max(1, optimizationSteps),
                    "realStd": float(logStd[..., :actionDim].exp().mean().item()),
                    "imagStd": float(logStd[..., actionDim:].exp().mean().item()),
                    "corpusEpoch": float(corpusStatus["epoch"]),
                    "corpusBlockIndex": float(corpusStatus["blockIndex"]),
                    "corpusBlockCount": float(corpusStatus["blockCount"]),
                    "corpusTargetsSeen": float(corpusStatus["targetsSeen"]),
                    "corpusBlockLoadSeconds": float(corpusStatus["blockLoadSeconds"]),
                }
                for name, value in evaluation_weighted_loss_means(
                    rollout.evaluation, self.lossParams
                ).items():
                    summary[f"mean{name[0].upper()}{name[1:]}Loss"] = value
                for name, value in evaluation_diagnostic_means(rollout.evaluation).items():
                    summary[f"mean{name[0].upper()}{name[1:]}Metric"] = value
                self._logMetrics(step, summary)
                prefetchLabel = (
                    "ready"
                    if corpusStatus["prefetchReady"]
                    else "loading"
                    if corpusStatus["prefetchLoading"]
                    else "off"
                )
                pbar.set_postfix(
                    {
                        "best": f"{bestReward:.4f}",
                        "mean": f"{meanReward:.4f}",
                        "epoch": int(corpusStatus["epoch"]),
                        "block": f"{int(corpusStatus['blockIndex'])}/{int(corpusStatus['blockCount'])}",
                        "prefetch": prefetchLabel,
                    }
                )

                self._queueCheckpoint(
                    step=step,
                    bestRewardOverall=bestRewardOverall,
                    bestStepOverall=bestStepOverall,
                    bestSampleOverall=bestSampleOverall,
                    historyBest=historyBest,
                    historyMean=historyMean,
                )
                pbar.update(1)

            finalPayload = {
                "experimentName": self.experimentName,
                "bestRewardOverall": bestRewardOverall,
                "bestStepOverall": bestStepOverall,
                "bestSampleOverall": bestSampleOverall,
                "history": {
                    "bestReward": torch.tensor(historyBest),
                    "meanReward": torch.tensor(historyMean),
                },
                "model": modelConfigToDict(self.modelConfig),
            }
            assert self.checkpointWriter is not None
            self.checkpointWriter.submit(self.archiveLocation / "final.pt", finalPayload)
            self.checkpointWriter.flush()
            return finalPayload
        finally:
            if pbar is not None:
                pbar.close()
            if self.writer is not None:
                self.writer.close()
                self.writer = None
            self._closeWriters()
            self.corpusLoader.close()


def buildPPOControllerFromConfig(
    configPath: str,
) -> tuple[PPOController, tuple[torch.device, torch.dtype, ExperimentConfig]]:
    runConfig = loadPPORunConfig(configPath)
    manifestPath, recordPaths = resolvePPOTargetRecordPaths(runConfig)
    selectionCount = "all" if runConfig.target.selectionCount is None else str(
        runConfig.target.selectionCount
    )
    loaderWorkers = (
        runConfig.target.loaderWorkers
        if runConfig.target.loaderWorkers != "auto"
        else min(8, max(1, (os.cpu_count() or 1) // 2))
    )
    print(
        "Loading PPO target corpus "
        f"from {manifestPath} "
        f"(effectiveCorpusSize={len(recordPaths)}, "
        f"selection={runConfig.target.selection}, "
        f"selectionCount={selectionCount}, "
        f"loadingMode={runConfig.target.loadingMode}, "
        f"blockSize={runConfig.target.blockSize}, "
        f"prefetchNextBlock={runConfig.target.prefetchNextBlock}, "
        f"decimate={runConfig.target.decimate}, "
        f"loaderWorkers={loaderWorkers})",
        flush=True,
    )
    device, dtype = torch.device(runConfig.device.device), {
        "float32": torch.float32,
        "float64": torch.float64,
    }[runConfig.device.dtype]

    class _ResolvedPPOController(PPOController):
        def _resolvedRunConfig(self) -> PPORunConfig:
            return runConfig

    controller = _ResolvedPPOController(
        config=runConfig.ppo,
        modelConfig=runConfig.model,
        corpusLoader=PPOCorpusLoader(
            recordPaths=recordPaths,
            targetConfig=runConfig.target,
            rolloutBatchSize=runConfig.ppo.rolloutBatchSize,
            seed=runConfig.target.selectionSeed
            if runConfig.target.selectionSeed is not None
            else runConfig.ppo.seed,
        ),
        arraySpec=runConfig.array,
        lossParams=runConfig.loss if runConfig.objectiveVersion == "v1" else runConfig.lossV2,
        objectiveVersion=runConfig.objectiveVersion,
        experimentName=runConfig.experiment.name,
        archiveRoot=runConfig.experiment.archiveDir,
        loggingConfig=runConfig.logging,
        checkpointConfig=runConfig.checkpoint,
        workerConfig=runConfig.workers,
        sourceConfigPath=runConfig.sourcePath,
        modelSourcePath=runConfig.modelSourcePath,
        resolvedConfig=ppoRunConfigToDict(runConfig),
        writerLogDir=runConfig.experiment.logDir,
    )
    return controller, (device, dtype, runConfig.experiment)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Helios PPO from YAML config")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    controller, (device, dtype, experiment) = buildPPOControllerFromConfig(args.config)
    controller.train(
        dtype=dtype,
        device=device,
        logDir=experiment.logDir,
        resume=experiment.resume,
    )


if __name__ == "__main__":
    main()
