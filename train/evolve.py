from dataclasses import dataclass
from pathlib import Path
import shutil

import torch
from scripts.arrayBatch import ArrayBatch, merge
from scripts.arraySpec import ArraySpec
from scripts.batchFactory import generateBatch
from scripts.plots import projectResponseOnTarget
from scripts.targetSpec import TargetSpec
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from train.objective import LossConfig, LossType, batchLoss


@dataclass
class EvolutionConfig:
    generator: torch.Generator | None = None
    batchSize: int = 100
    evolutionSteps: int = 100

    # must sum to 1
    cloneFraction: float = 0.2
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

    @property
    def cloneCount(self) -> int:
        return int(self.cloneFraction * self.batchSize)

    @property
    def mutateCount(self) -> int:
        return int(self.mutateFraction * self.batchSize)

    @property
    def randomCount(self) -> int:
        return self.batchSize - self.cloneCount - self.mutateCount

    @property
    def parentPoolCount(self) -> int:
        return int(self.parentPoolFraction * self.batchSize)

    def sigmaAt(self, step: int) -> tuple[float, float]:
        phaseSigma = max(self.phaseMinSigma, self.phaseSigma * (self.phaseSigmaDecay**step))
        amplitudeSigma = max(
            self.amplitudeMinSigma, self.amplitudeSigma * (self.amplitudeSigmaDecay**step)
        )
        return phaseSigma, amplitudeSigma

    def serializeEvolutionConfig(self) -> dict:
        return {
            "batchSize": self.batchSize,
            "cloneFraction": self.cloneFraction,
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
        }


@dataclass
class EvolutionController:
    config: EvolutionConfig
    targetSpec: TargetSpec
    arraySpec: ArraySpec
    lossParams: LossConfig
    experimentName: str = "evo_1"
    writer: SummaryWriter | None = None

    @property
    def archiveLocation(self) -> Path:
        return Path("data/archive") / self.experimentName

    def runLocation(self, logDir: str | Path | None) -> Path | None:
        if logDir is None:
            return None
        return Path(logDir) / self.experimentName

    def resetExperiment(self, logDir: str | Path | None) -> None:
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

    def loadResumeState(self, device: torch.device) -> dict | None:
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
        latestStep = int(latestPayload["step"])
        if latestStep < self.config.evolutionSteps - 1:
            scores = latestPayload["scores"].to(device)
            batch = self.evolutionStep(latestStep, batch, scores)

        return {
            "batch": batch,
            "startStep": latestStep + 1,
            "bestScoreOverall": bestScoreOverall,
            "bestStepOverall": bestStepOverall,
            "bestSampleOverall": bestSampleOverall,
            "historyBest": historyBest,
            "historyMean": historyMean,
            "historyWorst": historyWorst,
        }

    def logInit(self) -> Path:
        archiveDir = self.archiveLocation
        archiveDir.mkdir(parents=True, exist_ok=True)

        payload = {
            "experimentName": self.experimentName,
            "config": self.config.serializeEvolutionConfig(),
            "targetSpec": self.targetSpec.serializeTargetSpec(),
            "arraySpec": self.arraySpec.serializeArraySpec(),
        }

        savePath = archiveDir / "init.pt"
        torch.save(payload, savePath)
        return savePath

    def logStep(self, step: int, batch: ArrayBatch, scores: torch.Tensor) -> Path:
        archiveDir = self.archiveLocation
        archiveDir.mkdir(parents=True, exist_ok=True)

        if scores.ndim != 1:
            raise ValueError(f"scores must have shape [B], got {tuple(scores.shape)}")
        if scores.shape[0] != batch.batchSize:
            raise ValueError(
                f"scores length ({scores.shape[0]}) must match batch size ({batch.batchSize})"
            )

        bestIdx = int(torch.argmin(scores).item())
        worstIdx = int(torch.argmax(scores).item())

        payload = {
            "step": step,
            "summary": {
                "batchSize": batch.batchSize,
                "elementCount": batch.N,
                "bestScore": float(scores[bestIdx].item()),
                "meanScore": float(scores.mean().item()),
                "medianScore": float(scores.median().item()),
                "worstScore": float(scores[worstIdx].item()),
                "stdScore": float(scores.std(unbiased=False).item()),
                "bestIdx": bestIdx,
                "worstIdx": worstIdx,
            },
            "scores": scores.detach().cpu(),
            "batch": batch.serializeBatch(),
            "bestSample": {
                **batch.serializeBatchSample(bestIdx),
                "score": float(scores[bestIdx].item()),
            },
        }

        savePath = archiveDir / f"step_{step:05d}.pt"
        torch.save(payload, savePath)

        if self.writer:
            self.writer.add_scalar("Score/Best", payload["summary"]["bestScore"], step)
            self.writer.add_scalar("Score/Mean", payload["summary"]["meanScore"], step)
            self.writer.add_scalar("Score/Worst", payload["summary"]["worstScore"], step)
            self.writer.add_scalar("Score/Std", payload["summary"]["stdScore"], step)

            phaseSigma, amplitudeSigma = self.config.sigmaAt(step)
            self.writer.add_scalar("Param/PhaseSigma", phaseSigma, step)
            self.writer.add_scalar("Param/AmplitudeSigma", amplitudeSigma, step)

            randomFraction = max(
                self.config.minRandomFraction,
                self.config.randomFraction * (self.config.randomFractionDecay**step),
            )
            self.writer.add_scalar("Param/RandomFraction", randomFraction, step)

        return savePath

    def getScores(self, batch: ArrayBatch) -> torch.Tensor:
        return batchLoss(batch, self.targetSpec, self.lossParams, self.config.lossType)

    def sample(
        self,
        parentIDs: torch.Tensor,
        childCount: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        samples = torch.randint(
            0, parentIDs.shape[0], (childCount,), device=device, generator=generator
        )
        return parentIDs[samples]

    def initEvolution(self, dtype: torch.dtype, device: torch.device) -> ArrayBatch:
        self.logInit()
        return generateBatch(
            self.arraySpec,
            batchSize=self.config.batchSize,
            device=device,
            dtype=dtype,
            weightsType="random",
            targetLLA=self.targetSpec.hotspotCoordinates[0],
        )

    def evolutionStep(self, step: int, batch: ArrayBatch, scores: torch.Tensor) -> ArrayBatch:

        device, dtype = batch.device, batch.dtype
        sortedScoresIDs = torch.argsort(scores, dim=0, descending=False)
        cloneIDs = sortedScoresIDs[: self.config.cloneCount]
        parentIDs = sortedScoresIDs[: self.config.parentPoolCount]
        clones = batch.fetch(cloneIDs)

        mutationIDs = self.sample(parentIDs, self.config.mutateCount, device)
        mutationParents = batch.fetch(mutationIDs)
        phaseSigma, amplitudeSigma = self.config.sigmaAt(step)
        mutationChildren = mutationParents.mutateWeights(phaseSigma, amplitudeSigma)

        randomChildren = generateBatch(self.arraySpec, self.config.randomCount, device, dtype)

        nextBatch = merge([clones, mutationChildren, randomChildren])
        return nextBatch

    def train(
        self,
        dtype: torch.dtype,
        device: torch.device,
        logDir: str | Path | None = "runs",
        plotProjection: bool = False,
        resume: bool = True,
    ) -> dict:
        if not resume:
            self.resetExperiment(logDir)

        if self.writer is not None:
            self.writer.close()
            self.writer = None

        resumeState = self.loadResumeState(device=device) if resume else None
        if resumeState is None:
            batch = self.initEvolution(dtype=dtype, device=device)
            startStep = 0
            bestScoreOverall: float | None = None
            bestStepOverall: int | None = None
            bestSampleOverall: dict | None = None
            historyBest: list[float] = []
            historyMean: list[float] = []
            historyWorst: list[float] = []
        else:
            batch = resumeState["batch"]
            startStep = int(resumeState["startStep"])
            bestScoreOverall = resumeState["bestScoreOverall"]
            bestStepOverall = resumeState["bestStepOverall"]
            bestSampleOverall = resumeState["bestSampleOverall"]
            historyBest = resumeState["historyBest"]
            historyMean = resumeState["historyMean"]
            historyWorst = resumeState["historyWorst"]

        if logDir is not None:
            logPath = self.runLocation(logDir)
            assert logPath is not None
            self.writer = SummaryWriter(log_dir=str(logPath), purge_step=startStep)

        if startStep >= self.config.evolutionSteps:
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
            savePath = self.archiveLocation / "final.pt"
            torch.save(finalPayload, savePath)
            if self.writer:
                self.writer.close()
                self.writer = None
            return finalPayload

        pbar = tqdm(
            range(startStep, self.config.evolutionSteps), desc=f"Evolving {self.experimentName}"
        )
        for step in pbar:
            scores = self.getScores(batch)

            bestIdx = int(torch.argmin(scores).item())
            bestScore = float(scores[bestIdx].item())
            meanScore = float(scores.mean().item())
            worstScore = float(scores.max().item())

            bestBatchPath = self.archiveLocation / "best.pt"
            bestBatchPath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(batch.fetch(bestIdx).to(torch.device("cpu")), bestBatchPath)

            historyBest.append(bestScore)
            historyMean.append(meanScore)
            historyWorst.append(worstScore)

            pbar.set_postfix({"best": f"{bestScore:.4f}", "mean": f"{meanScore:.4f}"})

            if plotProjection:
                projectResponseOnTarget(
                    batch, self.targetSpec, sampleID=bestIdx, normalizedInputs=True
                )

            if bestScoreOverall is None or bestScore < bestScoreOverall:
                bestScoreOverall = bestScore
                bestStepOverall = step
                bestSampleOverall = {
                    **batch.serializeBatchSample(bestIdx),
                    "score": bestScore,
                    "step": step,
                }

            self.logStep(step, batch, scores)

            if step < self.config.evolutionSteps - 1:
                batch = self.evolutionStep(step, batch, scores)

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

        savePath = self.archiveLocation / "final.pt"
        torch.save(finalPayload, savePath)

        if self.writer:
            self.writer.close()
            self.writer = None

        return finalPayload
