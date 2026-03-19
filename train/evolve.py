from dataclasses import dataclass
from pathlib import Path

import torch
from scripts.arrayBatch import ArrayBatch, merge
from scripts.arraySpec import ArraySpec
from scripts.batchFactory import generateBatch
from scripts.targetSpec import TargetSpec

from train.reward import LossType, batchScore


@dataclass
class EvolutionConfig:
    generator: torch.Generator | None = None
    batchSize: int = 100
    evolutionSteps: int = 100

    # must sum to 1
    cloneFraction: float = 0.2
    mutateFraction: float = 0.6
    randomFraction: float = 0.2

    parentPoolFraction: float = 0.5

    phaseSigma: float = 0.15
    amplitudeSigma: float = 0.05
    gainSigma: float = 0

    phaseSigmaDecay: float = 0.98
    amplitudeSigmaDecay: float = 0.98
    randomFractionDecay: float = 0.995

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

    experimentName: str = "evo_1"

    @property
    def archiveLocation(self) -> Path:
        return Path("data/archive") / self.experimentName

    def logInit(self) -> Path:
        archiveDir = self.archiveLocation
        archiveDir.mkdir(parents=True, exist_ok=True)

        payload = {
            "experimentName": self.experimentName,
            "config": self.config.serializeEvolutionConfig(),
            "targetSpec": self.targetSpec.serializeTargetSpec(),
            "arraySpec": self.arraySpec.serializeArraySpec(),
        }

        savePath = archiveDir / "run.pt"
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
        return savePath

    def getScores(self, batch: ArrayBatch) -> torch.Tensor:
        return batchScore(batch, self.targetSpec, self.config.lossType)

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
            self.arraySpec, batchSize=self.config.batchSize, dtype=dtype, device=device
        )

    def evolutionStep(self, step: int, batch: ArrayBatch, scores: torch.Tensor) -> ArrayBatch:

        device, dtype = batch.device, batch.dtype
        sortedScoresIDs = torch.argsort(scores, dim=0, descending=False)
        cloneIDs = sortedScoresIDs[: self.config.cloneCount]
        parentIDs = sortedScoresIDs[: self.config.parentPoolCount]
        clones = batch.fetch(cloneIDs)

        mutationIDs = self.sample(parentIDs, self.config.mutateCount, device)
        mutationParents = batch.fetch(mutationIDs)
        mutationChildren = mutationParents.mutateWeights(self.config, step)

        randomChildren = generateBatch(self.arraySpec, self.config.randomCount, device, dtype)

        nextBatch = merge([clones, mutationChildren, randomChildren])
        return nextBatch

    def train(self, dtype: torch.dtype, device: torch.device) -> dict:
        batch = self.initEvolution(dtype=dtype, device=device)

        bestScoreOverall: float | None = None
        bestStepOverall: int | None = None
        bestSampleOverall: dict | None = None

        historyBest: list[float] = []
        historyMean: list[float] = []
        historyWorst: list[float] = []

        for step in range(self.config.evolutionSteps):
            scores = self.getScores(batch)

            bestIdx = int(torch.argmin(scores).item())
            bestScore = float(scores[bestIdx].item())
            meanScore = float(scores.mean().item())
            worstScore = float(scores.max().item())

            historyBest.append(bestScore)
            historyMean.append(meanScore)
            historyWorst.append(worstScore)

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

        return finalPayload
