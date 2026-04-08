from __future__ import annotations

import argparse
import statistics

import torch

from train.config import loadRunConfig, resolveDevice, resolveTarget, runConfigToDict
from train.evolve import EvolutionController
from train.objective import evaluateBatch


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _run_benchmark(
    controller: EvolutionController,
    batch,
    lossParams,
    runs: int,
    warmup: int,
    allowSharedTargetFastPath: bool,
) -> tuple[list[float], list[float], object]:
    totalTimings: list[float] = []
    gpuPeaks: list[float] = []
    lastEvaluation = None

    for iteration in range(warmup + runs):
        evaluation = evaluateBatch(
            batch=batch,
            target=controller.targetSpec,
            params=lossParams,
            targetMode=controller.targetMode,
            linearResponseChunkSize=controller.config.linearResponseChunkSize,
            wideResponseChunkSize=controller.config.wideResponseChunkSize,
            responseReductionTileCap=controller.config.responseReductionTileCap,
            allowSharedTargetFastPath=allowSharedTargetFastPath,
        )
        if batch.device.type == "cuda":
            torch.cuda.synchronize(batch.device)

        if iteration >= warmup:
            totalTimings.append(evaluation.diagnostics.totalMS)
            if evaluation.diagnostics.gpuMaxMemoryMB is not None:
                gpuPeaks.append(evaluation.diagnostics.gpuMaxMemoryMB)
        lastEvaluation = evaluation

    return totalTimings, gpuPeaks, lastEvaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Helios evaluation throughput")
    parser.add_argument("--config", default="configs/evo.yaml", help="Path to YAML config")
    parser.add_argument("--runs", type=int, default=3, help="Measured benchmark runs")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs")
    args = parser.parse_args()

    runConfig = loadRunConfig(args.config)
    target = resolveTarget(runConfig)
    device, dtype = resolveDevice(runConfig)
    controller = EvolutionController(
        config=runConfig.evolution,
        targetSpec=target,
        arraySpec=runConfig.array,
        lossParams=runConfig.loss,
        experimentName=runConfig.experiment.name,
        archiveRoot=runConfig.experiment.archiveDir,
        loggingConfig=runConfig.logging,
        checkpointConfig=runConfig.checkpoint,
        workerConfig=runConfig.workers,
        targetMode=runConfig.experiment.targetMode,
        sourceConfigPath=runConfig.sourcePath,
        resolvedConfig=runConfigToDict(runConfig),
        writerLogDir=runConfig.experiment.logDir,
    )

    with torch.no_grad():
        batch = controller.initEvolution(dtype=dtype, device=device)
        lossParams = controller._lossParamsForStep(0)

        standardTimings, standardGpuPeaks, standardEval = _run_benchmark(
            controller=controller,
            batch=batch,
            lossParams=lossParams,
            runs=args.runs,
            warmup=args.warmup,
            allowSharedTargetFastPath=False,
        )
        fastTimings, fastGpuPeaks, fastEval = _run_benchmark(
            controller=controller,
            batch=batch,
            lossParams=lossParams,
            runs=args.runs,
            warmup=args.warmup,
            allowSharedTargetFastPath=True,
        )

    assert standardEval is not None and fastEval is not None
    maxScoreDelta = (fastEval.totalLoss - standardEval.totalLoss).abs().max().item()

    print(f"config: {args.config}")
    print(f"device: {device} dtype={dtype}")
    print(f"batch_size: {batch.batchSize} elements: {batch.N}")
    print(f"target_mode: {fastEval.targetMode} fast_path_used: {fastEval.diagnostics.usedSharedTargetFastPath}")
    print(
        "standard_path_ms:"
        f" mean={_mean(standardTimings):.2f}"
        f" max_gpu_mb={_mean(standardGpuPeaks):.1f}"
    )
    print(
        "fast_path_ms:"
        f" mean={_mean(fastTimings):.2f}"
        f" max_gpu_mb={_mean(fastGpuPeaks):.1f}"
    )
    print(f"max_score_delta: {maxScoreDelta:.6e}")


if __name__ == "__main__":
    main()
