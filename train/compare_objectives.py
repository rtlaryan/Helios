from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from scripts.batchFactory import generateBatch
from scripts.targetSpec import TargetBatch, TargetLike, TargetSpec, inferTargetCenter
from train.config import loadRunConfig, resolveDevice, resolveTarget, runConfigToDict
from train.evaluation_utils import evaluation_loss_record
from train.objective import LossConfig, evaluateBatch
from train.objective_v2 import LossConfigV2, clearCachesV2, evaluateBatchV2


def _clearV1Caches() -> None:
    import train.objective as objective_v1

    objective_v1._SHARED_TARGET_CACHE.clear()
    objective_v1._WIDE_GRID_CACHE.clear()


def _with_seeded_rng(seed: int, device: torch.device, fn: Callable[[], Any]) -> Any:
    cpuState = torch.random.get_rng_state()
    cudaStates = None
    if torch.cuda.is_available():
        cudaStates = torch.cuda.get_rng_state_all()

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    try:
        return fn()
    finally:
        torch.random.set_rng_state(cpuState)
        if cudaStates is not None:
            torch.cuda.set_rng_state_all(cudaStates)


def _build_fixture_batch(target: TargetLike, config: Any, device: torch.device, dtype: torch.dtype) -> Any:
    targetLLA = None
    if config.evolution.initialWeightsType == "directed":
        targetLLA = inferTargetCenter(target)
    return _with_seeded_rng(
        1234,
        device,
        lambda: generateBatch(
            spec=config.array,
            batchSize=config.evolution.batchSize,
            device=device,
            dtype=dtype,
            weightsType=config.evolution.initialWeightsType,
            targetLLA=targetLLA,
            generator=None,
        ),
    )


def _loss_params_for_v1(config: Any) -> LossConfig:
    if getattr(config, "loss", None) is not None:
        return config.loss
    return LossConfig()


def _loss_params_for_v2(config: Any) -> LossConfigV2:
    if getattr(config, "lossV2", None) is not None:
        return config.lossV2
    return LossConfigV2()


def _evaluate_objective(
    objectiveVersion: str,
    batch: Any,
    target: TargetLike,
    config: Any,
    collectStats: bool = False,
) -> Any:
    if objectiveVersion == "v1":
        return evaluateBatch(
            batch=batch,
            target=target,
            params=_loss_params_for_v1(config),
            targetMode=config.experiment.targetMode,
            linearResponseChunkSize=config.evolution.linearResponseChunkSize,
            wideResponseChunkSize=config.evolution.wideResponseChunkSize,
        )
    if objectiveVersion == "v2":
        return evaluateBatchV2(
            batch=batch,
            target=target,
            params=_loss_params_for_v2(config),
            targetMode=config.experiment.targetMode,
            linearResponseChunkSize=config.evolution.linearResponseChunkSize,
            wideResponseChunkSize=config.evolution.wideResponseChunkSize,
            collectStats=collectStats,
        )
    raise ValueError(f"unsupported objectiveVersion: {objectiveVersion}")


def _timed_evaluations(
    objectiveVersion: str,
    batch: Any,
    target: TargetLike,
    config: Any,
    repeats: int,
    warmups: int,
    clearCaches: bool,
) -> dict[str, Any]:
    if clearCaches:
        _clearV1Caches()
        clearCachesV2()

    for _ in range(max(0, warmups)):
        _evaluate_objective(objectiveVersion, batch, target, config, collectStats=False)

    durations: list[float] = []
    stageSamples: dict[str, list[float]] = {}
    finalEvaluation = None
    for _ in range(max(1, repeats)):
        startedAt = time.perf_counter()
        evaluation = _evaluate_objective(
            objectiveVersion,
            batch,
            target,
            config,
            collectStats=(objectiveVersion == "v2"),
        )
        durations.append(time.perf_counter() - startedAt)
        finalEvaluation = evaluation
        if getattr(evaluation, "stats", None):
            for key, value in evaluation.stats.items():
                stageSamples.setdefault(key, []).append(float(value))
    summary = {
        "meanSeconds": float(statistics.mean(durations)),
        "medianSeconds": float(statistics.median(durations)),
        "stdSeconds": float(statistics.pstdev(durations) if len(durations) > 1 else 0.0),
        "repetitions": int(max(1, repeats)),
    }
    if stageSamples:
        summary["stages"] = {
            key: {
                "meanSeconds": float(statistics.mean(values)),
                "medianSeconds": float(statistics.median(values)),
                "stdSeconds": float(statistics.pstdev(values) if len(values) > 1 else 0.0),
            }
            for key, values in stageSamples.items()
        }
    if getattr(finalEvaluation, "metadata", None):
        summary["metadata"] = finalEvaluation.metadata
    return {"timing": summary, "evaluation": finalEvaluation}


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, dim=0)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(order.shape[0], device=values.device, dtype=torch.float32)
    return ranks


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() <= 1:
        return 1.0
    rankA = _rankdata(a)
    rankB = _rankdata(b)
    rankA = rankA - rankA.mean()
    rankB = rankB - rankB.mean()
    denominator = rankA.norm() * rankB.norm()
    if float(denominator.item()) <= 0.0:
        return 1.0
    return float((rankA * rankB).sum().item() / denominator.item())


def _topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    topA = set(torch.argsort(a)[:k].tolist())
    topB = set(torch.argsort(b)[:k].tolist())
    return float(len(topA & topB) / max(1, k))


def _serialize_target_mode(target: TargetLike) -> str:
    return "per_sample" if isinstance(target, TargetBatch) else "shared"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Helios objective_v1 and objective_v2")
    parser.add_argument("--config", required=True, help="Path to an evolution YAML config")
    parser.add_argument("--warmups", type=int, default=2, help="Number of untimed warmup evaluations")
    parser.add_argument("--repeats", type=int, default=5, help="Number of timed evaluations per objective")
    args = parser.parse_args()

    runConfig = loadRunConfig(args.config)
    device, dtype = resolveDevice(runConfig)
    target = resolveTarget(runConfig)
    batch = _build_fixture_batch(target, runConfig, device=device, dtype=dtype)

    coldV1 = _timed_evaluations("v1", batch, target, runConfig, repeats=1, warmups=0, clearCaches=True)
    coldV2 = _timed_evaluations("v2", batch, target, runConfig, repeats=1, warmups=0, clearCaches=True)
    warmV1 = _timed_evaluations("v1", batch, target, runConfig, repeats=args.repeats, warmups=args.warmups, clearCaches=False)
    warmV2 = _timed_evaluations("v2", batch, target, runConfig, repeats=args.repeats, warmups=args.warmups, clearCaches=False)

    evalV1 = warmV1["evaluation"]
    evalV2 = warmV2["evaluation"]
    batchSize = int(evalV1.totalLoss.shape[0])
    topk = min(10, batchSize)
    ranking = {
        "spearman": _spearman(evalV1.totalLoss.detach().cpu(), evalV2.totalLoss.detach().cpu()),
        "topKOverlap": _topk_overlap(evalV1.totalLoss.detach().cpu(), evalV2.totalLoss.detach().cpu(), topk),
        "winnerIdentityAgreement": bool(int(torch.argmin(evalV1.totalLoss).item()) == int(torch.argmin(evalV2.totalLoss).item())),
        "topK": int(topk),
        "v1BestSample": int(torch.argmin(evalV1.totalLoss).item()),
        "v2BestSample": int(torch.argmin(evalV2.totalLoss).item()),
    }
    warmV2OverV1 = warmV2["timing"]["meanSeconds"] / max(warmV1["timing"]["meanSeconds"], 1e-12)
    report = {
        "experimentName": runConfig.experiment.name,
        "mode": _serialize_target_mode(target),
        "objectiveVersion": runConfig.objectiveVersion,
        "config": runConfigToDict(runConfig),
        "cold": {
            "v1": coldV1["timing"],
            "v2": coldV2["timing"],
        },
        "warm": {
            "v1": warmV1["timing"],
            "v2": warmV2["timing"],
            "v2_over_v1_ratio": float(warmV2OverV1),
        },
        "ranking": ranking,
        "lossExample": {
            "v1": evaluation_loss_record(evalV1, int(torch.argmin(evalV1.totalLoss).item())),
            "v2": evaluation_loss_record(evalV2, int(torch.argmin(evalV2.totalLoss).item())),
        },
    }

    outputDir = Path(runConfig.experiment.archiveDir) / runConfig.experiment.name / "benchmarks"
    outputDir.mkdir(parents=True, exist_ok=True)
    outputPath = outputDir / f"objective_compare_{report['mode']}.json"
    with outputPath.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=False)

    print(f"Mode: {report['mode']}")
    print(f"Cold v1: {coldV1['timing']['meanSeconds']:.6f}s")
    print(f"Cold v2: {coldV2['timing']['meanSeconds']:.6f}s")
    print(f"Warm v1 mean: {warmV1['timing']['meanSeconds']:.6f}s")
    print(f"Warm v2 mean: {warmV2['timing']['meanSeconds']:.6f}s")
    print(f"Warm v2/v1 ratio: {report['warm']['v2_over_v1_ratio']:.3f}")
    print(f"Spearman: {ranking['spearman']:.4f}")
    print(f"Top-{ranking['topK']} overlap: {ranking['topKOverlap']:.4f}")
    print(f"Winner agreement: {ranking['winnerIdentityAgreement']}")
    print(f"Wrote benchmark report to {outputPath}")


if __name__ == "__main__":
    main()
