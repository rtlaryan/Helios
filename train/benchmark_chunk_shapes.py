from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from pathlib import Path

import torch

from scripts.arraySimulation import ChunkShapeStrategy, useChunkShapeStrategy
from train.config import loadRunConfig, resolveDevice, resolveTarget, runConfigToDict
from train.evolve import EvolutionController
from train.objective import evaluateBatch


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _stdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _parse_positive_int(raw: str) -> int:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid numeric value: {raw}") from exc

    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"value must be positive: {raw}")

    rounded = round(value)
    if not math.isclose(value, rounded, rel_tol=1e-9, abs_tol=1e-9):
        raise argparse.ArgumentTypeError(f"value must be an integer-sized quantity: {raw}")
    return int(rounded)


def _candidate_chunk_sizes(
    chunk_size: int,
    sweep_mode: str,
    evolution_config,
) -> tuple[int | None, int | None]:
    if sweep_mode == "linear":
        return chunk_size, evolution_config.wideResponseChunkSize
    if sweep_mode == "wide":
        return evolution_config.linearResponseChunkSize, chunk_size
    return chunk_size, chunk_size


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "strategy",
        "reduction_tile_cap",
        "requested_chunk_size",
        "resolved_linear_chunk_size",
        "resolved_wide_chunk_size",
        "mean_total_ms",
        "std_total_ms",
        "min_total_ms",
        "max_total_ms",
        "mean_gpu_mb",
        "mean_score",
        "timings_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["timings_ms"] = " ".join(
                f"{value:.3f}" for value in serialized["timings_ms"]
            )
            writer.writerow(serialized)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark named response chunk-shape heuristics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/evo.yaml", help="Path to YAML config")
    parser.add_argument(
        "--sweep",
        choices=("both", "linear", "wide"),
        default="linear",
        help="Which response chunk size setting to vary",
    )
    parser.add_argument(
        "--chunk-sizes",
        nargs="+",
        type=_parse_positive_int,
        required=True,
        help="Chunk sizes to benchmark for each strategy",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=("balanced", "cap_reduction", "grid_first"),
        default=("balanced", "cap_reduction", "grid_first"),
        help="Chunk-shape strategies to compare",
    )
    parser.add_argument(
        "--reduction-tile-cap",
        type=int,
        default=256,
        help="Maximum Nc used by reduction-capped strategies",
    )
    parser.add_argument("--runs", type=int, default=3, help="Measured runs per candidate")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per candidate")
    parser.add_argument(
        "--allow-shared-target-fast-path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the shared-target fast path when it applies",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional path to write aggregate results as CSV",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.runs <= 0:
        raise ValueError("runs must be positive")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.reduction_tile_cap <= 0:
        raise ValueError("reduction-tile-cap must be positive")

    run_config = loadRunConfig(args.config)
    target = resolveTarget(run_config)
    device, dtype = resolveDevice(run_config)
    controller = EvolutionController(
        config=run_config.evolution,
        targetSpec=target,
        arraySpec=run_config.array,
        lossParams=run_config.loss,
        experimentName=run_config.experiment.name,
        archiveRoot=run_config.experiment.archiveDir,
        loggingConfig=run_config.logging,
        checkpointConfig=run_config.checkpoint,
        workerConfig=run_config.workers,
        targetMode=run_config.experiment.targetMode,
        sourceConfigPath=run_config.sourcePath,
        resolvedConfig=runConfigToDict(run_config),
        writerLogDir=run_config.experiment.logDir,
    )

    rows: list[dict[str, object]] = []

    with torch.no_grad():
        batch = controller.initEvolution(dtype=dtype, device=device)
        loss_params = controller._lossParamsForStep(0)

        for strategy in args.strategies:
            typedStrategy = strategy
            with useChunkShapeStrategy(
                typedStrategy,
                reductionTileCap=args.reduction_tile_cap,
            ):
                for chunk_size in args.chunk_sizes:
                    linear_chunk_size, wide_chunk_size = _candidate_chunk_sizes(
                        chunk_size=chunk_size,
                        sweep_mode=args.sweep,
                        evolution_config=controller.config,
                    )
                    timings_ms: list[float] = []
                    gpu_peaks_mb: list[float] = []
                    score_means: list[float] = []
                    resolved_linear_chunk_size: int | None = None
                    resolved_wide_chunk_size: int | None = None

                    for iteration in range(args.warmup + args.runs):
                        _synchronize(batch.device)
                        start = time.perf_counter()
                        evaluation = evaluateBatch(
                            batch=batch,
                            target=controller.targetSpec,
                            params=loss_params,
                            targetMode=controller.targetMode,
                            linearResponseChunkSize=linear_chunk_size,
                            wideResponseChunkSize=wide_chunk_size,
                            responseChunkShapeStrategy=typedStrategy,
                            responseReductionTileCap=args.reduction_tile_cap,
                            allowSharedTargetFastPath=args.allow_shared_target_fast_path,
                        )
                        _synchronize(batch.device)
                        elapsed_ms = (time.perf_counter() - start) * 1000.0

                        if iteration >= args.warmup:
                            timings_ms.append(elapsed_ms)
                            score_means.append(float(evaluation.totalLoss.mean().item()))
                            if evaluation.diagnostics.gpuMaxMemoryMB is not None:
                                gpu_peaks_mb.append(float(evaluation.diagnostics.gpuMaxMemoryMB))

                        resolved_linear_chunk_size = (
                            evaluation.diagnostics.linearResponseChunkSize
                        )
                        resolved_wide_chunk_size = (
                            evaluation.diagnostics.wideResponseChunkSize
                        )

                    rows.append(
                        {
                            "strategy": typedStrategy,
                            "reduction_tile_cap": args.reduction_tile_cap,
                            "requested_chunk_size": chunk_size,
                            "resolved_linear_chunk_size": resolved_linear_chunk_size,
                            "resolved_wide_chunk_size": resolved_wide_chunk_size,
                            "mean_total_ms": _mean(timings_ms),
                            "std_total_ms": _stdev(timings_ms),
                            "min_total_ms": min(timings_ms) if timings_ms else 0.0,
                            "max_total_ms": max(timings_ms) if timings_ms else 0.0,
                            "mean_gpu_mb": _mean(gpu_peaks_mb),
                            "mean_score": _mean(score_means),
                            "timings_ms": timings_ms,
                        }
                    )

    rows.sort(key=lambda row: (int(row["requested_chunk_size"]), float(row["mean_total_ms"])))

    print(f"config: {args.config}")
    print(f"device: {device} dtype={dtype}")
    print(f"batch_size: {batch.batchSize} elements: {batch.N}")
    print(f"sweep: {args.sweep}")
    print(f"reduction_tile_cap: {args.reduction_tile_cap}")
    print(
        f"{'strategy':>14} {'req_chunk':>12} {'linear_chunk':>12} {'wide_chunk':>12} "
        f"{'mean_ms':>10} {'std_ms':>10} {'gpu_mb':>10}"
    )
    for row in rows:
        print(
            f"{str(row['strategy']):>14} "
            f"{int(row['requested_chunk_size']):>12d} "
            f"{int(row['resolved_linear_chunk_size']):>12d} "
            f"{int(row['resolved_wide_chunk_size']):>12d} "
            f"{float(row['mean_total_ms']):>10.2f} "
            f"{float(row['std_total_ms']):>10.2f} "
            f"{float(row['mean_gpu_mb']):>10.1f}"
        )

    if args.csv is not None:
        _write_csv(args.csv, rows)


if __name__ == "__main__":
    main()
