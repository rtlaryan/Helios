from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from pathlib import Path

import torch

from train.config import loadRunConfig, resolveDevice, resolveTarget, runConfigToDict
from train.evolve import EvolutionController
from train.nvtx import (
    BENCHMARK_CHUNK_SWEEP_ROOT_RANGE,
    benchmark_chunk_sweep_candidate_range,
    benchmark_chunk_sweep_iteration_range,
    nvtx_range,
)
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


def _default_chunk_candidates(reference: int | None) -> list[int]:
    base = reference if reference is not None else 2_000_000
    candidates = {
        max(1, int(round(base * scale)))
        for scale in (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    }
    return sorted(candidates)


def _geometric_chunk_candidates(start: int, stop: int, factor: float) -> list[int]:
    if stop < start:
        raise ValueError("chunk-stop must be >= chunk-start")
    if factor <= 1.0:
        raise ValueError("chunk-factor must be greater than 1.0")

    values: list[int] = []
    current = float(start)
    limit = float(stop)
    while current <= limit * (1.0 + 1e-12):
        values.append(max(1, int(round(current))))
        current *= factor

    values.append(stop)
    return sorted(set(values))


def _reference_chunk_size(run_config, sweep_mode: str) -> int | None:
    if sweep_mode == "linear":
        return run_config.evolution.linearResponseChunkSize
    if sweep_mode == "wide":
        return run_config.evolution.wideResponseChunkSize
    return (
        run_config.evolution.linearResponseChunkSize
        or run_config.evolution.wideResponseChunkSize
    )


def _resolve_chunk_sizes(args: argparse.Namespace, run_config) -> list[int]:
    if args.chunk_sizes:
        return sorted(set(args.chunk_sizes))
    if args.chunk_start is not None or args.chunk_stop is not None:
        if args.chunk_start is None or args.chunk_stop is None:
            raise ValueError("chunk-start and chunk-stop must be provided together")
        return _geometric_chunk_candidates(args.chunk_start, args.chunk_stop, args.chunk_factor)
    return _default_chunk_candidates(_reference_chunk_size(run_config, args.sweep))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def _run_candidate(
    *,
    chunk_size: int,
    runs: int,
    warmup: int,
    sweep_mode: str,
    controller: EvolutionController,
    batch,
    loss_params,
    allow_shared_target_fast_path: bool,
    clear_cache_between_chunks: bool,
) -> dict[str, object]:
    nvtxEnabled = batch.device.type == "cuda"
    if clear_cache_between_chunks and nvtxEnabled:
        torch.cuda.empty_cache()

    linear_chunk_size, wide_chunk_size = _candidate_chunk_sizes(
        chunk_size=chunk_size,
        sweep_mode=sweep_mode,
        evolution_config=controller.config,
    )
    timings_ms: list[float] = []
    gpu_peaks_mb: list[float] = []
    score_means: list[float] = []
    resolved_linear_chunk_size: int | None = None
    resolved_wide_chunk_size: int | None = None
    target_mode: str | None = None
    fast_path_used = False

    with nvtx_range(
        benchmark_chunk_sweep_candidate_range(chunk_size),
        enabled=nvtxEnabled,
    ):
        for iteration in range(warmup + runs):
            _synchronize(batch.device)
            start = time.perf_counter()
            with nvtx_range(
                benchmark_chunk_sweep_iteration_range(chunk_size, iteration),
                enabled=nvtxEnabled,
            ):
                evaluation = evaluateBatch(
                    batch=batch,
                    target=controller.targetSpec,
                    params=loss_params,
                    targetMode=controller.targetMode,
                    linearResponseChunkSize=linear_chunk_size,
                    wideResponseChunkSize=wide_chunk_size,
                    responseReductionTileCap=controller.config.responseReductionTileCap,
                    allowSharedTargetFastPath=allow_shared_target_fast_path,
                )
            _synchronize(batch.device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            if iteration >= warmup:
                timings_ms.append(elapsed_ms)
                score_means.append(float(evaluation.totalLoss.mean().item()))
                if evaluation.diagnostics.gpuMaxMemoryMB is not None:
                    gpu_peaks_mb.append(float(evaluation.diagnostics.gpuMaxMemoryMB))

            resolved_linear_chunk_size = evaluation.diagnostics.linearResponseChunkSize
            resolved_wide_chunk_size = evaluation.diagnostics.wideResponseChunkSize
            target_mode = evaluation.targetMode
            fast_path_used = evaluation.diagnostics.usedSharedTargetFastPath

    return {
        "requested_chunk_size": chunk_size,
        "resolved_linear_chunk_size": resolved_linear_chunk_size,
        "resolved_wide_chunk_size": resolved_wide_chunk_size,
        "mean_total_ms": _mean(timings_ms),
        "std_total_ms": _stdev(timings_ms),
        "min_total_ms": min(timings_ms) if timings_ms else 0.0,
        "max_total_ms": max(timings_ms) if timings_ms else 0.0,
        "mean_gpu_mb": _mean(gpu_peaks_mb),
        "mean_score": _mean(score_means),
        "target_mode": target_mode,
        "fast_path_used": fast_path_used,
        "timings_ms": timings_ms,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "requested_chunk_size",
        "resolved_linear_chunk_size",
        "resolved_wide_chunk_size",
        "mean_total_ms",
        "std_total_ms",
        "min_total_ms",
        "max_total_ms",
        "mean_gpu_mb",
        "mean_score",
        "target_mode",
        "fast_path_used",
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


def _print_results(rows: list[dict[str, object]]) -> None:
    headers = (
        "req_chunk",
        "linear_chunk",
        "wide_chunk",
        "mean_ms",
        "std_ms",
        "min_ms",
        "max_ms",
        "gpu_mb",
    )
    print(
        f"{headers[0]:>12} {headers[1]:>12} {headers[2]:>12} {headers[3]:>10} "
        f"{headers[4]:>10} {headers[5]:>10} {headers[6]:>10} {headers[7]:>10}"
    )
    for row in rows:
        print(
            f"{int(row['requested_chunk_size']):>12d} "
            f"{int(row['resolved_linear_chunk_size']):>12d} "
            f"{int(row['resolved_wide_chunk_size']):>12d} "
            f"{float(row['mean_total_ms']):>10.2f} "
            f"{float(row['std_total_ms']):>10.2f} "
            f"{float(row['min_total_ms']):>10.2f} "
            f"{float(row['max_total_ms']):>10.2f} "
            f"{float(row['mean_gpu_mb']):>10.1f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep response chunk sizes and benchmark evaluation time",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/evo.yaml", help="Path to YAML config")
    parser.add_argument(
        "--sweep",
        choices=("both", "linear", "wide"),
        default="both",
        help="Which chunk size setting to sweep",
    )
    parser.add_argument(
        "--chunk-sizes",
        nargs="+",
        type=_parse_positive_int,
        help="Explicit chunk sizes to benchmark",
    )
    parser.add_argument(
        "--chunk-start",
        type=_parse_positive_int,
        help="Start of geometric chunk-size sweep",
    )
    parser.add_argument(
        "--chunk-stop",
        type=_parse_positive_int,
        help="End of geometric chunk-size sweep",
    )
    parser.add_argument(
        "--chunk-factor",
        type=float,
        default=2.0,
        help="Multiplier for geometric chunk-size sweeps",
    )
    parser.add_argument("--runs", type=int, default=5, help="Measured runs per chunk size")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per chunk size")
    parser.add_argument(
        "--step",
        type=int,
        default=0,
        help="Training step whose loss settings should be benchmarked",
    )
    parser.add_argument(
        "--sort-by",
        choices=("mean_total_ms", "requested_chunk_size"),
        default="mean_total_ms",
        help="How to order the printed results",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional path to write aggregate benchmark results as CSV",
    )
    parser.add_argument(
        "--allow-shared-target-fast-path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the shared-target fast path when it applies",
    )
    parser.add_argument(
        "--clear-cache-between-chunks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Clear CUDA cache before each chunk-size candidate",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.runs <= 0:
        raise ValueError("runs must be positive")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.step < 0:
        raise ValueError("step must be non-negative")

    run_config = loadRunConfig(args.config)
    chunk_sizes = _resolve_chunk_sizes(args, run_config)
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

    print(f"config: {args.config}")
    print(f"device: {device} dtype={dtype}")
    print(f"sweep: {args.sweep}")
    print(f"benchmark_step: {args.step}")
    print(f"chunk_sizes: {', '.join(str(value) for value in chunk_sizes)}")

    with torch.no_grad():
        with nvtx_range(
            BENCHMARK_CHUNK_SWEEP_ROOT_RANGE,
            enabled=device.type == "cuda",
        ):
            batch = controller.initEvolution(dtype=dtype, device=device)
            loss_params = controller._lossParamsForStep(args.step)

            print(f"batch_size: {batch.batchSize} elements: {batch.N}")
            print(f"wide_grid_size: {loss_params.wide_grid_size}")
            if loss_params.wide_grid_size != controller.lossParams.wide_grid_size:
                print(
                    "wide_grid_size_final:"
                    f" {controller.lossParams.wide_grid_size}"
                )

            rows = [
                _run_candidate(
                    chunk_size=chunk_size,
                    runs=args.runs,
                    warmup=args.warmup,
                    sweep_mode=args.sweep,
                    controller=controller,
                    batch=batch,
                    loss_params=loss_params,
                    allow_shared_target_fast_path=args.allow_shared_target_fast_path,
                    clear_cache_between_chunks=args.clear_cache_between_chunks,
                )
                for chunk_size in chunk_sizes
            ]

    sorted_rows = sorted(rows, key=lambda row: row[args.sort_by])
    best = sorted_rows[0]

    print(
        "best:"
        f" requested={int(best['requested_chunk_size'])}"
        f" mean_total_ms={float(best['mean_total_ms']):.2f}"
        f" linear={int(best['resolved_linear_chunk_size'])}"
        f" wide={int(best['resolved_wide_chunk_size'])}"
    )
    _print_results(sorted_rows)

    if args.csv is not None:
        _write_csv(args.csv, sorted_rows)
        print(f"csv: {args.csv}")


if __name__ == "__main__":
    main()
