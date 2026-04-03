from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from train.nvtx import (
    BENCHMARK_CHUNK_SWEEP_CANDIDATE_PREFIX,
    BENCHMARK_CHUNK_SWEEP_ROOT_RANGE,
)


_CHUNK_SIZE_PATTERN = re.compile(
    rf"^{re.escape(BENCHMARK_CHUNK_SWEEP_CANDIDATE_PREFIX)}(?P<chunk_size>\d+)$"
)
_STAGE_PREFIXES = (
    "helios.evaluate.",
    BENCHMARK_CHUNK_SWEEP_CANDIDATE_PREFIX,
    "helios.benchmark_chunk_sweep.iteration.",
)


def _extract_json_payload(output: str) -> list[dict[str, Any]]:
    lines = output.splitlines()
    payloadStart = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip() in {"[", "[]"} or line.lstrip().startswith("[{")
        ),
        None,
    )
    if payloadStart is None:
        raise ValueError(f"unable to find JSON payload in nsys output:\n{output}")
    payload = json.loads("\n".join(lines[payloadStart:]))
    if not isinstance(payload, list):
        raise ValueError("expected nsys stats JSON output to be a list")
    return payload


def _normalize_nvtx_range_name(raw: str) -> str:
    return raw.lstrip(":")


def _sum_int(rows: list[dict[str, Any]], key: str) -> int:
    total = 0
    for row in rows:
        value = row.get(key, 0)
        if value is None:
            continue
        total += int(value)
    return total


def _sum_float(rows: list[dict[str, Any]], key: str) -> float:
    total = 0.0
    for row in rows:
        value = row.get(key, 0.0)
        if value is None:
            continue
        total += float(value)
    return total


def _report_path(output_prefix: Path, suffix: str) -> Path:
    return Path(f"{output_prefix}{suffix}")


def _resolve_nsys_input_path(output_prefix: Path) -> Path:
    report_path = _report_path(output_prefix, ".nsys-rep")
    sqlite_path = _report_path(output_prefix, ".sqlite")
    qdstrm_path = _report_path(output_prefix, ".qdstrm")

    if sqlite_path.exists():
        return sqlite_path
    if report_path.exists():
        return report_path
    if qdstrm_path.exists():
        raise RuntimeError(
            "nsys collected a raw .qdstrm trace but did not finish importing it into "
            f".nsys-rep/.sqlite.\n"
            f"Found: {qdstrm_path}\n"
            "This usually means the trace was too large or the importer failed.\n"
            "Try profiling fewer chunk sizes, using --runs 1 --warmup 0, or profiling one "
            "chunk size per Nsight run."
        )
    raise RuntimeError(
        "nsys did not produce a .sqlite, .nsys-rep, or .qdstrm artifact for output prefix "
        f"{output_prefix}"
    )


def _run_subprocess(command: list[str], capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def _run_nsys_stats(
    input_path: Path,
    report: str,
    filter_nvtx: str | None = None,
) -> list[dict[str, Any]]:
    command = [
        "nsys",
        "stats",
        "--report",
        report,
        "--format",
        "json",
        "--output",
        "-",
    ]
    if filter_nvtx is not None:
        command.extend(["--filter-nvtx", filter_nvtx])
    command.append(str(input_path))
    result = _run_subprocess(command, capture_output=True)
    return _extract_json_payload(result.stdout)


def _candidate_region_summary(input_path: Path, range_name: str, wall_time_ns: int) -> dict[str, Any]:
    kernel_rows = _run_nsys_stats(input_path, "cuda_gpu_kern_sum", filter_nvtx=range_name)
    api_rows = _run_nsys_stats(input_path, "cuda_api_sum", filter_nvtx=range_name)
    mem_time_rows = _run_nsys_stats(input_path, "cuda_gpu_mem_time_sum", filter_nvtx=range_name)
    mem_size_rows = _run_nsys_stats(input_path, "cuda_gpu_mem_size_sum", filter_nvtx=range_name)

    kernel_time_ns = _sum_int(kernel_rows, "Total Time (ns)")
    return {
        "range_name": range_name,
        "chunk_size": int(_CHUNK_SIZE_PATTERN.fullmatch(range_name).group("chunk_size")),
        "wall_time_ns": wall_time_ns,
        "gpu_kernel_time_ns": kernel_time_ns,
        "cuda_api_time_ns": _sum_int(api_rows, "Total Time (ns)"),
        "kernel_launch_count": _sum_int(kernel_rows, "Instances"),
        "gpu_mem_time_ns": _sum_int(mem_time_rows, "Total Time (ns)"),
        "gpu_mem_total_mb": _sum_float(mem_size_rows, "Total (MB)"),
        "gpu_active_ratio": (kernel_time_ns / wall_time_ns) if wall_time_ns > 0 else 0.0,
    }


def _build_summary(
    output_prefix: Path,
    command: list[str],
    capture_range: str,
) -> dict[str, Any]:
    sqlite_path = _report_path(output_prefix, ".sqlite")
    report_path = _report_path(output_prefix, ".nsys-rep")
    input_path = _resolve_nsys_input_path(output_prefix)
    version = _run_subprocess(["nsys", "--version"], capture_output=True).stdout.strip()
    nvtx_rows = _run_nsys_stats(input_path, "nvtx_sum")

    ranges: list[dict[str, Any]] = []
    candidate_ranges: list[dict[str, Any]] = []
    root_range: dict[str, Any] | None = None

    for row in nvtx_rows:
        range_name = _normalize_nvtx_range_name(str(row.get("Range", "")))
        total_time_ns = int(row.get("Total Time (ns)", 0))
        instances = int(row.get("Instances", 0))
        normalized = {
            "range_name": range_name,
            "wall_time_ns": total_time_ns,
            "instances": instances,
            "style": row.get("Style"),
        }
        if range_name.startswith(_STAGE_PREFIXES):
            ranges.append(normalized)
        if range_name == BENCHMARK_CHUNK_SWEEP_ROOT_RANGE:
            root_range = normalized
        if _CHUNK_SIZE_PATTERN.fullmatch(range_name):
            candidate_ranges.append(normalized)

    regions = [
        _candidate_region_summary(
            input_path=input_path,
            range_name=range_info["range_name"],
            wall_time_ns=range_info["wall_time_ns"],
        )
        for range_info in candidate_ranges
    ]
    regions.sort(key=lambda region: region["chunk_size"])
    ranges.sort(key=lambda row: row["range_name"])

    summary = {
        "nsys_version": version,
        "output_prefix": str(output_prefix),
        "report_path": str(report_path),
        "sqlite_path": str(sqlite_path),
        "capture_range": capture_range,
        "benchmark_command": command,
        "root_range": root_range,
        "regions": regions,
        "ranges": ranges,
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a Helios benchmark under nsys and extract a JSON summary",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Prefix for generated .nsys-rep, .sqlite, and summary files",
    )
    parser.add_argument(
        "--nsys-capture-range",
        choices=("none", "nvtx"),
        default="none",
        help="Optional nsys capture-range mode",
    )
    parser.add_argument(
        "--keep-nsys-rep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the generated .nsys-rep artifact after summary extraction",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Benchmark command to run after '--'",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("expected a benchmark command after '--'")

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    nsys_command = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--stats=true",
        "--force-overwrite=true",
        "-o",
        str(output_prefix),
    ]
    if args.nsys_capture_range == "nvtx":
        nsys_command.extend(
            [
                "--capture-range=nvtx",
                f"--nvtx-capture={BENCHMARK_CHUNK_SWEEP_ROOT_RANGE}",
                "--capture-range-end=stop-shutdown",
            ]
        )
    nsys_command.extend(command)

    print(f"output_prefix: {output_prefix}")
    print(f"nsys_capture_range: {args.nsys_capture_range}")
    print(f"command: {' '.join(command)}")

    _run_subprocess(nsys_command, capture_output=False)

    summary = _build_summary(
        output_prefix=output_prefix,
        command=command,
        capture_range=args.nsys_capture_range,
    )

    summary_path = _report_path(output_prefix, "_nsys_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"summary_json: {summary_path}")
    print(f"nsys_report: {_report_path(output_prefix, '.nsys-rep')}")
    print(f"nsys_sqlite: {_report_path(output_prefix, '.sqlite')}")

    report_path = _report_path(output_prefix, ".nsys-rep")
    if not args.keep_nsys_rep and report_path.exists():
        report_path.unlink()
        print(f"deleted_nsys_report: {report_path}")


if __name__ == "__main__":
    main()
