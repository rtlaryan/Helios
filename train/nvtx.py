from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


BENCHMARK_CHUNK_SWEEP_ROOT_RANGE = "helios.benchmark_chunk_sweep.run"
BENCHMARK_CHUNK_SWEEP_CANDIDATE_PREFIX = "helios.benchmark_chunk_sweep.chunk_size="
BENCHMARK_CHUNK_SWEEP_ITERATION_PREFIX = "helios.benchmark_chunk_sweep.iteration.chunk_size="

EVALUATE_TOTAL_RANGE = "helios.evaluate.total"
EVALUATE_TARGET_PROJECTION_RANGE = "helios.evaluate.target_projection"
EVALUATE_LINEAR_RESPONSE_RANGE = "helios.evaluate.linear_response"
EVALUATE_WIDE_RESPONSE_RANGE = "helios.evaluate.wide_response"
EVALUATE_SUPPORT_MASK_RANGE = "helios.evaluate.support_mask"
EVALUATE_LOSS_ASSEMBLY_RANGE = "helios.evaluate.loss_assembly"


def benchmark_chunk_sweep_candidate_range(chunk_size: int) -> str:
    return f"{BENCHMARK_CHUNK_SWEEP_CANDIDATE_PREFIX}{int(chunk_size)}"


def benchmark_chunk_sweep_iteration_range(chunk_size: int, iteration: int) -> str:
    return (
        f"{BENCHMARK_CHUNK_SWEEP_ITERATION_PREFIX}{int(chunk_size)}"
        f".iteration={int(iteration)}"
    )


@contextmanager
def nvtx_range(message: str, enabled: bool = True) -> Iterator[None]:
    if not enabled:
        yield
        return

    torch.cuda.nvtx.range_push(message)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
