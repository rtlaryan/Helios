# Array Response Tiling Notes

Last updated: 2026-04-01

## Goal

Find a response chunk tiling strategy that improves end-to-end evaluation time without changing the beamforming result in any meaningful way.

The hot path lives in:

- `scripts/arraySimulation.py`
- `train/objective.py`

## Current Code State

- `scripts/arraySimulation.py`
  - `_arrayResponseCoreReference(...)` is a frozen copy of the pre-optimization implementation and acts as the testing ground truth.
  - `arrayResponseCore(...)` now writes power directly into the real output tensor and reuses a complex scratch tile instead of materializing a full-grid complex staging buffer.
  - `chooseChunkShape(...)` still defaults to the original balanced heuristic.
  - Experimental strategies can now be selected explicitly:
    - `balanced`
    - `cap_reduction`
    - `grid_first`
- `tests/test_array_simulation.py`
  - exact default-parity tests against `_arrayResponseCoreReference(...)`
  - strategy-selection coverage for the new chunk-shape selector
- `train/benchmark_chunk_shapes.py`
  - compares named chunk-shape strategies at fixed chunk sizes

## Why This Investigation Exists

Observed behavior:

- very small chunk sizes are slow because launch count explodes
- medium chunk sizes perform best
- very large chunk sizes make the GPU look busier, but wall time rises again

Working interpretation:

- tiny chunks are launch-bound
- large chunks are dominated by heavier `phase -> exp -> accumulate` work and memory traffic

## Existing Measurement Summary

### 2026-03-31 Chunk Sweep Highlights

End-to-end chunk sweep on `configs/evo.yaml` showed:

- full `both` sweep best around `1.6e6` with `~14.13 s`
- `linear`-only sweep with wide chunk fixed high favored a smaller linear chunk, around `5e5`

Representative output recorded from the terminal:

| Mode | Best chunk(s) | Mean time |
| --- | --- | ---: |
| `both` | `1.6e6 / 1.6e6` | `14125.36 ms` |
| `linear` only | `5e5 / 2e9` | `30465.75 ms` |

### 2026-03-31 Nsight Systems Findings

Single-chunk Nsight profiles were captured under `data/profiling/`.

Important artifacts:

- `data/profiling/evo_linear_1e5_nsys_summary.json`
- `data/profiling/evo_linear_5e5_nsys_summary.json`
- `data/profiling/evo_linear_1e6_nsys_summary.json`
- `data/profiling/evo_linear_5e6_nsys_summary.json`
- `data/profiling/evo_linear_1e7_nsys_summary.json`

High-level pattern:

| Chunk | Wall time | Kernel launches | GPU active ratio |
| --- | ---: | ---: | ---: |
| `1e5` | `77.1 s` | `7.86M` | `0.215` |
| `5e5` | `15.3 s` | `1.56M` | `0.678` |
| `1e6` | `15.9 s` | `0.79M` | `0.870` |
| `5e6` | `18.5 s` | `0.16M` | `0.959` |
| `1e7` | `19.0 s` | `0.08M` | `0.968` |

Interpretation:

- small chunks: too many tiny kernels
- large chunks: fewer launches, but each launch gets much heavier
- `helios.evaluate.linear_response` is the dominant stage

### 2026-04-01 A/B Of The First Memory-Traffic Cleanup

The optimized `arrayResponseCore(...)` was benchmarked directly against `_arrayResponseCoreReference(...)` on the same evaluation path.

Results:

| Chunk | Reference | Optimized | Delta |
| --- | ---: | ---: | ---: |
| `5e5` | `15611.79 ms` | `15614.30 ms` | `-0.02%` |
| `1e6` | `15238.03 ms` | `15114.49 ms` | `+0.81%` |
| `5e6` | `18656.75 ms` | `18611.09 ms` | `+0.24%` |
| `1e7` | `19248.63 ms` | `19264.18 ms` | `-0.08%` |

The score delta was `0.0` at the displayed precision for all tested chunk sizes.

Conclusion:

- removing the full-grid complex staging buffer was safe
- but it did not materially change end-to-end performance
- the main large-chunk bottleneck is probably still tile shape, not final output staging

### 2026-04-01 First Strategy-Comparison Smoke Run

Command:

```bash
conda run -n helios python -m train.benchmark_chunk_shapes \
  --config configs/evo.yaml \
  --sweep linear \
  --chunk-sizes 1e6 \
  --strategies balanced cap_reduction grid_first \
  --runs 1 --warmup 0
```

Result:

| Strategy | Chunk | Mean time | Peak GPU MB |
| --- | ---: | ---: | ---: |
| `grid_first` | `1e6` | `14762.37 ms` | `1079.4` |
| `cap_reduction` | `1e6` | `15534.30 ms` | `1079.4` |
| `balanced` | `1e6` | `16377.18 ms` | `38216.2` |

Notes:

- this was only a 1-run smoke test, not a stable benchmark yet
- even so, it suggests tile shape can matter much more than the first output-buffer cleanup
- `grid_first` is worth repeating with more samples

## Current Heuristic Discussion

The balanced heuristic chooses `Bc`, `Nc`, and `Pc` almost symmetrically.

The current leading hypothesis is:

- treat `Nc` as the most dangerous axis because it deepens the reduction
- let `Pc` absorb extra chunk budget before `Nc` grows too much
- treat `Bc` as the least important axis to optimize around, because batch is already small

That is why the first experimental variants are:

- `balanced`: current heuristic
- `cap_reduction`: start from balanced, then cap `Nc` and reassign leftover budget to `Pc`
- `grid_first`: choose a bounded `Nc`, then size `Pc`, then let `Bc` take the remaining budget

## Suggested Next Experiments

Use the new benchmark to compare strategies directly:

```bash
conda run -n helios python -m train.benchmark_chunk_shapes \
  --config configs/evo.yaml \
  --sweep linear \
  --chunk-sizes 1e6 5e6 1e7 \
  --strategies balanced cap_reduction grid_first \
  --reduction-tile-cap 256
```

If one strategy looks promising:

1. rerun it with more samples
2. compare score deltas against the reference implementation
3. profile that strategy under `nsys` to confirm the kernel mix changes the way we expect
