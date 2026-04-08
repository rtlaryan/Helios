# Array Response Tiling Notes

Last updated: 2026-04-03

## Goal

Keep the array-response hot path fast without changing beamforming results.

The hot path lives in:

- `scripts/arraySimulation.py`
- `train/objective.py`

## Current Code State

- `arrayResponseCore(...)` is the only production implementation.
- `_arrayResponseCoreReference(...)` remains private and test-only as the parity oracle.
- The old `arrayResponseCoreOLD(...)` path has been removed.
- Response tiling now uses one heuristic:
  - cap `Nc` with `responseReductionTileCap`
  - if the remaining batch fits, use the full remaining batch as `Bc`
  - spend the remaining chunk budget on `Pc`
  - if the batch does not fit, fall back to batch striping
- When batch striping is required on CUDA, Helios processes up to 4 disjoint batch stripes concurrently with CUDA streams. CPU execution stays serial.

## Why This Shape

The main constraint is still the reduction depth across elements. Capping `Nc` keeps the reduction from growing into the most expensive axis, then the kernel tries to preserve batch parallelism before shrinking into single-grid-point tiles.

This keeps the tuning surface small:

- `linearResponseChunkSize`
- `wideResponseChunkSize`
- `responseReductionTileCap`

There is no longer a strategy menu to benchmark or reason about.

## Measurement Workflow

To tune chunk sizes on a specific config:

```bash
conda run -n helios python -m train.benchmark_chunk_sweep --config configs/evo.yaml
```

To profile a chosen chunk-size sweep under Nsight Systems:

```bash
conda run -n helios python -m train.profile_with_nsys \
  --output-prefix data/profiling/evo_linear \
  -- \
  python -m train.benchmark_chunk_sweep --config configs/evo.yaml --sweep linear --chunk-sizes 5e5
```

The benchmark and evaluation paths still emit NVTX ranges so `nsys stats --filter-nvtx ...` can isolate full runs, chunk candidates, per-iteration evaluations, and the major evaluation stages.

## Validation Expectations

When changing `arrayResponseCore(...)`, keep these checks in place:

- exact CPU parity against `_arrayResponseCoreReference(...)`
- CUDA parity for both full-batch tiles and batch-split tiles
- chunk-shape invariants:
  - `Nc <= responseReductionTileCap`
  - `Bc == batchRemaining` whenever `batchRemaining * Nc <= chunkSize`
  - fallback batch splitting uses `Pc == 1`
  - `Bc * Nc * Pc <= chunkSize`
