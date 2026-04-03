# Helios Training Pipeline

This directory contains the machine learning pipelines for optimizing phased array element configurations. It uses evolutionary search to generate near-optimal beamforming patterns based on arbitrary targets.

## Core Modules

- `evolve.py`: The entry point and main logic for the evolutionary trainer. It orchestrates population generation, evaluation, rank-weighted parent selection, crossover, mutation, adaptive exploration schedules, checkpointing, and dataset logging.
- `objective.py`: Defines the objective (loss/reward) functions used to evaluate the fitness of a specific array configuration. The current loss combines local target-map matching, wide-angle outside-support suppression, and a small efficiency regularizer.
- `config.py`: Loads typed YAML run configs and resolves target sources.
- `run.py`: CLI entrypoint for YAML-driven training runs.
- `persistence.py`: Async-capable dataset and checkpoint writers.
- `benchmark_evaluate.py`: Benchmarks evaluation timing/memory with and without the shared-target fast path.
- `benchmark_chunk_sweep.py`: Sweeps response chunk sizes and measures end-to-end evaluation time for each candidate.
- `benchmark_chunk_shapes.py`: Compares named chunk-shape heuristics at fixed chunk sizes.
- `profile_with_nsys.py`: Launches a benchmark under `nsys`, keeps the raw trace, and writes a JSON summary of NVTX-filtered GPU activity.

## YAML Workflow

Training can now be driven from a single YAML file that holds:

- experiment settings
- device and dtype
- array search space
- evolution settings
- loss settings
- logging and dataset settings
- checkpoint settings
- worker settings
- target loading settings

Run a config with:

```bash
conda run -n helios python -m train.run --config path/to/run.yaml
```

`evolution.initialWeightsType` controls how the first population is initialized. Supported values are `random`, `uniform`, and `directed`. When `directed` is used, the training pipeline derives a steering center automatically from the resolved target map.

The trainer now uses a clone/crossover/mutate/random population update. Parent choice is rank-weighted inside the configured parent pool. Random injection decays over time, the released budget is absorbed by mutation, and sigma values use both exponential decay and diversity-aware minimum floors. If progress stalls, the trainer temporarily boosts sigma values and random injection to re-open exploration. `wide_grid_size` can also ramp from a smaller start value to the final loss fidelity for faster early generations.
Shared-target runs with homogeneous array geometry now reuse target projection/support work across the batch, and response chunk sizes can be capped explicitly or left on a conservative CUDA-aware auto mode.

To tune chunk sizes on a specific config, run:

```bash
conda run -n helios python -m train.benchmark_chunk_sweep --config configs/evo.yaml
```

By default the script benchmarks a geometric sweep around the current configured chunk size. You can also provide explicit values with `--chunk-sizes 5e5 1e6 2e6 4e6`, or sweep only one knob with `--sweep linear` or `--sweep wide`.

For Nsight Systems profiling, run the benchmark through the wrapper:

```bash
conda run -n helios python -m train.profile_with_nsys \
  --output-prefix data/profiling/evo_linear \
  -- \
  python -m train.benchmark_chunk_sweep --config configs/evo.yaml --sweep linear --chunk-sizes 5e5
```

This keeps `data/profiling/evo_linear.nsys-rep`, the exported SQLite file, and writes `data/profiling/evo_linear_nsys_summary.json`. The chunk sweep and evaluation paths now emit NVTX ranges so `nsys stats --filter-nvtx ...` can isolate full runs, per-chunk candidates, per-iteration evaluations, and the major evaluation stages.

To compare tiling heuristics directly, run:

```bash
conda run -n helios python -m train.benchmark_chunk_shapes \
  --config configs/evo.yaml \
  --sweep linear \
  --chunk-sizes 1e6 5e6 1e7 \
  --strategies balanced cap_reduction grid_first \
  --reduction-tile-cap 256
```

The profiling history and current chunking conclusions are tracked in `docs/performance/array_response_tiling.md`.

## Example Config

```yaml
experiment:
  name: evo_yaml_demo
  logDir: runs
  archiveDir: data/archive
  resume: true
  plotProjection: false
  tensorboard: true
  targetMode: auto

device:
  device: cpu
  dtype: float32

array:
  allowedElementCount: [64]
  allowedAspectRatio: [1.0]
  latitudeRange: [0.0, 0.0]
  longitudeRange: [-83.0, -83.0]
  altitudeRange: [36000000.0, 36000000.0]

evolution:
  batchSize: 50
  evolutionSteps: 1000
  initialWeightsType: directed
  cloneFraction: 0.15
  crossoverFraction: 0.15
  mutateFraction: 0.45
  randomFraction: 0.25
  parentPoolFraction: 0.25
  phaseSigma: 0.2
  amplitudeSigma: 0.01
  phaseSigmaDecay: 0.999
  amplitudeSigmaDecay: 0.999
  randomFractionDecay: 0.999
  phaseMinSigma: 0.0005
  amplitudeMinSigma: 0.00005
  minRandomFraction: 0.05
  stagnationWindow: 200
  sigmaBoostDuration: 20
  sigmaBoostMultiplier: 2.0
  sigmaBoostRandomFraction: 0.15
  phaseAdaptiveSigmaFloor: true
  amplitudeAdaptiveSigmaFloor: true
  phaseMinSigmaScale: 0.1
  amplitudeMinSigmaScale: 0.1
  wideGridSizeStart: 32
  wideGridRampSteps: 200

loss:
  w_shape: 1.0
  w_eff: 0.1
  w_wide: 0.5
  wide_grid_size: 64
  wide_support_dilation_cells: 1

logging:
  logMode: metrics_only
  datasetFlushEverySteps: 10
  responseCompactSize: 64

checkpoint:
  checkpointMode: periodic
  checkpointEverySteps: 100

workers:
  asyncIO: true
  datasetWriterWorkers: 1
  checkpointWriterWorkers: 1
  ioQueueSize: 2

target:
  reference: directed.pt
  decimate: 1
```

## Evolution Settings

- `cloneFraction`, `crossoverFraction`, `mutateFraction`, and `randomFraction` must sum to `1.0`.
- `crossoverFraction` enables uniform complex crossover: each child inherits each element weight from one of two ranked parents, then amplitudes are renormalized.
- `parentPoolFraction` controls how much of the best-ranked population is eligible to reproduce.
- `randomFractionDecay` decays random injection over time, down to `minRandomFraction`. The released fraction is reallocated to mutation automatically.
- `phaseSigma` and `amplitudeSigma` are the starting mutation scales. Their decay terms define the baseline schedule.
- `phaseMinSigma` and `amplitudeMinSigma` are hard floors.
- `phaseAdaptiveSigmaFloor` and `amplitudeAdaptiveSigmaFloor` toggle the diversity-aware sigma floors on and off.
- `phaseMinSigmaScale` and `amplitudeMinSigmaScale` control the strength of those adaptive floors when enabled.
- `stagnationWindow`, `sigmaBoostDuration`, `sigmaBoostMultiplier`, and `sigmaBoostRandomFraction` control stagnation-triggered exploration boosts.
- `wideGridSizeStart` and `wideGridRampSteps` optionally ramp wide-angle fidelity from a coarse grid to the final `loss.wide_grid_size`.
- `linearResponseChunkSize` and `wideResponseChunkSize` optionally cap evaluation chunk sizes. When left unset, Helios chooses conservative CUDA-aware defaults.
- `responseChunkShapeStrategy` selects the chunk-shape heuristic used inside the response kernel tiling. Supported values are `balanced`, `cap_reduction`, and `grid_first`.
- `responseReductionTileCap` limits the `Nc` reduction tile used by the reduction-capped chunk-shape strategies.

## Target Config Options

Exactly one target source must be set.

### Shared target from a `.pt`

```yaml
target:
  reference: directed.pt
  decimate: 1
```

### Shared target from a UI zones JSON

```yaml
target:
  reference: directed.json
  format: zones_json
  resolutionDeg: 0.5
  decimate: 1
```

### Shared inline target

```yaml
target:
  inline:
    searchLatitudes: [[10.0, 10.5], [11.0, 11.5]]
    searchLongitudes: [[20.0, 20.5], [21.0, 21.5]]
    importanceMap: [[1.0, 0.8], [0.4, 0.2]]
    powerMap: [[0.0, -3.0], [-6.0, -9.0]]
    hotspotCoordinates: [[10.5, 20.5]]
    thresholdDB: 10
  decimate: 1
```

### Per-sample target batch

```yaml
target:
  references:
    - target_a.pt
    - target_b.pt
    - target_c.pt
  decimate: 1
```

If `references` or `inlineBatch` is used, the resolved target becomes a `TargetBatch`, and its length must match `evolution.batchSize`.
If `decimate` is greater than `1`, the resolved target is spatially downsampled before training. This is useful when you want to keep a high-quality source target on disk but run faster searches against a coarser grid.

## Logging Modes

- `off`: disable TensorBoard metrics and dataset logging.
- `metrics_only`: write TensorBoard metrics only.
- `dataset_compact`: write metrics plus per-sample dataset shards using compact `64x64` response payloads.
- `dataset_full`: write metrics plus full-resolution per-sample dataset shards.

Dataset shards are written to:

```text
data/archive/<experiment_name>/dataset/
```

Each shard contains:

- per-sample array state
- response payload
- loss payload with `total`, `shape`, `efficiency`, and `wideSupport`
- shared target once per shard for shared-target runs
- per-sample target payloads for `TargetBatch` runs

## Checkpoints And Outputs

For a run named `evo_yaml_demo`:

- run config copy: `runs/evo_yaml_demo/config.yaml`
- TensorBoard logs: `runs/evo_yaml_demo/`
- latest checkpoint: `data/archive/evo_yaml_demo/checkpoints/latest.pt`
- best sample: `data/archive/evo_yaml_demo/best.pt`
- final summary: `data/archive/evo_yaml_demo/final.pt`

If you resume an experiment and `runs/<experiment>/config.yaml` already exists, the trainer validates that the active config matches it before continuing.
When `resume: true`, it allows updates to scheduler-oriented evolution fields such as `phaseSigma`, `amplitudeSigma`, their decays and floors, stagnation/boost knobs, fidelity-ramp knobs, and `initialWeightsType`, while still rejecting structural changes like batch fractions.

## Notebook / Programmatic Usage

You can load the same YAML in a notebook and construct the controller directly:

```python
import torch

from train.config import loadRunConfig, resolveDevice, resolveTarget, runConfigToDict
from train.evolve import EvolutionController

runConfig = loadRunConfig("path/to/run.yaml")
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

controller.train(
    dtype=dtype,
    device=device,
    logDir=runConfig.experiment.logDir,
    plotProjection=runConfig.experiment.plotProjection,
    resume=runConfig.experiment.resume,
)
```

## Benchmarking

Benchmark the active evaluation path with and without the shared-target fast path:

```bash
conda run -n helios python -m train.benchmark_evaluate --config configs/evo.yaml
```

## Notes

- `targetMode: auto` resolves by target type:
  - `TargetSpec` -> shared-target path
  - `TargetBatch` -> per-sample-target path
- Shared-target evaluation avoids unnecessary batch expansion of shared grids.
- Exact cloned elites are reused without rescoring when the fidelity schedule is unchanged for the next generation.
- `projectResponseOnTarget(...)` now evaluates only the requested sample instead of the whole batch.
