# Generation

Offline dataset-generation utilities for Helios live here so they stay separate from the online training runtime.

## What This Contains

- `generate_targets.py`: CLI entrypoint for building a procedural target corpus
- `target_corpus.py`: config parsing, curriculum sampling, category synthesis, manifest writing
- `inspect_target_corpus.ipynb`: notebook for previewing generated maps and summarizing corpus statistics

The generator reuses the shared rasterization code in [scripts/target_generation.py](/home/aryan/projects/Helios/scripts/target_generation.py), so generated targets and UI-authored targets resolve to the same `TargetSpec` structure.

## Quick Start

```bash
python -m generation.generate_targets --config configs/target_corpus.yaml
```

This writes:

- one `TargetSpec` `.pt` file per generated sample
- one YAML manifest describing the corpus

For debugging and dataset characterization, open [inspect_target_corpus.ipynb](/home/aryan/projects/Helios/generation/inspect_target_corpus.ipynb) after generating a corpus.

## Config Notes

Key config sections in [configs/target_corpus.yaml](/home/aryan/projects/Helios/configs/target_corpus.yaml):

- `workers`: process count for corpus generation, or `auto` to use most CPU cores
- `output`: output root, manifest path, file prefix
- `grid`: lat/lon window, resolution, decimation, and `powerMode`
- `curriculum.weights`: class mix for `single_circle`, `irregular_shape`, `multibeam`
- `categories.*`: per-category sampling ranges

`grid.powerMode` defaults to `normalized`. Set it to `db` if you want generated power maps stored in dB instead.
Relative output paths in the generator config are resolved from the project root.

## Parallelism

The generator is CPU-bound: it samples geometry, rasterizes maps with NumPy, builds `TargetSpec` objects, and writes `.pt` files. It does not use GPU acceleration.

You can parallelize generation across CPU cores with either:

```yaml
workers: auto
```

or:

```bash
python -m generation.generate_targets --config configs/target_corpus.yaml --workers 8
```

Parallel runs remain deterministic because each sample uses a seed derived from the global corpus seed and its sample index.

Generation now shows a `tqdm` status bar in both terminals and notebooks when `tqdm` is available. If you call the API directly, you can control it with:

```python
from generation.target_corpus import generateTargetCorpus

manifest = generateTargetCorpus(
    "configs/target_corpus.yaml",
    workers="auto",
    showProgress=True,
    progressDescription="Building corpus",
)
```

## Training Integration

Training consumes generated corpora through the manifest path in [train/config.py](/home/aryan/projects/Helios/train/config.py):

```yaml
target:
  manifest: data/targets/sample_corpus/manifest.yaml
  selection: random_without_replacement
  selectionSeed: 7
```

At startup, Helios selects a batch from the manifest and resolves it into a `TargetBatch`.
