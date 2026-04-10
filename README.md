# Helios

**Phased array beamforming research framework**

Helios is a research tool for phased array beamforming. It includes a PyTorch simulation backend, an evolutionary training pipeline for optimizing array excitations against spatial targets, and a local web UI for visualization and configuration.

## Features

- **Phased Array Simulation**: Fast, PyTorch-accelerated calculations of complex beamforming patterns and coordinate transformations.
- **Evolutionary Optimization**: Optimize element excitations with directed initialization, crossover, adaptive mutation schedules, and stagnation-aware exploration boosts.
- **Interactive UI**: A local web interface (with a 3D globe) for editing target focuses, setting constraints, and visualizing the resulting beam patterns.

## Directory Structure

- `simulation/`: Core phased-array response kernels (`arraySim.py`).
- `scripts/`: Supporting geometry, batch generation, plotting, and target utilities.
- `train/`: Machine learning pipeline to optimize array configurations (`config.py`, `evolve.py`, `objective.py`).
- `ui/`: Web interface and backend server for interactive configuration and visualization.
- `data/` & `runs/` (ignored): Scratch space for tensorboard logs, saved models, and simulation datasets.

## Getting Started

### Prerequisites

You need `conda` (or `miniconda`) installed to easily manage dependencies.

### Installation

1. Create the conda environment from the provided configuration:
   ```bash
   conda env create -f environment.yml
   ```
2. Activate the environment:
   ```bash
   conda activate helios
   ```

### Running the Application

To start the interactive web application, run the UI server:

```bash
python ui/server.py
```
This will start a Flask server. Open your browser to the URL provided in the terminal (usually `http://127.0.0.1:8000`) to access the interface.

## YAML Training Runs

The training stack now supports typed YAML configs and a CLI entrypoint:

```bash
conda run -n helios python -m train.evolve --config path/to/run.yaml
```

For the full training/YAML guide, target config options, clone/crossover/mutate/random scheduling, adaptive sigma controls, logging modes, notebook usage, and output layout, see `train/README.md`.

## Formatting & Linting

This project uses `ruff` for code formatting and linting.
To check the code:
```bash
ruff check .
```
To format the code:
```bash
ruff format .
```
