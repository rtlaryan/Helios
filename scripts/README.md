# Helios Scripts

This directory contains the core Python modules for defining and simulating phased arrays, generating batches, and plotting results. They form the numerical and physical simulation backend for the Helios framework.

## Core Modules

- `arrayBatch.py` & `batchFactory.py`: Handle the generation and management of batches of simulation states, including weight mutation/crossover helpers and directed initialization for the evolutionary trainer.
- `arraySimulation.py`: The main simulation engine for calculating phased array responses, beam patterns, and performance metrics (like mainlobe power and sidelobe levels).
- `arraySpec.py`: Defines the physical specifications of the phased arrays.
- `coordinateTransforms.py`: Utilities for converting between different coordinate systems (e.g., spherical, Cartesian, latitude/longitude) used in targeting and visualization.
- `plots.py`: Utility functions to generate plots for analysis (e.g., cross-sections, 3D visualizations, training metrics).
- `targetSpec.py`: Contains specifications and requirements for the focus regions (targets) and their constraints.
- `phasedArray_old.py`: Legacy implementation of the phased array logic (kept for reference or backwards compatibility).

These scripts are imported by both the training pipeline (`train/`) and the web UI bridge (`ui/`) to run simulations. The training stack in `train/` now relies on these modules for rank-weighted parent selection, uniform complex crossover, adaptive mutation, and wide-angle response evaluation.
