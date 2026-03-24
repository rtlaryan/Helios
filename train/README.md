# Helios Training Pipeline

This directory contains the machine learning pipelines for optimizing phased array element configurations. It uses deep reinforcement learning (RL) and evolutionary algorithms to generate near-optimal beamforming patterns based on arbitrary targets.

## Core Modules

- `evolve.py`: The entry point and main logic for the evolutionary / reinforcement learning loop. It orchestrates the generation of batches, evaluation of the objective functions, and the update of neural network weights to improve phased array performance.
- `objective.py`: Defines the objective (loss/reward) functions used to evaluate the fitness of a specific array configuration. It factors in mainlobe alignment, sidelobe suppression, and region-specific rolloff constraints.

To start a training run, see the usage of `evolve.py`, which supports logging metrics locally or directly to TensorBoard. Ensure the Helios conda environment is activated before running the scripts.
