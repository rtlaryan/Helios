from __future__ import annotations

import argparse

from train.config import loadRunConfig, resolveDevice, resolveTarget, runConfigToDict
from train.evolve import EvolutionController


def buildControllerFromConfig(configPath: str) -> tuple[EvolutionController, tuple]:
    runConfig = loadRunConfig(configPath)
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
    return controller, (device, dtype, runConfig.experiment)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Helios evolution from YAML config")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    controller, (device, dtype, experiment) = buildControllerFromConfig(args.config)
    controller.train(
        dtype=dtype,
        device=device,
        logDir=experiment.logDir,
        plotProjection=experiment.plotProjection,
        resume=experiment.resume,
    )


if __name__ == "__main__":
    main()
