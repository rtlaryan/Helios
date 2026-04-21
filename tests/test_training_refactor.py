from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from scripts.arrayBatch import ArrayBatch
from scripts.arraySpec import ArraySpec
from scripts.batchFactory import (
    generateBatch,
    sampleDirectedWeights,
    sampleElementPositions,
    sampleRandomWeights,
    sampleUniformWeights,
)
from scripts.coordinateTransforms import LLAtoECEF
from scripts.targetSpec import TargetBatch, TargetSpec, inferTargetCenter
from train.config import loadRunConfig, resolveTarget, runConfigToDict
import train.evolve as evolve_module
from train.evolve import (
    CheckpointConfig,
    EvolutionConfig,
    EvolutionController,
    LoggingConfig,
    WorkerConfig,
)
from train.objective import (
    BatchEvaluation,
    LossConfig,
    _getWideGrid,
    _powerEfficiencyLoss,
    _prepareTarget,
    _rasterizeWideSupportMask,
    _resolveTargetMode,
    _shapeFidelityLoss,
    _wideAreaResponse,
    _wideSupportLossFromResponse,
    evaluateBatch,
)


def make_target() -> TargetSpec:
    latitudes = torch.tensor([[10.0, 10.5], [11.0, 11.5]], dtype=torch.float32)
    longitudes = torch.tensor([[20.0, 20.5], [21.0, 21.5]], dtype=torch.float32)
    importance = torch.tensor([[1.0, 0.8], [0.4, 0.2]], dtype=torch.float32)
    power = torch.tensor([[0.0, -3.0], [-6.0, -9.0]], dtype=torch.float32)
    hotspots = torch.tensor([[10.5, 20.5]], dtype=torch.float32)
    return TargetSpec(
        searchLatitudes=latitudes,
        searchLongitudes=longitudes,
        importanceMap=importance,
        powerMap=power,
        hotspotCoordinates=hotspots,
    )


def make_target_inline_payload() -> dict:
    target = make_target().serializeTargetSpec()
    return {
        key: value.tolist() if isinstance(value, torch.Tensor) else value
        for key, value in target.items()
    }


def make_array_spec() -> ArraySpec:
    return ArraySpec(
        allowedElementCount=(4,),
        allowedAspectRatio=(1.0,),
        latitudeRange=(0.0, 0.0),
        longitudeRange=(-83.0, -83.0),
        altitudeRange=(3.6e7, 3.6e7),
    )


def rows_are_identical(tensor: torch.Tensor) -> bool:
    return bool(torch.equal(tensor, tensor[:1].expand_as(tensor)))


def test_load_run_config_with_inline_target_accepts_deprecated_loss_keys(tmp_path: Path) -> None:
    payload = {
        "experiment": {"name": "yaml_inline"},
        "device": {"device": "cpu", "dtype": "float32"},
        "array": {"allowedElementCount": [4], "allowedAspectRatio": [1.0]},
        "evolution": {"batchSize": 2, "evolutionSteps": 1},
        "loss": {
            "wide_grid_size": 8,
            "wide_support_dilation_cells": 2,
            "psl_local_mix": 0.25,
        },
        "logging": {"logMode": "metrics_only"},
        "checkpoint": {"checkpointMode": "off"},
        "workers": {"asyncIO": False, "datasetWriterWorkers": 0, "checkpointWriterWorkers": 0},
        "target": {"inline": make_target_inline_payload()},
    }
    configPath = tmp_path / "run.yaml"
    with configPath.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    runConfig = loadRunConfig(configPath)
    resolvedTarget = resolveTarget(runConfig)
    serializedLoss = runConfigToDict(runConfig)["loss"]

    assert runConfig.sourcePath == configPath
    assert isinstance(resolvedTarget, TargetSpec)
    assert runConfig.loss.wide_support_dilation_cells == 2
    assert "psl_local_mix" not in serializedLoss


def test_load_run_config_accepts_directed_initial_weights(tmp_path: Path) -> None:
    payload = {
        "experiment": {"name": "yaml_directed"},
        "device": {"device": "cpu", "dtype": "float32"},
        "array": {"allowedElementCount": [4], "allowedAspectRatio": [1.0]},
        "evolution": {"batchSize": 2, "evolutionSteps": 1, "initialWeightsType": "directed"},
        "logging": {"logMode": "metrics_only"},
        "checkpoint": {"checkpointMode": "off"},
        "workers": {"asyncIO": False, "datasetWriterWorkers": 0, "checkpointWriterWorkers": 0},
        "target": {"inline": make_target_inline_payload()},
    }
    configPath = tmp_path / "run.yaml"
    with configPath.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    runConfig = loadRunConfig(configPath)

    assert runConfig.evolution.initialWeightsType == "directed"


def test_load_run_config_accepts_crossover_and_scheduler_fields(tmp_path: Path) -> None:
    payload = {
        "experiment": {"name": "yaml_scheduler"},
        "device": {"device": "cpu", "dtype": "float32"},
        "array": {"allowedElementCount": [4], "allowedAspectRatio": [1.0]},
        "evolution": {
            "batchSize": 4,
            "evolutionSteps": 3,
            "cloneFraction": 0.25,
            "crossoverFraction": 0.25,
            "mutateFraction": 0.35,
            "randomFraction": 0.15,
            "stagnationWindow": 5,
            "sigmaBoostDuration": 3,
            "sigmaBoostMultiplier": 1.5,
            "sigmaBoostRandomFraction": 0.2,
            "phaseAdaptiveSigmaFloor": False,
            "amplitudeAdaptiveSigmaFloor": True,
            "phaseMinSigmaScale": 0.15,
            "amplitudeMinSigmaScale": 0.05,
            "wideGridSizeStart": 8,
            "wideGridRampSteps": 10,
            "linearResponseChunkSize": 256,
            "wideResponseChunkSize": 128,
        },
        "logging": {"logMode": "metrics_only"},
        "checkpoint": {"checkpointMode": "off"},
        "workers": {"asyncIO": False, "datasetWriterWorkers": 0, "checkpointWriterWorkers": 0},
        "target": {"inline": make_target_inline_payload()},
    }
    configPath = tmp_path / "run.yaml"
    with configPath.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    runConfig = loadRunConfig(configPath)
    serialized = runConfigToDict(runConfig)

    assert runConfig.evolution.crossoverFraction == pytest.approx(0.25)
    assert runConfig.evolution.stagnationWindow == 5
    assert runConfig.evolution.sigmaBoostMultiplier == pytest.approx(1.5)
    assert runConfig.evolution.phaseAdaptiveSigmaFloor is False
    assert runConfig.evolution.amplitudeAdaptiveSigmaFloor is True
    assert runConfig.evolution.wideGridSizeStart == 8
    assert runConfig.evolution.linearResponseChunkSize == 256
    assert runConfig.evolution.wideResponseChunkSize == 128
    assert serialized["evolution"]["wideGridRampSteps"] == 10
    assert "responseChunkShapeStrategy" not in serialized["evolution"]


def test_load_run_config_rejects_removed_chunk_shape_strategy(tmp_path: Path) -> None:
    payload = {
        "experiment": {"name": "yaml_removed_strategy"},
        "device": {"device": "cpu", "dtype": "float32"},
        "array": {"allowedElementCount": [4], "allowedAspectRatio": [1.0]},
        "evolution": {
            "batchSize": 4,
            "evolutionSteps": 3,
            "responseChunkShapeStrategy": "cap_reduction",
        },
        "logging": {"logMode": "metrics_only"},
        "checkpoint": {"checkpointMode": "off"},
        "workers": {"asyncIO": False, "datasetWriterWorkers": 0, "checkpointWriterWorkers": 0},
        "target": {"inline": make_target_inline_payload()},
    }
    configPath = tmp_path / "run.yaml"
    with configPath.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    with pytest.raises(ValueError, match="responseChunkShapeStrategy has been removed"):
        loadRunConfig(configPath)


def test_infer_target_center_prefers_weighted_support_centroid() -> None:
    target = TargetSpec(
        searchLatitudes=torch.tensor([[0.0, 0.0], [10.0, 10.0]], dtype=torch.float32),
        searchLongitudes=torch.tensor([[0.0, 10.0], [0.0, 10.0]], dtype=torch.float32),
        importanceMap=torch.tensor([[0.0, 0.0], [1.0, 3.0]], dtype=torch.float32),
        powerMap=torch.zeros((2, 2), dtype=torch.float32),
        hotspotCoordinates=torch.tensor([[999.0, 999.0]], dtype=torch.float32),
    )

    center = inferTargetCenter(target)

    torch.testing.assert_close(center, torch.tensor([10.0, 7.5], dtype=torch.float32))


def test_infer_target_center_supports_target_batch() -> None:
    targetA = TargetSpec(
        searchLatitudes=torch.tensor([[0.0, 0.0], [10.0, 10.0]], dtype=torch.float32),
        searchLongitudes=torch.tensor([[0.0, 10.0], [0.0, 10.0]], dtype=torch.float32),
        importanceMap=torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32),
        powerMap=torch.zeros((2, 2), dtype=torch.float32),
        hotspotCoordinates=torch.tensor([[999.0, 999.0]], dtype=torch.float32),
    )
    targetB = TargetSpec(
        searchLatitudes=torch.tensor([[20.0, 20.0], [30.0, 30.0]], dtype=torch.float32),
        searchLongitudes=torch.tensor([[40.0, 50.0], [40.0, 50.0]], dtype=torch.float32),
        importanceMap=torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        powerMap=torch.zeros((2, 2), dtype=torch.float32),
        hotspotCoordinates=torch.tensor([[999.0, 999.0]], dtype=torch.float32),
    )

    centers = inferTargetCenter(TargetBatch.fromTargetSpecs([targetA, targetB]))

    torch.testing.assert_close(
        centers,
        torch.tensor([[0.0, 0.0], [30.0, 50.0]], dtype=torch.float32),
    )


def test_init_evolution_directed_uses_inferred_target_center(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = TargetSpec(
        searchLatitudes=torch.tensor([[0.0, 0.0], [10.0, 10.0]], dtype=torch.float32),
        searchLongitudes=torch.tensor([[0.0, 10.0], [0.0, 10.0]], dtype=torch.float32),
        importanceMap=torch.tensor([[0.0, 0.0], [1.0, 3.0]], dtype=torch.float32),
        powerMap=torch.zeros((2, 2), dtype=torch.float32),
        hotspotCoordinates=torch.tensor([[999.0, 999.0]], dtype=torch.float32),
    )
    controller = EvolutionController(
        config=EvolutionConfig(batchSize=2, initialWeightsType="directed"),
        targetSpec=target,
        arraySpec=make_array_spec(),
        lossParams=LossConfig(),
        loggingConfig=LoggingConfig(logMode="off"),
        checkpointConfig=CheckpointConfig(checkpointMode="off"),
        workerConfig=WorkerConfig(asyncIO=False, datasetWriterWorkers=0, checkpointWriterWorkers=0),
    )
    captured: dict[str, torch.Tensor | str | int] = {}

    def fake_generate_batch(*args, **kwargs):
        captured["weightsType"] = kwargs["weightsType"]
        captured["targetLLA"] = kwargs["targetLLA"].clone()
        return generateBatch(*args, **kwargs)

    monkeypatch.setattr("train.evolve.generateBatch", fake_generate_batch)

    controller.initEvolution(dtype=torch.float32, device=torch.device("cpu"))

    assert captured["weightsType"] == "directed"
    torch.testing.assert_close(
        captured["targetLLA"],
        torch.tensor([10.0, 7.5], dtype=torch.float32),
    )


def test_init_evolution_shared_target_keeps_a_homogeneous_template() -> None:
    controller = EvolutionController(
        config=EvolutionConfig(
            batchSize=3,
            evolutionSteps=1,
            initialWeightsType="random",
            generator=torch.Generator().manual_seed(0),
        ),
        targetSpec=make_target(),
        arraySpec=ArraySpec(
            allowedElementCount=(4, 9),
            allowedAspectRatio=(1.0, 0.5),
            positionJitterSTD=0.1,
            gainRange=(1.0, 2.0),
            latitudeRange=(0.0, 1.0),
            longitudeRange=(-83.0, -82.0),
            altitudeRange=(3.6e7, 3.7e7),
        ),
        lossParams=LossConfig(wide_grid_size=8),
    )

    batch = controller.initEvolution(dtype=torch.float32, device=torch.device("cpu"))

    assert controller.targetMode == "shared"
    assert rows_are_identical(batch.elementLocalPosition)
    assert rows_are_identical(batch.LLAPosition)
    assert rows_are_identical(batch.ECEFPosition)
    assert rows_are_identical(batch.gain)
    assert not rows_are_identical(batch.weights)


def test_rank_weighted_parent_sampling_prefers_better_ranks() -> None:
    controller = EvolutionController(
        config=EvolutionConfig(batchSize=4, parentPoolFraction=1.0),
        targetSpec=make_target(),
        arraySpec=make_array_spec(),
        lossParams=LossConfig(wide_grid_size=8),
    )
    parentIDs = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    generator = torch.Generator().manual_seed(0)

    sampled = controller.sample(
        parentIDs,
        childCount=4000,
        device=torch.device("cpu"),
        generator=generator,
    )
    counts = torch.bincount(sampled, minlength=4)

    assert counts[0] > counts[1] > counts[2] > counts[3]


def test_crossover_weights_mixes_parent_phases_and_preserves_norm() -> None:
    batch = generateBatch(
        make_array_spec(),
        batchSize=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        weightsType="uniform",
    )
    amplitude = torch.full((2, batch.N), 1.0 / (batch.N**0.5), dtype=torch.float32)
    phaseA = torch.zeros((2, batch.N), dtype=torch.float32)
    phaseB = torch.full((2, batch.N), torch.pi / 2, dtype=torch.float32)
    parentA = ArrayBatch(
        elementLocalPosition=batch.elementLocalPosition.clone(),
        weights=torch.polar(amplitude, phaseA),
        wavelength=batch.wavelength,
        gain=batch.gain.clone(),
        LLAPosition=batch.LLAPosition.clone(),
        ECEFPosition=batch.ECEFPosition.clone(),
        elementMask=None if batch.elementMask is None else batch.elementMask.clone(),
    )
    parentB = ArrayBatch(
        elementLocalPosition=batch.elementLocalPosition.clone(),
        weights=torch.polar(amplitude, phaseB),
        wavelength=batch.wavelength,
        gain=batch.gain.clone(),
        LLAPosition=batch.LLAPosition.clone(),
        ECEFPosition=batch.ECEFPosition.clone(),
        elementMask=None if batch.elementMask is None else batch.elementMask.clone(),
    )

    children = parentA.crossoverWeights(parentB, generator=torch.Generator().manual_seed(0))
    phases = children.weights.angle()

    torch.testing.assert_close(children.weights.abs().norm(dim=1), torch.ones(2))
    assert torch.any(torch.isclose(phases, torch.zeros_like(phases)))
    assert torch.any(torch.isclose(phases, torch.full_like(phases, torch.pi / 2)))


def test_generate_batch_respects_generator_for_variable_search_space() -> None:
    spec = ArraySpec(
        allowedElementCount=(4, 9),
        allowedAspectRatio=(1.0, 2.0),
        latitudeRange=(0.0, 1.0),
        longitudeRange=(-83.5, -82.5),
        altitudeRange=(3.5e7, 3.7e7),
    )
    generatorA = torch.Generator().manual_seed(1234)
    generatorB = torch.Generator().manual_seed(1234)

    batchA = generateBatch(
        spec,
        batchSize=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        weightsType="random",
        generator=generatorA,
    )
    batchB = generateBatch(
        spec,
        batchSize=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        weightsType="random",
        generator=generatorB,
    )

    assert batchA.N == batchB.N
    torch.testing.assert_close(batchA.elementLocalPosition, batchB.elementLocalPosition)
    torch.testing.assert_close(batchA.weights, batchB.weights)
    torch.testing.assert_close(batchA.LLAPosition, batchB.LLAPosition)
    torch.testing.assert_close(batchA.ECEFPosition, batchB.ECEFPosition)


@pytest.mark.parametrize("weights_type", ["random", "uniform", "directed"])
def test_masked_initializers_produce_unit_norm_active_weights(weights_type: str) -> None:
    spec = make_array_spec()
    batchSize = 2
    elementCount = 4
    device = torch.device("cpu")
    dtype = torch.float32
    elementMask = torch.tensor(
        [[True, False, True, False], [False, True, True, False]],
        dtype=torch.bool,
    )
    generator = torch.Generator().manual_seed(0)

    if weights_type == "random":
        weights = sampleRandomWeights(
            spec,
            batchSize,
            elementCount,
            device,
            dtype,
            elementMask,
            generator=generator,
        )
    elif weights_type == "uniform":
        weights = sampleUniformWeights(
            spec,
            batchSize,
            elementCount,
            device,
            dtype,
            elementMask,
            generator=generator,
        )
    else:
        localPositions = sampleElementPositions(
            spec,
            batchSize,
            elementCount,
            1.0,
            device,
            dtype,
            generator=generator,
        )
        arrayLLA = torch.tensor(
            [[0.0, -83.0, 3.6e7], [0.0, -83.0, 3.6e7]],
            dtype=dtype,
        )
        weights = sampleDirectedWeights(
            spec,
            batchSize,
            localPositions,
            device,
            dtype,
            targetLLA=torch.tensor([10.0, 20.0], dtype=dtype),
            arrayLLA=arrayLLA,
            elementMask=elementMask,
            generator=generator,
        )

    torch.testing.assert_close(weights.abs().norm(dim=1), torch.ones(batchSize))
    assert torch.allclose(
        weights.abs()[~elementMask],
        torch.zeros_like(weights.abs()[~elementMask]),
    )


def test_evaluate_batch_supports_target_batch() -> None:
    target = make_target()
    targetBatch = TargetBatch.fromTargetSpecs([target.clone(), target.clone()])
    arraySpec = make_array_spec()
    batch = generateBatch(
        arraySpec,
        batchSize=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        weightsType="random",
        targetLLA=target.hotspotCoordinates[0],
    )
    params = LossConfig(wide_grid_size=8)

    sharedEval = evaluateBatch(batch, target, params, targetMode="shared")
    batchedEval = evaluateBatch(batch, targetBatch, params, targetMode="per_sample")

    torch.testing.assert_close(sharedEval.totalLoss, batchedEval.totalLoss)
    assert sharedEval.linearResponse.shape == batchedEval.linearResponse.shape == (2, 2, 2)


def test_shared_target_fast_path_matches_standard_path() -> None:
    target = make_target()
    batch = generateBatch(
        make_array_spec(),
        batchSize=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        weightsType="random",
        targetLLA=target.hotspotCoordinates[0],
    )
    params = LossConfig(wide_grid_size=8)

    standardEval = evaluateBatch(
        batch,
        target,
        params,
        targetMode="shared",
        allowSharedTargetFastPath=False,
    )
    fastEval = evaluateBatch(
        batch,
        target,
        params,
        targetMode="shared",
        allowSharedTargetFastPath=True,
    )

    torch.testing.assert_close(fastEval.totalLoss, standardEval.totalLoss)
    torch.testing.assert_close(fastEval.shapeLoss, standardEval.shapeLoss)
    torch.testing.assert_close(fastEval.efficiencyLoss, standardEval.efficiencyLoss)
    torch.testing.assert_close(fastEval.wideSupportLoss, standardEval.wideSupportLoss)
    torch.testing.assert_close(fastEval.linearResponse, standardEval.linearResponse)


def test_shared_target_fast_path_skips_heterogeneous_batches() -> None:
    target = make_target()
    batch = generateBatch(
        make_array_spec(),
        batchSize=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        weightsType="random",
        targetLLA=target.hotspotCoordinates[0],
    )
    variedLLA = batch.LLAPosition.clone()
    variedLLA[1, 0] += 1.0
    heterogeneousBatch = ArrayBatch(
        elementLocalPosition=batch.elementLocalPosition.clone(),
        weights=batch.weights.clone(),
        wavelength=batch.wavelength,
        gain=batch.gain.clone(),
        LLAPosition=variedLLA,
        ECEFPosition=LLAtoECEF(variedLLA),
        elementMask=None if batch.elementMask is None else batch.elementMask.clone(),
    )

    standardEvaluation = evaluateBatch(
        heterogeneousBatch,
        target,
        LossConfig(wide_grid_size=8),
        targetMode="shared",
        allowSharedTargetFastPath=False,
    )
    fastEvaluation = evaluateBatch(
        heterogeneousBatch,
        target,
        LossConfig(wide_grid_size=8),
        targetMode="shared",
        allowSharedTargetFastPath=True,
    )

    torch.testing.assert_close(fastEvaluation.totalLoss, standardEvaluation.totalLoss)
    torch.testing.assert_close(fastEvaluation.linearResponse, standardEvaluation.linearResponse)


def test_explicit_chunk_sizes_preserve_evaluation_results() -> None:
    target = make_target()
    batch = generateBatch(
        make_array_spec(),
        batchSize=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        weightsType="random",
        targetLLA=target.hotspotCoordinates[0],
    )
    params = LossConfig(wide_grid_size=8)

    baseline = evaluateBatch(
        batch,
        target,
        params,
        targetMode="shared",
        linearResponseChunkSize=512,
        wideResponseChunkSize=512,
        allowSharedTargetFastPath=False,
    )
    chunked = evaluateBatch(
        batch,
        target,
        params,
        targetMode="shared",
        linearResponseChunkSize=4,
        wideResponseChunkSize=4,
        allowSharedTargetFastPath=False,
    )

    torch.testing.assert_close(chunked.totalLoss, baseline.totalLoss)
    torch.testing.assert_close(chunked.linearResponse, baseline.linearResponse)
    torch.testing.assert_close(chunked.wideSupportLoss, baseline.wideSupportLoss)


def test_float_like_chunk_sizes_from_config_remain_usable() -> None:
    target = make_target()
    batch = generateBatch(
        make_array_spec(),
        batchSize=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        weightsType="random",
        targetLLA=target.hotspotCoordinates[0],
    )

    evaluation = evaluateBatch(
        batch,
        target,
        LossConfig(wide_grid_size=8),
        targetMode="shared",
        linearResponseChunkSize=300.0,
        wideResponseChunkSize=120.0,
        allowSharedTargetFastPath=False,
    )

    assert evaluation.totalLoss.shape == (batch.batchSize,)


def test_shape_loss_uses_global_peak_referenced_response() -> None:
    target = make_target()
    arraySpec = make_array_spec()
    batch = generateBatch(
        arraySpec,
        batchSize=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        weightsType="random",
        targetLLA=target.hotspotCoordinates[0],
    )
    scaledBatch = ArrayBatch(
        elementLocalPosition=batch.elementLocalPosition.clone(),
        weights=batch.weights * 2.0,
        wavelength=batch.wavelength,
        gain=batch.gain.clone(),
        LLAPosition=batch.LLAPosition.clone(),
        ECEFPosition=batch.ECEFPosition.clone(),
        elementMask=None if batch.elementMask is None else batch.elementMask.clone(),
    )
    params = LossConfig(wide_grid_size=8)

    baseEval = evaluateBatch(batch, target, params, targetMode="shared")
    scaledEval = evaluateBatch(scaledBatch, target, params, targetMode="shared")

    torch.testing.assert_close(scaledEval.shapeLoss, baseEval.shapeLoss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(scaledEval.linearResponse, baseEval.linearResponse * 4.0)


def test_shape_loss_ignores_zero_importance_regions() -> None:
    params = LossConfig()
    target = make_target()
    target.importanceMap = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    target = target.to(torch.device("cpu"), torch.float32)
    prep = _prepareTarget(target, params, _resolveTargetMode(target, "shared"))

    baseResponse = torch.tensor([[[0.2, 0.1], [0.05, 0.02]]], dtype=torch.float32)
    ignoredOnlyChanged = torch.tensor([[[0.2, 0.9], [0.8, 0.7]]], dtype=torch.float32)

    baseLoss = _shapeFidelityLoss(baseResponse, prep)
    ignoredLoss = _shapeFidelityLoss(ignoredOnlyChanged, prep)

    torch.testing.assert_close(baseLoss, ignoredLoss)


def test_projected_support_rasterization_marks_only_positive_importance() -> None:
    params = LossConfig()
    target = make_target()
    target.importanceMap = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    target = target.to(torch.device("cpu"), torch.float32)
    prep = _prepareTarget(target, params, _resolveTargetMode(target, "shared"))
    wideAZGrid, wideELGrid = _getWideGrid(torch.device("cpu"), torch.float32, 5)

    targetAZEL = (
        torch.tensor(
            [[wideAZGrid[1, 0], wideAZGrid[2, 0], wideAZGrid[0, 0], wideAZGrid[3, 0]]],
            dtype=torch.float32,
        ),
        torch.tensor(
            [[wideELGrid[0, 1], wideELGrid[0, 2], wideELGrid[0, 3], wideELGrid[0, 4]]],
            dtype=torch.float32,
        ),
    )

    supportMask = _rasterizeWideSupportMask(
        targetAZEL, prep, wideAZGrid, wideELGrid, dilationCells=0
    )

    assert bool(supportMask[0, 1, 1])
    assert bool(supportMask[0, 3, 4])
    assert not bool(supportMask[0, 2, 2])
    assert not bool(supportMask[0, 0, 3])
    assert int(supportMask.sum().item()) == 2


def test_projected_support_dilation_fills_expected_neighbors() -> None:
    params = LossConfig()
    target = make_target()
    target.importanceMap = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    target = target.to(torch.device("cpu"), torch.float32)
    prep = _prepareTarget(target, params, _resolveTargetMode(target, "shared"))
    wideAZGrid, wideELGrid = _getWideGrid(torch.device("cpu"), torch.float32, 5)

    targetAZEL = (
        torch.tensor(
            [[wideAZGrid[2, 0], wideAZGrid[0, 0], wideAZGrid[0, 0], wideAZGrid[0, 0]]],
            dtype=torch.float32,
        ),
        torch.tensor(
            [[wideELGrid[0, 2], wideELGrid[0, 0], wideELGrid[0, 0], wideELGrid[0, 0]]],
            dtype=torch.float32,
        ),
    )

    supportMask = _rasterizeWideSupportMask(
        targetAZEL, prep, wideAZGrid, wideELGrid, dilationCells=0
    )
    dilatedMask = _rasterizeWideSupportMask(
        targetAZEL, prep, wideAZGrid, wideELGrid, dilationCells=1
    )

    assert int(supportMask.sum().item()) == 1
    assert int(dilatedMask.sum().item()) == 9
    assert bool(dilatedMask[0, 1, 1])
    assert bool(dilatedMask[0, 2, 2])
    assert bool(dilatedMask[0, 3, 3])


def test_wide_support_loss_matches_reused_wide_response_and_efficiency() -> None:
    target = make_target()
    arraySpec = make_array_spec()
    batch = generateBatch(
        arraySpec,
        batchSize=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        weightsType="random",
        targetLLA=target.hotspotCoordinates[0],
    )
    params = LossConfig(wide_grid_size=8)

    evaluation = evaluateBatch(batch, target, params, targetMode="shared")
    wideResponse, wideAZGrid, wideELGrid = _wideAreaResponse(batch, params.wide_grid_size)
    resolvedTarget = target.clone().to(batch.device, batch.dtype)
    prep = _prepareTarget(
        resolvedTarget,
        params,
        _resolveTargetMode(resolvedTarget, "shared"),
    )
    supportMask = _rasterizeWideSupportMask(
        evaluation.targetAZEL,
        prep,
        wideAZGrid,
        wideELGrid,
        params.wide_support_dilation_cells,
    )
    globalPeak = wideResponse.flatten(1).amax(dim=-1).clamp_min(1e-10)
    manualWide = _wideSupportLossFromResponse(wideResponse, globalPeak, supportMask)
    manualEfficiency = _powerEfficiencyLoss(evaluation.linearResponse, prep)

    torch.testing.assert_close(evaluation.wideSupportLoss, manualWide)
    torch.testing.assert_close(evaluation.efficiencyLoss, manualEfficiency)


def test_wide_support_loss_zero_inside_support_and_increases_outside() -> None:
    supportMask = torch.tensor(
        [[[False, False, False], [False, True, False], [False, False, False]]],
        dtype=torch.bool,
    )
    baselineResponse = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    spikyResponse = baselineResponse.clone()
    spikyResponse[0, 0, 0] = 0.08

    baselinePeak = baselineResponse.flatten(1).amax(dim=-1)
    spikyPeak = spikyResponse.flatten(1).amax(dim=-1)
    baselineLoss = _wideSupportLossFromResponse(baselineResponse, baselinePeak, supportMask)
    spikyLoss = _wideSupportLossFromResponse(spikyResponse, spikyPeak, supportMask)

    assert baselineLoss.item() == pytest.approx(0.0)
    assert spikyLoss.item() > baselineLoss.item()


def test_train_writes_yaml_dataset_and_checkpoint(tmp_path: Path) -> None:
    target = make_target()
    sourceConfig = tmp_path / "source.yaml"
    with sourceConfig.open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"experiment": {"name": "smoke"}}, handle)

    controller = EvolutionController(
        config=EvolutionConfig(batchSize=2, evolutionSteps=1),
        targetSpec=target,
        arraySpec=make_array_spec(),
        lossParams=LossConfig(wide_grid_size=8),
        experimentName="smoke",
        archiveRoot=tmp_path / "archive",
        loggingConfig=LoggingConfig(logMode="dataset_compact", datasetFlushEverySteps=1),
        checkpointConfig=CheckpointConfig(checkpointMode="periodic", checkpointEverySteps=1),
        workerConfig=WorkerConfig(
            asyncIO=False,
            datasetWriterWorkers=0,
            checkpointWriterWorkers=0,
            ioQueueSize=1,
        ),
        sourceConfigPath=sourceConfig,
        resolvedConfig={"experiment": {"name": "smoke"}},
        writerLogDir=tmp_path / "runs",
    )

    controller.train(
        dtype=torch.float32,
        device=torch.device("cpu"),
        logDir=tmp_path / "runs",
        plotProjection=False,
        resume=False,
    )

    datasetPath = tmp_path / "archive" / "smoke" / "dataset" / "shard_00000.pt"
    assert (tmp_path / "runs" / "smoke" / "config.yaml").exists()
    assert datasetPath.exists()
    assert (tmp_path / "archive" / "smoke" / "checkpoints" / "latest.pt").exists()
    assert (tmp_path / "archive" / "smoke" / "best.pt").exists()
    assert (tmp_path / "archive" / "smoke" / "final.pt").exists()

    dataset = torch.load(datasetPath, weights_only=False)
    record = dataset["records"][0]
    assert set(record["loss"]) == {"total", "shape", "efficiency", "wideSupport"}


def test_train_starts_progress_bar_before_initial_population_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller = EvolutionController(
        config=EvolutionConfig(batchSize=2, evolutionSteps=1),
        targetSpec=make_target(),
        arraySpec=make_array_spec(),
        lossParams=LossConfig(wide_grid_size=8),
        experimentName="progress_smoke",
        archiveRoot=tmp_path / "archive",
        loggingConfig=LoggingConfig(logMode="off"),
        checkpointConfig=CheckpointConfig(checkpointMode="off"),
        workerConfig=WorkerConfig(
            asyncIO=False,
            datasetWriterWorkers=0,
            checkpointWriterWorkers=0,
            ioQueueSize=1,
        ),
        writerLogDir=tmp_path / "runs",
    )

    batch = generateBatch(
        controller.arraySpec,
        batchSize=controller.config.batchSize,
        device=torch.device("cpu"),
        dtype=torch.float32,
        weightsType=controller.config.initialWeightsType,
    )
    targetPointCount = make_target().importanceMap.numel()
    events: list[tuple] = []

    class FakeProgressBar:
        def __init__(self, *args, total: int, initial: int, desc: str, **kwargs) -> None:
            events.append(("init", total, initial, desc, kwargs.get("dynamic_ncols")))

        def set_postfix(self, payload: dict[str, str]) -> None:
            events.append(("set_postfix", payload))

        def set_postfix_str(self, value: str) -> None:
            events.append(("set_postfix_str", value))

        def refresh(self) -> None:
            events.append(("refresh",))

        def update(self, n: int = 1) -> None:
            events.append(("update", n))

        def close(self) -> None:
            events.append(("close",))

    def fake_population_for_step(active_batch: ArrayBatch, step: int):
        events.append(
            (
                "population",
                step,
                any(event[0] == "init" for event in events),
                any(event[0] == "refresh" for event in events),
            )
        )
        zeros = torch.zeros(active_batch.batchSize, dtype=torch.float32)
        evaluation = BatchEvaluation(
            totalLoss=torch.tensor([1.0, 2.0], dtype=torch.float32),
            shapeLoss=zeros,
            efficiencyLoss=zeros,
            wideSupportLoss=zeros,
            linearResponse=torch.zeros((active_batch.batchSize, 2, 2), dtype=torch.float32),
            targetAZEL=(
                torch.zeros((active_batch.batchSize, targetPointCount), dtype=torch.float32),
                torch.zeros((active_batch.batchSize, targetPointCount), dtype=torch.float32),
            ),
            targetMode=controller.targetMode,
        )
        return evolve_module.PopulationState(
            batch=active_batch,
            evaluation=evaluation,
            lossParams=controller.lossParams,
        )

    monkeypatch.setattr(evolve_module, "tqdm", FakeProgressBar)
    monkeypatch.setattr(controller, "initEvolution", lambda **_: batch)
    monkeypatch.setattr(controller, "_populationForStep", fake_population_for_step)

    controller.train(
        dtype=torch.float32,
        device=torch.device("cpu"),
        logDir=tmp_path / "runs",
        plotProjection=False,
        resume=False,
    )

    populationEvent = next(event for event in events if event[0] == "population")
    assert populationEvent[2] is True
    assert populationEvent[3] is True
    assert ("set_postfix_str", "evaluating") in events
    assert ("update", 1) in events


def test_resume_rejects_mismatched_config(tmp_path: Path) -> None:
    runDir = tmp_path / "runs" / "exp"
    runDir.mkdir(parents=True, exist_ok=True)
    with (runDir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"experiment": {"name": "exp", "resume": True}}, handle)

    controller = EvolutionController(
        config=EvolutionConfig(batchSize=2, evolutionSteps=1),
        targetSpec=make_target(),
        arraySpec=make_array_spec(),
        lossParams=LossConfig(wide_grid_size=8),
        experimentName="exp",
        archiveRoot=tmp_path / "archive",
        loggingConfig=LoggingConfig(logMode="off"),
        checkpointConfig=CheckpointConfig(checkpointMode="off"),
        workerConfig=WorkerConfig(asyncIO=False, datasetWriterWorkers=0, checkpointWriterWorkers=0),
        resolvedConfig={"experiment": {"name": "different"}},
        writerLogDir=tmp_path / "runs",
    )

    with pytest.raises(ValueError):
        controller._writeRunConfig(runDir)


def test_init_evolution_uses_configured_initial_weights_type() -> None:
    controller = EvolutionController(
        config=EvolutionConfig(
            batchSize=2,
            evolutionSteps=1,
            initialWeightsType="uniform",
        ),
        targetSpec=make_target(),
        arraySpec=make_array_spec(),
        lossParams=LossConfig(wide_grid_size=8),
    )

    batch = controller.initEvolution(dtype=torch.float32, device=torch.device("cpu"))
    expectedAmplitude = torch.full_like(batch.weights.abs(), 1.0 / (batch.N**0.5))

    torch.testing.assert_close(batch.weights.abs(), expectedAmplitude)
    torch.testing.assert_close(
        batch.weights.angle(),
        torch.zeros_like(batch.weights.angle()),
        atol=1e-6,
        rtol=0.0,
    )


def test_shared_target_random_injection_reuses_population_template() -> None:
    controller = EvolutionController(
        config=EvolutionConfig(
            batchSize=3,
            evolutionSteps=2,
            cloneFraction=0.0,
            crossoverFraction=0.0,
            mutateFraction=0.0,
            randomFraction=1.0,
            parentPoolFraction=1.0,
            generator=torch.Generator().manual_seed(0),
        ),
        targetSpec=make_target(),
        arraySpec=ArraySpec(
            allowedElementCount=(4, 9),
            allowedAspectRatio=(1.0, 0.5),
            positionJitterSTD=0.1,
            gainRange=(1.0, 2.0),
            latitudeRange=(0.0, 1.0),
            longitudeRange=(-83.0, -82.0),
            altitudeRange=(3.6e7, 3.7e7),
        ),
        lossParams=LossConfig(wide_grid_size=8),
    )
    batch = controller.initEvolution(dtype=torch.float32, device=torch.device("cpu"))
    population = controller._populationForStep(batch, 0)

    nextPopulation = controller.evolutionStep(0, population, controller._initialSchedulerState())

    template = batch.fetch(0)
    torch.testing.assert_close(
        nextPopulation.batch.elementLocalPosition,
        template.elementLocalPosition.expand_as(nextPopulation.batch.elementLocalPosition),
    )
    torch.testing.assert_close(
        nextPopulation.batch.LLAPosition,
        template.LLAPosition.expand_as(nextPopulation.batch.LLAPosition),
    )
    torch.testing.assert_close(
        nextPopulation.batch.ECEFPosition,
        template.ECEFPosition.expand_as(nextPopulation.batch.ECEFPosition),
    )
    torch.testing.assert_close(
        nextPopulation.batch.gain,
        template.gain.expand_as(nextPopulation.batch.gain),
    )


def test_resume_allows_sigma_bump_and_updates_saved_run_config(tmp_path: Path) -> None:
    runDir = tmp_path / "runs" / "exp"
    runDir.mkdir(parents=True, exist_ok=True)
    existingConfig = {
        "experiment": {"name": "exp", "resume": True},
        "evolution": {
            "phaseSigma": 0.05,
            "amplitudeSigma": 0.0,
            "phaseSigmaDecay": 0.998,
            "amplitudeSigmaDecay": 1.0,
            "phaseMinSigma": 1e-4,
            "amplitudeMinSigma": 0.0,
            "initialWeightsType": "uniform",
        },
    }
    with (runDir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(existingConfig, handle, sort_keys=False)

    controller = EvolutionController(
        config=EvolutionConfig(batchSize=2, evolutionSteps=1),
        targetSpec=make_target(),
        arraySpec=make_array_spec(),
        lossParams=LossConfig(wide_grid_size=8),
        experimentName="exp",
        archiveRoot=tmp_path / "archive",
        loggingConfig=LoggingConfig(logMode="off"),
        checkpointConfig=CheckpointConfig(checkpointMode="off"),
        workerConfig=WorkerConfig(asyncIO=False, datasetWriterWorkers=0, checkpointWriterWorkers=0),
        resolvedConfig={
            "experiment": {"name": "exp", "resume": True},
            "evolution": {
                "phaseSigma": 0.25,
                "amplitudeSigma": 0.02,
                "phaseSigmaDecay": 0.999,
                "amplitudeSigmaDecay": 0.995,
                "phaseMinSigma": 1e-4,
                "amplitudeMinSigma": 0.0,
                "initialWeightsType": "random",
            },
        },
        writerLogDir=tmp_path / "runs",
    )

    controller._writeRunConfig(runDir, resume=True)

    with (runDir / "config.yaml").open("r", encoding="utf-8") as handle:
        updated = yaml.safe_load(handle)

    assert updated["evolution"]["phaseSigma"] == pytest.approx(0.25)
    assert updated["evolution"]["amplitudeSigma"] == pytest.approx(0.02)
    assert updated["evolution"]["initialWeightsType"] == "random"
    assert updated["experiment"]["resume"] is True


def test_scheduler_uses_adaptive_floor_and_stagnation_boosts() -> None:
    controller = EvolutionController(
        config=EvolutionConfig(
            batchSize=4,
            evolutionSteps=10,
            cloneFraction=0.25,
            crossoverFraction=0.25,
            mutateFraction=0.5,
            randomFraction=0.0,
            phaseSigma=0.05,
            phaseSigmaDecay=0.1,
            phaseMinSigma=1e-4,
            phaseMinSigmaScale=0.5,
            stagnationWindow=2,
            sigmaBoostDuration=3,
            sigmaBoostMultiplier=2.0,
            sigmaBoostRandomFraction=0.2,
        ),
        targetSpec=make_target(),
        arraySpec=make_array_spec(),
        lossParams=LossConfig(wide_grid_size=8),
    )
    batch = generateBatch(
        make_array_spec(),
        batchSize=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        weightsType="uniform",
    )
    phase = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
        ],
        dtype=torch.float32,
    )
    parentBatch = ArrayBatch(
        elementLocalPosition=batch.elementLocalPosition.clone(),
        weights=torch.polar(batch.weights.abs(), phase),
        wavelength=batch.wavelength,
        gain=batch.gain.clone(),
        LLAPosition=batch.LLAPosition.clone(),
        ECEFPosition=batch.ECEFPosition.clone(),
        elementMask=None if batch.elementMask is None else batch.elementMask.clone(),
    )

    state = controller._initialSchedulerState()
    state = controller._advanceScheduler(0, state, parentBatch, improved=False)
    assert state.currentPhaseSigma > controller.config.phaseMinSigma

    state = controller._advanceScheduler(1, state, parentBatch, improved=False)
    assert state.boostStepsRemaining == controller.config.sigmaBoostDuration
    assert state.currentRandomFraction == pytest.approx(controller.config.sigmaBoostRandomFraction)


def test_scheduler_can_disable_adaptive_phase_floor() -> None:
    controller = EvolutionController(
        config=EvolutionConfig(
            batchSize=4,
            evolutionSteps=10,
            cloneFraction=0.25,
            crossoverFraction=0.25,
            mutateFraction=0.5,
            randomFraction=0.0,
            phaseSigma=0.05,
            phaseSigmaDecay=0.1,
            phaseMinSigma=1e-4,
            phaseAdaptiveSigmaFloor=False,
            phaseMinSigmaScale=0.5,
        ),
        targetSpec=make_target(),
        arraySpec=make_array_spec(),
        lossParams=LossConfig(wide_grid_size=8),
    )
    batch = generateBatch(
        make_array_spec(),
        batchSize=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        weightsType="uniform",
    )
    phase = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
        ],
        dtype=torch.float32,
    )
    parentBatch = ArrayBatch(
        elementLocalPosition=batch.elementLocalPosition.clone(),
        weights=torch.polar(batch.weights.abs(), phase),
        wavelength=batch.wavelength,
        gain=batch.gain.clone(),
        LLAPosition=batch.LLAPosition.clone(),
        ECEFPosition=batch.ECEFPosition.clone(),
        elementMask=None if batch.elementMask is None else batch.elementMask.clone(),
    )

    state = controller._initialSchedulerState()
    state = controller._advanceScheduler(0, state, parentBatch, improved=False)
    expectedPhaseSigma, _ = controller.config.baseSigmaAt(1)
    assert state.currentPhaseSigma == pytest.approx(expectedPhaseSigma)


def test_wide_grid_schedule_ramps_to_final_size() -> None:
    controller = EvolutionController(
        config=EvolutionConfig(
            batchSize=2,
            evolutionSteps=20,
            wideGridSizeStart=8,
            wideGridRampSteps=4,
        ),
        targetSpec=make_target(),
        arraySpec=make_array_spec(),
        lossParams=LossConfig(wide_grid_size=16),
    )

    assert controller._wideGridSizeAt(0) == 8
    assert controller._wideGridSizeAt(2) == 12
    assert controller._wideGridSizeAt(4) == 16
    assert controller._wideGridSizeAt(10) == 16


def test_evolution_step_reuses_clone_evaluations_when_fidelity_is_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = EvolutionController(
        config=EvolutionConfig(
            batchSize=2,
            evolutionSteps=3,
            cloneFraction=0.5,
            crossoverFraction=0.0,
            mutateFraction=0.5,
            randomFraction=0.0,
            parentPoolFraction=0.5,
        ),
        targetSpec=make_target(),
        arraySpec=make_array_spec(),
        lossParams=LossConfig(wide_grid_size=8),
    )
    batch = generateBatch(
        make_array_spec(),
        batchSize=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        weightsType="random",
        targetLLA=make_target().hotspotCoordinates[0],
    )
    population = controller._populationForStep(batch, 0)
    schedulerState = controller._initialSchedulerState()
    calls: list[int] = []
    originalEvaluate = controller.evaluate

    def spy_evaluate(batch: ArrayBatch, lossParams: LossConfig | None = None) -> BatchEvaluation:
        calls.append(batch.batchSize)
        return originalEvaluate(batch, lossParams)

    monkeypatch.setattr(controller, "evaluate", spy_evaluate)

    nextPopulation = controller.evolutionStep(0, population, schedulerState)

    assert calls == [1]
    assert nextPopulation.batch.batchSize == 2


def test_evolution_step_reevaluates_full_batch_when_fidelity_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = EvolutionController(
        config=EvolutionConfig(
            batchSize=2,
            evolutionSteps=3,
            cloneFraction=0.5,
            crossoverFraction=0.0,
            mutateFraction=0.5,
            randomFraction=0.0,
            parentPoolFraction=0.5,
            wideGridSizeStart=4,
            wideGridRampSteps=1,
        ),
        targetSpec=make_target(),
        arraySpec=make_array_spec(),
        lossParams=LossConfig(wide_grid_size=8),
    )
    batch = generateBatch(
        make_array_spec(),
        batchSize=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        weightsType="random",
        targetLLA=make_target().hotspotCoordinates[0],
    )
    population = controller._populationForStep(batch, 0)
    schedulerState = controller._initialSchedulerState()
    calls: list[int] = []
    originalEvaluate = controller.evaluate

    def spy_evaluate(batch: ArrayBatch, lossParams: LossConfig | None = None) -> BatchEvaluation:
        calls.append(batch.batchSize)
        return originalEvaluate(batch, lossParams)

    monkeypatch.setattr(controller, "evaluate", spy_evaluate)

    controller.evolutionStep(0, population, schedulerState)

    assert calls == [2]


def test_resume_allows_runtime_toggle_changes(tmp_path: Path) -> None:
    runDir = tmp_path / "runs" / "exp"
    runDir.mkdir(parents=True, exist_ok=True)
    with (runDir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "experiment": {
                    "name": "exp",
                    "resume": False,
                    "plotProjection": True,
                },
                "evolution": {
                    "phaseSigma": 0.02,
                },
            },
            handle,
            sort_keys=False,
        )

    controller = EvolutionController(
        config=EvolutionConfig(batchSize=2, evolutionSteps=1),
        targetSpec=make_target(),
        arraySpec=make_array_spec(),
        lossParams=LossConfig(wide_grid_size=8),
        experimentName="exp",
        archiveRoot=tmp_path / "archive",
        loggingConfig=LoggingConfig(logMode="off"),
        checkpointConfig=CheckpointConfig(checkpointMode="off"),
        workerConfig=WorkerConfig(asyncIO=False, datasetWriterWorkers=0, checkpointWriterWorkers=0),
        resolvedConfig={
            "experiment": {
                "name": "exp",
                "resume": True,
                "plotProjection": False,
            },
            "evolution": {
                "phaseSigma": 0.05,
                "phaseAdaptiveSigmaFloor": False,
            },
        },
        writerLogDir=tmp_path / "runs",
    )

    controller._writeRunConfig(runDir, resume=True)

    with (runDir / "config.yaml").open("r", encoding="utf-8") as handle:
        updated = yaml.safe_load(handle)

    assert updated["experiment"]["resume"] is True
    assert updated["experiment"]["plotProjection"] is False
    assert updated["evolution"]["phaseSigma"] == pytest.approx(0.05)
    assert updated["evolution"]["phaseAdaptiveSigmaFloor"] is False


def test_resume_still_rejects_structural_evolution_change(tmp_path: Path) -> None:
    runDir = tmp_path / "runs" / "exp"
    runDir.mkdir(parents=True, exist_ok=True)
    with (runDir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "experiment": {"name": "exp", "resume": True},
                "evolution": {"cloneFraction": 0.2, "phaseSigma": 0.05},
            },
            handle,
            sort_keys=False,
        )

    controller = EvolutionController(
        config=EvolutionConfig(batchSize=2, evolutionSteps=1),
        targetSpec=make_target(),
        arraySpec=make_array_spec(),
        lossParams=LossConfig(wide_grid_size=8),
        experimentName="exp",
        archiveRoot=tmp_path / "archive",
        loggingConfig=LoggingConfig(logMode="off"),
        checkpointConfig=CheckpointConfig(checkpointMode="off"),
        workerConfig=WorkerConfig(asyncIO=False, datasetWriterWorkers=0, checkpointWriterWorkers=0),
        resolvedConfig={
            "experiment": {"name": "exp", "resume": True},
            "evolution": {"cloneFraction": 0.3, "phaseSigma": 0.25},
        },
        writerLogDir=tmp_path / "runs",
    )

    with pytest.raises(ValueError):
        controller._writeRunConfig(runDir, resume=True)
