from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml
from scripts.arrayBatch import ArrayBatch
from scripts.arrayBatch import merge
from scripts.batchFactory import generateBatch
from train.config import buildPPORunConfig, buildRunConfig, ppoRunConfigToDict, runConfigToDict
from train.evaluation_utils import evaluation_diagnostic_means, evaluation_weighted_loss_means
from train.evolve import EvolutionConfig, EvolutionController, LoggingConfig, CheckpointConfig, WorkerConfig
from train.objective_v2 import (
    LossConfigV2,
    _COARSE_SHARED_CACHE_V2,
    _FINE_SHARED_CACHE_V2,
    _prepareTargetV2,
    clearCachesV2,
    evaluateBatchV2,
)


def make_target(power_map: torch.Tensor | None = None) -> Any:
    from scripts.targetSpec import TargetSpec

    latitudes = torch.tensor([[10.0, 10.5], [11.0, 11.5]], dtype=torch.float32)
    longitudes = torch.tensor([[20.0, 20.5], [21.0, 21.5]], dtype=torch.float32)
    importance = torch.tensor([[1.0, 0.8], [0.4, 0.2]], dtype=torch.float32)
    power = (
        torch.tensor([[1.0, 0.5], [0.25, 0.125]], dtype=torch.float32)
        if power_map is None
        else power_map
    )
    hotspots = torch.tensor([[10.5, 20.5]], dtype=torch.float32)
    return TargetSpec(
        searchLatitudes=latitudes,
        searchLongitudes=longitudes,
        importanceMap=importance,
        powerMap=power,
        hotspotCoordinates=hotspots,
        thresholdDB=6.0,
    )


def make_array_spec() -> Any:
    from scripts.arraySpec import ArraySpec

    return ArraySpec(
        allowedElementCount=(4,),
        allowedAspectRatio=(1.0,),
        latitudeRange=(0.0, 0.0),
        longitudeRange=(-83.0, -83.0),
        altitudeRange=(3.6e7, 3.6e7),
    )


def make_run_payload() -> dict:
    return {
        "experiment": {"name": "obj_v2"},
        "device": {"device": "cpu", "dtype": "float32"},
        "array": {
            "allowedElementCount": [4],
            "allowedAspectRatio": [1.0],
            "latitudeRange": [0.0, 0.0],
            "longitudeRange": [-83.0, -83.0],
            "altitudeRange": [3.6e7, 3.6e7],
        },
        "evolution": {"batchSize": 2, "evolutionSteps": 1},
        "logging": {"logMode": "off"},
        "checkpoint": {"checkpointMode": "off"},
        "workers": {"asyncIO": False, "datasetWriterWorkers": 0, "checkpointWriterWorkers": 0},
        "target": {"inline": make_target().serializeTargetSpec()},
    }


def test_prepare_target_v2_normalizes_linear_and_db_inputs_equally() -> None:
    linear_target = make_target()
    db_target = make_target(power_map=10.0 * torch.log10(linear_target.powerMap))

    linear_prep = _prepareTargetV2(linear_target, LossConfigV2(), targetMode="shared")
    db_prep = _prepareTargetV2(db_target, LossConfigV2(), targetMode="shared")

    torch.testing.assert_close(linear_prep.targetNormFlat, db_prep.targetNormFlat)
    assert torch.equal(linear_prep.serviceMaskFlat, db_prep.serviceMaskFlat)


def test_prepare_target_v2_rejects_empty_power_support() -> None:
    target = make_target(power_map=torch.zeros((2, 2), dtype=torch.float32))

    with pytest.raises(ValueError, match="positive support"):
        _prepareTargetV2(target, LossConfigV2(), targetMode="shared")


def test_evaluate_batch_v2_uses_global_peak_reference() -> None:
    target = make_target()
    array_spec = make_array_spec()
    batch = generateBatch(
        array_spec,
        batchSize=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        weightsType="random",
        targetLLA=target.hotspotCoordinates[0],
    )
    scaled_batch = ArrayBatch(
        elementLocalPosition=batch.elementLocalPosition.clone(),
        weights=batch.weights * 2.0,
        wavelength=batch.wavelength,
        gain=batch.gain.clone(),
        LLAPosition=batch.LLAPosition.clone(),
        ECEFPosition=batch.ECEFPosition.clone(),
        elementMask=None if batch.elementMask is None else batch.elementMask.clone(),
    )

    base_eval = evaluateBatchV2(batch, target, LossConfigV2(), targetMode="shared")
    scaled_eval = evaluateBatchV2(scaled_batch, target, LossConfigV2(), targetMode="shared")

    torch.testing.assert_close(base_eval.shapeLoss, scaled_eval.shapeLoss, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(base_eval.efficiencyLoss, scaled_eval.efficiencyLoss, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(base_eval.nullPenaltyLoss, scaled_eval.nullPenaltyLoss, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(base_eval.shellLoss, scaled_eval.shellLoss, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(base_eval.globalLoss, scaled_eval.globalLoss, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(base_eval.peakLoss, scaled_eval.peakLoss, atol=1e-5, rtol=1e-4)


def test_v2_weighted_loss_means_and_diagnostics_are_exposed() -> None:
    target = make_target()
    batch = generateBatch(
        make_array_spec(),
        batchSize=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        weightsType="random",
        targetLLA=target.hotspotCoordinates[0],
    )
    params = LossConfigV2(
        w_shape=1.5,
        w_efficiency=2.0,
        w_null_penalty=2.5,
        w_shell=0.75,
        w_global=3.0,
        w_peak=4.0,
    )
    evaluation = evaluateBatchV2(batch, target, params, targetMode="shared")

    weighted = evaluation_weighted_loss_means(evaluation, params)
    diagnostics = evaluation_diagnostic_means(evaluation)

    assert weighted["shape"] == pytest.approx(float((evaluation.shapeLoss.mean() * params.w_shape).item()))
    assert weighted["efficiency"] == pytest.approx(
        float((evaluation.efficiencyLoss.mean() * params.w_efficiency).item())
    )
    assert weighted["nullPenalty"] == pytest.approx(
        float((evaluation.nullPenaltyLoss.mean() * params.w_null_penalty).item())
    )
    assert "serviceMeanNorm" in diagnostics
    assert "serviceCoverageFrac" in diagnostics
    assert "serviceEnergyFraction" in diagnostics
    assert "finePeakNorm" in diagnostics
    assert 0.0 <= diagnostics["serviceCoverageFrac"] <= 1.0
    assert 0.0 <= diagnostics["serviceEnergyFraction"] <= 1.000001


def test_v2_objective_simulation_backend_v2_fails_fast_on_cpu() -> None:
    target = make_target()
    batch = generateBatch(
        make_array_spec(),
        batchSize=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        weightsType="random",
        targetLLA=target.hotspotCoordinates[0],
    )

    with pytest.raises(RuntimeError, match="requires a CUDA ArrayBatch"):
        evaluateBatchV2(
            batch,
            target,
            LossConfigV2(min_front_grid_size=4, max_front_grid_size=4),
            targetMode="shared",
            simulationBackend="v2",
        )


def test_build_run_config_v2_requires_loss_v2_and_serializes_only_active_block() -> None:
    payload = make_run_payload()
    payload["objectiveVersion"] = "v2"
    payload["lossV2"] = {"peak_topk": 4}

    config = buildRunConfig(payload)
    serialized = runConfigToDict(config)

    assert config.objectiveVersion == "v2"
    assert config.loss is None
    assert config.lossV2 is not None
    assert "loss" not in serialized
    assert serialized["lossV2"]["peak_topk"] == 4


def test_build_run_config_v2_rejects_legacy_loss_block() -> None:
    payload = make_run_payload()
    payload["objectiveVersion"] = "v2"
    payload["loss"] = {"wide_grid_size": 8}

    with pytest.raises(ValueError, match="requires lossV2"):
        buildRunConfig(payload)


def test_build_ppo_run_config_v2_serializes_only_loss_v2(tmp_path: Path) -> None:
    model_path = tmp_path / "model.yaml"
    model_path.write_text(
        yaml.safe_dump(
            {
                "architecture": "cnn_mlp",
                "encoder": {"type": "cnn", "cnn": {"convChannels": [8], "kernelSizes": [3], "strides": [1], "paddings": [1]}},
                "decoder": {"type": "flat_action", "flat_action": {"mlpLayers": [8]}},
                "context": {"globalFeatures": [], "elementFeatures": []},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    target_path = tmp_path / "target.pt"
    torch.save(make_target(), target_path)
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump({"records": [{"path": str(target_path)}]}, sort_keys=False), encoding="utf-8")
    payload = {
        "experiment": {"name": "ppo_v2"},
        "device": {"device": "cpu", "dtype": "float32"},
        "array": {
            "allowedElementCount": [4],
            "allowedAspectRatio": [1.0],
            "latitudeRange": [0.0, 0.0],
            "longitudeRange": [-83.0, -83.0],
            "altitudeRange": [3.6e7, 3.6e7],
        },
        "objectiveVersion": "v2",
        "modelConfig": str(model_path),
        "ppo": {"rolloutBatchSize": 1, "updateSteps": 1, "ppoEpochs": 1, "minibatchSize": 1},
        "lossV2": {"peak_topk": 3},
        "logging": {"logMode": "off"},
        "checkpoint": {"checkpointMode": "off"},
        "workers": {"asyncIO": False, "datasetWriterWorkers": 0, "checkpointWriterWorkers": 0},
        "target": {"manifest": str(manifest_path), "loadingMode": "block_window", "blockSize": 1, "loaderWorkers": 1},
    }

    config = buildPPORunConfig(payload, basePath=tmp_path)
    serialized = ppoRunConfigToDict(config)

    assert config.objectiveVersion == "v2"
    assert config.lossV2 is not None
    assert "loss" not in serialized
    assert serialized["lossV2"]["peak_topk"] == 3


def test_compare_objectives_writes_benchmark_report(tmp_path: Path) -> None:
    payload = make_run_payload()
    payload["experiment"]["archiveDir"] = str(tmp_path / "archive")
    payload["experiment"]["name"] = "compare_smoke"
    payload["target"]["inline"] = {
        key: value.tolist() if isinstance(value, torch.Tensor) else value
        for key, value in make_target().serializeTargetSpec().items()
    }
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "train.compare_objectives", "--config", str(config_path), "--warmups", "1", "--repeats", "1"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )

    report_path = tmp_path / "archive" / "compare_smoke" / "benchmarks" / "objective_compare_shared.json"
    assert report_path.exists(), result.stderr
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["warm"]["v2_over_v1_ratio"] >= 0.0
    assert "spearman" in payload["ranking"]
    assert "cold" in payload and "warm" in payload


def test_compare_objectives_build_fixture_batch_uses_seeded_global_rng_without_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import train.compare_objectives as compare_module

    config = SimpleNamespace(
        array=make_array_spec(),
        evolution=SimpleNamespace(batchSize=2, initialWeightsType="random"),
    )
    generatorArgs: list[torch.Generator | None] = []

    def fake_generate_batch(**kwargs):
        generatorArgs.append(kwargs.get("generator"))
        return float(torch.rand(1).item())

    baselineState = torch.random.get_rng_state()
    monkeypatch.setattr(compare_module, "generateBatch", fake_generate_batch)

    first = compare_module._build_fixture_batch(
        make_target(),
        config,
        device=torch.device("cuda"),
        dtype=torch.float32,
    )
    afterFirstState = torch.random.get_rng_state()
    second = compare_module._build_fixture_batch(
        make_target(),
        config,
        device=torch.device("cuda"),
        dtype=torch.float32,
    )

    assert generatorArgs == [None, None]
    assert first == second
    assert torch.equal(baselineState, afterFirstState)
    assert torch.equal(baselineState, torch.random.get_rng_state())


def test_v2_evolution_step_does_not_call_wide_grid_schedule_for_clones(tmp_path: Path) -> None:
    target = make_target()
    array_spec = make_array_spec()
    batch = generateBatch(
        array_spec,
        batchSize=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        weightsType="random",
        targetLLA=target.hotspotCoordinates[0],
    )
    controller = EvolutionController(
        config=EvolutionConfig(
            batchSize=4,
            evolutionSteps=2,
            cloneFraction=0.5,
            crossoverFraction=0.0,
            mutateFraction=0.5,
            randomFraction=0.0,
            parentPoolFraction=0.5,
        ),
        targetSpec=target,
        arraySpec=array_spec,
        lossParams=LossConfigV2(),
        objectiveVersion="v2",
        experimentName="v2_clone_step",
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

    evaluation = controller.evaluate(batch)
    population = controller._populationForStep(batch, 0)
    scheduler = controller._initialSchedulerState()
    sorted_ids = torch.argsort(evaluation.totalLoss, dim=0, descending=False)

    next_population = controller.evolutionStep(0, population, scheduler, sortedScoresIDs=sorted_ids)

    assert next_population.batch.batchSize == batch.batchSize


def test_v2_shared_operator_caches_reuse_across_merged_batches(tmp_path: Path) -> None:
    target = make_target()
    array_spec = make_array_spec()
    controller = EvolutionController(
        config=EvolutionConfig(
            batchSize=4,
            evolutionSteps=2,
            cloneFraction=0.5,
            crossoverFraction=0.0,
            mutateFraction=0.5,
            randomFraction=0.0,
            parentPoolFraction=0.5,
        ),
        targetSpec=target,
        arraySpec=array_spec,
        lossParams=LossConfigV2(),
        objectiveVersion="v2",
        experimentName="v2_cache_reuse",
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

    clearCachesV2()
    batch = controller.initEvolution(dtype=torch.float32, device=torch.device("cpu"))
    assert batch.geometryCacheKey is not None

    evaluateBatchV2(batch, target, LossConfigV2(), targetMode="shared")
    fine_count = len(_FINE_SHARED_CACHE_V2)
    coarse_count = len(_COARSE_SHARED_CACHE_V2)

    merged_batch = merge([batch.fetch(torch.tensor([0, 1])), batch.fetch(torch.tensor([2, 3]))])
    assert merged_batch.geometryCacheKey == batch.geometryCacheKey

    evaluateBatchV2(merged_batch, target, LossConfigV2(), targetMode="shared")

    assert len(_FINE_SHARED_CACHE_V2) == fine_count
    assert len(_COARSE_SHARED_CACHE_V2) == coarse_count
