from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from generation.target_corpus import CATEGORY_NAMES, generateTargetCorpus
from scripts.arraySpec import ArraySpec
from scripts.targetSpec import TargetBatch, TargetSpec
from train.config import buildRunConfig, resolveTarget
from train.evolve import (
    CheckpointConfig,
    EvolutionConfig,
    EvolutionController,
    LoggingConfig,
    WorkerConfig,
)
from train.objective import LossConfig


def make_array_spec() -> ArraySpec:
    return ArraySpec(
        allowedElementCount=(4,),
        allowedAspectRatio=(1.0,),
        latitudeRange=(0.0, 0.0),
        longitudeRange=(-83.0, -83.0),
        altitudeRange=(3.6e7, 3.6e7),
    )


def write_corpus_config(
    path: Path,
    output_root: Path,
    *,
    count: int = 6,
    seed: int = 7,
    power_mode: str = "normalized",
) -> None:
    payload = {
        "count": count,
        "seed": seed,
        "output": {
            "root": str(output_root),
            "manifestPath": str(output_root / "manifest.yaml"),
            "prefix": "sample",
        },
        "grid": {
            "latRange": [0.0, 10.0],
            "lonRange": [20.0, 30.0],
            "resolutionDeg": 1.0,
            "decimate": 1,
            "powerMode": power_mode,
        },
        "curriculum": {
            "weights": {
                "single_circle": 1.0,
                "irregular_shape": 1.0,
                "multibeam": 1.0,
            }
        },
        "categories": {
            "single_circle": {
                "radiusRange": [1.0, 1.0],
                "rolloffRange": [0.5, 0.5],
                "peakDBRange": [0.0, 0.0],
            },
            "irregular_shape": {
                "componentCountRange": [2, 2],
                "circleComponentProbability": 0.5,
                "radiusRange": [1.0, 1.0],
                "polygonRadiusRange": [1.5, 1.5],
                "polygonVertexCountRange": [4, 4],
                "componentJitterRange": [1.0, 1.0],
                "rolloffRange": [0.5, 0.5],
                "peakDBRange": [0.0, 0.0],
            },
            "multibeam": {
                "beamCountRange": [2, 4],
                "radiusRange": [1.0, 1.0],
                "rolloffRange": [0.5, 0.5],
                "peakDBRange": [0.0, 0.0],
                "minSeparationDeg": 3.0,
                "placementAttempts": 64,
            },
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_target_from_record(manifest_path: Path, record: dict) -> TargetSpec:
    record_path = Path(record["path"])
    if not record_path.is_absolute():
        record_path = manifest_path.parent / record_path
    target = torch.load(record_path, weights_only=False)
    assert isinstance(target, TargetSpec)
    return target


def test_generate_target_corpus_is_deterministic_and_curriculum_driven(tmp_path: Path) -> None:
    config_path_a = tmp_path / "targets_a.yaml"
    config_path_b = tmp_path / "targets_b.yaml"
    output_root_a = tmp_path / "corpus_a"
    output_root_b = tmp_path / "corpus_b"
    write_corpus_config(config_path_a, output_root_a)
    write_corpus_config(config_path_b, output_root_b)

    manifest_a = generateTargetCorpus(config_path_a)
    manifest_b = generateTargetCorpus(config_path_b)

    assert manifest_a["curriculum"]["counts"] == {
        "single_circle": 2,
        "irregular_shape": 2,
        "multibeam": 2,
    }
    assert manifest_a["curriculum"]["counts"] == manifest_b["curriculum"]["counts"]
    assert [record["category"] for record in manifest_a["records"]] == [
        record["category"] for record in manifest_b["records"]
    ]

    manifest_path_a = output_root_a / "manifest.yaml"
    manifest_path_b = output_root_b / "manifest.yaml"
    for record_a, record_b in zip(manifest_a["records"], manifest_b["records"], strict=True):
        target_a = load_target_from_record(manifest_path_a, record_a)
        target_b = load_target_from_record(manifest_path_b, record_b)
        torch.testing.assert_close(target_a.powerMap, target_b.powerMap)
        torch.testing.assert_close(target_a.importanceMap, target_b.importanceMap)
        torch.testing.assert_close(target_a.hotspotCoordinates, target_b.hotspotCoordinates)


def test_generate_target_corpus_matches_single_worker_output_when_parallelized(
    tmp_path: Path,
) -> None:
    config_path_single = tmp_path / "targets_single.yaml"
    config_path_multi = tmp_path / "targets_multi.yaml"
    output_root_single = tmp_path / "corpus_single"
    output_root_multi = tmp_path / "corpus_multi"
    write_corpus_config(config_path_single, output_root_single, count=8)
    write_corpus_config(config_path_multi, output_root_multi, count=8)

    manifest_single = generateTargetCorpus(config_path_single, workers=1, showProgress=False)
    manifest_multi = generateTargetCorpus(config_path_multi, workers=2, showProgress=False)

    assert manifest_single["workers"] == 1
    assert manifest_multi["workers"] == 2
    assert [record["category"] for record in manifest_single["records"]] == [
        record["category"] for record in manifest_multi["records"]
    ]

    manifest_path_single = output_root_single / "manifest.yaml"
    manifest_path_multi = output_root_multi / "manifest.yaml"
    for record_single, record_multi in zip(
        manifest_single["records"], manifest_multi["records"], strict=True
    ):
        target_single = load_target_from_record(manifest_path_single, record_single)
        target_multi = load_target_from_record(manifest_path_multi, record_multi)
        torch.testing.assert_close(target_single.powerMap, target_multi.powerMap)
        torch.testing.assert_close(target_single.importanceMap, target_multi.importanceMap)
        torch.testing.assert_close(
            target_single.hotspotCoordinates, target_multi.hotspotCoordinates
        )


def test_generated_targets_are_valid_and_use_normalized_power_maps(tmp_path: Path) -> None:
    config_path = tmp_path / "targets.yaml"
    output_root = tmp_path / "corpus"
    write_corpus_config(config_path, output_root)

    manifest = generateTargetCorpus(config_path)
    manifest_path = output_root / "manifest.yaml"

    assert manifest["powerMode"] == "normalized"
    assert set(record["category"] for record in manifest["records"]) == set(CATEGORY_NAMES)

    for record in load_manifest(manifest_path)["records"]:
        target = load_target_from_record(manifest_path, record)
        assert target.hotspotCoordinates.shape == (4, 2)
        assert target.importanceMap.max().item() > 0.0
        assert target.powerMap.min().item() >= 0.0
        assert target.powerMap.max().item() <= 1.0 + 1e-6
        assert target.searchLatitudes.min().item() >= 0.0
        assert target.searchLatitudes.max().item() <= 10.0
        assert target.searchLongitudes.min().item() >= 20.0
        assert target.searchLongitudes.max().item() <= 30.0


def test_resolve_target_supports_manifest_selection_modes(tmp_path: Path) -> None:
    config_path = tmp_path / "targets.yaml"
    output_root = tmp_path / "corpus"
    write_corpus_config(config_path, output_root)
    generateTargetCorpus(config_path)
    manifest_path = output_root / "manifest.yaml"

    run_config = buildRunConfig(
        {
            "experiment": {"name": "manifest_first"},
            "device": {"device": "cpu", "dtype": "float32"},
            "array": {"allowedElementCount": [4], "allowedAspectRatio": [1.0]},
            "evolution": {"batchSize": 3, "evolutionSteps": 1},
            "logging": {"logMode": "off"},
            "checkpoint": {"checkpointMode": "off"},
            "workers": {"asyncIO": False, "datasetWriterWorkers": 0, "checkpointWriterWorkers": 0},
            "target": {"manifest": str(manifest_path), "selection": "first"},
        }
    )
    target = resolveTarget(run_config)

    assert isinstance(target, TargetBatch)
    assert target.batchSize == 3

    random_payload = {
        "experiment": {"name": "manifest_random"},
        "device": {"device": "cpu", "dtype": "float32"},
        "array": {"allowedElementCount": [4], "allowedAspectRatio": [1.0]},
        "evolution": {"batchSize": 3, "evolutionSteps": 1},
        "logging": {"logMode": "off"},
        "checkpoint": {"checkpointMode": "off"},
        "workers": {"asyncIO": False, "datasetWriterWorkers": 0, "checkpointWriterWorkers": 0},
        "target": {
            "manifest": str(manifest_path),
            "selection": "random_without_replacement",
            "selectionSeed": 11,
        },
    }
    target_a = resolveTarget(buildRunConfig(random_payload))
    target_b = resolveTarget(buildRunConfig(random_payload))
    assert isinstance(target_a, TargetBatch)
    assert isinstance(target_b, TargetBatch)
    torch.testing.assert_close(target_a.powerMap, target_b.powerMap)
    torch.testing.assert_close(target_a.hotspotCoordinates, target_b.hotspotCoordinates)


def test_resolve_target_manifest_selection_count_must_match_batch_size(tmp_path: Path) -> None:
    config_path = tmp_path / "targets.yaml"
    output_root = tmp_path / "corpus"
    write_corpus_config(config_path, output_root)
    generateTargetCorpus(config_path)
    manifest_path = output_root / "manifest.yaml"

    run_config = buildRunConfig(
        {
            "experiment": {"name": "manifest_mismatch"},
            "device": {"device": "cpu", "dtype": "float32"},
            "array": {"allowedElementCount": [4], "allowedAspectRatio": [1.0]},
            "evolution": {"batchSize": 3, "evolutionSteps": 1},
            "logging": {"logMode": "off"},
            "checkpoint": {"checkpointMode": "off"},
            "workers": {"asyncIO": False, "datasetWriterWorkers": 0, "checkpointWriterWorkers": 0},
            "target": {
                "manifest": str(manifest_path),
                "selection": "first",
                "selectionCount": 2,
            },
        }
    )

    with pytest.raises(ValueError):
        resolveTarget(run_config)


def test_generated_manifest_can_drive_a_smoke_training_run(tmp_path: Path) -> None:
    config_path = tmp_path / "targets.yaml"
    output_root = tmp_path / "corpus"
    write_corpus_config(config_path, output_root, count=4)
    generateTargetCorpus(config_path)
    manifest_path = output_root / "manifest.yaml"

    run_config = buildRunConfig(
        {
            "experiment": {"name": "manifest_smoke", "targetMode": "auto"},
            "device": {"device": "cpu", "dtype": "float32"},
            "array": {"allowedElementCount": [4], "allowedAspectRatio": [1.0]},
            "evolution": {"batchSize": 2, "evolutionSteps": 1},
            "loss": {"wide_grid_size": 8},
            "logging": {"logMode": "off"},
            "checkpoint": {"checkpointMode": "off"},
            "workers": {"asyncIO": False, "datasetWriterWorkers": 0, "checkpointWriterWorkers": 0},
            "target": {"manifest": str(manifest_path), "selection": "first"},
        }
    )
    target = resolveTarget(run_config)
    assert isinstance(target, TargetBatch)

    controller = EvolutionController(
        config=EvolutionConfig(batchSize=2, evolutionSteps=1),
        targetSpec=target,
        arraySpec=make_array_spec(),
        lossParams=LossConfig(wide_grid_size=8),
        experimentName="manifest_smoke",
        archiveRoot=tmp_path / "archive",
        loggingConfig=LoggingConfig(logMode="off"),
        checkpointConfig=CheckpointConfig(checkpointMode="off"),
        workerConfig=WorkerConfig(asyncIO=False, datasetWriterWorkers=0, checkpointWriterWorkers=0),
        targetMode="auto",
        writerLogDir=tmp_path / "runs",
    )

    controller.train(
        dtype=torch.float32,
        device=torch.device("cpu"),
        logDir=tmp_path / "runs",
        plotProjection=False,
        resume=False,
    )

    assert (tmp_path / "archive" / "manifest_smoke" / "best.pt").exists()
