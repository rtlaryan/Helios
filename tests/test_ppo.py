from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from model.base import ModelContext, ModelInput
from model.config import buildModelConfig, loadModelConfig
from model.registry import build_model
from scripts.targetSpec import TargetSpec
from train.config import loadPPORunConfig, loadTargetCorpus, ppoRunConfigToDict
from train.evolve import _payloadToCPU
from train.ppo import PPOCorpusLoader, buildPPOControllerFromConfig


def make_target(index: int = 0) -> TargetSpec:
    latitudes = torch.tensor([[10.0, 10.5], [11.0, 11.5]], dtype=torch.float32) + index
    longitudes = torch.tensor([[20.0, 20.5], [21.0, 21.5]], dtype=torch.float32) + index
    importance = torch.tensor([[1.0, 0.8], [0.4, 0.2]], dtype=torch.float32)
    power = torch.tensor([[0.0, -3.0], [-6.0, -9.0]], dtype=torch.float32)
    hotspots = torch.tensor([[10.5 + index, 20.5 + index]], dtype=torch.float32)
    return TargetSpec(
        searchLatitudes=latitudes,
        searchLongitudes=longitudes,
        importanceMap=importance,
        powerMap=power,
        hotspotCoordinates=hotspots,
    )


def make_target_with_shape(index: int, shape: tuple[int, int]) -> TargetSpec:
    height, width = shape
    latitudes = torch.arange(height * width, dtype=torch.float32).reshape(height, width) + index
    longitudes = latitudes + 100.0
    importance = torch.ones((height, width), dtype=torch.float32)
    power = torch.zeros((height, width), dtype=torch.float32)
    hotspots = torch.tensor([[float(index), float(index)]], dtype=torch.float32)
    return TargetSpec(
        searchLatitudes=latitudes,
        searchLongitudes=longitudes,
        importanceMap=importance,
        powerMap=power,
        hotspotCoordinates=hotspots,
    )


def write_manifest(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    for index in range(2):
        path = base_dir / f"target_{index}.pt"
        torch.save(make_target(index), path)
        records.append({"path": path.name})

    manifest_path = base_dir / "manifest.yaml"
    with manifest_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"records": records}, handle, sort_keys=False)
    return manifest_path


def write_manifest_with_shapes(base_dir: Path, shapes: list[tuple[int, int]]) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    for index, shape in enumerate(shapes):
        path = base_dir / f"target_{index}.pt"
        torch.save(make_target_with_shape(index, shape), path)
        records.append({"path": path.name})

    manifest_path = base_dir / "manifest.yaml"
    with manifest_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"records": records}, handle, sort_keys=False)
    return manifest_path


def write_model_config(
    base_dir: Path,
    architecture: str = "cnn_mlp",
    decoder_type: str = "flat_action",
    global_features: list[str] | None = None,
    element_features: list[str] | None = None,
) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "architecture": architecture,
        "common": {
            "activation": "gelu",
            "norm": "layernorm",
            "hiddenDim": 64,
            "logStdMode": "global",
            "logStdInit": -0.5,
        },
        "encoder": {
            "type": "cnn" if architecture == "cnn_mlp" else "transformer",
            "cnn": {
                "inChannels": 2,
                "convChannels": [8, 16],
                "kernelSizes": [3, 3],
                "strides": [1, 1],
                "paddings": [1, 1],
                "pooling": "adaptive_avg",
                "dropout": 0.0,
            },
            "transformer": {
                "patchSize": 4,
                "embedDim": 32,
                "depth": 2,
                "numHeads": 4,
                "mlpRatio": 2.0,
            },
        },
        "decoder": {
            "type": decoder_type,
            "flat_action": {
                "mlpLayers": [32],
                "dropout": 0.0,
            },
            "coordinate_conditioned": {
                "mlpLayers": [32],
                "dropout": 0.0,
            },
        },
        "context": {
            "globalFeatures": list(global_features or []),
            "elementFeatures": list(element_features or []),
        },
    }
    path = base_dir / f"{architecture}.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return path


def write_ppo_config(base_dir: Path, model_path: Path, manifest_path: Path) -> Path:
    config_dir = base_dir / "configs" / "ppo"
    model_dir = base_dir / "configs" / "models"
    config_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    copied_model = model_dir / model_path.name
    copied_model.write_text(model_path.read_text(encoding="utf-8"), encoding="utf-8")

    payload = {
        "experiment": {
            "name": "ppo_smoke",
            "logDir": str(base_dir / "runs"),
            "archiveDir": str(base_dir / "archive"),
            "resume": False,
        },
        "device": {"device": "cpu", "dtype": "float32"},
        "array": {
            "allowedElementCount": [4],
            "allowedAspectRatio": [1.0],
            "latitudeRange": [0.0, 0.0],
            "longitudeRange": [-83.0, -83.0],
            "altitudeRange": [3.6e7, 3.6e7],
        },
        "modelConfig": f"../models/{copied_model.name}",
        "ppo": {
            "rolloutBatchSize": 2,
            "updateSteps": 1,
            "ppoEpochs": 1,
            "minibatchSize": 1,
            "learningRate": 1e-3,
            "clipEpsilon": 0.2,
            "valueLossCoef": 0.5,
            "entropyCoef": 0.01,
            "maxGradNorm": 0.5,
            "seed": 3,
        },
        "loss": {"w_shape": 1.0, "w_eff": 0.1, "w_wide": 0.5, "wide_grid_size": 8},
        "logging": {"logMode": "off"},
        "checkpoint": {"checkpointMode": "periodic", "checkpointEverySteps": 1},
        "workers": {
            "asyncIO": False,
            "datasetWriterWorkers": 0,
            "checkpointWriterWorkers": 0,
            "ioQueueSize": 1,
        },
        "target": {
            "manifest": str(manifest_path),
            "selection": "first",
            "decimate": 1,
            "loadingMode": "block_window",
            "blockSize": 2,
            "prefetchNextBlock": True,
            "loaderWorkers": 1,
            "shuffleEachEpoch": True,
        },
    }
    config_path = config_dir / "run.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return config_path


def test_load_ppo_run_config_resolves_relative_model_path(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path / "data")
    model_path = write_model_config(tmp_path / "models")
    config_path = write_ppo_config(tmp_path, model_path, manifest_path)

    run_config = loadPPORunConfig(config_path)

    assert run_config.sourcePath == config_path
    assert run_config.modelSourcePath == (config_path.parent / "../models" / model_path.name).resolve()
    assert run_config.model.architecture == "cnn_mlp"
    assert run_config.model.encoder.type == "cnn"
    assert run_config.model.decoder.type == "flat_action"
    assert tuple(run_config.model.context.globalFeatures) == ()


def test_model_config_rejects_mismatched_cnn_lists() -> None:
    with pytest.raises(ValueError, match="matching lengths"):
        buildModelConfig(
            {
                "architecture": "cnn_mlp",
                "encoder": {
                    "type": "cnn",
                    "cnn": {
                        "convChannels": [8, 16],
                        "kernelSizes": [3],
                        "strides": [1, 1],
                        "paddings": [1, 1],
                    },
                },
            }
        )


def test_unknown_architecture_fails_clearly() -> None:
    with pytest.raises(ValueError, match="unsupported model architecture"):
        buildModelConfig({"architecture": "unknown"})


def test_unknown_context_feature_fails_clearly() -> None:
    with pytest.raises(ValueError, match="unknown model.context.elementFeatures entry"):
        buildModelConfig(
            {
                "architecture": "cnn_mlp",
                "decoder": {"type": "coordinate_conditioned"},
                "context": {"elementFeatures": ["not_a_real_feature"]},
            }
        )


def test_cnn_mlp_model_outputs_expected_shapes(tmp_path: Path) -> None:
    config = loadModelConfig(write_model_config(tmp_path / "models"))

    model = build_model(config, action_dim=8)
    outputs = model(torch.randn(3, 2, 4, 4))

    assert outputs.policyMean.shape == (3, 8)
    assert outputs.logStd.shape == (3, 8)
    assert outputs.value.shape == (3,)


def test_flat_action_model_accepts_global_context_features(tmp_path: Path) -> None:
    config = loadModelConfig(
        write_model_config(
            tmp_path / "models",
            decoder_type="flat_action",
            global_features=["array_lla", "gain"],
        )
    )

    model = build_model(config, action_dim=8)
    outputs = model(
        ModelInput(
            targetTensor=torch.randn(3, 2, 4, 4),
            context=ModelContext(
                globalFeatures=torch.randn(3, 4),
            ),
        )
    )

    assert outputs.policyMean.shape == (3, 8)
    assert outputs.logStd.shape == (3, 8)
    assert outputs.value.shape == (3,)


def test_coordinate_conditioned_model_outputs_expected_shapes(tmp_path: Path) -> None:
    config = loadModelConfig(
        write_model_config(
            tmp_path / "models",
            decoder_type="coordinate_conditioned",
            global_features=["array_lla", "gain"],
            element_features=["element_local_xyz", "element_mask"],
        )
    )

    model = build_model(config, action_dim=8)
    outputs = model(
        ModelInput(
            targetTensor=torch.randn(3, 2, 4, 4),
            context=ModelContext(
                globalFeatures=torch.randn(3, 4),
                elementFeatures=torch.randn(3, 4, 4),
                elementMask=torch.ones((3, 4), dtype=torch.bool),
            ),
        )
    )

    assert outputs.policyMean.shape == (3, 8)
    assert outputs.logStd.shape == (3, 8)
    assert outputs.value.shape == (3,)


def test_ppo_controller_builds_requested_array_context_features(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path / "data")
    model_path = write_model_config(
        tmp_path / "models",
        decoder_type="coordinate_conditioned",
        global_features=["array_lla", "geometry_one_hot"],
        element_features=["normalized_aperture_xyz", "element_mask"],
    )
    config_path = write_ppo_config(tmp_path, model_path, manifest_path)
    controller, (device, dtype, _) = buildPPOControllerFromConfig(str(config_path))

    controller._init_model(dtype=dtype, device=device)
    context = controller._build_model_context(batchSize=2, device=device, dtype=dtype)

    assert context.globalFeatures is not None
    assert context.elementFeatures is not None
    assert context.globalFeatures.shape == (2, 5)
    assert context.elementFeatures.shape == (2, 4, 4)
    assert context.arraySpec is not None
    assert context.arraySpec["geometry"] == "URA"


def test_transformer_architecture_is_scaffold_only(tmp_path: Path) -> None:
    config = loadModelConfig(write_model_config(tmp_path / "models", architecture="transformer"))

    with pytest.raises(NotImplementedError, match="not implemented yet"):
        build_model(config, action_dim=8)


def test_ppo_config_dict_includes_model_reference_and_resolved_model(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path / "data")
    model_path = write_model_config(tmp_path / "models")
    config_path = write_ppo_config(tmp_path, model_path, manifest_path)

    run_config = loadPPORunConfig(config_path)
    payload = ppoRunConfigToDict(run_config)

    assert payload["modelConfig"] == f"../models/{model_path.name}"
    assert payload["model"]["architecture"] == "cnn_mlp"
    assert payload["model"]["encoder"]["type"] == "cnn"
    assert tuple(payload["model"]["encoder"]["cnn"]["convChannels"]) == (8, 16)
    assert tuple(payload["model"]["decoder"]["flat_action"]["mlpLayers"]) == (32,)
    assert payload["model"]["context"]["globalFeatures"] == []
    assert payload["model"]["context"]["elementFeatures"] == []


def test_legacy_cnn_mlp_model_config_still_loads() -> None:
    config = buildModelConfig(
        {
            "architecture": "cnn_mlp",
            "common": {"activation": "gelu"},
            "cnn_mlp": {
                "inChannels": 2,
                "convChannels": [8, 16],
                "kernelSizes": [3, 3],
                "strides": [1, 1],
                "paddings": [1, 1],
                "pooling": "adaptive_avg",
                "mlpLayers": [32, 16],
                "dropout": 0.1,
            },
        }
    )

    assert config.encoder.type == "cnn"
    assert tuple(config.encoder.cnn.convChannels) == (8, 16)
    assert tuple(config.decoder.flat_action.mlpLayers) == (32, 16)
    assert config.encoder.cnn.dropout == pytest.approx(0.1)
    assert config.decoder.flat_action.dropout == pytest.approx(0.1)


def test_load_target_corpus_uses_manifest_targets(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path / "data")
    model_path = write_model_config(tmp_path / "models")
    config_path = write_ppo_config(tmp_path, model_path, manifest_path)

    run_config = loadPPORunConfig(config_path)
    corpus = loadTargetCorpus(run_config)

    assert len(corpus) == 2
    assert corpus[0].targetShape == corpus[1].targetShape == torch.Size([2, 2])


def test_build_controller_does_not_preload_corpus_blocks(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path = write_manifest(tmp_path / "data")
    model_path = write_model_config(tmp_path / "models")
    config_path = write_ppo_config(tmp_path, model_path, manifest_path)

    controller, _ = buildPPOControllerFromConfig(str(config_path))
    captured = capsys.readouterr().out

    assert "effectiveCorpusSize=2" in captured
    assert "loadingMode=block_window" in captured
    assert controller.corpusLoader.activeBlock is None
    assert len(controller.corpusLoader.recordPaths) == 2


def test_selection_count_caps_effective_corpus_before_block_loading(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path / "data")
    model_path = write_model_config(tmp_path / "models")
    config_path = write_ppo_config(tmp_path, model_path, manifest_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["target"]["selectionCount"] = 1
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    controller, _ = buildPPOControllerFromConfig(str(config_path))

    assert len(controller.corpusLoader.recordPaths) == 1


def test_block_loader_traverses_full_manifest_over_epoch(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path / "data")
    model_path = write_model_config(tmp_path / "models")
    config_path = write_ppo_config(tmp_path, model_path, manifest_path)
    run_config = loadPPORunConfig(config_path)
    run_config.target.blockSize = 1
    run_config.target.prefetchNextBlock = False
    run_config.target.shuffleEachEpoch = False

    _, recordPaths = (Path(run_config.target.manifest), [])  # placate typing in this scope
    from train.config import resolvePPOTargetRecordPaths

    _, recordPaths = resolvePPOTargetRecordPaths(run_config)
    loader = PPOCorpusLoader(
        recordPaths=recordPaths,
        targetConfig=run_config.target,
        rolloutBatchSize=1,
        seed=run_config.ppo.seed,
    )
    batchA = loader.next_target_batch(torch.device("cpu"), torch.float32, showProgress=False)
    batchB = loader.next_target_batch(torch.device("cpu"), torch.float32, showProgress=False)

    assert batchA.batchSize == batchB.batchSize == 1
    assert batchA.searchLatitudes[0, 0, 0].item() == pytest.approx(10.0)
    assert batchB.searchLatitudes[0, 0, 0].item() == pytest.approx(11.0)
    assert loader.targetsSeen == 2
    loader.close()


def test_block_loader_prefetches_next_block(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path / "data")
    model_path = write_model_config(tmp_path / "models")
    config_path = write_ppo_config(tmp_path, model_path, manifest_path)
    run_config = loadPPORunConfig(config_path)
    run_config.target.blockSize = 1
    run_config.target.prefetchNextBlock = True

    from train.config import resolvePPOTargetRecordPaths

    _, recordPaths = resolvePPOTargetRecordPaths(run_config)
    loader = PPOCorpusLoader(
        recordPaths=recordPaths,
        targetConfig=run_config.target,
        rolloutBatchSize=1,
        seed=run_config.ppo.seed,
    )
    loader._ensure_initialized(showProgress=False)
    if loader.nextBlockFuture is not None:
        loader.nextBlock = loader.nextBlockFuture.result()
        loader.nextBlockFuture = None
    status = loader.status()

    assert status["prefetchReady"] is True or status["prefetchLoading"] is True
    loader.close()


def test_block_loader_state_round_trips(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path / "data")
    model_path = write_model_config(tmp_path / "models")
    config_path = write_ppo_config(tmp_path, model_path, manifest_path)
    run_config = loadPPORunConfig(config_path)
    run_config.target.blockSize = 1
    run_config.target.prefetchNextBlock = False
    run_config.target.shuffleEachEpoch = False

    from train.config import resolvePPOTargetRecordPaths

    _, recordPaths = resolvePPOTargetRecordPaths(run_config)
    loader = PPOCorpusLoader(
        recordPaths=recordPaths,
        targetConfig=run_config.target,
        rolloutBatchSize=1,
        seed=run_config.ppo.seed,
    )
    batchA = loader.next_target_batch(torch.device("cpu"), torch.float32, showProgress=False)
    state = loader.serializeState()

    restored = PPOCorpusLoader(
        recordPaths=recordPaths,
        targetConfig=run_config.target,
        rolloutBatchSize=1,
        seed=run_config.ppo.seed,
    )
    restored.restoreState(state)
    batchB = restored.next_target_batch(torch.device("cpu"), torch.float32, showProgress=False)

    assert batchA.searchLatitudes[0, 0, 0].item() == pytest.approx(10.0)
    assert batchB.searchLatitudes[0, 0, 0].item() == pytest.approx(11.0)
    loader.close()
    restored.close()


def test_block_loader_shape_mismatch_fails_clearly(tmp_path: Path) -> None:
    manifest_path = write_manifest_with_shapes(tmp_path / "data", [(2, 2), (3, 3)])
    model_path = write_model_config(tmp_path / "models")
    config_path = write_ppo_config(tmp_path, model_path, manifest_path)
    run_config = loadPPORunConfig(config_path)
    run_config.target.blockSize = 2
    run_config.target.prefetchNextBlock = False

    from train.config import resolvePPOTargetRecordPaths

    _, recordPaths = resolvePPOTargetRecordPaths(run_config)
    loader = PPOCorpusLoader(
        recordPaths=recordPaths,
        targetConfig=run_config.target,
        rolloutBatchSize=1,
        seed=run_config.ppo.seed,
    )

    with pytest.raises(ValueError, match="share the same spatial shape"):
        loader._ensure_initialized(showProgress=False)
    loader.close()


def test_payload_to_cpu_converts_nested_tensors() -> None:
    payload = {
        "tensor": torch.ones(2),
        "nested": {
            "list": [torch.zeros(1), (torch.full((1,), 3.0),)],
        },
    }

    converted = _payloadToCPU(payload)

    assert isinstance(converted["tensor"], torch.Tensor)
    assert converted["tensor"].device.type == "cpu"
    assert converted["nested"]["list"][0].device.type == "cpu"
    assert converted["nested"]["list"][1][0].device.type == "cpu"


def test_ppo_log_std_clamping_keeps_sampling_finite(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path / "data")
    model_path = write_model_config(tmp_path / "models")
    config_path = write_ppo_config(tmp_path, model_path, manifest_path)
    controller, (device, dtype, _) = buildPPOControllerFromConfig(str(config_path))

    controller._init_model(dtype=dtype, device=device)
    assert controller.model is not None
    assert hasattr(controller.model, "logStd")

    with torch.no_grad():
        controller.model.logStd.fill_(float("inf"))

    outputs = controller.model(torch.randn(2, 2, 2, 2))
    actions = controller._sample_actions(outputs.policyMean, outputs.logStd)
    log_prob = controller._log_prob(outputs.policyMean, outputs.logStd, actions)
    entropy = controller._entropy(outputs.logStd)

    assert torch.isfinite(actions).all()
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()

    controller._clamp_model_parameters()
    assert torch.isfinite(controller.model.logStd).all()
    assert float(controller.model.logStd.max().item()) <= 2.0


def test_ppo_train_writes_run_and_model_artifacts(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path / "data")
    model_path = write_model_config(tmp_path / "models")
    config_path = write_ppo_config(tmp_path, model_path, manifest_path)
    controller, (device, dtype, experiment) = buildPPOControllerFromConfig(str(config_path))

    controller.train(dtype=dtype, device=device, logDir=experiment.logDir, resume=experiment.resume)

    run_dir = tmp_path / "runs" / "ppo_smoke"
    archive_dir = tmp_path / "archive" / "ppo_smoke"
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "model.yaml").exists()
    assert (archive_dir / "checkpoints" / "latest.pt").exists()
    assert (archive_dir / "best.pt").exists()
    assert (archive_dir / "final.pt").exists()

    with (run_dir / "config.yaml").open("r", encoding="utf-8") as handle:
        saved = yaml.safe_load(handle)
    assert saved["model"]["architecture"] == "cnn_mlp"
    assert saved["model"]["encoder"]["type"] == "cnn"


def test_swapping_model_configs_changes_loaded_architecture_params(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path / "data")
    model_path_a = write_model_config(tmp_path / "models_a")
    model_path_b = write_model_config(tmp_path / "models_b")

    payload_b = yaml.safe_load(model_path_b.read_text(encoding="utf-8"))
    payload_b["decoder"]["flat_action"]["mlpLayers"] = [64, 32]
    with model_path_b.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload_b, handle, sort_keys=False)

    run_config_a = loadPPORunConfig(write_ppo_config(tmp_path / "run_a", model_path_a, manifest_path))
    run_config_b = loadPPORunConfig(write_ppo_config(tmp_path / "run_b", model_path_b, manifest_path))

    assert tuple(run_config_a.model.decoder.flat_action.mlpLayers) == (32,)
    assert tuple(run_config_b.model.decoder.flat_action.mlpLayers) == (64, 32)
