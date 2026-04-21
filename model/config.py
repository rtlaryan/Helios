from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

import yaml

ArchitectureName = Literal["cnn_mlp", "transformer"]
ActivationName = Literal["relu", "gelu", "tanh"]
NormName = Literal["none", "layernorm"]
LogStdMode = Literal["global", "per_dimension"]
PoolingName = Literal["adaptive_avg", "adaptive_max"]
EncoderName = Literal["cnn", "transformer"]
DecoderName = Literal["flat_action", "coordinate_conditioned"]
GlobalContextFeatureName: TypeAlias = Literal[
    "array_lla",
    "array_ecef",
    "gain",
    "wavelength",
    "element_spacing",
    "element_count",
    "geometry_one_hot",
]
ElementContextFeatureName: TypeAlias = Literal[
    "element_local_xyz",
    "element_global_xyz",
    "normalized_aperture_xyz",
    "element_mask",
]

SUPPORTED_ARCHITECTURES = {"cnn_mlp", "transformer"}
GLOBAL_CONTEXT_FEATURE_DIMS: dict[GlobalContextFeatureName, int] = {
    "array_lla": 3,
    "array_ecef": 3,
    "gain": 1,
    "wavelength": 1,
    "element_spacing": 1,
    "element_count": 1,
    "geometry_one_hot": 2,
}
ELEMENT_CONTEXT_FEATURE_DIMS: dict[ElementContextFeatureName, int] = {
    "element_local_xyz": 3,
    "element_global_xyz": 3,
    "normalized_aperture_xyz": 3,
    "element_mask": 1,
}


@dataclass
class CommonModelConfig:
    activation: ActivationName = "gelu"
    norm: NormName = "layernorm"
    hiddenDim: int = 256
    logStdMode: LogStdMode = "global"
    logStdInit: float = -0.5


@dataclass
class CNNEncoderConfig:
    inChannels: int = 2
    convChannels: tuple[int, ...] = (32, 64, 128)
    kernelSizes: tuple[int, ...] = (5, 3, 3)
    strides: tuple[int, ...] = (2, 2, 2)
    paddings: tuple[int, ...] = (2, 1, 1)
    pooling: PoolingName = "adaptive_avg"
    dropout: float = 0.0


@dataclass
class TransformerEncoderConfig:
    patchSize: int = 4
    embedDim: int = 256
    depth: int = 6
    numHeads: int = 8
    mlpRatio: float = 4.0


@dataclass
class FlatActionDecoderConfig:
    mlpLayers: tuple[int, ...] = (256, 256)
    dropout: float = 0.0


@dataclass
class CoordinateConditionedDecoderConfig:
    mlpLayers: tuple[int, ...] = (256, 256)
    dropout: float = 0.0


@dataclass
class ContextConfig:
    globalFeatures: tuple[GlobalContextFeatureName, ...] = ()
    elementFeatures: tuple[ElementContextFeatureName, ...] = ()


@dataclass
class EncoderConfig:
    type: EncoderName = "cnn"
    cnn: CNNEncoderConfig = field(default_factory=CNNEncoderConfig)
    transformer: TransformerEncoderConfig = field(default_factory=TransformerEncoderConfig)


@dataclass
class DecoderConfig:
    type: DecoderName = "flat_action"
    flat_action: FlatActionDecoderConfig = field(default_factory=FlatActionDecoderConfig)
    coordinate_conditioned: CoordinateConditionedDecoderConfig = field(
        default_factory=CoordinateConditionedDecoderConfig
    )


@dataclass
class CNNMLPModelConfig:
    inChannels: int = 2
    convChannels: tuple[int, ...] = (32, 64, 128)
    kernelSizes: tuple[int, ...] = (5, 3, 3)
    strides: tuple[int, ...] = (2, 2, 2)
    paddings: tuple[int, ...] = (2, 1, 1)
    pooling: PoolingName = "adaptive_avg"
    mlpLayers: tuple[int, ...] = (256, 256)
    dropout: float = 0.0


@dataclass
class ModelConfig:
    architecture: ArchitectureName = "cnn_mlp"
    common: CommonModelConfig = field(default_factory=CommonModelConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    # Legacy aliases kept so older call sites and configs still work.
    cnn_mlp: CNNMLPModelConfig = field(default_factory=CNNMLPModelConfig)
    transformer: TransformerEncoderConfig = field(default_factory=TransformerEncoderConfig)


def _ensure_dict(payload: Any, section: str) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"{section} must be a mapping")
    return payload


def _convert_sequences(payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    normalized = dict(payload)
    for key in keys:
        value = normalized.get(key)
        if isinstance(value, list):
            normalized[key] = tuple(value)
    return normalized


def _as_tuple(payload: Any) -> tuple[Any, ...]:
    if payload is None:
        return ()
    if isinstance(payload, tuple):
        return payload
    if isinstance(payload, list):
        return tuple(payload)
    raise ValueError("context feature lists must be sequences")


def _build_common_config(payload: dict[str, Any]) -> CommonModelConfig:
    return CommonModelConfig(
        activation=str(payload.get("activation", "gelu")),
        norm=str(payload.get("norm", "layernorm")),
        hiddenDim=int(payload.get("hiddenDim", 256)),
        logStdMode=str(payload.get("logStdMode", "global")),
        logStdInit=float(payload.get("logStdInit", -0.5)),
    )


def _build_cnn_encoder_config(payload: dict[str, Any]) -> CNNEncoderConfig:
    normalized = _convert_sequences(
        payload,
        ("convChannels", "kernelSizes", "strides", "paddings"),
    )
    return CNNEncoderConfig(
        inChannels=int(normalized.get("inChannels", 2)),
        convChannels=tuple(int(value) for value in normalized.get("convChannels", (32, 64, 128))),
        kernelSizes=tuple(int(value) for value in normalized.get("kernelSizes", (5, 3, 3))),
        strides=tuple(int(value) for value in normalized.get("strides", (2, 2, 2))),
        paddings=tuple(int(value) for value in normalized.get("paddings", (2, 1, 1))),
        pooling=str(normalized.get("pooling", "adaptive_avg")),
        dropout=float(normalized.get("dropout", 0.0)),
    )


def _build_transformer_encoder_config(payload: dict[str, Any]) -> TransformerEncoderConfig:
    return TransformerEncoderConfig(
        patchSize=int(payload.get("patchSize", 4)),
        embedDim=int(payload.get("embedDim", 256)),
        depth=int(payload.get("depth", 6)),
        numHeads=int(payload.get("numHeads", 8)),
        mlpRatio=float(payload.get("mlpRatio", 4.0)),
    )


def _build_flat_action_decoder_config(payload: dict[str, Any]) -> FlatActionDecoderConfig:
    normalized = _convert_sequences(payload, ("mlpLayers",))
    return FlatActionDecoderConfig(
        mlpLayers=tuple(int(value) for value in normalized.get("mlpLayers", (256, 256))),
        dropout=float(normalized.get("dropout", 0.0)),
    )


def _build_coordinate_conditioned_decoder_config(
    payload: dict[str, Any]
) -> CoordinateConditionedDecoderConfig:
    normalized = _convert_sequences(payload, ("mlpLayers",))
    return CoordinateConditionedDecoderConfig(
        mlpLayers=tuple(int(value) for value in normalized.get("mlpLayers", (256, 256))),
        dropout=float(normalized.get("dropout", 0.0)),
    )


def _build_context_config(payload: dict[str, Any]) -> ContextConfig:
    return ContextConfig(
        globalFeatures=tuple(str(value) for value in _as_tuple(payload.get("globalFeatures"))),
        elementFeatures=tuple(str(value) for value in _as_tuple(payload.get("elementFeatures"))),
    )


def _legacy_cnn_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return _convert_sequences(
        _ensure_dict(payload.get("cnn_mlp"), "cnn_mlp"),
        ("convChannels", "kernelSizes", "strides", "paddings", "mlpLayers"),
    )


def _derive_encoder_payload(
    payload: dict[str, Any],
    architecture: str,
    legacyCNNPayload: dict[str, Any],
) -> EncoderConfig:
    encoderPayload = _ensure_dict(payload.get("encoder"), "encoder")
    encoderType = str(
        encoderPayload.get("type", "cnn" if architecture == "cnn_mlp" else "transformer")
    )

    cnnPayload = _ensure_dict(encoderPayload.get("cnn"), "encoder.cnn")
    if not cnnPayload and legacyCNNPayload:
        cnnPayload = {
            key: legacyCNNPayload[key]
            for key in (
                "inChannels",
                "convChannels",
                "kernelSizes",
                "strides",
                "paddings",
                "pooling",
                "dropout",
            )
            if key in legacyCNNPayload
        }

    transformerPayload = _ensure_dict(encoderPayload.get("transformer"), "encoder.transformer")
    if not transformerPayload:
        transformerPayload = _ensure_dict(payload.get("transformer"), "transformer")

    return EncoderConfig(
        type=encoderType,
        cnn=_build_cnn_encoder_config(cnnPayload),
        transformer=_build_transformer_encoder_config(transformerPayload),
    )


def _derive_decoder_payload(payload: dict[str, Any], legacyCNNPayload: dict[str, Any]) -> DecoderConfig:
    decoderPayload = _ensure_dict(payload.get("decoder"), "decoder")
    decoderType = str(decoderPayload.get("type", "flat_action"))

    flatPayload = _ensure_dict(decoderPayload.get("flat_action"), "decoder.flat_action")
    coordinatePayload = _ensure_dict(
        decoderPayload.get("coordinate_conditioned"),
        "decoder.coordinate_conditioned",
    )
    if not flatPayload and legacyCNNPayload:
        flatPayload = {
            key: legacyCNNPayload[key]
            for key in ("mlpLayers", "dropout")
            if key in legacyCNNPayload
        }

    return DecoderConfig(
        type=decoderType,
        flat_action=_build_flat_action_decoder_config(flatPayload),
        coordinate_conditioned=_build_coordinate_conditioned_decoder_config(coordinatePayload),
    )


def _legacy_cnn_alias(encoder: EncoderConfig, decoder: DecoderConfig) -> CNNMLPModelConfig:
    legacyDecoder = (
        decoder.flat_action
        if decoder.type == "flat_action"
        else decoder.coordinate_conditioned
    )
    return CNNMLPModelConfig(
        inChannels=encoder.cnn.inChannels,
        convChannels=tuple(encoder.cnn.convChannels),
        kernelSizes=tuple(encoder.cnn.kernelSizes),
        strides=tuple(encoder.cnn.strides),
        paddings=tuple(encoder.cnn.paddings),
        pooling=encoder.cnn.pooling,
        mlpLayers=tuple(legacyDecoder.mlpLayers),
        dropout=float(legacyDecoder.dropout),
    )


def globalContextFeatureDim(config: ContextConfig) -> int:
    return sum(GLOBAL_CONTEXT_FEATURE_DIMS[name] for name in config.globalFeatures)


def elementContextFeatureDim(config: ContextConfig) -> int:
    return sum(ELEMENT_CONTEXT_FEATURE_DIMS[name] for name in config.elementFeatures)


def buildModelConfig(payload: dict[str, Any]) -> ModelConfig:
    if not isinstance(payload, dict):
        raise ValueError("model config root must be a mapping")

    architecture = str(payload.get("architecture", "cnn_mlp"))
    common = _build_common_config(_ensure_dict(payload.get("common"), "common"))
    legacyCNNPayload = _legacy_cnn_payload(payload)
    encoder = _derive_encoder_payload(payload, architecture, legacyCNNPayload)
    decoder = _derive_decoder_payload(payload, legacyCNNPayload)
    context = _build_context_config(_ensure_dict(payload.get("context"), "context"))
    legacyCNN = _legacy_cnn_alias(encoder, decoder)

    config = ModelConfig(
        architecture=architecture,
        common=common,
        encoder=encoder,
        decoder=decoder,
        context=context,
        cnn_mlp=legacyCNN,
        transformer=encoder.transformer,
    )
    validateModelConfig(config)
    return config


def validateModelConfig(config: ModelConfig) -> None:
    if config.architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"unsupported model architecture: {config.architecture} "
            f"(expected one of {sorted(SUPPORTED_ARCHITECTURES)})"
        )
    if config.common.activation not in {"relu", "gelu", "tanh"}:
        raise ValueError("model.common.activation must be one of 'relu', 'gelu', or 'tanh'")
    if config.common.norm not in {"none", "layernorm"}:
        raise ValueError("model.common.norm must be either 'none' or 'layernorm'")
    if config.common.logStdMode not in {"global", "per_dimension"}:
        raise ValueError(
            "model.common.logStdMode must be either 'global' or 'per_dimension'"
        )
    if config.common.hiddenDim <= 0:
        raise ValueError("model.common.hiddenDim must be positive")

    if config.encoder.type not in {"cnn", "transformer"}:
        raise ValueError("model.encoder.type must be either 'cnn' or 'transformer'")
    if config.decoder.type not in {"flat_action", "coordinate_conditioned"}:
        raise ValueError(
            "model.decoder.type must be either 'flat_action' or 'coordinate_conditioned'"
        )

    if config.architecture == "cnn_mlp" and config.encoder.type != "cnn":
        raise ValueError("model architecture 'cnn_mlp' requires model.encoder.type='cnn'")
    if config.architecture == "transformer" and config.encoder.type != "transformer":
        raise ValueError(
            "model architecture 'transformer' requires model.encoder.type='transformer'"
        )

    cnn = config.encoder.cnn
    layerLengths = {
        len(cnn.convChannels),
        len(cnn.kernelSizes),
        len(cnn.strides),
        len(cnn.paddings),
    }
    if cnn.inChannels <= 0:
        raise ValueError("model.encoder.cnn.inChannels must be positive")
    if len(layerLengths) != 1:
        raise ValueError(
            "model.encoder.cnn convChannels/kernelSizes/strides/paddings must have matching lengths"
        )
    if len(cnn.convChannels) == 0:
        raise ValueError("model.encoder.cnn must define at least one convolution layer")
    if any(value <= 0 for value in cnn.convChannels):
        raise ValueError("model.encoder.cnn.convChannels values must be positive")
    if any(value <= 0 for value in cnn.kernelSizes):
        raise ValueError("model.encoder.cnn.kernelSizes values must be positive")
    if any(value <= 0 for value in cnn.strides):
        raise ValueError("model.encoder.cnn.strides values must be positive")
    if any(value < 0 for value in cnn.paddings):
        raise ValueError("model.encoder.cnn.paddings values must be non-negative")
    if cnn.pooling not in {"adaptive_avg", "adaptive_max"}:
        raise ValueError(
            "model.encoder.cnn.pooling must be either 'adaptive_avg' or 'adaptive_max'"
        )
    if not 0.0 <= cnn.dropout < 1.0:
        raise ValueError("model.encoder.cnn.dropout must be in [0, 1)")

    flatAction = config.decoder.flat_action
    coordinateDecoder = config.decoder.coordinate_conditioned
    if any(value <= 0 for value in flatAction.mlpLayers):
        raise ValueError("model.decoder.flat_action.mlpLayers values must be positive")
    if not 0.0 <= flatAction.dropout < 1.0:
        raise ValueError("model.decoder.flat_action.dropout must be in [0, 1)")
    if any(value <= 0 for value in coordinateDecoder.mlpLayers):
        raise ValueError("model.decoder.coordinate_conditioned.mlpLayers values must be positive")
    if not 0.0 <= coordinateDecoder.dropout < 1.0:
        raise ValueError("model.decoder.coordinate_conditioned.dropout must be in [0, 1)")

    transformer = config.encoder.transformer
    if transformer.patchSize <= 0:
        raise ValueError("model.encoder.transformer.patchSize must be positive")
    if transformer.embedDim <= 0:
        raise ValueError("model.encoder.transformer.embedDim must be positive")
    if transformer.depth <= 0:
        raise ValueError("model.encoder.transformer.depth must be positive")
    if transformer.numHeads <= 0:
        raise ValueError("model.encoder.transformer.numHeads must be positive")
    if transformer.mlpRatio <= 0:
        raise ValueError("model.encoder.transformer.mlpRatio must be positive")

    for feature in config.context.globalFeatures:
        if feature not in GLOBAL_CONTEXT_FEATURE_DIMS:
            raise ValueError(f"unknown model.context.globalFeatures entry: {feature}")
    for feature in config.context.elementFeatures:
        if feature not in ELEMENT_CONTEXT_FEATURE_DIMS:
            raise ValueError(f"unknown model.context.elementFeatures entry: {feature}")
    if config.decoder.type == "coordinate_conditioned" and len(config.context.elementFeatures) == 0:
        raise ValueError(
            "model.decoder.type='coordinate_conditioned' requires at least one model.context.elementFeatures entry"
        )


def modelConfigToDict(config: ModelConfig) -> dict[str, Any]:
    return {
        "architecture": config.architecture,
        "common": {
            "activation": config.common.activation,
            "norm": config.common.norm,
            "hiddenDim": config.common.hiddenDim,
            "logStdMode": config.common.logStdMode,
            "logStdInit": config.common.logStdInit,
        },
        "encoder": {
            "type": config.encoder.type,
            "cnn": {
                "inChannels": config.encoder.cnn.inChannels,
                "convChannels": list(config.encoder.cnn.convChannels),
                "kernelSizes": list(config.encoder.cnn.kernelSizes),
                "strides": list(config.encoder.cnn.strides),
                "paddings": list(config.encoder.cnn.paddings),
                "pooling": config.encoder.cnn.pooling,
                "dropout": config.encoder.cnn.dropout,
            },
            "transformer": {
                "patchSize": config.encoder.transformer.patchSize,
                "embedDim": config.encoder.transformer.embedDim,
                "depth": config.encoder.transformer.depth,
                "numHeads": config.encoder.transformer.numHeads,
                "mlpRatio": config.encoder.transformer.mlpRatio,
            },
        },
        "decoder": {
            "type": config.decoder.type,
            "flat_action": {
                "mlpLayers": list(config.decoder.flat_action.mlpLayers),
                "dropout": config.decoder.flat_action.dropout,
            },
            "coordinate_conditioned": {
                "mlpLayers": list(config.decoder.coordinate_conditioned.mlpLayers),
                "dropout": config.decoder.coordinate_conditioned.dropout,
            },
        },
        "context": {
            "globalFeatures": list(config.context.globalFeatures),
            "elementFeatures": list(config.context.elementFeatures),
        },
    }


def loadModelConfig(path: str | Path) -> ModelConfig:
    configPath = Path(path)
    with configPath.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return buildModelConfig(payload)
