from __future__ import annotations

from typing import Any, Literal

import torch

ObjectiveVersion = Literal["v1", "v2"]

LOSS_TERM_KEYS_V1: tuple[str, ...] = ("shape", "efficiency", "wideSupport")
LOSS_TERM_KEYS_V2: tuple[str, ...] = ("shape", "efficiency", "nullPenalty", "shell", "global", "peak")


def loss_term_keys_for_objective(objectiveVersion: ObjectiveVersion) -> tuple[str, ...]:
    if objectiveVersion == "v1":
        return LOSS_TERM_KEYS_V1
    if objectiveVersion == "v2":
        return LOSS_TERM_KEYS_V2
    raise ValueError(f"unsupported objectiveVersion: {objectiveVersion}")


def evaluation_loss_tensors(evaluation: Any) -> dict[str, torch.Tensor]:
    if hasattr(evaluation, "efficiencyLoss") and hasattr(evaluation, "wideSupportLoss"):
        return {
            "shape": evaluation.shapeLoss,
            "efficiency": evaluation.efficiencyLoss,
            "wideSupport": evaluation.wideSupportLoss,
        }
    if (
        hasattr(evaluation, "efficiencyLoss")
        and hasattr(evaluation, "nullPenaltyLoss")
        and hasattr(evaluation, "shellLoss")
        and hasattr(evaluation, "globalLoss")
        and hasattr(evaluation, "peakLoss")
    ):
        return {
            "shape": evaluation.shapeLoss,
            "efficiency": evaluation.efficiencyLoss,
            "nullPenalty": evaluation.nullPenaltyLoss,
            "shell": evaluation.shellLoss,
            "global": evaluation.globalLoss,
            "peak": evaluation.peakLoss,
        }
    raise TypeError(f"unsupported evaluation payload: {type(evaluation)!r}")


def evaluation_loss_means(evaluation: Any) -> dict[str, float]:
    return {
        name: float(tensor.mean().item())
        for name, tensor in evaluation_loss_tensors(evaluation).items()
    }


def evaluation_weighted_loss_means(evaluation: Any, lossParams: Any) -> dict[str, float]:
    if hasattr(evaluation, "wideSupportLoss"):
        weightNames = {
            "shape": "w_shape",
            "efficiency": "w_eff",
            "wideSupport": "w_wide",
        }
    else:
        weightNames = {
            "shape": "w_shape",
            "efficiency": "w_efficiency",
            "nullPenalty": "w_null_penalty",
            "shell": "w_shell",
            "global": "w_global",
            "peak": "w_peak",
        }
    weighted: dict[str, float] = {}
    for name, tensor in evaluation_loss_tensors(evaluation).items():
        weight = float(getattr(lossParams, weightNames[name], 1.0))
        weighted[name] = weight * float(tensor.mean().item())
    return weighted


def evaluation_diagnostic_means(evaluation: Any) -> dict[str, float]:
    diagnostics = getattr(evaluation, "diagnostics", None)
    if diagnostics is None:
        return {}
    return {
        name: float(tensor.mean().item())
        for name, tensor in diagnostics.items()
    }


def evaluation_loss_record(evaluation: Any, index: int) -> dict[str, float]:
    record = {"total": float(evaluation.totalLoss[index].item())}
    for name, tensor in evaluation_loss_tensors(evaluation).items():
        record[name] = float(tensor[index].item())
    return record
