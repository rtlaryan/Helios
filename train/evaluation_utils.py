from __future__ import annotations

from typing import Any, Literal

import torch

ObjectiveVersion = Literal["v1", "v2"]

LOSS_TERM_KEYS_V1: tuple[str, ...] = ("shape", "efficiency", "wideSupport")
LOSS_TERM_KEYS_V2: tuple[str, ...] = ("shape", "coverage", "shell", "global", "peak")


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
        hasattr(evaluation, "coverageLoss")
        and hasattr(evaluation, "shellLoss")
        and hasattr(evaluation, "globalLoss")
        and hasattr(evaluation, "peakLoss")
    ):
        return {
            "shape": evaluation.shapeLoss,
            "coverage": evaluation.coverageLoss,
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


def evaluation_loss_record(evaluation: Any, index: int) -> dict[str, float]:
    record = {"total": float(evaluation.totalLoss[index].item())}
    for name, tensor in evaluation_loss_tensors(evaluation).items():
        record[name] = float(tensor[index].item())
    return record
