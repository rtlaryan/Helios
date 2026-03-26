from dataclasses import dataclass, field
from typing import Literal

import torch
from scripts.arrayBatch import ArrayBatch
from scripts.arraySimulation import arrayResponseBatch
from scripts.coordinateTransforms import mapLLAtoArrayAZEL
from scripts.targetSpec import TargetSpec

LossType = Literal["MSE", "HUBER"]  # kept for backward compat, unused internally


@dataclass
class LossConfig:
    """Configuration for the 4-term beamforming loss.

    Terms:
        shape  — KL divergence + cosine similarity blend (distribution shape match)
        eff    — power efficiency: fraction of power inside importance-weighted region
        psl    — peak sidelobe level penalty relative to weakest beam peak
        wide   — wide-area spill: power fraction outside the search area entirely

    All four terms are naturally O(1), so weights represent true relative importance.
    """

    w_shape: float = 1.0
    w_eff: float = 0.5
    w_psl: float = 0.5
    w_wide: float = 0.4

    # KL vs cosine blend in shape term (1.0 = pure KL, 0.0 = pure cosine)
    alpha: float = 0.7

    # PSL floor in the bounded ratio space: outsidePeak / (minZonePeak + outsidePeak).
    # 0.05 ≈ outsidePeak is 1/19 of min zone peak (~−26 dB), 0.09 ≈ −20 dB
    sidelobe_floor: float = 0.05

    # Gaussian sigma (degrees) used to compute per-zone peak power for PSL reference
    hotspot_sigma: float = 2.0

    # Wide-area scan: coarse grid resolution (wide_grid_size × wide_grid_size points)
    # over the full sphere (AZ ∈ [-π, π], EL ∈ [-π/2, π/2]).
    wide_grid_size: int = 256

    # Extra AZ/EL margin (radians) added to the search area bounding box before
    # classifying wide-grid cells as "inside".  Allows the beam's near-field
    # rolloff to stay outside the penalty region.  0.0 = strict search area boundary.
    wide_padding: float = 0.05


# ---------------------------------------------------------------------------
# Backward-compatible alias so existing notebooks/scripts don't break
# ---------------------------------------------------------------------------
LossParameters = LossConfig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_linear(powerMap: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Convert powerMap to linear scale.

    Detects dB scale by the presence of any negative values.
    Assumes normalized linear maps already lie in [0, 1].
    """
    if torch.any(powerMap < 0):
        # dB scale detected — convert to linear
        return torch.pow(10.0, powerMap / 10.0).clamp_min(eps)
    return powerMap.clamp_min(0.0)


def _as_distribution(tensor: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Normalise flat [N] or batched [B, N] tensor to a probability distribution."""
    return tensor / tensor.sum(dim=-1, keepdim=True).clamp_min(eps)


# ---------------------------------------------------------------------------
# Term 1: Shape Fidelity
# ---------------------------------------------------------------------------


def shapeFidelityLoss(
    linearResponse: torch.Tensor,
    target: TargetSpec,
    alpha: float = 0.7,
    eps: float = 1e-10,
) -> torch.Tensor:
    """KL divergence + cosine similarity blend between response and target distributions.

    Args:
        linearResponse: [B, H, W] raw linear power (unnormalised)
        target:         TargetSpec with powerMap [H, W]
        alpha:          weight given to KL term (1-alpha goes to cosine)
        eps:            numerical floor

    Returns:
        [B] loss — lower is better
    """
    B = linearResponse.shape[0]
    responseFlat = linearResponse.flatten(1)  # [B, N]

    targetLinear = _to_linear(target.powerMap, eps=eps).flatten()  # [N]
    P_resp = _as_distribution(responseFlat, eps=eps)  # [B, N]
    P_targ = _as_distribution(targetLinear.unsqueeze(0), eps=eps)  # [1, N]

    # --- KL(target || response) — penalises response mass missing from target ---
    # Guard: where P_targ == 0 the KL contribution is 0 by definition,
    # but 0 * log(0 / anything) = 0 * -inf = NaN in floating point.
    # Use torch.where so the log is never evaluated at zero-target cells.
    log_ratio = torch.where(
        P_targ > eps,
        torch.log((P_targ / P_resp.clamp_min(eps)).clamp_min(eps)),
        torch.zeros_like(P_targ),
    )
    kl = (P_targ * log_ratio).sum(dim=-1)  # [B]

    # --- 1 − cosine similarity ---
    dot = (P_resp * P_targ).sum(dim=-1)  # [B]
    norm_resp = P_resp.norm(dim=-1).clamp_min(eps)
    norm_targ = P_targ.norm(dim=-1).clamp_min(eps)
    cos_sim = dot / (norm_resp * norm_targ)
    one_minus_cos = 1.0 - cos_sim  # [B]

    return alpha * kl + (1.0 - alpha) * one_minus_cos


# ---------------------------------------------------------------------------
# Term 2: Power Efficiency
# ---------------------------------------------------------------------------


def powerEfficiencyLoss(
    linearResponse: torch.Tensor,
    target: TargetSpec,
    eps: float = 1e-10,
) -> torch.Tensor:
    """1 − (power inside importance-weighted region / total power).

    Uses the importanceMap as a soft mask, so the gradient is smooth and it
    works naturally for multi-beam (importanceMap has nonzero weight at all zones).

    Args:
        linearResponse: [B, H, W] raw linear power
        target:         TargetSpec with importanceMap [H, W] in [0, 1]

    Returns:
        [B] loss ∈ [0, 1] — lower is better
    """
    responseFlat = linearResponse.flatten(1)  # [B, N]

    # importanceMap is already normalised [0, 1]; use as soft mask directly
    softMask = target.importanceMap.flatten().clamp(0.0, 1.0).unsqueeze(0)  # [1, N]

    insidePower = (responseFlat * softMask).sum(dim=-1)           # [B]
    totalPower = responseFlat.sum(dim=-1).clamp_min(eps)           # [B]
    efficiency = insidePower / totalPower                           # [B] ∈ [0, 1]

    return 1.0 - efficiency


# ---------------------------------------------------------------------------
# Term 3: Peak Sidelobe Level (PSL)
# ---------------------------------------------------------------------------


def peakSidelobeLoss(
    linearResponse: torch.Tensor,
    target: TargetSpec,
    sidelobe_floor: float = 0.05,
    hotspot_sigma: float = 2.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Bounded peak sidelobe level (PSL) penalty.

    Computes  pslRatio = outsidePeak / (minZonePeak + outsidePeak) ∈ [0, 1]
    which is always bounded — avoids division blow-up when the global beam
    maximum is outside the target area.

    Interpretation:
      pslRatio → 0   all power inside the target footprint (ideal)
      pslRatio → 0.5 equal peak power inside and outside (random-like)
      pslRatio → 1   peak power entirely outside the target (off-target beam)

    For multi-beam the zone reference is the *weakest* beam peak, so a strong
    beam cannot mask sidelobe problems near a weaker one.

    Args:
        linearResponse: [B, H, W] raw linear power (any absolute scale)
        target:         TargetSpec with hotspotCoordinates [K, 2] and importanceMap
        sidelobe_floor: PSL ratio below which no penalty is applied.
                        0.05 ≈ outsidePeak is 1/19 of the weakest zone peak (~−26 dB).
                        0.09 ≈ −20 dB, 0.003 ≈ −30 dB.
        hotspot_sigma:  Gaussian half-width (deg) that defines each beam's footprint.
                        Cells where the Gaussian > 0.5 (≈ within 1.18σ) are treated
                        as inside that zone.

    Returns:
        [B] loss ∈ [0, 1] — lower is better.
    """
    responseFlat = linearResponse.flatten(1)          # [B, N]
    searchLat = target.searchLatitudes.flatten()      # [N]
    searchLon = target.searchLongitudes.flatten()     # [N]
    hotspots = target.hotspotCoordinates              # [K, 2]
    softMask = target.importanceMap.flatten().clamp(0.0, 1.0)   # [N]
    outsideMask = (1.0 - softMask).clamp(0.0, 1.0)             # [N]

    # Normalise response to peak=1. PSL is a purely relative metric — absolute
    # scale cancels — and normalisation keeps all values in [0, 1].
    globalPeak = responseFlat.amax(dim=-1, keepdim=True).clamp_min(eps)  # [B, 1]
    responseNorm = responseFlat / globalPeak                               # [B, N] ∈ [0, 1]

    # --- Per-zone peak: true maximum inside each hotspot's Gaussian footprint ---
    # Hard-threshold the Gaussian at 0.5 → binary footprint within ~1.18σ of centre.
    zonePeakList: list[torch.Tensor] = []
    for k in range(hotspots.shape[0]):
        latDiff = searchLat - hotspots[k, 0]
        lonDiff = searchLon - hotspots[k, 1]
        dist2 = latDiff.square() + lonDiff.square()
        gaussian = torch.exp(-0.5 * dist2 / (hotspot_sigma ** 2))   # [N] ∈ [0, 1]
        zoneMask = (gaussian > 0.5).to(responseNorm.dtype)           # binary [N]
        zonePeak = (responseNorm * zoneMask.unsqueeze(0)).amax(dim=-1)   # [B]
        zonePeakList.append(zonePeak)

    # Reference: weakest beam peak across all zones
    minZonePeak = torch.stack(zonePeakList, dim=-1).min(dim=-1).values  # [B]

    # --- Peak sidelobe: true maximum outside the importance-weighted region ---
    outsidePeak = (responseNorm * outsideMask.unsqueeze(0)).amax(dim=-1)  # [B] ∈ [0, 1]

    # --- Bounded PSL ratio ∈ [0, 1] ---
    # Using the sum in the denominator avoids blow-up when minZonePeak is small.
    denom = (minZonePeak + outsidePeak).clamp_min(eps)
    pslRatio = outsidePeak / denom                            # [B] ∈ [0, 1]
    loss = torch.relu(pslRatio - sidelobe_floor)             # [B] ∈ [0, 1]

    return loss


# ---------------------------------------------------------------------------
# Term 4: Wide-Area Spill
# ---------------------------------------------------------------------------


def wideAreaSpillLoss(
    batch: ArrayBatch,
    searchAZEL: tuple[torch.Tensor, torch.Tensor],
    grid_size: int = 128,
    padding: float = 0.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Penalise power radiated outside the search area on a full AZ/EL hemisphere scan.

    Scans the complete sphere AZ ∈ [-π, π], EL ∈ [-π/2, π/2] at coarse resolution
    to detect out-of-window spill that the PSL term (which only looks inside the
    search window) cannot see.

    The inside mask is **per-batch**: each array sample has a different pose, so the
    search area occupies different AZ/EL cells for each.  We compute the AZ/EL bounding
    box of the search area for each sample from the pre-computed *searchAZEL* tensors.

    Args:
        batch:      ArrayBatch of candidate configurations
        searchAZEL: (azimuth [B, N], elevation [B, N]) of the search-area points,
                    already computed by mapLLAtoArrayAZEL.  Passed in to avoid
                    redundant computation.
        grid_size:  Side length of the full-sphere scan grid (grid_size² total points).
                    128 → 16384 pts; 256 → 65536 pts.
        padding:    Extra margin (radians) added to the search area AZ/EL bounding box
                    before classifying cells as "inside".  Use this to give the beam's
                    near-field rolloff room to breathe without being penalised.
                    0.0 = strict search area boundary.

    Returns:
        [B] loss ∈ [0, 1] — lower is better.
    """
    B = batch.batchSize
    device = batch.elementLocalPosition.device
    dtype = batch.elementLocalPosition.dtype

    # --- Full-sphere AZ/EL grid (shared, not batched) ---
    wideAZ  = torch.linspace(-torch.pi, torch.pi,       grid_size, device=device, dtype=dtype)  # [G]
    wideEL  = torch.linspace(-torch.pi / 2, torch.pi / 2, grid_size, device=device, dtype=dtype)  # [G]
    wideAZgrid, wideELgrid = torch.meshgrid(wideAZ, wideEL, indexing="ij")  # [G, G]
    wideAZflat = wideAZgrid.reshape(-1)  # [G²]
    wideELflat = wideELgrid.reshape(-1)  # [G²]

    # --- Per-batch inside mask from search area AZ/EL bounding box ---
    searchAZ, searchEL = searchAZEL              # each [B, N]
    azMin = searchAZ.min(dim=-1).values - padding  # [B]
    azMax = searchAZ.max(dim=-1).values + padding  # [B]
    elMin = searchEL.min(dim=-1).values - padding  # [B]
    elMax = searchEL.max(dim=-1).values + padding  # [B]

    # Broadcast: [B, 1] vs [G²] → [B, G²]
    inAZ = (wideAZflat.unsqueeze(0) >= azMin.unsqueeze(1)) & \
           (wideAZflat.unsqueeze(0) <= azMax.unsqueeze(1))
    inEL = (wideELflat.unsqueeze(0) >= elMin.unsqueeze(1)) & \
           (wideELflat.unsqueeze(0) <= elMax.unsqueeze(1))
    insideMask = (inAZ & inEL).to(dtype)         # [B, G²]

    # --- Full-sphere response: pass shared [G, G] grid directly ---
    wideResponse = arrayResponseBatch(batch, (wideAZgrid, wideELgrid))  # [B, G, G]
    wideFlat = wideResponse.flatten(1)                                    # [B, G²]

    # --- Bounded spill ratio ∈ [0, 1] ---
    insidePower  = (wideFlat * insideMask).sum(dim=-1)              # [B]
    outsidePower = (wideFlat * (1.0 - insideMask)).sum(dim=-1)      # [B]
    return outsidePower / (insidePower + outsidePower).clamp_min(eps)  # [B]


# ---------------------------------------------------------------------------
# Combined batch loss
# ---------------------------------------------------------------------------


def batchLoss(
    batch: ArrayBatch,
    target: TargetSpec,
    params: LossConfig,
    lossType: LossType = "MSE",  # unused; kept for API compatibility
    logTerms: bool = False,
) -> torch.Tensor:
    """Evaluate the 4-term beamforming loss for a batch of array configurations.

    Args:
        batch:    ArrayBatch of candidate phased-array weight configurations
        target:   TargetSpec defining the desired beam pattern
        params:   LossConfig with per-term weights and hyperparameters
        lossType: legacy argument — ignored
        logTerms: if True, print per-term contributions for debugging

    Returns:
        [B] scalar loss — lower is better
    """
    target = target.to(batch.device, batch.dtype)
    targetAZEL = mapLLAtoArrayAZEL(batch, target.targetCoordinates)
    linearResponse = arrayResponseBatch(batch, targetAZEL)  # [B, H, W] linear power

    shapeTerm = shapeFidelityLoss(linearResponse, target, alpha=params.alpha)
    effTerm = powerEfficiencyLoss(linearResponse, target)
    pslTerm = peakSidelobeLoss(
        linearResponse,
        target,
        sidelobe_floor=params.sidelobe_floor,
        hotspot_sigma=params.hotspot_sigma,
    )
    wideTerm = wideAreaSpillLoss(
        batch,
        searchAZEL=targetAZEL,
        grid_size=params.wide_grid_size,
        padding=params.wide_padding,
    )

    if logTerms:
        print(
            f"shape={params.w_shape * shapeTerm.mean().item():.4f} | "
            f"efficiency={params.w_eff * effTerm.mean().item():.4f} | "
            f"psl={params.w_psl * pslTerm.mean().item():.4f} | "
            f"wide={params.w_wide * wideTerm.mean().item():.4f}"
        )

    loss = (
        params.w_shape * shapeTerm
        + params.w_eff * effTerm
        + params.w_psl * pslTerm
        + params.w_wide * wideTerm
    )
    return loss
