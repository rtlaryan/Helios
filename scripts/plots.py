import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import torch

from .arrayBatch import ArrayBatch
from .arraySimulation import arrayResponseBatch, arrayResponseSample, normalizePower, todB
from .coordinateTransforms import mapLLAtoArrayAZEL
from .targetSpec import TargetLike, fetchTargetSample


def _render_figure(fig) -> None:
    try:
        from IPython import get_ipython
        from IPython.display import display

        ipython = get_ipython()
        if ipython is not None and getattr(ipython, "kernel", None) is not None:
            display(fig)
            plt.close(fig)
            return
    except Exception:
        pass

    plt.show()


def plotArrayGeometry(batch: ArrayBatch, sampleID: int = 0, plotWeights: bool = True) -> None:
    x, y, z = (batch.elementLocalPosition[sampleID] * 1000).cpu().unbind(0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=10, azim=-8, roll=1)

    if plotWeights:
        weights = batch.weights[sampleID].cpu()
        sc = ax.scatter(
            x.numpy(),
            y.numpy(),
            z.numpy(),  # type: ignore
            c=torch.angle(weights).numpy(),
            s=torch.abs(weights).numpy() * 1000,  # type: ignore
            cmap="viridis",
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.5)
        cbar.set_label("Array Phase")
    else:
        ax.scatter(
            x.numpy(),
            y.numpy(),
            z.numpy(),  # type: ignore
        )

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_xticks([])

    ax.set_title("Antenna Array Geometry")
    ax.set_box_aspect((0.1, 1, 1))

    _render_figure(fig)


def plotArrayFactor(
    batch: ArrayBatch,
    sampleID: int = 0,
    projection: str | None = "polar",
    plot3d: bool = True,
    azimuthCutAngle: float = 0.0,
    elevationCutAngle: float = 0.0,
    xProjectionScale: float = 10.0,
    plotResolution: int = 500,
    plotRange: float = 2 * torch.pi,
    stride: int = 4,
) -> None:
    fig = plt.figure(figsize=(10, 5))

    # -------------------------
    # AF vs azimuth
    # -------------------------
    azimuthCutAxis = torch.linspace(
        -plotRange / 2, plotRange / 2, 2 * plotResolution, device=batch.device, dtype=batch.dtype
    )
    elevationCutTensor = torch.tensor([elevationCutAngle], device=batch.device, dtype=batch.dtype)
    azimuthResponse = arrayResponseSample(batch, sampleID, azimuthCutAxis, elevationCutTensor)

    ax1 = fig.add_subplot(131, projection=projection)
    azimuthCutAxis_cpu = azimuthCutAxis.detach().cpu().numpy()
    azimuthResponse_cpu = azimuthResponse.detach().cpu().numpy()
    if projection in {None, "rectilinear"}:
        ax1.plot(torch.as_tensor(azimuthCutAxis_cpu).rad2deg().numpy(), azimuthResponse_cpu)
        ax1.set_xlabel("Azimuth (deg)")
    else:
        ax1.plot(azimuthCutAxis_cpu, azimuthResponse_cpu)
        ax1.set_xlabel("Azimuth (rad)")
    ax1.set_ylim(-100, 10)
    ax1.set_title("TX Power (dB) Azimuth Cut")

    # -------------------------
    # AF vs elevation
    # -------------------------
    elevationCutAxis = torch.linspace(
        -plotRange / 2, plotRange / 2, 2 * plotResolution, device=batch.device, dtype=batch.dtype
    )
    azimuthCutTensor = torch.tensor([azimuthCutAngle], device=batch.device, dtype=batch.dtype)
    elevationResponse = arrayResponseSample(batch, sampleID, azimuthCutTensor, elevationCutAxis)

    ax2 = fig.add_subplot(132, projection=projection)
    elevationCutAxis_cpu = elevationCutAxis.detach().cpu().numpy()
    elevationResponse_cpu = elevationResponse.detach().cpu().numpy()
    if projection in {None, "rectilinear"}:
        ax2.plot(torch.as_tensor(elevationCutAxis_cpu).rad2deg().numpy(), elevationResponse_cpu)
        ax2.set_xlabel("Elevation (deg)")
    else:
        ax2.plot(elevationCutAxis_cpu, elevationResponse_cpu)
        ax2.set_xlabel("Elevation (rad)")
    ax2.set_ylim(-100, 10)
    ax2.set_title("TX Power (dB) Elevation Cut")

    # -------------------------
    # 3D pattern
    # -------------------------
    if plot3d:
        azimuthVector = torch.linspace(
            -torch.pi, torch.pi, 2 * plotResolution, device=batch.device, dtype=batch.dtype
        )
        elevationVector = torch.linspace(
            -torch.pi / 2, torch.pi / 2, plotResolution, device=batch.device, dtype=batch.dtype
        )
        azimuthGrid, elevationGrid = torch.meshgrid(azimuthVector, elevationVector, indexing="ij")

        # Compute once; reuse pmax; avoid extra temporaries; avoid CPU scalar tensors
        fullResponse = arrayResponseSample(batch, sampleID, azimuthGrid, elevationGrid, dB=False)
        fullPower = fullResponse.abs().square()
        pmax = fullPower.max().clamp_min(1e-12)

        fullPowerNormalized = fullPower / pmax
        fullPowerdB = 10.0 * torch.log10(fullPowerNormalized.clamp_min(1e-12))

        floordB = -40.0
        maskdB = fullPowerdB < floordB

        # Mask on-device; clamp for numerical safety; radius = sqrt(normalized power)
        fullPowerNormalized = torch.where(
            maskdB, fullPowerNormalized.new_zeros(()), fullPowerNormalized
        )
        R = fullPowerNormalized.clamp_min(0.0).sqrt()

        # Precompute trig once (saves repeated cos/sin calls)
        ce = torch.cos(elevationGrid)
        se = torch.sin(elevationGrid)
        ca = torch.cos(azimuthGrid)
        sa = torch.sin(azimuthGrid)

        # Convert only what matplotlib needs (single device->CPU transfer per tensor)
        X = (R * ce * ca).detach().cpu().numpy()
        Y = (R * ce * sa).detach().cpu().numpy()
        Z = (R * se).detach().cpu().numpy()
        C = fullPowerdB.detach().cpu().numpy()
        rmax = R.max().item() / 10.0

        ax3 = fig.add_subplot(133, projection="3d")
        norm = colors.Normalize(vmin=floordB, vmax=0.0)
        facecolors = cm.viridis(norm(C))  # type: ignore

        ax3.plot_surface(
            X,
            Y,
            Z,
            facecolors=facecolors,
            rstride=stride,
            cstride=stride,
            linewidth=0,
            antialiased=False,
            shade=False,
        )

        mappable = cm.ScalarMappable(norm=norm, cmap=cm.viridis)  # type: ignore
        mappable.set_array([])
        fig.colorbar(mappable, ax=ax3, pad=0.1, shrink=0.3, label="Power (dB)")

        ax3.set_title("TX Power 3D Pattern")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_zlabel("z")
        ax3.set_xticks([-1, 0.0, 1])
        ax3.set_yticks([-1 / xProjectionScale, 0.0, 1 / xProjectionScale])
        ax3.set_zticks([-1 / xProjectionScale, 0.0, 1 / xProjectionScale])  # type: ignore
        ax3.set_xlim(-xProjectionScale * rmax, xProjectionScale * rmax)
        ax3.set_ylim(-rmax, rmax)
        ax3.set_zlim(-rmax, rmax)
        ax3.set_box_aspect((1, 1, 1))

    plt.tight_layout()
    _render_figure(fig)


def projectResponseOnTarget(
    batch: ArrayBatch,
    target: TargetLike,
    sampleID: int = 0,
    normalizedInputs: bool = False,
    responseTensor: torch.Tensor | None = None,
):
    sampleBatch = batch.fetch(sampleID)
    sampleTarget = fetchTargetSample(target, sampleID).to(batch.device, batch.dtype)

    if responseTensor is None:
        targetAZEL = mapLLAtoArrayAZEL(sampleBatch, sampleTarget.targetCoordinates)
        batchResponse = arrayResponseBatch(sampleBatch, targetAZEL, dB=False, normalize=False)
        responseTensor = batchResponse[0]

    if responseTensor.shape != sampleTarget.targetShape:
        responseTensor = responseTensor.reshape(sampleTarget.targetShape)

    if normalizedInputs:
        targetPowerTensor = todB(sampleTarget.powerMap.clamp_min(1e-12))
        responseDB = todB(normalizePower(responseTensor.unsqueeze(0)))[0]
    else:
        targetPowerTensor = sampleTarget.powerMap
        responseDB = todB(normalizePower(responseTensor.unsqueeze(0)))[0]

    targetPower = targetPowerTensor.detach().cpu().numpy()
    responsePower = responseDB.detach().cpu().numpy()
    errorPower = responsePower - targetPower
    latitudes = sampleTarget.searchLatitudes.detach().cpu().numpy()
    longitudes = sampleTarget.searchLongitudes.detach().cpu().numpy()

    vmin = min(float(targetPowerTensor.min().item()), float(responseDB.min().item()))
    vmax = max(float(targetPowerTensor.max().item()), float(responseDB.max().item()))
    errorLimit = max(abs(float(errorPower.min())), abs(float(errorPower.max())), 1e-6)
    powerLabel = "Normalized Power (dB)" if normalizedInputs else "Power (dB)"
    targetTitle = (
        "Target Power Map (Normalized Input)" if normalizedInputs else "Target Power Map (dB)"
    )
    responseTitle = (
        f"Normalized Array Response (Sample {sampleID})"
        if normalizedInputs
        else f"Array Response (Sample {sampleID})"
    )
    errorTitle = "Normalized Response Error (dB)" if normalizedInputs else "Response Error (dB)"
    fullAzimuth = torch.linspace(-torch.pi, torch.pi, 361, device=batch.device, dtype=batch.dtype)
    fullElevation = torch.linspace(
        -torch.pi / 2, torch.pi / 2, 181, device=batch.device, dtype=batch.dtype
    )
    fullAzimuthGrid, fullElevationGrid = torch.meshgrid(fullAzimuth, fullElevation, indexing="ij")
    fullResponse = arrayResponseSample(
        batch,
        sampleID,
        fullAzimuthGrid,
        fullElevationGrid,
        dB=True,
        normalize=True,
    )
    fullResponsePower = fullResponse.detach().cpu().numpy()
    fullAzimuthDeg = torch.rad2deg(fullAzimuth).detach().cpu().numpy()
    fullElevationDeg = torch.rad2deg(fullElevation).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(14, 3), constrained_layout=True)

    targetIm = axes[0].imshow(
        targetPower,
        origin="lower",
        aspect="auto",
        extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()],
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    axes[0].set_title(targetTitle)
    axes[0].set_xlabel("Longitude (deg)")
    axes[0].set_ylabel("Latitude (deg)")
    fig.colorbar(targetIm, ax=axes[0], pad=0.02, label=powerLabel)

    responseIm = axes[1].imshow(
        responsePower,
        origin="lower",
        aspect="auto",
        extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()],
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    axes[1].set_title(responseTitle)
    axes[1].set_xlabel("Longitude (deg)")
    axes[1].set_ylabel("Latitude (deg)")
    fig.colorbar(responseIm, ax=axes[1], pad=0.02, label=powerLabel)

    errorIm = axes[2].imshow(
        errorPower,
        origin="lower",
        aspect="auto",
        extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()],
        vmin=-errorLimit,
        vmax=errorLimit,
        cmap="coolwarm",
    )
    axes[2].set_title(errorTitle)
    axes[2].set_xlabel("Longitude (deg)")
    axes[2].set_ylabel("Latitude (deg)")
    fig.colorbar(errorIm, ax=axes[2], pad=0.02, label="Error (dB)")

    fullResponseIm = axes[3].imshow(
        fullResponsePower.T,
        origin="lower",
        aspect="auto",
        extent=[
            fullAzimuthDeg.min(),
            fullAzimuthDeg.max(),
            fullElevationDeg.min(),
            fullElevationDeg.max(),
        ],
        vmin=-40.0,
        vmax=0.0,
        cmap="viridis",
    )
    axes[3].set_title(f"Full Array Response (dB, Sample {sampleID})")
    axes[3].set_xlabel("Azimuth (deg)")
    axes[3].set_ylabel("Elevation (deg)")
    fig.colorbar(fullResponseIm, ax=axes[3], pad=0.02, label="Power (dB)")

    _render_figure(fig)
