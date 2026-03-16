# scripts/antenna_gen.py

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch

c = 3e8  # speed of light
semiMajorAxis = 6378137.0
flattening = 1.0 / 298.257223563
matData = scipy.io.loadmat("GAmap")
GALatitudes, GALongitudes = (
    -1 * torch.tensor(matData["GA"]["lat"][0][0], dtype=torch.float32),
    torch.tensor(matData["GA"]["lon"][0][0], dtype=torch.float32),
)
GAMap = matData["GA"]["data"][0][0]


class antenna:
    def __init__(
        self,
        fc: float = 5.8e9,
        shape: str = "UCA",
        llaPosition: torch.Tensor = torch.tensor([0, -83, 3.6e7]),
        y_elements: int = 4,
        z_elements: int = 4,
        spacing: float = 0,
        dtype=torch.float32,
    ):
        self.fc = fc
        self.shape = shape.upper()
        self.y_elements = y_elements
        self.z_elements = z_elements
        self.n_elements = 0
        self.wavelength = c / fc

        if spacing:
            self.spacing = spacing
        else:
            self.spacing = 0.5 * (c / fc)

        self.x_span = 0
        self.y_span = (y_elements - 1) * self.spacing
        self.z_span = (z_elements - 1) * self.spacing

        self.latitude = llaPosition[0]
        self.longitude = llaPosition[1]
        self.altitude = llaPosition[2]

        self.dtype = dtype

        self.localPosition = self.getLocalPosition()
        self.ECEF = self.geodeticToECEF(self.latitude, self.longitude, self.altitude)

    def getLocalPosition(self) -> torch.Tensor:
        y = torch.linspace(
            (-0.5 * self.y_span), (0.5 * self.y_span), self.y_elements, dtype=self.dtype
        )
        z = torch.linspace(
            (-0.5 * self.z_span), (0.5 * self.z_span), self.z_elements, dtype=self.dtype
        )
        y_grid, z_grid = torch.meshgrid(y, z, indexing="ij")
        y_grid, z_grid = y_grid.flatten(), z_grid.flatten()

        if self.shape in {"CIRCULAR", "UCA"}:
            # Round the rectangular grid into a disk by removing corner points.
            y_step = (
                torch.abs(y[1] - y[0])
                if self.y_elements > 1
                else torch.tensor(0.0, dtype=self.dtype)
            )
            z_step = (
                torch.abs(z[1] - z[0])
                if self.z_elements > 1
                else torch.tensor(0.0, dtype=self.dtype)
            )
            radius = 0.5 * min(self.y_span, self.z_span) + 0.5 * float(min(y_step, z_step))

            keep = (y_grid.pow(2) + z_grid.pow(2)) <= (radius**2)
            y_grid = y_grid[keep]
            z_grid = z_grid[keep]
        elif self.shape != "URA":
            raise ValueError(
                f"Unsupported array shape '{self.shape}'. Use 'URA' or 'CIRCULAR'/'UCA'."
            )

        self.n_elements = y_grid.numel()
        x_grid = torch.zeros_like(y_grid)
        return torch.stack((x_grid, y_grid, z_grid))

    def setRandomWeights(self) -> torch.Tensor:
        amplitude = torch.rand(self.n_elements)
        amplitude = amplitude / torch.norm(amplitude)
        phase = 2 * torch.pi * torch.rand(self.n_elements)
        self.weights = amplitude * torch.exp(1j * phase)
        return self.weights

    def setUniformWeights(self) -> torch.Tensor:
        amplitude = torch.ones(self.n_elements) / self.n_elements
        phase = torch.zeros(self.n_elements)
        self.weights = amplitude * torch.exp(1j * phase)
        return self.weights

    def setDirectedWeights(
        self, targetLatitude: torch.Tensor, targetLongitude: torch.Tensor
    ) -> torch.Tensor:
        targetLatitude = torch.as_tensor(targetLatitude, dtype=self.dtype)
        targetLongitude = torch.as_tensor(targetLongitude, dtype=self.dtype)
        if targetLatitude.numel() != 1 or targetLongitude.numel() != 1:
            raise ValueError(
                "setDirectedWeights expects scalar targetLatitude and targetLongitude."
            )
        targetECEF = self.geodeticToECEF(targetLatitude.reshape(()), targetLongitude.reshape(()))
        directionECEF = targetECEF - self.ECEF
        directionLocal = self.toAntennaLocalFrame(directionECEF)
        directionLocal = directionLocal / torch.norm(directionLocal).clamp_min(1e-12)

        # Local frame is [down, east, north]. Positive local x points toward Earth (nadir).
        if float(directionLocal[0]) <= 0:
            raise ValueError("Target direction is not in the Earth-facing (+x/down) hemisphere.")

        waveVector = (2 * torch.pi / self.wavelength) * directionLocal.reshape(3, 1)
        phase = (self.localPosition.T @ waveVector).squeeze(-1)

        amplitude = torch.ones(self.n_elements, dtype=self.dtype) / self.n_elements
        self.weights = amplitude * torch.exp(1j * phase)
        return self.weights

    @staticmethod
    def _complex_dtype_for(real_dtype: torch.dtype) -> torch.dtype:
        if real_dtype == torch.float64:
            return torch.complex128
        return torch.complex64

    def getArrayResponse(
        self,
        azimuth: torch.Tensor,
        elevation: torch.Tensor,
        chunkSize: int | None = None,
        computeDtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if chunkSize is not None and chunkSize <= 0:
            raise ValueError("chunkSize must be greater than 0 when provided.")

        realDtype = self.dtype if computeDtype is None else computeDtype
        complexDtype = self._complex_dtype_for(realDtype)

        azimuth = torch.as_tensor(azimuth, dtype=realDtype)
        elevation = torch.as_tensor(elevation, dtype=realDtype)
        azimuth, elevation = torch.broadcast_tensors(azimuth, elevation)
        inputShape = azimuth.shape

        xyzToSpherical = torch.stack(
            [
                torch.cos(elevation) * torch.cos(azimuth),
                torch.cos(elevation) * torch.sin(azimuth),
                torch.sin(elevation),
            ],
            dim=0,
        )

        waveVector = 2 * torch.pi / self.wavelength * xyzToSpherical
        waveVectorFlat = waveVector.reshape(3, -1)
        nPoints = waveVectorFlat.shape[1]
        if chunkSize is None:
            chunkSize = nPoints

        localPositionT = self.localPosition.T.to(dtype=realDtype)
        weightsConj = self.weights.to(dtype=complexDtype).conj()
        response = torch.empty(nPoints, dtype=complexDtype)

        for start in range(0, nPoints, chunkSize):
            end = min(start + chunkSize, nPoints)
            phaseChunk = localPositionT @ waveVectorFlat[:, start:end]
            arrayManifoldChunk = torch.exp(1j * phaseChunk)
            response[start:end] = (weightsConj[:, None] * arrayManifoldChunk).sum(dim=0)

        return response.reshape(inputShape)

    def geodeticToECEF(
        self,
        latitude: torch.Tensor,
        longitude: torch.Tensor,
        altitude: torch.Tensor = torch.tensor(0),
        degrees: bool = True,
    ) -> torch.Tensor:
        latitude, longitude, altitude = torch.broadcast_tensors(latitude, longitude, altitude)
        if degrees:
            latitude, longitude = latitude.deg2rad(), longitude.deg2rad()

        e2 = flattening * (2.0 - flattening)
        sinLatitude = torch.sin(latitude)
        cosLatitude = torch.cos(latitude)
        sinLongitude = torch.sin(longitude)
        cosLongitude = torch.cos(longitude)

        primeRadius = semiMajorAxis / torch.sqrt(1.0 - e2 * sinLatitude.pow(2))

        x = (primeRadius + altitude) * cosLatitude * cosLongitude
        y = (primeRadius + altitude) * cosLatitude * sinLongitude
        z = ((1.0 - e2) * primeRadius + altitude) * sinLatitude

        return torch.stack((x, y, z), dim=0)

    def toAntennaLocalFrame(self, vectorECEF: torch.Tensor) -> torch.Tensor:
        latitudeRad = self.latitude.deg2rad()
        longitudeRad = self.longitude.deg2rad()
        sinLatitude, cosLatitude = torch.sin(latitudeRad), torch.cos(latitudeRad)
        sinLongitude, cosLongitude = torch.sin(longitudeRad), torch.cos(longitudeRad)

        ecefToENU = torch.stack(
            [
                torch.stack([-sinLongitude, cosLongitude, torch.zeros_like(cosLatitude)]),
                torch.stack(
                    [-sinLatitude * cosLongitude, -sinLatitude * sinLongitude, cosLatitude]
                ),
                torch.stack([cosLatitude * cosLongitude, cosLatitude * sinLongitude, sinLatitude]),
            ]
        )

        vectorFlat = vectorECEF.reshape(3, -1)
        vectorENU = ecefToENU @ vectorFlat
        vectorENU = vectorENU.reshape_as(vectorECEF)

        down = -vectorENU[2]
        east = vectorENU[0]
        north = vectorENU[1]
        return torch.stack([down, east, north], dim=0)

    def getLandRadianceAngles(
        self, targetLatitudes: torch.Tensor, targetLongitudes: torch.Tensor, outputDeg: bool = False
    ) -> tuple:
        mapECEF = self.geodeticToECEF(targetLatitudes, targetLongitudes)
        ecefOrigin = self.ECEF.view(3, *([1] * (mapECEF.ndim - 1)))
        directionVectorECEF = mapECEF - ecefOrigin
        directionVectorLocal = self.toAntennaLocalFrame(directionVectorECEF)
        x, y, z = directionVectorLocal[0], directionVectorLocal[1], directionVectorLocal[2]
        r = torch.sqrt(x * x + y * y + z * z).clamp_min(1e-12)

        azimuth = torch.atan2(y, x)
        elevation = torch.asin(z / r)

        if outputDeg:
            azimuth = torch.rad2deg(azimuth)
            elevation = torch.rad2deg(elevation)

        return azimuth, elevation

    def getLandRadiancePower(
        self,
        targetLatitudes: torch.Tensor = GALatitudes,
        targetLongitudes: torch.Tensor = GALongitudes,
        latitudeStride: int = 1,
        longitudeStride: int = 1,
        normalize: bool = True,
        chunkSize: int | None = None,
        computeDtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if latitudeStride <= 0 or longitudeStride <= 0:
            raise ValueError("latitudeStride and longitudeStride must be greater than 0.")

        realDtype = self.dtype if computeDtype is None else computeDtype
        targetLatitudes = torch.as_tensor(targetLatitudes, dtype=realDtype)
        targetLongitudes = torch.as_tensor(targetLongitudes, dtype=realDtype)

        if targetLatitudes.ndim >= 2 and targetLongitudes.ndim >= 2:
            targetLatitudes = targetLatitudes[::latitudeStride, ::longitudeStride]
            targetLongitudes = targetLongitudes[::latitudeStride, ::longitudeStride]
        elif targetLatitudes.ndim >= 1 and targetLongitudes.ndim >= 1:
            stride = max(latitudeStride, longitudeStride)
            targetLatitudes = targetLatitudes[::stride]
            targetLongitudes = targetLongitudes[::stride]

        azimuth, elevation = self.getLandRadianceAngles(
            targetLatitudes, targetLongitudes, outputDeg=False
        )
        response = self.getArrayResponse(
            azimuth,
            elevation,
            chunkSize=chunkSize,
            computeDtype=realDtype,
        )
        power = response.abs().pow(2)

        if normalize:
            power = power / power.max().clamp_min(1e-12)

        return power

    def plotLandRadianceOnMap(
        self,
        mapBitmap: np.ndarray | None = None,
        targetLatitudes: torch.Tensor = GALatitudes,
        targetLongitudes: torch.Tensor = GALongitudes,
        useDB: bool = True,
        floorDB: float = -40.0,
        cmap: str = "inferno",
        alpha: float = 0.65,
        zoomOutFactor: float = 1.0,
        focusLatitude: float | None = None,
        focusLongitude: float | None = None,
        latitudeStride: int = 1,
        longitudeStride: int = 1,
        chunkSize: int | None = None,
        computeDtype: torch.dtype | None = None,
        showArrayLocation: bool = True,
        showFocusLocation: bool = False,
        show: bool = True,
    ):
        if latitudeStride <= 0 or longitudeStride <= 0:
            raise ValueError("latitudeStride and longitudeStride must be greater than 0.")

        latNP = targetLatitudes.detach().cpu().numpy()
        lonNP = targetLongitudes.detach().cpu().numpy()
        lonMin, lonMax = float(lonNP.min()), float(lonNP.max())
        latMin, latMax = float(latNP.min()), float(latNP.max())
        mapExtent = [lonMin, lonMax, latMin, latMax]
        if zoomOutFactor <= 0:
            raise ValueError("zoomOutFactor must be greater than 0.")

        if focusLatitude is None:
            centerLat = 0.5 * (latMin + latMax)
        else:
            centerLat = float(focusLatitude)

        if focusLongitude is None:
            centerLon = 0.5 * (lonMin + lonMax)
        else:
            centerLon = float(focusLongitude)

        halfLatSpan = 0.5 * (latMax - latMin) * zoomOutFactor
        halfLonSpan = 0.5 * (lonMax - lonMin) * zoomOutFactor
        viewLatMin, viewLatMax = centerLat - halfLatSpan, centerLat + halfLatSpan
        viewLonMin, viewLonMax = centerLon - halfLonSpan, centerLon + halfLonSpan

        # Rebuild a computation grid over the full displayed view window.
        latVecFromGrid = np.unique(latNP)
        lonVecFromGrid = np.unique(lonNP)
        dLat = (
            float(np.min(np.diff(np.sort(latVecFromGrid))))
            if latVecFromGrid.size > 1
            else (latMax - latMin)
        )
        dLon = (
            float(np.min(np.diff(np.sort(lonVecFromGrid))))
            if lonVecFromGrid.size > 1
            else (lonMax - lonMin)
        )
        if dLat <= 0:
            dLat = max((latMax - latMin) / max(1, latNP.shape[0] - 1), 1e-3)
        if dLon <= 0:
            dLon = max((lonMax - lonMin) / max(1, lonNP.shape[1] - 1), 1e-3)

        nLatBase = max(2, int(round((viewLatMax - viewLatMin) / dLat)) + 1)
        nLonBase = max(2, int(round((viewLonMax - viewLonMin) / dLon)) + 1)
        nLat = max(2, (nLatBase - 1) // latitudeStride + 1)
        nLon = max(2, (nLonBase - 1) // longitudeStride + 1)
        realDtype = self.dtype if computeDtype is None else computeDtype
        latGridVec = torch.linspace(viewLatMin, viewLatMax, nLat, dtype=realDtype)
        lonGridVec = torch.linspace(viewLonMin, viewLonMax, nLon, dtype=realDtype)
        computeLatGrid, computeLonGrid = torch.meshgrid(latGridVec, lonGridVec, indexing="ij")

        power = self.getLandRadiancePower(
            computeLatGrid,
            computeLonGrid,
            latitudeStride=1,
            longitudeStride=1,
            normalize=True,
            chunkSize=chunkSize,
            computeDtype=realDtype,
        )
        if useDB:
            overlayData = 10.0 * torch.log10(power.clamp_min(1e-12))
            overlayData = torch.maximum(overlayData, torch.tensor(floorDB, dtype=overlayData.dtype))
            vmin, vmax = floorDB, 0.0
            colorbarLabel = "Radiance (dB)"
        else:
            overlayData = power
            vmin, vmax = 0.0, 1.0
            colorbarLabel = "Normalized Radiance"

        overlayNP = overlayData.detach().cpu().numpy()
        overlayExtent = [viewLonMin, viewLonMax, viewLatMin, viewLatMax]

        fig, ax = plt.subplots(figsize=(10, 8))

        if mapBitmap is not None:
            # Align bitmap row direction to the provided latitude grid.
            # If first row has larger latitude than last row, data is north-to-south (origin="upper").
            if latNP.ndim >= 2:
                firstRowLat = float(latNP[0, 0])
                lastRowLat = float(latNP[-1, 0])
            else:
                firstRowLat = float(latNP[0])
                lastRowLat = float(latNP[-1])
            mapOrigin = "upper" if firstRowLat > lastRowLat else "lower"

            ax.imshow(
                mapBitmap, extent=mapExtent, origin=mapOrigin, cmap="gray", interpolation="nearest"
            )

        im = ax.imshow(
            overlayNP,
            extent=overlayExtent,
            origin="lower",
            cmap=cmap,
            alpha=alpha,
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(im, ax=ax, label=colorbarLabel)

        if showArrayLocation:
            arrayLon = float(self.longitude)
            arrayLat = float(self.latitude)
            inBounds = (viewLonMin <= arrayLon <= viewLonMax) and (
                viewLatMin <= arrayLat <= viewLatMax
            )
            if inBounds:
                ax.scatter(
                    arrayLon,
                    arrayLat,
                    c="cyan",
                    s=40,
                    marker="x",
                    label="Array",
                )
                ax.legend(loc="upper right")

        if showFocusLocation and (focusLatitude is not None) and (focusLongitude is not None):
            focusLon = float(focusLongitude)
            focusLat = float(focusLatitude)
            inBounds = (viewLonMin <= focusLon <= viewLonMax) and (
                viewLatMin <= focusLat <= viewLatMax
            )
            if inBounds:
                ax.scatter(
                    focusLon,
                    focusLat,
                    c="lime",
                    s=36,
                    marker="+",
                    label="Focus",
                )
                ax.legend(loc="upper right")

        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.set_title("Antenna Land Radiance over GA Map")
        # Lock to requested view window around either map center or focus point.
        ax.set_xlim(viewLonMin, viewLonMax)
        ax.set_ylim(viewLatMin, viewLatMax)

        if show:
            plt.show()

        return fig, ax

    def plotArrayGeometry(self, plotWeights: bool = True) -> None:
        x, y, z = self.getLocalPosition().unbind(0)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # ax.view_init(elev=75, azim=-45, roll=45)

        if plotWeights:
            print(torch.abs(self.weights).size(), x.size())
            sc = ax.scatter(
                x.numpy(),
                y.numpy(),
                z.numpy(),  # type: ignore
                c=torch.angle(self.weights).numpy(),
                s=torch.abs(self.weights).numpy() * 1000,  # type: ignore
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

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_xticks([])

        ax.set_title("Antenna Array Geometry")
        fig.text(0.5, 0.875, f"X_Span = {self.x_span:.4f}", ha="center", va="top", fontsize=6)
        fig.text(0.5, 0.85, f"Y_Span = {self.y_span:.4f}", ha="center", va="top", fontsize=6)
        fig.text(0.5, 0.825, f"Z_Span = {self.z_span:.4f}", ha="center", va="top", fontsize=6)

        ax.set_box_aspect((0.1, 1, 1))

        plt.show()

    def plotArrayFactor(
        self,
        projection: matplotlib.projections = "polar",
        plot3d: bool = True,
        azimuthCutAngle: float = 0.0,
        elevationCutAngle: float = 0.0,
        xProjectionScale: float = 10.0,
        plotResolution: int = 500,
        plotRange: float = 2 * torch.pi,
    ) -> None:  # type: ignore
        fig = plt.figure(figsize=(16, 7))

        # AF vs azimuth
        azimuthCutAxis = torch.linspace(-plotRange / 2, plotRange / 2, 2 * plotResolution)
        azimuthResponse = self.getArrayResponse(azimuthCutAxis, torch.tensor([elevationCutAngle]))
        azimuthPower = azimuthResponse.abs() ** 2
        azimuthPowerdB = 10 * torch.log10(azimuthPower / azimuthPower.max().clamp_min(1e-12))

        ax1 = fig.add_subplot(131, projection=projection)
        if projection in {None, "rectilinear"}:
            ax1.plot(azimuthCutAxis.rad2deg(), azimuthPowerdB)
        else:
            ax1.plot(azimuthCutAxis, azimuthPowerdB)
        ax1.set_ylim(-100, 10)
        ax1.set_xlabel("Azimuth (deg)")
        ax1.set_title("TX Power (dB) Azimuth Cut")

        # AF vs elevation
        elevationCutAxis = torch.linspace(-plotRange / 2, plotRange / 2, 2 * plotResolution)
        elevationResponse = self.getArrayResponse(torch.tensor([azimuthCutAngle]), elevationCutAxis)
        elevationPower = elevationResponse.abs() ** 2
        elevationPowerdB = 10 * torch.log10(elevationPower / elevationPower.max().clamp_min(1e-12))

        ax2 = fig.add_subplot(132, projection=projection)
        if projection in {None, "rectilinear"}:
            ax2.plot(elevationCutAxis.rad2deg(), elevationPowerdB)
        else:
            ax2.plot(elevationCutAxis, elevationPowerdB)
        ax2.set_ylim(-100, 10)
        ax2.set_xlabel("Elevation (deg)")
        ax2.set_title("TX Power (dB) Elevation Cut")

        if plot3d:
            # Full Response Grid
            azimuthVector = torch.linspace(-torch.pi, torch.pi, 2 * plotResolution)
            elevationVector = torch.linspace(
                -torch.pi / 2, torch.pi / 2, plotResolution
            )  # unique elevation domain
            azimuthGrid, elevationGrid = torch.meshgrid(
                azimuthVector, elevationVector, indexing="ij"
            )

            fullResponse = self.getArrayResponse(azimuthGrid, elevationGrid)
            fullPower = fullResponse.abs() ** 2
            fullPowerNormalized = fullPower / fullPower.max().clamp_min(1e-12)
            fullPowerdB = 10 * torch.log10(fullPower / fullPower.max().clamp_min(1e-12))

            floordB = -40.0
            maskdB = fullPowerdB < floordB
            fullPowerNormalized = torch.where(maskdB, torch.tensor(0.0), fullPowerNormalized)

            R = fullPowerNormalized.sqrt()
            X = (R * torch.cos(elevationGrid) * torch.cos(azimuthGrid)).numpy()
            Y = (R * torch.cos(elevationGrid) * torch.sin(azimuthGrid)).numpy()
            Z = (R * torch.sin(elevationGrid)).numpy()
            C = fullPowerdB.numpy()
            rmax = np.max(np.abs(np.stack([X, Y, Z]))) / 10

            ax3 = fig.add_subplot(133, projection="3d")
            norm = colors.Normalize(vmin=-40, vmax=0)
            facecolors = cm.viridis(norm(C))  # type: ignore
            surf = ax3.plot_surface(
                X,
                Y,
                Z,
                facecolors=facecolors,
                rstride=4,
                cstride=4,
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
            ax3.set_xlim(-xProjectionScale * rmax, xProjectionScale * rmax)
            ax3.set_ylim(-rmax, rmax)
            ax3.set_zlim(-rmax, rmax)
            ax3.set_box_aspect((1, 1, 1))

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    arr = antenna(fc=30e9, shape="UCA", y_elements=8, z_elements=8)
    print(arr.ECEF)
    arr.setUniformWeights()
    # arr.setDirectedWeights(torch.tensor(33), torch.tensor(-83))
    arr.plotArrayFactor(projection=None, plot3d=True)
    # fig, ax = arr.plotLandRadianceOnMap(
    #     mapBitmap=GAMap,
    #     targetLatitudes=GALatitudes,
    #     targetLongitudes=GALongitudes,
    #     zoomOutFactor=2.0,  # show 2x wider/taller region
    #     focusLatitude=33.5,  # center view around target
    #     focusLongitude=-82.5,
    #     latitudeStride=16,
    #     longitudeStride=16,
    #     showFocusLocation=True,
    #     show=True,
    # )
