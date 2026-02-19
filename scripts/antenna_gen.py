#scripts/antenna_gen.py
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np


c = 3e8 #speed of light
class antenna:
    def __init__(self, fc: float = 5.8e9,  shape: str = "URA", y_elements: int = 4, z_elements: int = 4, spacing: float = 0, dtype = torch.float32):
        self.fc = fc
        self.shape = shape.upper()
        self.y_elements = y_elements
        self.z_elements = z_elements
        self.n_elements = 0
        self.wavelength = c/fc

 
        if spacing:
            self.spacing = spacing
        else:
            self.spacing = 0.5 * (c/fc)

        self.y_span = (y_elements -1) * self.spacing
        self.z_span = (z_elements - 1) * self.spacing
        self.x_span = 0

        self.dtype = dtype

        self.localPosition = self.getLocalPosition()

    def getLocalPosition(self) -> torch.Tensor:
        y = torch.linspace((-0.5 * self.y_span), (0.5 * self.y_span), self.y_elements, dtype=self.dtype)
        z = torch.linspace((-0.5 * self.z_span), (0.5 * self.z_span), self.z_elements, dtype=self.dtype)
        y_grid, z_grid = torch.meshgrid(y,z, indexing="ij")
        y_grid, z_grid = y_grid.flatten(), z_grid.flatten()

        if self.shape in {"CIRCULAR", "UCA"}:
            # Round the rectangular grid into a disk by removing corner points.
            y_step = torch.abs(y[1] - y[0]) if self.y_elements > 1 else torch.tensor(0.0, dtype=self.dtype)
            z_step = torch.abs(z[1] - z[0]) if self.z_elements > 1 else torch.tensor(0.0, dtype=self.dtype)
            radius = 0.5 * min(self.y_span, self.z_span) + 0.5 * float(min(y_step, z_step))

            keep = (y_grid.pow(2) + z_grid.pow(2)) <= (radius ** 2)
            y_grid = y_grid[keep]
            z_grid = z_grid[keep]
        elif self.shape != "URA":
            raise ValueError(f"Unsupported array shape '{self.shape}'. Use 'URA' or 'CIRCULAR'/'UCA'.")

        self.n_elements = y_grid.numel()
        x_grid = torch.zeros_like(y_grid)
        return torch.stack((x_grid,y_grid,z_grid))


        
    def setRandomWeights(self) -> torch.Tensor:
        amplitude = torch.rand(self.n_elements)
        amplitude = amplitude / torch.norm(amplitude)
        phase = 2 * torch.pi * torch.rand(self.n_elements)
        self.weights = amplitude * torch.exp(1j * phase)
        return self.weights
    
    def setUniformWeights(self) -> torch.Tensor:
        amplitude = torch.ones(self.n_elements) / self.n_elements
        phase = torch.zeros(self.n_elements)
        self.weights = amplitude * torch.exp(1j*phase)
        return self.weights

    def getArrayResponse(self, azimuth: torch.Tensor, elevation: torch.Tensor) -> torch.Tensor: 
        azimuth, elevation = torch.broadcast_tensors(azimuth, elevation)
        inputShape = azimuth.shape

        coordinateMap = torch.stack([
            torch.cos(elevation) * torch.cos(azimuth), 
            torch.cos(elevation) * torch.sin(azimuth), 
            torch.sin(elevation) 
            ], dim=0) 
        
        waveVector = 2*torch.pi/self.wavelength * coordinateMap 
        waveVectorFlat = waveVector.reshape(3,-1)
        phase = self.localPosition.T @ waveVectorFlat
        arrayManifold = torch.exp(1j * phase)
        response = (self.weights.conj()[:,None] * arrayManifold).sum(dim=0)

        return response.reshape(inputShape)


    def plotArrayGeometry(self, plotWeights: bool = True) -> None:
        x, y, z = self.getLocalPosition().unbind(0)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=75, azim=-45, roll=45)
        
        if plotWeights:
            print(torch.abs(self.weights).size(), x.size())
            sc = ax.scatter(
                x.numpy(),
                y.numpy(),
                z.numpy(),
                c = torch.angle(self.weights).numpy(),
                s = torch.abs(self.weights).numpy() * 1000, 
                cmap="viridis"               
            )
            cbar = fig.colorbar(sc, ax=ax, pad=0.1, shrink = 0.5)
            cbar.set_label("Array Phase")
        else:
            ax.scatter(
                x.numpy(),
                y.numpy(),
                z.numpy(),
            )

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")


        ax.set_title("Antenna Array Geometry")
        fig.text(0.5, 0.875, f'X_Span = {self.x_span:.4f}', ha='center', va='top', fontsize=6)
        fig.text(0.5, 0.85,  f'Y_Span = {self.y_span:.4f}', ha='center', va='top', fontsize=6)
        fig.text(0.5, 0.825, f'Z_Span = {self.z_span:.4f}', ha='center', va='top', fontsize=6)

        ax.set_box_aspect((1, 1, 0.25))

        plt.show()

    def plotArrayFactor(self, projection:  matplotlib.projections = "polar", azimuthCutAngle: float = 0, elevationCutAngle: float = 0, xProjectionScale: float = 10, float = torch.pi, plotResolution: int = 500) -> None:        

        fig = plt.figure(figsize=(16, 7))        
    
        #AF vs azimuth
        azimuthCutAxis = torch.linspace(-torch.pi, torch.pi, 2 * plotResolution)
        azimuthResponse = self.getArrayResponse(azimuthCutAxis, torch.tensor([elevationCutAngle]))
        azimuthPower = azimuthResponse.abs() ** 2
        azimuthPowerdB = 10 * torch.log10(azimuthPower / azimuthPower.max().clamp_min(1e-12))

        ax1 = fig.add_subplot(131, projection=projection)
        if projection in {None, "rectilinear"}: ax1.plot(azimuthCutAxis.rad2deg(), azimuthPowerdB)
        else: ax1.plot(azimuthCutAxis, azimuthPowerdB)
        ax1.set_ylim(-100, 10)
        ax1.set_xlabel("Azimuth (deg)")
        ax1.set_title("TX Power (dB) Azimuth Cut")

        #AF vs elevation
        elevationCutAxis = torch.linspace(-torch.pi, torch.pi, 2 * plotResolution)
        elevationResponse = self.getArrayResponse(torch.tensor([azimuthCutAngle]), elevationCutAxis)
        elevationPower = elevationResponse.abs() ** 2
        elevationPowerdB = 10 * torch.log10(elevationPower / elevationPower.max().clamp_min(1e-12))

        ax2 = fig.add_subplot(132, projection=projection)
        if projection in {None, "rectilinear"}: ax2.plot(elevationCutAxis.rad2deg(), elevationPowerdB)
        else: ax2.plot(elevationCutAxis, elevationPowerdB)
        ax2.set_ylim(-100, 10)
        ax2.set_xlabel("Elevation (deg)")
        ax2.set_title("TX Power (dB) Elevation Cut")

        #Full Response Grid
        azimuthVector = torch.linspace(-torch.pi, torch.pi, 2 * plotResolution)
        elevationVector = torch.linspace(-torch.pi / 2, torch.pi / 2, plotResolution)     # unique elevation domain
        azimuthGrid, elevationGrid = torch.meshgrid(azimuthVector, elevationVector, indexing="ij")

        fullResponse = self.getArrayResponse(azimuthGrid, elevationGrid) 
        fullPower = fullResponse.abs() ** 2
        fullPowerNormalized = fullPower / fullPower.max().clamp_min(1e-12)
        fullPowerdB = 10*torch.log10(fullPower / fullPower.max().clamp_min(1e-12) )

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
        facecolors = cm.viridis(norm(C))
        surf = ax3.plot_surface(
            X, Y, Z,
            facecolors=facecolors,
            rstride=4, cstride=4,
            linewidth=0, antialiased=False, shade=False
        )
        mappable = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
        mappable.set_array([])
        fig.colorbar(mappable, ax=ax3, pad=0.1, shrink=0.3, label="Power (dB)")
        ax3.set_title("TX Power 3D Pattern")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_zlabel("z")
        ax3.set_xlim(-xProjectionScale*rmax, xProjectionScale*rmax)
        ax3.set_ylim(-rmax, rmax)
        ax3.set_zlim(-rmax, rmax)
        ax3.set_box_aspect((1, 1, 1))


        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    arr = antenna(fc=5.8e9, shape="UCA", y_elements=16, z_elements=16)
    arr.setUniformWeights()
    
    #arr.plotArrayGeometry()
    arr.plotArrayFactor()
