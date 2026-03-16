import torch

from .arrayBatch import ArrayBatch

semiMajorAxis = 6378137.0
flattening = 1.0 / 298.257223563
e2 = flattening * (2.0 - flattening)


def LLAtoECEF(LLAPosition: torch.Tensor) -> torch.Tensor:
    """
    LLAPosition: [..., 3] = [Latitude_deg, Longitude_deg, Altitude_m]
    returns:  [..., 3] = [x, y, z] meters (ECEF)
    """
    latitude, longitude, altitude = LLAPosition.unbind(dim=-1)
    latitude, longitude = latitude.deg2rad(), longitude.deg2rad()

    sinLatitude, cosLatitude = torch.sin(latitude), torch.cos(latitude)
    sinLongitude, cosLongitude = torch.sin(longitude), torch.cos(longitude)

    primeRadius = semiMajorAxis / torch.sqrt(1.0 - e2 * sinLatitude.square())

    x = (primeRadius + altitude) * cosLatitude * cosLongitude
    y = (primeRadius + altitude) * cosLatitude * sinLongitude
    z = ((1.0 - e2) * primeRadius + altitude) * sinLatitude

    return torch.stack([x, y, z], dim=-1)


def getECEFtoENUMapping(LLABatch: torch.Tensor) -> torch.Tensor:
    """
    LLABatch: [B, 3] = [Latitude_deg, Longitude_deg, Altitude_m]
    returns:  [B, 3, 3] rotation matrix mapping ECEF -> ENU

    Rows are the ENU basis vectors expressed in ECEF:
      R[b,0,:] = East
      R[b,1,:] = North
      R[b,2,:] = Up
    """
    latitude_deg, longitude_deg, _altitude_m = LLABatch.unbind(dim=-1)
    latitude = latitude_deg.deg2rad()
    longitude = longitude_deg.deg2rad()

    sinLatitude, cosLatitude = torch.sin(latitude), torch.cos(latitude)
    sinLongitude, cosLongitude = torch.sin(longitude), torch.cos(longitude)
    z = torch.zeros_like(latitude)

    east = torch.stack([-sinLongitude, cosLongitude, z], dim=-1)  # [B,3]
    north = torch.stack(
        [-sinLatitude * cosLongitude, -sinLatitude * sinLongitude, cosLatitude], dim=-1
    )
    up = torch.stack([cosLatitude * cosLongitude, cosLatitude * sinLongitude, sinLatitude], dim=-1)

    return torch.stack([east, north, up], dim=1)  # [B,3,3]


def mapLLAtoArrayAZEL(
    batch: ArrayBatch, targetCoordinates: torch.Tensor, toDeg: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    batch.ECEFPosition: [B,3]
    batch.LLAPosition: [B,3]
    targetCoordinates: [..., 2] or [B, ..., 2] = [Latitude_deg, Longitude_deg]
      or [..., 3] / [B, ..., 3] if altitude is provided.

    returns:
        azimuth: [B, ...]
        elevation: [B, ...]
    """

    batchSize = batch.batchSize
    # Ensure targetCoordinates is LLA with altitude column
    if targetCoordinates.shape[-1] == 2:
        zeros = torch.zeros_like(targetCoordinates[..., :1])
        targetLLA = torch.cat([targetCoordinates, zeros], dim=-1)
    elif targetCoordinates.shape[-1] == 3:
        targetLLA = targetCoordinates
    else:
        raise ValueError(
            "targetCoordinates must have shape [..., 2] or [..., 3] (got shape {})".format(
                targetCoordinates.shape
            )
        )
    targetECEF = LLAtoECEF(targetLLA)
    rotationMatrix = getECEFtoENUMapping(batch.LLAPosition)

    # Normalize and Flatten Inputs
    if targetECEF.shape[0] == batchSize:
        originalShape = targetECEF.shape[1:-1]
        flatECEF = targetECEF.reshape(batchSize, -1, 3)
    else:
        originalShape = targetECEF.shape[:-1]
        flatECEF = targetECEF.reshape(1, -1, 3).expand(batchSize, -1, 3)

    # Get ENU coordinates WRT Array Position
    directionVector = flatECEF - batch.ECEFPosition[:, None, :]
    east, north, up = torch.einsum("bij,bpj->bpi", rotationMatrix, directionVector).unbind(dim=-1)
    x, y, z = -up, east, north

    azimuth = torch.atan2(y, x)
    rho = torch.hypot(x, y).clamp_min(1e-12)
    elevation = torch.atan2(z, rho)

    # Reshape back to [B, ...]
    azimuth = azimuth.reshape(batchSize, *originalShape)
    elevation = elevation.reshape(batchSize, *originalShape)

    if toDeg:
        azimuth = torch.rad2deg(azimuth)
        elevation = torch.rad2deg(elevation)

    return azimuth, elevation
