import torch

def LLAtoECEF(LLAPosition: torch.Tensor) -> torch.Tensor:
    latitude, longitude, altitude = LLAPosition.unbind(dim=-1)
    latitude, longitude = latitude.deg2rad(), longitude.deg2rad()
    semiMajorAxis = 6378137.0
    flattening = 1.0 / 298.257223563
    e2 = flattening * (2.0 - flattening)

    sinLatitude = torch.sin(latitude)
    cosLatitude = torch.cos(latitude)
    sinLongitude = torch.sin(longitude)
    cosLongitude = torch.cos(longitude)

    primeRadius = semiMajorAxis / torch.sqrt(1.0 - e2 * sinLatitude.square())

    x = (primeRadius + altitude) * cosLatitude * cosLongitude
    y = (primeRadius + altitude) * cosLatitude * sinLongitude
    z = ((1.0 - e2) * primeRadius + altitude) * sinLatitude

    return torch.stack([x, y, z], dim=1)





