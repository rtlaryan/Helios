import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")

# Ensure project root is importable when this test is run directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.phasedArray_old as phasedArray_old


def test_localPosition_ura_shape_matches_element_count():
    arr = phasedArray_old.antenna(shape="URA", y_elements=4, z_elements=5)
    localPos = arr.getLocalPosition()
    assert localPos.shape == (3, arr.y_elements * arr.z_elements)


def test_localPosition_circular_masks_corners():
    arr = phasedArray_old.antenna(shape="circular", y_elements=5, z_elements=5)
    localPos = arr.getLocalPosition()
    assert localPos.shape[0] == 3
    assert localPos.shape[1] < arr.y_elements * arr.z_elements


def test_localPosition_circular_even_grid_not_overtrimmed():
    arr = phasedArray_old.antenna(shape="circular", y_elements=6, z_elements=6)
    localPos = arr.getLocalPosition()
    assert localPos.shape[1] > 16


def test_geodeticToECEF_equator_prime_meridian():
    arr = phasedArray_old.antenna()
    ecef = arr.geodeticToECEF(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))
    assert torch.isclose(ecef[0], torch.tensor(phasedArray_old.semiMajorAxis), atol=1e-3)
    assert torch.isclose(ecef[1], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(ecef[2], torch.tensor(0.0), atol=1e-6)


def test_toAntennaLocalFrame_axes_at_equator_prime_meridian():
    arr = phasedArray_old.antenna(llaPosition=torch.tensor([0.0, 0.0, 0.0]))

    up_ecef = torch.tensor([1.0, 0.0, 0.0])
    east_ecef = torch.tensor([0.0, 1.0, 0.0])
    north_ecef = torch.tensor([0.0, 0.0, 1.0])

    up_local = arr.toAntennaLocalFrame(up_ecef)
    east_local = arr.toAntennaLocalFrame(east_ecef)
    north_local = arr.toAntennaLocalFrame(north_ecef)

    assert torch.allclose(up_local, torch.tensor([-1.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(east_local, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
    assert torch.allclose(north_local, torch.tensor([0.0, 0.0, 1.0]), atol=1e-6)


def test_getLandRadianceAngles_nadir_is_zero_az_el():
    arr = phasedArray_old.antenna(llaPosition=torch.tensor([0.0, 0.0, 1000.0]))
    azimuth, elevation = arr.getLandRadianceAngles(torch.tensor(0.0), torch.tensor(0.0))
    assert torch.isclose(azimuth, torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(elevation, torch.tensor(0.0), atol=1e-5)


def test_getLandRadianceAngles_degree_output_matches_radians():
    arr = phasedArray_old.antenna(llaPosition=torch.tensor([10.0, -20.0, 1000.0]))
    targetLat = torch.tensor([10.0, 11.0])
    targetLon = torch.tensor([-20.0, -19.0])

    az_rad, el_rad = arr.getLandRadianceAngles(targetLat, targetLon, outputDeg=False)
    az_deg, el_deg = arr.getLandRadianceAngles(targetLat, targetLon, outputDeg=True)

    assert torch.allclose(az_deg, torch.rad2deg(az_rad), atol=1e-4)
    assert torch.allclose(el_deg, torch.rad2deg(el_rad), atol=1e-4)


def test_getLandRadiancePower_shape_and_normalization():
    arr = phasedArray_old.antenna(llaPosition=torch.tensor([10.0, -20.0, 1000.0]))
    arr.setUniformWeights()

    lat = torch.tensor([[10.0, 10.2], [10.4, 10.6]])
    lon = torch.tensor([[-20.0, -19.8], [-19.6, -19.4]])
    power = arr.getLandRadiancePower(lat, lon, normalize=True)

    assert power.shape == lat.shape
    assert torch.isfinite(power).all()
    assert torch.isclose(power.max(), torch.tensor(1.0), atol=1e-6)
    assert (power >= 0).all()


def test_getLandRadiancePower_stride_reduces_grid_size():
    arr = phasedArray_old.antenna(llaPosition=torch.tensor([10.0, -20.0, 1000.0]))
    arr.setUniformWeights()

    lat = torch.arange(0, 24, dtype=torch.float32).reshape(6, 4)
    lon = torch.arange(0, 24, dtype=torch.float32).reshape(6, 4)

    power = arr.getLandRadiancePower(lat, lon, latitudeStride=2, longitudeStride=2, normalize=True)
    assert power.shape == (3, 2)


def test_getArrayResponse_chunking_matches_full_compute():
    arr = phasedArray_old.antenna(llaPosition=torch.tensor([10.0, -20.0, 1000.0]), y_elements=12, z_elements=12)
    arr.setUniformWeights()

    az = torch.linspace(-0.8, 0.8, 97)
    el = torch.linspace(-0.5, 0.5, 97)
    full = arr.getArrayResponse(az, el)
    chunked = arr.getArrayResponse(az, el, chunkSize=11)

    assert torch.allclose(chunked, full, atol=1e-5, rtol=1e-5)


def test_getLandRadiancePower_compute_dtype_controls_output_precision():
    arr = phasedArray_old.antenna(llaPosition=torch.tensor([10.0, -20.0, 1000.0]), dtype=torch.float64)
    arr.setUniformWeights()

    lat = torch.tensor([[10.0, 10.2], [10.4, 10.6]], dtype=torch.float64)
    lon = torch.tensor([[-20.0, -19.8], [-19.6, -19.4]], dtype=torch.float64)
    power = arr.getLandRadiancePower(lat, lon, normalize=True, computeDtype=torch.float32, chunkSize=2)

    assert power.dtype == torch.float32
    assert torch.isclose(power.max(), torch.tensor(1.0, dtype=torch.float32), atol=1e-6)


def test_setDirectedWeights_steers_toward_target():
    arr = phasedArray_old.antenna(llaPosition=torch.tensor([33.0, -83.0, 36000.0]), y_elements=6, z_elements=6)
    weights = arr.setDirectedWeights(torch.tensor(33.5), torch.tensor(-82.5))

    expected_amp = torch.ones(arr.n_elements) / arr.n_elements
    assert torch.allclose(weights.abs(), expected_amp, atol=1e-6)

    az_target, el_target = arr.getLandRadianceAngles(torch.tensor(33.5), torch.tensor(-82.5))
    response_target = arr.getArrayResponse(az_target, el_target).abs()
    response_opposite = arr.getArrayResponse(az_target + torch.pi, -el_target).abs()

    assert response_target > response_opposite


def test_plotLandRadianceOnMap_returns_fig_ax_without_show():
    arr = phasedArray_old.antenna(llaPosition=torch.tensor([10.0, -20.0, 1000.0]))
    arr.setUniformWeights()

    lat = torch.tensor([[10.0, 10.2], [10.4, 10.6]])
    lon = torch.tensor([[-20.0, -19.8], [-19.6, -19.4]])
    bitmap = np.zeros((2, 2), dtype=np.float32)

    fig, ax = arr.plotLandRadianceOnMap(
        mapBitmap=bitmap,
        targetLatitudes=lat,
        targetLongitudes=lon,
        show=False,
    )

    assert fig is not None
    assert ax is not None


def test_plotLandRadianceOnMap_keeps_axes_at_map_extent_when_array_outside():
    arr = phasedArray_old.antenna(llaPosition=torch.tensor([0.0, -83.0, 1000.0]))
    arr.setUniformWeights()

    lat = torch.tensor([[30.0, 30.0], [35.0, 35.0]])
    lon = torch.tensor([[-85.0, -80.0], [-85.0, -80.0]])
    bitmap = np.zeros((2, 2), dtype=np.float32)

    _, ax = arr.plotLandRadianceOnMap(
        mapBitmap=bitmap,
        targetLatitudes=lat,
        targetLongitudes=lon,
        show=False,
    )

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    assert np.isclose(min(xlim), -85.0)
    assert np.isclose(max(xlim), -80.0)
    assert np.isclose(min(ylim), 30.0)
    assert np.isclose(max(ylim), 35.0)


def test_plotLandRadianceOnMap_zoomOutFactor_expands_extent_around_map_center():
    arr = phasedArray_old.antenna(llaPosition=torch.tensor([33.0, -83.0, 1000.0]))
    arr.setUniformWeights()

    lat = torch.tensor([[30.0, 30.0], [35.0, 35.0]])
    lon = torch.tensor([[-85.0, -80.0], [-85.0, -80.0]])
    bitmap = np.zeros((2, 2), dtype=np.float32)

    _, ax = arr.plotLandRadianceOnMap(
        mapBitmap=bitmap,
        targetLatitudes=lat,
        targetLongitudes=lon,
        zoomOutFactor=2.0,
        show=False,
    )

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    assert np.isclose(min(xlim), -87.5)
    assert np.isclose(max(xlim), -77.5)
    assert np.isclose(min(ylim), 27.5)
    assert np.isclose(max(ylim), 37.5)


def test_plotLandRadianceOnMap_focus_centered_view():
    arr = phasedArray_old.antenna(llaPosition=torch.tensor([33.0, -83.0, 1000.0]))
    arr.setUniformWeights()

    lat = torch.tensor([[30.0, 30.0], [35.0, 35.0]])
    lon = torch.tensor([[-85.0, -80.0], [-85.0, -80.0]])
    bitmap = np.zeros((2, 2), dtype=np.float32)

    _, ax = arr.plotLandRadianceOnMap(
        mapBitmap=bitmap,
        targetLatitudes=lat,
        targetLongitudes=lon,
        zoomOutFactor=2.0,
        focusLatitude=33.0,
        focusLongitude=-83.0,
        show=False,
    )

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    assert np.isclose(min(xlim), -88.0)
    assert np.isclose(max(xlim), -78.0)
    assert np.isclose(min(ylim), 28.0)
    assert np.isclose(max(ylim), 38.0)


def test_plotLandRadianceOnMap_zoomOut_recomputes_larger_radiance_grid():
    arr = phasedArray_old.antenna(llaPosition=torch.tensor([33.0, -83.0, 1000.0]))
    arr.setUniformWeights()

    lat = torch.tensor([[30.0, 30.0], [35.0, 35.0]])
    lon = torch.tensor([[-85.0, -80.0], [-85.0, -80.0]])
    bitmap = np.zeros((2, 2), dtype=np.float32)

    _, ax_base = arr.plotLandRadianceOnMap(
        mapBitmap=bitmap,
        targetLatitudes=lat,
        targetLongitudes=lon,
        zoomOutFactor=1.0,
        show=False,
    )
    base_overlay_shape = ax_base.images[-1].get_array().shape

    _, ax_zoom = arr.plotLandRadianceOnMap(
        mapBitmap=bitmap,
        targetLatitudes=lat,
        targetLongitudes=lon,
        zoomOutFactor=2.0,
        show=False,
    )
    zoom_overlay_shape = ax_zoom.images[-1].get_array().shape

    assert zoom_overlay_shape[0] > base_overlay_shape[0]
    assert zoom_overlay_shape[1] > base_overlay_shape[1]


def test_plotLandRadianceOnMap_stride_reduces_overlay_resolution():
    arr = phasedArray_old.antenna(llaPosition=torch.tensor([33.0, -83.0, 1000.0]))
    arr.setUniformWeights()

    lat = torch.tensor([[30.0, 30.0], [35.0, 35.0]])
    lon = torch.tensor([[-85.0, -80.0], [-85.0, -80.0]])
    bitmap = np.zeros((2, 2), dtype=np.float32)

    _, ax_full = arr.plotLandRadianceOnMap(
        mapBitmap=bitmap,
        targetLatitudes=lat,
        targetLongitudes=lon,
        zoomOutFactor=4.0,
        latitudeStride=1,
        longitudeStride=1,
        show=False,
    )
    full_shape = ax_full.images[-1].get_array().shape

    _, ax_strided = arr.plotLandRadianceOnMap(
        mapBitmap=bitmap,
        targetLatitudes=lat,
        targetLongitudes=lon,
        zoomOutFactor=4.0,
        latitudeStride=4,
        longitudeStride=4,
        show=False,
    )
    strided_shape = ax_strided.images[-1].get_array().shape

    assert strided_shape[0] < full_shape[0]
    assert strided_shape[1] < full_shape[1]


def test_plotLandRadianceOnMap_map_origin_tracks_latitude_row_order():
    arr = phasedArray_old.antenna(llaPosition=torch.tensor([33.0, -83.0, 1000.0]))
    arr.setUniformWeights()
    bitmap = np.zeros((2, 2), dtype=np.float32)

    lat_ascending = torch.tensor([[30.0, 30.0], [35.0, 35.0]])
    lon = torch.tensor([[-85.0, -80.0], [-85.0, -80.0]])
    _, ax_asc = arr.plotLandRadianceOnMap(
        mapBitmap=bitmap,
        targetLatitudes=lat_ascending,
        targetLongitudes=lon,
        show=False,
    )
    assert ax_asc.images[0].origin == "lower"

    lat_descending = torch.tensor([[35.0, 35.0], [30.0, 30.0]])
    _, ax_desc = arr.plotLandRadianceOnMap(
        mapBitmap=bitmap,
        targetLatitudes=lat_descending,
        targetLongitudes=lon,
        show=False,
    )
    assert ax_desc.images[0].origin == "upper"
