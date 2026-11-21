from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import routing
from mapeo_ruta.config import (
    DEFAULT_DESTINATION,
    DEFAULT_ORIGIN,
    get_default_scenario,
)
from mapeo_ruta.copernicus_client import generate_simulated_netcdf
from mapeo_ruta.ocean_data import OceanCurrentField, load_ocean_currents
from mapeo_ruta.route_optimizer import snap_point_to_water

client = TestClient(app)


def _default_request() -> dict[str, object]:
    return {
        "origin": {"lat": DEFAULT_ORIGIN.lat, "lon": DEFAULT_ORIGIN.lon},
        "destination": {"lat": DEFAULT_DESTINATION.lat, "lon": DEFAULT_DESTINATION.lon},
    }


@pytest.mark.parametrize(
    "body",
    [
        {"origin": {"lat": 95, "lon": 0}, "destination": {"lat": 0, "lon": 0}},
        {"origin": {"lat": 0, "lon": -200}, "destination": {"lat": 1, "lon": 1}},
        {"origin": {"lat": 0, "lon": 0}},
        {},
    ],
)
def test_invalid_requests_return_422(body: dict[str, object]) -> None:
    assert client.post("/api/v1/routes", json=body).status_code == 422


def test_same_point_and_excessive_span_return_422() -> None:
    same = {"origin": {"lat": 20, "lon": -86}, "destination": {"lat": 20, "lon": -86}}
    far = {"origin": {"lat": 20, "lon": -86}, "destination": {"lat": 24, "lon": -86}}
    assert client.post("/api/v1/routes", json=same).status_code == 422
    assert client.post("/api/v1/routes", json=far).status_code == 422


def test_synthetic_route_returns_lon_lat_geojson(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = generate_simulated_netcdf(get_default_scenario(), output_path=tmp_path / "synthetic.nc")
    monkeypatch.setattr(routing, "acquire_ocean_data", lambda **kwargs: result)
    monkeypatch.setattr(routing, "crosses_land", lambda coords, field: False)
    response = client.post("/api/v1/routes", json=_default_request())
    assert response.status_code == 200, response.text
    payload = response.json()
    route = payload["route"]
    assert route["type"] == "LineString"
    assert len(route["coordinates"]) >= 2
    assert all(len(coord) == 2 and -180 <= coord[0] <= 180 and -90 <= coord[1] <= 90 for coord in route["coordinates"])
    smoothing = payload["meta"]["smoothing"]
    assert smoothing["applied"] is True
    assert smoothing["method"] == "cubic_bspline_clamped"
    assert len(route["coordinates"]) > smoothing["raw_points"]
    field = load_ocean_currents(result.netcdf_path, target_datetime=get_default_scenario().target_datetime)
    origin_snap = snap_point_to_water(field, DEFAULT_ORIGIN, "Origen")
    destination_snap = snap_point_to_water(field, DEFAULT_DESTINATION, "Destino")
    assert route["coordinates"][0][0] == pytest.approx(origin_snap.node_lon, abs=1e-6)
    assert route["coordinates"][0][1] == pytest.approx(origin_snap.node_lat, abs=1e-6)
    assert route["coordinates"][-1][0] == pytest.approx(destination_snap.node_lon, abs=1e-6)
    assert route["coordinates"][-1][1] == pytest.approx(destination_snap.node_lat, abs=1e-6)
    assert "SIMULADOS" in payload["meta"]["data_source"]


def test_synthetic_land_guard_keeps_raw_route(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = generate_simulated_netcdf(get_default_scenario(), output_path=tmp_path / "synthetic.nc")
    monkeypatch.setattr(routing, "acquire_ocean_data", lambda **kwargs: result)
    field = load_ocean_currents(result.netcdf_path, target_datetime=get_default_scenario().target_datetime)
    land_row, land_col = np.argwhere(~field.water_mask)[0]
    land_point = [float(field.lons[land_col]), float(field.lats[land_row])]
    original_smoother = routing.cubic_bspline

    def append_land_point(coords: list[list[float]], samples: int) -> list[list[float]]:
        return [*original_smoother(coords, samples), land_point]

    monkeypatch.setattr(routing.settings, "smoothing", False)
    raw_response = client.post("/api/v1/routes", json=_default_request())
    assert raw_response.status_code == 200, raw_response.text
    raw_coordinates = raw_response.json()["route"]["coordinates"]

    monkeypatch.setattr(routing.settings, "smoothing", True)
    monkeypatch.setattr(routing, "cubic_bspline", append_land_point)
    guarded_response = client.post("/api/v1/routes", json=_default_request())
    assert guarded_response.status_code == 200, guarded_response.text
    guarded = guarded_response.json()
    assert guarded["meta"]["smoothing"]["applied"] is False
    assert "tierra" in guarded["meta"]["smoothing"]["reason"]
    assert guarded["route"]["coordinates"] == raw_coordinates


def test_blocked_field_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routing, "acquire_ocean_data", lambda **kwargs: type("Acq", (), {
        "netcdf_path": Path("unused.nc"), "source_label": "DATOS SINTÉTICOS"
    })())
    lats = np.array([20.70, 20.80, 20.90])
    lons = np.array([-86.90, -86.80, -86.70])
    values = np.full((3, 3), np.nan)
    values[0, 0] = 0.0
    values[2, 2] = 0.0
    field = OceanCurrentField(
        lats, lons, values.copy(), values.copy(), "2026-11-20T12:00:00", 0.5,
        {"latitude": 3, "longitude": 3}, ["uo", "vo"], True,
    )
    monkeypatch.setattr(routing, "load_ocean_currents", lambda *args, **kwargs: field)
    monkeypatch.setattr(routing, "refine_field_for_routing", lambda field, factor: field)
    request = {
        "origin": {"lat": 20.70, "lon": -86.90},
        "destination": {"lat": 20.90, "lon": -86.70},
    }
    response = client.post("/api/v1/routes", json=request)
    assert response.status_code == 404
    assert "No existe" in response.json()["detail"]


def test_acquisition_failure_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(**kwargs: object) -> None:
        raise PermissionError("Copernicus credentials unavailable")

    monkeypatch.setattr(routing, "acquire_ocean_data", fail)
    response = client.post("/api/v1/routes", json=_default_request())
    assert response.status_code == 503
    assert "credentials" in response.json()["detail"]


@pytest.mark.e2e
@pytest.mark.skipif(not Path.home().joinpath(".copernicusmarine/.copernicusmarine-credentials").exists(), reason="Copernicus credentials not configured")
def test_real_copernicus_route_cancun_to_playa() -> None:
    request = {
        "origin": {"lat": 21.1619, "lon": -86.8515},
        "destination": {"lat": 20.6296, "lon": -87.0739},
    }
    response = client.post("/api/v1/routes", json=request)
    assert response.status_code in (200, 404, 503), response.text
    if response.status_code == 200:
        route = response.json()["route"]
        assert route["type"] == "LineString"
        assert len(route["coordinates"]) >= 2
        assert "SIMULADOS" not in response.json()["meta"]["data_source"]
