from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import Point as ShPoint

from app import alerts, precompute_alerts
from app.main import app
from app.settings import settings

RUN = "2026-06-29T16-00-00Z_test"
D = 0.01  # cell size in degrees
ZONE_CENTER = (-86.5, 21.0)  # lon, lat
client = TestClient(app)


def cell(i, j, density):
    x, y = -87.0 + j * D, 20.5 + i * D
    ring = [[x, y], [x, y + D], [x + D, y + D], [x + D, y], [x, y]]
    return {"type": "Feature", "properties": {"density": density, "i": i, "j": j},
            "geometry": {"type": "Polygon", "coordinates": [ring]}}


def frame_cells(minutes):
    """Three waves through the zone at 60-90, 240-270 and 480; a faint blob elsewhere; haze everywhere."""
    feats = [cell(0, 0, 1e-5), cell(0, 1, 1e-5)]  # below min_density
    feats += [cell(80, 80, 0.01), cell(80, 81, 0.01)]  # far dense mass, never in zone
    zi, zj = 50, 50  # zone center cell (lat 21.0, lon -86.5)
    wave = {60: 0.003, 90: 0.002, 240: 0.005, 270: 0.004, 480: 0.002}.get(minutes)
    if wave:
        feats += [cell(zi, zj, wave), cell(zi + 1, zj, wave), cell(zi + 1, zj + 1, wave / 2)]
    return {"type": "FeatureCollection", "features": feats}


@pytest.fixture
def export(tmp_path, monkeypatch):
    run = tmp_path / "exports" / RUN
    (run / "frames").mkdir(parents=True)
    minutes = list(range(0, 541, 30))
    for m in minutes:
        (run / "frames" / f"t{m:04d}.geojson").write_text(json.dumps(frame_cells(m)))
    manifest = {"run_id": RUN, "t0": "2026-06-29T16:00:00Z", "step_minutes": 30,
                "frames": [{"minutes": m, "valid_utc": f"2026-06-29T{16 + m // 60:02d}:{m % 60:02d}:00Z"} for m in minutes]}
    (run / "manifest.json").write_text(json.dumps(manifest))
    zone = ShPoint(*ZONE_CENTER).buffer(0.015)
    (run / "collection_areas.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"id": "zone-a", "center": list(ZONE_CENTER)},
         "geometry": json.loads(json.dumps(zone.__geo_interface__))}]}))
    monkeypatch.setattr(settings, "sargassum_export_dir", tmp_path / "exports")
    monkeypatch.setattr(alerts, "ALERTS_DIR", tmp_path / "alerts")
    return run


def test_masses_8_connected_and_threshold():
    cells = [alerts.load_cells({"features": [cell(0, 0, 1), cell(1, 1, 1), cell(5, 5, 1), cell(9, 9, 1e-6)]})][0]
    masses = alerts.build_masses(cells, 1e-3)
    assert sorted(len(m) for m in masses) == [1, 2]  # diagonal joins; faint cell dropped


def test_detect_three_alerts_peaks(export):
    found, _, info = alerts.detect_alerts(export, min_density=1e-3)
    assert [a["frame_minutes"] for a in found] == [60, 240, 480]
    assert info["episodes"] == 3 and not info["fallback_used"]
    a = found[0]
    assert a["intersection"]["cell_count"] >= 1 and a["mass"]["cell_count"] == 3
    assert ShPoint(a["destination"]["lon"], a["destination"]["lat"]).distance(ShPoint(*ZONE_CENTER)) < 0.015
    assert a["mass"]["geometry"]["type"] in ("Polygon", "MultiPolygon")
    assert a["mass"]["bbox"][0] < a["mass"]["bbox"][2]


def test_top_two_keeps_strongest(export):
    found, _, _ = alerts.detect_alerts(export, min_density=1e-3, top=2)
    assert [a["frame_minutes"] for a in found] == [60, 240]  # 480 is the weakest


def test_fallback_when_fewer_than_three_episodes(export):
    found, _, info = alerts.detect_alerts(export, min_density=1e-3, top=5)
    assert info["fallback_used"] and "note" in info
    mins = [a["frame_minutes"] for a in found]
    assert all(b - a >= 180 for a, b in zip(mins, mins[1:]))


def test_destination_moves_to_water():
    zone = ShPoint(0, 0).buffer(1.0)
    top = alerts.Cell(0, 0, 1.0, ShPoint(0.5, 0).buffer(0.1))
    lons, lats = np.array([-0.5, 0.5, 1.5]), np.array([0.0])
    land_there = type("F", (), {"lats": lats, "lons": lons, "water_mask": np.array([[True, False, False]])})()
    lat, lon = alerts.pick_destination([top], zone, land_there)
    assert (lat, lon) == (0.0, -0.5)


def fake_router(port, dest):
    if port["id"] == "cozumel":
        return {"status": "error", "error": "boom"}
    t = {"puerto-morelos": 30.0}.get(port["id"], 90.0)
    return {"status": "ok", "route": {"type": "LineString", "coordinates": [[port["lng"], port["lat"]], [dest["lon"], dest["lat"]]]},
            "meta": {"time_minutes": t}}


def test_cli_and_endpoints(export, monkeypatch):
    monkeypatch.setattr(precompute_alerts, "route_entry", fake_router)
    monkeypatch.setattr(precompute_alerts, "load_zone_field", lambda lat, lon: None)
    precompute_alerts.main(["--run-id", RUN, "--min-density", "1e-3"])
    assert (alerts.ALERTS_DIR / f"{RUN}.json").is_file()

    r = client.get("/api/v1/alerts")
    body = r.json()
    assert r.status_code == 200 and body["run_id"] == RUN and body["collection_area_ids"] == ["zone-a"]
    assert len(body["alerts"]) == 3 and "routes" not in body["alerts"][0]
    first = body["alerts"][0]
    assert first["recommended_port_id"] == "puerto-morelos" and first["route_count"] == 4

    d = client.get(f"/api/v1/alerts/{first['id']}").json()
    assert d["routes"]["cozumel"] == {"status": "error", "route": None, "meta": None, "error": "boom"}
    assert d["routes"]["puerto-morelos"]["route"]["type"] == "LineString"
    assert len(d["routes"]) == 5

    assert client.get("/api/v1/alerts/nope").status_code == 404
    assert client.get("/api/v1/alerts/..%2Fx").status_code in (404, 422)


def test_cache_missing_is_503(tmp_path, monkeypatch):
    monkeypatch.setattr(alerts, "ALERTS_DIR", tmp_path / "none")
    r = client.get("/api/v1/alerts")
    assert r.status_code == 503 and "precompute_alerts" in r.json()["detail"]
    assert client.get("/api/v1/alerts/alert-t0060").status_code == 503
