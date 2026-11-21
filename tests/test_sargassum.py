from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.settings import settings

RUN = "2026-06-29T16-00-00Z_test"
client = TestClient(app)


@pytest.fixture(autouse=True)
def export_dir(tmp_path, monkeypatch):
    run = tmp_path / RUN
    (run / "frames").mkdir(parents=True)
    # big enough to trigger gzip (>1024 bytes)
    cells = [{"type": "Feature", "properties": {"density": 0.5, "i": i, "j": 0}, "geometry": None} for i in range(60)]
    for m in (0, 30):
        (run / "frames" / f"t{m:04d}.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": cells}))
    manifest = {"run_id": RUN, "t0": "2026-06-29T16:00:00Z", "step_minutes": 30, "bbox": [-87, 20, -86, 21],
                "frames": [{"minutes": 0}, {"minutes": 30}]}
    (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "arrivals.json").write_text('{"areas": {}}')
    (run / "arrivals_origin.geojson").write_text('{"type": "FeatureCollection", "features": []}')
    (run / "collection_areas.geojson").write_text('{"type": "FeatureCollection", "features": []}')
    (tmp_path / "secret.json").write_text("secret")
    monkeypatch.setattr(settings, "sargassum_export_dir", tmp_path)
    return tmp_path


base = f"/api/v1/sargassum/runs/{RUN}"


def test_runs_list():
    r = client.get("/api/v1/sargassum/runs")
    assert r.status_code == 200
    assert r.json()[0]["run_id"] == RUN and r.json()[0]["n_frames"] == 2


def test_runs_list_missing_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "sargassum_export_dir", tmp_path / "nope")
    assert client.get("/api/v1/sargassum/runs").json() == []


def test_manifest_arrivals_areas():
    assert client.get(f"{base}/manifest").json()["step_minutes"] == 30
    assert client.get(f"{base}/arrivals").status_code == 200
    r = client.get(f"{base}/arrivals/origin")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/geo+json")
    assert client.get("/api/v1/sargassum/collection-areas", params={"run_id": RUN}).status_code == 200


def test_frame_ok_headers_and_gzip():
    r = client.get(f"{base}/frames/30")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/geo+json")
    assert "immutable" in r.headers["cache-control"]
    assert r.headers["content-encoding"] == "gzip"
    assert len(r.json()["features"]) == 60


def test_404s():
    assert client.get(f"{base}/frames/60").status_code == 404
    assert client.get("/api/v1/sargassum/runs/nope/manifest").status_code == 404
    assert client.get("/api/v1/sargassum/collection-areas", params={"run_id": "nope"}).status_code == 404


@pytest.mark.parametrize("path", [
    f"{base}/frames/abc", f"{base}/frames/-1", f"{base}/frames/123456",
    "/api/v1/sargassum/runs/..%2F/manifest", "/api/v1/sargassum/runs/../manifest",
    "/api/v1/sargassum/runs/%2e%2e/manifest", "/api/v1/sargassum/runs/a%2Fb/manifest",
])
def test_invalid_or_traversal_rejected(path):
    assert client.get(path).status_code in (404, 422)


def test_traversal_query_rejected():
    assert client.get("/api/v1/sargassum/collection-areas", params={"run_id": ".."}).status_code == 422
    assert client.get("/api/v1/sargassum/collection-areas", params={"run_id": "../x"}).status_code == 422


def test_symlink_escape_blocked(export_dir):
    outside = export_dir.parent / "outside_secret"
    outside.mkdir(exist_ok=True)
    (outside / "manifest.json").write_text("{}")
    (export_dir / "evil").symlink_to(outside)
    assert client.get("/api/v1/sargassum/runs/evil/manifest").status_code == 404
