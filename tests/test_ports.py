from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_ports_real_file():
    r = client.get("/api/v1/ports")
    assert r.status_code == 200
    ports = r.json()
    assert len(ports) == 5
    assert {"id", "name", "short_name", "municipality", "lat", "lng", "description", "facilities"} <= ports[0].keys()
    assert "puerto-morelos" in {p["id"] for p in ports}
