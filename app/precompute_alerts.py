"""CLI: detect alerts for a run and pre-generate routes from every port.

    uv run python -m app.precompute_alerts [--run-id RUN] [--min-density D] [--top N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app import alerts as alerts_mod
from app.ports import load_ports
from app.schemas import Point
from app.settings import settings


def load_zone_field(lat: float, lon: float) -> Any | None:
    """Routing grid around the zone (Copernicus, cached) to check the destination is water."""
    from mapeo_ruta.config import BoundingBox, GeoPoint, ScenarioConfig
    from mapeo_ruta.copernicus_client import acquire_ocean_data
    from mapeo_ruta.ocean_data import load_ocean_currents, refine_field_for_routing

    from app import routing

    m = settings.bbox_margin_deg
    bbox = BoundingBox(lon_min=lon - m, lon_max=lon + m, lat_min=lat - m, lat_max=lat + m)
    target = routing.rounded_target_datetime()
    scenario = ScenarioConfig(GeoPoint(lat, lon, "z"), GeoPoint(lat, lon, "z"), settings.vessel_speed_knots, target, bbox)
    path = routing._cache_path(bbox, target)
    path.parent.mkdir(parents=True, exist_ok=True)
    acq = acquire_ocean_data(scenario=scenario, netcdf_path=path, allow_simulated_fallback=False, use_simulated=False)
    return refine_field_for_routing(load_ocean_currents(acq.netcdf_path, target_datetime=target), factor=settings.refine_factor)


def route_entry(port: dict[str, Any], dest: dict[str, float]) -> dict[str, Any]:
    from app.routing import calculate_route

    try:
        result = calculate_route(Point(lat=port["lat"], lon=port["lng"]), Point(lat=dest["lat"], lon=dest["lon"]))
    except Exception as exc:  # per-route failures are recorded, not fatal
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    if "not_found" in result:
        return {"status": "error", "error": str(result["not_found"])}
    return {"status": "ok", "route": result["route"], "meta": result["meta"]}


def attach_routes(alert: dict[str, Any], ports: list[dict[str, Any]]) -> None:
    routes = {p["id"]: route_entry(p, alert["destination"]) for p in ports}
    ok = {k: v for k, v in routes.items() if v["status"] == "ok"}
    alert["routes"] = routes
    alert["route_count"] = len(ok)
    alert["recommended_port_id"] = min(ok, key=lambda k: ok[k]["meta"]["time_minutes"]) if ok else None


def latest_run(root: Path) -> str:
    runs = sorted(p.parent.name for p in root.glob("*/manifest.json"))
    if not runs:
        sys.exit(f"no runs with manifest.json under {root}")
    return runs[-1]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id")
    ap.add_argument("--min-density", type=float, default=alerts_mod.MIN_DENSITY)
    ap.add_argument("--top", type=int, default=alerts_mod.TOP_ALERTS)
    args = ap.parse_args(argv)

    root = settings.sargassum_export_dir
    run_id = args.run_id or latest_run(root)
    run_dir = root / run_id

    # first pass without field only to locate the zone; destinations are then water-checked
    zone_feature = json.loads((run_dir / "collection_areas.geojson").read_text())["features"][0]
    cx, cy = zone_feature["properties"]["center"]
    field, field_note = None, None
    try:
        field = load_zone_field(cy, cx)
    except Exception as exc:
        field_note = f"destination water check skipped: {type(exc).__name__}: {exc}"
        print(field_note, file=sys.stderr)
    alerts, manifest, info = alerts_mod.detect_alerts(run_dir, args.min_density, field, args.top)
    if field_note:
        info["warning"] = field_note
    print(f"{len(alerts)} alerts: " + ", ".join(a["prediction_utc"] for a in alerts), flush=True)

    ports = load_ports()
    t0 = time.monotonic()
    for alert in alerts:
        attach_routes(alert, ports)
        errs = {k: v["error"] for k, v in alert["routes"].items() if v["status"] == "error"}
        print(f"{alert['id']}: {alert['route_count']}/{len(ports)} routes ok, recommended={alert['recommended_port_id']}"
              f" ({time.monotonic() - t0:.0f}s)" + (f" errors={errs}" if errs else ""), flush=True)

    payload = {
        "run_id": manifest["run_id"],
        "t0": manifest["t0"],
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "collection_area_ids": [info["area_id"]],
        "alerts": alerts,
        "selection": info,
    }
    out = alerts_mod.ALERTS_DIR / f"{run_dir.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
