"""Detection of sargassum masses reaching a collection zone (pure functions over export files)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import shapely
from fastapi import APIRouter, HTTPException, Path as PathParam, Query
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import unary_union

from app.alert_schemas import AlertDetail, AlertsResponse

# Relative density per 1 km cell (frames have median ~6e-5, p90 ~6e-4, max ~1e-2).
# 2.5e-4 (~p95) keeps the dense cores and drops the diffuse haze: at 1e-4 one mass reaches
# 279 cells and episodes merge, at 5e-4 episodes fragment into 1-cell blips; override with --min-density.
MIN_DENSITY = 2.5e-4
TOP_ALERTS = 3
MIN_SEPARATION_MIN = 180  # fallback: distinct frames at least 3 h apart


@dataclass
class Cell:
    i: int
    j: int
    density: float
    geom: Polygon


@dataclass
class Match:
    """Best mass of one frame that touches the zone."""

    minutes: int
    cells: list[Cell]  # whole mass
    hit: list[Cell]  # cells of the mass intersecting the zone
    score: float  # density summed over the hit cells


def load_cells(frame: dict[str, Any]) -> list[Cell]:
    return [
        Cell(f["properties"]["i"], f["properties"]["j"], float(f["properties"]["density"]), shape(f["geometry"]))
        for f in frame["features"]
    ]


def build_masses(cells: list[Cell], min_density: float) -> list[list[Cell]]:
    """8-connected components (on the i,j grid) of cells with density >= min_density."""
    index = {(c.i, c.j): c for c in cells if c.density >= min_density}
    seen: set[tuple[int, int]] = set()
    masses: list[list[Cell]] = []
    for start in index:
        if start in seen:
            continue
        seen.add(start)
        stack, group = [start], []
        while stack:
            i, j = stack.pop()
            group.append(index[i, j])
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nxt = (i + di, j + dj)
                    if nxt in index and nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
        masses.append(group)
    return masses


def best_match(minutes: int, masses: list[list[Cell]], zone: Polygon) -> Match | None:
    """Mass with the highest density inside the zone, or None if no mass touches it."""
    best: Match | None = None
    for mass in masses:
        touches = shapely.intersects(np.array([c.geom for c in mass], dtype=object), zone)
        hit = [c for c, t in zip(mass, touches) if t]
        if not hit:
            continue
        score = sum(c.density for c in hit)
        if best is None or score > best.score:
            best = Match(minutes, mass, hit, score)
    return best


def group_episodes(matches: list[Match], step: int) -> list[list[Match]]:
    """Consecutive matching frames (spaced `step` minutes) form one episode."""
    episodes: list[list[Match]] = []
    for m in sorted(matches, key=lambda m: m.minutes):
        if episodes and m.minutes - episodes[-1][-1].minutes <= step:
            episodes[-1].append(m)
        else:
            episodes.append([m])
    return episodes


def select_peaks(matches: list[Match], step: int, top: int = TOP_ALERTS) -> tuple[list[Match], dict[str, Any]]:
    """Peak frame of each of the top episodes; falls back to separated frames when episodes < top."""
    episodes = group_episodes(matches, step)
    peaks = sorted((max(ep, key=lambda m: m.score) for ep in episodes), key=lambda m: -m.score)[:top]
    info: dict[str, Any] = {"episodes": len(episodes), "fallback_used": False}
    if len(peaks) < top:
        info["fallback_used"] = True
        for m in sorted(matches, key=lambda m: -m.score):
            if len(peaks) == top:
                break
            if all(abs(m.minutes - p.minutes) >= MIN_SEPARATION_MIN for p in peaks):
                peaks.append(m)
        info["note"] = f"only {len(episodes)} episode(s); filled with frames >= {MIN_SEPARATION_MIN // 60} h apart"
    return sorted(peaks, key=lambda m: m.minutes), info


def nearest_water(field: Any, lat: float, lon: float) -> bool:
    row = int(np.argmin(np.abs(field.lats - lat)))
    col = int(np.argmin(np.abs(field.lons - lon)))
    return bool(field.water_mask[row, col])


def pick_destination(hit: list[Cell], zone: Polygon, field: Any | None = None) -> tuple[float, float]:
    """(lat, lon) inside the zone: centroid of the densest hit cell clipped to the zone.

    If `field` (routing grid) is given and that point is not water, take the nearest
    water grid node inside the zone; if none exists keep the densest cell centroid.
    """
    top = max(hit, key=lambda c: c.density)
    p = top.geom.intersection(zone).centroid
    if field is None or nearest_water(field, p.y, p.x):
        return p.y, p.x
    lons, lats = np.meshgrid(field.lons, field.lats)
    ok = field.water_mask & shapely.contains_xy(zone, lons, lats)
    if not ok.any():
        return p.y, p.x
    dist = (lats - p.y) ** 2 + ((lons - p.x) * np.cos(np.radians(p.y))) ** 2
    dist = np.where(ok, dist, np.inf)
    r, c = np.unravel_index(np.argmin(dist), dist.shape)
    return float(field.lats[r]), float(field.lons[c])


def cell_area_km2(cell: Cell) -> float:
    minx, miny, maxx, maxy = cell.geom.bounds
    lat = (miny + maxy) / 2
    return (maxx - minx) * 111.32 * np.cos(np.radians(lat)) * (maxy - miny) * 110.57


def build_alert(match: Match, zone: Polygon, area_id: str, valid_utc: str, field: Any | None = None) -> dict[str, Any]:
    union = unary_union([c.geom for c in match.cells])
    cx = union.centroid
    lat, lon = pick_destination(match.hit, zone, field)
    area = float(sum(cell_area_km2(c) for c in match.cells))
    hh = valid_utc[5:16].replace("T", " ")
    return {
        "id": f"alert-t{match.minutes:04d}",
        "title": f"Masa de sargazo hacia la zona de recolección ({hh} UTC)",
        "message": (
            f"Masa de {area:.1f} km² prevista dentro de la zona de recolección el {hh} UTC; "
            f"{len(match.hit)} celdas dentro de la zona."
        ),
        "prediction_utc": valid_utc,
        "frame_minutes": match.minutes,
        "area_id": area_id,
        "mass": {
            "id": f"mass-t{match.minutes:04d}",
            "cell_count": len(match.cells),
            "area_km2": round(area, 3),
            "density_sum": float(sum(c.density for c in match.cells)),
            "centroid": {"lat": cx.y, "lon": cx.x},
            "bbox": list(union.bounds),
            "geometry": mapping(union),
        },
        "intersection": {"cell_count": len(match.hit), "density_sum": match.score},
        "destination": {"lat": lat, "lon": lon},
    }


def detect_alerts(
    run_dir: Path, min_density: float = MIN_DENSITY, field: Any | None = None, top: int = TOP_ALERTS
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Return (alerts without routes, manifest, selection info) for the first collection area."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    areas = json.loads((run_dir / "collection_areas.geojson").read_text())["features"]
    if not areas:
        raise ValueError("collection_areas.geojson has no zone")
    feature = areas[0]
    zone, area_id = shape(feature["geometry"]), feature["properties"]["id"]
    step = int(manifest["step_minutes"])
    valid = {f["minutes"]: f["valid_utc"] for f in manifest["frames"]}
    matches = []
    for minutes in sorted(valid):
        frame = json.loads((run_dir / "frames" / f"t{minutes:04d}.geojson").read_text())
        m = best_match(minutes, build_masses(load_cells(frame), min_density), zone)
        if m:
            matches.append(m)
    peaks, info = select_peaks(matches, step, top)
    info.update(min_density=min_density, matching_frames=len(matches), area_id=area_id)
    alerts = [build_alert(m, zone, area_id, valid[m.minutes], field) for m in peaks]
    return alerts, manifest, info


# ---- read-only API over the precomputed cache (data/alerts/<run_id>.json) ----

ALERTS_DIR = Path(__file__).resolve().parent.parent / "data" / "alerts"

router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])


def _cache(run_id: str | None) -> dict[str, Any]:
    if run_id is None:
        files = sorted(ALERTS_DIR.glob("*.json")) if ALERTS_DIR.is_dir() else []
        path = files[-1] if files else None
    else:
        path = ALERTS_DIR / f"{run_id}.json"
    if path is None or not path.is_file():
        raise HTTPException(
            status_code=503,
            detail="Alert cache missing: run `uv run python -m app.precompute_alerts` first.",
        )
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"Alert cache unreadable: {exc}") from exc


RunQuery = Annotated[str | None, Query(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")]


@router.get("", response_model=AlertsResponse)
def list_alerts(run_id: RunQuery = None) -> dict[str, Any]:
    data = _cache(run_id)
    return {**data, "alerts": [{k: v for k, v in a.items() if k != "routes"} for a in data["alerts"]]}


@router.get("/{alert_id}", response_model=AlertDetail)
def get_alert(alert_id: Annotated[str, PathParam(pattern=r"^[A-Za-z0-9_.-]{1,64}$")], run_id: RunQuery = None) -> dict[str, Any]:
    for alert in _cache(run_id)["alerts"]:
        if alert["id"] == alert_id:
            return alert
    raise HTTPException(status_code=404, detail="alert not found")
