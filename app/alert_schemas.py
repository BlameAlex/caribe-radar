from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class LatLon(BaseModel):
    lat: float
    lon: float


class Mass(BaseModel):
    id: str
    cell_count: int
    area_km2: float
    density_sum: float
    centroid: LatLon
    bbox: list[float]  # [west, south, east, north]
    geometry: dict[str, Any]  # GeoJSON Polygon | MultiPolygon


class Intersection(BaseModel):
    cell_count: int
    density_sum: float


class AlertSummary(BaseModel):
    id: str
    title: str
    message: str
    prediction_utc: str
    frame_minutes: int
    area_id: str
    mass: Mass
    intersection: Intersection
    destination: LatLon
    recommended_port_id: str | None
    route_count: int


class RouteEntry(BaseModel):
    status: Literal["ok", "error"]
    route: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None
    error: str | None = None


class AlertDetail(AlertSummary):
    routes: dict[str, RouteEntry]


class AlertsResponse(BaseModel):
    run_id: str
    t0: str
    generated_utc: str
    collection_area_ids: list[str]
    alerts: list[AlertSummary]
    # additive, backward compatible: how the alerts were selected
    selection: dict[str, Any] | None = None
