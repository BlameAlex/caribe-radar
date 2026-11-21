from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from mapeo_ruta.config import BoundingBox, GeoPoint, NAVIGATION_DISCLAIMER, ScenarioConfig
from mapeo_ruta.copernicus_client import acquire_ocean_data
from mapeo_ruta.ocean_data import load_ocean_currents, refine_field_for_routing
from mapeo_ruta.route_optimizer import optimize_maritime_route

from app.schemas import Point
from app.settings import settings


class RouteInputError(ValueError):
    """Los puntos recibidos no forman una solicitud de ruta válida."""


class DataAcquisitionError(RuntimeError):
    """No se pudieron obtener datos oceanográficos reales."""


def rounded_target_datetime(now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    current = current.astimezone(timezone.utc)
    hour = current.hour - current.hour % 6
    return current.replace(hour=hour, minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%S")


def _cache_path(bbox: BoundingBox, target_datetime: str):
    key = f"{bbox.lat_min:.6f},{bbox.lat_max:.6f},{bbox.lon_min:.6f},{bbox.lon_max:.6f},{target_datetime}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:20]
    return settings.cache_dir / f"{digest}.nc"


def calculate_route(origin: Point, destination: Point) -> dict[str, object]:
    a = GeoPoint(lat=origin.lat, lon=origin.lon, name="Origen (A)")
    b = GeoPoint(lat=destination.lat, lon=destination.lon, name="Destino (B)")
    if a.lat == b.lat and a.lon == b.lon:
        raise RouteInputError("El origen y el destino deben ser distintos.")
    if abs(a.lat - b.lat) > settings.max_span_deg or abs(a.lon - b.lon) > settings.max_span_deg:
        raise RouteInputError(f"La distancia entre puntos excede el máximo de {settings.max_span_deg} grados.")

    margin = settings.bbox_margin_deg
    bbox = BoundingBox(
        lon_min=min(a.lon, b.lon) - margin,
        lon_max=max(a.lon, b.lon) + margin,
        lat_min=max(-90.0, min(a.lat, b.lat) - margin),
        lat_max=min(90.0, max(a.lat, b.lat) + margin),
    )
    target_datetime = rounded_target_datetime()
    scenario = ScenarioConfig(a, b, settings.vessel_speed_knots, target_datetime, bbox)
    cache_path = _cache_path(bbox, target_datetime)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Acquisition errors (credentials, network, product availability) are handled as 503.
    try:
        acquisition = acquire_ocean_data(
            scenario=scenario,
            netcdf_path=cache_path,
            allow_simulated_fallback=False,
            use_simulated=False,
        )
    except Exception as exc:
        raise DataAcquisitionError(str(exc)) from exc
    field = load_ocean_currents(acquisition.netcdf_path, target_datetime=target_datetime)
    routing_field = refine_field_for_routing(field, factor=settings.refine_factor)
    report = optimize_maritime_route(routing_field, a, b, settings.vessel_speed_knots)
    route_result = report.optimized_route
    if not route_result.found:
        return {"not_found": route_result.message}
    geojson = route_result.to_geojson_feature()
    if geojson is None:
        return {"not_found": route_result.message}
    return {
        "route": geojson["geometry"],
        "meta": {
            "distance_km": route_result.distance_km,
            "time_minutes": route_result.time_minutes,
            "data_source": acquisition.source_label,
            "selected_time": routing_field.selected_time,
            "snap_warnings": [
                warning for warning in (report.origin_snap.warning, report.destination_snap.warning) if warning
            ],
            "disclaimer": NAVIGATION_DISCLAIMER,
        },
    }
