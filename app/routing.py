from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np

from mapeo_ruta.config import BoundingBox, GeoPoint, NAVIGATION_DISCLAIMER, ScenarioConfig
from mapeo_ruta.copernicus_client import acquire_ocean_data
from mapeo_ruta.ocean_data import load_ocean_currents, refine_field_for_routing
from mapeo_ruta.geographic_utils import knots_to_mps
from mapeo_ruta.route_optimizer import run_astar



from app.landmask import load_land_mask
from app.schemas import Point
from app.settings import settings
from app.smoothing import METHOD, SmoothingResult, haversine_km, haversine_km_array, smooth_route


COAST_KM = 1.0  # celda costera propia del puerto: el conector puede cruzar tierra solo hasta aquí
MAX_SNAP_KM = 30.0
SNAP_CANDIDATES = 6


def _connector_clear(point: list[float], node: list[float], mask) -> bool:
    """El conector point→node no pisa tierra (máscara fina) más allá de la celda costera del punto."""
    if mask is None:
        return True
    km = haversine_km(point, node)
    t = np.linspace(0.0, 1.0, max(2, int(km / 0.4) + 2))
    pts = np.column_stack((point[0] + (node[0] - point[0]) * t, point[1] + (node[1] - point[1]) * t))
    covered, land = mask.query(pts)
    far = haversine_km_array(point[0], point[1], pts[:, 0], pts[:, 1]) > COAST_KM
    return not bool((land & far).any())


def _candidate_nodes(field, point: list[float], mask) -> list[tuple[int, int]]:
    """Nodos de agua cercanos, por distancia, cuyo conector no cruza tierra (si hay máscara fina).

    Si ninguno cumple se devuelven los más cercanos igualmente (el suavizado lo reportará).
    """
    rows, cols = np.nonzero(field.water_mask)
    if len(rows) == 0:
        return []
    dist = haversine_km_array(point[0], point[1], field.lons[cols], field.lats[rows])
    order = [i for i in np.argsort(dist) if dist[i] <= MAX_SNAP_KM] or [int(np.argmin(dist))]
    clear = []
    for i in order:
        if _connector_clear(point, [float(field.lons[cols[i]]), float(field.lats[rows[i]])], mask):
            clear.append((int(rows[i]), int(cols[i])))
            if len(clear) == SNAP_CANDIDATES:
                break
    return clear or [(int(rows[i]), int(cols[i])) for i in order[:SNAP_CANDIDATES]]


def _snap_warning(label: str, point: list[float], node: list[float]) -> str | None:
    km = haversine_km(point, node)
    if km <= COAST_KM:
        return None
    return (
        f"El punto '{label}' ({point[1]:.6f}, {point[0]:.6f}) se conectó al nodo marino más cercano "
        f"alcanzable sin cruzar tierra ({node[1]:.6f}, {node[0]:.6f}), a {km:.2f} km. Depende de la "
        f"resolución del modelo y NO garantiza un punto de salida o navegación segura."
    )


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
    mask = load_land_mask(settings.land_mask_path)
    origin_ll, destination_ll = [a.lon, a.lat], [b.lon, b.lat]
    starts = _candidate_nodes(routing_field, origin_ll, mask)
    goals = _candidate_nodes(routing_field, destination_ll, mask)
    speed_mps = knots_to_mps(settings.vessel_speed_knots)
    route_result = None
    pairs = sorted(((i + j, i, j) for i in range(len(starts)) for j in range(len(goals))))
    for _, i, j in pairs[:12]:
        route_result = run_astar(routing_field, starts[i], goals[j], speed_mps, optimize_for="time")
        if route_result.found:
            start_node, goal_node = starts[i], goals[j]
            break
    if route_result is None or not route_result.found:
        return {"not_found": route_result.message if route_result else "No hay nodos de agua cerca de los puntos."}
    geojson = route_result.to_geojson_feature()
    if geojson is None:
        return {"not_found": route_result.message}
    raw = geojson["geometry"]["coordinates"]
    # Conectores rectos entre el punto pedido y el nodo A* (la cuadrícula es de ~3-9 km): se suman a
    # distancia y tiempo (a velocidad del buque, sin corriente). El resto es el valor de la ruta A*
    # validada; la curva suavizada solo mejora la geometría (se desvía `max_deviation_km`).
    origin_km = haversine_km(origin_ll, raw[0])
    destination_km = haversine_km(raw[-1], destination_ll)
    connector_min = (origin_km + destination_km) / (settings.vessel_speed_knots * 1.852) * 60.0
    if settings.smoothing:
        smoothed = smooth_route(raw, routing_field, origin_ll, destination_ll, mask=mask)
    else:
        smoothed = SmoothingResult(
            [origin_ll, *raw, destination_ll], False, len(raw), 0.0, "El suavizado está desactivado en la configuración."
        )
    coordinates = smoothed.coordinates
    return {
        "route": {**geojson["geometry"], "coordinates": coordinates},
        "meta": {
            "distance_km": round(route_result.distance_km + origin_km + destination_km, 3),
            "origin_connector_km": round(origin_km, 3),
            "destination_connector_km": round(destination_km, 3),
            "time_minutes": round(route_result.time_minutes + connector_min, 2),
            "data_source": acquisition.source_label,
            "selected_time": routing_field.selected_time,
            "snap_warnings": [
                warning
                for warning in (
                    _snap_warning("Origen (A)", origin_ll, raw[0]),
                    _snap_warning("Destino (B)", destination_ll, raw[-1]),
                )
                if warning
            ],
            "disclaimer": NAVIGATION_DISCLAIMER,
            "smoothing": {
                "applied": smoothed.applied,
                "method": METHOD,
                "raw_points": len(raw),
                "simplified_points": smoothed.simplified_points,
                "max_deviation_km": round(smoothed.max_deviation_km, 3),
                "reason": smoothed.reason,
            },
        },
    }
