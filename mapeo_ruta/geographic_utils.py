"""Utilidades geográficas y físicas para navegación marítima.

Incluye conversión de unidades, distancia Haversine, validación de región,
vectores unitarios locales (este-norte) y cálculo vectorial de velocidad
sobre el fondo considerando corrientes oceánicas (incluyendo deriva lateral).
"""

from __future__ import annotations

import math
from typing import Optional

from mapeo_ruta.config import BoundingBox, EARTH_RADIUS_M, GeoPoint, KNOTS_TO_MPS


def knots_to_mps(knots: float) -> float:
    """Convierte velocidad en nudos internacionales a metros por segundo (m/s)."""
    if knots < 0:
        raise ValueError("La velocidad en nudos no puede ser negativa.")
    return knots * KNOTS_TO_MPS


def mps_to_knots(mps: float) -> float:
    """Convierte velocidad en metros por segundo (m/s) a nudos."""
    if mps < 0:
        raise ValueError("La velocidad en m/s no puede ser negativa.")
    return mps / KNOTS_TO_MPS


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula la distancia geodésica aproximada en metros usando Haversine."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return EARTH_RADIUS_M * c


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula la distancia geodésica aproximada en kilómetros usando Haversine."""
    return haversine_m(lat1, lon1, lat2, lon2) / 1000.0


def validate_points_in_bbox(
    origin: GeoPoint, destination: GeoPoint, bbox: BoundingBox
) -> None:
    """Verifica que tanto el origen como el destino estén dentro de la región."""
    if not bbox.contains(origin):
        raise ValueError(
            f"El punto de origen ({origin.lat}, {origin.lon}) está fuera de la región "
            f"lat=[{bbox.lat_min}, {bbox.lat_max}], lon=[{bbox.lon_min}, {bbox.lon_max}]."
        )
    if not bbox.contains(destination):
        raise ValueError(
            f"El punto de destino ({destination.lat}, {destination.lon}) está fuera de la región "
            f"lat=[{bbox.lat_min}, {bbox.lat_max}], lon=[{bbox.lon_min}, {bbox.lon_max}]."
        )


def local_unit_direction_en(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> tuple[float, float]:
    """Devuelve el vector unitario local (este, norte) desde (lat1, lon1) hacia (lat2, lon2).

    Utiliza proyección tangente local equirectangular válida para distancias cortas
    entre celdas vecinas de una cuadrícula oceanográfica.
    """
    mean_lat_rad = math.radians(0.5 * (lat1 + lat2))
    d_east = math.radians(lon2 - lon1) * math.cos(mean_lat_rad) * EARTH_RADIUS_M
    d_north = math.radians(lat2 - lat1) * EARTH_RADIUS_M
    norm = math.hypot(d_east, d_north)
    if norm == 0.0:
        raise ValueError("Los dos puntos son idénticos; no existe dirección definida.")
    return d_east / norm, d_north / norm


def current_magnitude(uo: float, vo: float) -> float:
    """Calcula la magnitud escalar de la corriente (m/s) a partir de uo (este) y vo (norte)."""
    if math.isnan(uo) or math.isnan(vo):
        return float("nan")
    return math.hypot(uo, vo)


def current_direction_deg(uo: float, vo: float) -> float:
    """Dirección hacia donde fluye la corriente en grados náuticos (0°=N, 90°=E, 180°=S, 270°=O)."""
    if math.isnan(uo) or math.isnan(vo):
        return float("nan")
    angle = math.degrees(math.atan2(uo, vo))
    return (angle + 360.0) % 360.0


def speed_over_ground_mps(
    vessel_speed_mps: float,
    dir_east: float,
    dir_north: float,
    uo: float,
    vo: float,
) -> Optional[float]:
    """Calcula la velocidad efectiva sobre el fondo (SOG) en m/s manteniendo el rumbo deseado.

    Modelo vectorial con corrección de deriva (crabbing):
      - c_parallel = uo * dir_east + vo * dir_north  (componente a favor/en contra)
      - c_perp = |uo * dir_north - vo * dir_east|    (componente transversal)

    La embarcación debe dirigir parte de su velocidad en el agua a contrarrestar
    la corriente transversal c_perp:
      - Si c_perp >= vessel_speed_mps, la embarcación no puede mantener el rumbo -> None.
      - v_water_along = sqrt(vessel_speed_mps^2 - c_perp^2)
      - v_sog = v_water_along + c_parallel
      - Si v_sog <= 0 (corriente en contra demasiado fuerte) -> None.
    """
    if vessel_speed_mps <= 0:
        raise ValueError("La velocidad de la embarcación debe ser mayor a 0.")
    if math.isnan(uo) or math.isnan(vo):
        return None

    c_parallel = uo * dir_east + vo * dir_north
    c_perp = abs(uo * dir_north - vo * dir_east)

    if c_perp >= vessel_speed_mps:
        return None

    v_water_along = math.sqrt(vessel_speed_mps**2 - c_perp**2)
    v_sog = v_water_along + c_parallel

    if v_sog <= 0.0:
        return None
    return v_sog


def compute_geometric_baseline(
    origin: GeoPoint,
    destination: GeoPoint,
    vessel_speed_knots: float,
) -> dict[str, object]:
    """Calcula la ruta base geométrica directa A -> B sin considerar corrientes ni tierra.

    Esta ruta es únicamente una referencia geométrica (Etapa 5), no una ruta marítima validada.
    """
    distance_m = haversine_m(origin.lat, origin.lon, destination.lat, destination.lon)
    distance_km = distance_m / 1000.0
    speed_mps = knots_to_mps(vessel_speed_knots)
    time_seconds = distance_m / speed_mps if speed_mps > 0 else float("inf")

    return {
        "tipo": "referencia_geometrica_directa",
        "es_ruta_navegable_validada": False,
        "nota": (
            "Línea directa geodésica entre A y B sin considerar corrientes, "
            "línea de costa ni obstáculos. Solo referencia geométrica."
        ),
        "distancia_km": round(distance_km, 3),
        "velocidad_nominal_nudos": vessel_speed_knots,
        "velocidad_nominal_mps": round(speed_mps, 4),
        "tiempo_estimado_min": round(time_seconds / 60.0, 2),
        "tiempo_estimado_s": round(time_seconds, 1),
    }
