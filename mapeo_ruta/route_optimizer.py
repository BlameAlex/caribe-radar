"""Optimización de rutas marítimas mediante A* considerando corrientes oceánicas.

Etapas 6, 7 y 8:
  - Representación en cuadrícula de nodos de agua válidos (excluye tierra y NaN).
  - Detección explícita y ajuste de puntos sobre tierra al nodo marino más cercano.
  - Conectividad en 8 direcciones prohibiendo diagonales que corten esquinas terrestres.
  - Modelo vectorial de velocidad sobre el fondo (SOG) con corrección de deriva lateral.
  - Algoritmo A* con heurística admisible para minimizar tiempo estimado de navegación.
  - Comparación consistente entre la ruta más corta transitable y la ruta optimizada
    evaluadas bajo el mismo campo de corrientes.
  - Exportación a JSON y GeoJSON (LineString en orden [longitud, latitud]).

Simplificación documentada (MVP):
  - Se utiliza una única instantánea superficial de corrientes asumida constante
    durante todo el recorrido.
  - Una celda oceanográfica válida NO garantiza navegación segura (sin cartas náuticas
    ni batimetría).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from mapeo_ruta.config import GeoPoint, NAVIGATION_DISCLAIMER
from mapeo_ruta.geographic_utils import (
    compute_geometric_baseline,
    haversine_m,
    knots_to_mps,
    local_unit_direction_en,
    speed_over_ground_mps,
)
from mapeo_ruta.ocean_data import OceanCurrentField

# 8 direcciones vecinas: ortogonales y diagonales
_NEIGHBOR_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


@dataclass(frozen=True)
class SnappedNode:
    """Resultado de asignar un punto geográfico a un nodo marino válido de la cuadrícula."""

    label: str
    requested_lat: float
    requested_lon: float
    row: int
    col: int
    node_lat: float
    node_lon: float
    was_on_land_or_missing: bool
    displacement_m: float
    warning: Optional[str] = None

    def to_dict(self) -> dict[str, object]:
        return {
            "etiqueta": self.label,
            "coordenada_solicitada": {
                "lat": round(self.requested_lat, 6),
                "lon": round(self.requested_lon, 6),
            },
            "nodo_marino_asignado": {
                "lat": round(self.node_lat, 6),
                "lon": round(self.node_lon, 6),
                "fila": self.row,
                "columna": self.col,
            },
            "estaba_en_tierra_o_sin_dato": self.was_on_land_or_missing,
            "desplazamiento_m": round(self.displacement_m, 1),
            "advertencia": self.warning,
        }


@dataclass(frozen=True)
class RouteResult:
    """Resultado de un cálculo de trayectoria sobre la cuadrícula marina."""

    found: bool
    mode: str  # "time_optimized" o "shortest_navigable"
    nodes: list[tuple[int, int]]
    coordinates: list[dict[str, float]]  # [{"lat": ..., "lon": ...}, ...]
    distance_km: Optional[float]
    time_seconds: Optional[float]
    time_minutes: Optional[float]
    avg_sog_mps: Optional[float]
    expanded_nodes: int
    message: str

    def to_geojson_feature(self, properties: Optional[dict[str, object]] = None) -> Optional[dict[str, object]]:
        """Genera un GeoJSON Feature (LineString) con orden [longitud, latitud]."""
        if not self.found or len(self.coordinates) < 2:
            return None
        coords_lon_lat = [[pt["lon"], pt["lat"]] for pt in self.coordinates]
        props = {
            "mode": self.mode,
            "distance_km": self.distance_km,
            "time_minutes": self.time_minutes,
            "is_validated_safe_navigation": False,
        }
        if properties:
            props.update(properties)
        return {
            "type": "Feature",
            "properties": props,
            "geometry": {
                "type": "LineString",
                "coordinates": coords_lon_lat,
            },
        }


@dataclass(frozen=True)
class OptimizationReport:
    """Resultado integral que combina ruta base, ruta corta transitable y ruta A*."""

    origin_snap: SnappedNode
    destination_snap: SnappedNode
    geometric_baseline: dict[str, object]
    shortest_navigable: RouteResult
    optimized_route: RouteResult
    time_saved_minutes: Optional[float]
    time_saved_percent: Optional[float]
    distance_delta_km: Optional[float]


def snap_point_to_water(
    field: OceanCurrentField,
    point: GeoPoint,
    label: str,
) -> SnappedNode:
    """Asigna un punto geográfico a un nodo marino válido.

    Si la celda más cercana de la cuadrícula corresponde a tierra o dato faltante (NaN),
    busca explícitamente el nodo de agua válido más cercano por distancia Haversine e
    informa el desplazamiento realizado, aclarando que NO constituye un punto de salida
    o llegada validado para navegación real.
    """
    water = field.water_mask
    if not np.any(water):
        raise ValueError("La cuadrícula oceanográfica no contiene ninguna celda de agua válida.")

    nearest_row = int(np.argmin(np.abs(field.lats - point.lat)))
    nearest_col = int(np.argmin(np.abs(field.lons - point.lon)))

    if bool(water[nearest_row, nearest_col]):
        node_lat = float(field.lats[nearest_row])
        node_lon = float(field.lons[nearest_col])
        disp_m = haversine_m(point.lat, point.lon, node_lat, node_lon)
        return SnappedNode(
            label=label,
            requested_lat=point.lat,
            requested_lon=point.lon,
            row=nearest_row,
            col=nearest_col,
            node_lat=node_lat,
            node_lon=node_lon,
            was_on_land_or_missing=False,
            displacement_m=disp_m,
            warning=None,
        )

    # La celda más cercana es tierra o NaN -> buscar el nodo de agua válido más cercano
    water_rows, water_cols = np.where(water)
    best_idx = 0
    best_dist = float("inf")
    for idx in range(len(water_rows)):
        r = int(water_rows[idx])
        c = int(water_cols[idx])
        d = haversine_m(point.lat, point.lon, float(field.lats[r]), float(field.lons[c]))
        if d < best_dist:
            best_dist = d
            best_idx = idx

    snapped_r = int(water_rows[best_idx])
    snapped_c = int(water_cols[best_idx])
    snapped_lat = float(field.lats[snapped_r])
    snapped_lon = float(field.lons[snapped_c])

    warning_msg = (
        f"El punto '{label}' ({point.lat:.6f}, {point.lon:.6f}) cayó sobre una celda "
        f"terrestre o sin datos del modelo. Se ajustó al nodo marino más cercano "
        f"({snapped_lat:.6f}, {snapped_lon:.6f}), desplazamiento de {best_dist:.1f} m "
        f"({best_dist / 1000.0:.2f} km). Este ajuste depende de la resolución del modelo "
        f"y NO garantiza un punto de salida o navegación segura."
    )

    return SnappedNode(
        label=label,
        requested_lat=point.lat,
        requested_lon=point.lon,
        row=snapped_r,
        col=snapped_c,
        node_lat=snapped_lat,
        node_lon=snapped_lon,
        was_on_land_or_missing=True,
        displacement_m=best_dist,
        warning=warning_msg,
    )


def _edge_kinematics(
    field: OceanCurrentField,
    node_a: tuple[int, int],
    node_b: tuple[int, int],
    vessel_speed_mps: float,
) -> Optional[tuple[float, float, float]]:
    """Calcula (distancia_m, tiempo_s, sog_mps) al transitar de `node_a` a `node_b`.

    Devuelve None si la corriente transversal o en contra impide avanzar por la arista.
    """
    r1, c1 = node_a
    r2, c2 = node_b

    lat1, lon1 = float(field.lats[r1]), float(field.lons[c1])
    lat2, lon2 = float(field.lats[r2]), float(field.lons[c2])

    dist_m = haversine_m(lat1, lon1, lat2, lon2)
    if dist_m == 0.0:
        return 0.0, 0.0, vessel_speed_mps

    u_edge = 0.5 * (float(field.uo[r1, c1]) + float(field.uo[r2, c2]))
    v_edge = 0.5 * (float(field.vo[r1, c1]) + float(field.vo[r2, c2]))

    dir_e, dir_n = local_unit_direction_en(lat1, lon1, lat2, lon2)
    sog = speed_over_ground_mps(vessel_speed_mps, dir_e, dir_n, u_edge, v_edge)
    if sog is None:
        return None

    time_s = dist_m / sog
    return dist_m, time_s, sog


def _valid_neighbors(
    water_mask: np.ndarray,
    node: tuple[int, int],
) -> list[tuple[int, int]]:
    """Devuelve vecinos válidos en 8 direcciones, prohibiendo cortar esquinas terrestres."""
    n_rows, n_cols = water_mask.shape
    r, c = node
    neighbors: list[tuple[int, int]] = []

    for dr, dc in _NEIGHBOR_OFFSETS:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < n_rows and 0 <= nc < n_cols):
            continue
        if not bool(water_mask[nr, nc]):
            continue
        # Prohibir diagonales que atraviesen esquinas terrestres o sin datos
        if dr != 0 and dc != 0:
            if not (bool(water_mask[r + dr, c]) and bool(water_mask[r, c + dc])):
                continue
        neighbors.append((nr, nc))

    return neighbors


def _reconstruct_path(
    came_from: dict[tuple[int, int], tuple[int, int]],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    path = [goal]
    cur = goal
    while cur in came_from:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path


def _evaluate_path_metrics(
    field: OceanCurrentField,
    path: list[tuple[int, int]],
    vessel_speed_mps: float,
) -> Optional[tuple[float, float]]:
    """Evalúa distancia total (m) y tiempo total (s) de un camino bajo el campo de corrientes."""
    total_dist_m = 0.0
    total_time_s = 0.0
    for idx in range(len(path) - 1):
        kin = _edge_kinematics(field, path[idx], path[idx + 1], vessel_speed_mps)
        if kin is None:
            return None
        dist_m, time_s, _ = kin
        total_dist_m += dist_m
        total_time_s += time_s
    return total_dist_m, total_time_s


def run_astar(
    field: OceanCurrentField,
    start: tuple[int, int],
    goal: tuple[int, int],
    vessel_speed_mps: float,
    optimize_for: str = "time",
) -> RouteResult:
    """Ejecuta A* sobre nodos de agua válidos minimizando tiempo o distancia.

    Parámetros:
      - optimize_for="time": minimiza tiempo estimado considerando corrientes (Etapa 6).
        Heurística admisible: h(n) = haversine(n, goal) / (v_ship + c_max).
      - optimize_for="distance": minimiza distancia transitable sobre agua (Etapa 7),
        evaluando posteriormente su tiempo real bajo las mismas corrientes.
        Heurística admisible: h(n) = haversine(n, goal).
    """
    if optimize_for not in ("time", "distance"):
        raise ValueError("optimize_for debe ser 'time' o 'distance'.")
    if vessel_speed_mps <= 0:
        raise ValueError("La velocidad de la embarcación debe ser positiva.")

    water = field.water_mask
    mode_label = "time_optimized" if optimize_for == "time" else "shortest_navigable"

    if not bool(water[start[0], start[1]]) or not bool(water[goal[0], goal[1]]):
        return RouteResult(
            found=False,
            mode=mode_label,
            nodes=[],
            coordinates=[],
            distance_km=None,
            time_seconds=None,
            time_minutes=None,
            avg_sog_mps=None,
            expanded_nodes=0,
            message="El nodo de inicio o destino no corresponde a una celda de agua válida.",
        )

    if start == goal:
        lat = round(float(field.lats[start[0]]), 6)
        lon = round(float(field.lons[start[1]]), 6)
        return RouteResult(
            found=True,
            mode=mode_label,
            nodes=[start],
            coordinates=[{"lat": lat, "lon": lon}],
            distance_km=0.0,
            time_seconds=0.0,
            time_minutes=0.0,
            avg_sog_mps=vessel_speed_mps,
            expanded_nodes=1,
            message="El nodo de inicio y el de destino coinciden en la cuadrícula.",
        )

    goal_lat = float(field.lats[goal[0]])
    goal_lon = float(field.lons[goal[1]])
    max_possible_sog = vessel_speed_mps + field.max_current_speed_mps

    def heuristic(node: tuple[int, int]) -> float:
        d_m = haversine_m(float(field.lats[node[0]]), float(field.lons[node[1]]), goal_lat, goal_lon)
        if optimize_for == "time":
            return d_m / max_possible_sog
        return d_m

    open_heap: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (heuristic(start), 0.0, start))

    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start: 0.0}
    closed: set[tuple[int, int]] = set()
    expanded = 0

    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        expanded += 1

        if current == goal:
            path = _reconstruct_path(came_from, goal)
            metrics = _evaluate_path_metrics(field, path, vessel_speed_mps)
            if metrics is None:
                return RouteResult(
                    found=False,
                    mode=mode_label,
                    nodes=[],
                    coordinates=[],
                    distance_km=None,
                    time_seconds=None,
                    time_minutes=None,
                    avg_sog_mps=None,
                    expanded_nodes=expanded,
                    message="La trayectoria reconstruida resultó intransitable bajo las corrientes.",
                )
            total_dist_m, total_time_s = metrics
            coords = [
                {
                    "lat": round(float(field.lats[r]), 6),
                    "lon": round(float(field.lons[c]), 6),
                }
                for r, c in path
            ]
            avg_sog = total_dist_m / total_time_s if total_time_s > 0 else vessel_speed_mps
            return RouteResult(
                found=True,
                mode=mode_label,
                nodes=path,
                coordinates=coords,
                distance_km=round(total_dist_m / 1000.0, 3),
                time_seconds=round(total_time_s, 1),
                time_minutes=round(total_time_s / 60.0, 2),
                avg_sog_mps=round(avg_sog, 4),
                expanded_nodes=expanded,
                message="Ruta calculada correctamente.",
            )

        for neighbor in _valid_neighbors(water, current):
            if neighbor in closed:
                continue

            kin = _edge_kinematics(field, current, neighbor, vessel_speed_mps)
            if kin is None:
                continue
            dist_m, time_s, _ = kin

            step_cost = time_s if optimize_for == "time" else dist_m
            tentative_g = current_g + step_cost

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor)
                heapq.heappush(open_heap, (f_score, tentative_g, neighbor))

    return RouteResult(
        found=False,
        mode=mode_label,
        nodes=[],
        coordinates=[],
        distance_km=None,
        time_seconds=None,
        time_minutes=None,
        avg_sog_mps=None,
        expanded_nodes=expanded,
        message=(
            "No existe una ruta transitable entre el origen y el destino "
            "(bloqueada por tierra, datos faltantes o corrientes superiores a la embarcación)."
        ),
    )


def optimize_maritime_route(
    field: OceanCurrentField,
    origin: GeoPoint,
    destination: GeoPoint,
    vessel_speed_knots: float,
) -> OptimizationReport:
    """Ejecuta el flujo completo de comparación: ruta base, ruta corta y ruta A* optimizada."""
    vessel_speed_mps = knots_to_mps(vessel_speed_knots)
    baseline = compute_geometric_baseline(origin, destination, vessel_speed_knots)

    origin_snap = snap_point_to_water(field, origin, label=origin.name or "Punto A (Origen)")
    dest_snap = snap_point_to_water(field, destination, label=destination.name or "Punto B (Destino)")

    start_node = (origin_snap.row, origin_snap.col)
    goal_node = (dest_snap.row, dest_snap.col)

    shortest = run_astar(
        field=field,
        start=start_node,
        goal=goal_node,
        vessel_speed_mps=vessel_speed_mps,
        optimize_for="distance",
    )
    optimized = run_astar(
        field=field,
        start=start_node,
        goal=goal_node,
        vessel_speed_mps=vessel_speed_mps,
        optimize_for="time",
    )

    time_saved_min: Optional[float] = None
    time_saved_pct: Optional[float] = None
    dist_delta_km: Optional[float] = None

    if (
        shortest.found
        and optimized.found
        and shortest.time_seconds is not None
        and optimized.time_seconds is not None
        and shortest.distance_km is not None
        and optimized.distance_km is not None
    ):
        diff_s = shortest.time_seconds - optimized.time_seconds
        time_saved_min = round(diff_s / 60.0, 2)
        if shortest.time_seconds > 0:
            time_saved_pct = round((diff_s / shortest.time_seconds) * 100.0, 2)
        else:
            time_saved_pct = 0.0
        dist_delta_km = round(optimized.distance_km - shortest.distance_km, 3)

    return OptimizationReport(
        origin_snap=origin_snap,
        destination_snap=dest_snap,
        geometric_baseline=baseline,
        shortest_navigable=shortest,
        optimized_route=optimized,
        time_saved_minutes=time_saved_min,
        time_saved_percent=time_saved_pct,
        distance_delta_km=dist_delta_km,
    )


def build_export_payload(
    origin: GeoPoint,
    destination: GeoPoint,
    vessel_speed_knots: float,
    field: OceanCurrentField,
    report: OptimizationReport,
    source_label: str,
) -> dict[str, object]:
    """Construye el diccionario JSON final (Etapa 8) compatible con frontends de mapas."""
    opt = report.optimized_route
    short = report.shortest_navigable

    geojson = opt.to_geojson_feature(
        properties={
            "name": "Ruta optimizada A* (corrientes superficiales)",
            "vessel_speed_knots": vessel_speed_knots,
            "is_simulated_currents": field.is_simulated,
            "time_saved_min": report.time_saved_minutes,
        }
    )

    return {
        "origen": origin.to_dict(),
        "destino": destination.to_dict(),
        "velocidad_nudos": vessel_speed_knots,
        "distancia_km": opt.distance_km,
        "tiempo_estimado_min": opt.time_minutes,
        "ahorro_tiempo_min": report.time_saved_minutes,
        "ruta": opt.coordinates,
        "geojson": geojson,
        "ruta_encontrada": opt.found,
        "aviso_navegacion": NAVIGATION_DISCLAIMER,
        "fuente_datos": {
            "descripcion": source_label,
            "es_simulado": field.is_simulated,
            "tiempo_seleccionado": field.selected_time,
            "profundidad_m": field.selected_depth_m,
            "simplificacion": (
                "Se utiliza una única instantánea de corrientes superficiales "
                "asumida constante durante el trayecto."
            ),
        },
        "ajustes_nodos": {
            "origen": report.origin_snap.to_dict(),
            "destino": report.destination_snap.to_dict(),
        },
        "ruta_base_geometrica": report.geometric_baseline,
        "ruta_corta_transitable": {
            "encontrada": short.found,
            "distancia_km": short.distance_km,
            "tiempo_estimado_min": short.time_minutes,
            "velocidad_media_sog_mps": short.avg_sog_mps,
            "ruta": short.coordinates,
            "geojson": short.to_geojson_feature(
                properties={"name": "Ruta más corta transitable (evaluada con corrientes)"}
            ),
        },
        "comparacion": {
            "ahorro_tiempo_min": report.time_saved_minutes,
            "ahorro_tiempo_porcentaje": report.time_saved_percent,
            "diferencia_distancia_km": report.distance_delta_km,
        },
    }
