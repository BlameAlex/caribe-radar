"""Pruebas unitarias y de validación matemática sin conexión a internet (Sección 7).

Cubre:
  1. Conversión de nudos a m/s.
  2. Distancia Haversine y validación de región.
  3. Interpretación vectorial de uo y vo.
  4. Corrientes a favor, en contra y transversales (deriva / crabbing).
  5. Preservación de datos faltantes (NaN) sin reemplazarlos por cero.
  6. Detección y ajuste explícito de puntos sobre tierra.
  7. Prohibición de diagonales que cortan esquinas terrestres.
  8. Ausencia de ruta transitable.
  9. Comparación consistente entre ruta corta transitable y ruta optimizada A*.
 10. Exportación JSON y GeoJSON (orden [lon, lat]).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from mapeo_ruta.config import BoundingBox, DEFAULT_BBOX, DEFAULT_DESTINATION, DEFAULT_ORIGIN, GeoPoint
from mapeo_ruta.copernicus_client import generate_simulated_netcdf
from mapeo_ruta.config import get_default_scenario
from mapeo_ruta.geographic_utils import (
    current_direction_deg,
    current_magnitude,
    haversine_km,
    haversine_m,
    knots_to_mps,
    mps_to_knots,
    speed_over_ground_mps,
    validate_points_in_bbox,
)
from mapeo_ruta.ocean_data import OceanCurrentField, load_ocean_currents, query_point_current
from mapeo_ruta.route_optimizer import (
    build_export_payload,
    optimize_maritime_route,
    run_astar,
    snap_point_to_water,
)


def _make_synthetic_field(
    lats: np.ndarray,
    lons: np.ndarray,
    uo: np.ndarray,
    vo: np.ndarray,
) -> OceanCurrentField:
    return OceanCurrentField(
        lats=lats.astype(np.float64),
        lons=lons.astype(np.float64),
        uo=uo.astype(np.float64),
        vo=vo.astype(np.float64),
        selected_time="2026-11-20T12:00:00",
        selected_depth_m=0.494,
        dimensions={"latitude": len(lats), "longitude": len(lons)},
        available_variables=["uo", "vo"],
        is_simulated=True,
    )


# ---------------------------------------------------------------------------
# 1. Conversión de unidades y Haversine
# ---------------------------------------------------------------------------
def test_knots_to_mps_conversion() -> None:
    assert math.isclose(knots_to_mps(1.0), 1852.0 / 3600.0, rel_tol=1e-9)
    assert math.isclose(knots_to_mps(8.0), 4.1155555556, rel_tol=1e-6)
    assert math.isclose(mps_to_knots(knots_to_mps(8.0)), 8.0, rel_tol=1e-9)
    with pytest.raises(ValueError):
        knots_to_mps(-1.0)


def test_haversine_distance() -> None:
    # 1 grado de latitud ≈ 111.195 km
    d_km = haversine_km(20.0, -86.0, 21.0, -86.0)
    assert 111.0 < d_km < 111.4
    assert haversine_m(20.84, -86.87, 20.84, -86.87) == 0.0

    # Distancia Puerto Morelos -> Zona sargazo (~25.5 km)
    d_pm = haversine_km(
        DEFAULT_ORIGIN.lat,
        DEFAULT_ORIGIN.lon,
        DEFAULT_DESTINATION.lat,
        DEFAULT_DESTINATION.lon,
    )
    assert 24.0 < d_pm < 27.0


def test_validate_points_in_bbox() -> None:
    validate_points_in_bbox(DEFAULT_ORIGIN, DEFAULT_DESTINATION, DEFAULT_BBOX)
    outside = GeoPoint(lat=25.0, lon=-86.8)
    with pytest.raises(ValueError):
        validate_points_in_bbox(outside, DEFAULT_DESTINATION, DEFAULT_BBOX)


# ---------------------------------------------------------------------------
# 2. Interpretación de uo y vo, corrientes a favor, en contra y transversales
# ---------------------------------------------------------------------------
def test_uo_vo_interpretation() -> None:
    # uo > 0 -> hacia el Este (90°); vo > 0 -> hacia el Norte (0°)
    assert math.isclose(current_magnitude(3.0, 4.0), 5.0, rel_tol=1e-9)
    assert math.isclose(current_direction_deg(1.0, 0.0), 90.0, abs_tol=1e-6)
    assert math.isclose(current_direction_deg(0.0, 1.0), 0.0, abs_tol=1e-6)
    assert math.isclose(current_direction_deg(0.0, -1.0), 180.0, abs_tol=1e-6)
    assert math.isclose(current_direction_deg(-1.0, 0.0), 270.0, abs_tol=1e-6)
    assert math.isnan(current_magnitude(float("nan"), 1.0))


def test_following_and_opposing_currents() -> None:
    v_ship = 4.0  # m/s
    # Rumbo al Este: dir=(1, 0)
    sog_favor = speed_over_ground_mps(v_ship, 1.0, 0.0, uo=1.0, vo=0.0)
    sog_contra = speed_over_ground_mps(v_ship, 1.0, 0.0, uo=-1.0, vo=0.0)
    assert sog_favor is not None and math.isclose(sog_favor, 5.0, rel_tol=1e-9)
    assert sog_contra is not None and math.isclose(sog_contra, 3.0, rel_tol=1e-9)

    # Corriente en contra igual o superior a la velocidad del barco -> intransitable
    assert speed_over_ground_mps(v_ship, 1.0, 0.0, uo=-4.0, vo=0.0) is None
    assert speed_over_ground_mps(v_ship, 1.0, 0.0, uo=-5.0, vo=0.0) is None


def test_cross_current_crabbing() -> None:
    v_ship = 5.0  # m/s
    # Rumbo al Este (1, 0), corriente perpendicular al Norte (uo=0, vo=3)
    # El barco debe compensar 3 m/s hacia el Sur -> avance neto = sqrt(5^2 - 3^2) = 4 m/s
    sog_cross = speed_over_ground_mps(v_ship, 1.0, 0.0, uo=0.0, vo=3.0)
    assert sog_cross is not None and math.isclose(sog_cross, 4.0, rel_tol=1e-9)

    # Corriente transversal >= velocidad del barco -> imposible mantener el rumbo
    assert speed_over_ground_mps(v_ship, 1.0, 0.0, uo=0.0, vo=5.0) is None
    assert speed_over_ground_mps(v_ship, 1.0, 0.0, uo=0.0, vo=6.0) is None


# ---------------------------------------------------------------------------
# 3. Datos faltantes (NaN) y puntos sobre tierra
# ---------------------------------------------------------------------------
def test_missing_data_preserved_as_nan_never_zero() -> None:
    lats = np.array([20.80, 20.82, 20.84])
    lons = np.array([-86.90, -86.85, -86.80])
    uo = np.array(
        [
            [np.nan, 0.5, 0.5],
            [np.nan, 0.5, 0.5],
            [np.nan, 0.5, 0.5],
        ]
    )
    vo = np.array(
        [
            [np.nan, 0.2, 0.2],
            [np.nan, 0.2, 0.2],
            [np.nan, 0.2, 0.2],
        ]
    )
    field = _make_synthetic_field(lats, lons, uo, vo)

    assert np.all(field.land_or_missing_mask[:, 0])
    assert np.all(field.water_mask[:, 1:])
    assert np.all(np.isnan(field.magnitude[:, 0]))

    q = query_point_current(field, GeoPoint(lat=20.82, lon=-86.90), label="Punto Tierra")
    assert not q.is_valid_water
    assert math.isnan(q.uo) and math.isnan(q.vo) and math.isnan(q.magnitude)


def test_snap_point_on_land_reports_displacement_and_warning() -> None:
    lats = np.array([20.80, 20.82, 20.84])
    lons = np.array([-86.90, -86.85, -86.80])
    uo = np.array(
        [
            [np.nan, 0.3, 0.3],
            [np.nan, 0.3, 0.3],
            [np.nan, 0.3, 0.3],
        ]
    )
    vo = np.where(np.isnan(uo), np.nan, 0.1)
    field = _make_synthetic_field(lats, lons, uo, vo)

    land_point = GeoPoint(lat=20.82, lon=-86.90, name="Muelle en tierra")
    snapped = snap_point_to_water(field, land_point, label="Origen")

    assert snapped.was_on_land_or_missing is True
    assert (snapped.row, snapped.col) == (1, 1)
    assert snapped.displacement_m > 4000.0  # ~5.2 km entre -86.90 y -86.85
    assert snapped.warning is not None
    assert "NO garantiza" in snapped.warning


# ---------------------------------------------------------------------------
# 4. Restricciones topológicas: sin corte de esquinas y sin ruta posible
# ---------------------------------------------------------------------------
def test_no_diagonal_corner_cutting_across_land() -> None:
    # Cuadrícula 2x2 donde (0,0) y (1,1) son agua, pero (0,1) y (1,0) son tierra.
    # No debe permitirse saltar en diagonal de (0,0) a (1,1).
    lats = np.array([20.80, 20.82])
    lons = np.array([-86.85, -86.83])
    uo = np.array([[0.0, np.nan], [np.nan, 0.0]])
    vo = np.array([[0.0, np.nan], [np.nan, 0.0]])
    field = _make_synthetic_field(lats, lons, uo, vo)

    res = run_astar(field, start=(0, 0), goal=(1, 1), vessel_speed_mps=4.0, optimize_for="time")
    assert res.found is False


def test_unreachable_destination_blocked_by_land_wall() -> None:
    lats = np.linspace(20.80, 20.84, 5)
    lons = np.linspace(-86.85, -86.75, 5)
    uo = np.zeros((5, 5))
    vo = np.zeros((5, 5))
    # Muro completo de tierra en columna 2
    uo[:, 2] = np.nan
    vo[:, 2] = np.nan
    field = _make_synthetic_field(lats, lons, uo, vo)

    res = run_astar(field, start=(2, 0), goal=(2, 4), vessel_speed_mps=4.0, optimize_for="time")
    assert res.found is False
    assert res.distance_km is None
    assert res.time_minutes is None


# ---------------------------------------------------------------------------
# 5. Comparación consistente: ruta corta vs ruta optimizada A*
# ---------------------------------------------------------------------------
def test_optimized_route_beats_shortest_route_in_strong_head_current() -> None:
    # Cuadrícula 7x11:
    # - Fila central (r=3) tiene una fuerte corriente en contra (uo = -3.0 m/s).
    # - Fila inferior (r=1) tiene una corriente favorable al Este (uo = +2.0 m/s).
    # Embarcación a 8 nudos (~4.115 m/s) yendo de (3, 0) a (3, 10):
    # - La ruta más corta cruza recto por la fila 3 (SOG ~ 1.1 m/s).
    # - La ruta optimizada A* debe desviarse al corredor favorable y llegar mucho antes.
    lats = np.linspace(20.76, 20.88, 7)
    lons = np.linspace(-86.85, -86.65, 11)
    uo = np.zeros((7, 11))
    vo = np.zeros((7, 11))

    uo[3, 1:10] = -3.0  # Fuerte corriente en contra en el camino recto
    uo[2, 1:10] = -1.5
    uo[4, 1:10] = -1.5
    uo[1, 1:10] = 2.0  # Corredor favorable al sur

    field = _make_synthetic_field(lats, lons, uo, vo)
    origin = GeoPoint(lat=float(lats[3]), lon=float(lons[0]), name="A")
    dest = GeoPoint(lat=float(lats[3]), lon=float(lons[10]), name="B")

    report = optimize_maritime_route(field, origin, dest, vessel_speed_knots=8.0)

    assert report.shortest_navigable.found is True
    assert report.optimized_route.found is True
    assert report.shortest_navigable.time_seconds is not None
    assert report.optimized_route.time_seconds is not None

    # La ruta optimizada recorre más distancia, pero tarda estrictamente menos tiempo
    assert report.optimized_route.time_seconds < report.shortest_navigable.time_seconds
    assert report.optimized_route.distance_km > report.shortest_navigable.distance_km
    assert report.time_saved_minutes is not None and report.time_saved_minutes > 0.0
    assert report.time_saved_percent is not None and report.time_saved_percent > 10.0

    # Validar exportación JSON y orden GeoJSON [lon, lat]
    payload = build_export_payload(
        origin=origin,
        destination=dest,
        vessel_speed_knots=8.0,
        field=field,
        report=report,
        source_label="Sintético prueba",
    )
    assert payload["distancia_km"] == report.optimized_route.distance_km
    assert payload["tiempo_estimado_min"] == report.optimized_route.time_minutes
    assert payload["ahorro_tiempo_min"] == report.time_saved_minutes
    assert payload["geojson"]["geometry"]["type"] == "LineString"
    first_coord = payload["geojson"]["geometry"]["coordinates"][0]
    assert first_coord[0] < 0 and first_coord[1] > 0  # [lon, lat] para Caribe mexicano


def test_simulated_netcdf_roundtrip(tmp_path: Path) -> None:
    scenario = get_default_scenario()
    nc_path = tmp_path / "sim_test.nc"
    acq = generate_simulated_netcdf(scenario, output_path=nc_path)
    assert acq.is_simulated is True
    assert nc_path.exists()

    field = load_ocean_currents(nc_path, target_datetime=scenario.target_datetime)
    assert field.is_simulated is True
    assert np.any(field.land_or_missing_mask)
    assert np.any(field.water_mask)

    report = optimize_maritime_route(
        field=field,
        origin=scenario.origin,
        destination=scenario.destination,
        vessel_speed_knots=scenario.vessel_speed_knots,
    )
    assert report.origin_snap.was_on_land_or_missing is True
    assert report.optimized_route.found is True
    assert report.shortest_navigable.found is True
    assert report.optimized_route.time_seconds <= report.shortest_navigable.time_seconds
