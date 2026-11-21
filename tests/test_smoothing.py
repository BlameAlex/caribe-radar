from __future__ import annotations

import math

import numpy as np

from app.smoothing import catmull_rom, crosses_land, simplify_path, smooth_route
from app.landmask import LandMask
from mapeo_ruta.ocean_data import OceanCurrentField


def _field(water: np.ndarray) -> OceanCurrentField:
    n_rows, n_cols = water.shape
    return OceanCurrentField(
        lats=np.arange(n_rows) * 0.1,
        lons=10.0 + np.arange(n_cols) * 0.1,
        uo=np.where(water, 0.0, np.nan),
        vo=np.where(water, 0.0, np.nan),
        selected_time="2026-11-20T00:00:00",
        selected_depth_m=0.5,
        dimensions={"latitude": n_rows, "longitude": n_cols},
        available_variables=["uo", "vo"],
        is_simulated=True,
    )


def _turns(coords: list[list[float]]) -> list[float]:
    out = []
    for a, b, c in zip(coords, coords[1:], coords[2:]):
        d = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(b[1] - a[1], b[0] - a[0]))
        out.append(abs((d + 180) % 360 - 180))
    return out


def _l_path() -> list[list[float]]:
    # camino A* en L: 10 nodos hacia el este y luego 10 hacia el norte
    return [[10.0 + 0.1 * i, 0.0] for i in range(10)] + [[10.9, 0.1 * j] for j in range(1, 10)]


def test_crosses_land_uses_nearest_grid_cell() -> None:
    water = np.ones((3, 3), dtype=bool)
    water[1, 1] = False
    field = _field(water)

    assert crosses_land([[10.01, 0.01], [10.1, 0.1]], field)
    assert not crosses_land([[10.01, 0.01], [10.2, 0.2]], field)


def test_collinear_run_collapses_to_endpoints() -> None:
    field = _field(np.ones((12, 12), dtype=bool))
    raw = [[10.0 + 0.1 * i, 0.0] for i in range(10)]
    assert simplify_path(raw, field) == [raw[0], raw[-1]]


def test_simplify_keeps_corner_when_land_blocks_shortcut() -> None:
    water = np.ones((12, 12), dtype=bool)
    water[2:12, 0:9] = False  # tierra en el interior de la L
    field = _field(water)
    raw = _l_path()
    kept = simplify_path(raw, field)
    assert len(kept) == 3 and kept[0] == raw[0] and kept[-1] == raw[-1] and kept[1] in raw[8:12]


def test_l_path_becomes_gentle_curve_in_water_with_exact_endpoints() -> None:
    water = np.ones((12, 12), dtype=bool)
    water[2:12, 0:9] = False
    field = _field(water)
    raw = _l_path()
    result = smooth_route(raw, field)

    assert result.applied and result.simplified_points == 3
    assert result.coordinates[0] == raw[0] and result.coordinates[-1] == raw[-1]
    assert not crosses_land(result.coordinates, field)
    assert max(_turns(result.coordinates)) < 20  # la L original gira 90° de golpe
    assert result.max_deviation_km > 0


def test_curve_touching_land_falls_back_to_raw() -> None:
    water = np.ones((12, 12), dtype=bool)
    water[1, 10] = False  # tierra pegada a la esquina: la curva redondeada la toca, la L no
    field = _field(water)
    raw = [[10.0 + 0.1 * i, 0.0] for i in range(10)] + [[10.9, 0.0 + 0.1 * j] for j in range(1, 10)]
    raw = [c for c in raw if not (abs(c[0] - 11.0) < 1e-9 and abs(c[1] - 0.1) < 1e-9)]
    result = smooth_route(raw, field)
    if not result.applied:
        assert result.coordinates == raw and "tierra" in result.reason
    assert not crosses_land(result.coordinates, field)


def test_catmull_rom_passes_through_controls() -> None:
    controls = [[10.0, 0.0], [10.5, 0.2], [10.9, 0.9]]
    curve = catmull_rom(controls)
    assert curve[0] == controls[0] and curve[-1] == controls[-1]
    assert any(np.allclose(p, controls[1]) for p in curve)


def test_connectors_keep_requested_endpoints_and_smooth_joints() -> None:
    field = _field(np.ones((12, 12), dtype=bool))
    raw = [[10.2 + 0.1 * i, 0.3] for i in range(6)]
    origin, destination = [10.1, 0.35], [10.8, 0.45]
    result = smooth_route(raw, field, origin, destination)

    assert result.applied
    assert result.coordinates[0] == origin and result.coordinates[-1] == destination
    assert max(_turns(result.coordinates)) < 15  # sin quiebres en las uniones conector/A*


def test_connector_over_coast_is_exempt_but_astar_part_is_checked() -> None:
    water = np.ones((12, 12), dtype=bool)
    water[:, 0:2] = False  # franja costera sin datos de ~0.2°
    field = _field(water)
    raw = [[10.2 + 0.1 * i, 0.3] for i in range(6)]
    origin_on_land = [10.0, 0.3]  # conector de ~22 km sobre la franja sin datos
    assert smooth_route(raw, field, origin_on_land, [10.8, 0.3]).applied

    water[2:4, 3:8] = False  # tierra sobre el tramo A* (curva) → se rechaza
    field = _field(water)
    blocked = smooth_route(raw, field, origin_on_land, [10.8, 0.3])
    assert not blocked.applied and blocked.coordinates[0] == origin_on_land


def _mask(land: np.ndarray) -> LandMask:
    n_rows, n_cols = land.shape
    return LandMask(np.arange(n_rows) * 0.01, 10.0 + np.arange(n_cols) * 0.01, land)


def test_landmask_reports_coverage_and_land() -> None:
    land = np.zeros((10, 10), dtype=bool)
    land[5, 5] = True
    covered, is_land = _mask(land).query(np.array([[10.05, 0.05], [10.02, 0.02], [11.0, 0.05]]))
    assert covered.tolist() == [True, True, False]
    assert is_land.tolist() == [True, False, False]


def test_overshooting_end_node_is_dropped_instead_of_hairpin() -> None:
    field = _field(np.ones((12, 12), dtype=bool))
    raw = [[10.2 + 0.05 * i, 0.3] for i in range(10)]  # el A* llega a 10.65, pasado el destino
    destination = [10.55, 0.3]
    result = smooth_route(raw, field, [10.15, 0.3], destination)
    assert result.applied and result.coordinates[-1] == destination
    assert max(_turns(result.coordinates)) < 5


def test_fine_mask_rejects_curve_over_land_beyond_port_cell() -> None:
    field = _field(np.ones((12, 12), dtype=bool))
    land = np.zeros((120, 120), dtype=bool)
    land[28:33, 3:7] = True  # tierra sobre el conector, a ~3-7 km del origen (celdas de ~1 km)
    mask = _mask(land)
    raw = [[10.8, 0.3], [10.9, 0.3]]
    blocked = smooth_route(raw, field, [10.0, 0.3], [10.95, 0.3], mask=mask)
    assert not blocked.applied
    assert smooth_route(raw, field, [10.0, 0.6], [10.95, 0.6], mask=mask).applied
