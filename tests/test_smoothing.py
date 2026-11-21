from __future__ import annotations

import numpy as np

from app.smoothing import crosses_land, cubic_bspline
from mapeo_ruta.ocean_data import OceanCurrentField


def _field(water: np.ndarray) -> OceanCurrentField:
    return OceanCurrentField(
        lats=np.array([0.0, 1.0, 2.0]),
        lons=np.array([10.0, 11.0, 12.0]),
        uo=np.where(water, 0.0, np.nan),
        vo=np.where(water, 0.0, np.nan),
        selected_time="2026-11-20T00:00:00",
        selected_depth_m=0.5,
        dimensions={"latitude": 3, "longitude": 3},
        available_variables=["uo", "vo"],
        is_simulated=True,
    )


def test_clamped_spline_preserves_exact_endpoints_and_sample_count() -> None:
    controls = [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0], [4.0, 0.0]]
    samples_per_segment = 5
    result = cubic_bspline(controls, samples_per_segment)

    assert result[0] == controls[0]
    assert result[-1] == controls[-1]
    assert len(result) == (len(controls) - 3) * samples_per_segment + 1


def test_collinear_control_points_remain_collinear() -> None:
    controls = [[0.0, 2.0], [1.0, 4.0], [2.0, 6.0], [3.0, 8.0], [4.0, 10.0]]
    result = cubic_bspline(controls, samples_per_segment=4)
    assert all(np.isclose(lat, 2.0 * lon + 2.0) for lon, lat in result)


def test_fewer_than_four_points_are_returned_unchanged() -> None:
    controls = [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]
    result = cubic_bspline(controls, samples_per_segment=8)
    assert result is controls


def test_crosses_land_uses_nearest_grid_cell() -> None:
    water = np.ones((3, 3), dtype=bool)
    water[1, 1] = False
    field = _field(water)

    assert crosses_land([[10.05, 0.05], [11.0, 1.0]], field)
    assert not crosses_land([[10.05, 0.05], [12.0, 2.0]], field)
