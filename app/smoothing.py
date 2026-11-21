from __future__ import annotations

import numpy as np

from mapeo_ruta.ocean_data import OceanCurrentField


def cubic_bspline(coords: list[list[float]], samples_per_segment: int) -> list[list[float]]:
    """Evalúa una B-spline cúbica clamped usando los puntos recibidos como controles."""
    if len(coords) < 4:
        return coords
    if samples_per_segment < 1:
        raise ValueError("samples_per_segment debe ser al menos 1.")

    points = np.asarray(coords, dtype=float)
    degree = 3
    segment_count = len(points) - degree
    knots = np.concatenate(
        (
            np.zeros(degree + 1),
            np.arange(1, segment_count, dtype=float) / segment_count,
            np.ones(degree + 1),
        )
    )
    parameters = [
        (segment + sample / samples_per_segment) / segment_count
        for segment in range(segment_count)
        for sample in range(samples_per_segment)
    ]
    parameters.append(1.0)

    evaluated: list[list[float]] = []
    for t in parameters:
        if t == 1.0:
            basis = np.zeros(len(points), dtype=float)
            basis[-1] = 1.0
        else:
            basis = ((knots[:-1] <= t) & (t < knots[1:])).astype(float)
            for current_degree in range(1, degree + 1):
                count = len(basis) - 1
                indices = np.arange(count)
                left_denominator = knots[indices + current_degree] - knots[indices]
                right_denominator = knots[indices + current_degree + 1] - knots[indices + 1]
                left = np.divide(
                    (t - knots[indices]) * basis[:-1],
                    left_denominator,
                    out=np.zeros(count, dtype=float),
                    where=left_denominator != 0,
                )
                right = np.divide(
                    (knots[indices + current_degree + 1] - t) * basis[1:],
                    right_denominator,
                    out=np.zeros(count, dtype=float),
                    where=right_denominator != 0,
                )
                basis = left + right
        evaluated.append((basis @ points).tolist())

    # Preserve the clamped endpoints exactly despite floating point evaluation.
    evaluated[0] = list(coords[0])
    evaluated[-1] = list(coords[-1])
    return evaluated


def crosses_land(coords: list[list[float]], field: OceanCurrentField) -> bool:
    """True si algún punto cae en tierra o dato faltante según la celda más cercana."""
    water = field.water_mask
    for lon, lat in coords:
        row = int(np.argmin(np.abs(field.lats - lat)))
        col = int(np.argmin(np.abs(field.lons - lon)))
        if not water[row, col]:
            return True
    return False
