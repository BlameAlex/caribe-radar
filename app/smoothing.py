from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mapeo_ruta.ocean_data import OceanCurrentField

_KM_PER_DEG = 111.32
METHOD = "los_simplify_catmull_rom"


def _nearest_index(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Índice de la celda más cercana en un eje ascendente (mismo criterio que argmin|axis-v|)."""
    idx = np.clip(np.searchsorted(axis, values), 1, len(axis) - 1) if len(axis) > 1 else np.zeros(len(values), int)
    if len(axis) > 1:
        idx = np.where(np.abs(values - axis[idx - 1]) <= np.abs(values - axis[idx]), idx - 1, idx)
    return idx


def crosses_land(coords: list[list[float]], field: OceanCurrentField) -> bool:
    """True si algún punto cae en tierra o dato faltante según la celda más cercana."""
    if not coords:
        return False
    pts = np.asarray(coords, dtype=float)
    rows = _nearest_index(field.lats, pts[:, 1])
    cols = _nearest_index(field.lons, pts[:, 0])
    return not bool(field.water_mask[rows, cols].all())


def _segment_is_clear(a: list[float], b: list[float], field: OceanCurrentField) -> bool:
    """Visibilidad directa sobre el agua: muestrea a-b a un cuarto de celda y prueba cada muestra."""
    cell = min(
        float(np.min(np.diff(field.lons))) if len(field.lons) > 1 else 1.0,
        float(np.min(np.diff(field.lats))) if len(field.lats) > 1 else 1.0,
    )
    span = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    steps = max(2, math.ceil(span / (cell / 4.0)) + 1)
    t = np.linspace(0.0, 1.0, steps)
    samples = np.column_stack((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return not crosses_land(samples.tolist(), field)


def simplify_path(
    coords: list[list[float]], field: OceanCurrentField, max_hop: int | None = None
) -> list[list[float]]:
    """String pulling: conserva un nodo solo si el segmento al siguiente nodo conservado cruzaría tierra.

    `max_hop` limita cuántos nodos originales puede saltar un segmento (None = sin límite).
    """
    n = len(coords)
    if n < 3:
        return list(coords)
    kept = [coords[0]]
    i = 0
    while i < n - 1:
        far = n - 1 if max_hop is None else min(n - 1, i + max_hop)
        j = far
        while j > i + 1 and not _segment_is_clear(coords[i], coords[j], field):
            j -= 1
        kept.append(coords[j])
        i = j
    return kept


def catmull_rom(coords: list[list[float]], samples_per_km: float = 0.7, min_samples: int = 8) -> list[list[float]]:
    """Catmull-Rom centrípeta (Barry-Goldman) que pasa por todos los puntos; extremos exactos."""
    if len(coords) < 2:
        return [list(c) for c in coords]
    mean_lat = math.radians(float(np.mean([c[1] for c in coords])))
    scale = np.array([math.cos(mean_lat), 1.0])  # lon, lat → plano local
    pts = np.asarray(coords, dtype=float) * scale
    padded = np.vstack((2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]))

    out: list[np.ndarray] = []
    for k in range(len(pts) - 1):
        p0, p1, p2, p3 = padded[k : k + 4]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-12) ** 0.5
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-12) ** 0.5
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-12) ** 0.5
        seg_km = float(np.linalg.norm(p2 - p1)) * _KM_PER_DEG
        count = max(min_samples, math.ceil(seg_km * samples_per_km))
        for t in np.linspace(t1, t2, count, endpoint=False):
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    out.append(pts[-1])
    result = (np.asarray(out) / scale).tolist()
    result[0], result[-1] = list(coords[0]), list(coords[-1])  # extremos exactos
    return result


def max_deviation_km(curve: list[list[float]], reference: list[list[float]]) -> float:
    """Distancia máxima (km) de las muestras de `curve` a la polilínea `reference`."""
    mean_lat = math.radians(float(np.mean([c[1] for c in reference])))
    scale = np.array([math.cos(mean_lat), 1.0]) * _KM_PER_DEG
    p = np.asarray(curve, dtype=float) * scale
    ref = np.asarray(reference, dtype=float) * scale
    a, b = ref[:-1], ref[1:]
    ab = b - a
    denom = np.maximum((ab * ab).sum(axis=1), 1e-18)
    t = np.clip(((p[:, None, :] - a) * ab).sum(axis=2) / denom, 0.0, 1.0)
    proj = a + t[..., None] * ab
    return float(np.linalg.norm(p[:, None, :] - proj, axis=2).min(axis=1).max())


@dataclass(frozen=True)
class SmoothingResult:
    coordinates: list[list[float]]
    applied: bool
    simplified_points: int
    max_deviation_km: float
    reason: str | None


def smooth_route(raw: list[list[float]], field: OceanCurrentField) -> SmoothingResult:
    """Simplifica el camino A* por línea de visión y lo suaviza con Catmull-Rom, validando contra tierra.

    Intenta simplificación total, luego una menos agresiva (saltos de hasta 3 nodos); si ninguna
    curva está libre de tierra, devuelve el camino A* original.
    """
    if len(raw) < 3:
        return SmoothingResult(raw, False, len(raw), 0.0, "La ruta tiene menos de 3 puntos; no hay nada que suavizar.")
    for max_hop in (None, 3):
        simplified = simplify_path(raw, field, max_hop)
        curve = catmull_rom(simplified)
        if not crosses_land(curve, field):
            return SmoothingResult(curve, True, len(simplified), max_deviation_km(curve, raw), None)
    return SmoothingResult(
        raw, False, len(raw), 0.0, "El suavizado toca tierra o celdas sin datos; se conserva la ruta A* original."
    )
