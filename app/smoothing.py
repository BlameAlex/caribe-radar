from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.landmask import LandMask
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


def haversine_km_array(lon1, lat1, lon2, lat2):
    """Haversine vectorizada (km); admite escalares y arreglos."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(np.asarray(lon2) - lon1) / 2) ** 2
    return 12742.0 * np.arcsin(np.sqrt(h))


def haversine_km(a: list[float], b: list[float]) -> float:
    """Distancia (km) entre dos puntos [lon, lat]."""
    return float(haversine_km_array(a[0], a[1], b[0], b[1]))


class LandGuard:
    """Decide qué muestras [lon, lat] están sobre tierra.

    - Donde hay máscara de alta resolución: manda ella (1 km, costa real).
    - Fuera de su cobertura: celda más cercana del modelo.
    - Alrededor de origen/destino (puertos en la costa) no se evalúa: `coast_km` bajo la máscara fina;
      sin ella, el radio dado en `ends` (la franja costera sin datos del modelo es más ancha que la
      celda y no se puede verificar).
    """

    def __init__(
        self,
        field: OceanCurrentField,
        mask: LandMask | None = None,
        ends: list[tuple[list[float], float]] | None = None,
        coast_km: float = 1.0,
    ) -> None:
        self.field, self.mask, self.ends, self.coast_km = field, mask, ends or [], coast_km

    def blocked(self, pts: np.ndarray) -> np.ndarray:
        model_land = ~self.field.water_mask[
            _nearest_index(self.field.lats, pts[:, 1]), _nearest_index(self.field.lons, pts[:, 0])
        ]
        if self.mask is not None:
            covered, fine_land = self.mask.query(pts)
            land = np.where(covered, fine_land, model_land)
        else:
            covered, land = np.zeros(len(pts), bool), model_land
        for (lon, lat), exempt_km in self.ends:
            near = haversine_km_array(lon, lat, pts[:, 0], pts[:, 1])
            land = land & ~(near <= np.where(covered, self.coast_km, exempt_km))
        return land

    def step_deg(self) -> float:
        cell = min(float(np.min(np.diff(self.field.lons))), float(np.min(np.diff(self.field.lats))))
        if self.mask is not None:
            cell = min(cell, self.mask.dlat)
        return cell / 2.0

    def segment_clear(self, a: list[float], b: list[float]) -> bool:
        span = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
        t = np.linspace(0.0, 1.0, max(2, math.ceil(span / self.step_deg()) + 1))
        samples = np.column_stack((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
        return not bool(self.blocked(samples).any())

    def path_clear(self, curve: list[list[float]]) -> bool:
        """Muestras y tramos entre muestras consecutivas libres de tierra."""
        return all(self.segment_clear(curve[i], curve[i + 1]) for i in range(len(curve) - 1))


def simplify_path(
    coords: list[list[float]],
    field: OceanCurrentField | None = None,
    max_hop: int | None = None,
    guard: LandGuard | None = None,
) -> list[list[float]]:
    """String pulling: conserva un nodo solo si el segmento al siguiente nodo conservado cruzaría tierra.

    `max_hop` limita cuántos nodos originales puede saltar un segmento (None = sin límite).
    """
    guard = guard or LandGuard(field)
    n = len(coords)
    if n < 3:
        return list(coords)
    kept = [coords[0]]
    i = 0
    while i < n - 1:
        j = n - 1 if max_hop is None else min(n - 1, i + max_hop)
        while j > i + 1 and not guard.segment_clear(coords[i], coords[j]):
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


def smooth_route(
    raw: list[list[float]],
    field: OceanCurrentField,
    origin: list[float] | None = None,
    destination: list[float] | None = None,
    exempt_km: float = 2.0,
    mask: LandMask | None = None,
) -> SmoothingResult:
    """Simplifica por línea de visión el camino [origen] + A* + [destino] y lo suaviza con Catmull-Rom.

    `origin`/`destination` ([lon, lat] solicitados) forman parte del mismo camino: los conectores se
    simplifican y suavizan junto con el A*, así un nodo A* que sobrepasa el destino se descarta y no
    hay quiebres en las uniones. Cada curva se valida con `LandGuard` (máscara fina si hay, si no el
    modelo). Intenta simplificación total, luego saltos de hasta 3 nodos, luego sin simplificar; si
    ninguna curva está libre de tierra, devuelve el A* con conectores rectos.
    `simplified_points` = puntos de control de la curva final, extremos incluidos.
    """
    head = [list(origin)] if origin and list(origin) != raw[0] else []
    tail = [list(destination)] if destination and list(destination) != raw[-1] else []
    full = head + raw + tail
    if len(full) < 3:
        return SmoothingResult(raw, False, len(raw), 0.0, "La ruta tiene menos de 3 puntos; no hay nada que suavizar.")
    ends = []
    if head:
        ends.append((head[0], max(exempt_km, haversine_km(head[0], raw[0]))))
    if tail:
        ends.append((tail[0], max(exempt_km, haversine_km(raw[-1], tail[0]))))
    guard = LandGuard(field, mask, ends)
    for max_hop in (None, 3, 1):
        simplified = simplify_path(full, max_hop=max_hop, guard=guard)
        curve = catmull_rom(simplified)
        if guard.path_clear(curve):
            return SmoothingResult(curve, True, len(simplified), max_deviation_km(curve, full), None)
    return SmoothingResult(
        full, False, len(raw), 0.0, "El suavizado toca tierra o celdas sin datos; se conserva la ruta A* original."
    )
