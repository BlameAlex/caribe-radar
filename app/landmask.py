from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import xarray as xr


class LandMask:
    """Máscara de tierra de alta resolución (~1 km, variable `land_mask`, 1 = tierra)."""

    def __init__(self, lats: np.ndarray, lons: np.ndarray, land: np.ndarray) -> None:
        self.lats, self.lons, self.land = lats, lons, land.astype(bool)
        self.dlat = float(np.median(np.diff(lats)))
        self.dlon = float(np.median(np.diff(lons)))

    def query(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(cubierto, tierra) por punto [lon, lat]; fuera de la cobertura `cubierto` es False."""
        lon, lat = pts[:, 0], pts[:, 1]
        covered = (
            (lat >= self.lats[0] - self.dlat / 2)
            & (lat <= self.lats[-1] + self.dlat / 2)
            & (lon >= self.lons[0] - self.dlon / 2)
            & (lon <= self.lons[-1] + self.dlon / 2)
        )
        rows = np.clip(np.rint((lat - self.lats[0]) / self.dlat).astype(int), 0, len(self.lats) - 1)
        cols = np.clip(np.rint((lon - self.lons[0]) / self.dlon).astype(int), 0, len(self.lons) - 1)
        return covered, self.land[rows, cols] & covered


@lru_cache(maxsize=2)
def load_land_mask(path: Path | None) -> LandMask | None:
    """Carga la máscara si hay ruta configurada y existe; si no, None (se usa la malla del modelo)."""
    if path is None or not Path(path).exists():
        return None
    with xr.open_dataset(path) as ds:
        return LandMask(ds["latitude"].values, ds["longitude"].values, ds["land_mask"].values)
