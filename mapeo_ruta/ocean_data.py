"""Lectura y procesamiento de archivos NetCDF de corrientes oceánicas con xarray.

Etapa 3:
  - Inspecciona dimensiones y variables reales del archivo `.nc`.
  - Extrae la capa superficial y el instante temporal más cercano solicitado.
  - Preserva valores faltantes / celdas terrestres como NaN (nunca los convierte a 0).
  - Consulta corrientes cercanas a puntos de interés distinguiendo coordenadas
    solicitadas frente a coordenadas reales de las celdas del modelo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

from mapeo_ruta.config import GeoPoint
from mapeo_ruta.geographic_utils import current_direction_deg, current_magnitude, haversine_m


@dataclass
class OceanCurrentField:
    """Instantánea 2D de corrientes superficiales sobre una cuadrícula regular lat/lon."""

    lats: np.ndarray  # 1D ascendente, shape (n_lat,)
    lons: np.ndarray  # 1D ascendente, shape (n_lon,)
    uo: np.ndarray  # 2D float64, shape (n_lat, n_lon), NaN en tierra/faltantes
    vo: np.ndarray  # 2D float64, shape (n_lat, n_lon), NaN en tierra/faltantes
    selected_time: str
    selected_depth_m: Optional[float]
    dimensions: dict[str, int]
    available_variables: list[str]
    is_simulated: bool
    source_path: Optional[Path] = None

    @property
    def water_mask(self) -> np.ndarray:
        """Máscara booleana 2D: True donde tanto uo como vo son finitos (celda válida)."""
        return np.isfinite(self.uo) & np.isfinite(self.vo)

    @property
    def land_or_missing_mask(self) -> np.ndarray:
        """Máscara booleana 2D: True en celdas terrestres o con datos faltantes."""
        return ~self.water_mask

    @property
    def magnitude(self) -> np.ndarray:
        """Magnitud escalar sqrt(uo^2 + vo^2) en m/s, preservando NaN donde no hay datos."""
        mag = np.full_like(self.uo, np.nan, dtype=np.float64)
        valid = self.water_mask
        mag[valid] = np.hypot(self.uo[valid], self.vo[valid])
        return mag

    @property
    def max_current_speed_mps(self) -> float:
        """Velocidad máxima de corriente válida en toda la cuadrícula (para heurística A*)."""
        mag = self.magnitude
        if not np.any(np.isfinite(mag)):
            return 0.0
        return float(np.nanmax(mag))

    def summary_stats(self) -> dict[str, object]:
        """Resumen de cobertura de agua, celdas terrestres/faltantes y rango de magnitudes."""
        total = int(self.uo.size)
        valid_count = int(np.sum(self.water_mask))
        missing_count = total - valid_count
        mag = self.magnitude
        return {
            "dimensiones_archivo": self.dimensions,
            "variables_disponibles": self.available_variables,
            "forma_cuadricula": {"lat": int(len(self.lats)), "lon": int(len(self.lons))},
            "tiempo_seleccionado": self.selected_time,
            "profundidad_m": self.selected_depth_m,
            "es_simulado": self.is_simulated,
            "celdas_totales": total,
            "celdas_agua_validas": valid_count,
            "celdas_tierra_o_faltantes": missing_count,
            "magnitud_min_mps": round(float(np.nanmin(mag)), 4) if valid_count > 0 else None,
            "magnitud_max_mps": round(float(np.nanmax(mag)), 4) if valid_count > 0 else None,
            "magnitud_media_mps": round(float(np.nanmean(mag)), 4) if valid_count > 0 else None,
        }


@dataclass(frozen=True)
class PointCurrentQuery:
    """Resultado de consultar la corriente del modelo más cercana a un punto geográfico."""

    label: str
    requested_lat: float
    requested_lon: float
    model_row: int
    model_col: int
    model_lat: float
    model_lon: float
    offset_from_requested_m: float
    uo: float
    vo: float
    magnitude: float
    direction_deg: float
    is_valid_water: bool

    def format_report(self) -> str:
        """Formatea la consulta distinguiendo coordenadas solicitadas vs celda del modelo."""
        if self.is_valid_water:
            uo_str = f"{self.uo:.4f} m/s"
            vo_str = f"{self.vo:.4f} m/s"
            mag_str = f"{self.magnitude:.4f} m/s (hacia {self.direction_deg:.1f}°)"
            estado = "Agua válida en el modelo"
        else:
            uo_str = "NaN (tierra o sin dato)"
            vo_str = "NaN (tierra o sin dato)"
            mag_str = "NaN (tierra o sin dato)"
            estado = "TIERRA / SIN DATO en la celda más cercana del modelo"

        return (
            f"{self.label}:\n"
            f"  Latitud solicitada: {self.requested_lat:.6f}\n"
            f"  Longitud solicitada: {self.requested_lon:.6f}\n"
            f"  Latitud (celda modelo): {self.model_lat:.6f}\n"
            f"  Longitud (celda modelo): {self.model_lon:.6f}\n"
            f"  Desfase celda modelo: {self.offset_from_requested_m:.1f} m\n"
            f"  Estado celda: {estado}\n"
            f"  uo: {uo_str}\n"
            f"  vo: {vo_str}\n"
            f"  Magnitud de corriente: {mag_str}"
        )


def _find_coord_name(ds: xr.Dataset, candidates: tuple[str, ...]) -> Optional[str]:
    """Busca el nombre real de una coordenada/dimensión dentro de un Dataset xarray."""
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    return None


def load_ocean_currents(
    netcdf_path: Path,
    target_datetime: Optional[str] = None,
    uo_var: str = "uo",
    vo_var: str = "vo",
) -> OceanCurrentField:
    """Abre un archivo NetCDF con xarray y extrae el campo superficial 2D de (uo, vo).

    Comprueba dinámicamente los nombres de dimensiones (`latitude`/`lat`,
    `longitude`/`lon`, `time`, `depth`/`lev`) y nunca reemplaza NaN por cero.
    """
    netcdf_path = Path(netcdf_path)
    if not netcdf_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo NetCDF: {netcdf_path}")

    with xr.open_dataset(netcdf_path) as ds:
        dimensions = {str(k): int(v) for k, v in ds.sizes.items()}
        available_variables = list(ds.data_vars.keys())

        if uo_var not in ds.data_vars or vo_var not in ds.data_vars:
            raise KeyError(
                f"El archivo {netcdf_path} debe contener las variables '{uo_var}' y '{vo_var}'. "
                f"Variables encontradas: {available_variables}"
            )

        lat_name = _find_coord_name(ds, ("latitude", "lat", "y"))
        lon_name = _find_coord_name(ds, ("longitude", "lon", "x"))
        time_name = _find_coord_name(ds, ("time", "t"))
        depth_name = _find_coord_name(ds, ("depth", "lev", "z"))

        if lat_name is None or lon_name is None:
            raise KeyError(
                f"No se identificaron coordenadas de latitud/longitud en {list(ds.coords.keys())}."
            )

        sub = ds[[uo_var, vo_var]]

        # Selección temporal (instante más cercano al solicitado)
        selected_time = "sin_dimension_temporal"
        if time_name is not None and time_name in sub.dims:
            if target_datetime is not None:
                target_np = np.datetime64(target_datetime.strip().replace("Z", ""))
                sub = sub.sel({time_name: target_np}, method="nearest")
            else:
                sub = sub.isel({time_name: 0})
            selected_time = str(sub.coords[time_name].values)

        # Selección de capa superficial (profundidad mínima disponible)
        selected_depth_m: Optional[float] = None
        if depth_name is not None and depth_name in sub.dims:
            depth_vals = sub.coords[depth_name].values
            surface_idx = int(np.argmin(np.abs(depth_vals)))
            sub = sub.isel({depth_name: surface_idx})
            selected_depth_m = float(sub.coords[depth_name].values)
        elif depth_name is not None and depth_name in sub.coords:
            selected_depth_m = float(np.asarray(sub.coords[depth_name].values).flat[0])

        # Asegurar orden ascendente de latitud y longitud para indexación y gráficos
        sub = sub.sortby([lat_name, lon_name])

        lats = np.asarray(sub.coords[lat_name].values, dtype=np.float64)
        lons = np.asarray(sub.coords[lon_name].values, dtype=np.float64)
        uo = np.asarray(sub[uo_var].transpose(lat_name, lon_name).values, dtype=np.float64)
        vo = np.asarray(sub[vo_var].transpose(lat_name, lon_name).values, dtype=np.float64)

        is_simulated = bool(
            ds.attrs.get("is_simulated", 0) == 1
            or str(ds.attrs.get("data_source", "")).upper() == "SIMULATED_TEST_DATA"
        )

    return OceanCurrentField(
        lats=lats,
        lons=lons,
        uo=uo,
        vo=vo,
        selected_time=selected_time,
        selected_depth_m=selected_depth_m,
        dimensions=dimensions,
        available_variables=available_variables,
        is_simulated=is_simulated,
        source_path=netcdf_path,
    )


def query_point_current(
    field: OceanCurrentField,
    point: GeoPoint,
    label: Optional[str] = None,
) -> PointCurrentQuery:
    """Consulta la celda más cercana del modelo NetCDF a un punto geográfico."""
    row = int(np.argmin(np.abs(field.lats - point.lat)))
    col = int(np.argmin(np.abs(field.lons - point.lon)))

    model_lat = float(field.lats[row])
    model_lon = float(field.lons[col])
    uo_val = float(field.uo[row, col])
    vo_val = float(field.vo[row, col])
    is_valid = math.isfinite(uo_val) and math.isfinite(vo_val)

    offset_m = haversine_m(point.lat, point.lon, model_lat, model_lon)
    mag = current_magnitude(uo_val, vo_val) if is_valid else float("nan")
    direction = current_direction_deg(uo_val, vo_val) if is_valid else float("nan")

    return PointCurrentQuery(
        label=label or point.name or "Punto",
        requested_lat=point.lat,
        requested_lon=point.lon,
        model_row=row,
        model_col=col,
        model_lat=model_lat,
        model_lon=model_lon,
        offset_from_requested_m=offset_m,
        uo=uo_val,
        vo=vo_val,
        magnitude=mag,
        direction_deg=direction,
        is_valid_water=is_valid,
    )


def refine_field_for_routing(
    field: OceanCurrentField,
    factor: int = 1,
) -> OceanCurrentField:
    """Opcionalmente refina una cuadrícula gruesa (ej. 0.083°) mediante interpolación bilineal.

    Regla estricta de integridad:
      - Si `factor <= 1`, devuelve el campo original sin modificar.
      - Solo se interpolan puntos rodeados completamente por celdas de agua válidas;
        cualquier celda influenciada por NaN/tierra permanece NaN (nunca extrapola hacia tierra).
    """
    if factor <= 1:
        return field
    if len(field.lats) < 2 or len(field.lons) < 2:
        return field

    new_n_lat = (len(field.lats) - 1) * factor + 1
    new_n_lon = (len(field.lons) - 1) * factor + 1
    new_lats = np.linspace(field.lats[0], field.lats[-1], new_n_lat)
    new_lons = np.linspace(field.lons[0], field.lons[-1], new_n_lon)

    # Bilineal en numpy (sin scipy), renormalizada sobre las esquinas válidas. La máscara de agua es
    # la de la celda gruesa más cercana: así la conectividad del agua nunca es menor que en la malla
    # original (con NaN propagado se cortaban los pasos diagonales y quedaban bolsas aisladas) y no se
    # extrapola más allá de la mitad de una celda gruesa de agua.
    def _bilinear(values: np.ndarray) -> np.ndarray:
        def axis(old: np.ndarray, new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            lo = np.clip(np.searchsorted(old, new, side="right") - 1, 0, len(old) - 2)
            return lo, (new - old[lo]) / (old[lo + 1] - old[lo])

        r0, wr = axis(field.lats, new_lats)
        c0, wc = axis(field.lons, new_lons)
        wr2, wc2 = wr[:, None], wc[None, :]
        total = np.zeros((len(new_lats), len(new_lons)))
        norm = np.zeros_like(total)
        for dr, wrow in ((0, 1 - wr2), (1, wr2)):
            for dc, wcol in ((0, 1 - wc2), (1, wc2)):
                corner = values[(r0 + dr)[:, None], (c0 + dc)[None, :]]
                weight = np.where(np.isfinite(corner), wrow * wcol, 0.0)
                total += weight * np.nan_to_num(corner)
                norm += weight
        nearest = np.isfinite(
            values[(r0 + (wr > 0.5))[:, None], (c0 + (wc > 0.5))[None, :]]
        )
        return np.where(nearest & (norm > 1e-12), total / np.maximum(norm, 1e-12), np.nan)

    uo_interp = _bilinear(field.uo)
    vo_interp = _bilinear(field.vo)

    return OceanCurrentField(
        lats=new_lats,
        lons=new_lons,
        uo=np.asarray(uo_interp, dtype=np.float64),
        vo=np.asarray(vo_interp, dtype=np.float64),
        selected_time=field.selected_time,
        selected_depth_m=field.selected_depth_m,
        dimensions=field.dimensions,
        available_variables=field.available_variables,
        is_simulated=field.is_simulated,
        source_path=field.source_path,
    )
