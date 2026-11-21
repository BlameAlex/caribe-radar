"""Cliente para descarga de corrientes oceánicas desde Copernicus Marine.

Gestiona la verificación de compatibilidad de la librería `copernicusmarine`,
la descarga de subconjuntos NetCDF (`uo`, `vo`) reutilizando archivos existentes,
el manejo de errores de red/autenticación y un modo simulado explícito para
pruebas sin conexión.
"""

from __future__ import annotations

import inspect
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

from mapeo_ruta.config import (
    DEFAULT_NETCDF_PATH,
    SIMULATED_NETCDF_PATH,
    ScenarioConfig,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AcquisitionResult:
    """Resultado de la etapa de adquisición de datos oceanográficos."""

    netcdf_path: Path
    is_simulated: bool
    reused_existing: bool
    source_label: str
    message: str


def check_copernicus_compatibility(dataset_id: str) -> dict[str, object]:
    """Comprueba si `copernicusmarine` está instalado y qué parámetros acepta `subset`."""
    try:
        import copernicusmarine
    except ImportError:
        return {
            "installed": False,
            "version": None,
            "dataset_id": dataset_id,
            "supported_subset_params": [],
            "message": "La librería `copernicusmarine` no está instalada.",
        }

    version = getattr(copernicusmarine, "__version__", "desconocida")
    sig = inspect.signature(copernicusmarine.subset)
    params = list(sig.parameters.keys())

    required_core = {
        "dataset_id",
        "variables",
        "minimum_longitude",
        "maximum_longitude",
        "minimum_latitude",
        "maximum_latitude",
        "start_datetime",
        "end_datetime",
        "minimum_depth",
        "maximum_depth",
    }
    missing = sorted(required_core - set(params))

    return {
        "installed": True,
        "version": version,
        "dataset_id": dataset_id,
        "supported_subset_params": params,
        "missing_required_params": missing,
        "compatible": len(missing) == 0,
        "message": (
            f"copernicusmarine v{version} compatible con subset()."
            if not missing
            else f"copernicusmarine v{version} carece de parámetros esperados: {missing}"
        ),
    }


def _has_copernicus_credentials() -> bool:
    """Detecta si existen credenciales configuradas vía variables de entorno o archivo local."""
    env_user = os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")
    env_pass = os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")
    if env_user and env_pass:
        return True
    creds_file = Path.home() / ".copernicusmarine" / ".copernicusmarine-credentials"
    return creds_file.exists()


def _parse_iso_datetime(dt_str: str) -> datetime:
    """Convierte una cadena ISO 8601 (con o sin sufijo Z) a datetime naive UTC."""
    cleaned = dt_str.strip().replace("Z", "")
    return datetime.fromisoformat(cleaned)


def download_copernicus_currents(
    scenario: ScenarioConfig,
    output_path: Path = DEFAULT_NETCDF_PATH,
    force_download: bool = False,
) -> AcquisitionResult:
    """Descarga las variables `uo` y `vo` desde Copernicus Marine para la región configurada.

    Reutiliza el archivo existente si `force_download=False`.
    Lanza excepciones descriptivas si no es posible autenticar o conectar.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force_download:
        return AcquisitionResult(
            netcdf_path=output_path,
            is_simulated=False,
            reused_existing=True,
            source_label="Copernicus Marine (archivo local reutilizado)",
            message=f"Se reutiliza el archivo NetCDF existente: {output_path}",
        )

    compat = check_copernicus_compatibility(scenario.copernicus.dataset_id)
    if not compat["installed"]:
        raise RuntimeError(str(compat["message"]))
    if not compat["compatible"]:
        raise RuntimeError(str(compat["message"]))

    if not _has_copernicus_credentials():
        raise PermissionError(
            "No se encontraron credenciales de Copernicus Marine. "
            "Configura las variables de entorno COPERNICUSMARINE_SERVICE_USERNAME y "
            "COPERNICUSMARINE_SERVICE_PASSWORD, o ejecuta `copernicusmarine login`."
        )

    import copernicusmarine

    target_dt = _parse_iso_datetime(scenario.target_datetime)
    half_window = timedelta(hours=max(6, scenario.time_window_hours // 2))
    start_dt = (target_dt - half_window).strftime("%Y-%m-%dT%H:%M:%S")
    end_dt = (target_dt + half_window).strftime("%Y-%m-%dT%H:%M:%S")

    subset_params = set(inspect.signature(copernicusmarine.subset).parameters.keys())
    kwargs: dict[str, object] = {
        "dataset_id": scenario.copernicus.dataset_id,
        "variables": list(scenario.copernicus.variables),
        "minimum_longitude": scenario.bbox.lon_min,
        "maximum_longitude": scenario.bbox.lon_max,
        "minimum_latitude": scenario.bbox.lat_min,
        "maximum_latitude": scenario.bbox.lat_max,
        "start_datetime": start_dt,
        "end_datetime": end_dt,
        "minimum_depth": scenario.copernicus.min_depth_m,
        "maximum_depth": scenario.copernicus.max_depth_m,
        "output_filename": output_path.name,
        "output_directory": str(output_path.parent),
    }

    # Compatibilidad entre copernicusmarine v1.x (`force_download`) y v2.x (`overwrite`)
    if "overwrite" in subset_params:
        kwargs["overwrite"] = force_download
    elif "force_download" in subset_params:
        kwargs["force_download"] = force_download

    try:
        copernicusmarine.subset(**kwargs)
    except Exception as exc:
        raise ConnectionError(
            f"Error al descargar el dataset '{scenario.copernicus.dataset_id}' "
            f"desde Copernicus Marine: {exc}"
        ) from exc

    if not output_path.exists():
        raise FileNotFoundError(
            f"Copernicus Marine finalizó sin generar el archivo esperado en {output_path}."
        )

    return AcquisitionResult(
        netcdf_path=output_path,
        is_simulated=False,
        reused_existing=False,
        source_label=f"Copernicus Marine ({scenario.copernicus.dataset_id})",
        message=f"Archivo descargado correctamente en {output_path}",
    )


def generate_simulated_netcdf(
    scenario: ScenarioConfig,
    output_path: Path = SIMULATED_NETCDF_PATH,
    grid_step_deg: float = 0.01,
) -> AcquisitionResult:
    """Genera un archivo NetCDF de prueba con corrientes simuladas claramente etiquetadas.

    NOTA: Estos datos NO provienen de Copernicus Marine. Se utilizan únicamente para
    desarrollo, pruebas sin conexión y demostración cuando no hay acceso al servicio real.

    Características del campo simulado (Puerto Morelos):
      - Máscara costera realista en el borde oeste (tierra = NaN, nunca 0).
      - Punto A (Muelle Comercial) cae deliberadamente en celda terrestre/costera
        para ejercitar la estrategia explícita de ajuste al nodo marino cercano.
      - Corriente de Yucatán hacia el norte/noreste con una zona de flujo favorable
        al sur de la línea recta y una corriente lateral/en contra en la franja central,
        permitiendo verificar la optimización A*.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lons = np.round(
        np.arange(scenario.bbox.lon_min, scenario.bbox.lon_max + grid_step_deg / 2, grid_step_deg),
        5,
    )
    lats = np.round(
        np.arange(scenario.bbox.lat_min, scenario.bbox.lat_max + grid_step_deg / 2, grid_step_deg),
        5,
    )
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Línea de costa aproximada de Quintana Roo cerca de Puerto Morelos (tierra al oeste)
    coast_lon = -86.872 + 0.25 * (lat_grid - 20.84)
    land_mask = lon_grid <= coast_lon

    # Campo de corrientes simulado (m/s):
    # - Flujo base hacia el norte (Corriente de Yucatán) que se intensifica mar adentro.
    # - Franja central (lat ~ 20.83) con componente oeste/norte fuerte (en contra de A -> B).
    # - Corredor sur (lat ~ 20.775) con componente este favorable hacia el destino.
    offshore_factor = np.clip((lon_grid - coast_lon) / 0.25, 0.0, 1.0)

    adverse_core = np.exp(-((lat_grid - 20.832) / 0.024) ** 2) * np.exp(
        -((lon_grid - (-86.75)) / 0.09) ** 2
    )
    favorable_jet = np.exp(-((lat_grid - 20.772) / 0.022) ** 2) * np.exp(
        -((lon_grid - (-86.74)) / 0.11) ** 2
    )

    uo_2d = 0.15 * offshore_factor - 1.35 * adverse_core + 1.15 * favorable_jet
    vo_2d = 0.45 * offshore_factor + 0.95 * adverse_core - 0.15 * favorable_jet

    # Aplicar máscara terrestre: tierra = NaN (nunca 0.0)
    uo_2d = np.where(land_mask, np.nan, uo_2d).astype(np.float32)
    vo_2d = np.where(land_mask, np.nan, vo_2d).astype(np.float32)

    target_dt = np.datetime64(_parse_iso_datetime(scenario.target_datetime), "ns")
    times = np.array([target_dt])
    depths = np.array([0.494025], dtype=np.float32)

    ds = xr.Dataset(
        data_vars={
            "uo": (
                ("time", "depth", "latitude", "longitude"),
                uo_2d[np.newaxis, np.newaxis, :, :],
                {
                    "long_name": "Eastward velocity (SIMULATED)",
                    "standard_name": "eastward_sea_water_velocity",
                    "units": "m s-1",
                },
            ),
            "vo": (
                ("time", "depth", "latitude", "longitude"),
                vo_2d[np.newaxis, np.newaxis, :, :],
                {
                    "long_name": "Northward velocity (SIMULATED)",
                    "standard_name": "northward_sea_water_velocity",
                    "units": "m s-1",
                },
            ),
        },
        coords={
            "time": times,
            "depth": (
                "depth",
                depths,
                {"long_name": "Depth", "units": "m", "positive": "down"},
            ),
            "latitude": (
                "latitude",
                lats.astype(np.float32),
                {"long_name": "Latitude", "units": "degrees_north"},
            ),
            "longitude": (
                "longitude",
                lons.astype(np.float32),
                {"long_name": "Longitude", "units": "degrees_east"},
            ),
        },
        attrs={
            "title": "Datos simulados de corrientes superficiales - Puerto Morelos (MODO PRUEBA)",
            "data_source": "SIMULATED_TEST_DATA",
            "is_simulated": 1,
            "warning": (
                "ARCHIVO SIMULADO PARA PRUEBAS Y DEMOSTRACION. "
                "NO CORRESPONDE A DATOS REALES DE COPERNICUS MARINE."
            ),
        },
    )

    ds.to_netcdf(output_path)

    return AcquisitionResult(
        netcdf_path=output_path,
        is_simulated=True,
        reused_existing=False,
        source_label="DATOS SIMULADOS DE PRUEBA (sin conexión a Copernicus Marine)",
        message=f"Archivo NetCDF simulado generado en {output_path}",
    )


def acquire_ocean_data(
    scenario: ScenarioConfig,
    netcdf_path: Optional[Path] = None,
    force_download: bool = False,
    allow_simulated_fallback: bool = True,
    use_simulated: bool = False,
) -> AcquisitionResult:
    """Orquesta la obtención de datos: archivo local, descarga Copernicus o fallback simulado."""
    if use_simulated:
        return generate_simulated_netcdf(scenario, output_path=SIMULATED_NETCDF_PATH)

    target_path = Path(netcdf_path) if netcdf_path is not None else DEFAULT_NETCDF_PATH

    try:
        return download_copernicus_currents(
            scenario=scenario,
            output_path=target_path,
            force_download=force_download,
        )
    except Exception as exc:
        if not allow_simulated_fallback:
            raise
        logger.warning(
            "No fue posible obtener datos reales de Copernicus Marine (%s). "
            "Activando modo de prueba con datos simulados.",
            exc,
        )
        sim_result = generate_simulated_netcdf(scenario, output_path=SIMULATED_NETCDF_PATH)
        return AcquisitionResult(
            netcdf_path=sim_result.netcdf_path,
            is_simulated=True,
            reused_existing=False,
            source_label=sim_result.source_label,
            message=f"Fallback a modo simulado ({exc}). {sim_result.message}",
        )
