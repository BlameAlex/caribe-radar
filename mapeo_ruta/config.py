"""Configuración central del modelo de optimización de rutas marítimas.

Define coordenadas iniciales (Puerto Morelos), parámetros de embarcación,
región geográfica de análisis y configuración de Copernicus Marine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEFAULT_NETCDF_PATH = DATA_DIR / "corrientes_puerto_morelos.nc"
SIMULATED_NETCDF_PATH = DATA_DIR / "corrientes_puerto_morelos_simulado.nc"
DEFAULT_JSON_OUTPUT = OUTPUTS_DIR / "ruta_resultado.json"
DEFAULT_CURRENTS_PLOT = OUTPUTS_DIR / "mapa_corrientes.png"
DEFAULT_ROUTES_PLOT = OUTPUTS_DIR / "mapa_rutas.png"

# Constantes físicas y geográficas
KNOTS_TO_MPS = 1852.0 / 3600.0  # 0.5144444444... m/s exacto (nudo internacional)
EARTH_RADIUS_M = 6_371_000.0

# Aviso de seguridad obligatorio para resultados de demostración
NAVIGATION_DISCLAIMER = (
    "AVISO: Esta ruta ha sido calculada con fines de demostración para un MVP "
    "utilizando datos oceanográficos de resolución media y sin cartas náuticas, "
    "batimetría, arrecifes ni avisos a los marinos. NO es una ruta validada ni "
    "segura para navegación real."
)


@dataclass(frozen=True)
class GeoPoint:
    """Coordenada geográfica en grados decimales (WGS84)."""

    lat: float
    lon: float
    name: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, float]:
        return {"lat": round(self.lat, 6), "lon": round(self.lon, 6)}


@dataclass(frozen=True)
class BoundingBox:
    """Región geográfica rectangular en grados decimales."""

    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float

    def contains(self, point: GeoPoint) -> bool:
        """Verifica si un punto geográfico se encuentra dentro de la región."""
        return (
            self.lon_min <= point.lon <= self.lon_max
            and self.lat_min <= point.lat <= self.lat_max
        )


@dataclass(frozen=True)
class CopernicusConfig:
    """Parámetros del producto y dataset de Copernicus Marine."""

    product_id: str = "GLOBAL_ANALYSISFORECAST_PHY_001_024"
    dataset_id: str = "cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i"
    variables: tuple[str, ...] = ("uo", "vo")
    min_depth_m: float = 0.0
    max_depth_m: float = 1.0


@dataclass(frozen=True)
class ScenarioConfig:
    """Configuración completa de un escenario de cálculo de ruta."""

    origin: GeoPoint
    destination: GeoPoint
    vessel_speed_knots: float
    target_datetime: str  # Formato ISO UTC, ej. "2026-11-20T12:00:00"
    bbox: BoundingBox
    copernicus: CopernicusConfig = CopernicusConfig()
    time_window_hours: int = 12


# Escenario por defecto: Puerto Morelos, Quintana Roo, México
DEFAULT_ORIGIN = GeoPoint(
    lat=20.841667,
    lon=-86.878056,
    name="Puerto Morelos (Referencia Muelle Comercial)",
    description=(
        "Referencia geográfica del Muelle Portuario Comercial de Puerto Morelos. "
        "No corresponde a una posición de salida validada náuticamente."
    ),
)

DEFAULT_DESTINATION = GeoPoint(
    lat=20.808922,
    lon=-86.634994,
    name="Zona de recolección de sargazo",
    description="Ubicación objetivo para la recolección de sargazo.",
)

DEFAULT_BBOX = BoundingBox(
    lon_min=-86.96,
    lon_max=-86.55,
    lat_min=20.74,
    lat_max=20.92,
)

DEFAULT_VESSEL_SPEED_KNOTS = 8.0
DEFAULT_TARGET_DATETIME = "2026-11-20T12:00:00"


def get_default_scenario() -> ScenarioConfig:
    """Devuelve el escenario de prueba configurado para Puerto Morelos."""
    return ScenarioConfig(
        origin=DEFAULT_ORIGIN,
        destination=DEFAULT_DESTINATION,
        vessel_speed_knots=DEFAULT_VESSEL_SPEED_KNOTS,
        target_datetime=DEFAULT_TARGET_DATETIME,
        bbox=DEFAULT_BBOX,
    )
