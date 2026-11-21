from pydantic import BaseModel, ConfigDict, Field


class Point(BaseModel):
    model_config = ConfigDict(extra="ignore")

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class RouteRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    origin: Point
    destination: Point


class SmoothingMeta(BaseModel):
    applied: bool
    method: str
    raw_points: int
    simplified_points: int = 0
    max_deviation_km: float = 0.0
    reason: str | None = None


class RouteMeta(BaseModel):
    distance_km: float
    time_minutes: float
    data_source: str
    selected_time: str
    snap_warnings: list[str]
    disclaimer: str
    smoothing: SmoothingMeta


class RouteGeometry(BaseModel):
    type: str = "LineString"
    coordinates: list[list[float]]


class RouteResponse(BaseModel):
    route: RouteGeometry
    meta: RouteMeta
