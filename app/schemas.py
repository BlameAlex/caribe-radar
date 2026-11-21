from pydantic import BaseModel, ConfigDict, Field


class Point(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class RouteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    origin: Point
    destination: Point


class SmoothingMeta(BaseModel):
    applied: bool
    method: str
    raw_points: int
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
