from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.routing import DataAcquisitionError, RouteInputError, calculate_route
from app.sargassum import router as sargassum_router
from app.schemas import RouteRequest, RouteResponse
from app.settings import settings

app = FastAPI(title="Caribe Radar API", version="0.1.0")
app.add_middleware(GZipMiddleware, minimum_size=1024)
if settings.cors_origins:
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"], allow_headers=["*"])
app.include_router(sargassum_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/routes", response_model=RouteResponse)
def create_route(request: RouteRequest) -> dict[str, object]:
    try:
        result = calculate_route(request.origin, request.destination)
    except RouteInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DataAcquisitionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if "not_found" in result:
        raise HTTPException(status_code=404, detail=result["not_found"])
    return result
