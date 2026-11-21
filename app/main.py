from fastapi import FastAPI, HTTPException

from app.routing import DataAcquisitionError, RouteInputError, calculate_route
from app.schemas import RouteRequest, RouteResponse

app = FastAPI(title="Caribe Radar API", version="0.1.0")


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
