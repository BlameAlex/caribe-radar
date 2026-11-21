"""Read-only serving of static sargassum export files (produced by caribe-radar-ml)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path as PathParam, Query
from fastapi.responses import FileResponse

from app.settings import settings

router = APIRouter(prefix="/api/v1/sargassum", tags=["sargassum"])

RUN_ID = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"
RunId = Annotated[str, PathParam(pattern=RUN_ID)]
Minutes = Annotated[str, PathParam(pattern=r"^[0-9]{1,5}$")]

GEOJSON = "application/geo+json"
IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}


def _file(run_id: str, *parts: str) -> Path:
    """Path of a file inside the run dir; 404 if missing or if it resolves outside the export dir."""
    root = settings.sargassum_export_dir.resolve()
    path = root.joinpath(run_id, *parts).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return path


def _serve(path: Path, media_type: str) -> FileResponse:
    return FileResponse(path, media_type=media_type, headers=IMMUTABLE)


@router.get("/runs")
def list_runs() -> list[dict[str, object]]:
    root = settings.sargassum_export_dir.resolve()
    runs = []
    for manifest in sorted(root.glob("*/manifest.json")) if root.is_dir() else []:
        try:
            m = json.loads(manifest.read_text())
        except (OSError, ValueError):
            continue
        runs.append(
            {
                "run_id": m.get("run_id", manifest.parent.name),
                "t0": m.get("t0"),
                "step_minutes": m.get("step_minutes"),
                "n_frames": len(m.get("frames", [])),
                "bbox": m.get("bbox"),
                "manifest_url": f"/api/v1/sargassum/runs/{manifest.parent.name}/manifest",
            }
        )
    return runs


@router.get("/runs/{run_id}/manifest")
def get_manifest(run_id: RunId) -> FileResponse:
    return _serve(_file(run_id, "manifest.json"), "application/json")


@router.get("/runs/{run_id}/frames/{minutes}")
def get_frame(run_id: RunId, minutes: Minutes) -> FileResponse:
    return _serve(_file(run_id, "frames", f"t{int(minutes):04d}.geojson"), GEOJSON)


@router.get("/runs/{run_id}/arrivals")
def get_arrivals(run_id: RunId) -> FileResponse:
    return _serve(_file(run_id, "arrivals.json"), "application/json")


@router.get("/runs/{run_id}/arrivals/origin")
def get_arrivals_origin(run_id: RunId) -> FileResponse:
    return _serve(_file(run_id, "arrivals_origin.geojson"), GEOJSON)


@router.get("/collection-areas")
def get_collection_areas(run_id: Annotated[str, Query(pattern=RUN_ID)]) -> FileResponse:
    return _serve(_file(run_id, "collection_areas.geojson"), GEOJSON)
