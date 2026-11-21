"""Departure ports (real ports, stored in data/ports.json)."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

PORTS_FILE = Path(__file__).resolve().parent.parent / "data" / "ports.json"

router = APIRouter(prefix="/api/v1", tags=["ports"])


class PortOut(BaseModel):
    id: str
    name: str
    short_name: str
    municipality: str
    lat: float
    lng: float
    description: str
    facilities: str


def load_ports() -> list[dict[str, object]]:
    return json.loads(PORTS_FILE.read_text())


@router.get("/ports", response_model=list[PortOut])
def list_ports() -> list[dict[str, object]]:
    try:
        return load_ports()
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="ports file unavailable") from exc
