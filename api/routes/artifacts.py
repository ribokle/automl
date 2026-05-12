"""Artifact serving."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(prefix="/artifacts", tags=["artifacts"])


@router.get("/{run_id}/{path:path}")
async def get_artifact(run_id: str, path: str) -> FileResponse:
    from api.deps import get_run_dir

    base = get_run_dir() / run_id / "artifacts"
    target = (base / path).resolve()
    if not str(target).startswith(str(base.resolve())):
        raise HTTPException(status_code=400, detail="invalid path")
    if not target.exists():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(target)
