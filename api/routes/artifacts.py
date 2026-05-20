"""Artifact serving.

Serves any file under `runs/<run_id>/`. Path-traversal is blocked by
verifying the resolved path stays inside the run directory.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from api.auth import require_auth
from api.deps import get_run_dir

router = APIRouter(prefix="/artifacts", tags=["artifacts"], dependencies=[Depends(require_auth)])


@router.get("/{run_id}/{path:path}")
async def get_artifact(run_id: str, path: str) -> FileResponse:
    base = (get_run_dir() / run_id).resolve()
    target = (base / path).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(target)
