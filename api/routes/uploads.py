"""CSV upload endpoint."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

from api.deps import get_run_dir

router = APIRouter(prefix="/uploads", tags=["uploads"])


@router.post("")
async def upload_csv(
    file: UploadFile = File(...),
    run_dir_base: Path = Depends(get_run_dir),
) -> dict[str, str]:
    upload_dir = run_dir_base.parent / "data"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / (file.filename or "upload.csv")
    with dest.open("wb") as f:
        f.write(await file.read())
    return {"path": str(dest.resolve())}
