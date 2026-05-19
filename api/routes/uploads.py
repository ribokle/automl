"""CSV upload endpoint.

Streams the incoming file to disk in fixed-size chunks so a giant body
doesn't blow out memory. Validation is intentionally narrow — this endpoint
only accepts the kind of weekly panel CSV the rest of the pipeline reads.
"""
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from api.auth import require_auth
from api.deps import get_run_dir

router = APIRouter(prefix="/uploads", tags=["uploads"], dependencies=[Depends(require_auth)])

CHUNK_BYTES = 1 << 20  # 1 MiB
ALLOWED_EXTENSIONS = {".csv"}
ALLOWED_CONTENT_TYPES = {"text/csv", "application/csv", "application/vnd.ms-excel", "text/plain"}
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _max_upload_bytes() -> int:
    raw = os.environ.get("MAX_UPLOAD_MB", "200")
    try:
        mb = int(raw)
    except ValueError:
        mb = 200
    return max(1, mb) * 1024 * 1024


def _safe_filename(raw: str | None) -> str:
    candidate = Path(raw or "upload.csv").name
    candidate = unicodedata.normalize("NFKD", candidate).encode("ascii", "ignore").decode("ascii")
    candidate = SAFE_NAME_RE.sub("_", candidate).strip("._")
    return candidate or "upload.csv"


@router.post("")
async def upload_csv(
    file: UploadFile = File(...),
    run_dir_base: Path = Depends(get_run_dir),
) -> dict[str, str | int]:
    name = _safe_filename(file.filename)
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"only {sorted(ALLOWED_EXTENSIONS)} uploads are accepted",
        )
    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"unsupported content-type: {file.content_type}",
        )

    upload_dir = run_dir_base.parent / "data"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / name

    cap = _max_upload_bytes()
    written = 0
    try:
        with dest.open("wb") as out:
            while True:
                chunk = await file.read(CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                if written > cap:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"file exceeds {cap // (1024 * 1024)} MiB cap",
                    )
                out.write(chunk)
    finally:
        await file.close()

    if written == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="empty upload")

    return {"path": str(dest.resolve()), "bytes": written, "filename": name}
