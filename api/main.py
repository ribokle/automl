"""FastAPI application factory."""
from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.deps import get_run_dir
from api.routes import approvals, artifacts, events, runs, uploads

log = logging.getLogger("api.main")


def _allowed_origins() -> list[str]:
    raw = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or ["http://localhost:3000"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    counters = runs.rehydrate_runs(get_run_dir())
    log.info(
        "rehydrated runs from disk: loaded=%s orphaned=%s",
        counters["loaded"],
        counters["orphaned"],
    )
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="AutoML Agentic Pricing API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
        allow_credentials=False,
    )
    app.include_router(runs.router)
    app.include_router(events.router)
    app.include_router(uploads.router)
    app.include_router(artifacts.router)
    app.include_router(approvals.router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
