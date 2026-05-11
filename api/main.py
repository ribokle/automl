"""FastAPI application factory."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import approvals, artifacts, events, runs, uploads


def create_app() -> FastAPI:
    app = FastAPI(title="AutoML Agentic Pricing API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
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
