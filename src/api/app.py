"""
api/app.py — FastAPI app factory for the pull API.

No CORS/middleware beyond auth: this is service-to-service traffic over a
private WireGuard tunnel, not browser-facing.
"""

from __future__ import annotations

from fastapi import FastAPI

from api.routes.stats import router as stats_router
from api.routes.visits import router as visits_router


def create_app() -> FastAPI:
    app = FastAPI(title="medkard pull API")
    app.include_router(visits_router)
    app.include_router(stats_router)
    return app
