"""FastAPI entrypoint: loads the persisted model registry once at startup."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.live_pipeline import LiveInputsCache
from app.routes import pages, predictions
from edf.models.registry import DEFAULT_REGISTRY_DIR, load_registry

APP_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.registry = load_registry(DEFAULT_REGISTRY_DIR)
    app.state.live_cache = LiveInputsCache()
    yield


app = FastAPI(title="GB Demand Forecast — Live Demo", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
app.include_router(pages.router)
app.include_router(predictions.router)
