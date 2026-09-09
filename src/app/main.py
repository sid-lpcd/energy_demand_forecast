"""FastAPI entrypoint: loads the persisted model registry once at startup."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.live_pipeline import LiveInputsCache
from app.routes import history, pages, predictions
from edf.models.history_comparison import DEFAULT_OUT_PATH as HISTORY_COMPARISON_PATH
from edf.models.registry import DEFAULT_REGISTRY_DIR, load_registry

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent.parent


def _load_history_comparison(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.registry = load_registry(DEFAULT_REGISTRY_DIR)
    app.state.live_cache = LiveInputsCache()
    app.state.history_comparison = _load_history_comparison(REPO_ROOT / HISTORY_COMPARISON_PATH)
    yield


app = FastAPI(title="GB Demand Forecast — Live Demo", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
app.include_router(pages.router)
app.include_router(predictions.router)
app.include_router(history.router)
