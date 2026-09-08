"""HTML pages: intro (`/`), live predictions (`/predictions`), write-up (`/report`)."""

from __future__ import annotations

from pathlib import Path

import markdown
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

APP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = APP_DIR.parent.parent
REPORTS_DIR = REPO_ROOT / "reports"

templates = Jinja2Templates(directory=APP_DIR / "templates")
router = APIRouter()


def _render_markdown(path: Path) -> str:
    if not path.exists():
        return "<p><em>Not available in this deployment.</em></p>"
    return markdown.markdown(path.read_text(), extensions=["tables", "fenced_code"])


@router.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@router.get("/predictions", response_class=HTMLResponse)
def predictions_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "predictions.html")


@router.get("/report", response_class=HTMLResponse)
def report_page(request: Request) -> HTMLResponse:
    context = {
        "technical_report": _render_markdown(REPORTS_DIR / "technical_report.md"),
        "findings": _render_markdown(REPORTS_DIR / "FINDINGS.md"),
    }
    return templates.TemplateResponse(request, "report.html", context)
