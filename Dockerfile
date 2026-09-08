# Live-inference demo (src/app) -- see README's "Deploying the demo" section.
# Trained model artifacts (models_registry/) are committed to the repo and copied in as-is;
# this image never trains anything itself.
FROM python:3.11-slim

# libgomp1: LightGBM's compiled extension needs OpenMP at import time, not present on the slim
# base image by default -- omitting this fails with "libgomp.so.1: cannot open shared object
# file" the first time a persisted model is loaded, not at build time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src/ ./src/
COPY models_registry/ ./models_registry/
COPY reports/ ./reports/
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}"
ENV PORT=7860

# 7860 is Hugging Face Spaces' expected container port for its Docker SDK; Render and most other
# Docker hosts instead inject their own $PORT at runtime and expect the app to bind to it, hence
# the shell-form CMD below rather than a fixed --port.
EXPOSE 7860

# `exec` replaces the shell process with uvicorn (PID 1) so it receives shutdown signals
# directly, rather than a shell that may not forward them -- shell form is still needed here
# only for the ${PORT} expansion.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
