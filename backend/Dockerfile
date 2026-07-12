# syntax=docker/dockerfile:1
#
# Backend image for the docker-compose deployment. Build context is the repo
# root, so source paths are prefixed with backend/.

FROM python:3.11-slim

# All runtime deps ship as wheels (fastapi, pydantic-core, uvicorn, …), so no
# compiler/toolchain layer is needed.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app/backend

# -e keeps the source importable as `app`. No `[dev]` extra → pytest/mypy/ruff
# are excluded from the runtime image.
COPY backend/ ./
RUN pip install -e .

EXPOSE 8000

# Migrate, then serve. Single worker (no --reload) — the in-process rate
# limiter and turn-activity registry assume one worker. Bind 0.0.0.0 so the
# frontend's nginx container can reach us over the compose network — this
# service publishes no host port.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
