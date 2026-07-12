from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter

from app.api import routes_agent, routes_voice
from app.config import get_settings
from app.fleet.manifests import fleet_from_settings
from app.fleet.tools import build_delegate_tools, default_client_factory, render_fleet_section
from app.logging_config import RequestIDMiddleware, configure_logging

logger = structlog.get_logger(__name__)

api_router = APIRouter()
api_router.include_router(routes_agent.router)
api_router.include_router(routes_voice.router)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Discover the fleet once per process: register the ``ask_<app>`` tools on
    the shared registry and stash the prompt's fleet layer for
    ``routes_agent.get_agent_loop``. The MCP server does the same in its own
    ``main()`` — two drivers, one registry shape."""
    configure_logging()
    fleet = fleet_from_settings()
    build_delegate_tools(fleet, default_client_factory())
    app.state.fleet_section = render_fleet_section(fleet) or None
    logger.info(
        "startup",
        env=get_settings().app_env,
        agents=[fleet_app.name for fleet_app in fleet.agent_apps()],
    )
    yield


app = FastAPI(title="Conductor", lifespan=lifespan)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_origin_regex=get_settings().cors_origin_regex,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router, prefix="/api")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "env": get_settings().app_env}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=get_settings().api_host,
        port=get_settings().api_port,
        reload=True,
    )
