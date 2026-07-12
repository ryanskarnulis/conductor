from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter

from app.config import get_settings

# Delegate routes (conductor → per-app agents) land under /api in later
# slices; the router exists now so main.py's shape doesn't change later.
api_router = APIRouter()

app = FastAPI(title="Conductor")
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
